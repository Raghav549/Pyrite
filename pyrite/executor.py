"""Native streaming executor for Qwen3-MoE GGUF checkpoints.

This is the "native streaming kernel" the runtime was missing: a real,
dependency-free decoder that

* reads tensors from the GGUF file through :class:`~pyrite.stream.BlockStreamer`,
  so resident memory stays inside the configured budget and evictions are real;
* streams individual experts out of the stacked 3D ``ffn_*_exps.weight`` tensors
  instead of materializing all experts of a layer;
* applies the architecture exactly as published (RMSNorm, per-head QK-norm
  before RoPE, GQA attention, softmax router with normalized top-k weights).

It is pure Python, so throughput is far below a native build; the point is
correct, bounded, verifiable local execution rather than pretending a missing
kernel exists.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from .adapters.gguf_adapter import GGUFAdapter
from .config import RuntimeConfig
from .engine import PyriteRuntime
from .ggml_types import spec as ggml_spec
from .ggml_types import type_name
from .qwen3_moe import EXPERT_TENSORS, Qwen3MoECheckpoint, TensorSlice
from .sampler import Sampler
from .stream import BlockStreamer
from .tensor_ops import DECODABLE_TYPES, decode_vector, silu, softmax


class UnsupportedTensorType(ValueError):
    """Raised when a checkpoint uses a quantization Pyrite cannot decode."""


@dataclass
class ExecutorStats:
    steps: int = 0
    layers_executed: int = 0
    tensors_streamed: int = 0
    expert_slices_streamed: int = 0
    resident_bytes: int = 0
    peak_resident_bytes: int = 0
    evictions: int = 0
    cache_hits: int = 0
    prefetched: int = 0
    wasted_prefetch_bytes: int = 0
    kv_tokens: int = 0
    kv_bytes: int = 0
    kv_budget_bytes: int = 0
    bytes_loaded: int = 0

    def to_dict(self) -> dict[str, int]:
        return dict(self.__dict__)


@dataclass
class LayerKV:
    """Per-layer KV cache as flat per-head vectors."""

    keys: list[list[float]] = field(default_factory=list)
    values: list[list[float]] = field(default_factory=list)

    def append(self, key: list[float], value: list[float]) -> None:
        self.keys.append(key)
        self.values.append(value)

    @property
    def length(self) -> int:
        return len(self.keys)

    @property
    def byte_size(self) -> int:
        return sum(len(vector) for vector in self.keys + self.values) * 4

    def truncate(self, max_tokens: int) -> None:
        if len(self.keys) > max_tokens:
            del self.keys[: len(self.keys) - max_tokens]
            del self.values[: len(self.values) - max_tokens]


class _NoTokenizer:
    def encode(self, text: str) -> list[int]:
        raise RuntimeError("this checkpoint does not carry a byte-level BPE tokenizer")

    def decode(self, tokens: Sequence[int]) -> str:
        raise RuntimeError("this checkpoint does not carry a byte-level BPE tokenizer")


class Qwen3MoEExecutor:
    """Real, bounded Qwen3-MoE inference over a GGUF checkpoint."""

    #: Rows of the vocabulary projection streamed per read.
    OUTPUT_CHUNK_ROWS = 128

    DENSE_LAYER_TENSORS: tuple[str, ...] = (
        "attn_norm.weight",
        "attn_q.weight",
        "attn_k.weight",
        "attn_v.weight",
        "attn_output.weight",
        "attn_q_norm.weight",
        "attn_k_norm.weight",
        "ffn_norm.weight",
        "ffn_gate_inp.weight",
    )

    def __init__(
        self,
        path: str | Path,
        runtime: PyriteRuntime | None = None,
        config: RuntimeConfig | None = None,
        *,
        workers: int = 2,
        prefetch: bool = True,
        resident_bytes: int | None = None,
        max_tensor_bytes: int | None = None,
        sampler_seed: int | None = None,
        cache_small_tensors: bool = True,
    ):
        self.runtime = runtime or PyriteRuntime(config)
        self.path = Path(path)
        self.checkpoint = Qwen3MoECheckpoint(self.path)
        self.checkpoint.validate_contract()
        self.adapter = GGUFAdapter(self.path)
        self.prefetch_enabled = prefetch
        budget = int(resident_bytes) if resident_bytes is not None else self.runtime.config.resident_byte_budget
        if budget <= 0:
            raise ValueError("resident_bytes must be positive")
        self.resident_bytes = budget
        self.max_tensor_bytes = max_tensor_bytes or budget
        self.sampler = Sampler(seed=sampler_seed)
        self.streamer = BlockStreamer(
            self.adapter,
            resident_blocks=None,  # the byte budget is authoritative here
            resident_bytes=budget,
            workers=workers,
        )
        self.stats = ExecutorStats()
        self._small: dict[str, list[float]] = {}
        self._cache_small_tensors = cache_small_tensors
        self._kv: dict[int, LayerKV] = {}
        self._last_experts: dict[int, tuple[int, ...]] = {}

        cfg = self.checkpoint.config
        kv_bytes = (
            self.runtime.config.max_kv_tokens
            * cfg.num_hidden_layers
            * 2  # keys and values
            * cfg.kv_proj_dim
            * 4  # fp32 trace
        )
        working_set = self.runtime.config.working_set_mb * 1024 * 1024
        if kv_bytes > working_set:
            raise ValueError(
                f"KV cache for {self.runtime.config.max_kv_tokens} tokens needs "
                f"{kv_bytes / 1024 / 1024:.0f} MiB, which does not fit the "
                f"{self.runtime.config.working_set_mb} MiB working set; lower "
                "PYRITE_KV_TOKENS or raise PYRITE_RAM_MB"
            )
        self.kv_budget_bytes = kv_bytes
        self.stats.kv_budget_bytes = kv_bytes

        from .tokenizer import load_gguf_tokenizer

        self.tokenizer = load_gguf_tokenizer(self.checkpoint.reader) or _NoTokenizer()

    # ------------------------------------------------------------------ helpers
    @property
    def config_(self):
        return self.checkpoint.config

    def close(self) -> None:
        self.streamer.close()

    def __enter__(self) -> Qwen3MoEExecutor:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def unsupported_tensor_types(self) -> tuple[str, ...]:
        names = {
            type_name(tensor.ggml_type)
            for tensor in self.checkpoint.reader.tensor_index()
            if tensor.ggml_type not in DECODABLE_TYPES
        }
        return tuple(sorted(names))

    def ensure_decodable(self) -> None:
        unsupported = self.unsupported_tensor_types()
        if unsupported:
            raise UnsupportedTensorType(
                "checkpoint uses tensor types without a reference decoder: "
                + ", ".join(unsupported)
                + "; Pyrite will not approximate weights"
            )

    def reset_kv(self) -> None:
        """Drop the KV cache so the next token starts a fresh sequence."""
        self._kv.clear()
        self._last_experts.clear()
        self._record_kv()

    def reset(self) -> None:
        """Drop KV state and reset statistics."""
        self.stats = ExecutorStats()
        self.stats.kv_budget_bytes = self.kv_budget_bytes
        self._kv.clear()
        self._last_experts.clear()

    def kv_bytes(self) -> int:
        return sum(cache.byte_size for cache in self._kv.values())

    def kv_tokens(self) -> int:
        """Tokens currently held in the KV cache (the longest layer trace)."""
        return max((cache.length for cache in self._kv.values()), default=0)

    def _record_kv(self) -> None:
        self.stats.kv_tokens = max((cache.length for cache in self._kv.values()), default=0)
        self.stats.kv_bytes = self.kv_bytes()

    def _check_size(self, name: str, size: int) -> None:
        if size > self.max_tensor_bytes:
            raise MemoryError(
                f"tensor {name!r} is {size} bytes, larger than the resident budget "
                f"({self.max_tensor_bytes}); raise PYRITE_RAM_MB or stream smaller slices"
            )

    def _block_bytes(self, name: str) -> bytes:
        block_id = f"tensor:{name}"
        block = self.streamer.index.get(block_id)
        if block is None:
            raise KeyError(f"missing tensor: {name}")
        self._check_size(name, block.size)
        self.stats.tensors_streamed += 1
        payload = bytes(self.streamer.get(block_id))
        self._after_load()
        return payload

    def _dense_vector(self, name: str) -> list[float]:
        """Small tensor (norm/router) with a bounded in-process cache."""
        if self._cache_small_tensors and name in self._small:
            return self._small[name]
        tensor = self.checkpoint.tensor(name)
        values = decode_vector(tensor.ggml_type, self._block_bytes(name), tensor.element_count)
        if self._cache_small_tensors and tensor.size <= max(64 * 1024, self.max_tensor_bytes // 16):
            self._small[name] = values
        return values

    def _tensor_vector(self, name: str) -> list[float]:
        tensor = self.checkpoint.tensor(name)
        return decode_vector(tensor.ggml_type, self._block_bytes(name), tensor.element_count)

    def _row_bytes(self, name: str, first_element: int, element_count: int) -> tuple[int, int]:
        """Byte range covering a block-aligned element range of a tensor."""
        tensor = self.checkpoint.tensor(name)
        block = ggml_spec(tensor.ggml_type)
        if first_element % block.block_size or element_count % block.block_size:
            raise ValueError(
                f"{name} rows must align to the {block.name} block size ({block.block_size})"
            )
        byte_offset = (first_element // block.block_size) * block.bytes_per_block
        byte_length = (element_count // block.block_size) * block.bytes_per_block
        return byte_offset, byte_length

    def _matrix_rows(self, name: str, first_row: int, row_count: int) -> list[list[float]]:
        """Stream ``row_count`` rows of a 2D ``(cols, rows)`` ggml tensor."""
        tensor = self.checkpoint.tensor(name)
        if len(tensor.dims) != 2:
            raise ValueError(f"{name} must be rank-2 for row streaming")
        cols, rows = tensor.dims
        if first_row < 0 or row_count < 0 or first_row + row_count > rows:
            raise ValueError(f"row range [{first_row}, {first_row + row_count}) outside {name}")
        if row_count == 0:
            return []
        byte_offset, byte_length = self._row_bytes(name, first_row * cols, row_count * cols)
        self._check_size(name, byte_length)
        self.stats.tensors_streamed += 1
        payload = bytes(self.streamer.get_range(f"tensor:{name}", byte_offset, byte_length))
        self._after_load()
        values = decode_vector(tensor.ggml_type, payload, row_count * cols)
        return [values[row * cols:(row + 1) * cols] for row in range(row_count)]

    def _slice_vector(self, layer: int, expert: int, component: str) -> list[float]:
        slice_: TensorSlice = self.checkpoint.expert_slice(layer, expert, component)
        self._check_size(slice_.tensor_name, slice_.byte_length)
        self.stats.tensors_streamed += 1
        self.stats.expert_slices_streamed += 1
        payload = bytes(
            self.streamer.get_range(slice_.block_id, slice_.block_offset, slice_.byte_length)
        )
        self._after_load()
        return decode_vector(slice_.ggml_type, payload, slice_.element_count)

    def _after_load(self) -> None:
        resident = self.streamer.cache.stats.estimated_bytes
        self.stats.resident_bytes = resident
        self.stats.peak_resident_bytes = max(self.stats.peak_resident_bytes, resident)
        self.stats.evictions = self.streamer.cache.stats.evictions
        self.stats.cache_hits = self.streamer.cache.stats.hits
        self.stats.prefetched = self.streamer.prefetched
        self.stats.wasted_prefetch_bytes = self.streamer.wasted_prefetch_bytes
        self.stats.bytes_loaded = self.streamer.bytes_loaded

    # ------------------------------------------------------------------- math
    @staticmethod
    def _matvec(weights: Sequence[float], rows: int, cols: int, vector: Sequence[float]) -> list[float]:
        return [
            sum(w * x for w, x in zip(weights[row * cols:(row + 1) * cols], vector, strict=True))
            for row in range(rows)
        ]

    @staticmethod
    def _matvec_transposed(weights: Sequence[float], rows: int, cols: int, vector: Sequence[float]) -> list[float]:
        """``weights^T @ vector`` for weights shaped ``(rows, cols)``."""
        result = [0.0] * cols
        for row in range(rows):
            value = vector[row]
            if value == 0.0:
                continue
            base = row * cols
            for col in range(cols):
                result[col] += value * weights[base + col]
        return result

    @staticmethod
    def _rms_norm(x: Sequence[float], weight: Sequence[float], eps: float) -> list[float]:
        inv = 1.0 / math.sqrt(sum(value * value for value in x) / max(1, len(x)) + eps)
        return [value * inv * w for value, w in zip(x, weight, strict=True)]

    @staticmethod
    def _rope(vector: list[float], position: int, theta: float) -> list[float]:
        """NEOX-style RoPE over the whole head dimension (llama.cpp qwen3 style)."""
        n_rot = len(vector)
        half = n_rot // 2
        for i in range(half):
            freq = theta ** (-2.0 * i / n_rot)
            angle = position * freq
            cos_a, sin_a = math.cos(angle), math.sin(angle)
            x0, x1 = vector[i], vector[i + half]
            vector[i] = x0 * cos_a - x1 * sin_a
            vector[i + half] = x0 * sin_a + x1 * cos_a
        return vector

    # ------------------------------------------------------------------- model
    def _embed(self, token_id: int) -> list[float]:
        cfg = self.checkpoint.config
        if not 0 <= token_id < cfg.vocab_size:
            raise ValueError(f"token id {token_id} outside vocabulary ({cfg.vocab_size})")
        # Only the embedding row for this token is read, never the whole table.
        return self._matrix_rows("token_embd.weight", token_id, 1)[0]

    def _prefetch_layer(self, layer: int) -> None:
        if not self.prefetch_enabled:
            return
        cfg = self.checkpoint.config
        if not 0 <= layer < cfg.num_hidden_layers:
            return
        ids = [f"tensor:blk.{layer}.{name}" for name in self.DENSE_LAYER_TENSORS]
        # Routed experts are mostly stable across adjacent tokens; speculate on
        # the previous choice and account for any wasted bytes.
        for expert in self._last_experts.get(layer, ()):
            for component in EXPERT_TENSORS:
                slice_ = self.checkpoint.expert_slice(layer, expert, component)
                self.streamer.prefetch_ranges(
                    [(slice_.block_id, slice_.block_offset, slice_.byte_length)],
                    confidence=0.6,
                )
        self.streamer.prefetch(ids)

    def _expert(self, layer: int, expert: int, hidden: Sequence[float]) -> list[float]:
        cfg = self.checkpoint.config
        gate = self._matvec(self._slice_vector(layer, expert, "gate"), cfg.moe_intermediate_size, cfg.hidden_size, hidden)
        up = self._matvec(self._slice_vector(layer, expert, "up"), cfg.moe_intermediate_size, cfg.hidden_size, hidden)
        act = [s * u for s, u in zip(silu(gate), up, strict=True)]
        down = self._slice_vector(layer, expert, "down")
        return self._matvec_transposed(down, cfg.moe_intermediate_size, cfg.hidden_size, act)

    def _attention(self, layer: int, hidden: Sequence[float], position: int) -> list[float]:
        cfg = self.checkpoint.config
        head_dim = cfg.head_dim
        q = self._matvec(self._tensor_vector(f"blk.{layer}.attn_q.weight"), cfg.q_proj_dim, cfg.hidden_size, hidden)
        k = self._matvec(self._tensor_vector(f"blk.{layer}.attn_k.weight"), cfg.kv_proj_dim, cfg.hidden_size, hidden)
        v = self._matvec(self._tensor_vector(f"blk.{layer}.attn_v.weight"), cfg.kv_proj_dim, cfg.hidden_size, hidden)

        q_norm = self._dense_vector(f"blk.{layer}.attn_q_norm.weight")
        k_norm = self._dense_vector(f"blk.{layer}.attn_k_norm.weight")

        q_heads = [
            self._rope(self._rms_norm(q[h * head_dim:(h + 1) * head_dim], q_norm, cfg.rms_norm_eps), position, cfg.rope_theta)
            for h in range(cfg.num_attention_heads)
        ]
        k_heads = [
            self._rope(self._rms_norm(k[h * head_dim:(h + 1) * head_dim], k_norm, cfg.rms_norm_eps), position, cfg.rope_theta)
            for h in range(cfg.num_key_value_heads)
        ]
        v_heads = [v[h * head_dim:(h + 1) * head_dim] for h in range(cfg.num_key_value_heads)]

        cache = self._kv.setdefault(layer, LayerKV())
        cache.append([value for head in k_heads for value in head], [value for head in v_heads for value in head])
        cache.truncate(self.runtime.config.max_kv_tokens)
        self._record_kv()

        scale = 1.0 / math.sqrt(head_dim)
        group = cfg.kv_heads_per_group
        output: list[float] = []
        for h, query in enumerate(q_heads):
            kv_head = h // group
            base = kv_head * head_dim
            keys = cache.keys
            values = cache.values
            scores = []
            for position_index in range(len(keys)):
                key = keys[position_index][base: base + head_dim]
                scores.append(sum(a * b for a, b in zip(query, key, strict=True)) * scale)
            probs = softmax(scores)
            head_out = [0.0] * head_dim
            for position_index, weight in enumerate(probs):
                if weight == 0.0:
                    continue
                value = values[position_index][base: base + head_dim]
                for i in range(head_dim):
                    head_out[i] += weight * value[i]
            output.extend(head_out)
        return output

    def _moe(self, layer: int, hidden: Sequence[float]) -> list[float]:
        cfg = self.checkpoint.config
        router = self._matvec(
            self._tensor_vector(f"blk.{layer}.ffn_gate_inp.weight"),
            cfg.num_experts,
            cfg.hidden_size,
            hidden,
        )
        probs = softmax(router)
        ranked = sorted(range(cfg.num_experts), key=lambda e: (-probs[e], e))
        chosen = ranked[: cfg.num_experts_per_tok]
        weights = [probs[e] for e in chosen]
        if cfg.norm_topk_prob:
            total = sum(weights)
            if total > 0:
                weights = [value / total for value in weights]
        if cfg.expert_weights_scale != 1.0:
            weights = [value * cfg.expert_weights_scale for value in weights]

        self._last_experts[layer] = tuple(chosen)
        # Start all selected expert slices at once so their I/O overlaps.
        requested = []
        for expert in chosen:
            for component in EXPERT_TENSORS:
                slice_ = self.checkpoint.expert_slice(layer, expert, component)
                requested.append((slice_.block_id, slice_.block_offset, slice_.byte_length))
        if self.prefetch_enabled:
            self.streamer.prefetch_ranges(requested)

        result = [0.0] * cfg.hidden_size
        for weight, expert in zip(weights, chosen, strict=True):
            if weight == 0.0:
                continue
            expert_out = self._expert(layer, expert, hidden)
            for i in range(cfg.hidden_size):
                result[i] += weight * expert_out[i]
        return result

    def _layer(self, layer: int, hidden: list[float], position: int) -> list[float]:
        cfg = self.checkpoint.config
        normed = self._rms_norm(hidden, self._dense_vector(f"blk.{layer}.attn_norm.weight"), cfg.rms_norm_eps)
        attention = self._attention(layer, normed, position)
        attention = self._matvec(
            self._tensor_vector(f"blk.{layer}.attn_output.weight"),
            cfg.hidden_size,
            cfg.q_proj_dim,
            attention,
        )
        hidden = [a + b for a, b in zip(hidden, attention, strict=True)]

        self._prefetch_layer(layer + 1)
        ffn_normed = self._rms_norm(hidden, self._dense_vector(f"blk.{layer}.ffn_norm.weight"), cfg.rms_norm_eps)
        moe = self._moe(layer, ffn_normed)
        self.stats.layers_executed += 1
        return [a + b for a, b in zip(hidden, moe, strict=True)]

    def logits(self, token_id: int, position: int) -> list[float]:
        """Forward pass for one token; returns the vocabulary logits."""
        if position < 0:
            raise ValueError("position must be non-negative")
        if position >= self.runtime.config.max_kv_tokens:
            raise MemoryError(
                f"position {position} exceeds the configured KV token budget "
                f"({self.runtime.config.max_kv_tokens})"
            )
        self.ensure_decodable()
        hidden = self._embed(token_id)
        for layer in range(self.checkpoint.config.num_hidden_layers):
            hidden = self._layer(layer, hidden, position)
        hidden = self._rms_norm(
            hidden,
            self._dense_vector("output_norm.weight"),
            self.checkpoint.config.rms_norm_eps,
        )
        output_name = "output.weight" if self.checkpoint.has_tensor("output.weight") else "token_embd.weight"
        self.stats.steps += 1
        # The vocabulary projection is streamed in bounded chunks so a large
        # output tensor never has to be resident all at once.
        logits: list[float] = []
        for start in range(0, self.checkpoint.config.vocab_size, self.OUTPUT_CHUNK_ROWS):
            count = min(self.OUTPUT_CHUNK_ROWS, self.checkpoint.config.vocab_size - start)
            rows = self._matrix_rows(output_name, start, count)
            logits.extend(
                sum(w * x for w, x in zip(row, hidden, strict=True))
                for row in rows
            )
        return logits

    # ------------------------------------------------------------ public entry
    def generate(
        self,
        input_ids: list[int],
        max_new_tokens: int,
        temperature: float = 0.0,
        top_p: float = 1.0,
        stop_ids: Sequence[int] | None = None,
    ) -> list[int]:
        """InferenceBackend-compatible greedy/sampled decoding."""
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")
        if not input_ids:
            raise ValueError("input_ids must contain at least one token")
        # Every call decodes a fresh sequence: stale KV entries from an earlier
        # prompt would otherwise leak into the attention context.
        self.reset_kv()
        self.ensure_decodable()
        stops = set(stop_ids if stop_ids is not None else ())
        if self.checkpoint.config.eos_token_id is not None:
            stops.add(self.checkpoint.config.eos_token_id)

        max_kv = self.runtime.config.max_kv_tokens
        position = 0
        generated = list(input_ids)
        logits: list[float] = []
        for token_id in input_ids:
            if position >= max_kv:
                raise MemoryError("prompt exceeds the configured KV token budget")
            logits = self.logits(token_id, position)
            position += 1

        for _ in range(max_new_tokens):
            if position >= max_kv:
                break
            next_token = self.sampler.sample(logits, temperature, top_p)
            generated.append(next_token)
            if next_token in stops:
                break
            logits = self.logits(next_token, position)
            position += 1
        return generated

    def generate_text(
        self,
        prompt: str,
        max_new_tokens: int = 32,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> dict[str, object]:
        """Tokenize, generate and decode; returns text plus token accounting."""
        if isinstance(self.tokenizer, _NoTokenizer):
            raise RuntimeError(  # checkpoint state, not a bad argument type
                "checkpoint has no byte-level BPE tokenizer; pass token ids to generate() instead"
            )
        prompt_ids = self.tokenizer.encode(prompt)
        if not prompt_ids:
            raise ValueError("prompt encodes to zero tokens")
        output_ids = self.generate(prompt_ids, max_new_tokens, temperature, top_p)
        new_ids = output_ids[len(prompt_ids):]
        return {
            "prompt": prompt,
            "text": self.tokenizer.decode(output_ids),
            "completion": self.tokenizer.decode(new_ids),
            "prompt_tokens": len(prompt_ids),
            "new_tokens": len(new_ids),
            "output_ids": output_ids,
        }

    def checkpoint_report(self) -> dict[str, object]:
        cfg = self.checkpoint.config
        total_bytes = sum(tensor.size for tensor in self.checkpoint.reader.tensor_index())
        return {
            "path": str(self.path),
            "architecture": "qwen3moe",
            "layers": cfg.num_hidden_layers,
            "hidden_size": cfg.hidden_size,
            "experts": cfg.num_experts,
            "experts_per_token": cfg.num_experts_per_tok,
            "vocab_size": cfg.vocab_size,
            "tensor_count": self.checkpoint.tensor_count,
            "total_bytes": total_bytes,
            "resident_budget_bytes": self.runtime.config.resident_byte_budget,
            "kv_budget_bytes": self.kv_budget_bytes,
            "unsupported_tensor_types": list(self.unsupported_tensor_types()),
            "tokenizer": type(self.tokenizer).__name__,
        }
