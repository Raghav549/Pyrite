"""Bounded CPU executor for Qwen3-MoE and dense GGUF checkpoints.

The executor reads ranges through :class:`~pyrite.stream.BlockStreamer`,
streams individual experts from stacked 3D tensors, and keeps compact KV state.
RMSNorm, RoPE, attention, routing, sampling, and most tensor types use the
Python reference path. Optional portable C kernels fuse dequantization with
matvec for Q4_K and Q6_K weights; they do not materialize a dequantized matrix.
"""
from __future__ import annotations

import math
import sys
from array import array
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter

from .adapters.gguf_adapter import GGUFAdapter
from .config import RuntimeConfig
from .engine import PyriteRuntime
from .ggml_types import spec as ggml_spec
from .ggml_types import type_name
from .kernels.native import dequantize_rows as native_dequantize_rows
from .kernels.native import matvec as native_matvec
from .kernels.native import native_available, supports_native
from .memory import RSSMonitor
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
    kv_evictions: int = 0
    bytes_loaded: int = 0
    prefetch_hits: int = 0
    io_seconds: float = 0.0
    decode_seconds: float = 0.0
    native_kernel_calls: int = 0
    native_bytes: int = 0
    native_kernel_seconds: float = 0.0
    peak_working_set_bytes: int = 0
    working_set_budget_bytes: int = 0
    rss_current_bytes: int = 0
    rss_peak_bytes: int = 0
    rss_limit_bytes: int = 0
    rss_current_mb: float = 0.0
    rss_peak_mb: float = 0.0
    rss_limit_mb: float = 0.0
    rss_samples: int = 0
    rss_unknown: bool = False
    native_kernel_available: bool = False

    def to_dict(self) -> dict[str, int | float | bool]:
        return dict(self.__dict__)


@dataclass
class LayerKV:
    """Compact per-layer fp32 KV vectors with oldest-first eviction."""

    keys: list[array] = field(default_factory=list)
    values: list[array] = field(default_factory=list)
    _payload_bytes: int = field(default=0, init=False, repr=False)
    _vector_overhead: int = field(
        default=sys.getsizeof(array("f")) + 8,
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        if len(self.keys) != len(self.values):
            raise ValueError("KV keys and values must have equal token counts")
        self._payload_bytes = sum(
            len(vector) * vector.itemsize for vector in self.keys
        ) + sum(len(vector) * vector.itemsize for vector in self.values)

    def append(self, key: Sequence[float], value: Sequence[float]) -> None:
        key_array = array("f", key)
        value_array = array("f", value)
        self.keys.append(key_array)
        try:
            self.values.append(value_array)
        except BaseException:
            self.keys.pop()
            raise
        self._payload_bytes += (len(key_array) + len(value_array)) * key_array.itemsize

    @property
    def length(self) -> int:
        return len(self.keys)

    @property
    def byte_size(self) -> int:
        return self._payload_bytes

    @property
    def estimated_bytes(self) -> int:
        # Include array headers and list references in the managed budget;
        # byte_size intentionally remains the exact fp32 payload for API parity.
        return self._payload_bytes + self.length * 2 * self._vector_overhead

    def evict_oldest(self, count: int = 1) -> int:
        if count < 0:
            raise ValueError("eviction count must be non-negative")
        removed = min(count, self.length)
        if removed:
            removed_bytes = sum(
                len(vector) * vector.itemsize for vector in self.keys[:removed]
            ) + sum(len(vector) * vector.itemsize for vector in self.values[:removed])
            del self.keys[:removed]
            del self.values[:removed]
            self._payload_bytes -= removed_bytes
        return removed

    def truncate(self, max_tokens: int) -> int:
        if max_tokens < 0:
            raise ValueError("max_tokens must be non-negative")
        excess = max(0, self.length - max_tokens)
        return self.evict_oldest(excess)


class _NoTokenizer:
    def encode(self, text: str) -> list[int]:
        raise RuntimeError("this checkpoint does not carry a byte-level BPE tokenizer")

    def decode(self, tokens: Sequence[int]) -> str:
        raise RuntimeError("this checkpoint does not carry a byte-level BPE tokenizer")


class Qwen3MoEExecutor:
    """Real, bounded Qwen3-MoE inference over a GGUF checkpoint."""

    #: Checkpoint contract validated at startup; dense subclasses override this.
    CHECKPOINT_CLS = Qwen3MoECheckpoint

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
        self.checkpoint = self.CHECKPOINT_CLS(self.path)
        self.checkpoint.validate_contract()
        self.adapter = GGUFAdapter(self.path, reader=self.checkpoint.reader)
        self.prefetch_enabled = bool(prefetch)
        self.sampler = Sampler(seed=sampler_seed)

        cfg = self.checkpoint.config
        mib = 1024 * 1024
        working_set = self.runtime.config.resident_byte_budget
        if array("f").itemsize != 4:
            raise RuntimeError("Pyrite requires a 4-byte C float for its compact KV cache")
        kv_payload_bytes = (
            self.runtime.config.max_kv_tokens
            * cfg.num_hidden_layers
            * cfg.kv_cache_dim
            * 4
        )
        kv_overhead_bytes = (
            self.runtime.config.max_kv_tokens
            * cfg.num_hidden_layers
            * (2 * (sys.getsizeof(array("f")) + 8))
        )
        self.kv_budget_bytes = kv_payload_bytes + kv_overhead_bytes
        self.small_cache_budget_bytes = min(32 * mib, max(1, working_set // 32))
        self.transient_reserve_bytes = min(64 * mib, max(16 * mib, working_set // 16))
        available_stream_bytes = (
            working_set
            - self.kv_budget_bytes
            - self.small_cache_budget_bytes
            - self.transient_reserve_bytes
        )
        if available_stream_bytes <= 0:
            raise ValueError(
                f"KV cache for {self.runtime.config.max_kv_tokens} tokens needs "
                f"{self.kv_budget_bytes / mib:.0f} MiB, leaving no streamed-weight "
                f"working set inside {self.runtime.config.working_set_mb} MiB; "
                "lower PYRITE_KV_TOKENS or raise PYRITE_RAM_MB"
            )
        requested_stream_bytes = (
            int(resident_bytes) if resident_bytes is not None else available_stream_bytes
        )
        if requested_stream_bytes <= 0:
            raise ValueError("resident_bytes must be positive")
        self.resident_bytes = min(requested_stream_bytes, available_stream_bytes)
        if max_tensor_bytes is not None and max_tensor_bytes <= 0:
            raise ValueError("max_tensor_bytes must be positive when provided")
        self.max_tensor_bytes = (
            int(max_tensor_bytes) if max_tensor_bytes is not None else self.resident_bytes
        )
        self._cache_small_tensors = cache_small_tensors
        self._small: dict[str, array] = {}
        self._small_sizes: dict[str, int] = {}
        self._small_cache_bytes = 0
        self._kv: dict[int, LayerKV] = {}
        self._last_experts: dict[int, tuple[int, ...]] = {}
        self._closed = False

        from .tokenizer import load_gguf_tokenizer

        self.tokenizer = load_gguf_tokenizer(self.checkpoint.reader) or _NoTokenizer()
        self.rss_monitor = RSSMonitor(self.runtime.config.ram_budget_mb * mib)
        self.streamer = BlockStreamer(
            self.adapter,
            resident_blocks=None,
            resident_bytes=self.resident_bytes,
            workers=workers,
            allow_oversized_reads=False,
        )
        self.stats = ExecutorStats(
            kv_budget_bytes=self.kv_budget_bytes,
            working_set_budget_bytes=working_set,
            rss_limit_bytes=self.runtime.config.ram_budget_mb * mib,
            rss_limit_mb=float(self.runtime.config.ram_budget_mb),
            native_kernel_available=native_available(),
        )
        try:
            self._sample_memory("executor-startup")
        except BaseException:
            self._closed = True
            self.streamer.close()
            raise

    # ------------------------------------------------------------------ helpers
    @property
    def config_(self):
        return self.checkpoint.config

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("executor is closed")

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.streamer.close()
        self._sync_stream_stats()
        self._sample_memory("executor-close")

    def __enter__(self) -> Qwen3MoEExecutor:
        self._ensure_open()
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
        self._ensure_open()
        self._kv.clear()
        self._last_experts.clear()
        self._record_kv()
        self._sample_memory("kv-reset")

    def reset(self) -> None:
        """Drop KV state and reset statistics without resetting the RSS monitor."""
        self._ensure_open()
        self._kv.clear()
        self._last_experts.clear()
        self.stats = ExecutorStats(
            kv_budget_bytes=self.kv_budget_bytes,
            working_set_budget_bytes=self.runtime.config.resident_byte_budget,
            rss_limit_bytes=self.runtime.config.ram_budget_mb * 1024 * 1024,
            rss_limit_mb=float(self.runtime.config.ram_budget_mb),
            native_kernel_available=native_available(),
        )
        self._sync_stream_stats()
        self._record_kv()
        self._sample_memory("statistics-reset")

    def kv_bytes(self) -> int:
        return sum(cache.byte_size for cache in self._kv.values())

    def kv_tokens(self) -> int:
        """Tokens currently held in the KV cache (the longest layer trace)."""
        return max((cache.length for cache in self._kv.values()), default=0)

    def _record_kv(self) -> None:
        self.stats.kv_tokens = max((cache.length for cache in self._kv.values()), default=0)
        self.stats.kv_bytes = self.kv_bytes()

    def _managed_working_set_bytes(self) -> int:
        stream_bytes = self.streamer.stats().resident_bytes
        kv_bytes = sum(cache.estimated_bytes for cache in self._kv.values())
        return stream_bytes + kv_bytes + self._small_cache_bytes

    def _sample_memory(self, stage: str) -> None:
        current = self.rss_monitor.sample(stage)
        self.stats.rss_current_bytes = current
        self.stats.rss_peak_bytes = self.rss_monitor.peak_bytes
        self.stats.rss_current_mb = current / (1024 * 1024)
        self.stats.rss_peak_mb = self.rss_monitor.peak_bytes / (1024 * 1024)
        self.stats.rss_samples = self.rss_monitor.samples
        self.stats.rss_unknown = self.rss_monitor.unknown
        managed = self._managed_working_set_bytes()
        self.stats.peak_working_set_bytes = max(self.stats.peak_working_set_bytes, managed)
        if managed > self.stats.working_set_budget_bytes:
            raise MemoryError(
                f"managed working set reached {managed / (1024 * 1024):.1f} MiB, "
                f"above budget {self.stats.working_set_budget_bytes / (1024 * 1024):.0f} MiB"
            )

    def _sync_stream_stats(self) -> None:
        stream = self.streamer.stats()
        self.stats.resident_bytes = stream.resident_bytes
        self.stats.peak_resident_bytes = max(self.stats.peak_resident_bytes, stream.resident_bytes)
        self.stats.evictions = stream.evictions
        self.stats.cache_hits = stream.cache_hits
        self.stats.prefetched = stream.prefetched
        self.stats.wasted_prefetch_bytes = stream.wasted_prefetch_bytes
        self.stats.bytes_loaded = stream.bytes_loaded
        self.stats.prefetch_hits = stream.prefetch_hits
        self.stats.io_seconds = stream.io_seconds

    def _check_size(self, name: str, size: int) -> None:
        if size > self.max_tensor_bytes:
            raise MemoryError(
                f"tensor {name!r} is {size} bytes, larger than the resident budget "
                f"({self.max_tensor_bytes}); raise PYRITE_RAM_MB or stream smaller slices"
            )

    def _block_bytes(self, name: str) -> memoryview:
        block_id = f"tensor:{name}"
        block = self.streamer.index.get(block_id)
        if block is None:
            raise KeyError(f"missing tensor: {name}")
        self._check_size(name, block.size)
        self.stats.tensors_streamed += 1
        payload = self.streamer.get(block_id)
        self._after_load("tensor-read")
        return payload

    def _dense_vector(self, name: str) -> Sequence[float]:
        """Small tensor with a byte-capped compact fp32 LRU cache."""
        if self._cache_small_tensors and name in self._small:
            values = self._small.pop(name)
            self._small[name] = values
            return values
        tensor = self.checkpoint.tensor(name)
        values = array(
            "f", self._decode(tensor.ggml_type, self._block_bytes(name), tensor.element_count)
        )
        cached_bytes = len(values) * values.itemsize
        if (
            self._cache_small_tensors
            and cached_bytes <= self.small_cache_budget_bytes
            and cached_bytes <= max(64 * 1024, self.max_tensor_bytes // 16)
        ):
            while self._small and self._small_cache_bytes + cached_bytes > self.small_cache_budget_bytes:
                oldest = next(iter(self._small))
                del self._small[oldest]
                self._small_cache_bytes -= self._small_sizes.pop(oldest)
            self._small[name] = values
            self._small_sizes[name] = cached_bytes
            self._small_cache_bytes += cached_bytes
        self._sample_memory("small-tensor-decode")
        return values

    def _tensor_vector(self, name: str) -> list[float]:
        tensor = self.checkpoint.tensor(name)
        return self._decode(tensor.ggml_type, self._block_bytes(name), tensor.element_count)

    def _decode(self, ggml_type: int, payload: bytes | memoryview, count: int) -> list[float]:
        """Reference-decode one tensor payload and record CPU time."""
        started = perf_counter()
        try:
            return decode_vector(ggml_type, payload, count)
        finally:
            self.stats.decode_seconds += perf_counter() - started

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

    def _read_range(self, name: str, byte_offset: int, byte_length: int) -> memoryview:
        self._check_size(name, byte_length)
        block_id = f"tensor:{name}"
        payload = self.streamer.get_range(block_id, byte_offset, byte_length)
        self.stats.tensors_streamed += 1
        self._after_load("weight-range-read")
        return payload

    def _matrix_rows(self, name: str, first_row: int, row_count: int) -> list[list[float]]:
        """Stream and decode rows of a 2D GGML tensor with bounded allocation."""
        tensor = self.checkpoint.tensor(name)
        if len(tensor.dims) != 2:
            raise ValueError(f"{name} must be rank-2 for row streaming")
        cols, rows = tensor.dims
        if first_row < 0 or row_count < 0 or first_row + row_count > rows:
            raise ValueError(f"row range [{first_row}, {first_row + row_count}) outside {name}")
        if row_count == 0:
            return []
        byte_offset, byte_length = self._row_bytes(name, first_row * cols, row_count * cols)
        payload = self._read_range(name, byte_offset, byte_length)
        if supports_native(tensor.ggml_type):
            started = perf_counter()
            values = native_dequantize_rows(tensor.ggml_type, payload, row_count, cols)
            self.stats.native_kernel_calls += 1
            self.stats.native_bytes += byte_length
            self.stats.native_kernel_seconds += perf_counter() - started
        else:
            values = self._decode(tensor.ggml_type, payload, row_count * cols)
        self._sample_memory("matrix-row-decode")
        return [values[row * cols:(row + 1) * cols] for row in range(row_count)]

    def _matrix_matvec(
        self,
        name: str,
        first_row: int,
        row_count: int,
        vector: Sequence[float],
    ) -> list[float]:
        """Read a contiguous row range and multiply without expanding K-quants."""
        tensor = self.checkpoint.tensor(name)
        if len(tensor.dims) != 2:
            raise ValueError(f"{name} must be rank-2 for matrix-vector execution")
        cols, rows = tensor.dims
        if len(vector) != cols:
            raise ValueError(f"matrix/vector shape mismatch for {name}")
        if first_row < 0 or row_count < 0 or first_row + row_count > rows:
            raise ValueError(f"row range [{first_row}, {first_row + row_count}) outside {name}")
        if row_count == 0:
            return []
        byte_offset, byte_length = self._row_bytes(name, first_row * cols, row_count * cols)
        payload = self._read_range(name, byte_offset, byte_length)
        if supports_native(tensor.ggml_type):
            started = perf_counter()
            result = native_matvec(tensor.ggml_type, payload, vector, row_count, cols)
            self.stats.native_kernel_calls += 1
            self.stats.native_bytes += byte_length
            self.stats.native_kernel_seconds += perf_counter() - started
            return result
        values = self._decode(tensor.ggml_type, payload, row_count * cols)
        return self._matvec(values, row_count, cols, vector)

    def _tensor_matvec(
        self,
        name: str,
        rows: int,
        cols: int,
        vector: Sequence[float],
    ) -> list[float]:
        """Bounded matrix-vector product, splitting large matrices by rows."""
        tensor = self.checkpoint.tensor(name)
        if len(tensor.dims) != 2 or tensor.dims != (cols, rows):
            raise ValueError(
                f"{name} has shape {tensor.dims}; expected GGML dimensions {(cols, rows)}"
            )
        _, row_bytes = self._row_bytes(name, 0, cols)
        range_budget = min(self.resident_bytes, self.max_tensor_bytes)
        if row_bytes > range_budget:
            raise MemoryError(
                f"one row of tensor {name!r} needs {row_bytes} bytes, above the "
                f"streaming range budget ({range_budget} bytes)"
            )
        rows_per_chunk = max(1, range_budget // row_bytes)
        if supports_native(tensor.ggml_type):
            rows_per_chunk = min(rows_per_chunk, 4096)
        else:
            # Python floats and list pointers expand compact GGUF payloads by
            # much more than 4x.  Cap each reference decode to about 8 MiB of
            # Python objects even when the file tensor itself is several GB.
            rows_per_chunk = min(rows_per_chunk, max(1, 262_144 // cols))
        result: list[float] = []
        for start in range(0, rows, rows_per_chunk):
            count = min(rows_per_chunk, rows - start)
            result.extend(self._matrix_matvec(name, start, count, vector))
        return result

    def _slice_matvec(
        self,
        layer: int,
        expert: int,
        component: str,
        rows: int,
        cols: int,
        vector: Sequence[float],
    ) -> list[float]:
        """Matvec one expert slice in bounded row ranges (never the full stack)."""
        slice_: TensorSlice = self.checkpoint.expert_slice(layer, expert, component)
        if slice_.element_count != rows * cols or len(vector) != cols:
            raise ValueError(f"expert {component} matrix/vector shape mismatch")
        block = ggml_spec(slice_.ggml_type)
        if cols % block.block_size:
            raise ValueError(
                f"expert {component} row width {cols} is not aligned to {block.name} blocks"
            )
        row_bytes = (cols // block.block_size) * block.bytes_per_block
        range_budget = min(self.resident_bytes, self.max_tensor_bytes)
        if row_bytes > range_budget:
            raise MemoryError(
                f"one {component} expert row needs {row_bytes} bytes, above the "
                f"streaming range budget ({range_budget} bytes)"
            )
        rows_per_chunk = max(1, range_budget // row_bytes)
        if supports_native(slice_.ggml_type):
            rows_per_chunk = min(rows_per_chunk, 4096)
        else:
            rows_per_chunk = min(rows_per_chunk, max(1, 262_144 // cols))
        result: list[float] = []
        for first in range(0, rows, rows_per_chunk):
            count = min(rows_per_chunk, rows - first)
            byte_offset = slice_.block_offset + first * row_bytes
            byte_length = count * row_bytes
            self._check_size(slice_.tensor_name, byte_length)
            payload = self.streamer.get_range(slice_.block_id, byte_offset, byte_length)
            self.stats.tensors_streamed += 1
            self.stats.expert_slices_streamed += 1
            self._after_load("expert-slice-read")
            if supports_native(slice_.ggml_type):
                started = perf_counter()
                result.extend(native_matvec(slice_.ggml_type, payload, vector, count, cols))
                self.stats.native_kernel_calls += 1
                self.stats.native_bytes += byte_length
                self.stats.native_kernel_seconds += perf_counter() - started
            else:
                values = self._decode(slice_.ggml_type, payload, count * cols)
                result.extend(self._matvec(values, count, cols, vector))
        return result

    def _after_load(self, stage: str = "tensor-read") -> None:
        self._sync_stream_stats()
        self._sample_memory(stage)

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
    def _rope(
        vector: list[float], position: int, theta: float, rotary_dim: int | None = None
    ) -> list[float]:
        """NEOX-style RoPE (llama.cpp style).

        The whole head rotates unless ``rotary_dim`` is smaller, in which case
        only the first ``rotary_dim`` values rotate and the rest pass through.
        """
        n_rot = len(vector) if rotary_dim is None else min(rotary_dim, len(vector))
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
        gate = self._slice_matvec(
            layer, expert, "gate", cfg.moe_intermediate_size, cfg.hidden_size, hidden
        )
        up = self._slice_matvec(
            layer, expert, "up", cfg.moe_intermediate_size, cfg.hidden_size, hidden
        )
        act = [s * u for s, u in zip(silu(gate), up, strict=True)]
        return self._slice_matvec(
            layer, expert, "down", cfg.hidden_size, cfg.moe_intermediate_size, act
        )

    def _attention(self, layer: int, hidden: Sequence[float], position: int) -> list[float]:
        cfg = self.checkpoint.config
        head_dim = cfg.head_dim
        value_dim = cfg.value_dim
        q = self._tensor_matvec(
            f"blk.{layer}.attn_q.weight", cfg.q_proj_dim, cfg.hidden_size, hidden
        )
        k = self._tensor_matvec(
            f"blk.{layer}.attn_k.weight", cfg.kv_proj_dim, cfg.hidden_size, hidden
        )
        v = self._tensor_matvec(
            f"blk.{layer}.attn_v.weight", cfg.v_proj_dim, cfg.hidden_size, hidden
        )

        # QK-norm exists on Qwen3(-MoE) but not on Llama; apply it only when
        # the checkpoint contract declares and supplies those tensors.
        q_norm_name = f"blk.{layer}.attn_q_norm.weight"
        k_norm_name = f"blk.{layer}.attn_k_norm.weight"
        q_norm = self._dense_vector(q_norm_name) if self.checkpoint.has_tensor(q_norm_name) else None
        k_norm = self._dense_vector(k_norm_name) if self.checkpoint.has_tensor(k_norm_name) else None
        rotary_dim = getattr(cfg, "rotary_dim", head_dim) or head_dim

        def prep_head(vec: list[float], norm: Sequence[float] | None) -> list[float]:
            if norm is not None:
                vec = self._rms_norm(vec, norm, cfg.rms_norm_eps)
            return self._rope(vec, position, cfg.rope_theta, rotary_dim)

        q_heads = [
            prep_head(q[h * head_dim:(h + 1) * head_dim], q_norm)
            for h in range(cfg.num_attention_heads)
        ]
        k_heads = [
            prep_head(k[h * head_dim:(h + 1) * head_dim], k_norm)
            for h in range(cfg.num_key_value_heads)
        ]
        v_heads = [
            v[h * value_dim:(h + 1) * value_dim]
            for h in range(cfg.num_key_value_heads)
        ]

        cache = self._kv.setdefault(layer, LayerKV())
        cache.append(
            [value for head in k_heads for value in head],
            [value for head in v_heads for value in head],
        )
        evicted = cache.truncate(self.runtime.config.max_kv_tokens)
        if evicted:
            self.stats.kv_evictions += evicted
        self._record_kv()
        self._sample_memory("kv-cache-update")

        scale = 1.0 / math.sqrt(head_dim)
        group = cfg.kv_heads_per_group
        output: list[float] = []
        for h, query in enumerate(q_heads):
            kv_head = h // group
            key_base = kv_head * head_dim
            value_base = kv_head * value_dim
            scores = [
                sum(query[i] * key[key_base + i] for i in range(head_dim)) * scale
                for key in cache.keys
            ]
            probs = softmax(scores)
            head_out = [0.0] * value_dim
            for weight, value in zip(probs, cache.values, strict=True):
                if weight == 0.0:
                    continue
                for i in range(value_dim):
                    head_out[i] += weight * value[value_base + i]
            output.extend(head_out)
        return output

    def _moe(self, layer: int, hidden: Sequence[float]) -> list[float]:
        cfg = self.checkpoint.config
        router = self._tensor_matvec(
            f"blk.{layer}.ffn_gate_inp.weight",
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
        attention = self._tensor_matvec(
            f"blk.{layer}.attn_output.weight",
            cfg.hidden_size,
            cfg.attn_output_dim,
            attention,
        )
        hidden = [a + b for a, b in zip(hidden, attention, strict=True)]

        self._prefetch_layer(layer + 1)
        ffn_normed = self._rms_norm(hidden, self._dense_vector(f"blk.{layer}.ffn_norm.weight"), cfg.rms_norm_eps)
        moe = self._moe(layer, ffn_normed)
        self.stats.layers_executed += 1
        return [a + b for a, b in zip(hidden, moe, strict=True)]

    def _context_limit(self) -> int:
        model_limit = self.checkpoint.config.max_position_embeddings
        if model_limit <= 0:
            model_limit = self.runtime.config.max_context_tokens
        return min(self.runtime.config.max_context_tokens, model_limit)

    def logits(self, token_id: int, position: int) -> list[float]:
        """Forward one autoregressive token with bounded KV and weight state."""
        self._ensure_open()
        if position < 0:
            raise ValueError("position must be non-negative")
        context_limit = self._context_limit()
        if position >= context_limit:
            raise MemoryError(
                f"position {position} exceeds the model/runtime context limit "
                f"({context_limit})"
            )
        if (
            self.runtime.config.kv_cache_policy == "stop"
            and position >= self.runtime.config.max_kv_tokens
        ):
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
        output_tensor = self.checkpoint.tensor(output_name)
        output_cols, output_rows = output_tensor.dims
        _, output_row_bytes = self._row_bytes(output_name, 0, output_cols)
        range_budget = min(self.resident_bytes, self.max_tensor_bytes)
        if output_row_bytes > range_budget:
            raise MemoryError(
                f"one output row needs {output_row_bytes} bytes, above the streaming "
                f"range budget ({range_budget} bytes)"
            )
        output_chunk_rows = min(
            self.OUTPUT_CHUNK_ROWS,
            max(1, range_budget // output_row_bytes),
        )
        if not supports_native(output_tensor.ggml_type):
            output_chunk_rows = min(output_chunk_rows, max(1, 262_144 // output_cols))
        logits: list[float] = []
        for start in range(0, output_rows, output_chunk_rows):
            count = min(output_chunk_rows, output_rows - start)
            logits.extend(self._matrix_matvec(output_name, start, count, hidden))
        self._sync_stream_stats()
        self._sample_memory("token-forward-complete")
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
        self._ensure_open()
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")
        if not input_ids:
            raise ValueError("input_ids must contain at least one token")
        if not math.isfinite(temperature) or temperature < 0:
            raise ValueError("temperature must be finite and non-negative")
        if not math.isfinite(top_p) or not 0.0 < top_p <= 1.0:
            raise ValueError("top_p must be finite and in (0, 1]")
        if any(not isinstance(token, int) or not 0 <= token < self.checkpoint.config.vocab_size for token in input_ids):
            raise ValueError("input token id is outside the checkpoint vocabulary")
        # Every call decodes a fresh sequence: stale KV entries from an earlier
        # prompt would otherwise leak into the attention context.
        self.reset_kv()
        if max_new_tokens == 0:
            return list(input_ids)
        self.ensure_decodable()
        stops = set(stop_ids if stop_ids is not None else ())
        if self.checkpoint.config.eos_token_id is not None:
            stops.add(self.checkpoint.config.eos_token_id)
        if any(not isinstance(token, int) or not 0 <= token < self.checkpoint.config.vocab_size for token in stops):
            raise ValueError("stop token id is outside the checkpoint vocabulary")

        context_limit = self._context_limit()
        if len(input_ids) > context_limit:
            raise MemoryError(
                f"prompt has {len(input_ids)} tokens, above context limit {context_limit}"
            )
        max_kv = self.runtime.config.max_kv_tokens
        if self.runtime.config.kv_cache_policy == "stop" and len(input_ids) > max_kv:
            raise MemoryError("prompt exceeds the configured KV token budget")

        position = 0
        generated = list(input_ids)
        logits: list[float] = []
        for token_id in input_ids:
            logits = self.logits(token_id, position)
            position += 1

        for _ in range(max_new_tokens):
            if position >= context_limit:
                break
            if self.runtime.config.kv_cache_policy == "stop" and position >= max_kv:
                break
            next_token = self.sampler.sample(logits, temperature, top_p)
            generated.append(next_token)
            if next_token in stops:
                break
            logits = self.logits(next_token, position)
            position += 1
        self._sync_stream_stats()
        self._sample_memory("generation-complete")
        return generated

    def generate_text(
        self,
        prompt: str,
        max_new_tokens: int = 32,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> dict[str, object]:
        """Tokenize, generate and decode; returns text plus token accounting."""
        self._ensure_open()
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
            "architecture": str(
                self.checkpoint.metadata.get("general.architecture", "qwen3moe")
            ),
            "layers": cfg.num_hidden_layers,
            "hidden_size": cfg.hidden_size,
            "experts": cfg.num_experts,
            "experts_per_token": cfg.num_experts_per_tok,
            "vocab_size": cfg.vocab_size,
            "tensor_count": self.checkpoint.tensor_count,
            "total_bytes": total_bytes,
            "resident_budget_bytes": self.resident_bytes,
            "working_set_budget_bytes": self.runtime.config.resident_byte_budget,
            "rss_limit_bytes": self.runtime.config.ram_budget_mb * 1024 * 1024,
            "kv_budget_bytes": self.kv_budget_bytes,
            "kv_cache_policy": self.runtime.config.kv_cache_policy,
            "cpu_execution": True,
            "native_kernel_available": native_available(),
            "native_kernel_types": ["Q4_K", "Q6_K"] if native_available() else [],
            "unsupported_tensor_types": list(self.unsupported_tensor_types()),
            "tokenizer": type(self.tokenizer).__name__,
        }


def detect_architecture(path: str | Path) -> str:
    """Read ``general.architecture`` from a GGUF checkpoint."""
    from .adapters.gguf import GGUFReader

    reader = GGUFReader(Path(path))
    return str(reader.metadata().get("general.architecture", ""))


def open_executor(
    path: str | Path,
    runtime: PyriteRuntime | None = None,
    config: RuntimeConfig | None = None,
    **kwargs,
) -> Qwen3MoEExecutor:
    """Open the streaming executor matching the checkpoint architecture.

    ``qwen3moe`` checkpoints use the MoE executor; ``qwen3`` and ``llama``
    checkpoints use the dense executor.  Anything else raises a clear error
    instead of running the wrong architecture.
    """
    arch = detect_architecture(path)
    if arch == "qwen3moe":
        return Qwen3MoEExecutor(path, runtime=runtime, config=config, **kwargs)
    if arch in ("qwen3", "llama"):
        from .dense import DenseExecutor

        return DenseExecutor(path, runtime=runtime, config=config, **kwargs)
    raise ValueError(
        f"unsupported GGUF architecture {arch or 'unknown'!r}: "
        "Pyrite executes qwen3moe, qwen3 and llama checkpoints"
    )
