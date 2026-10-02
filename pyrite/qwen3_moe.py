"""Qwen3-MoE checkpoint contract.

The layout and metadata keys here follow the published GGUF contract used by
llama.cpp for the ``qwen3moe`` architecture:

* metadata keys live under the ``qwen3moe.`` prefix (for example
  ``qwen3moe.expert_feed_forward_length``, ``qwen3moe.expert_weights_norm``);
* routed experts are stored as three stacked 3D tensors per layer
  (``ffn_gate_exps.weight``, ``ffn_up_exps.weight``, ``ffn_down_exps.weight``),
  so a single expert is a contiguous slice and can be streamed on its own;
* attention QK-norm tensors (``attn_q_norm.weight``/``attn_k_norm.weight``) have
  one value per head dimension and are applied before RoPE;
* Qwen3-MoE has no shared expert tensors.

The module validates a checkpoint instead of pretending a missing kernel exists.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from .adapters.gguf import GGUFReader, GGUFTensor
from .ggml_types import row_size, type_name
from .ggml_types import spec as ggml_spec
from .tensor_ops import DECODABLE_TYPES

ARCHITECTURE = "qwen3moe"

#: Component name -> stacked 3D expert tensor for that projection.
EXPERT_TENSORS: dict[str, str] = {
    "gate": "ffn_gate_exps.weight",
    "up": "ffn_up_exps.weight",
    "down": "ffn_down_exps.weight",
}

#: Dense ("shared") attention/MLP tensors every layer must provide.
LAYER_TENSORS: tuple[str, ...] = (
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


class Qwen3MoEContractError(ValueError):
    """Raised when a GGUF file does not satisfy the Qwen3-MoE contract."""


@dataclass(frozen=True)
class TensorSlice:
    """A contiguous, block-aligned region of a checkpoint tensor."""

    tensor_name: str
    block_id: str
    byte_offset: int
    """Absolute offset of the slice inside the checkpoint file."""

    block_offset: int
    """Offset of the slice relative to the start of its tensor/block."""

    byte_length: int
    element_offset: int
    element_count: int
    shape: tuple[int, ...]
    ggml_type: int

    @property
    def dtype(self) -> str:
        return type_name(self.ggml_type)

    @property
    def is_quantized(self) -> bool:
        return ggml_spec(self.ggml_type).quantized


@dataclass(frozen=True)
class Qwen3MoEConfig:
    hidden_size: int
    intermediate_size: int
    moe_intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    num_experts: int
    num_experts_per_tok: int
    vocab_size: int
    head_dim: int
    value_dim: int
    rms_norm_eps: float
    rope_theta: float
    max_position_embeddings: int
    bos_token_id: int | None
    eos_token_id: int | None
    norm_topk_prob: bool
    tie_word_embeddings: bool
    expert_weights_scale: float

    @property
    def q_proj_dim(self) -> int:
        return self.num_attention_heads * self.head_dim

    @property
    def kv_proj_dim(self) -> int:
        return self.num_key_value_heads * self.head_dim

    @property
    def v_proj_dim(self) -> int:
        return self.num_key_value_heads * self.value_dim

    @property
    def attn_output_dim(self) -> int:
        return self.num_attention_heads * self.value_dim

    @property
    def kv_cache_dim(self) -> int:
        return self.num_key_value_heads * (self.head_dim + self.value_dim)

    @property
    def kv_heads_per_group(self) -> int:
        return max(1, self.num_attention_heads // max(1, self.num_key_value_heads))

    @classmethod
    def from_metadata(cls, metadata: Mapping[str, object]) -> Qwen3MoEConfig:
        def get(key: str, default: object = None) -> object:
            return metadata.get(key, default)

        arch = str(get("general.architecture", ""))
        if arch != ARCHITECTURE:
            raise Qwen3MoEContractError(f"unsupported GGUF architecture: {arch or 'unknown'}")

        def n(suffix: str, default: object = None) -> object:
            return get(f"{ARCHITECTURE}.{suffix}", default)

        hidden = int(n("embedding_length", 0) or 0)
        layers = int(n("block_count", 0) or 0)
        heads = int(n("attention.head_count", 0) or 0)
        raw_kv_heads = n("attention.head_count_kv", None)
        kv_heads = int(heads if raw_kv_heads is None else raw_kv_heads)
        intermediate = int(n("feed_forward_length", 0) or 0)
        moe_intermediate = int(n("expert_feed_forward_length", 0) or 0)
        experts = int(n("expert_count", 0) or 0)
        per_tok = int(n("expert_used_count", 0) or 0)
        vocab = _resolve_vocab_size(metadata, n)
        raw_head_dim = n("attention.key_length", None)
        head_dim = int(hidden // heads if raw_head_dim is None and heads else raw_head_dim or 0)
        raw_value_dim = n("attention.value_length", None)
        value_dim = int(head_dim if raw_value_dim is None else raw_value_dim)
        max_pos = int(n("context_length", 0) or 0)
        raw_eps = n("attention.layer_norm_rms_epsilon", None)
        eps = float(1e-6 if raw_eps is None else raw_eps)
        raw_rope_theta = n("rope.freq_base", None)
        rope_theta = float(1_000_000.0 if raw_rope_theta is None else raw_rope_theta)
        bos = get("tokenizer.ggml.bos_token_id")
        eos = get("tokenizer.ggml.eos_token_id")
        tied = _metadata_bool(n("tied_word_embeddings"), False, "tied_word_embeddings")
        norm_topk = _metadata_bool(n("expert_weights_norm"), True, "expert_weights_norm")
        raw_expert_scale = n("expert_weights_scale", None)
        expert_scale = float(1.0 if raw_expert_scale is None else raw_expert_scale)

        required = (hidden, layers, heads, kv_heads, intermediate, moe_intermediate, experts, per_tok, vocab, head_dim)
        if any(value <= 0 for value in required):
            raise Qwen3MoEContractError("GGUF is missing required Qwen3-MoE metadata")
        if kv_heads > heads or heads % kv_heads:
            raise Qwen3MoEContractError("attention head count must be divisible by KV head count")
        if per_tok > experts:
            raise Qwen3MoEContractError("expert_used_count cannot exceed expert_count")
        if value_dim <= 0 or not math.isfinite(eps) or eps <= 0:
            raise Qwen3MoEContractError("attention value length or RMSNorm epsilon is invalid")
        if not math.isfinite(rope_theta) or rope_theta <= 0:
            raise Qwen3MoEContractError("RoPE frequency base must be finite and positive")
        if not math.isfinite(expert_scale) or expert_scale <= 0:
            raise Qwen3MoEContractError("expert weight scale must be finite and positive")

        return cls(
            hidden_size=hidden,
            intermediate_size=intermediate,
            moe_intermediate_size=moe_intermediate,
            num_hidden_layers=layers,
            num_attention_heads=heads,
            num_key_value_heads=kv_heads,
            num_experts=experts,
            num_experts_per_tok=per_tok,
            vocab_size=vocab,
            head_dim=head_dim,
            value_dim=value_dim,
            rms_norm_eps=eps,
            rope_theta=rope_theta,
            max_position_embeddings=max_pos,
            bos_token_id=int(bos) if bos is not None else None,
            eos_token_id=int(eos) if eos is not None else None,
            norm_topk_prob=norm_topk,
            tie_word_embeddings=tied,
            expert_weights_scale=expert_scale,
        )

    @property
    def is_canonical_qwen3_moe(self) -> bool:
        """Whether metadata matches the published 235B-A22B architecture."""
        return (
            self.num_hidden_layers == 94
            and self.num_experts == 128
            and self.num_experts_per_tok == 8
        )

    def to_dict(self) -> dict[str, object]:
        return dict(self.__dict__)


def _metadata_bool(value: object, default: bool, name: str) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    raise Qwen3MoEContractError(f"{name} metadata must be a boolean")


def _resolve_vocab_size(metadata: Mapping[str, object], get_key) -> int:
    """Resolve the vocabulary size the way real GGUFs store it.

    ``tokenizer.ggml.vocab_size`` is not part of the published GGUF contract;
    files carry the token array and/or ``{arch}.vocab_size``.  Prefer the
    explicit key, then the token array length, then the legacy key.

    When both the explicit key and a token array are present they must agree:
    llama.cpp derives the vocabulary from the tokenizer and refuses files
    where the embedding table disagrees, so Pyrite refuses them too instead of
    generating undecodable token ids.
    """
    explicit = metadata.get(f"{ARCHITECTURE}.vocab_size")
    tokens = metadata.get("tokenizer.ggml.tokens")
    token_count = len(tokens) if isinstance(tokens, (list, tuple)) and tokens else 0
    if explicit is not None:
        if token_count and int(explicit) != token_count:
            raise Qwen3MoEContractError(
                f"{ARCHITECTURE}.vocab_size is {int(explicit)} but the tokenizer "
                f"carries {token_count} tokens; refusing a self-inconsistent checkpoint"
            )
        return int(explicit)
    if token_count:
        return token_count
    legacy = metadata.get("tokenizer.ggml.vocab_size")
    if legacy is not None:
        return int(legacy)
    return 0


class Qwen3MoECheckpoint:
    """Inspect, validate and slice a Qwen3-MoE GGUF checkpoint."""

    def __init__(self, path: str | Path, reader: GGUFReader | None = None):
        self.path = Path(path)
        self.reader = reader or GGUFReader(self.path)
        if self.reader.path != self.path:
            raise ValueError("GGUF reader belongs to a different checkpoint")
        self.metadata = self.reader.metadata()
        self.config = Qwen3MoEConfig.from_metadata(self.metadata)
        self._tensors = {tensor.name: tensor for tensor in self.reader.tensor_index()}

    @property
    def tensor_count(self) -> int:
        return len(self._tensors)

    def tensor(self, name: str) -> GGUFTensor:
        try:
            return self._tensors[name]
        except KeyError as exc:
            raise KeyError(f"missing tensor: {name}") from exc

    def has_tensor(self, name: str) -> bool:
        return name in self._tensors

    # ------------------------------------------------------------------ slicing
    def stacked_expert_tensor(self, layer: int, component: str) -> GGUFTensor:
        """Return the 3D ``ffn_*_exps.weight`` tensor for ``layer``."""
        if component not in EXPERT_TENSORS:
            raise KeyError(f"unknown expert component: {component}")
        if not 0 <= layer < self.config.num_hidden_layers:
            raise IndexError(layer)
        return self.tensor(f"blk.{layer}.{EXPERT_TENSORS[component]}")

    def expert_slice(self, layer: int, expert: int, component: str) -> TensorSlice:
        """Byte/element slice of one expert inside a stacked 3D tensor.

        The ggml layout is row-major with the first dimension fastest, so the
        slice for expert ``e`` is contiguous and can be streamed without
        touching any other expert.
        """
        tensor = self.stacked_expert_tensor(layer, component)
        if not 0 <= expert < self.config.num_experts:
            raise IndexError(expert)
        if len(tensor.dims) != 3:
            raise Qwen3MoEContractError(
                f"{tensor.name} must be a stacked 3D expert tensor, got dims {tensor.dims}"
            )
        inner = tensor.dims[0] * tensor.dims[1]
        block = ggml_spec(tensor.ggml_type)
        if inner % block.block_size:
            raise Qwen3MoEContractError(
                f"{tensor.name} expert slice of {inner} values is not a multiple of "
                f"the {block.name} block size {block.block_size}"
            )
        element_offset = expert * inner
        block_offset = (element_offset // block.block_size) * block.bytes_per_block
        byte_length = (inner // block.block_size) * block.bytes_per_block
        return TensorSlice(
            tensor_name=tensor.name,
            block_id=f"tensor:{tensor.name}",
            byte_offset=tensor.offset + block_offset,
            block_offset=block_offset,
            byte_length=byte_length,
            element_offset=element_offset,
            element_count=inner,
            shape=(tensor.dims[0], tensor.dims[1]),
            ggml_type=tensor.ggml_type,
        )

    # ------------------------------------------------------------- validation
    def validate_contract(self) -> dict[str, object]:
        """Check tensor presence and shapes; raise on any violation."""
        cfg = self.config
        problems: list[str] = []

        def require(name: str, shape: tuple[int, ...]) -> None:
            tensor = self._tensors.get(name)
            if tensor is None:
                problems.append(f"missing tensor: {name}")
                return
            if tuple(tensor.dims) != shape:
                problems.append(f"{name} has shape {tensor.dims}, expected {shape}")

        require("token_embd.weight", (cfg.hidden_size, cfg.vocab_size))
        require("output_norm.weight", (cfg.hidden_size,))
        if "output.weight" in self._tensors:
            require("output.weight", (cfg.hidden_size, cfg.vocab_size))
        elif not cfg.tie_word_embeddings:
            problems.append("missing tensor: output.weight (checkpoint does not declare tied embeddings)")

        for layer in range(cfg.num_hidden_layers):
            prefix = f"blk.{layer}."
            require(prefix + "attn_norm.weight", (cfg.hidden_size,))
            require(prefix + "attn_q.weight", (cfg.hidden_size, cfg.q_proj_dim))
            require(prefix + "attn_k.weight", (cfg.hidden_size, cfg.kv_proj_dim))
            require(prefix + "attn_v.weight", (cfg.hidden_size, cfg.v_proj_dim))
            require(prefix + "attn_output.weight", (cfg.attn_output_dim, cfg.hidden_size))
            require(prefix + "attn_q_norm.weight", (cfg.head_dim,))
            require(prefix + "attn_k_norm.weight", (cfg.head_dim,))
            require(prefix + "ffn_norm.weight", (cfg.hidden_size,))
            require(prefix + "ffn_gate_inp.weight", (cfg.hidden_size, cfg.num_experts))
            require(
                prefix + EXPERT_TENSORS["gate"],
                (cfg.hidden_size, cfg.moe_intermediate_size, cfg.num_experts),
            )
            require(
                prefix + EXPERT_TENSORS["up"],
                (cfg.hidden_size, cfg.moe_intermediate_size, cfg.num_experts),
            )
            require(
                prefix + EXPERT_TENSORS["down"],
                (cfg.moe_intermediate_size, cfg.hidden_size, cfg.num_experts),
            )

        if problems:
            raise Qwen3MoEContractError(
                "Qwen3-MoE GGUF contract violated:\n  - " + "\n  - ".join(problems[:20])
            )
        return self.routing_summary()

    def routing_summary(self) -> dict[str, object]:
        cfg = self.config
        attention_layers = 0
        moe_layers = 0
        qk_norm_layers = 0
        complete_layers = 0
        layer_requirements = (*LAYER_TENSORS, *EXPERT_TENSORS.values())
        for layer in range(cfg.num_hidden_layers):
            prefix = f"blk.{layer}."
            attention_layers += int(
                all(
                    f"{prefix}{name}" in self._tensors
                    for name in ("attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_output.weight")
                )
            )
            moe_layers += int(
                all(f"{prefix}{name}" in self._tensors for name in EXPERT_TENSORS.values())
            )
            qk_norm_layers += int(
                f"{prefix}attn_q_norm.weight" in self._tensors
                and f"{prefix}attn_k_norm.weight" in self._tensors
            )
            complete_layers += int(
                all(f"{prefix}{name}" in self._tensors for name in layer_requirements)
            )
        output_complete = (
            "output.weight" in self._tensors or cfg.tie_word_embeddings
        )
        globals_complete = (
            "token_embd.weight" in self._tensors
            and "output_norm.weight" in self._tensors
            and output_complete
        )
        unsupported_types = sorted(
            {
                type_name(tensor.ggml_type)
                for tensor in self._tensors.values()
                if tensor.ggml_type not in DECODABLE_TYPES
            }
        )
        structurally_complete = (
            complete_layers == cfg.num_hidden_layers and globals_complete
        )
        return {
            "architecture": ARCHITECTURE,
            "layers": cfg.num_hidden_layers,
            "experts": cfg.num_experts,
            "experts_per_token": cfg.num_experts_per_tok,
            "canonical_qwen3_235b_a22b": cfg.is_canonical_qwen3_moe,
            "required_canonical_contract": {
                "layers": 94,
                "experts": 128,
                "experts_per_token": 8,
            },
            "norm_topk_prob": cfg.norm_topk_prob,
            "attention_complete_layers": attention_layers,
            "qk_norm_layers": qk_norm_layers,
            "moe_layers_detected": moe_layers,
            "complete_layers": complete_layers,
            "global_tensors_complete": globals_complete,
            "unsupported_tensor_types": unsupported_types,
            "tensor_types_decodable": not unsupported_types,
            "native_generation_ready": structurally_complete and not unsupported_types,
        }

    def validate_for_streaming(self, resident_bytes: int) -> dict[str, object]:
        """Report what fits in the resident budget; never claims more than that."""
        if resident_bytes < 0:
            raise ValueError("resident_bytes must be non-negative")
        tensors = list(self._tensors.values())
        oversized = [tensor for tensor in tensors if tensor.size > resident_bytes]
        expert_slices = self._expert_slice_sizes()
        largest_expert = max(expert_slices) if expert_slices else 0
        largest_row = max(
            (row_size(tensor.dims[0], tensor.ggml_type) for tensor in tensors),
            default=0,
        )
        experts_fit = largest_expert <= resident_bytes
        rows_fit = largest_row <= resident_bytes
        return {
            "resident_bytes": resident_bytes,
            "tensor_count": len(tensors),
            "total_bytes": sum(tensor.size for tensor in tensors),
            "oversized_tensors": len(oversized),
            "largest_row_bytes": largest_row,
            "largest_expert_slice_bytes": largest_expert,
            "dense_rows_fit_resident_budget": rows_fit,
            "experts_fit_resident_budget": experts_fit,
            "can_stream_storage": experts_fit and rows_fit,
            "whole_model_residency_required": False,
            "full_model_loaded": False,
            # Kept as an explicit negative capability: Pyrite streams the model
            # one tensor/expert at a time and never expands the complete model
            # into Python objects at once.
            "can_decode_full_model_in_python": False,
        }

    def _expert_slice_sizes(self) -> list[int]:
        sizes: list[int] = []
        for component in EXPERT_TENSORS:
            tensor = self._tensors.get(f"blk.0.{EXPERT_TENSORS[component]}")
            if tensor is None:
                continue
            block = ggml_spec(tensor.ggml_type)
            inner = tensor.dims[0] * tensor.dims[1] if len(tensor.dims) == 3 else 0
            if inner and inner % block.block_size == 0:
                sizes.append((inner // block.block_size) * block.bytes_per_block)
        return sizes

    def expert_tensor(self, layer: int, expert: int, component: str) -> GGUFTensor:
        """Deprecated accessor kept for compatibility; returns the stacked tensor."""
        if not 0 <= expert < self.config.num_experts:
            raise IndexError(expert)
        return self.stacked_expert_tensor(layer, component)
