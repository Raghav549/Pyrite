"""Qwen3-MoE checkpoint metadata and routing support.

This module intentionally stops at the model-contract boundary until a native
streaming tensor kernel is present. It must never silently substitute a fake
tokenizer or fake transformer for a real checkpoint.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from .adapters.gguf import GGUFTensor, GGUFReader


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
    rms_norm_eps: float
    rope_theta: float
    max_position_embeddings: int
    bos_token_id: int | None
    eos_token_id: int | None
    norm_topk_prob: bool
    tie_word_embeddings: bool

    @classmethod
    def from_metadata(cls, metadata: Mapping[str, object]) -> "Qwen3MoEConfig":
        def get(key: str, default: object = None) -> object:
            return metadata.get(key, default)

        arch = str(get("general.architecture", ""))
        if arch != "qwen3moe":
            raise ValueError(f"unsupported GGUF architecture: {arch or 'unknown'}")

        def n(suffix: str, default: object = None) -> object:
            return get(f"qwen3moe.{suffix}", default)

        hidden = int(n("embedding_length", 0))
        layers = int(n("block_count", 0))
        heads = int(n("attention.head_count", 0))
        kv_heads = int(n("attention.head_count_kv", 0))
        intermediate = int(n("feed_forward_length", 0))
        moe_intermediate = int(n("expert_feed_forward_length", 0))
        experts = int(n("expert_count", 0))
        per_tok = int(n("expert_used_count", 0))
        vocab = int(get("tokenizer.ggml.vocab_size", 0))
        head_dim = int(n("attention.key_length", hidden // max(1, heads)))
        max_pos = int(n("context_length", 0))
        eps = float(n("attention.layer_norm_rms_epsilon", 1e-6))
        rope_theta = float(n("rope.freq_base", 10000.0))
        bos = get("tokenizer.ggml.bos_token_id")
        eos = get("tokenizer.ggml.eos_token_id")
        tied = bool(n("tied_word_embeddings", False))
        norm_topk = bool(n("expert_weights_normalized", True))

        required = (hidden, layers, heads, kv_heads, intermediate,
                    moe_intermediate, experts, per_tok, vocab, head_dim)
        if any(value <= 0 for value in required):
            raise ValueError("GGUF is missing required Qwen3-MoE metadata")
        if heads % kv_heads:
            raise ValueError("attention head count must be divisible by KV head count")

        return cls(
            hidden, intermediate, moe_intermediate, layers, heads, kv_heads,
            experts, per_tok, vocab, head_dim, eps, rope_theta, max_pos,
            int(bos) if bos is not None else None,
            int(eos) if eos is not None else None,
            norm_topk, tied,
        )


class Qwen3MoECheckpoint:
    """Inspect and route a Qwen3-MoE GGUF checkpoint without fake inference."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.reader = GGUFReader(self.path)
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

    def expert_tensor(self, layer: int, expert: int, component: str) -> GGUFTensor:
        if not 0 <= layer < self.config.num_hidden_layers:
            raise IndexError(layer)
        if not 0 <= expert < self.config.num_experts:
            raise IndexError(expert)
        aliases = (
            f"blk.{layer}.ffn_{component}.{expert}.weight",
            f"blk.{layer}.ffn_{component}.{expert}",
        )
        for name in aliases:
            tensor = self._tensors.get(name)
            if tensor is not None:
                return tensor
        raise KeyError(f"expert tensor not found for layer={layer}, expert={expert}, component={component}")

    def routing_summary(self) -> dict[str, int | bool | str]:
        cfg = self.config
        attention_layers = 0
        moe_layers = 0
        for layer in range(cfg.num_hidden_layers):
            prefix = f"blk.{layer}."
            attention_layers += int(all(
                f"{prefix}{name}" in self._tensors
                for name in ("attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_output.weight")
            ))
            moe_layers += int(any(
                name.startswith(prefix + "ffn_") for name in self._tensors
            ))
        return {
            "architecture": "qwen3moe",
            "layers": cfg.num_hidden_layers,
            "experts": cfg.num_experts,
            "experts_per_token": cfg.num_experts_per_tok,
            "attention_complete_layers": attention_layers,
            "moe_layers_detected": moe_layers,
            "native_generation_ready": False,
        }

    def validate_for_streaming(self, resident_bytes: int) -> dict[str, object]:
        if resident_bytes <= 0:
            raise ValueError("resident_bytes must be positive")
        oversized = [
            tensor for tensor in self._tensors.values()
            if tensor.size > resident_bytes
        ]
        return {
            "resident_bytes": resident_bytes,
            "tensor_count": len(self._tensors),
            "oversized_tensors": len(oversized),
            "can_stream_storage": True,
            "can_decode_full_model_in_python": False,
        }
