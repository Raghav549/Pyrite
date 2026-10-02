"""Build real (tiny) dense ``qwen3``/``llama`` GGUF checkpoints for tests.

Same conventions as :mod:`tests.tiny_qwen3moe`: genuine GGUF files with the
published metadata keys, dense SwiGLU tensors and the same byte-level-BPE
tokenizer metadata, read from disk by the code under test.  Nothing is mocked.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .gguf_builder import GGUFFileBuilder, f32_bytes
from .tiny_qwen3moe import build_tokenizer_metadata


def _rng(seed: int):
    """Small deterministic LCG so tests need no numpy."""
    state = seed & 0xFFFFFFFF

    def next_float() -> float:
        nonlocal state
        state = (1103515245 * state + 12345) & 0x7FFFFFFF
        return (state / 0x7FFFFFFF) * 2.0 - 1.0

    return next_float


def _tensor(next_float, elements: int, scale: float = 0.15) -> bytes:
    return f32_bytes(next_float() * scale for _ in range(elements))


@dataclass(frozen=True)
class TinyDenseConfig:
    architecture: str = "llama"
    hidden_size: int = 32
    intermediate_size: int = 48
    num_hidden_layers: int = 2
    num_attention_heads: int = 4
    num_key_value_heads: int = 2
    head_dim: int = 8
    vocab_size: int = 271
    max_position_embeddings: int = 64
    rope_theta: float = 500000.0
    rms_norm_eps: float = 1e-5
    qk_norm: bool = False

    @property
    def q_proj_dim(self) -> int:
        return self.num_attention_heads * self.head_dim

    @property
    def kv_proj_dim(self) -> int:
        return self.num_key_value_heads * self.head_dim


def qwen3_dense_config(**overrides) -> TinyDenseConfig:
    base = {
        "architecture": "qwen3",
        "rope_theta": 1000000.0,
        "rms_norm_eps": 1e-6,
        "qk_norm": True,
    }
    base.update(overrides)
    return TinyDenseConfig(**base)


def build_tiny_dense_checkpoint(
    path: Path,
    config: TinyDenseConfig | None = None,
    *,
    seed: int = 4321,
    write_output_weight: bool = True,
) -> tuple[Path, TinyDenseConfig, dict[str, bytes]]:
    cfg = config or TinyDenseConfig()
    if cfg.architecture not in ("qwen3", "llama"):
        raise ValueError(f"unsupported dense fixture architecture: {cfg.architecture}")
    builder = GGUFFileBuilder(cfg.architecture)
    next_float = _rng(seed)
    tensors: dict[str, bytes] = {}

    def add(name: str, dims: tuple[int, ...], scale: float = 0.15) -> bytes:
        elements = 1
        for dim in dims:
            elements *= dim
        payload = f32_bytes(next_float() * scale for _ in range(elements))
        builder.add_tensor(name, dims, 0, payload)
        tensors[name] = payload
        return payload

    arch = cfg.architecture
    builder.add(f"{arch}.embedding_length", cfg.hidden_size)
    builder.add(f"{arch}.block_count", cfg.num_hidden_layers)
    builder.add(f"{arch}.feed_forward_length", cfg.intermediate_size)
    builder.add(f"{arch}.attention.head_count", cfg.num_attention_heads)
    builder.add(f"{arch}.attention.head_count_kv", cfg.num_key_value_heads)
    builder.add(f"{arch}.attention.key_length", cfg.head_dim)
    builder.add(f"{arch}.attention.value_length", cfg.head_dim)
    builder.add(f"{arch}.attention.layer_norm_rms_epsilon", cfg.rms_norm_eps)
    builder.add(f"{arch}.rope.freq_base", cfg.rope_theta)
    builder.add(f"{arch}.context_length", cfg.max_position_embeddings)
    builder.add(f"{arch}.vocab_size", cfg.vocab_size)
    builder.add(f"{arch}.tied_word_embeddings", not write_output_weight)
    builder.add(f"{arch}.attention.qk_norm", cfg.qk_norm)
    build_tokenizer_metadata(builder, cfg.vocab_size)

    add("token_embd.weight", (cfg.hidden_size, cfg.vocab_size), 0.2)
    add("output_norm.weight", (cfg.hidden_size,), 0.05)
    if write_output_weight:
        add("output.weight", (cfg.hidden_size, cfg.vocab_size), 0.2)

    for layer in range(cfg.num_hidden_layers):
        prefix = f"blk.{layer}."
        add(prefix + "attn_norm.weight", (cfg.hidden_size,), 0.05)
        add(prefix + "attn_q.weight", (cfg.hidden_size, cfg.q_proj_dim))
        add(prefix + "attn_k.weight", (cfg.hidden_size, cfg.kv_proj_dim))
        add(prefix + "attn_v.weight", (cfg.hidden_size, cfg.kv_proj_dim))
        add(prefix + "attn_output.weight", (cfg.q_proj_dim, cfg.hidden_size))
        if cfg.qk_norm:
            add(prefix + "attn_q_norm.weight", (cfg.head_dim,), 0.05)
            add(prefix + "attn_k_norm.weight", (cfg.head_dim,), 0.05)
        add(prefix + "ffn_norm.weight", (cfg.hidden_size,), 0.05)
        add(prefix + "ffn_gate.weight", (cfg.hidden_size, cfg.intermediate_size))
        add(prefix + "ffn_up.weight", (cfg.hidden_size, cfg.intermediate_size))
        add(prefix + "ffn_down.weight", (cfg.intermediate_size, cfg.hidden_size))

    builder.write(path)
    return path, cfg, tensors
