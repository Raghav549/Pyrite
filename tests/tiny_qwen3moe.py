"""Build a real (tiny) Qwen3-MoE GGUF checkpoint for end-to-end tests.

Everything here produces actual bytes in the published GGUF/layout conventions:
the same metadata keys, the same stacked 3D expert tensors, the same
byte-level-BPE tokenizer metadata.  Nothing is mocked or monkeypatched; the
executor under test reads this file from disk.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pyrite.ggml_types import spec as ggml_spec
from pyrite.ggml_types import tensor_size
from pyrite.tokenizer import bytes_to_unicode

from .gguf_builder import INT32, GGUFFileBuilder, f32_bytes


@dataclass(frozen=True)
class TinyConfig:
    hidden_size: int = 32
    num_hidden_layers: int = 2
    num_attention_heads: int = 4
    num_key_value_heads: int = 2
    head_dim: int = 8
    moe_intermediate_size: int = 16
    num_experts: int = 4
    num_experts_per_tok: int = 2
    vocab_size: int = 271  # 256 byte symbols + 13 merges + 2 specials
    max_position_embeddings: int = 64

    @property
    def q_proj_dim(self) -> int:
        return self.num_attention_heads * self.head_dim

    @property
    def kv_proj_dim(self) -> int:
        return self.num_key_value_heads * self.head_dim


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


# Merge ranks must be consistent: a longer merge has to outrank the pair it
# depends on (real vocabularies are built that way; "Ġ w" < "w o").
MERGES = [
    "Ġ w", "Ġw o", "Ġwo r", "Ġwor l", "Ġworl d",
    "h e", "he l", "hel l", "hell o",
    "w o", "wo r", "wor l", "worl d",
]
SPECIAL_TOKENS = ["<|im_start|>", "<|im_end|>"]


def tokenizer_vocab() -> tuple[list[str], list[str], list[int], int, int]:
    """Real byte-level BPE vocabulary: base alphabet + merges + specials."""
    vocab = [bytes_to_unicode()[byte] for byte in range(256)]
    for merge in MERGES:
        left, right = merge.split(" ")
        vocab.append(left + right)
    vocab.extend(SPECIAL_TOKENS)
    token_types = [1] * (len(vocab) - len(SPECIAL_TOKENS)) + [3] * len(SPECIAL_TOKENS)
    bos = len(vocab) - 2
    eos = len(vocab) - 1
    return vocab, list(MERGES), token_types, bos, eos


def build_tokenizer_metadata(builder: GGUFFileBuilder, vocab_size: int | None = None) -> None:
    vocab, merges, token_types, bos, eos = tokenizer_vocab()
    if vocab_size is not None and vocab_size > len(vocab):
        # Pad with reserved normal tokens so the declared vocabulary matches
        # embedding/output rows on larger-than-tiny fixtures.  llama.cpp
        # derives the expected vocab from this list and refuses mismatches.
        extra = vocab_size - len(vocab)
        vocab = vocab + [f"<reserved_{i}>" for i in range(extra)]
        token_types = token_types + [1] * extra
    elif vocab_size is not None and vocab_size != len(vocab):
        raise ValueError(f"cannot shrink the fixture vocabulary to {vocab_size}")
    builder.add("tokenizer.ggml.model", "gpt2")
    builder.add("tokenizer.ggml.pre", "qwen2")
    builder.add("tokenizer.ggml.tokens", vocab)
    builder.add("tokenizer.ggml.merges", merges)
    # Real GGUFs store token types as int32; llama.cpp refuses uint32 here.
    builder.add("tokenizer.ggml.token_type", token_types, INT32)
    builder.add("tokenizer.ggml.bos_token_id", bos)
    builder.add("tokenizer.ggml.eos_token_id", eos)


def build_tiny_checkpoint(
    path: Path,
    config: TinyConfig | None = None,
    *,
    seed: int = 1234,
    write_output_weight: bool = True,
    tokenizer: bool = True,
    ggml_type: int = 0,
) -> tuple[Path, TinyConfig, dict[str, bytes]]:
    """Write a tiny Qwen3-MoE GGUF; returns (path, config, raw tensors).

    ``ggml_type`` overrides the element type of every tensor.  The weights are
    then zero payloads of the right size, which is enough to exercise the
    contract/type checks without paying to quantize a fixture.
    """
    cfg = config or TinyConfig()
    builder = GGUFFileBuilder("qwen3moe")
    next_float = _rng(seed)
    tensors: dict[str, bytes] = {}

    def add(name: str, dims: tuple[int, ...], scale: float = 0.15) -> bytes:
        elements = _prod(dims)
        # Norms and other small vectors stay F32, exactly like shipped files;
        # only block-aligned matrices take the requested quantized type.
        block_size = ggml_spec(ggml_type).block_size
        use_type = ggml_type if dims and dims[0] % block_size == 0 and elements % block_size == 0 else 0
        if use_type == 0:
            payload = _tensor(next_float, elements, scale)
        else:
            payload = bytes(tensor_size(elements, use_type))
        builder.add_tensor(name, dims, use_type, payload)
        tensors[name] = payload
        return payload

    builder.add("qwen3moe.embedding_length", cfg.hidden_size)
    builder.add("qwen3moe.block_count", cfg.num_hidden_layers)
    builder.add("qwen3moe.feed_forward_length", cfg.moe_intermediate_size * cfg.num_experts_per_tok)
    builder.add("qwen3moe.expert_feed_forward_length", cfg.moe_intermediate_size)
    builder.add("qwen3moe.expert_count", cfg.num_experts)
    builder.add("qwen3moe.expert_used_count", cfg.num_experts_per_tok)
    builder.add("qwen3moe.expert_weights_norm", True)
    builder.add("qwen3moe.expert_weights_scale", 1.0)
    builder.add("qwen3moe.attention.head_count", cfg.num_attention_heads)
    builder.add("qwen3moe.attention.head_count_kv", cfg.num_key_value_heads)
    builder.add("qwen3moe.attention.key_length", cfg.head_dim)
    builder.add("qwen3moe.attention.value_length", cfg.head_dim)
    builder.add("qwen3moe.attention.layer_norm_rms_epsilon", 1e-6)
    builder.add("qwen3moe.rope.freq_base", 10000.0)
    builder.add("qwen3moe.context_length", cfg.max_position_embeddings)
    builder.add("qwen3moe.vocab_size", cfg.vocab_size)
    builder.add("qwen3moe.tied_word_embeddings", not write_output_weight)
    if tokenizer:
        build_tokenizer_metadata(builder)

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
        add(prefix + "attn_q_norm.weight", (cfg.head_dim,), 0.05)
        add(prefix + "attn_k_norm.weight", (cfg.head_dim,), 0.05)
        add(prefix + "ffn_norm.weight", (cfg.hidden_size,), 0.05)
        add(prefix + "ffn_gate_inp.weight", (cfg.hidden_size, cfg.num_experts))
        add(prefix + "ffn_gate_exps.weight", (cfg.hidden_size, cfg.moe_intermediate_size, cfg.num_experts))
        add(prefix + "ffn_up_exps.weight", (cfg.hidden_size, cfg.moe_intermediate_size, cfg.num_experts))
        add(prefix + "ffn_down_exps.weight", (cfg.moe_intermediate_size, cfg.hidden_size, cfg.num_experts))

    builder.write(path)
    return path, cfg, tensors


def _prod(dims: tuple[int, ...]) -> int:
    total = 1
    for dim in dims:
        total *= dim
    return total


def make_test_tokenizer():
    """Build the tokenizer the same way the runtime loads it from GGUF."""
    from pyrite.tokenizer import GGUFBPETokenizer

    vocab, merges, token_types, bos, eos = tokenizer_vocab()
    return GGUFBPETokenizer(vocab, merges, token_types=token_types, bos_id=bos, eos_id=eos)


