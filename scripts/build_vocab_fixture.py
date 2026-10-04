"""Build a tiny dense GGUF that carries a *real* tokenizer vocabulary.

The tiny fixtures under ``tests/`` use a hand-written vocabulary, which is fine
for unit tests but useless for comparing against a reference runtime: neither
engine can be driven from the same natural-language prompt with meaningful
token ids.

This script grafts the vocabulary out of a real ``ggml-vocab-*.gguf`` file onto
a minimal Qwen3/LLaMA-shaped model so that ``llama.cpp`` and Pyrite can both be
asked to tokenize and generate from the same text.  The weights are random, so
the *text* is meaningless - the point is that both engines must agree on the
token ids, the metadata interpretation and the arithmetic.

    python scripts/build_vocab_fixture.py \\
        --vocab-gguf /path/to/ggml-vocab-qwen2.gguf \\
        --out models/qwen3-vocab-fixture.gguf
"""
from __future__ import annotations

import argparse
import sys
from array import array
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import random

from pyrite.adapters.gguf import GGUFReader
from tests.gguf_builder import INT32, GGUFFileBuilder
from tests.tiny_dense import TinyDenseConfig

# GGML type ids for the quantizations this script can emit.
QUANT_TYPES: dict[str, int] = {"f32": 0, "q8_0": 8, "q4_0": 2}


def _f16(value: float) -> bytes:
    """IEEE half, little-endian, via struct (no numpy needed)."""
    import struct

    return struct.pack("<e", value)


def _round_half_away(value: float) -> int:
    """C's ``roundf``: halves go away from zero, not to even.

    Python's built-in ``round`` is banker's rounding, so it disagrees with ggml
    on exact halves (``round(0.5) == 0`` but ``roundf(0.5f) == 1``).
    """
    return int(value + 0.5) if value >= 0 else -int(-value + 0.5)


def encode_q8_0(values: list[float]) -> bytes:
    """Mirror ``quantize_row_q8_0_ref`` in ggml/src/ggml-quants.c."""
    out = bytearray()
    for start in range(0, len(values), 32):
        block = values[start : start + 32]
        if len(block) < 32:
            block = block + [0.0] * (32 - len(block))
        amax = max((abs(v) for v in block), default=0.0)
        d = amax / 127.0
        inv = (1.0 / d) if d else 0.0
        out += _f16(d)
        out += bytes(max(-127, min(127, _round_half_away(v * inv))) & 0xFF for v in block)
    return bytes(out)


def encode_q4_0(values: list[float]) -> bytes:
    """Mirror ``quantize_row_q4_0_ref``: ``d = max / -8``, nibbles offset by 8."""
    out = bytearray()
    for start in range(0, len(values), 32):
        block = values[start : start + 32]
        if len(block) < 32:
            block = block + [0.0] * (32 - len(block))
        amax = 0.0
        best = 0.0
        for v in block:
            if amax < abs(v):
                amax = abs(v)
                best = v
        d = best / -8.0
        inv = (1.0 / d) if d else 0.0
        out += _f16(d)
        for j in range(16):
            lo = min(15, max(0, int(block[j] * inv + 8.5)))
            hi = min(15, max(0, int(block[j + 16] * inv + 8.5)))
            out.append(lo | (hi << 4))
    return bytes(out)


QUANT_ENCODERS = {8: encode_q8_0, 2: encode_q4_0}


def load_real_vocab(path: Path) -> dict[str, object]:
    reader = GGUFReader(path)
    metadata = reader.metadata()
    tokens = list(metadata.get("tokenizer.ggml.tokens") or ())
    merges = list(metadata.get("tokenizer.ggml.merges") or ())
    types = list(metadata.get("tokenizer.ggml.token_type") or ())
    if not tokens:
        raise ValueError(f"{path} has no tokenizer.ggml.tokens")
    return {
        "model": metadata.get("tokenizer.ggml.model"),
        "pre": metadata.get("tokenizer.ggml.pre", ""),
        "tokens": tokens,
        "merges": merges,
        "token_types": types or [1] * len(tokens),
        "bos": metadata.get("tokenizer.ggml.bos_token_id", -1),
        "eos": metadata.get("tokenizer.ggml.eos_token_id", -1),
        "unknown": metadata.get("tokenizer.ggml.unknown_token_id", -1),
        "padding": metadata.get("tokenizer.ggml.padding_token_id", -1),
    }


def build(
    out: Path,
    vocab: dict[str, object],
    *,
    architecture: str,
    hidden_size: int,
    num_layers: int,
    intermediate_size: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    max_context: int,
    seed: int,
    num_experts: int = 0,
    num_experts_used: int = 0,
    moe_ffn: int = 0,
    quant: str = "f32",
    keep_f32: tuple[str, ...] = (),
) -> Path:
    tokens = list(vocab["tokens"])  # type: ignore[arg-type]
    vocab_size = len(tokens)
    config = TinyDenseConfig(
        architecture=architecture,
        hidden_size=hidden_size,
        num_hidden_layers=num_layers,
        intermediate_size=intermediate_size,
        num_attention_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        head_dim=head_dim,
        vocab_size=vocab_size,
        max_position_embeddings=max_context,
        qk_norm=architecture == "qwen3",
    )
    builder = GGUFFileBuilder(architecture)
    next_float = random.Random(seed).random  # noqa: S311 - fixture weights, not security  # C-implemented: fast enough for GB fixtures

    def add(name: str, dims: tuple[int, ...], scale: float) -> None:
        elements = 1
        for dim in dims:
            elements *= dim

        from pyrite.ggml_types import spec

        ggml_type = QUANT_TYPES[quant]
        if ggml_type != 0 and name.endswith("_norm.weight"):
            # RMSNorm weights are multiplied element-wise against f32
            # activations, and ggml's CPU binary ops refuse a mixed-type
            # operand ("binary_op: unsupported types: dst f32, src1 q8_0").
            # Every real quantized GGUF leaves the norm vectors in F32 for
            # exactly this reason, so this fixture does too.
            ggml_type = 0
        if ggml_type != 0 and any(name.startswith(prefix) for prefix in keep_f32):
            ggml_type = 0
        if ggml_type != 0:
            # ggml quantizes block by block *within a row*, so it is the row
            # length (dims[0]) that has to be a multiple of the block size; a
            # total element count that happens to divide is not enough.  Real
            # quantized GGUFs leave exactly the tensors that fail this test -
            # the small norm vectors - in F32, and so does llama.cpp's loader.
            row_length = dims[0] if dims else 1
            if row_length % spec(ggml_type).block_size:
                ggml_type = 0

        def produce(_name=name, _elements=elements, _scale=scale, _type=ggml_type) -> bytes:
            # Generated when the file is written, so peak memory is one tensor.
            values = [
                (next_float() * 2.0 - 1.0) * _scale for _ in range(_elements)
            ]
            if _type == 0:
                return array("f", values).tobytes()
            return QUANT_ENCODERS[_type](values)

        builder.add_tensor_lazy(name, dims, ggml_type, produce)

    is_moe = architecture == "qwen3moe"
    if is_moe and min(num_experts, num_experts_used, moe_ffn) <= 0:
        raise ValueError(
            "qwen3moe needs --experts, --experts-used and --moe-ffn to all be positive"
        )
    builder.add(f"{architecture}.embedding_length", config.hidden_size)
    builder.add(f"{architecture}.block_count", config.num_hidden_layers)
    builder.add(
        f"{architecture}.feed_forward_length",
        moe_ffn * num_experts_used if is_moe else config.intermediate_size,
    )
    if is_moe:
        builder.add(f"{architecture}.expert_feed_forward_length", moe_ffn)
        builder.add(f"{architecture}.expert_count", num_experts)
        builder.add(f"{architecture}.expert_used_count", num_experts_used)
        # llama.cpp normalises the router weights for Qwen3-MoE.
        builder.add(f"{architecture}.expert_weights_norm", True)
        builder.add(f"{architecture}.expert_weights_scale", 1.0)
    builder.add(f"{architecture}.attention.head_count", config.num_attention_heads)
    builder.add(f"{architecture}.attention.head_count_kv", config.num_key_value_heads)
    builder.add(f"{architecture}.attention.key_length", config.head_dim)
    builder.add(f"{architecture}.attention.value_length", config.head_dim)
    builder.add(f"{architecture}.attention.layer_norm_rms_epsilon", config.rms_norm_eps)
    builder.add(f"{architecture}.rope.freq_base", config.rope_theta)
    builder.add(f"{architecture}.context_length", config.max_position_embeddings)
    builder.add(f"{architecture}.vocab_size", vocab_size)
    builder.add(f"{architecture}.tied_word_embeddings", False)
    if architecture in ("qwen3", "qwen3moe"):
        builder.add(f"{architecture}.attention.qk_norm", True)

    builder.add("tokenizer.ggml.model", vocab["model"])
    builder.add("tokenizer.ggml.pre", vocab["pre"])
    builder.add("tokenizer.ggml.tokens", tokens)
    builder.add("tokenizer.ggml.merges", list(vocab["merges"]))  # type: ignore[arg-type]
    builder.add("tokenizer.ggml.token_type", list(vocab["token_types"]), INT32)
    builder.add("tokenizer.ggml.bos_token_id", int(vocab["bos"]))  # type: ignore[arg-type]
    builder.add("tokenizer.ggml.eos_token_id", int(vocab["eos"]))  # type: ignore[arg-type]

    add("token_embd.weight", (config.hidden_size, vocab_size), 0.2)
    add("output_norm.weight", (config.hidden_size,), 0.05)
    add("output.weight", (config.hidden_size, vocab_size), 0.2)
    for layer in range(config.num_hidden_layers):
        prefix = f"blk.{layer}."
        add(f"{prefix}attn_norm.weight", (config.hidden_size,), 0.05)
        q_dim = config.head_dim * config.num_attention_heads
        kv_dim = config.head_dim * config.num_key_value_heads
        add(f"{prefix}attn_q.weight", (config.hidden_size, q_dim), 0.15)
        add(f"{prefix}attn_k.weight", (config.hidden_size, kv_dim), 0.15)
        add(f"{prefix}attn_v.weight", (config.hidden_size, kv_dim), 0.15)
        add(f"{prefix}attn_output.weight", (q_dim, config.hidden_size), 0.15)
        if architecture in ("qwen3", "qwen3moe"):
            add(f"{prefix}attn_q_norm.weight", (config.head_dim,), 0.05)
            add(f"{prefix}attn_k_norm.weight", (config.head_dim,), 0.05)
        add(f"{prefix}ffn_norm.weight", (config.hidden_size,), 0.05)
        if is_moe:
            # llama.cpp's qwen3moe loads no shared-expert tensor: routed experts
            # only, via a stacked 3D tensor per projection.
            add(f"{prefix}ffn_gate_inp.weight", (config.hidden_size, num_experts), 0.15)
            add(
                f"{prefix}ffn_gate_exps.weight",
                (config.hidden_size, moe_ffn, num_experts),
                0.15,
            )
            add(
                f"{prefix}ffn_up_exps.weight",
                (config.hidden_size, moe_ffn, num_experts),
                0.15,
            )
            add(
                f"{prefix}ffn_down_exps.weight",
                (moe_ffn, config.hidden_size, num_experts),
                0.15,
            )
        else:
            add(f"{prefix}ffn_gate.weight", (config.hidden_size, config.intermediate_size), 0.15)
            add(f"{prefix}ffn_up.weight", (config.hidden_size, config.intermediate_size), 0.15)
            add(
                f"{prefix}ffn_down.weight",
                (config.intermediate_size, config.hidden_size),
                0.15,
            )

    out.parent.mkdir(parents=True, exist_ok=True)
    builder.write_streaming(out)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vocab-gguf", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--architecture", default="qwen3")
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--intermediate-size", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-kv-heads", type=int, default=2)
    parser.add_argument("--head-dim", type=int, default=16)
    parser.add_argument("--max-context", type=int, default=512)
    parser.add_argument("--seed", type=int, default=4321)
    parser.add_argument("--experts", type=int, default=0, help="qwen3moe routed expert count")
    parser.add_argument("--experts-used", type=int, default=0, help="qwen3moe top-k")
    parser.add_argument("--moe-ffn", type=int, default=0, help="qwen3moe per-expert FFN width")
    parser.add_argument(
        "--quant",
        default="f32",
        choices=tuple(QUANT_TYPES),
        help="GGML quantization for the weight tensors",
    )
    parser.add_argument(
        "--keep-f32",
        default="",
        help="comma-separated tensor-name prefixes to leave in F32",
    )
    args = parser.parse_args(argv)

    vocab = load_real_vocab(args.vocab_gguf)
    path = build(
        args.out,
        vocab,
        architecture=args.architecture,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        intermediate_size=args.intermediate_size,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        head_dim=args.head_dim,
        max_context=args.max_context,
        seed=args.seed,
        num_experts=args.experts,
        num_experts_used=args.experts_used,
        moe_ffn=args.moe_ffn,
        quant=args.quant,
        keep_f32=tuple(args.keep_f32.split(",")) if args.keep_f32 else (),
    )
    print(
        f"wrote {path} ({path.stat().st_size:,} bytes) arch={args.architecture} "
        f"vocab={len(vocab['tokens'])} pre={vocab['pre']!r} "
        f"layers={args.num_layers} hidden={args.hidden_size} quant={args.quant}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
