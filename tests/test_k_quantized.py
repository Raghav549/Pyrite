"""K-quant execution: naive-encoder payloads verified against the oracle, then run.

:class:`tests.k_quant` produces valid K-quant bytes (packing ported from the
reference ``ggml`` quantizers); every assertion about *values* here is checked
against the independent ``gguf`` dequantizer, and execution on a K-quantized
checkpoint is checked against the naive reference running on oracle-decoded
weights — the same pattern as the simple-quant tests.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from pyrite.config import RuntimeConfig
from pyrite.engine import PyriteRuntime
from pyrite.executor import Qwen3MoEExecutor
from pyrite.tensor_ops import decode_vector

from .gguf_builder import GGUFFileBuilder
from .k_quant import K_ENCODERS
from .tiny_qwen3moe import TinyConfig, build_tiny_checkpoint
from .tiny_reference import NaiveQwen3MoE

K_TYPE_NAMES = {10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K"}


def _oracle():
    quants = pytest.importorskip("gguf.quants", reason="decoder oracle")
    np = pytest.importorskip("numpy", reason="the gguf oracle needs numpy")
    return quants, np


def _k_config() -> TinyConfig:
    return TinyConfig(
        hidden_size=256,
        num_hidden_layers=1,
        num_attention_heads=8,
        num_key_value_heads=8,
        head_dim=64,
        moe_intermediate_size=256,
        num_experts=2,
        num_experts_per_tok=2,
        vocab_size=271,
    )


def _runtime() -> PyriteRuntime:
    return PyriteRuntime(RuntimeConfig(ram_budget_mb=1024, reserve_mb=768))


def k_requantize_checkpoint(src: Path, dst: Path, ggml_type: int) -> Path:
    """K-encode every 256-aligned F32 tensor of ``src`` with the naive encoder."""
    import struct

    from pyrite.adapters.gguf import GGUFReader

    reader = GGUFReader(src)
    builder = GGUFFileBuilder("qwen3moe")
    for key, value in reader.metadata().items():
        if key not in {"general.architecture", "general.alignment"}:
            builder.add(key, value)
    encoder = K_ENCODERS[ggml_type]
    converted = 0
    for tensor in reader.tensor_index():
        payload = reader.read_tensor(tensor)
        if (
            tensor.ggml_type == 0
            and tensor.element_count > 0
            and tensor.element_count % 256 == 0
        ):
            values = list(struct.unpack(f"<{tensor.element_count}f", payload))
            builder.add_tensor(tensor.name, tensor.dims, ggml_type, encoder(values))
            converted += 1
        else:
            builder.add_tensor(tensor.name, tensor.dims, tensor.ggml_type, payload)
    assert converted >= 10, "expected most fixture tensors to be K-quantized"
    return builder.write(dst)


def oracle_dequantize_checkpoint(src: Path, dst: Path, qtype_name: str) -> Path:
    """Decode ``src`` with the oracle and store the result as F32."""
    from pyrite.adapters.gguf import GGUFReader

    quants, np = _oracle()
    reader = GGUFReader(src)
    builder = GGUFFileBuilder("qwen3moe")
    for key, value in reader.metadata().items():
        if key not in {"general.architecture", "general.alignment"}:
            builder.add(key, value)
    for tensor in reader.tensor_index():
        payload = reader.read_tensor(tensor)
        if tensor.ggml_type != 0:
            raw = np.frombuffer(payload, dtype=np.uint8).copy()
            decoded = quants.dequantize(raw, getattr(quants.GGMLQuantizationType, qtype_name))
            builder.add_tensor(tensor.name, tensor.dims, 0, decoded.astype(np.float32).tobytes())
        else:
            builder.add_tensor(tensor.name, tensor.dims, tensor.ggml_type, payload)
    return builder.write(dst)


@pytest.mark.parametrize("ggml_type", sorted(K_ENCODERS))
def test_k_decoders_match_oracle_on_encoder_bytes(tmp_path: Path, ggml_type: int) -> None:
    from pyrite.adapters.gguf import GGUFReader

    quants, np = _oracle()
    src, _, _ = build_tiny_checkpoint(tmp_path / "tiny.gguf", _k_config())
    quantized = k_requantize_checkpoint(src, tmp_path / "k.gguf", ggml_type)
    reader = GGUFReader(quantized)
    checked = 0
    for tensor in reader.tensor_index():
        if tensor.ggml_type != ggml_type:
            continue
        payload = reader.read_tensor(tensor)
        expected = list(
            quants.dequantize(
                np.frombuffer(payload, dtype=np.uint8).copy(),
                getattr(quants.GGMLQuantizationType, K_TYPE_NAMES[ggml_type]),
            )
        )
        got = decode_vector(tensor.ggml_type, payload, tensor.element_count)
        worst = max(abs(a - b) for a, b in zip(got, expected, strict=True))
        assert worst <= 1e-6, f"{tensor.name}: max abs diff {worst}"
        checked += 1
    assert checked >= 10


@pytest.mark.parametrize("ggml_type", sorted(K_ENCODERS))
def test_k_quantized_execution_matches_oracle_decoded_reference(
    tmp_path: Path, ggml_type: int
) -> None:
    src, _, _ = build_tiny_checkpoint(tmp_path / "tiny.gguf", _k_config())
    quantized = k_requantize_checkpoint(src, tmp_path / "k.gguf", ggml_type)
    dequantized = oracle_dequantize_checkpoint(quantized, tmp_path / "deq.gguf", K_TYPE_NAMES[ggml_type])
    reference = NaiveQwen3MoE(dequantized)
    tokens = [3, 17, 42]
    # Bound justification: the same comparison on the pure-F32 K-fixture
    # (no quantization at all) already measures ~1.2e-3 from float
    # accumulation order across 256-wide projections, so 5e-3 keeps 4x
    # headroom while still catching any layout bug (wrong nibble order or
    # scales shift logits by 1e-1 or more).
    with Qwen3MoEExecutor(quantized, runtime=_runtime()) as executor:
        assert executor.unsupported_tensor_types() == ()
        for position, token in enumerate(tokens):
            got = executor.logits(token, position)
            expected = reference.logits(tokens[: position + 1])
            worst = max(abs(a - b) for a, b in zip(got, expected, strict=True))
            assert worst <= 5e-3, f"position {position}: max abs diff {worst}"
        assert executor.stats.tensors_streamed > 0
        assert executor.stats.expert_slices_streamed > 0


@pytest.mark.parametrize("ggml_type", sorted(K_ENCODERS))
def test_k_quantized_generation_is_deterministic_and_valid(
    tmp_path: Path, ggml_type: int
) -> None:
    from pyrite.qwen3_moe import Qwen3MoECheckpoint

    src, _, _ = build_tiny_checkpoint(tmp_path / "tiny.gguf", _k_config())
    quantized = k_requantize_checkpoint(src, tmp_path / "k.gguf", ggml_type)
    vocab = Qwen3MoECheckpoint(quantized).config.vocab_size
    with Qwen3MoEExecutor(quantized, runtime=_runtime(), sampler_seed=0) as executor:
        first = executor.generate([5, 9, 11], max_new_tokens=3, temperature=0.0)
        second = executor.generate([5, 9, 11], max_new_tokens=3, temperature=0.0)
        text = executor.generate_text("hello world", max_new_tokens=2, temperature=0.0)
    assert first == second
    assert all(0 <= token < vocab for token in first)
    assert text["new_tokens"] == 2
    assert text["text"].startswith("hello world")
