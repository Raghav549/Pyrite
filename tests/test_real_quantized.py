"""Real-quantizer validation: official ``gguf`` quantizer output, executed by Pyrite.

The weight *values* in these fixtures are synthetic, but the *quantization* is
not: tensors are quantized with the reference ``gguf`` package (the same
layouts ``llama.cpp`` ships), written to genuine GGUF files, then parsed,
decoded, executed and generated from by Pyrite with no mocks.  Decode parity is
checked against the independent ``gguf`` dequantizer, and execution is checked
against the naive reference running on oracle-dequantized weights.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from pyrite.backends.real import GGUFReferenceBackend
from pyrite.config import RuntimeConfig
from pyrite.engine import PyriteRuntime
from pyrite.executor import Qwen3MoEExecutor
from pyrite.tensor_ops import decode_vector

from .gguf_builder import GGUFFileBuilder
from .tiny_qwen3moe import build_tiny_checkpoint
from .tiny_reference import NaiveQwen3MoE

# gguf block size for the simple (non-K) quants exercised here.
SIMPLE_BLOCK = 32


def _oracle():
    quants = pytest.importorskip("gguf.quants", reason="official quantizer oracle")
    np = pytest.importorskip("numpy", reason="the gguf oracle needs numpy")
    return quants, np


def _runtime() -> PyriteRuntime:
    return PyriteRuntime(RuntimeConfig(ram_budget_mb=1024, reserve_mb=768))


def _copy_metadata(builder: GGUFFileBuilder, reader) -> None:
    for key, value in reader.metadata().items():
        if key not in {"general.architecture", "general.alignment"}:
            builder.add(key, value)


def requantize_checkpoint(src: Path, dst: Path, qtype_name: str, ggml_type: int) -> Path:
    """Re-quantize every block-aligned F32 tensor of ``src`` with ``gguf``."""
    from pyrite.adapters.gguf import GGUFReader

    quants, np = _oracle()
    qtype = getattr(quants.GGMLQuantizationType, qtype_name)
    reader = GGUFReader(src)
    builder = GGUFFileBuilder("qwen3moe")
    _copy_metadata(builder, reader)
    converted = 0
    for tensor in reader.tensor_index():
        payload = reader.read_tensor(tensor)
        if (
            tensor.ggml_type == 0
            and tensor.element_count > 0
            and tensor.element_count % SIMPLE_BLOCK == 0
        ):
            array = np.frombuffer(payload, dtype=np.float32).copy()
            payload = quants.quantize(array, qtype).tobytes()
            builder.add_tensor(tensor.name, tensor.dims, ggml_type, payload)
            converted += 1
        else:
            builder.add_tensor(tensor.name, tensor.dims, tensor.ggml_type, payload)
    assert converted > 0, "fixture has no quantizable tensors"
    return builder.write(dst)


def dequantize_checkpoint_to_f32(src: Path, dst: Path, qtype_name: str) -> Path:
    """Decode ``src`` with the oracle and store the result as F32.

    The naive reference only understands F32, so this is how execution on a
    quantized file is cross-checked: Pyrite runs the quantized bytes while the
    reference runs the oracle-decoded weights.
    """
    from pyrite.adapters.gguf import GGUFReader

    quants, np = _oracle()
    qtype = getattr(quants.GGMLQuantizationType, qtype_name)
    reader = GGUFReader(src)
    builder = GGUFFileBuilder("qwen3moe")
    _copy_metadata(builder, reader)
    for tensor in reader.tensor_index():
        payload = reader.read_tensor(tensor)
        if tensor.ggml_type != 0:
            raw = np.frombuffer(payload, dtype=np.uint8).copy()
            payload = quants.dequantize(raw, qtype).astype(np.float32).tobytes()
            builder.add_tensor(tensor.name, tensor.dims, 0, payload)
        else:
            builder.add_tensor(tensor.name, tensor.dims, tensor.ggml_type, payload)
    return builder.write(dst)


@pytest.mark.parametrize(
    ("qtype_name", "ggml_type"),
    [("Q8_0", 8), ("Q4_0", 2), ("Q4_1", 3), ("Q5_0", 6), ("Q5_1", 7)],
)
def test_pyrite_decoders_match_oracle_on_real_quantizer_bytes(
    tmp_path: Path, qtype_name: str, ggml_type: int
) -> None:
    from pyrite.adapters.gguf import GGUFReader

    quants, np = _oracle()
    src, _, _ = build_tiny_checkpoint(tmp_path / "tiny.gguf")
    quantized = requantize_checkpoint(src, tmp_path / "q.gguf", qtype_name, ggml_type)
    reader = GGUFReader(quantized)
    checked = 0
    for tensor in reader.tensor_index():
        if tensor.ggml_type != ggml_type:
            continue
        payload = reader.read_tensor(tensor)
        expected = list(
            quants.dequantize(
                np.frombuffer(payload, dtype=np.uint8).copy(),
                getattr(quants.GGMLQuantizationType, qtype_name),
            )
        )
        got = decode_vector(tensor.ggml_type, payload, tensor.element_count)
        assert len(got) == len(expected) == tensor.element_count
        worst = max(abs(a - b) for a, b in zip(got, expected, strict=True))
        assert worst <= 1e-6, f"{tensor.name}: max abs diff {worst}"
        checked += 1
    assert checked >= 20, "expected most fixture tensors to be quantized"


@pytest.mark.parametrize(("qtype_name", "ggml_type"), [("Q8_0", 8), ("Q4_0", 2)])
def test_quantized_execution_matches_oracle_decoded_reference(
    tmp_path: Path, qtype_name: str, ggml_type: int
) -> None:
    src, _, _ = build_tiny_checkpoint(tmp_path / "tiny.gguf")
    quantized = requantize_checkpoint(src, tmp_path / "q.gguf", qtype_name, ggml_type)
    dequantized = dequantize_checkpoint_to_f32(quantized, tmp_path / "deq.gguf", qtype_name)
    reference = NaiveQwen3MoE(dequantized)
    tokens = [3, 17, 42]
    with Qwen3MoEExecutor(quantized, runtime=_runtime()) as executor:
        assert executor.unsupported_tensor_types() == ()
        for position, token in enumerate(tokens):
            got = executor.logits(token, position)
            expected = reference.logits(tokens[: position + 1])
            worst = max(abs(a - b) for a, b in zip(got, expected, strict=True))
            assert worst <= 1e-3, f"position {position}: max abs diff {worst}"
        assert executor.stats.tensors_streamed > 0
        assert executor.stats.bytes_loaded > 0


@pytest.mark.parametrize(("qtype_name", "ggml_type"), [("Q8_0", 8), ("Q4_0", 2)])
def test_quantized_generation_is_deterministic_and_valid(
    tmp_path: Path, qtype_name: str, ggml_type: int
) -> None:
    from pyrite.qwen3_moe import Qwen3MoECheckpoint

    src, _, _ = build_tiny_checkpoint(tmp_path / "tiny.gguf")
    quantized = requantize_checkpoint(src, tmp_path / "q.gguf", qtype_name, ggml_type)
    vocab = Qwen3MoECheckpoint(quantized).config.vocab_size
    with Qwen3MoEExecutor(quantized, runtime=_runtime(), sampler_seed=0) as executor:
        first = executor.generate([5, 9, 11], max_new_tokens=4, temperature=0.0)
        second = executor.generate([5, 9, 11], max_new_tokens=4, temperature=0.0)
        text = executor.generate_text("hello world", max_new_tokens=2, temperature=0.0)
    assert first == second
    assert len(first) == 3 + 4
    assert all(0 <= token < vocab for token in first)
    assert text["new_tokens"] == 2
    assert text["text"].startswith("hello world")


def test_quantized_matvec_matches_the_numpy_oracle(tmp_path: Path) -> None:
    quants, np = _oracle()
    src, _, _ = build_tiny_checkpoint(tmp_path / "tiny.gguf")
    quantized = requantize_checkpoint(src, tmp_path / "q.gguf", "Q8_0", 8)
    backend = GGUFReferenceBackend(quantized)
    name = "blk.0.attn_q.weight"
    tensor = backend.reader.tensor(name)
    rng = np.random.RandomState(7)
    vector = rng.rand(tensor.dims[0]).astype(np.float64)
    got = backend.matvec_tensor(name, [float(v) for v in vector])
    raw = np.frombuffer(backend.load(name), dtype=np.uint8).copy()
    weights = np.array(
        quants.dequantize(raw, quants.GGMLQuantizationType.Q8_0), dtype=np.float64
    ).reshape(tensor.dims[1], tensor.dims[0])
    expected = weights @ vector
    worst = max(abs(a - b) for a, b in zip(got, expected, strict=True))
    assert worst <= 1e-4, f"max abs diff {worst}"
