"""Dense decoder-only path: contract, execution, quantized execution, dispatch."""
from __future__ import annotations

import math
from pathlib import Path

import pytest

from pyrite.config import RuntimeConfig
from pyrite.dense import DenseCheckpoint, DenseConfig, DenseContractError, DenseExecutor
from pyrite.engine import PyriteRuntime
from pyrite.executor import detect_architecture, open_executor

from .tiny_dense import TinyDenseConfig, build_tiny_dense_checkpoint, qwen3_dense_config


def _runtime(**overrides) -> PyriteRuntime:
    config = RuntimeConfig(ram_budget_mb=1024, reserve_mb=768, **overrides)
    return PyriteRuntime(config)


@pytest.mark.parametrize("arch", ["llama", "qwen3"])
def test_dense_contract_and_report(tmp_path: Path, arch: str):
    config = TinyDenseConfig() if arch == "llama" else qwen3_dense_config()
    path, cfg, _ = build_tiny_dense_checkpoint(tmp_path / "dense.gguf", config)
    checkpoint = DenseCheckpoint(path)
    assert checkpoint.config.architecture == arch
    assert checkpoint.config.vocab_size == cfg.vocab_size
    assert checkpoint.config.q_proj_dim == cfg.q_proj_dim
    assert checkpoint.config.kv_proj_dim == cfg.kv_proj_dim
    summary = checkpoint.validate_contract()
    assert summary["native_generation_ready"] is True
    per_layer = 9 + (2 if cfg.qk_norm else 0)
    assert checkpoint.tensor_count == 3 + cfg.num_hidden_layers * per_layer


def test_dense_contract_rejects_unknown_architectures(tmp_path: Path):
    path, _, _ = build_tiny_dense_checkpoint(tmp_path / "dense.gguf")
    from .gguf_builder import GGUFFileBuilder

    meta = DenseCheckpoint(path).metadata
    builder = GGUFFileBuilder("qwen3moe")
    for key, value in meta.items():
        if key not in {"general.architecture", "general.alignment"}:
            builder.add(key, value)
    bad = builder.write(tmp_path / "bad.gguf")
    with pytest.raises(DenseContractError):
        DenseCheckpoint(bad)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("qwen3.attention.head_count_kv", 0),
        ("qwen3.attention.layer_norm_rms_epsilon", 0.0),
        ("qwen3.rope.freq_base", 0.0),
    ],
)
def test_dense_metadata_does_not_replace_explicit_zero_with_a_default(
    tmp_path: Path, key: str, value: object
):
    path, _, _ = build_tiny_dense_checkpoint(tmp_path / "dense.gguf", qwen3_dense_config())
    metadata = DenseCheckpoint(path).metadata
    metadata[key] = value
    with pytest.raises(DenseContractError):
        DenseConfig.from_metadata(metadata)


def test_dense_contract_rejects_missing_tensors(tmp_path: Path):
    path, _, _ = build_tiny_dense_checkpoint(tmp_path / "dense.gguf")
    from pyrite.adapters.gguf import GGUFReader

    from .gguf_builder import GGUFFileBuilder

    reader = GGUFReader(path)
    builder = GGUFFileBuilder("llama")
    for key, value in reader.metadata().items():
        if key not in {"general.architecture", "general.alignment"}:
            builder.add(key, value)
    for tensor in reader.tensor_index():
        if tensor.name == "blk.0.ffn_gate.weight":
            continue
        builder.add_tensor(tensor.name, tensor.dims, tensor.ggml_type, reader.read_tensor(tensor))
    bad = builder.write(tmp_path / "bad.gguf")
    with pytest.raises(DenseContractError, match="ffn_gate"):
        DenseCheckpoint(bad).validate_contract()


@pytest.mark.parametrize("arch", ["llama", "qwen3"])
def test_dense_generation_is_deterministic_and_valid(tmp_path: Path, arch: str):
    config = TinyDenseConfig() if arch == "llama" else qwen3_dense_config()
    path, cfg, _ = build_tiny_dense_checkpoint(tmp_path / "dense.gguf", config)
    with DenseExecutor(path, runtime=_runtime(), sampler_seed=0) as executor:
        assert executor.unsupported_tensor_types() == ()
        logits = executor.logits(3, 0)
        assert len(logits) == cfg.vocab_size
        assert all(math.isfinite(v) for v in logits)
        first = executor.generate([5, 9, 11], max_new_tokens=4, temperature=0.0)
        second = executor.generate([5, 9, 11], max_new_tokens=4, temperature=0.0)
        assert first == second
        assert all(0 <= token < cfg.vocab_size for token in first)
        text = executor.generate_text("hello world", max_new_tokens=2, temperature=0.0)
        assert text["new_tokens"] == 2
        assert text["text"].startswith("hello world")
        assert executor.stats.layers_executed > 0
        assert executor.stats.bytes_loaded > 0


@pytest.mark.parametrize("arch", ["llama", "qwen3"])
def test_dense_survives_a_tiny_resident_budget(tmp_path: Path, arch: str):
    config = TinyDenseConfig() if arch == "llama" else qwen3_dense_config()
    path, _, _ = build_tiny_dense_checkpoint(tmp_path / "dense.gguf", config)
    with DenseExecutor(path, runtime=_runtime(), resident_bytes=4096,
                       max_tensor_bytes=1 << 20) as tight:
        streamed = tight.generate([7, 21], max_new_tokens=3, temperature=0.0)
        assert tight.stats.peak_resident_bytes <= 4096
        assert tight.streamer.stats().evictions > 0
    with DenseExecutor(path, runtime=_runtime()) as roomy:
        relaxed = roomy.generate([7, 21], max_new_tokens=3, temperature=0.0)
    assert streamed == relaxed


def test_dense_tied_embeddings_fallback(tmp_path: Path):
    path, _, _ = build_tiny_dense_checkpoint(tmp_path / "tied.gguf", write_output_weight=False)
    with DenseExecutor(path, runtime=_runtime()) as executor:
        logits = executor.logits(6, 0)
        assert len(logits) == 271
        assert all(math.isfinite(v) for v in logits)


def test_dense_padded_vocabulary_stays_consistent(tmp_path: Path):
    """Vocabularies larger than the tiny alphabet pad the tokenizer metadata.

    llama.cpp derives the expected embedding rows from the token list and
    refuses the file on mismatch, so the fixture must keep them in sync.
    """
    from pyrite.adapters.gguf import GGUFReader

    config = qwen3_dense_config(
        hidden_size=64, intermediate_size=96, num_hidden_layers=3,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16,
        vocab_size=300, max_position_embeddings=64,
    )
    path, cfg, _ = build_tiny_dense_checkpoint(tmp_path / "padded.gguf", config)
    reader = GGUFReader(path)
    assert len(reader.metadata()["tokenizer.ggml.tokens"]) == cfg.vocab_size
    embd = reader.tensor("token_embd.weight")
    assert embd.dims[1] == cfg.vocab_size
    with DenseExecutor(path, runtime=_runtime(), sampler_seed=0) as executor:
        first = executor.generate([5, 9, 11], max_new_tokens=2, temperature=0.0)
        second = executor.generate([5, 9, 11], max_new_tokens=2, temperature=0.0)
        assert first == second
        assert all(0 <= token < cfg.vocab_size for token in first)


def test_open_executor_dispatches_on_architecture(tmp_path: Path):
    from .tiny_qwen3moe import build_tiny_checkpoint

    moe_path, _, _ = build_tiny_checkpoint(tmp_path / "moe.gguf")
    dense_path, _, _ = build_tiny_dense_checkpoint(tmp_path / "dense.gguf")
    assert detect_architecture(moe_path) == "qwen3moe"
    assert detect_architecture(dense_path) == "llama"
    with open_executor(moe_path, runtime=_runtime()) as executor:
        assert type(executor).__name__ == "Qwen3MoEExecutor"
    with open_executor(dense_path, runtime=_runtime()) as executor:
        assert type(executor).__name__ == "DenseExecutor"
        assert executor.checkpoint_report()["architecture"] == "llama"


def test_open_executor_refuses_unknown_architectures(tmp_path: Path):
    from .gguf_builder import GGUFFileBuilder

    builder = GGUFFileBuilder("falcon")
    path = builder.write(tmp_path / "other.gguf")
    with pytest.raises(ValueError, match="unsupported GGUF architecture"):
        open_executor(path, runtime=_runtime())


def test_dense_q8_0_execution_uses_real_quantized_weights(tmp_path: Path):
    """Q8_0 dense weights must move the logits a little, not to zero."""
    quants = pytest.importorskip("gguf.quants", reason="official quantizer oracle")
    np = pytest.importorskip("numpy", reason="the gguf oracle needs numpy")

    from pyrite.adapters.gguf import GGUFReader

    from .gguf_builder import GGUFFileBuilder

    src, _, _ = build_tiny_dense_checkpoint(tmp_path / "dense.gguf")
    reader = GGUFReader(src)
    builder = GGUFFileBuilder("llama")
    for key, value in reader.metadata().items():
        if key not in {"general.architecture", "general.alignment"}:
            builder.add(key, value)
    converted = 0
    for tensor in reader.tensor_index():
        payload = reader.read_tensor(tensor)
        if (
            tensor.ggml_type == 0
            and tensor.element_count > 0
            and tensor.dims[0] % 32 == 0
            and tensor.element_count % 32 == 0
        ):
            array = np.frombuffer(payload, dtype=np.float32).copy()
            payload = quants.quantize(array, quants.GGMLQuantizationType.Q8_0).tobytes()
            builder.add_tensor(tensor.name, tensor.dims, 8, payload)
            converted += 1
        else:
            builder.add_tensor(tensor.name, tensor.dims, tensor.ggml_type, payload)
    assert converted >= 10
    quantized = builder.write(tmp_path / "dense_q8.gguf")

    with DenseExecutor(src, runtime=_runtime()) as f32_exec:
        f32_logits = f32_exec.logits(3, 0)
    with DenseExecutor(quantized, runtime=_runtime(), sampler_seed=0) as q8_exec:
        assert q8_exec.unsupported_tensor_types() == ()
        q8_logits = q8_exec.logits(3, 0)
        first = q8_exec.generate([5, 9, 11], max_new_tokens=3, temperature=0.0)
        second = q8_exec.generate([5, 9, 11], max_new_tokens=3, temperature=0.0)
    assert first == second
    diff = max(abs(a - b) for a, b in zip(f32_logits, q8_logits, strict=True))
    assert 0.0 < diff < 0.5, f"Q8_0 moved logits by {diff}"
