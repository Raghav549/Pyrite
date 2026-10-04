"""The published Qwen3 head geometry: head_dim 128 at hidden 1024.

Every published dense Qwen3 uses ``head_dim = 128``, and for the smaller sizes
that is *not* ``hidden_size / num_attention_heads``.  Qwen3-0.6B is the sharp
case: hidden 1024 over 16 heads is 64, but the real head dimension is 128, so

* ``attn_q.weight`` is ``(1024, 2048)``, not ``(1024, 1024)``;
* ``attn_output.weight`` is ``(2048, 1024)``;
* the QK-norm vectors are 128 long.

An engine that silently defaulted ``head_dim`` to ``hidden / heads`` would
build the wrong attention shapes and still "work" on a toy model, so the
geometry is exercised here at the published dimensions, and the contradiction
paths are asserted to be *rejected* rather than quietly accepted.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from pyrite.adapters.gguf import GGUFReader
from pyrite.config import RuntimeConfig
from pyrite.dense import DenseCheckpoint, DenseContractError
from pyrite.engine import PyriteRuntime
from pyrite.executor import open_executor

from .gguf_builder import GGUFFileBuilder
from .tiny_dense import TinyDenseConfig, build_tiny_dense_checkpoint

# Qwen3-0.6B's published attention geometry.  Layer count and intermediate size
# are kept real too; only the vocabulary is small, to keep the fixture small.
PUBLISHED_0_6B = {
    "architecture": "qwen3",
    "hidden_size": 1024,
    "num_attention_heads": 16,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "intermediate_size": 3072,
    "num_hidden_layers": 2,
    "max_position_embeddings": 512,
    "qk_norm": True,
}


def _build(tmp_path: Path, name: str, **overrides) -> Path:
    cfg = TinyDenseConfig(**{**PUBLISHED_0_6B, **overrides})
    return build_tiny_dense_checkpoint(tmp_path / f"{name}.gguf", cfg)[0]


def _rewrite_metadata(
    tmp_path: Path, source: Path, out_name: str, *, drop: str = "", set_key_length: int | None = None
) -> Path:
    """Copy a checkpoint, editing only metadata keys.

    Tensor payloads and shapes are copied verbatim, so any contract failure
    the rewritten file produces is genuinely a metadata/tensor contradiction
    rather than a malformed tensor.
    """
    reader = GGUFReader(source)
    builder = GGUFFileBuilder("qwen3")
    for key, value in reader.metadata().items():
        if key in ("general.architecture", "general.alignment"):
            continue
        if drop and key.endswith(drop):
            continue
        if set_key_length is not None and key.endswith("attention.key_length"):
            builder.add(key, set_key_length)
            continue
        builder.add(key, value)
    for tensor in reader.tensor_index():
        builder.add_tensor(
            tensor.name, tuple(tensor.dims), tensor.ggml_type, bytes(reader.read_tensor(tensor))
        )
    return builder.write(tmp_path / f"{out_name}.gguf")


def test_published_0_6b_head_geometry_is_accepted(tmp_path: Path) -> None:
    path = _build(tmp_path, "qwen3-0p6b-heads")
    checkpoint = DenseCheckpoint(path)
    cfg = checkpoint.config

    assert cfg.head_dim == 128
    assert cfg.q_proj_dim == 16 * 128
    assert cfg.kv_proj_dim == 8 * 128

    tensors = {t.name: tuple(t.dims) for t in GGUFReader(path).tensor_index()}
    assert tensors["blk.0.attn_q.weight"] == (1024, 2048)
    assert tensors["blk.0.attn_k.weight"] == (1024, 1024)
    assert tensors["blk.0.attn_v.weight"] == (1024, 1024)
    assert tensors["blk.0.attn_output.weight"] == (2048, 1024)
    # QK-norm runs per head over head_dim, so these are 128 long - not 64.
    assert tensors["blk.0.attn_q_norm.weight"] == (128,)
    assert tensors["blk.0.attn_k_norm.weight"] == (128,)

    # validate_contract raises DenseContractError on any shape mismatch, so
    # reaching a summary at all is the assertion; check what it reports too.
    summary = checkpoint.validate_contract()
    assert summary["architecture"] == "qwen3"
    assert summary["lm_head"]["tensor"] == "output.weight"
    assert summary["native_generation_ready"] is True


def test_head_dim_128_genuinely_disagrees_with_hidden_over_heads(tmp_path: Path) -> None:
    """Guards the premise of this file: the naive default would be wrong."""
    cfg = DenseCheckpoint(_build(tmp_path, "qwen3-0p6b-trap")).config
    assert cfg.hidden_size // cfg.num_attention_heads == 64
    assert cfg.head_dim == 128
    assert cfg.head_dim != cfg.hidden_size // cfg.num_attention_heads


def test_metadata_head_dim_contradicting_the_tensors_is_rejected(tmp_path: Path) -> None:
    """Metadata claiming 128 over tensors built for 64 must not be accepted."""
    narrow = _build(tmp_path, "qwen3-head64", head_dim=64)
    liar = _rewrite_metadata(tmp_path, narrow, "qwen3-head64-lies", set_key_length=128)

    with pytest.raises(DenseContractError) as excinfo:
        DenseCheckpoint(liar).validate_contract()

    message = str(excinfo.value)
    assert "attn_q.weight" in message
    assert "(1024, 1024)" in message and "(1024, 2048)" in message
    assert "expected" in message


def test_missing_key_length_is_shape_checked_not_silently_assumed(tmp_path: Path) -> None:
    """Without ``attention.key_length`` Pyrite infers hidden/heads - and the
    inference is still validated against the tensors, so a real 128-head
    checkpoint is rejected rather than silently mis-shaped."""
    path = _build(tmp_path, "qwen3-0p6b-nokey")
    dropped = _rewrite_metadata(tmp_path, path, "qwen3-0p6b-dropped", drop="attention.key_length")

    with pytest.raises(DenseContractError) as excinfo:
        DenseCheckpoint(dropped).validate_contract()

    # The inferred 64 makes the real 2048-wide projection a mismatch.
    assert "attn_q.weight" in str(excinfo.value)
    assert "(1024, 2048)" in str(excinfo.value)


def test_published_head_geometry_generates_end_to_end(tmp_path: Path) -> None:
    """The 128-dim head path actually runs, including per-head QK-norm."""
    path = _build(tmp_path, "qwen3-0p6b-runs")
    # tests/conftest.py pins PYRITE_RAM_MB, so the memory gate is exercised
    # deterministically instead of against whatever RAM happens to be free.
    runtime = PyriteRuntime(RuntimeConfig.from_env())
    with open_executor(path, runtime=runtime, sampler_seed=11) as executor:
        result = executor.generate_text("hello world", max_new_tokens=3, temperature=0.0)
        assert result["new_tokens"] == 3
        assert result["completion"]
        again = executor.generate_text("hello world", max_new_tokens=3, temperature=0.0)
        assert again["output_ids"][-3:] == result["output_ids"][-3:]
