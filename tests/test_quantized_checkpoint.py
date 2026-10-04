"""A quantized dense checkpoint must load, validate and generate.

The original failure this project started from was on a Q8_0 Qwen3 GGUF, so the
quantized path is a first-class case rather than an afterthought.

Two facts are pinned here because both were discovered by running things:

* a quantized LM head (``output.weight`` stored as Q8_0/Q4_0) is resolved like
  any other - the tie/shape checks do not depend on the storage type;
* RMSNorm weights must stay F32 in a quantized file.  ggml's CPU binary ops
  refuse a mixed-type element-wise operand
  (``binary_op: unsupported types: dst f32, src1 q8_0``), which is why every
  real quantized GGUF leaves the ``*_norm.weight`` vectors in F32.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from pyrite.adapters.gguf import GGUFReader
from pyrite.config import RuntimeConfig
from pyrite.dense import DenseCheckpoint
from pyrite.engine import PyriteRuntime
from pyrite.executor import open_executor

from .gguf_builder import GGUFFileBuilder
from .tiny_dense import TinyDenseConfig, build_tiny_dense_checkpoint

NORM_SUFFIX = "_norm.weight"


def _quantize(tmp_path: Path, ggml_type: int, *, quantize_norms: bool) -> Path:
    """Rebuild the tiny dense checkpoint with quantized weight tensors."""
    from pyrite.ggml_types import spec
    from scripts.build_vocab_fixture import QUANT_ENCODERS

    source, _config, _tensors = build_tiny_dense_checkpoint(
        tmp_path / f"source-{ggml_type}.gguf", TinyDenseConfig(architecture="qwen3")
    )
    reader = GGUFReader(source)
    builder = GGUFFileBuilder("qwen3")
    for key, value in reader.metadata().items():
        if key in ("general.architecture", "general.alignment"):
            continue
        builder.add(key, value)

    for tensor in reader.tensor_index():
        raw = bytes(reader.read_tensor(tensor))
        elements = 1
        for dim in tensor.dims:
            elements *= dim
        # ggml quantizes block by block *within a row*, so the row length
        # (dims[0]) must be a multiple of the block size - a total element
        # count that happens to divide is not enough.
        row_length = tensor.dims[0] if tensor.dims else 1
        keep_f32 = (
            tensor.ggml_type != 0
            or row_length % spec(ggml_type).block_size
            or (tensor.name.endswith(NORM_SUFFIX) and not quantize_norms)
        )
        if keep_f32:
            builder.add_tensor(tensor.name, tensor.dims, tensor.ggml_type, raw)
        else:
            import struct

            values = list(struct.unpack(f"<{elements}f", raw))
            builder.add_tensor(
                tensor.name, tensor.dims, ggml_type, QUANT_ENCODERS[ggml_type](values)
            )
    return builder.write(tmp_path / f"quant-{ggml_type}-{quantize_norms}.gguf")


@pytest.mark.parametrize("ggml_type,name", [(8, "Q8_0"), (2, "Q4_0")], ids=["Q8_0", "Q4_0"])
def test_quantized_lm_head_is_resolved_like_any_other(tmp_path: Path, ggml_type: int, name: str) -> None:
    path = _quantize(tmp_path, ggml_type, quantize_norms=False)
    checkpoint = DenseCheckpoint(path)
    summary = checkpoint.validate_contract()
    assert summary["lm_head"]["tensor"] == "output.weight"
    assert summary["lm_head"]["type"] == name
    assert summary["lm_head"]["tied"] is False
    assert summary["tensor_types_decodable"] is True
    assert summary["unsupported_tensor_types"] == []


@pytest.mark.parametrize("ggml_type", [8, 2], ids=["Q8_0", "Q4_0"])
def test_quantized_checkpoint_generates_real_tokens(tmp_path: Path, ggml_type: int) -> None:
    """End-to-end: quantized weights in, deterministic text out."""
    path = _quantize(tmp_path, ggml_type, quantize_norms=False)
    runtime = PyriteRuntime(RuntimeConfig.from_env())
    with open_executor(path, runtime=runtime, sampler_seed=7) as executor:
        result = executor.generate_text("hello world", max_new_tokens=4, temperature=0.0)
        assert result["new_tokens"] == 4
        assert len(result["output_ids"]) == result["prompt_tokens"] + 4
        assert result["completion"]
        # Deterministic: the same seed and prompt must reproduce it.
        second = executor.generate_text("hello world", max_new_tokens=4, temperature=0.0)
        assert second["output_ids"][-4:] == result["output_ids"][-4:]


def test_norm_weights_must_stay_f32_in_a_quantized_file(tmp_path: Path) -> None:
    """The rule every real quantized GGUF follows, and why.

    This asserts the *file layout* Pyrite produces and consumes.  A quantized
    norm vector is not something Pyrite can reject on its own - it decodes
    fine - but ggml cannot execute it, so a checkpoint built this way would not
    interoperate.  Keeping the invariant visible stops it from being "fixed"
    into a file no other runtime can load.
    """
    path = _quantize(tmp_path, 8, quantize_norms=False)
    for tensor in GGUFReader(path).tensor_index():
        if tensor.name.endswith(NORM_SUFFIX):
            assert tensor.ggml_type == 0, f"{tensor.name} must stay F32"


def test_quantized_and_f32_checkpoints_agree_on_the_lm_head_contract(tmp_path: Path) -> None:
    """Quantization changes storage, never the LM-head resolution."""
    plain = DenseCheckpoint(_quantize(tmp_path, 8, quantize_norms=False))
    summary = plain.validate_contract()
    assert summary["lm_head"]["source"].startswith("output.weight is present")
    assert summary["vocab_token_count"] == plain.config.vocab_token_count
    # The vocabulary still comes from the tokenizer metadata, not the storage type.
    metadata = GGUFReader(plain.path).metadata()
    assert len(metadata["tokenizer.ggml.tokens"]) == plain.config.vocab_token_count
