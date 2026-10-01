import struct
from pathlib import Path

import pytest

from pyrite.adapters.gguf import GGUFReader
from pyrite.backends.real import GGUFReferenceBackend


def _gguf(tmp_path: Path, ggml_type: int = 0, dims=(2,)):
    tensor_name = b"x"
    desc = (
        struct.pack("<Q", len(tensor_name)) + tensor_name
        + struct.pack("<I", len(dims))
        + b"".join(struct.pack("<Q", d) for d in dims)
        + struct.pack("<I", ggml_type)
        + struct.pack("<Q", 0)
    )
    header = b"GGUF" + struct.pack("<IQQ", 3, 1, 0)
    pos = len(header) + len(desc)
    pad = b"\0" * ((32 - pos % 32) % 32)
    payload = struct.pack("<2f", 1.0, 2.0)
    path = tmp_path / "tiny.gguf"
    path.write_bytes(header + desc + pad + payload)
    return path


def test_gguf_tensor_index(tmp_path: Path):
    path = _gguf(tmp_path)
    reader = GGUFReader(path)
    tensors = reader.tensor_index()
    assert tensors[0].name == "x"
    assert tensors[0].dims == (2,)
    assert tensors[0].size == 8


def test_gguf_real_reference_backend(tmp_path: Path):
    path = _gguf(tmp_path)
    backend = GGUFReferenceBackend(path)
    assert backend.tensor("x") == [1.0, 2.0]


def test_ggml_type_sizes_are_consistent(tmp_path: Path):
    """Type ids, block geometry and byte sizes follow llama.cpp's enum."""
    from pyrite.ggml_types import spec, tensor_size

    # Ids are the published ggml_type values, including the retired gaps.
    assert spec(0).name == "F32"
    assert spec(2).name == "Q4_0"
    assert spec(8).name == "Q8_0"
    assert spec(10).name == "Q2_K"
    assert spec(11).name == "Q3_K"
    assert spec(12).name == "Q4_K"
    assert spec(14).name == "Q6_K"
    assert spec(24).name == "I8"
    assert spec(30).name == "BF16"

    # Byte sizes from the static_asserts in llama.cpp's ggml-common.h.
    assert tensor_size(256, 10) == 84  # Q2_K
    assert tensor_size(256, 11) == 110  # Q3_K
    assert tensor_size(256, 12) == 144  # Q4_K
    assert tensor_size(256, 13) == 176  # Q5_K
    assert tensor_size(256, 14) == 210  # Q6_K
    assert tensor_size(256, 15) == 292  # Q8_K
    assert tensor_size(32, 2) == 18  # Q4_0
    assert tensor_size(32, 8) == 34  # Q8_0
    assert tensor_size(2, 1) == 4  # F16

    # K-quant dims must be block aligned; silently rounding up would make the
    # parser accept a tensor extent that does not match the file.
    assert tensor_size(512, 12) == 2 * 144
    with pytest.raises(ValueError):
        tensor_size(300, 12)
