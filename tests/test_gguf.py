from pathlib import Path
import struct

from pyrite.adapters.gguf import GGUFReader
from pyrite.backends.real import GGUFReferenceBackend


def _gguf(tmp_path: Path, ggml_type: int = 0, dims=(2,)):
    metadata = b""
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
