import json
import struct
from pathlib import Path

from pyrite.adapters.checkpoint import ChunkedFileAdapter, detect_checkpoint
from pyrite.adapters.safetensors import SafetensorsAdapter


def test_chunked_checkpoint_adapter(tmp_path: Path):
    path = tmp_path / "model.bin"
    path.write_bytes(b"0123456789")
    info = detect_checkpoint(path)
    assert info.format == "ggml-or-raw"
    adapter = ChunkedFileAdapter(path, chunk_bytes=4)
    blocks = adapter.blocks()
    assert [b.size for b in blocks] == [4, 4, 2]
    assert bytes(adapter.load(blocks[1])) == b"4567"


def test_safetensors_adapter(tmp_path: Path):
    path = tmp_path / "model.safetensors"
    header = json.dumps({
        "x": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]}
    }, separators=(",", ":")).encode()
    path.write_bytes(struct.pack("<Q", len(header)) + header + b"12345678")
    adapter = SafetensorsAdapter(path)
    blocks = adapter.blocks()
    assert len(blocks) == 1
    assert blocks[0].offset == 8 + len(header)
    assert bytes(adapter.load(blocks[0])) == b"12345678"
