from pathlib import Path

from pyrite.adapters.checkpoint import ChunkedFileAdapter, detect_checkpoint


def test_chunked_checkpoint_adapter(tmp_path: Path):
    path = tmp_path / "model.bin"
    path.write_bytes(b"0123456789")
    info = detect_checkpoint(path)
    assert info.format == "ggml-or-raw"
    adapter = ChunkedFileAdapter(path, chunk_bytes=4)
    blocks = adapter.blocks()
    assert [b.size for b in blocks] == [4, 4, 2]
    assert bytes(adapter.load(blocks[1])) == b"4567"
