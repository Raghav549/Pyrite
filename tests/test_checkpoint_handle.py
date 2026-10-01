from pathlib import Path

import pytest

from pyrite.checkpoint import open_checkpoint


def test_open_checkpoint(tmp_path: Path):
    path = tmp_path / "model.ggml"
    path.write_bytes(b"abcdef")
    handle = open_checkpoint(path, chunk_bytes=2)
    assert handle.info.size_bytes == 6
    assert len(handle.adapter.blocks()) == 3


def test_reject_unknown(tmp_path: Path):
    path = tmp_path / "model.xyz"
    path.write_bytes(b"x")
    with pytest.raises(ValueError):
        open_checkpoint(path)
