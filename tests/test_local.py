from pathlib import Path

from pyrite.config import RuntimeConfig
from pyrite.local import LocalModel


def test_local_model_streams_in_budget(tmp_path: Path):
    path = tmp_path / "model.bin"
    path.write_bytes(b"abcdefghij")
    config = RuntimeConfig(ram_budget_mb=4096, reserve_mb=768, resident_blocks=2)
    with LocalModel(path, config=config, chunk_bytes=3) as model:
        info = model.info()
        assert info.block_count == 4
        first = model.checkpoint.adapter.blocks()[0].block_id
        assert bytes(model.load_block(first)) == b"abc"
