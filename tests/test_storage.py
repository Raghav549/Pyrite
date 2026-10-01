from pathlib import Path

from pyrite.storage import LocalBlockStore


def test_block_store_round_trip(tmp_path: Path):
    store = LocalBlockStore(tmp_path)
    ref = store.add_bytes("layer:0", b"hello")
    assert ref.size_bytes == 5
    assert ref.path.read_bytes() == b"hello"
    found = list(store.iter_blocks())
    assert len(found) == 1
    assert found[0].sha256 == ref.sha256
