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


def test_block_ids_cannot_escape_the_store(tmp_path: Path):
    store = LocalBlockStore(tmp_path)
    for bad in ("../escape", "a/b", "", ".hidden", "x" * 200):
        try:
            store.add_bytes(bad, b"data")
        except ValueError:
            continue
        raise AssertionError(f"block id {bad!r} should have been rejected")


def test_metadata_pointing_outside_the_store_is_ignored(tmp_path: Path):
    store = LocalBlockStore(tmp_path)
    outside = tmp_path.parent / "outside.bin"
    outside.write_bytes(b"secret")
    (tmp_path / "evil.json").write_text(
        "{" + f'"block_id": "evil", "path": "{outside}", "size_bytes": 6, "sha256": "{"0" * 64}"' + "}",
        encoding="utf-8",
    )
    assert list(store.iter_blocks()) == []
    assert store.get("evil") is None
