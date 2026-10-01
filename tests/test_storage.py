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


def test_block_ids_with_colons_use_windows_legal_filenames(tmp_path: Path):
    """Ids like ``layer:0001`` are valid, but ``:`` is illegal on Windows."""
    import string

    store = LocalBlockStore(tmp_path)
    ref = store.add_bytes("layer:0001", b"payload")
    assert ref.block_id == "layer:0001"
    assert ref.path.read_bytes() == b"payload"
    assert store.get("layer:0001") is not None

    illegal = set('<>:"/\\|?*') | set(chr(c) for c in range(32))
    for path in tmp_path.iterdir():
        assert not (set(path.name) & illegal), path.name
        assert all(c in string.printable for c in path.name)

    # Distinct ids must never collide on disk.
    store.add_bytes("layer@0001", b"other")
    assert store.get("layer:0001").path.read_bytes() == b"payload"
    assert store.get("layer@0001").path.read_bytes() == b"other"
    assert store.get("layer:0001").path != store.get("layer@0001").path
