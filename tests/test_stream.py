from pathlib import Path

import pytest

from pyrite.adapters.raw_shards import RawShardAdapter
from pyrite.stream import BlockStreamer


def test_streamer_prefetch_and_cache(tmp_path: Path):
    for name, data in [("a", b"aaa"), ("b", b"bbb"), ("c", b"ccc")]:
        (tmp_path / f"{name}.bin").write_bytes(data)

    adapter = RawShardAdapter(tmp_path)
    with BlockStreamer(adapter, resident_blocks=2, workers=1) as stream:
        stream.prefetch(["a", "b"])
        assert bytes(stream.get("a")) == b"aaa"
        assert bytes(stream.get("a")) == b"aaa"
        stats = stream.stats()
        with pytest.raises(KeyError):
            stream.get("missing")

    assert stats.prefetched == 2
    assert stats.loads == 2
    assert stats.cache_hits >= 1
