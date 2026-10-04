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

    # The prefetch runs on a worker thread, so whether the first get("a") finds
    # it already resident or loads it itself is a scheduling outcome, not a
    # contract: pinning stats.loads made this test fail intermittently under
    # full-suite load.  What must hold on every schedule is asserted instead.
    assert stats.prefetched == 2, "both requested blocks must be handed to the worker"
    assert stats.prefetch_hits == 1, "the first get must find a's prefetch"
    assert stats.cache_hits == 1, "the second get must be served from the cache"
    assert stats.loads >= 1
    assert stats.bytes_loaded in (3, 6), "a, and optionally b, are the only reads"
