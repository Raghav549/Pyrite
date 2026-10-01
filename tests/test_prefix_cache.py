from pyrite.prefix_cache import PrefixKVCache


def test_prefix_cache_reuses_entry():
    cache = PrefixKVCache(max_bytes=8)
    tokens = [1, 2, 3]
    cache.put(tokens, b"abcd")
    entry = cache.get(tokens)
    assert entry is not None
    assert entry.token_count == 3
    assert entry.payload == b"abcd"


def test_prefix_cache_evicts_to_budget():
    cache = PrefixKVCache(max_bytes=5)
    cache.put([1], b"abc")
    cache.put([2], b"def")
    assert cache.bytes_used <= 5
