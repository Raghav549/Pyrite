from pyrite.memory import LRUResidentCache


def test_byte_budget_evicts_oldest():
    cache = LRUResidentCache[bytes](max_items=4, max_bytes=5)
    cache.put("a", b"aaa", 3)
    cache.put("b", b"bb", 2)
    cache.put("c", b"c", 1)
    assert cache.stats.estimated_bytes <= 5
    assert "a" not in cache
    assert "c" in cache


def test_oversized_item_fails():
    cache = LRUResidentCache[bytes](max_items=2, max_bytes=4)
    try:
        cache.put("large", b"12345", 5)
    except MemoryError:
        pass
    else:
        raise AssertionError("expected MemoryError")


def test_current_rss_is_non_negative():
    from pyrite.memory import process_memory_mb
    assert process_memory_mb() >= 0
