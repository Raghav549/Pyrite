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


def test_import_and_measure_without_resource_module(monkeypatch):
    """Windows has no ``resource`` module; import and RSS must survive that."""
    import sys

    import pyrite.memory as memory

    monkeypatch.setitem(sys.modules, "resource", None)
    assert memory._rusage_peak_mb() is None
    # On Linux the /proc fallback still measures; elsewhere it degrades to 0.0.
    assert memory.process_memory_mb() >= 0.0
    assert memory._peak_rss_mb() >= 0.0


def test_rss_degrades_gracefully_when_everything_is_missing(monkeypatch):
    import pyrite.memory as memory

    monkeypatch.setattr(memory, "_proc_rss_mb", lambda: None)
    monkeypatch.setattr(memory, "_rusage_peak_mb", lambda: None)
    monkeypatch.setattr(memory, "_windows_working_set_mb", lambda: None)
    assert memory.process_memory_mb() == 0.0


def test_rss_monitor_enforces_the_configured_ceiling(monkeypatch):
    import pytest

    import pyrite.memory as memory

    monkeypatch.setattr(memory, "process_rss_bytes", lambda: 2048)
    monkeypatch.setattr(memory, "process_peak_rss_mb", lambda: 0.0)
    monitor = memory.RSSMonitor(limit_bytes=1024)
    with pytest.raises(MemoryError, match="above configured limit"):
        monitor.sample("unit-test")
    assert monitor.current_bytes == 2048
    assert monitor.peak_bytes == 2048
    assert monitor.last_stage == "unit-test"
