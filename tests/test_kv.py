from pyrite.kv import AdaptiveKVCache, KVBudget


def test_kv_budget_is_bounded():
    kv = KVBudget(4)
    assert kv.accept(3) == 3
    assert kv.accept(3) == 1
    assert kv.should_truncate()
    assert kv.stats.tokens_dropped == 2


def test_adaptive_kv_precision():
    assert AdaptiveKVCache.choose_precision(0.95) == 16
    assert AdaptiveKVCache.choose_precision(0.60) == 8
    assert AdaptiveKVCache.choose_precision(0.30) == 4
    assert AdaptiveKVCache.choose_precision(0.10) == 2


def test_adaptive_kv_stays_in_budget():
    cache = AdaptiveKVCache(max_bytes=64, bytes_per_token_fp16=16)
    for importance in [0.9, 0.7, 0.4, 0.1, 0.1, 0.1]:
        cache.add(importance)
    assert cache.stats.estimated_bytes <= 64
