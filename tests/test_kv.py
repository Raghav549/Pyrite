from pyrite.executor import LayerKV
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


def test_executor_kv_uses_compact_payload_and_accounts_for_eviction():
    cache = LayerKV()
    cache.append([1.0, 2.0], [3.0, 4.0])
    cache.append([5.0, 6.0], [7.0, 8.0])
    assert cache.byte_size == 32  # 2 tokens x 2 vectors x 2 fp32 values
    assert cache.estimated_bytes > cache.byte_size
    assert cache.evict_oldest() == 1
    assert cache.byte_size == 16
    assert cache.keys[0].tolist() == [5.0, 6.0]
    assert cache.truncate(0) == 1
    assert cache.byte_size == 0
