from pyrite.kv import KVBudget


def test_kv_budget_is_bounded():
    kv = KVBudget(4)
    assert kv.accept(3) == 3
    assert kv.accept(3) == 1
    assert kv.should_truncate()
    assert kv.stats.tokens_dropped == 2
