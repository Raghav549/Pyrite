from pyrite.residency import ResidencyLedger


def test_residency_ledger_is_hard_bounded():
    ledger = ResidencyLedger(10)
    ledger.reserve("a", 6)
    ledger.reserve("b", 4)
    assert ledger.snapshot().resident_bytes == 10
    try:
        ledger.reserve("c", 1)
        assert False
    except MemoryError:
        pass


def test_release_frees_budget():
    ledger = ResidencyLedger(10)
    ledger.reserve("a", 8)
    ledger.release("a")
    ledger.reserve("b", 8)
    assert ledger.snapshot().resident_bytes == 8
