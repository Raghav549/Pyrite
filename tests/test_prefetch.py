from pyrite.prefetch import PrefetchPredictor


def test_prefetch_learns_transition():
    p = PrefetchPredictor()
    p.observe("a")
    p.observe("b")
    p.observe("a")
    p.observe("b")
    assert p.predict("a", ["b", "c"], 1) == ("b",)
