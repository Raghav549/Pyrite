from pyrite.sampler import Sampler


def test_greedy():
    assert Sampler(seed=1).greedy([0.1, 2.0, 0.2]) == 1
