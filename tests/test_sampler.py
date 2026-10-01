from pyrite.sampler import Sampler


def test_greedy():
    assert Sampler(seed=1).greedy([0.1, 2.0, 0.2]) == 1


def test_sampling_is_valid():
    token = Sampler(seed=4).sample([0.0, 2.0, 0.0], temperature=1.0, top_p=1.0)
    assert token in {0, 1, 2}
