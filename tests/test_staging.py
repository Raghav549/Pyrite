from pyrite.staging import TimedStager


def test_stager_waits_until_use_is_close():
    stager = TimedStager()
    assert not stager.decide("kv", 10.0, 1.0, True).commit
    assert stager.decide("kv", 0.5, 1.0, True).commit


def test_stager_respects_capacity():
    stager = TimedStager()
    assert not stager.decide("kv", 0.1, 1.0, False).commit
