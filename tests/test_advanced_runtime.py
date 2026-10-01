
from pyrite.io_scheduler import IOComputeScheduler, IOComputeTask
from pyrite.plan_feedback import ExecutionPlanFeedback
from pyrite.prefetch import PrefetchPredictor


def test_confidence_prefetch_is_bounded():
    predictor = PrefetchPredictor()
    for item in ["a", "b", "a", "c", "a", "b"]:
        predictor.observe(item)
    candidates = predictor.candidates("a", ["b", "c", "d"], 3)
    assert len(candidates) == 3
    assert all(0.0 < c.confidence <= 1.0 for c in candidates)


def test_io_compute_scheduler_overlaps_when_io_dominates():
    scheduler = IOComputeScheduler()
    scheduler.observe(1.0, 0.1)
    decisions = scheduler.schedule([IOComputeTask("x", 100, 1.0)])
    assert decisions[0].overlap


def test_execution_plan_feedback_learns():
    feedback = ExecutionPlanFeedback()
    feedback.record("p", 100, 0.1, 0.2, True)
    feedback.record("p", 100, 0.1, 0.2, True)
    assert feedback.stats["p"].cache_hit_rate == 1.0
    assert feedback.score("p") > 0
