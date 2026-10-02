"""Prefetch behaviour and the execution-plan feedback loop.

Prefetch: hits, misses, wrong predictions, cancellation, contention and the
hard guarantee that prefetching never violates the resident budget.
Planning: an initial plan, measured telemetry, feedback, an updated plan and
execution that follows the update.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from pyrite.adapters.raw_shards import RawShardAdapter
from pyrite.execution_loop import FeedbackExecutionLoop
from pyrite.io_scheduler import IOComputeScheduler, IOComputeTask
from pyrite.models.planner import ModelExecutionPlanner
from pyrite.models.spec import BlockKind, ModelBlock, ModelManifest
from pyrite.plan_feedback import ExecutionPlanFeedback
from pyrite.prefetch import PrefetchPredictor
from pyrite.stream import BlockStreamer


def _shards(tmp_path: Path, sizes: dict[str, int]) -> RawShardAdapter:
    for name, size in sizes.items():
        (tmp_path / f"{name}.bin").write_bytes(bytes([len(name) % 251]) * size)
    return RawShardAdapter(tmp_path)


# ------------------------------------------------------------------ predictor
def test_predictor_confidence_prefers_observed_transitions():
    predictor = PrefetchPredictor()
    for _ in range(4):
        predictor.observe("a")
        predictor.observe("b")
    predictor.observe("a")
    predictor.observe("c")
    ranked = predictor.candidates("a", ["b", "c"], 2)
    assert [c.block_id for c in ranked] == ["b", "c"]
    assert ranked[0].confidence > ranked[1].confidence > 0.0


def test_predictor_discounts_expensive_io():
    predictor = PrefetchPredictor()
    ranked = predictor.candidates("a", ["cheap", "pricey"], 2,
                                  io_cost={"pricey": 100.0})
    assert [c.block_id for c in ranked] == ["cheap", "pricey"]
    assert ranked[0].priority > ranked[1].priority


def test_predictor_handles_empty_and_zero_k():
    predictor = PrefetchPredictor()
    assert predictor.candidates("a", [], 2) == ()
    assert predictor.candidates("a", ["b"], 0) == ()
    assert predictor.predict("a", [], 2) == ()
    predictor.observe("a")
    predictor.clear()
    assert predictor.predict("a", ["a"], 1) == ("a",)


def test_predictor_rejects_a_non_positive_history():
    with pytest.raises(ValueError):
        PrefetchPredictor(history_size=0)


# ------------------------------------------------------------------ streamer
def test_prefetch_hit_avoids_a_second_load(tmp_path: Path):
    adapter = _shards(tmp_path, {"a": 64, "b": 64})
    with BlockStreamer(adapter, resident_blocks=4, workers=1) as stream:
        stream.prefetch(["a"])
        assert bytes(stream.get("a")) == bytes([len("a") % 251]) * 64
        stats = stream.stats()
        assert stats.prefetch_hits == 1
        assert stats.loads == 1
        # The cached copy serves further reads without more I/O.
        stream.get("a")
        assert stream.stats().loads == 1
        assert stream.stats().cache_hits >= 1


def test_wrong_prediction_is_evicted_and_counted(tmp_path: Path):
    adapter = _shards(tmp_path, {"a": 64, "b": 64, "c": 64})
    with BlockStreamer(adapter, resident_blocks=2, workers=1) as stream:
        stream.prefetch(["a", "b", "c"])  # c is never used
        stream.get("a")
        stream.get("b")
        stats = stream.stats()
        assert stats.prefetch_hits == 2
        assert stats.pending == 1  # the unused prediction is still queued


def test_prefetch_ranges_hit_and_miss(tmp_path: Path):
    adapter = _shards(tmp_path, {"a": 256})
    with BlockStreamer(adapter, resident_blocks=8, workers=1) as stream:
        stream.prefetch_ranges([("a", 64, 32)], confidence=1.0)
        assert bytes(stream.get_range("a", 64, 32)) == bytes([len("a") % 251]) * 32
        assert stream.stats().prefetch_hits == 1
        # Zero confidence refuses the prefetch and accounts the bytes as waste.
        before = stream.stats().wasted_prefetch_bytes
        stream.prefetch_ranges([("a", 128, 32)], confidence=0.0)
        assert stream.stats().wasted_prefetch_bytes == before + 32
        with pytest.raises(ValueError):
            stream.prefetch_ranges([("a", 0, 8)], confidence=-1.0)


def test_range_reads_reuse_a_cached_whole_block(tmp_path: Path):
    adapter = _shards(tmp_path, {"a": 256})
    expected = bytes([len("a") % 251]) * 32
    with BlockStreamer(adapter, resident_blocks=4, resident_bytes=512) as stream:
        stream.get("a")
        assert bytes(stream.get_range("a", 64, 32)) == expected
        assert stream.stats().loads == 1


def test_range_read_consumes_a_full_block_prefetch(tmp_path: Path):
    adapter = _shards(tmp_path, {"a": 256})
    expected = bytes([len("a") % 251]) * 32
    with BlockStreamer(adapter, resident_blocks=4, resident_bytes=512, workers=1) as stream:
        stream.prefetch(["a"])
        assert bytes(stream.get_range("a", 64, 32)) == expected
        assert stream.stats().loads == 1
        assert stream.stats().prefetch_hits == 1


def test_oversized_range_is_usable_but_never_cached(tmp_path: Path):
    adapter = _shards(tmp_path, {"big": 1024})
    with BlockStreamer(adapter, resident_blocks=8, resident_bytes=128, workers=1) as stream:
        payload = bytes(stream.get("big"))
        assert len(payload) == 1024
        stats = stream.stats()
        assert stats.uncached_bytes == 1024
        assert stats.resident_bytes == 0


def test_range_outside_the_block_is_rejected(tmp_path: Path):
    adapter = _shards(tmp_path, {"a": 64})
    with BlockStreamer(adapter, resident_blocks=4, workers=1) as stream:
        with pytest.raises(ValueError):
            stream.get_range("a", 0, 65)
        with pytest.raises(ValueError):
            stream.get_range("a", -1, 8)


def test_streamer_rejects_non_positive_budgets(tmp_path: Path):
    adapter = _shards(tmp_path, {"a": 8})
    with pytest.raises(ValueError):
        BlockStreamer(adapter, resident_bytes=0)
    with pytest.raises(ValueError):
        BlockStreamer(adapter, resident_blocks=0)


# ------------------------------------------------------------ plan feedback
def _manifest() -> ModelManifest:
    return ModelManifest(
        name="demo",
        architecture="toy",
        parameter_count=100,
        quantization="q8",
        blocks=(
            ModelBlock("slow", BlockKind.LAYER, 4, 0),
            ModelBlock("fast", BlockKind.LAYER, 4, 1),
        ),
    )


def test_measured_feedback_reorders_the_plan():
    planner = ModelExecutionPlanner(_manifest(), max_unit_bytes=4)
    units = planner.units()
    assert [u.block_ids for u in units] == [("slow",), ("fast",)]
    assert planner.score(units[0]) == 0.0  # unknown before telemetry
    planner.observe(units[0], io_seconds=1.0, compute_seconds=1.0, cache_hit=False)
    planner.observe(units[1], io_seconds=0.01, compute_seconds=0.01, cache_hit=True)
    assert planner.score(units[1]) > planner.score(units[0])
    reordered = planner.reorder_for_locality(units)
    assert [u.block_ids for u in reordered] == [("fast",), ("slow",)]
    assert set(planner.observations()) == {"slow", "fast"}


def test_oversized_blocks_get_their_own_unit():
    manifest = ModelManifest(
        name="demo", architecture="toy", parameter_count=10, quantization="q8",
        blocks=(ModelBlock("huge", BlockKind.LAYER, 99, 0),),
    )
    units = ModelExecutionPlanner(manifest, max_unit_bytes=4).units()
    assert [u.block_ids for u in units] == [("huge",)]


def test_feedback_loop_feeds_measurements_back(tmp_path: Path):
    loop = FeedbackExecutionLoop()
    tasks = [IOComputeTask("b", io_cost_bytes=8, compute_cost=0.0, priority=0.0),
             IOComputeTask("a", io_cost_bytes=8, compute_cost=0.0, priority=5.0)]
    seen: list[str] = []
    results = loop.run("plan", tasks, lambda task_id: seen.append(task_id) or task_id,
                       lambda task_id, value: value.upper())
    assert [r.task_id for r in results] == ["a", "b"]
    assert seen == ["a", "b"]
    stats = loop.feedback.stats["plan"]
    assert stats.executions == 2
    assert stats.total_io_bytes == 16
    assert stats.avg_io_seconds >= 0.0


def test_plan_feedback_rejects_negative_metrics():
    feedback = ExecutionPlanFeedback()
    with pytest.raises(ValueError):
        feedback.record("p", -1, 0.0, 0.0, True)
    assert feedback.score("unknown") == 0.0


def test_io_scheduler_learns_ema_and_ranks_tasks():
    scheduler = IOComputeScheduler()
    assert scheduler.schedule([]) == ()
    scheduler.observe(2.0, 0.1)
    assert scheduler.ema_io_seconds == pytest.approx(2.0)
    decisions = scheduler.schedule([
        IOComputeTask("low", io_cost_bytes=10, compute_cost=1.0, priority=0.0),
        IOComputeTask("high", io_cost_bytes=10, compute_cost=1.0, priority=9.0),
    ])
    assert [d.task_id for d in decisions] == ["high", "low"]
    assert all(d.overlap for d in decisions)
    with pytest.raises(ValueError):
        scheduler.observe(-1.0, 0.0)
