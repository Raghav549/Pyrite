from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from time import perf_counter

from .io_scheduler import IOComputeScheduler, IOComputeTask
from .plan_feedback import ExecutionPlanFeedback


@dataclass(frozen=True)
class StepResult:
    task_id: str
    io_seconds: float
    compute_seconds: float
    output: object


class FeedbackExecutionLoop:
    """Runs bounded local work while feeding measured I/O/compute back into policy."""

    def __init__(self, scheduler: IOComputeScheduler | None = None, feedback: ExecutionPlanFeedback | None = None):
        self.scheduler = scheduler or IOComputeScheduler()
        self.feedback = feedback or ExecutionPlanFeedback()

    def run(
        self,
        plan_id: str,
        tasks: Iterable[IOComputeTask],
        load: Callable[[str], object],
        compute: Callable[[str, object], object],
    ) -> tuple[StepResult, ...]:
        task_list = list(tasks)
        decisions = self.scheduler.schedule(task_list)
        by_id = {task.task_id: task for task in task_list}
        results: list[StepResult] = []

        for decision in decisions:
            task = by_id[decision.task_id]
            t0 = perf_counter()
            value = load(task.task_id)
            t1 = perf_counter()
            output = compute(task.task_id, value)
            t2 = perf_counter()
            io_seconds = t1 - t0
            compute_seconds = t2 - t1
            results.append(StepResult(task.task_id, io_seconds, compute_seconds, output))
            self.scheduler.observe(io_seconds, compute_seconds)
            self.feedback.record(
                plan_id,
                task.io_cost_bytes,
                io_seconds,
                compute_seconds,
                cache_hit=False,
            )

        return tuple(results)
