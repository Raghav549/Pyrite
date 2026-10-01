from __future__ import annotations

from dataclasses import dataclass
from time import monotonic


@dataclass(frozen=True)
class IOComputeTask:
    task_id: str
    io_cost_bytes: int
    compute_cost: float
    priority: float = 0.0


@dataclass(frozen=True)
class IOComputeDecision:
    task_id: str
    overlap: bool


class IOComputeScheduler:
    """Lightweight online scheduler for overlapping storage and compute work."""

    def __init__(self):
        self._ema_io = 0.0
        self._ema_compute = 0.0
        self._samples = 0

    def observe(self, io_seconds: float, compute_seconds: float) -> None:
        if io_seconds < 0 or compute_seconds < 0:
            raise ValueError("timings must be non-negative")
        alpha = 0.2
        if self._samples == 0:
            self._ema_io = io_seconds
            self._ema_compute = compute_seconds
        else:
            self._ema_io = alpha * io_seconds + (1 - alpha) * self._ema_io
            self._ema_compute = alpha * compute_seconds + (1 - alpha) * self._ema_compute
        self._samples += 1

    def schedule(self, tasks: list[IOComputeTask]) -> tuple[IOComputeDecision, ...]:
        if not tasks:
            return ()
        results = []
        io_dominant = self._ema_io > self._ema_compute
        ranked = sorted(tasks, key=lambda t: (-t.priority, t.io_cost_bytes, -t.compute_cost))
        for task in ranked:
            # Prefer overlap whenever historical I/O is the dominant critical path.
            results.append(IOComputeDecision(task.task_id, io_dominant or task.io_cost_bytes > 0))
        return tuple(results)

    @property
    def ema_io_seconds(self) -> float:
        return self._ema_io

    @property
    def ema_compute_seconds(self) -> float:
        return self._ema_compute
