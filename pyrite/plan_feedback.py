from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PlanStats:
    executions: int = 0
    total_io_bytes: int = 0
    total_io_seconds: float = 0.0
    total_compute_seconds: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0

    @property
    def avg_io_seconds(self) -> float:
        return self.total_io_seconds / self.executions if self.executions else 0.0

    @property
    def avg_compute_seconds(self) -> float:
        return self.total_compute_seconds / self.executions if self.executions else 0.0

    @property
    def cache_hit_rate(self) -> float:
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total else 0.0


@dataclass
class ExecutionPlanFeedback:
    stats: dict[str, PlanStats] = field(default_factory=dict)

    def record(self, plan_id: str, io_bytes: int, io_seconds: float, compute_seconds: float, cache_hit: bool) -> None:
        if io_bytes < 0 or io_seconds < 0 or compute_seconds < 0:
            raise ValueError("execution metrics must be non-negative")
        s = self.stats.setdefault(plan_id, PlanStats())
        s.executions += 1
        s.total_io_bytes += io_bytes
        s.total_io_seconds += io_seconds
        s.total_compute_seconds += compute_seconds
        if cache_hit:
            s.cache_hits += 1
        else:
            s.cache_misses += 1

    def score(self, plan_id: str) -> float:
        s = self.stats.get(plan_id)
        if s is None:
            return 0.0
        # Higher is better: reward cache hits and penalize measured I/O/compute latency.
        latency = s.avg_io_seconds + s.avg_compute_seconds
        return s.cache_hit_rate / max(1e-6, latency)
