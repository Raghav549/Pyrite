from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StagingDecision:
    key: str
    commit: bool
    lead_seconds: float


class TimedStager:
    """Commits scarce resident capacity only when use is close enough to justify it."""

    def __init__(self, safety_factor: float = 1.25):
        if safety_factor < 1.0:
            raise ValueError("safety_factor must be >= 1")
        self.safety_factor = safety_factor

    def decide(self, key: str, eta_seconds: float, load_seconds: float, capacity_available: bool) -> StagingDecision:
        if eta_seconds < 0 or load_seconds < 0:
            raise ValueError("timings must be non-negative")
        lead = load_seconds * self.safety_factor
        return StagingDecision(key, bool(capacity_available and eta_seconds <= lead), lead)


@dataclass(frozen=True)
class PageHeat:
    key: str
    hits: int
    last_epoch: int


class PredictiveRetention:
    """Tracks access heat separately from recency for storage-page retention."""

    def __init__(self):
        self._heat: dict[str, PageHeat] = {}
        self._epoch = 0

    def observe(self, key: str) -> None:
        self._epoch += 1
        item = self._heat.get(key)
        self._heat[key] = PageHeat(key, 1 if item is None else item.hits + 1, self._epoch)

    def score(self, key: str) -> float:
        item = self._heat.get(key)
        if item is None:
            return 0.0
        age = self._epoch - item.last_epoch
        return item.hits / (1.0 + age)

    def rank(self, keys: list[str]) -> tuple[str, ...]:
        return tuple(sorted(keys, key=lambda k: (-self.score(k), k)))
