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
        return StagingDecision(
            key=key,
            commit=bool(capacity_available and eta_seconds <= lead),
            lead_seconds=lead,
        )
