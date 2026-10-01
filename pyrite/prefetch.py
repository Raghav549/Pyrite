from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class PrefetchCandidate:
    block_id: str
    confidence: float
    expected_reuse: float = 0.0
    io_cost: float = 0.0


class PrefetchPredictor:
    """Confidence-aware predictor with cost-aware admission."""

    def __init__(self, history_size: int = 64):
        self.history: deque[str] = deque(maxlen=history_size)
        self.transitions: Counter[tuple[str, str]] = Counter()

    def observe(self, block_id: str) -> None:
        if self.history:
            self.transitions[(self.history[-1], block_id)] += 1
        self.history.append(block_id)

    def candidates(self, current: str, candidates: list[str], k: int = 2) -> tuple[PrefetchCandidate, ...]:
        if k <= 0:
            return ()
        total = sum(self.transitions[(current, c)] for c in candidates)
        ranked = []
        for index, candidate in enumerate(candidates):
            hits = self.transitions[(current, candidate)]
            confidence = (hits + 1.0) / (total + len(candidates))
            reuse = math.log1p(hits)
            ranked.append(PrefetchCandidate(candidate, confidence, reuse, 1.0))
        ranked.sort(key=lambda c: (c.confidence * (1.0 + c.expected_reuse), c.block_id), reverse=True)
        return tuple(ranked[:k])

    def predict(self, current: str, candidates: list[str], k: int = 2) -> tuple[str, ...]:
        return tuple(c.block_id for c in self.candidates(current, candidates, k))
