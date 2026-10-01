from __future__ import annotations

import math
from collections import Counter, deque
from dataclasses import dataclass


@dataclass(frozen=True)
class PrefetchCandidate:
    block_id: str
    confidence: float
    expected_reuse: float = 0.0
    io_cost: float = 0.0

    @property
    def priority(self) -> float:
        """Higher is better: confidence scaled by reuse, discounted by I/O cost."""
        return self.confidence * (1.0 + self.expected_reuse) / max(1e-9, 1.0 + self.io_cost)


class PrefetchPredictor:
    """Confidence-aware predictor with cost-aware admission.

    The predictor is a first-order Markov model over observed block transitions.
    Confidence uses add-one smoothing so unseen transitions get a small non-zero
    score instead of a confident zero (which would disable the fallback
    sequential plan).
    """

    def __init__(self, history_size: int = 64):
        if history_size < 1:
            raise ValueError("history_size must be positive")
        self.history: deque[str] = deque(maxlen=history_size)
        self.transitions: Counter[tuple[str, str]] = Counter()

    def observe(self, block_id: str) -> None:
        if self.history:
            self.transitions[(self.history[-1], block_id)] += 1
        self.history.append(block_id)

    def candidates(
        self,
        current: str,
        candidates: list[str],
        k: int = 2,
        io_cost: dict[str, float] | None = None,
    ) -> tuple[PrefetchCandidate, ...]:
        if k <= 0:
            return ()
        unique: list[str] = []
        seen: set[str] = set()
        for candidate in candidates:
            if candidate not in seen:
                seen.add(candidate)
                unique.append(candidate)
        if not unique:
            return ()

        cost = io_cost or {}
        total = sum(self.transitions[(current, candidate)] for candidate in unique)
        ranked = []
        for candidate in unique:
            hits = self.transitions[(current, candidate)]
            confidence = (hits + 1.0) / (total + len(unique))
            ranked.append(
                PrefetchCandidate(
                    candidate,
                    confidence,
                    math.log1p(hits),
                    max(0.0, float(cost.get(candidate, 0.0))),
                )
            )
        ranked.sort(key=lambda item: (-item.priority, item.block_id))
        return tuple(ranked[:k])

    def predict(
        self,
        current: str,
        candidates: list[str],
        k: int = 2,
        io_cost: dict[str, float] | None = None,
    ) -> tuple[str, ...]:
        return tuple(c.block_id for c in self.candidates(current, candidates, k, io_cost))

    def clear(self) -> None:
        self.history.clear()
        self.transitions.clear()
