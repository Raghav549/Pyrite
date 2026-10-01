from __future__ import annotations

from collections import Counter, deque


class PrefetchPredictor:
    """Small locality predictor for upcoming blocks."""

    def __init__(self, history_size: int = 64):
        self.history: deque[str] = deque(maxlen=history_size)
        self.transitions: Counter[tuple[str, str]] = Counter()

    def observe(self, block_id: str) -> None:
        if self.history:
            self.transitions[(self.history[-1], block_id)] += 1
        self.history.append(block_id)

    def predict(self, current: str, candidates: list[str], k: int = 2) -> tuple[str, ...]:
        ranked = sorted(
            enumerate(candidates),
            key=lambda item: (self.transitions[(current, item[1])], -item[0]),
            reverse=True,
        )
        return tuple(candidate for _, candidate in ranked[:max(0, k)])
