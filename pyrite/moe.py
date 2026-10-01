from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExpertChoice:
    expert_id: str
    score: float


class ExpertRouter:
    """Dependency-free top-k expert selector.

    A trained router can replace route() while preserving this interface.
    """

    def __init__(self, top_k: int = 2):
        self.top_k = max(1, top_k)

    def route(self, scores: dict[str, float]) -> tuple[ExpertChoice, ...]:
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        return tuple(
            ExpertChoice(expert_id, float(score))
            for expert_id, score in ranked[: self.top_k]
        )
