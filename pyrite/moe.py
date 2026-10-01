from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExpertChoice:
    expert_id: str
    score: float


class ExpertRouter:
    def __init__(self, top_k: int = 2):
        self.top_k = max(1, top_k)

    def route(self, scores: dict[str, float]) -> tuple[ExpertChoice, ...]:
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        return tuple(
            ExpertChoice(expert_id, float(score))
            for expert_id, score in ranked[: self.top_k]
        )

    def ids(self, scores: dict[str, float]) -> tuple[str, ...]:
        return tuple(choice.expert_id for choice in self.route(scores))
