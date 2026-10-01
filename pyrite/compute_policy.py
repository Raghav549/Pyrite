from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TokenDifficulty:
    uncertainty: float
    repetition: float
    reasoning_hint: float


@dataclass(frozen=True)
class ComputePlan:
    layers: tuple[int, ...]
    confidence: float


class AdaptiveLayerPolicy:
    """Conservative hook for learned/token-aware layer skipping."""

    def __init__(self, min_layers: int = 1):
        self.min_layers = max(1, min_layers)

    def plan(self, total_layers: int, difficulty: TokenDifficulty) -> ComputePlan:
        if total_layers <= 0:
            raise ValueError("total_layers must be positive")
        uncertainty = max(0.0, min(1.0, difficulty.uncertainty))
        reasoning = max(0.0, min(1.0, difficulty.reasoning_hint))
        repetition = max(0.0, min(1.0, difficulty.repetition))

        pressure = max(uncertainty, reasoning) - 0.5 * repetition
        use_fraction = 0.5 + 0.5 * max(0.0, min(1.0, pressure))
        count = max(self.min_layers, min(total_layers, round(total_layers * use_fraction)))

        layers = tuple(range(count))
        confidence = max(0.0, min(1.0, 1.0 - abs(use_fraction - 1.0) * 0.5))
        return ComputePlan(layers, confidence)
