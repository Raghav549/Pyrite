from __future__ import annotations

import math
import random


class Sampler:
    def __init__(self, seed: int | None = None):
        # Sampling does not need cryptographic randomness.
        self.rng = random.Random(seed)  # noqa: S311

    def greedy(self, logits: list[float]) -> int:
        if not logits:
            raise ValueError("empty logits")
        return max(range(len(logits)), key=logits.__getitem__)

    def sample(self, logits: list[float], temperature: float = 1.0, top_p: float = 1.0) -> int:
        if not logits:
            raise ValueError("empty logits")
        if temperature < 0:
            raise ValueError("temperature must be non-negative")
        if not 0.0 < top_p <= 1.0:
            raise ValueError("top_p must be in (0, 1]")
        if temperature == 0:
            return self.greedy(logits)
        scaled = [x / temperature for x in logits]
        m = max(scaled)
        probs = [math.exp(x - m) for x in scaled]
        total = sum(probs)
        probs = [x / total for x in probs]
        ranked = sorted(range(len(probs)), key=probs.__getitem__, reverse=True)
        if top_p < 1.0:
            keep: list[int] = []
            acc = 0.0
            for idx in ranked:
                keep.append(idx)
                acc += probs[idx]
                if acc >= top_p:
                    break
            z = sum(probs[i] for i in keep)
            if z <= 0.0:  # pragma: no cover - only reachable with degenerate logits
                return ranked[0]
            keep_set = set(keep)
            probs = [probs[i] / z if i in keep_set else 0.0 for i in range(len(probs))]
        return self.rng.choices(range(len(probs)), weights=probs, k=1)[0]
