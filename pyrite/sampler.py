from __future__ import annotations

import math
import random


class Sampler:
    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed)

    def greedy(self, logits: list[float]) -> int:
        if not logits:
            raise ValueError("empty logits")
        return max(range(len(logits)), key=logits.__getitem__)

    def sample(self, logits: list[float], temperature: float = 1.0, top_p: float = 1.0) -> int:
        if not logits:
            raise ValueError("empty logits")
        if temperature <= 0:
            return self.greedy(logits)
        scaled = [x / temperature for x in logits]
        m = max(scaled)
        probs = [math.exp(x - m) for x in scaled]
        total = sum(probs)
        probs = [x / total for x in probs]
        ranked = sorted(range(len(probs)), key=probs.__getitem__, reverse=True)
        if 0 < top_p < 1:
            keep = []
            acc = 0.0
            for idx in ranked:
                keep.append(idx)
                acc += probs[idx]
                if acc >= top_p:
                    break
            z = sum(probs[i] for i in keep)
            ranked = keep
            probs = [probs[i] / z if i in keep else 0.0 for i in range(len(probs))]
        return self.rng.choices(range(len(probs)), weights=probs, k=1)[0]
