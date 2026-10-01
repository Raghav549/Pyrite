from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ComputeConfig:
    threads: int = 1
    use_accelerator: bool = False


class ComputeBackend:
    name = "reference"

    def __init__(self, config: ComputeConfig | None = None):
        self.config = config or ComputeConfig()

    def matvec_f32(self, matrix: list[list[float]], vector: list[float]) -> list[float]:
        if not matrix:
            return []
        width = len(matrix[0])
        if len(vector) != width:
            raise ValueError("vector width does not match matrix width")
        if any(len(row) != width for row in matrix):
            raise ValueError("matrix rows must have equal width")
        return [
            sum(float(w) * float(x) for w, x in zip(row, vector))
            for row in matrix
        ]
