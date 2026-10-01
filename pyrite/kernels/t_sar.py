from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class TernaryKernelSpec:
    bits_per_weight: int = 2
    values: tuple[int, int, int] = (-1, 0, 1)


class ReferenceTernaryKernel:
    """Portable correctness reference for a future SIMD ternary backend."""

    def __init__(self, spec: TernaryKernelSpec | None = None):
        self.spec = spec or TernaryKernelSpec()

    def matvec(self, matrix: Sequence[Sequence[int]], vector: Sequence[int]) -> list[int]:
        if not matrix:
            return []
        width = len(matrix[0])
        if len(vector) != width:
            raise ValueError("vector width does not match matrix width")
        if any(len(row) != width for row in matrix):
            raise ValueError("matrix rows must have equal width")
        return [sum(int(w) * int(x) for w, x in zip(row, vector)) for row in matrix]

    def estimate_bits(self, rows: int, cols: int) -> int:
        if rows < 0 or cols < 0:
            raise ValueError("matrix dimensions must be non-negative")
        return rows * cols * self.spec.bits_per_weight
