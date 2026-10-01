from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class DenseTensor:
    rows: int
    cols: int
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        if self.rows < 0 or self.cols < 0:
            raise ValueError("tensor dimensions must be non-negative")
        if len(self.values) != self.rows * self.cols:
            raise ValueError("tensor data length does not match shape")

    def row(self, index: int) -> tuple[float, ...]:
        if index < 0 or index >= self.rows:
            raise IndexError(index)
        start = index * self.cols
        return self.values[start : start + self.cols]


def matvec(tensor: DenseTensor, vector: Sequence[float]) -> tuple[float, ...]:
    if len(vector) != tensor.cols:
        raise ValueError("vector width does not match tensor")
    return tuple(
        sum(weight * value for weight, value in zip(tensor.row(i), vector))
        for i in range(tensor.rows)
    )
