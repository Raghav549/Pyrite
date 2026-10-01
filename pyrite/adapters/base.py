from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class WeightBlock:
    """A bounded, independently loadable byte range inside a checkpoint."""

    block_id: str
    path: Path
    offset: int
    size: int
    dtype: str
    kind: str = "layer"
    index: int = 0


@dataclass(frozen=True)
class TensorMeta:
    """Format-neutral tensor description used by loaders and executors."""

    name: str
    block_id: str
    shape: tuple[int, ...]
    dtype: str
    ggml_type: int | None
    offset: int
    size: int

    @property
    def element_count(self) -> int:
        total = 1
        for dim in self.shape:
            total *= dim
        return total if self.shape else 0


class ModelAdapter(Protocol):
    name: str

    def blocks(self) -> Sequence[WeightBlock]:
        ...

    def load(self, block: WeightBlock) -> memoryview:
        ...

    def tensors(self) -> Sequence[TensorMeta]:
        ...
