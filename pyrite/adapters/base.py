from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence


@dataclass(frozen=True)
class WeightBlock:
    block_id: str
    path: Path
    offset: int
    size: int
    dtype: str
    kind: str = "layer"
    index: int = 0


class ModelAdapter(Protocol):
    name: str

    def blocks(self) -> Sequence[WeightBlock]:
        ...

    def load(self, block: WeightBlock) -> memoryview:
        ...
