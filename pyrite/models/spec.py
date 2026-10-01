from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum


class BlockKind(str, Enum):
    LAYER = "layer"
    EXPERT = "expert"
    SHARED = "shared"


@dataclass(frozen=True)
class ModelBlock:
    block_id: str
    kind: BlockKind
    size_bytes: int
    index: int
    dependencies: tuple[str, ...] = ()


@dataclass(frozen=True)
class ModelManifest:
    name: str
    architecture: str
    parameter_count: int
    quantization: str
    blocks: tuple[ModelBlock, ...]

    @property
    def total_bytes(self) -> int:
        return sum(block.size_bytes for block in self.blocks)

    def ordered_blocks(self) -> tuple[ModelBlock, ...]:
        return tuple(sorted(self.blocks, key=lambda block: block.index))


def blocks_from_rows(rows: Iterable[dict]) -> tuple[ModelBlock, ...]:
    return tuple(
        ModelBlock(
            block_id=str(row["block_id"]),
            kind=BlockKind(str(row.get("kind", "layer"))),
            size_bytes=int(row["size_bytes"]),
            index=int(row["index"]),
            dependencies=tuple(row.get("dependencies", ())),
        )
        for row in rows
    )
