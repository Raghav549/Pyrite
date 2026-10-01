from __future__ import annotations

from dataclasses import dataclass

from .spec import BlockKind, ModelBlock, ModelManifest


@dataclass(frozen=True)
class ExecutionUnit:
    block_ids: tuple[str, ...]
    estimated_bytes: int


class ModelExecutionPlanner:
    """Turns a model manifest into bounded execution units."""

    def __init__(self, manifest: ModelManifest, max_unit_bytes: int):
        if max_unit_bytes <= 0:
            raise ValueError("max_unit_bytes must be positive")
        self.manifest = manifest
        self.max_unit_bytes = max_unit_bytes

    def units(self) -> tuple[ExecutionUnit, ...]:
        result: list[ExecutionUnit] = []
        current: list[str] = []
        current_bytes = 0

        for block in self.manifest.ordered_blocks():
            if block.size_bytes > self.max_unit_bytes:
                result.append(
                    ExecutionUnit((block.block_id,), block.size_bytes)
                )
                current = []
                current_bytes = 0
                continue

            if current and current_bytes + block.size_bytes > self.max_unit_bytes:
                result.append(ExecutionUnit(tuple(current), current_bytes))
                current = []
                current_bytes = 0

            current.append(block.block_id)
            current_bytes += block.size_bytes

        if current:
            result.append(ExecutionUnit(tuple(current), current_bytes))

        return tuple(result)

    @staticmethod
    def select_experts(
        blocks: tuple[ModelBlock, ...],
        preferred: tuple[str, ...],
    ) -> tuple[str, ...]:
        available = {block.block_id for block in blocks if block.kind == BlockKind.EXPERT}
        return tuple(block_id for block_id in preferred if block_id in available)
