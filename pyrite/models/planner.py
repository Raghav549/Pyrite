from __future__ import annotations

from dataclasses import dataclass

from .spec import BlockKind, ModelBlock, ModelManifest


@dataclass(frozen=True)
class ExecutionUnit:
    block_ids: tuple[str, ...]
    estimated_bytes: int


@dataclass(frozen=True)
class PlanObservation:
    unit_id: str
    io_seconds: float
    compute_seconds: float
    cache_hit: bool


class ModelExecutionPlanner:
    """Turns a model manifest into bounded units and learns measured plan costs."""

    def __init__(self, manifest: ModelManifest, max_unit_bytes: int):
        if max_unit_bytes <= 0:
            raise ValueError("max_unit_bytes must be positive")
        self.manifest = manifest
        self.max_unit_bytes = max_unit_bytes
        self._observations: dict[str, list[PlanObservation]] = {}

    def units(self) -> tuple[ExecutionUnit, ...]:
        """Group ordered blocks into units that each fit ``max_unit_bytes``."""
        result: list[ExecutionUnit] = []
        current: list[str] = []
        current_bytes = 0

        for block in self.manifest.ordered_blocks():
            if block.size_bytes > self.max_unit_bytes:
                # Oversized block: it still has to be executed, on its own.
                if current:
                    result.append(ExecutionUnit(tuple(current), current_bytes))
                    current = []
                    current_bytes = 0
                result.append(ExecutionUnit((block.block_id,), block.size_bytes))
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
    def select_experts(blocks: tuple[ModelBlock, ...], preferred: tuple[str, ...]) -> tuple[str, ...]:
        available = {block.block_id for block in blocks if block.kind == BlockKind.EXPERT}
        return tuple(block_id for block_id in preferred if block_id in available)

    @staticmethod
    def unit_id(unit: ExecutionUnit) -> str:
        return "|".join(unit.block_ids)

    def observe(
        self,
        unit: ExecutionUnit,
        io_seconds: float,
        compute_seconds: float,
        cache_hit: bool,
    ) -> PlanObservation:
        if io_seconds < 0 or compute_seconds < 0:
            raise ValueError("timings must be non-negative")
        observation = PlanObservation(self.unit_id(unit), io_seconds, compute_seconds, cache_hit)
        self._observations.setdefault(observation.unit_id, []).append(observation)
        return observation

    def score(self, unit: ExecutionUnit) -> float:
        """Measured quality of a unit: cache hits per second of latency."""
        records = self._observations.get(self.unit_id(unit), [])
        if not records:
            return 0.0
        total = len(records)
        hit_rate = sum(r.cache_hit for r in records) / total
        latency = sum(r.io_seconds + r.compute_seconds for r in records) / total
        return hit_rate / max(1e-9, latency)

    def reorder_for_locality(self, units: tuple[ExecutionUnit, ...]) -> tuple[ExecutionUnit, ...]:
        """Stable-sort units by measured score, keeping plan order for unknowns."""
        return tuple(
            unit for _, unit in sorted(enumerate(units), key=lambda item: (-self.score(item[1]), item[0]))
        )

    def observations(self) -> dict[str, tuple[PlanObservation, ...]]:
        return {key: tuple(value) for key, value in self._observations.items()}
