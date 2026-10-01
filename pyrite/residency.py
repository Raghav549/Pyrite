from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ResidencySnapshot:
    budget_bytes: int
    resident_bytes: int
    pinned_bytes: int

    @property
    def headroom_bytes(self) -> int:
        return max(0, self.budget_bytes - self.resident_bytes - self.pinned_bytes)


class ResidencyLedger:
    """Single source of truth for application-level resident memory accounting."""

    def __init__(self, budget_bytes: int):
        if budget_bytes <= 0:
            raise ValueError("budget_bytes must be positive")
        self.budget_bytes = budget_bytes
        self._resident: dict[str, int] = {}
        self._pinned: dict[str, int] = {}

    def reserve(self, key: str, size_bytes: int, pinned: bool = False) -> None:
        size_bytes = max(0, int(size_bytes))
        current = self._resident.get(key, 0)
        delta = size_bytes - current
        if delta <= 0:
            self._set(key, size_bytes, pinned)
            return
        if self.snapshot().headroom_bytes < delta:
            raise MemoryError(f"resident memory budget exceeded by {delta} bytes")
        self._set(key, size_bytes, pinned)

    def _set(self, key: str, size: int, pinned: bool) -> None:
        self._resident[key] = size
        self._pinned.pop(key, None)
        if pinned:
            self._pinned[key] = size

    def release(self, key: str) -> None:
        self._resident.pop(key, None)
        self._pinned.pop(key, None)

    def snapshot(self) -> ResidencySnapshot:
        resident = sum(self._resident.values())
        pinned = sum(self._pinned.values())
        return ResidencySnapshot(self.budget_bytes, resident, pinned)
