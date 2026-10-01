from __future__ import annotations

from dataclasses import dataclass
from collections import deque


@dataclass(frozen=True)
class LoadPlan:
    required: tuple[str, ...]
    prefetch: tuple[str, ...]
    evict: tuple[str, ...]


class BlockScheduler:
    """Plans a small hot working set around sequential execution."""

    def __init__(self, resident_limit: int = 2, prefetch_depth: int = 2):
        self.resident_limit = max(1, resident_limit)
        self.prefetch_depth = max(0, prefetch_depth)
        self._history: deque[str] = deque(maxlen=32)

    def plan(self, blocks: list[str], current_index: int, resident: set[str]) -> LoadPlan:
        required = (blocks[current_index],) if 0 <= current_index < len(blocks) else ()
        future = tuple(blocks[current_index + 1: current_index + 1 + self.prefetch_depth])
        keep = set(required) | set(future)
        evict = tuple(x for x in resident if x not in keep)
        self._history.extend(required)
        return LoadPlan(required, future, evict)
