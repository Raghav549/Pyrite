from __future__ import annotations

from dataclasses import dataclass, field
from collections import OrderedDict
import os
import resource
from typing import Generic, Iterator, TypeVar

T = TypeVar("T")


@dataclass
class MemoryStats:
    resident_items: int = 0
    estimated_bytes: int = 0
    hits: int = 0
    misses: int = 0
    evictions: int = 0


@dataclass
class LRUResidentCache(Generic[T]):
    """Bounded in-process cache used for hot model blocks/experts."""

    max_items: int
    _data: OrderedDict[str, T] = field(default_factory=OrderedDict)
    _sizes: dict[str, int] = field(default_factory=dict)
    stats: MemoryStats = field(default_factory=MemoryStats)

    def get(self, key: str) -> T | None:
        if key not in self._data:
            self.stats.misses += 1
            return None
        self.stats.hits += 1
        value = self._data.pop(key)
        self._data[key] = value
        return value

    def put(self, key: str, value: T, estimated_bytes: int = 0) -> None:
        if key in self._data:
            self._data.pop(key)
            self.stats.estimated_bytes -= self._sizes.pop(key, 0)
        self._data[key] = value
        self._sizes[key] = max(0, estimated_bytes)
        self.stats.estimated_bytes += self._sizes[key]
        while len(self._data) > max(1, self.max_items):
            old_key, _ = self._data.popitem(last=False)
            self.stats.estimated_bytes -= self._sizes.pop(old_key, 0)
            self.stats.evictions += 1
        self.stats.resident_items = len(self._data)

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def keys(self) -> Iterator[str]:
        return iter(self._data.keys())


def process_memory_mb() -> float:
    """Best-effort process RSS in MiB without third-party dependencies."""
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if os.name == "nt":
        return rss / (1024 * 1024)
    return rss / 1024
