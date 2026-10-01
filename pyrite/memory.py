from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
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
    max_items: int
    max_bytes: int | None = None
    _data: OrderedDict[str, T] = field(default_factory=OrderedDict)
    _sizes: dict[str, int] = field(default_factory=dict)
    stats: MemoryStats = field(default_factory=MemoryStats)

    def _would_fit(self, extra: int) -> bool:
        if self.max_bytes is None:
            return True
        return self.stats.estimated_bytes + max(0, extra) <= self.max_bytes

    def get(self, key: str) -> T | None:
        if key not in self._data:
            self.stats.misses += 1
            return None
        self.stats.hits += 1
        value = self._data.pop(key)
        self._data[key] = value
        return value

    def put(self, key: str, value: T, estimated_bytes: int = 0) -> None:
        estimated_bytes = max(0, estimated_bytes)
        if key in self._data:
            self._data.pop(key)
            self.stats.estimated_bytes -= self._sizes.pop(key, 0)

        if self.max_bytes is not None and estimated_bytes > self.max_bytes:
            raise MemoryError(f"block {key!r} exceeds resident byte budget")

        self._data[key] = value
        self._sizes[key] = estimated_bytes
        self.stats.estimated_bytes += estimated_bytes

        while (
            len(self._data) > max(1, self.max_items)
            or not self._would_fit(0)
        ):
            old_key, _ = self._data.popitem(last=False)
            self.stats.estimated_bytes -= self._sizes.pop(old_key, 0)
            self.stats.evictions += 1

        self.stats.resident_items = len(self._data)

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def keys(self) -> Iterator[str]:
        return iter(self._data.keys())


def process_memory_mb() -> float:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if os.name == "nt":
        return rss / (1024 * 1024)
    return rss / 1024
