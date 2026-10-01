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
    """Best-effort current RSS in MB, falling back to platform peak RSS."""
    if os.name == "nt":
        try:
            import ctypes
            class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
                _fields_ = [
                    ("cb", ctypes.c_ulong),
                    ("PageFaultCount", ctypes.c_ulong),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                ]
            counters = PROCESS_MEMORY_COUNTERS()
            counters.cb = ctypes.sizeof(counters)
            handle = ctypes.windll.kernel32.GetCurrentProcess()
            ok = ctypes.windll.psapi.GetProcessMemoryInfo(
                handle, ctypes.byref(counters), counters.cb
            )
            if ok:
                return counters.WorkingSetSize / (1024 * 1024)
        except Exception:
            pass
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)

    try:
        with open("/proc/self/status", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return float(line.split()[1]) / 1024.0
    except (OSError, ValueError):
        pass
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
