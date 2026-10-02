from __future__ import annotations

import os
import sys
from collections import OrderedDict
from collections.abc import Iterator
from contextlib import suppress
from dataclasses import dataclass, field
from typing import Generic, TypeVar

T = TypeVar("T")


@dataclass
class MemoryStats:
    resident_items: int = 0
    estimated_bytes: int = 0
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    rejected: int = 0


@dataclass
class LRUResidentCache(Generic[T]):
    """Bounded LRU cache with both an item and a byte ceiling.

    The byte ceiling is authoritative: a single item larger than ``max_bytes``
    is rejected with ``MemoryError`` rather than silently blowing the budget.
    """

    max_items: int | None
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
            self.stats.rejected += 1
            raise MemoryError(f"block {key!r} exceeds resident byte budget")

        self._data[key] = value
        self._sizes[key] = estimated_bytes
        self.stats.estimated_bytes += estimated_bytes

        while (
            (self.max_items is not None and len(self._data) > max(1, self.max_items))
            or not self._would_fit(0)
        ):
            old_key, _ = self._data.popitem(last=False)
            self.stats.estimated_bytes -= self._sizes.pop(old_key, 0)
            self.stats.evictions += 1

        self.stats.resident_items = len(self._data)

    def discard(self, key: str) -> None:
        """Drop ``key`` if present, keeping byte accounting consistent."""
        if key in self._data:
            self._data.pop(key)
            self.stats.estimated_bytes -= self._sizes.pop(key, 0)
            self.stats.resident_items = len(self._data)

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def __len__(self) -> int:
        return len(self._data)

    def keys(self) -> Iterator[str]:
        return iter(self._data.keys())


def _windows_working_set_mb() -> float | None:
    """Current process working set in MB via the Win32 API.

    Returns ``None`` when the measurement is unavailable instead of raising,
    so callers can fall back to another source.  ``ctypes.windll`` only exists
    on Windows; this function is never called on other platforms.
    """
    with suppress(AttributeError, OSError, TypeError, ValueError):
        import ctypes

        class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
            # Full Win32 PROCESS_MEMORY_COUNTERS layout (10 fields).  Passing
            # a truncated struct can make GetProcessMemoryInfo fail, because
            # the API validates ``cb`` against the expected size.
            _fields_ = [
                ("cb", ctypes.c_ulong),
                ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = PROCESS_MEMORY_COUNTERS()
        counters.cb = ctypes.sizeof(counters)
        handle = ctypes.windll.kernel32.GetCurrentProcess()
        ok = ctypes.windll.psapi.GetProcessMemoryInfo(
            handle, ctypes.byref(counters), counters.cb
        )
        if ok:
            return counters.WorkingSetSize / (1024 * 1024)
    return None


def _proc_rss_mb() -> float | None:
    """Current RSS in MB from ``/proc`` (Linux), or ``None`` if unreadable."""
    try:
        with open("/proc/self/status", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return float(line.split()[1]) / 1024.0
    except (OSError, ValueError):
        return None
    return None


def _rusage_peak_mb() -> float | None:
    """Peak RSS in MB via ``resource`` (Unix only), or ``None`` if unavailable.

    The ``resource`` module does not exist on Windows, so it is imported lazily
    and its absence is not an error.  ``ru_maxrss`` is reported in kilobytes
    on Linux and in bytes on macOS.
    """
    try:
        import resource
    except ImportError:
        return None
    with suppress(OSError, ValueError, AttributeError):
        raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform == "darwin":
            return raw / (1024 * 1024)
        return raw / 1024.0
    return None


def process_memory_mb() -> float:
    """Best-effort current resident memory of this process, in MB.

    Measurement sources, in order of preference:

    * Windows: Win32 working set (``GetProcessMemoryInfo``);
    * Linux: ``/proc/self/status`` ``VmRSS``;
    * any Unix: ``resource.getrusage`` peak RSS (Linux/macOS fallback).

    Returns ``0.0`` when no source is available (for example a sandboxed
    interpreter without ``/proc`` or ``resource``); ``0.0`` therefore means
    "unknown", never "no memory used".  This function never raises for a
    measurement failure: memory *accounting* raises (see
    :class:`LRUResidentCache`), memory *observation* degrades gracefully.
    """
    if os.name == "nt":
        measured = _windows_working_set_mb()
        if measured is not None:
            return measured
        return _rusage_peak_mb() or 0.0

    measured = _proc_rss_mb()
    if measured is not None:
        return measured
    return _rusage_peak_mb() or 0.0


def process_rss_bytes() -> int:
    """Best-effort current resident set size in bytes (zero means unknown)."""
    measured = process_memory_mb()
    return max(0, int(measured * 1024 * 1024))


def process_peak_rss_mb() -> float:
    """Best-effort process high-water RSS, normalized to MiB."""
    peak = _rusage_peak_mb()
    return peak if peak is not None else process_memory_mb()


def _peak_rss_mb() -> float:
    """Backwards-compatible private alias for peak RSS measurement."""
    return process_peak_rss_mb()


class RSSMonitor:
    """Sample and enforce an absolute per-process RSS ceiling.

    The runtime separately budgets managed working-set items (KV, cached GGUF
    byte ranges and small tensors).  This monitor catches Python/runtime
    overhead and transient allocations that object-size accounting cannot
    predict exactly.  RSS reporting is best-effort on platforms where the OS
    does not expose current process memory.
    """

    def __init__(self, limit_bytes: int):
        if limit_bytes <= 0:
            raise ValueError("RSS limit must be positive")
        self.limit_bytes = int(limit_bytes)
        self.current_bytes = 0
        self.peak_bytes = 0
        self.samples = 0
        self.last_stage = "startup"
        self.unknown = False

    def sample(self, stage: str, *, enforce: bool = True) -> int:
        current = process_rss_bytes()
        self.samples += 1
        self.last_stage = stage
        if current == 0:
            self.unknown = True
            return 0
        self.current_bytes = current
        self.peak_bytes = max(self.peak_bytes, current)
        os_peak = process_peak_rss_mb()
        if os_peak > 0:
            self.peak_bytes = max(self.peak_bytes, int(os_peak * 1024 * 1024))
        if enforce and self.peak_bytes > self.limit_bytes:
            raise MemoryError(
                f"process RSS reached {self.peak_bytes / (1024 * 1024):.1f} MiB "
                f"by {stage}, above configured limit "
                f"{self.limit_bytes / (1024 * 1024):.0f} MiB"
            )
        return current

    def to_dict(self) -> dict[str, int | float | str | bool]:
        mib = 1024 * 1024
        return {
            "rss_current_bytes": self.current_bytes,
            "rss_current_mb": self.current_bytes / mib,
            "rss_peak_bytes": self.peak_bytes,
            "rss_peak_mb": self.peak_bytes / mib,
            "rss_limit_bytes": self.limit_bytes,
            "rss_limit_mb": self.limit_bytes / mib,
            "rss_samples": self.samples,
            "rss_last_stage": self.last_stage,
            "rss_unknown": self.unknown,
        }
