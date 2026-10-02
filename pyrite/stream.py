from __future__ import annotations

from collections.abc import Iterable
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from time import perf_counter

from .adapters.base import ModelAdapter, WeightBlock
from .memory import LRUResidentCache


@dataclass(frozen=True)
class StreamStats:
    loads: int
    prefetched: int
    prefetch_hits: int
    cache_hits: int
    pending: int
    resident_bytes: int
    evictions: int
    wasted_prefetch_bytes: int
    uncached_bytes: int
    bytes_loaded: int
    io_seconds: float


class BlockStreamer:
    """Bounded async streamer with whole-block and byte-range reads.

    ``resident_bytes`` is a single ceiling shared by cache entries and
    prefetched-but-unconsumed buffers.  Prefetch admission evicts the least
    recently used cached blocks before reserving memory; it never gets a second
    independent budget.  Ranged reads are used for streamed matrix rows and
    individual slices of stacked MoE expert tensors.
    """

    def __init__(
        self,
        adapter: ModelAdapter,
        resident_blocks: int | None = None,
        resident_bytes: int | None = None,
        workers: int = 1,
        allow_oversized_reads: bool = True,
    ):
        if resident_bytes is not None and resident_bytes <= 0:
            raise ValueError("resident_bytes must be positive when provided")
        if resident_blocks is not None and resident_blocks < 1:
            raise ValueError("resident_blocks must be at least 1 when provided")
        if workers < 1:
            raise ValueError("workers must be at least 1")
        self.allow_oversized_reads = bool(allow_oversized_reads)
        self.adapter = adapter
        self.blocks = list(adapter.blocks())
        self.index = {block.block_id: block for block in self.blocks}
        self.cache = LRUResidentCache[memoryview](
            None if resident_blocks is None else max(1, resident_blocks),
            resident_bytes,
        )
        self.executor = ThreadPoolExecutor(max_workers=workers)
        self.pending: dict[str, Future[memoryview]] = {}
        self.max_pending_bytes = resident_bytes
        self.pending_bytes = 0
        self.loads = 0
        self.prefetched = 0
        self.prefetch_hits = 0
        self.wasted_prefetch_bytes = 0
        self.uncached_bytes = 0
        self.bytes_loaded = 0
        self.io_seconds = 0.0
        self._stats_lock = Lock()
        self._closed = False

    # ------------------------------------------------------------------ loading
    def _load(
        self,
        block: WeightBlock,
        byte_offset: int = 0,
        byte_length: int | None = None,
    ) -> memoryview:
        length = block.size - byte_offset if byte_length is None else byte_length
        if byte_offset < 0 or length < 0 or byte_offset + length > block.size:
            raise ValueError(
                f"range [{byte_offset}, {byte_offset + length}) is outside {block.block_id}"
            )
        if (
            self.cache.max_bytes is not None
            and length > self.cache.max_bytes
            and not self.allow_oversized_reads
        ):
            raise MemoryError(
                f"requested range is {length} bytes, larger than the resident budget "
                f"({self.cache.max_bytes} bytes)"
            )
        with self._stats_lock:
            self.loads += 1
        started = perf_counter()
        with Path(block.path).open("rb") as fh:
            fh.seek(block.offset + byte_offset)
            payload = fh.read(length)
        elapsed = perf_counter() - started
        if len(payload) != length:
            raise ValueError(f"truncated block read: {block.block_id}")
        with self._stats_lock:
            self.io_seconds += elapsed
            self.bytes_loaded += length
        return memoryview(payload)

    @staticmethod
    def _range_id(
        block_id: str,
        byte_offset: int,
        byte_length: int,
        block_size: int | None = None,
    ) -> str:
        # A range beginning at byte zero is not a whole block unless its length
        # equals the block's full size.  Treating all offset-zero ranges as full
        # blocks used to load every expert (and the entire vocabulary matrix)
        # when callers requested only the first slice.
        if byte_offset == 0 and block_size is not None and byte_length == block_size:
            return block_id
        return f"{block_id}@{byte_offset}+{byte_length}"

    def _make_room(self, incoming_bytes: int) -> bool:
        """Evict LRU cache data until cache + pending + incoming fits."""
        budget = self.cache.max_bytes
        if budget is None:
            return True
        if incoming_bytes < 0 or incoming_bytes > budget:
            return False
        target_cache = budget - self.pending_bytes - incoming_bytes
        if target_cache < 0:
            return False
        while self.cache.stats.estimated_bytes > target_cache:
            oldest = next(self.cache.keys(), None)
            if oldest is None:
                break
            self.cache.discard(oldest)
            self.cache.stats.evictions += 1
        return self.cache.stats.estimated_bytes <= target_cache

    def _make_room_for_demand(self, incoming_bytes: int) -> bool:
        """Cancel speculative reads when a demanded range needs their space."""
        if self._make_room(incoming_bytes):
            return True
        for key in list(self.pending):
            self.wasted_prefetch_bytes += self._pending_size(key, self.index.get(key.split("@", 1)[0]))
            self._cancel_pending(key)
            if self._make_room(incoming_bytes):
                return True
        return self._make_room(incoming_bytes)

    def _submit(self, key: str, block: WeightBlock, byte_offset: int, byte_length: int) -> bool:
        if byte_length < 0 or byte_offset < 0 or byte_offset + byte_length > block.size:
            raise ValueError(f"invalid prefetch range for {block.block_id}")
        if byte_length == 0:
            return False
        if not self._make_room(byte_length):
            self.wasted_prefetch_bytes += byte_length
            return False
        try:
            future = self.executor.submit(self._load, block, byte_offset, byte_length)
        except RuntimeError:
            if not self._closed:
                raise
            return False
        self.pending[key] = future
        self.pending_bytes += byte_length
        self.prefetched += 1
        return True

    def _pending_size(self, key: str, block: WeightBlock) -> int:
        return _future_bytes(block, key)

    def _cancel_pending(self, key: str) -> None:
        future = self.pending.pop(key, None)
        if future is None:
            return
        block = self.index.get(key.split("@", 1)[0])
        reserved = self._pending_size(key, block) if block is not None else 0
        # If a read has already started, wait for it and release the buffer
        # before releasing the reservation.  Otherwise cancellation is enough.
        if not future.cancel():
            with suppress(Exception):
                future.result()  # a speculative read is discarded either way
        self.pending_bytes = max(0, self.pending_bytes - reserved)

    # ---------------------------------------------------------------- public API
    def prefetch(
        self,
        block_ids: Iterable[str],
        confidence: dict[str, float] | None = None,
        min_confidence: float = 0.0,
    ) -> None:
        confidence = confidence or {}
        for block_id in block_ids:
            if block_id in self.cache or block_id in self.pending:
                continue
            block = self.index.get(block_id)
            if block is None:
                continue
            score = max(0.0, min(1.0, confidence.get(block_id, 1.0)))
            if score < min_confidence:
                self.wasted_prefetch_bytes += block.size
                continue
            self._submit(block_id, block, 0, block.size)

    def prefetch_ranges(
        self,
        ranges: Iterable[tuple[str, int, int]],
        confidence: float = 1.0,
    ) -> None:
        """Prefetch ``(block_id, byte_offset, byte_length)`` ranges."""
        if confidence < 0.0:
            raise ValueError("confidence must be non-negative")
        for block_id, byte_offset, byte_length in ranges:
            block = self.index.get(block_id)
            if block is None:
                continue
            if byte_offset < 0 or byte_length < 0 or byte_offset + byte_length > block.size:
                raise ValueError(f"invalid prefetch range for {block_id}")
            key = self._range_id(block_id, byte_offset, byte_length, block.size)
            if key in self.cache or key in self.pending:
                continue
            if confidence == 0.0:
                self.wasted_prefetch_bytes += byte_length
                continue
            self._submit(key, block, byte_offset, byte_length)

    def _maybe_cache(self, key: str, payload: memoryview) -> None:
        budget = self.cache.max_bytes
        if budget is not None and len(payload) > budget:
            if self.allow_oversized_reads:
                self.uncached_bytes += len(payload)
                return
            raise MemoryError(
                f"loaded range is {len(payload)} bytes, larger than resident budget ({budget} bytes)"
            )
        if not self._make_room_for_demand(len(payload)):
            raise MemoryError("resident byte budget is fully reserved by pending reads")
        self.cache.put(key, payload, len(payload))

    def get(self, block_id: str) -> memoryview:
        hit = self.cache.get(block_id)
        if hit is not None:
            return hit
        block = self.index.get(block_id)
        if block is None:
            raise KeyError(f"unknown block: {block_id}")
        future = self.pending.pop(block_id, None)
        if future is not None:
            reserved = self._pending_size(block_id, block)
            try:
                payload = future.result()
            finally:
                self.pending_bytes = max(0, self.pending_bytes - reserved)
            self.prefetch_hits += 1
        else:
            oversized_allowed = (
                self.allow_oversized_reads
                and self.cache.max_bytes is not None
                and block.size > self.cache.max_bytes
            )
            if not oversized_allowed and not self._make_room_for_demand(block.size):
                raise MemoryError("resident byte budget is fully reserved by pending reads")
            payload = self._load(block)
        self._maybe_cache(block_id, payload)
        return payload

    def get_range(self, block_id: str, byte_offset: int, byte_length: int) -> memoryview:
        """Load (or reuse) a bounded byte range from a tensor block."""
        if byte_length < 0 or byte_offset < 0:
            raise ValueError("range offsets must be non-negative")
        block = self.index.get(block_id)
        if block is None:
            raise KeyError(f"unknown block: {block_id}")
        if byte_offset + byte_length > block.size:
            raise ValueError(
                f"range [{byte_offset}, {byte_offset + byte_length}) is outside {block_id}"
            )
        if byte_length == 0:
            return memoryview(b"")
        key = self._range_id(block_id, byte_offset, byte_length, block.size)
        if key == block_id:
            return self.get(block_id)
        hit = self.cache.get(key)
        if hit is not None:
            return hit
        if block_id in self.pending:
            full_block = self.get(block_id)
            return full_block[byte_offset:byte_offset + byte_length]
        if block_id in self.cache:
            full_block = self.cache.get(block_id)
            if full_block is not None:
                return full_block[byte_offset:byte_offset + byte_length]

        future = self.pending.pop(key, None)
        if future is not None:
            reserved = self._pending_size(key, block)
            try:
                payload = future.result()
            finally:
                self.pending_bytes = max(0, self.pending_bytes - reserved)
            self.prefetch_hits += 1
        else:
            oversized_allowed = (
                self.allow_oversized_reads
                and self.cache.max_bytes is not None
                and byte_length > self.cache.max_bytes
            )
            if not oversized_allowed and not self._make_room_for_demand(byte_length):
                raise MemoryError("resident byte budget is fully reserved by pending reads")
            payload = self._load(block, byte_offset, byte_length)
        self._maybe_cache(key, payload)
        return payload

    def cancel_prefetch(self, block_ids: Iterable[str]) -> None:
        for block_id in block_ids:
            self._cancel_pending(block_id)

    def clear_prefetch(self) -> None:
        """Cancel all queued work and wait for in-flight buffers to be dropped."""
        for key in list(self.pending):
            self._cancel_pending(key)
        self.pending.clear()
        self.pending_bytes = 0

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.clear_prefetch()
        self.executor.shutdown(wait=True, cancel_futures=True)

    def stats(self) -> StreamStats:
        with self._stats_lock:
            loads = self.loads
            bytes_loaded = self.bytes_loaded
            io_seconds = self.io_seconds
        return StreamStats(
            loads=loads,
            prefetched=self.prefetched,
            prefetch_hits=self.prefetch_hits,
            cache_hits=self.cache.stats.hits,
            pending=len(self.pending),
            resident_bytes=self.cache.stats.estimated_bytes + self.pending_bytes,
            evictions=self.cache.stats.evictions,
            wasted_prefetch_bytes=self.wasted_prefetch_bytes,
            uncached_bytes=self.uncached_bytes,
            bytes_loaded=bytes_loaded,
            io_seconds=io_seconds,
        )

    def __enter__(self) -> BlockStreamer:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def _future_bytes(block: WeightBlock | None, key: str) -> int:
    """Bytes reserved by a pending prefetch key."""
    if block is None:
        return 0
    if key == block.block_id:
        return block.size
    _, _, tail = key.partition("@")
    _, _, length = tail.partition("+")
    try:
        return int(length)
    except ValueError:
        return block.size
