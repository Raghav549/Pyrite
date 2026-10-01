from __future__ import annotations

from collections.abc import Iterable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from .adapters.base import ModelAdapter, WeightBlock
from .memory import LRUResidentCache


@dataclass(frozen=True)
class StreamStats:
    loads: int
    prefetched: int
    cache_hits: int
    pending: int
    resident_bytes: int
    evictions: int
    wasted_prefetch_bytes: int
    uncached_bytes: int
    bytes_loaded: int


class BlockStreamer:
    """Bounded async streamer with confidence/cost-aware admission.

    Supports both whole blocks and byte ranges inside a block.  Ranged loads are
    what let MoE runtimes stream a single expert out of a stacked 3D tensor
    without materializing the other experts.
    """

    def __init__(
        self,
        adapter: ModelAdapter,
        resident_blocks: int | None = None,
        resident_bytes: int | None = None,
        workers: int = 1,
    ):
        if resident_bytes is not None and resident_bytes <= 0:
            raise ValueError("resident_bytes must be positive when provided")
        if resident_blocks is not None and resident_blocks < 1:
            raise ValueError("resident_blocks must be at least 1 when provided")
        self.adapter = adapter
        self.blocks = list(adapter.blocks())
        self.index = {block.block_id: block for block in self.blocks}
        # A count cap without a byte budget mirrors the classic "keep N layers
        # resident" policy; a byte budget is authoritative for tensor-granular
        # streaming, where a handful of small entries must not be evicted just
        # because a large one arrived.
        self.cache = LRUResidentCache[memoryview](
            None if resident_blocks is None else max(1, resident_blocks),
            resident_bytes,
        )
        self.executor = ThreadPoolExecutor(max_workers=max(1, workers))
        self.pending: dict[str, Future[memoryview]] = {}
        # Prefetched-but-unconsumed bytes are real memory; cap them at the same
        # budget as the cache so prefetching cannot blow the promise.
        self.max_pending_bytes = resident_bytes
        self.pending_bytes = 0
        self.loads = 0
        self.prefetched = 0
        self.wasted_prefetch_bytes = 0
        self.uncached_bytes = 0
        self.bytes_loaded = 0
        self._closed = False

    # ------------------------------------------------------------------ loading
    def _load(self, block: WeightBlock, byte_offset: int = 0, byte_length: int | None = None) -> memoryview:
        length = block.size - byte_offset if byte_length is None else byte_length
        if byte_offset < 0 or length < 0 or byte_offset + length > block.size:
            raise ValueError(f"range [{byte_offset}, {byte_offset + length}) is outside {block.block_id}")
        self.loads += 1
        with Path(block.path).open("rb") as fh:
            fh.seek(block.offset + byte_offset)
            payload = fh.read(length)
        if len(payload) != length:
            raise ValueError(f"truncated block read: {block.block_id}")
        self.bytes_loaded += length
        return memoryview(payload)

    @staticmethod
    def _range_id(block_id: str, byte_offset: int, byte_length: int) -> str:
        if byte_offset == 0:
            return block_id
        return f"{block_id}@{byte_offset}+{byte_length}"

    def _submit(self, key: str, block: WeightBlock, byte_offset: int, byte_length: int) -> bool:
        if self.max_pending_bytes is not None and self.pending_bytes + byte_length > self.max_pending_bytes:
            self.wasted_prefetch_bytes += byte_length
            return False
        self.pending[key] = self.executor.submit(self._load, block, byte_offset, byte_length)
        self.pending_bytes += byte_length
        self.prefetched += 1
        return True

    def _take_pending(self, key: str) -> Future[memoryview] | None:
        future = self.pending.pop(key, None)
        if future is not None:
            block = self.index.get(key.split("@", 1)[0])
            self.pending_bytes -= _future_bytes(block, key)
        return future

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

    def prefetch_ranges(self, ranges: Iterable[tuple[str, int, int]], confidence: float = 1.0) -> None:
        """Prefetch ``(block_id, byte_offset, byte_length)`` ranges."""
        if confidence < 0.0:
            raise ValueError("confidence must be non-negative")
        for block_id, byte_offset, byte_length in ranges:
            key = self._range_id(block_id, byte_offset, byte_length)
            if key in self.cache or key in self.pending:
                continue
            block = self.index.get(block_id)
            if block is None:
                continue
            if confidence == 0.0:
                self.wasted_prefetch_bytes += byte_length
                continue
            self._submit(key, block, byte_offset, byte_length)

    def _maybe_cache(self, key: str, payload: memoryview) -> None:
        """Cache a payload unless it is larger than the whole cache budget.

        A load bigger than the budget can still be *used*, it just cannot be
        retained; refusing it outright would make large tensors unreadable.
        """
        budget = self.cache.max_bytes
        if budget is not None and len(payload) > budget:
            self.uncached_bytes += len(payload)
            return
        self.cache.put(key, payload, len(payload))

    def get(self, block_id: str) -> memoryview:
        hit = self.cache.get(block_id)
        if hit is not None:
            return hit
        block = self.index.get(block_id)
        if block is None:
            raise KeyError(f"unknown block: {block_id}")
        future = self._take_pending(block_id)
        payload = future.result() if future is not None else self._load(block)
        self._maybe_cache(block_id, payload)
        return payload

    def get_range(self, block_id: str, byte_offset: int, byte_length: int) -> memoryview:
        """Load (or reuse) an exact byte range of ``block_id``."""
        if byte_length < 0 or byte_offset < 0:
            raise ValueError("range offsets must be non-negative")
        key = self._range_id(block_id, byte_offset, byte_length)
        if key != block_id:
            hit = self.cache.get(key)
            if hit is not None:
                return hit
        else:
            return self.get(block_id)

        block = self.index.get(block_id)
        if block is None:
            raise KeyError(f"unknown block: {block_id}")
        future = self._take_pending(key)
        payload = future.result() if future is not None else self._load(block, byte_offset, byte_length)
        self._maybe_cache(key, payload)
        return payload

    def cancel_prefetch(self, block_ids: Iterable[str]) -> None:
        for block_id in block_ids:
            future = self._take_pending(block_id)
            if future is not None:
                future.cancel()

    def clear_prefetch(self) -> None:
        """Drop every queued prefetch, keeping the pending-byte ledger honest."""
        for key in list(self.pending):
            self._take_pending(key)
        self.pending.clear()
        self.pending_bytes = 0

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.clear_prefetch()
        self.executor.shutdown(wait=True, cancel_futures=True)

    def stats(self) -> StreamStats:
        return StreamStats(
            loads=self.loads,
            prefetched=self.prefetched,
            cache_hits=self.cache.stats.hits,
            pending=len(self.pending),
            resident_bytes=self.cache.stats.estimated_bytes + self.pending_bytes,
            evictions=self.cache.stats.evictions,
            wasted_prefetch_bytes=self.wasted_prefetch_bytes,
            uncached_bytes=self.uncached_bytes,
            bytes_loaded=self.bytes_loaded,
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
