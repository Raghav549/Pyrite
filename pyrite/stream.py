from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass

from .adapters.base import ModelAdapter, WeightBlock
from .memory import LRUResidentCache


@dataclass(frozen=True)
class StreamStats:
    loads: int
    prefetched: int
    cache_hits: int


class BlockStreamer:
    """Bounded async streamer for local model blocks."""

    def __init__(self, adapter: ModelAdapter, resident_blocks: int = 2, workers: int = 1):
        self.adapter = adapter
        self.blocks = list(adapter.blocks())
        self.index = {block.block_id: block for block in self.blocks}
        self.cache = LRUResidentCache[memoryview](max(1, resident_blocks))
        self.executor = ThreadPoolExecutor(max_workers=max(1, workers))
        self.pending: dict[str, Future[memoryview]] = {}
        self.loads = 0
        self.prefetched = 0

    def _load(self, block: WeightBlock) -> memoryview:
        self.loads += 1
        return self.adapter.load(block)

    def prefetch(self, block_ids: list[str]) -> None:
        for block_id in block_ids:
            if block_id in self.cache or block_id in self.pending:
                continue
            block = self.index.get(block_id)
            if block is None:
                continue
            self.pending[block_id] = self.executor.submit(self._load, block)
            self.prefetched += 1

    def get(self, block_id: str) -> memoryview:
        hit = self.cache.get(block_id)
        if hit is not None:
            return hit
        block = self.index.get(block_id)
        if block is None:
            raise KeyError(f"unknown block: {block_id}")
        future = self.pending.pop(block_id, None)
        payload = future.result() if future is not None else self._load(block)
        self.cache.put(block_id, payload, len(payload))
        return payload

    def close(self) -> None:
        self.executor.shutdown(wait=True, cancel_futures=True)

    def stats(self) -> StreamStats:
        return StreamStats(self.loads, self.prefetched, self.cache.stats.hits)

    def __enter__(self) -> "BlockStreamer":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
