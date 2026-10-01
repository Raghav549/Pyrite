from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .checkpoint import open_checkpoint
from .config import RuntimeConfig
from .engine import PyriteRuntime
from .stream import BlockStreamer


@dataclass(frozen=True)
class LocalModelInfo:
    path: Path
    format: str
    block_count: int
    total_bytes: int


class LocalModel:
    """Owns the local checkpoint, runtime budget and streaming layer."""

    def __init__(
        self,
        path: str | Path,
        config: RuntimeConfig | None = None,
        chunk_bytes: int = 64 * 1024 * 1024,
    ):
        self.runtime = PyriteRuntime(config)
        self.checkpoint = open_checkpoint(path, chunk_bytes=chunk_bytes)
        self.streamer = BlockStreamer(
            self.checkpoint.adapter,
            resident_blocks=self.runtime.config.resident_blocks,
            resident_bytes=self.runtime.config.resident_byte_budget,
            workers=1,
        )

    def info(self) -> LocalModelInfo:
        blocks = self.checkpoint.adapter.blocks()
        return LocalModelInfo(
            path=self.checkpoint.info.path,
            format=self.checkpoint.info.format,
            block_count=len(blocks),
            total_bytes=sum(block.size for block in blocks),
        )

    def prefetch_next(self, index: int) -> None:
        blocks = self.checkpoint.adapter.blocks()
        ids = [block.block_id for block in blocks[index + 1 : index + 1 + self.runtime.config.prefetch_depth]]
        self.streamer.prefetch(ids)

    def load_block(self, block_id: str) -> memoryview:
        return self.streamer.get(block_id)

    def close(self) -> None:
        self.streamer.close()

    def __enter__(self) -> LocalModel:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
