from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import struct

from .base import WeightBlock


@dataclass(frozen=True)
class CheckpointInfo:
    format: str
    path: Path
    size_bytes: int


def detect_checkpoint(path: Path) -> CheckpointInfo:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("rb") as fh:
        magic = fh.read(4)

    suffix = path.suffix.lower()
    if magic == b"GGUF":
        fmt = "gguf"
    elif suffix in {".safetensors", ".safetensor"}:
        fmt = "safetensors"
    elif suffix in {".ggml", ".bin"}:
        fmt = "ggml-or-raw"
    else:
        fmt = "unknown"
    return CheckpointInfo(fmt, path, path.stat().st_size)


class ChunkedFileAdapter:
    """Format-neutral adapter that exposes a large local file as bounded chunks.

    This is the bridge used before a full tensor-aware GGUF/Safetensors parser
    is attached. Chunks can be streamed without loading the whole checkpoint.
    """

    name = "chunked-file"

    def __init__(self, path: Path, chunk_bytes: int = 64 * 1024 * 1024):
        if chunk_bytes <= 0:
            raise ValueError("chunk_bytes must be positive")
        self.path = path
        self.chunk_bytes = chunk_bytes
        self.info = detect_checkpoint(path)

    def blocks(self) -> list[WeightBlock]:
        size = self.path.stat().st_size
        result = []
        offset = 0
        index = 0
        while offset < size:
            length = min(self.chunk_bytes, size - offset)
            result.append(
                WeightBlock(
                    block_id=f"chunk:{index:06d}",
                    path=self.path,
                    offset=offset,
                    size=length,
                    dtype=self.info.format,
                    kind="layer",
                    index=index,
                )
            )
            offset += length
            index += 1
        return result

    def load(self, block: WeightBlock) -> memoryview:
        with self.path.open("rb") as fh:
            fh.seek(block.offset)
            return memoryview(fh.read(block.size))
