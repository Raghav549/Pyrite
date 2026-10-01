from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import struct


@dataclass(frozen=True)
class GGUFHeader:
    version: int
    tensor_count: int
    metadata_count: int


class GGUFReader:
    """Minimal GGUF header reader; tensor decoding remains backend-specific."""

    MAGIC = b"GGUF"

    def __init__(self, path: Path):
        self.path = path

    def header(self) -> GGUFHeader:
        with self.path.open("rb") as fh:
            magic = fh.read(4)
            if magic != self.MAGIC:
                raise ValueError("not a GGUF checkpoint")
            raw = fh.read(8 + 8 + 8)
            if len(raw) != 24:
                raise ValueError("truncated GGUF header")
            version, tensor_count, metadata_count = struct.unpack("<QQQ", raw)
            return GGUFHeader(version, tensor_count, metadata_count)
