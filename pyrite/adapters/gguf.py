from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import struct


@dataclass(frozen=True)
class GGUFHeader:
    version: int
    tensor_count: int
    metadata_count: int


@dataclass(frozen=True)
class GGUFMinimalTensor:
    name: str
    offset: int
    size: int


class GGUFReader:
    MAGIC = b"GGUF"

    def __init__(self, path: Path):
        self.path = path

    def header(self) -> GGUFHeader:
        with self.path.open("rb") as fh:
            magic = fh.read(4)
            if magic != self.MAGIC:
                raise ValueError("not a GGUF checkpoint")
            raw = fh.read(24)
            if len(raw) != 24:
                raise ValueError("truncated GGUF header")
            version, tensor_count, metadata_count = struct.unpack("<QQQ", raw)
            return GGUFHeader(version, tensor_count, metadata_count)

    def tensor_index(self) -> tuple[GGUFMinimalTensor, ...]:
        """Return an empty-safe index contract until full GGUF metadata parsing is attached.

        GGUF contains variable-length metadata and tensor descriptors. We avoid
        guessing descriptor layouts here; a format-complete parser must decode
        the official GGUF value types before producing real tensor offsets.
        """
        header = self.header()
        return tuple(
            GGUFMinimalTensor(name=f"tensor:{i}", offset=0, size=0)
            for i in range(header.tensor_count)
        )
