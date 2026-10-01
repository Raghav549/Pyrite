from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import struct

from .base import WeightBlock


@dataclass(frozen=True)
class SafeTensor:
    name: str
    dtype: str
    shape: tuple[int, ...]
    offset: int
    size: int


class SafetensorsAdapter:
    name = "safetensors"

    def __init__(self, path: Path):
        self.path = Path(path)
        self._tensors = self._read_header()

    def _read_header(self) -> tuple[SafeTensor, ...]:
        with self.path.open("rb") as fh:
            raw = fh.read(8)
            if len(raw) != 8:
                raise ValueError("truncated safetensors header")
            header_len = struct.unpack("<Q", raw)[0]
            if header_len > 128 * 1024 * 1024:
                raise ValueError("safetensors header exceeds safety limit")
            header = fh.read(header_len)
            if len(header) != header_len:
                raise ValueError("truncated safetensors header")
        try:
            data = json.loads(header.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("invalid safetensors header") from exc
        if not isinstance(data, dict):
            raise ValueError("safetensors header must be an object")

        result: list[SafeTensor] = []
        for name, meta in data.items():
            if name == "__metadata__":
                continue
            if not isinstance(meta, dict):
                raise ValueError(f"invalid tensor metadata: {name}")
            dtype = meta.get("dtype")
            shape = meta.get("shape")
            offsets = meta.get("data_offsets")
            if not isinstance(dtype, str) or not isinstance(shape, list) or not isinstance(offsets, list) or len(offsets) != 2:
                raise ValueError(f"invalid tensor descriptor: {name}")
            start, end = offsets
            if not isinstance(start, int) or not isinstance(end, int) or start < 0 or end < start:
                raise ValueError(f"invalid tensor offsets: {name}")
            if not all(isinstance(x, int) and x >= 0 for x in shape):
                raise ValueError(f"invalid tensor shape: {name}")
            result.append(SafeTensor(name, dtype, tuple(shape), start, end - start))
        return tuple(result)

    def blocks(self) -> list[WeightBlock]:
        header_bytes = 8
        with self.path.open("rb") as fh:
            header_len = struct.unpack("<Q", fh.read(8))[0]
        data_start = header_bytes + header_len
        return [
            WeightBlock(
                block_id=f"tensor:{tensor.name}",
                path=self.path,
                offset=data_start + tensor.offset,
                size=tensor.size,
                dtype=tensor.dtype,
                kind="layer",
                index=index,
            )
            for index, tensor in enumerate(self._tensors)
        ]

    def load(self, block: WeightBlock) -> memoryview:
        if block.path != self.path:
            raise ValueError("block belongs to a different checkpoint")
        with self.path.open("rb") as fh:
            fh.seek(block.offset)
            payload = fh.read(block.size)
        if len(payload) != block.size:
            raise ValueError(f"truncated safetensors tensor: {block.block_id}")
        return memoryview(payload)
