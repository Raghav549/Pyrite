from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path

from .base import TensorMeta, WeightBlock

#: Safetensors dtype name -> (byte width, ggml type id or None).
DTYPES: dict[str, tuple[int, int | None]] = {
    "F64": (8, 28),
    "F32": (4, 0),
    "F16": (2, 1),
    "BF16": (2, 30),
    "I64": (8, 27),
    "I32": (4, 26),
    "I16": (2, 25),
    "I8": (1, 24),
    # U8/I8 have no signed ggml equivalent for the unsigned case; keep None so
    # consumers fall back to the safetensors dtype instead of guessing.
    "U8": (1, None),
}


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
        self.header_bytes, self._tensors = self._read_header()
        self._blocks = self._build_blocks()

    def _read_header(self) -> tuple[int, tuple[SafeTensor, ...]]:
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
            raise TypeError("safetensors header must be an object")

        data_start = 8 + header_len
        file_size = self.path.stat().st_size
        result: list[SafeTensor] = []
        for name, meta in data.items():
            if name == "__metadata__":
                continue
            if not isinstance(meta, dict) or not isinstance(name, str) or not name:
                raise ValueError(f"invalid tensor metadata: {name!r}")
            dtype = meta.get("dtype")
            shape = meta.get("shape")
            offsets = meta.get("data_offsets")
            if not isinstance(dtype, str) or not isinstance(shape, list) or not isinstance(offsets, list) or len(offsets) != 2:
                raise ValueError(f"invalid tensor descriptor: {name}")
            if dtype not in DTYPES:
                raise ValueError(f"unsupported safetensors dtype {dtype!r} for {name}")
            start, end = offsets
            if not isinstance(start, int) or not isinstance(end, int) or start < 0 or end < start:
                raise ValueError(f"invalid tensor offsets: {name}")
            if not all(isinstance(x, int) and x >= 0 for x in shape):
                raise ValueError(f"invalid tensor shape: {name}")
            elements = 1
            for dim in shape:
                elements *= dim
            expected = elements * DTYPES[dtype][0]
            if end - start != expected:
                raise ValueError(
                    f"tensor {name!r} declares {end - start} payload bytes but "
                    f"{dtype} x {shape} needs {expected}"
                )
            if data_start + end > file_size:
                raise ValueError(f"tensor {name!r} extends beyond the checkpoint")
            result.append(SafeTensor(name, dtype, tuple(shape), start, end - start))
        return header_len, tuple(result)

    def _build_blocks(self) -> list[WeightBlock]:
        data_start = 8 + self.header_bytes
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

    def blocks(self) -> list[WeightBlock]:
        return list(self._blocks)

    def tensors(self) -> list[TensorMeta]:
        return [
            TensorMeta(
                name=tensor.name,
                block_id=f"tensor:{tensor.name}",
                shape=tensor.shape,
                dtype=tensor.dtype,
                ggml_type=DTYPES[tensor.dtype][1],
                offset=block.offset,
                size=tensor.size,
            )
            for tensor, block in zip(self._tensors, self._blocks, strict=True)
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
