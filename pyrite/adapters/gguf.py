from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import struct
from typing import Any

from ..ggml_types import tensor_size


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


@dataclass(frozen=True)
class GGUFTensor:
    name: str
    dims: tuple[int, ...]
    ggml_type: int
    offset: int
    size: int

    @property
    def end(self) -> int:
        return self.offset + self.size


class _Reader:
    def __init__(self, fh):
        self.fh = fh

    def u8(self) -> int: return struct.unpack("<B", self._read(1))[0]
    def i8(self) -> int: return struct.unpack("<b", self._read(1))[0]
    def u16(self) -> int: return struct.unpack("<H", self._read(2))[0]
    def i16(self) -> int: return struct.unpack("<h", self._read(2))[0]
    def u32(self) -> int: return struct.unpack("<I", self._read(4))[0]
    def i32(self) -> int: return struct.unpack("<i", self._read(4))[0]
    def u64(self) -> int: return struct.unpack("<Q", self._read(8))[0]
    def i64(self) -> int: return struct.unpack("<q", self._read(8))[0]
    def f32(self) -> float: return struct.unpack("<f", self._read(4))[0]
    def f64(self) -> float: return struct.unpack("<d", self._read(8))[0]

    def raw(self, size: int) -> bytes:
        return self._read(size)

    def string(self) -> str:
        size = self.u64()
        if size > 1024 * 1024:
            raise ValueError("GGUF string exceeds safety limit")
        raw = self._read(size)
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("invalid UTF-8 string in GGUF") from exc

    def _read(self, size: int) -> bytes:
        data = self.fh.read(size)
        if len(data) != size:
            raise ValueError("truncated GGUF checkpoint")
        return data


class GGUFReader:
    MAGIC = b"GGUF"
    DEFAULT_ALIGNMENT = 32
    TYPE_UINT8, TYPE_INT8, TYPE_UINT16, TYPE_INT16 = 0, 1, 2, 3
    TYPE_UINT32, TYPE_INT32, TYPE_FLOAT32, TYPE_BOOL = 4, 5, 6, 7
    TYPE_STRING, TYPE_ARRAY, TYPE_UINT64, TYPE_INT64, TYPE_FLOAT64 = 8, 9, 10, 11, 12

    def __init__(self, path: Path):
        self.path = Path(path)
        self._parsed = False
        self._header: GGUFHeader | None = None
        self._metadata: dict[str, Any] = {}
        self._tensors: tuple[GGUFTensor, ...] = ()
        self._data_offset = 0

    def header(self) -> GGUFHeader:
        self._ensure_parsed(); assert self._header is not None; return self._header

    def metadata(self) -> dict[str, Any]:
        self._ensure_parsed(); return dict(self._metadata)

    def tensor_index(self) -> tuple[GGUFTensor, ...]:
        self._ensure_parsed(); return self._tensors

    @property
    def tensor_data_offset(self) -> int:
        self._ensure_parsed(); return self._data_offset

    def read_tensor(self, tensor: GGUFTensor) -> bytes:
        self._ensure_parsed()
        with self.path.open("rb") as fh:
            fh.seek(tensor.offset); data = fh.read(tensor.size)
        if len(data) != tensor.size:
            raise ValueError(f"truncated tensor payload: {tensor.name}")
        return data

    def _ensure_parsed(self) -> None:
        if self._parsed: return
        if not self.path.is_file(): raise FileNotFoundError(self.path)
        with self.path.open("rb") as fh:
            r = _Reader(fh)
            if r.raw(4) != self.MAGIC: raise ValueError("not a GGUF checkpoint")
            version = r.u32()
            if version < 1 or version > 3: raise ValueError(f"unsupported GGUF version: {version}")
            tensor_count, metadata_count = r.u64(), r.u64()
            if tensor_count > 10_000_000 or metadata_count > 1_000_000:
                raise ValueError("GGUF counts exceed safety limits")
            metadata: dict[str, Any] = {}
            alignment = self.DEFAULT_ALIGNMENT
            for _ in range(metadata_count):
                key, value_type = r.string(), r.u32()
                value = self._read_value(r, value_type); metadata[key] = value
                if key == "general.alignment" and isinstance(value, int) and value > 0: alignment = value
            descriptors: list[tuple[str, tuple[int, ...], int, int]] = []
            for _ in range(tensor_count):
                name, n_dims = r.string(), r.u32()
                if n_dims > 8: raise ValueError("GGUF tensor has too many dimensions")
                dims = tuple(r.u64() for _ in range(n_dims))
                descriptors.append((name, dims, r.u32(), r.u64()))
            position = fh.tell()
            data_offset = position + (alignment - position % alignment) % alignment if tensor_count else position
            tensors: list[GGUFTensor] = []
            for index, (name, dims, ggml_type, relative_offset) in enumerate(descriptors):
                if alignment <= 0 or relative_offset % alignment != 0:
                    raise ValueError(f"unaligned GGUF tensor offset: {name}")
                size = self._tensor_size(dims, ggml_type); absolute = data_offset + relative_offset
                if absolute + size > self.path.stat().st_size: raise ValueError(f"GGUF tensor extends beyond file: {name}")
                if index + 1 < len(descriptors) and data_offset + descriptors[index + 1][3] < absolute:
                    raise ValueError("GGUF tensor offsets are not monotonic")
                tensors.append(GGUFTensor(name, dims, ggml_type, absolute, size))
        self._header = GGUFHeader(version, tensor_count, metadata_count)
        self._metadata, self._tensors, self._data_offset, self._parsed = metadata, tuple(tensors), data_offset, True

    def _read_value(self, r: _Reader, value_type: int) -> Any:
        if value_type == self.TYPE_UINT8: return r.u8()
        if value_type == self.TYPE_INT8: return r.i8()
        if value_type == self.TYPE_UINT16: return r.u16()
        if value_type == self.TYPE_INT16: return r.i16()
        if value_type == self.TYPE_UINT32: return r.u32()
        if value_type == self.TYPE_INT32: return r.i32()
        if value_type == self.TYPE_FLOAT32: return r.f32()
        if value_type == self.TYPE_BOOL: return bool(r.u8())
        if value_type == self.TYPE_STRING: return r.string()
        if value_type == self.TYPE_ARRAY:
            element_type, count = r.u32(), r.u64()
            if count > 10_000_000: raise ValueError("GGUF metadata array exceeds safety limit")
            return tuple(self._read_value(r, element_type) for _ in range(count))
        if value_type == self.TYPE_UINT64: return r.u64()
        if value_type == self.TYPE_INT64: return r.i64()
        if value_type == self.TYPE_FLOAT64: return r.f64()
        raise ValueError(f"unsupported GGUF metadata type: {value_type}")

    def _tensor_size(self, dims: tuple[int, ...], ggml_type: int) -> int:
        if not dims: return 0
        elements = 1
        for dim in dims:
            if dim < 0: raise ValueError("GGUF tensor dimensions cannot be negative")
            if dim == 0: return 0
            elements *= dim
        return tensor_size(elements, ggml_type)


class GGUFAdapter:
    """Compatibility adapter retained for existing backend imports."""
    name = "gguf"

    def __init__(self, path: Path):
        self.path = Path(path)
        self.reader = GGUFReader(self.path)

    def blocks(self):
        from .base import WeightBlock
        return [WeightBlock(block_id=f"tensor:{t.name}", path=self.path, offset=t.offset,
                            size=t.size, dtype=f"ggml:{t.ggml_type}", kind="layer", index=i)
                for i, t in enumerate(self.reader.tensor_index())]

    def load(self, block):
        if block.path != self.path: raise ValueError("block belongs to a different checkpoint")
        if block.offset < 0 or block.size < 0: raise ValueError("invalid GGUF block range")
        with self.path.open("rb") as fh:
            fh.seek(block.offset); payload = fh.read(block.size)
        if len(payload) != block.size: raise ValueError(f"truncated GGUF tensor block: {block.block_id}")
        return memoryview(payload)
