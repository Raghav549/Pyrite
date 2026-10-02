from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..ggml_types import row_size
from ..ggml_types import spec as ggml_spec


@dataclass(frozen=True)
class GGUFHeader:
    version: int
    tensor_count: int
    metadata_count: int
    alignment: int
    data_offset: int


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

    @property
    def element_count(self) -> int:
        total = 1
        for dim in self.dims:
            total *= dim
        return total if self.dims else 0


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
    """Bounds-checked reader for GGUF v1-v3 checkpoints."""

    MAGIC = b"GGUF"
    DEFAULT_ALIGNMENT = 32
    MAX_TENSORS = 10_000_000
    MAX_METADATA = 1_000_000
    MAX_ARRAY = 100_000_000

    TYPE_UINT8, TYPE_INT8, TYPE_UINT16, TYPE_INT16 = 0, 1, 2, 3
    TYPE_UINT32, TYPE_INT32, TYPE_FLOAT32, TYPE_BOOL = 4, 5, 6, 7
    TYPE_STRING, TYPE_ARRAY, TYPE_UINT64, TYPE_INT64, TYPE_FLOAT64 = 8, 9, 10, 11, 12

    def __init__(self, path: Path):
        self.path = Path(path)
        self._parsed = False
        self._header: GGUFHeader | None = None
        self._metadata: dict[str, Any] = {}
        self._tensors: tuple[GGUFTensor, ...] = ()
        self._index: dict[str, GGUFTensor] = {}
        self._size = 0

    def header(self) -> GGUFHeader:
        self._ensure_parsed()
        assert self._header is not None
        return self._header

    def metadata(self) -> dict[str, Any]:
        self._ensure_parsed()
        return dict(self._metadata)

    def get(self, key: str, default: Any = None) -> Any:
        self._ensure_parsed()
        return self._metadata.get(key, default)

    def tensor_index(self) -> tuple[GGUFTensor, ...]:
        self._ensure_parsed()
        return self._tensors

    def tensor(self, name: str) -> GGUFTensor:
        self._ensure_parsed()
        try:
            return self._index[name]
        except KeyError as exc:
            raise KeyError(f"missing GGUF tensor: {name}") from exc

    def has_tensor(self, name: str) -> bool:
        self._ensure_parsed()
        return name in self._index

    def tensor_names(self) -> tuple[str, ...]:
        return tuple(tensor.name for tensor in self.tensor_index())

    @property
    def tensor_data_offset(self) -> int:
        self._ensure_parsed()
        assert self._header is not None
        return self._header.data_offset

    @property
    def size_bytes(self) -> int:
        self._ensure_parsed()
        return self._size

    def read_tensor(self, tensor: GGUFTensor) -> bytes:
        self._ensure_parsed()
        if tensor.offset < 0 or tensor.end > self._size:
            raise ValueError(f"tensor {tensor.name!r} lies outside the checkpoint")
        with self.path.open("rb") as fh:
            fh.seek(tensor.offset)
            data = fh.read(tensor.size)
        if len(data) != tensor.size:
            raise ValueError(f"truncated tensor payload: {tensor.name}")
        return data

    def metadata_summary(self, max_array_items: int = 8) -> dict[str, Any]:
        """Small, printable view of the metadata (skips huge token arrays)."""

        def summarize(value: Any) -> Any:
            if isinstance(value, (list, tuple)):
                if len(value) <= max_array_items:
                    return [summarize(item) for item in value]
                return {
                    "count": len(value),
                    "first": [summarize(item) for item in value[:max_array_items]],
                }
            if isinstance(value, (bytes, bytearray)):
                return f"<{len(value)} bytes>"
            if isinstance(value, (int, float, bool, str)) or value is None:
                return value
            return repr(value)

        self._ensure_parsed()
        return {key: summarize(value) for key, value in self._metadata.items()}

    def _ensure_parsed(self) -> None:
        if self._parsed:
            return
        if not self.path.is_file():
            raise FileNotFoundError(self.path)

        size = self.path.stat().st_size
        with self.path.open("rb") as fh:
            r = _Reader(fh)
            if r.raw(4) != self.MAGIC:
                raise ValueError("not a GGUF checkpoint")
            version = r.u32()
            if version < 1 or version > 3:
                raise ValueError(f"unsupported GGUF version: {version}")
            tensor_count, metadata_count = r.u64(), r.u64()
            if tensor_count > self.MAX_TENSORS or metadata_count > self.MAX_METADATA:
                raise ValueError("GGUF counts exceed safety limits")

            metadata: dict[str, Any] = {}
            alignment = self.DEFAULT_ALIGNMENT
            for _ in range(metadata_count):
                key, value_type = r.string(), r.u32()
                if key in metadata:
                    raise ValueError(f"duplicate GGUF metadata key: {key}")
                value = self._read_value(r, value_type)
                metadata[key] = value
                if key == "general.alignment":
                    if not isinstance(value, int) or value <= 0 or value & (value - 1):
                        raise ValueError(f"invalid GGUF alignment: {value!r}")
                    alignment = value

            descriptors: list[tuple[str, tuple[int, ...], int, int]] = []
            seen: set[str] = set()
            for _ in range(tensor_count):
                name, n_dims = r.string(), r.u32()
                if name in seen:
                    raise ValueError(f"duplicate GGUF tensor name: {name}")
                seen.add(name)
                if n_dims == 0 or n_dims > 8:
                    raise ValueError(f"GGUF tensor {name!r} has invalid dimension count: {n_dims}")
                dims = tuple(r.u64() for _ in range(n_dims))
                if any(dim == 0 for dim in dims):
                    raise ValueError(f"GGUF tensor {name!r} has a zero-sized dimension")
                descriptors.append((name, dims, r.u32(), r.u64()))

            position = fh.tell()
            data_offset = (
                position + (alignment - position % alignment) % alignment
                if tensor_count
                else position
            )
            if data_offset > size:
                raise ValueError("GGUF tensor data starts beyond end of file")
            tensors: list[GGUFTensor] = []
            extents: list[tuple[int, int, str]] = []
            for name, dims, ggml_type, relative_offset in descriptors:
                if relative_offset % alignment:
                    raise ValueError(f"unaligned GGUF tensor offset: {name}")
                tensor_bytes = self._tensor_size(dims, ggml_type, name)
                absolute = data_offset + relative_offset
                end = absolute + tensor_bytes
                if end > size:
                    raise ValueError(f"GGUF tensor extends beyond file: {name}")
                tensors.append(GGUFTensor(name, dims, ggml_type, absolute, tensor_bytes))
                extents.append((absolute, end, name))
            previous_end = data_offset
            previous_name = ""
            for start, end, name in sorted(extents):
                if start < previous_end:
                    raise ValueError(
                        f"GGUF tensor payload overlaps another tensor: {previous_name}, {name}"
                    )
                previous_end = end
                previous_name = name

        self._header = GGUFHeader(version, tensor_count, metadata_count, alignment, data_offset)
        self._metadata = metadata
        self._tensors = tuple(tensors)
        self._index = {tensor.name: tensor for tensor in tensors}
        self._size = size
        self._parsed = True

    def _read_value(self, r: _Reader, value_type: int) -> Any:
        if value_type == self.TYPE_UINT8: return r.u8()
        if value_type == self.TYPE_INT8: return r.i8()
        if value_type == self.TYPE_UINT16: return r.u16()
        if value_type == self.TYPE_INT16: return r.i16()
        if value_type == self.TYPE_UINT32: return r.u32()
        if value_type == self.TYPE_INT32: return r.i32()
        if value_type == self.TYPE_FLOAT32: return r.f32()
        if value_type == self.TYPE_BOOL:
            value = r.u8()
            if value not in (0, 1):
                raise ValueError(f"invalid GGUF boolean value: {value}")
            return bool(value)
        if value_type == self.TYPE_STRING: return r.string()
        if value_type == self.TYPE_ARRAY:
            element_type, count = r.u32(), r.u64()
            if count > self.MAX_ARRAY:
                raise ValueError("GGUF metadata array exceeds safety limit")
            return tuple(self._read_value(r, element_type) for _ in range(count))
        if value_type == self.TYPE_UINT64: return r.u64()
        if value_type == self.TYPE_INT64: return r.i64()
        if value_type == self.TYPE_FLOAT64: return r.f64()
        raise ValueError(f"unsupported GGUF metadata type: {value_type}")

    @staticmethod
    def _tensor_size(dims: tuple[int, ...], ggml_type: int, name: str = "") -> int:
        if not dims:
            raise ValueError("GGUF tensor must have at least one dimension")
        if any(dim <= 0 for dim in dims):
            raise ValueError("GGUF tensor dimensions must be positive")
        rows = 1
        for dim in dims[1:]:
            rows *= dim
        try:
            return row_size(dims[0], ggml_type) * rows
        except (TypeError, ValueError) as exc:
            label = f"tensor {name!r}: " if name else ""
            raise ValueError(f"{label}{exc}") from exc

    def describe_type(self, ggml_type: int) -> str:
        return ggml_spec(ggml_type).name


# Compatibility import for legacy callers.
try:
    from .gguf_adapter import GGUFAdapter
except ImportError:
    GGUFAdapter = None
