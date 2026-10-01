"""Minimal GGUF v3 writer used to build *real* checkpoint files in tests.

The output is a genuine GGUF file: header, typed metadata, aligned tensor
directory and aligned tensor payloads, exactly the layout ``pyrite`` (and
llama.cpp) reads.  It exists so the test suite exercises real bytes on disk
instead of mocking the parser.
"""
from __future__ import annotations

import math
import struct
from collections.abc import Iterable, Sequence
from pathlib import Path

from pyrite.ggml_types import spec as ggml_spec
from pyrite.ggml_types import tensor_size

# GGUF metadata value types.
UINT8, INT8, UINT16, INT16 = 0, 1, 2, 3
UINT32, INT32, FLOAT32, BOOL = 4, 5, 6, 7
STRING, ARRAY, UINT64, INT64, FLOAT64 = 8, 9, 10, 11, 12

TYPE_SIZES = {
    UINT8: 1, INT8: 1, UINT16: 2, INT16: 2, UINT32: 4, INT32: 4,
    FLOAT32: 4, BOOL: 1, UINT64: 8, INT64: 8, FLOAT64: 8,
}

_PACK = {
    UINT8: "<B", INT8: "<b", UINT16: "<H", INT16: "<h", UINT32: "<I",
    INT32: "<i", FLOAT32: "<f", BOOL: "<B", UINT64: "<Q", INT64: "<q", FLOAT64: "<d",
}


def _encode_string(value: str) -> bytes:
    raw = value.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def _value_type(value: object) -> int:
    if isinstance(value, bool):
        return BOOL
    if isinstance(value, int):
        if value < 0:
            return INT32 if value >= -(2 ** 31) else INT64
        return UINT32 if value <= 0xFFFFFFFF else UINT64
    if isinstance(value, float):
        return FLOAT32
    if isinstance(value, str):
        return STRING
    raise TypeError(f"unsupported metadata value: {value!r}")


def encode_value(value: object, value_type: int | None = None) -> tuple[int, bytes]:
    """Encode one metadata value; arrays take their element type from item 0."""
    if isinstance(value, (list, tuple)):
        items = list(value)
        element_type = value_type if value_type is not None else _value_type(items[0])
        payload = struct.pack("<IQ", element_type, len(items))
        for item in items:
            _, raw = encode_value(item, element_type)
            payload += raw
        return ARRAY, payload
    kind = value_type if value_type is not None else _value_type(value)
    if kind == STRING:
        return kind, _encode_string(str(value))
    if kind == BOOL:
        return kind, struct.pack("<B", 1 if value else 0)
    return kind, struct.pack(_PACK[kind], value)


class GGUFFileBuilder:
    """Build a GGUF v3 file with aligned tensors."""

    def __init__(self, architecture: str = "qwen3moe", alignment: int = 32):
        if alignment <= 0 or alignment & (alignment - 1):
            raise ValueError("alignment must be a positive power of two")
        self.architecture = architecture
        self.alignment = alignment
        self.metadata: list[tuple[str, bytes]] = []
        self.tensors: list[tuple[str, tuple[int, ...], int, bytes]] = []
        self.add("general.architecture", architecture)
        self.add("general.alignment", alignment)

    # ---------------------------------------------------------------- metadata
    def add(self, key: str, value: object, value_type: int | None = None) -> GGUFFileBuilder:
        kind, payload = encode_value(value, value_type)
        self.metadata.append((key, struct.pack("<I", kind) + payload))
        return self

    def set(self, key: str, value: object, value_type: int | None = None) -> GGUFFileBuilder:
        self.metadata = [(k, raw) for k, raw in self.metadata if k != key]
        return self.add(key, value, value_type)

    # ----------------------------------------------------------------- tensors
    def add_tensor(
        self,
        name: str,
        dims: Sequence[int],
        ggml_type: int,
        payload: bytes,
    ) -> GGUFFileBuilder:
        elements = math.prod(int(d) for d in dims) if dims else 0
        expected = tensor_size(elements, ggml_type)
        if len(payload) != expected:
            raise ValueError(
                f"tensor {name}: payload is {len(payload)} bytes, expected {expected} "
                f"for {ggml_spec(ggml_type).name} x {tuple(dims)}"
            )
        self.tensors.append((name, tuple(int(d) for d in dims), ggml_type, payload))
        return self

    # ------------------------------------------------------------------- write
    def to_bytes(self) -> bytes:
        header = b"GGUF" + struct.pack("<IQQ", 3, len(self.tensors), len(self.metadata))
        meta = b"".join(_encode_string(key) + raw for key, raw in self.metadata)

        directory = bytearray()
        offset = 0
        offsets: list[int] = []
        for name, dims, _ggml_type, payload in self.tensors:
            offsets.append(offset)
            directory += _encode_string(name)
            directory += struct.pack("<I", len(dims))
            directory += b"".join(struct.pack("<Q", dim) for dim in dims)
            directory += struct.pack("<IQ", _ggml_type, offset)
            offset += len(payload)
            offset += (-offset) % self.alignment

        body = bytearray()
        for (_name, _dims, _ggml_type, payload), tensor_offset in zip(self.tensors, offsets, strict=True):
            while len(body) < tensor_offset:
                body.append(0)
            body += payload
        while len(body) % self.alignment:
            body.append(0)

        prefix = bytearray(header + meta + bytes(directory))
        while len(prefix) % self.alignment:
            prefix.append(0)
        return bytes(prefix + body)

    def write(self, path: Path) -> Path:
        path = Path(path)
        path.write_bytes(self.to_bytes())
        return path


def f32_bytes(values: Iterable[float]) -> bytes:
    values = list(values)
    return struct.pack("<" + "f" * len(values), *values)


def f16_bytes(values: Iterable[float]) -> bytes:
    values = list(values)
    return struct.pack("<" + "e" * len(values), *values)
