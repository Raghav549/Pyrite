from __future__ import annotations

import math
import struct
from typing import Sequence


def f32_vector(data: bytes, count: int) -> list[float]:
    if count < 0 or len(data) < count * 4:
        raise ValueError("insufficient F32 data")
    return list(struct.unpack("<" + "f" * count, data[:count * 4]))


def bf16_vector(data: bytes, count: int) -> list[float]:
    if count < 0 or len(data) < count * 2:
        raise ValueError("insufficient BF16 data")
    out = []
    for i in range(count):
        word = struct.unpack_from("<H", data, i * 2)[0]
        out.append(struct.unpack("<f", struct.pack("<I", word << 16))[0])
    return out


def f16_vector(data: bytes, count: int) -> list[float]:
    if count < 0 or len(data) < count * 2:
        raise ValueError("insufficient F16 data")
    return [struct.unpack_from("<e", data, i * 2)[0] for i in range(count)]


def q4_0_vector(data: bytes, count: int) -> list[float]:
    if count % 32:
        raise ValueError("Q4_0 count must be divisible by 32")
    required = (count // 32) * 18
    if len(data) < required:
        raise ValueError("insufficient Q4_0 data")
    out: list[float] = []
    pos = 0
    for _ in range(count // 32):
        scale = struct.unpack_from("<e", data, pos)[0]
        pos += 2
        qs = data[pos:pos + 16]
        pos += 16
        for byte in qs:
            out.append(scale * ((byte & 0x0F) - 8))
            out.append(scale * ((byte >> 4) - 8))
    return out


def q8_0_vector(data: bytes, count: int) -> list[float]:
    if count % 32:
        raise ValueError("Q8_0 count must be divisible by 32")
    required = (count // 32) * 34
    if len(data) < required:
        raise ValueError("insufficient Q8_0 data")
    out: list[float] = []
    pos = 0
    for _ in range(count // 32):
        scale = struct.unpack_from("<e", data, pos)[0]
        pos += 2
        for i in range(32):
            q = struct.unpack_from("<b", data, pos)[0]
            pos += 1
            out.append(scale * q)
    return out


def matvec(values: Sequence[float], rows: int, cols: int, vector: Sequence[float]) -> list[float]:
    if rows <= 0 or cols <= 0 or len(values) != rows * cols or len(vector) != cols:
        raise ValueError("matrix/vector shape mismatch")
    return [
        sum(values[r * cols + c] * vector[c] for c in range(cols))
        for r in range(rows)
    ]


def rms_norm(x: Sequence[float], weight: Sequence[float], eps: float = 1e-5) -> list[float]:
    if len(x) != len(weight):
        raise ValueError("RMSNorm shape mismatch")
    inv = 1.0 / math.sqrt(sum(v * v for v in x) / max(1, len(x)) + eps)
    return [x[i] * inv * weight[i] for i in range(len(x))]
