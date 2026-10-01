from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class QuantizedBlock:
    scale: float
    zero: int
    values: bytes


def dequantize_int8(block: QuantizedBlock) -> list[float]:
    return [(int(v) - block.zero) * block.scale for v in block.values]


def quantize_int8(values: list[float]) -> QuantizedBlock:
    if not values:
        return QuantizedBlock(1.0, 0, b"")
    lo, hi = min(values), max(values)
    if hi == lo:
        return QuantizedBlock(1.0, 0, bytes(len(values)))
    scale = (hi - lo) / 255.0
    raw = bytes(max(0, min(255, round((value - lo) / scale))) for value in values)
    return QuantizedBlock(scale, 0, raw)


def pack_ternary(values: list[int]) -> bytes:
    if any(value not in (-1, 0, 1) for value in values):
        raise ValueError("ternary values must be -1, 0 or 1")
    out = bytearray((len(values) + 3) // 4)
    for index, value in enumerate(values):
        code = {-1: 0, 0: 1, 1: 2}[value]
        shift = (index % 4) * 2
        out[index // 4] |= code << shift
    return bytes(out)


def unpack_ternary(payload: bytes, count: int) -> list[int]:
    if count < 0:
        raise ValueError("count must be non-negative")
    result = []
    reverse = {0: -1, 1: 0, 2: 1}
    for index in range(count):
        shift = (index % 4) * 2
        code = (payload[index // 4] >> shift) & 0x3
        if code == 3:
            raise ValueError("invalid ternary code")
        result.append(reverse[code])
    return result
