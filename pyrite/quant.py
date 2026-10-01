from __future__ import annotations

from dataclasses import dataclass
import struct


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
        return QuantizedBlock(1.0, 0, bytes([0] * len(values)))
    scale = (hi - lo) / 255.0
    raw = [max(0, min(255, round((x - lo) / scale))) for x in values]
    return QuantizedBlock(scale, 0, bytes(raw))
