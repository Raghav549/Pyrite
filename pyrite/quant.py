from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class QuantizedBlock:
    """Affine int8 block: ``value = zero + code * scale``."""

    scale: float
    zero: float
    values: bytes


def dequantize_int8(block: QuantizedBlock) -> list[float]:
    """Invert :func:`quantize_int8`.

    The affine offset is applied back, so the original value range is restored
    (a plain multiply by ``scale`` would compress every block to ``[0, hi - lo]``).
    """
    return [block.zero + int(code) * block.scale for code in block.values]


def quantize_int8(values: list[float]) -> QuantizedBlock:
    if not values:
        return QuantizedBlock(1.0, 0.0, b"")
    lo, hi = min(values), max(values)
    if hi == lo:
        # Constant block: codes are meaningless, decoding must return the value.
        return QuantizedBlock(0.0, lo, bytes(len(values)))
    scale = (hi - lo) / 255.0
    raw = bytes(max(0, min(255, round((value - lo) / scale))) for value in values)
    return QuantizedBlock(scale, lo, raw)


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
    if len(payload) * 4 < count:
        raise ValueError("payload is too small for the requested ternary count")
    result = []
    reverse = {0: -1, 1: 0, 2: 1}
    for index in range(count):
        shift = (index % 4) * 2
        code = (payload[index // 4] >> shift) & 0x3
        if code == 3:
            raise ValueError("invalid ternary code")
        result.append(reverse[code])
    return result
