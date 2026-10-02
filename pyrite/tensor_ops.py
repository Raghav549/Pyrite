"""Tensor payload decoders.

Every quantized layout here is a line-for-line mirror of the reference
dequantizers in upstream ``ggml`` (``ggml/src/ggml-quants.c``).  The layouts are
not interchangeable: for example Q4_0 stores the *low* nibbles of a block in the
first half of the output and the *high* nibbles in the second half, so an
"interleaved nibble" reader silently produces a permuted tensor.  Bugs like that
still run; they just compute the wrong model.

Supported payload types are listed in :data:`DECODABLE_TYPES`.  Anything else
raises a clear error instead of returning approximated weights.
"""
from __future__ import annotations

import math
import struct
from collections.abc import Callable, Sequence

from .ggml_types import type_name


def _require(data: bytes, size: int, label: str) -> None:
    if len(data) < size:
        raise ValueError(f"insufficient {label} data: need {size} bytes, got {len(data)}")


def f32_vector(data: bytes, count: int) -> list[float]:
    if count < 0:
        raise ValueError("count must be non-negative")
    _require(data, count * 4, "F32")
    return list(struct.unpack("<" + "f" * count, data[: count * 4]))


def f64_vector(data: bytes, count: int) -> list[float]:
    if count < 0:
        raise ValueError("count must be non-negative")
    _require(data, count * 8, "F64")
    return list(struct.unpack("<" + "d" * count, data[: count * 8]))


def bf16_vector(data: bytes, count: int) -> list[float]:
    if count < 0:
        raise ValueError("count must be non-negative")
    _require(data, count * 2, "BF16")
    return [
        struct.unpack("<f", struct.pack("<I", word << 16))[0]
        for word in struct.unpack_from("<" + "H" * count, data, 0)
    ]


def f16_vector(data: bytes, count: int) -> list[float]:
    if count < 0:
        raise ValueError("count must be non-negative")
    _require(data, count * 2, "F16")
    return list(struct.unpack_from("<" + "e" * count, data, 0))


def i8_vector(data: bytes, count: int) -> list[float]:
    if count < 0:
        raise ValueError("count must be non-negative")
    _require(data, count, "I8")
    return [float(value) for value in struct.unpack_from("<" + "b" * count, data, 0)]


def i16_vector(data: bytes, count: int) -> list[float]:
    if count < 0:
        raise ValueError("count must be non-negative")
    _require(data, count * 2, "I16")
    return [float(value) for value in struct.unpack_from("<" + "h" * count, data, 0)]


def i32_vector(data: bytes, count: int) -> list[float]:
    if count < 0:
        raise ValueError("count must be non-negative")
    _require(data, count * 4, "I32")
    return [float(value) for value in struct.unpack_from("<" + "i" * count, data, 0)]


def i64_vector(data: bytes, count: int) -> list[float]:
    if count < 0:
        raise ValueError("count must be non-negative")
    _require(data, count * 8, "I64")
    return [float(value) for value in struct.unpack_from("<" + "q" * count, data, 0)]


def _half(data: bytes, offset: int) -> float:
    return struct.unpack_from("<e", data, offset)[0]


def q4_0_vector(data: bytes, count: int) -> list[float]:
    """Q4_0: ``y[j] = d * ((qs[j] & 0xF) - 8)`` and ``y[j + 16] = d * (qs[j] >> 4) - 8``."""
    if count % 32:
        raise ValueError("Q4_0 count must be divisible by 32")
    _require(data, (count // 32) * 18, "Q4_0")
    out: list[float] = []
    pos = 0
    for _ in range(count // 32):
        scale = _half(data, pos)
        pos += 2
        block = data[pos: pos + 16]
        pos += 16
        out.extend(scale * ((byte & 0x0F) - 8) for byte in block)
        out.extend(scale * ((byte >> 4) - 8) for byte in block)
    return out


def q4_1_vector(data: bytes, count: int) -> list[float]:
    """Q4_1: affine 4-bit blocks with ``y = d * q + m``."""
    if count % 32:
        raise ValueError("Q4_1 count must be divisible by 32")
    _require(data, (count // 32) * 20, "Q4_1")
    out: list[float] = []
    pos = 0
    for _ in range(count // 32):
        scale = _half(data, pos)
        minimum = _half(data, pos + 2)
        pos += 4
        block = data[pos: pos + 16]
        pos += 16
        out.extend(scale * (byte & 0x0F) + minimum for byte in block)
        out.extend(scale * (byte >> 4) + minimum for byte in block)
    return out


def q5_0_vector(data: bytes, count: int) -> list[float]:
    """Q5_0: 5-bit symmetric blocks (low nibbles + packed 5th bit)."""
    if count % 32:
        raise ValueError("Q5_0 count must be divisible by 32")
    _require(data, (count // 32) * 22, "Q5_0")
    out: list[float] = []
    pos = 0
    for _ in range(count // 32):
        scale = _half(data, pos)
        qh = struct.unpack_from("<I", data, pos + 2)[0]
        qs = data[pos + 6: pos + 22]
        pos += 22
        low = []
        high = []
        for j, byte in enumerate(qs):
            xh0 = ((qh >> j) << 4) & 0x10
            xh1 = (qh >> (j + 12)) & 0x10
            low.append(((byte & 0x0F) | xh0) - 16)
            high.append(((byte >> 4) | xh1) - 16)
        out.extend(scale * value for value in low)
        out.extend(scale * value for value in high)
    return out


def q5_1_vector(data: bytes, count: int) -> list[float]:
    """Q5_1: affine 5-bit blocks."""
    if count % 32:
        raise ValueError("Q5_1 count must be divisible by 32")
    _require(data, (count // 32) * 24, "Q5_1")
    out: list[float] = []
    pos = 0
    for _ in range(count // 32):
        scale = _half(data, pos)
        minimum = _half(data, pos + 2)
        qh = struct.unpack_from("<I", data, pos + 4)[0]
        qs = data[pos + 8: pos + 24]
        pos += 24
        low = []
        high = []
        for j, byte in enumerate(qs):
            xh0 = ((qh >> j) << 4) & 0x10
            xh1 = (qh >> (j + 12)) & 0x10
            low.append((byte & 0x0F) | xh0)
            high.append((byte >> 4) | xh1)
        out.extend(scale * value + minimum for value in low)
        out.extend(scale * value + minimum for value in high)
    return out


def q8_0_vector(data: bytes, count: int) -> list[float]:
    """Q8_0: ``y = d * q`` with int8 quants."""
    if count % 32:
        raise ValueError("Q8_0 count must be divisible by 32")
    _require(data, (count // 32) * 34, "Q8_0")
    out: list[float] = []
    pos = 0
    for _ in range(count // 32):
        scale = _half(data, pos)
        pos += 2
        out.extend(scale * value for value in struct.unpack_from("<32b", data, pos))
        pos += 32
    return out


def q8_1_vector(data: bytes, count: int) -> list[float]:
    """Q8_1: Q8_0 layout plus the stored ``d * sum(qs)`` helper half-word."""
    if count % 32:
        raise ValueError("Q8_1 count must be divisible by 32")
    _require(data, (count // 32) * 36, "Q8_1")
    out: list[float] = []
    pos = 0
    for _ in range(count // 32):
        scale = _half(data, pos)
        pos += 4  # d, s
        out.extend(scale * value for value in struct.unpack_from("<32b", data, pos))
        pos += 32
    return out


def _get_scale_min_k4(j: int, scales: bytes) -> tuple[int, int]:
    """Mirror of ``get_scale_min_k4`` from ``ggml-quants.c``."""
    if j < 4:
        return scales[j] & 63, scales[j + 4] & 63
    return (
        (scales[j + 4] & 0x0F) | ((scales[j - 4] >> 6) << 4),
        (scales[j + 4] >> 4) | ((scales[j - 0] >> 6) << 4),
    )


def q2_k_vector(data: bytes, count: int) -> list[float]:
    """Q2_K super-blocks (256 values, 84 bytes)."""
    if count % 256:
        raise ValueError("Q2_K count must be divisible by 256")
    _require(data, (count // 256) * 84, "Q2_K")
    out: list[float] = []
    pos = 0
    for _ in range(count // 256):
        # block_q2_K layout: scales[16], qs[64], d, dmin
        scales = data[pos: pos + 16]
        quants = data[pos + 16: pos + 80]
        scale = _half(data, pos + 80)
        minimum = _half(data, pos + 82)
        pos += 84
        is_ = 0
        q = 0
        for _n in range(0, 256, 128):
            shift = 0
            for _j in range(4):
                sc = scales[is_]
                is_ += 1
                dl = scale * (sc & 0x0F)
                ml = minimum * (sc >> 4)
                out.extend(dl * ((quants[q + l] >> shift) & 3) - ml for l in range(16))
                sc = scales[is_]
                is_ += 1
                dl = scale * (sc & 0x0F)
                ml = minimum * (sc >> 4)
                out.extend(dl * ((quants[q + 16 + l] >> shift) & 3) - ml for l in range(16))
                shift += 2
            q += 32
    return out


def q3_k_vector(data: bytes, count: int) -> list[float]:
    """Q3_K super-blocks (256 values, 110 bytes)."""
    if count % 256:
        raise ValueError("Q3_K count must be divisible by 256")
    _require(data, (count // 256) * 110, "Q3_K")
    out: list[float] = []
    pos = 0
    mask1 = 0x03030303
    mask2 = 0x0F0F0F0F
    for _ in range(count // 256):
        d_all = _half(data, pos + 108)
        # block_q3_K: hmask[32] | qs[64] | scales[12] | d(fp16)
        hmask = data[pos: pos + 32]
        quants = data[pos + 32: pos + 96]
        raw = data[pos + 96: pos + 108]
        pos += 110

        aux = list(struct.unpack("<4I", bytes(raw) + b"\0\0\0\0"))
        tmp = aux[2]
        aux[2] = ((aux[0] >> 4) & mask2) | (((tmp >> 4) & mask1) << 4)
        aux[3] = ((aux[1] >> 4) & mask2) | (((tmp >> 6) & mask1) << 4)
        aux[0] = (aux[0] & mask2) | (((tmp >> 0) & mask1) << 4)
        aux[1] = (aux[1] & mask2) | (((tmp >> 2) & mask1) << 4)
        scales = struct.unpack("<16b", struct.pack("<4I", *aux))

        is_ = 0
        q = 0
        m = 1
        for _n in range(0, 256, 128):
            shift = 0
            for _j in range(4):
                dl = d_all * (scales[is_] - 32)
                is_ += 1
                out.extend(
                    dl * (((quants[q + l] >> shift) & 3) - (0 if hmask[l] & m else 4))
                    for l in range(16)
                )
                dl = d_all * (scales[is_] - 32)
                is_ += 1
                out.extend(
                    dl * (((quants[q + 16 + l] >> shift) & 3) - (0 if hmask[16 + l] & m else 4))
                    for l in range(16)
                )
                shift += 2
                m <<= 1
                m &= 0xFF
            q += 32
    return out


def q4_k_vector(data: bytes, count: int) -> list[float]:
    """Q4_K super-blocks (256 values, 144 bytes)."""
    if count % 256:
        raise ValueError("Q4_K count must be divisible by 256")
    _require(data, (count // 256) * 144, "Q4_K")
    out: list[float] = []
    pos = 0
    for _ in range(count // 256):
        d = _half(data, pos)
        dmin = _half(data, pos + 2)
        scales = data[pos + 4: pos + 16]
        quants = data[pos + 16: pos + 144]
        pos += 144
        is_ = 0
        q = 0
        for _j in range(0, 256, 64):
            sc, m = _get_scale_min_k4(is_ + 0, scales)
            d1, m1 = d * sc, dmin * m
            sc, m = _get_scale_min_k4(is_ + 1, scales)
            d2, m2 = d * sc, dmin * m
            out.extend(d1 * (quants[q + l] & 0x0F) - m1 for l in range(32))
            out.extend(d2 * (quants[q + l] >> 4) - m2 for l in range(32))
            q += 32
            is_ += 2
    return out


def q5_k_vector(data: bytes, count: int) -> list[float]:
    """Q5_K super-blocks (256 values, 176 bytes)."""
    if count % 256:
        raise ValueError("Q5_K count must be divisible by 256")
    _require(data, (count // 256) * 176, "Q5_K")
    out: list[float] = []
    pos = 0
    for _ in range(count // 256):
        d = _half(data, pos)
        dmin = _half(data, pos + 2)
        scales = data[pos + 4: pos + 16]
        qh = data[pos + 16: pos + 48]
        ql = data[pos + 48: pos + 176]
        pos += 176
        is_ = 0
        q = 0
        u1, u2 = 1, 2
        for _j in range(0, 256, 64):
            sc, m = _get_scale_min_k4(is_ + 0, scales)
            d1, m1 = d * sc, dmin * m
            sc, m = _get_scale_min_k4(is_ + 1, scales)
            d2, m2 = d * sc, dmin * m
            out.extend(
                d1 * ((ql[q + l] & 0x0F) + (16 if qh[l] & u1 else 0)) - m1 for l in range(32)
            )
            out.extend(
                d2 * ((ql[q + l] >> 4) + (16 if qh[l] & u2 else 0)) - m2 for l in range(32)
            )
            q += 32
            is_ += 2
            u1 <<= 2
            u2 <<= 2
    return out


def q6_k_vector(data: bytes, count: int) -> list[float]:
    """Q6_K super-blocks (256 values, 210 bytes)."""
    if count % 256:
        raise ValueError("Q6_K count must be divisible by 256")
    _require(data, (count // 256) * 210, "Q6_K")
    out: list[float] = []
    pos = 0
    for _ in range(count // 256):
        ql = data[pos: pos + 128]
        qh = data[pos + 128: pos + 192]
        scales = struct.unpack_from("<16b", data, pos + 192)
        d = _half(data, pos + 208)
        pos += 210

        # The reference writes each 128-value half through an explicit index map
        # (l, l+32, l+64, l+96); the values must land at those offsets, not in
        # append order.
        for half in (0, 1):
            ql_off = half * 64
            qh_off = half * 32
            sc_off = half * 8
            block = [0.0] * 128
            for l in range(32):
                is_ = l // 16
                q1 = ((ql[ql_off + l] & 0x0F) | (((qh[qh_off + l] >> 0) & 3) << 4)) - 32
                q2 = ((ql[ql_off + l + 32] & 0x0F) | (((qh[qh_off + l] >> 2) & 3) << 4)) - 32
                q3 = ((ql[ql_off + l] >> 4) | (((qh[qh_off + l] >> 4) & 3) << 4)) - 32
                q4 = ((ql[ql_off + l + 32] >> 4) | (((qh[qh_off + l] >> 6) & 3) << 4)) - 32
                block[l + 0] = d * scales[sc_off + is_ + 0] * q1
                block[l + 32] = d * scales[sc_off + is_ + 2] * q2
                block[l + 64] = d * scales[sc_off + is_ + 4] * q3
                block[l + 96] = d * scales[sc_off + is_ + 6] * q4
            out.extend(block)
    return out


#: ggml type id -> decoder.  Types absent from this map are reported as
#: unsupported instead of being decoded with a guessed layout.
DECODABLE_TYPES: dict[int, Callable[[bytes, int], list[float]]] = {
    0: f32_vector,
    1: f16_vector,
    2: q4_0_vector,
    3: q4_1_vector,
    6: q5_0_vector,
    7: q5_1_vector,
    8: q8_0_vector,
    9: q8_1_vector,
    10: q2_k_vector,
    11: q3_k_vector,
    12: q4_k_vector,
    13: q5_k_vector,
    14: q6_k_vector,
    24: i8_vector,
    25: i16_vector,
    26: i32_vector,
    27: i64_vector,
    28: f64_vector,
    30: bf16_vector,
}


def supports_type(type_id: int) -> bool:
    return type_id in DECODABLE_TYPES


def decode_vector(type_id: int, data: bytes, count: int) -> list[float]:
    """Decode ``count`` consecutive values of ``type_id`` from ``data``."""
    decoder = DECODABLE_TYPES.get(type_id)
    if decoder is None:
        raise ValueError(
            f"GGML tensor type {type_id} ({type_name(type_id)}) has no reference "
            "decoder in Pyrite; refusing to approximate weights"
        )
    return decoder(data, count)


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


def silu(x: Sequence[float]) -> list[float]:
    return [value / (1.0 + math.exp(-value)) if value > -60.0 else 0.0 for value in x]


def softmax(values: Sequence[float]) -> list[float]:
    if not values:
        return []
    peak = max(values)
    exps = [math.exp(value - peak) for value in values]
    total = sum(exps)
    if total <= 0.0:
        return [1.0 / len(values)] * len(values)
    return [value / total for value in exps]
