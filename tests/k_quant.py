"""Naive K-quant encoders for test fixtures (test-only, not a quantizer product).

These implement the *exact* byte packing of the reference ``ggml`` quantizers
(``quantize_row_q*_K_ref`` in ``ggml-quants.c``) with deliberately naive
min/max scale selection: output bytes are always valid and decode identically
in Pyrite and in the independent ``gguf`` oracle, but quality is worse than a
real quantizer.  That is all fixtures need: the layout is what is under test,
and every encoder here is verified byte-for-byte against the oracle.
"""
from __future__ import annotations

import struct

_QK_K = 256
_F16_MAX = 65504.0


def _f16(value: float) -> float:
    """Round to fp16 the way the stored super-block scales are read back."""
    clamped = max(-_F16_MAX, min(_F16_MAX, value))
    return struct.unpack("<e", struct.pack("<e", clamped))[0]


def _f16_bytes(value: float) -> bytes:
    clamped = max(-_F16_MAX, min(_F16_MAX, value))
    return struct.pack("<e", clamped)


def _check(values: list[float]) -> list[list[float]]:
    if len(values) % _QK_K:
        raise ValueError(f"K-quant input must be a multiple of {_QK_K} values")
    return [values[i:i + _QK_K] for i in range(0, len(values), _QK_K)]


def _pack_k4_scales(pairs: list[tuple[int, int]]) -> bytes:
    """Pack 8 (scale, min) 6-bit pairs the way ``quantize_row_q4_K_ref`` does."""
    assert len(pairs) == 8
    scales = [0] * 12
    for j, (sc, mn) in enumerate(pairs):
        sc = max(0, min(63, sc))
        mn = max(0, min(63, mn))
        if j < 4:
            scales[j] = sc
            scales[j + 4] = mn
        else:
            scales[j + 4] = (sc & 0xF) | ((mn & 0xF) << 4)
            scales[j - 4] |= (sc >> 4) << 6
            scales[j] |= (mn >> 4) << 6
    return bytes(scales)


def _unpack_k4_scales(scales: bytes, j: int) -> tuple[int, int]:
    if j < 4:
        return scales[j] & 63, scales[j + 4] & 63
    return (
        (scales[j + 4] & 0x0F) | ((scales[j - 4] >> 6) << 4),
        (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4),
    )


def encode_q4_k(values: list[float]) -> bytes:
    out = bytearray()
    for block in _check(values):
        groups = [block[i * 32:(i + 1) * 32] for i in range(8)]
        scales_f: list[float] = []
        mins_f: list[float] = []
        for group in groups:
            lo, hi = min(group), max(group)
            if hi > lo:
                scales_f.append((hi - lo) / 15.0)
                mins_f.append(-lo)
            else:
                scales_f.append(0.0)
                mins_f.append(0.0)
        max_scale = max(scales_f)
        max_min = max(mins_f)
        inv_scale = 63.0 / max_scale if max_scale > 0 else 0.0
        inv_min = 63.0 / max_min if max_min > 0 else 0.0
        pairs = [
            (round(inv_scale * s), round(inv_min * m)) for s, m in zip(scales_f, mins_f, strict=True)
        ]
        packed = _pack_k4_scales(pairs)
        d = _f16(max_scale / 63.0) if max_scale > 0 else 0.0
        dmin = _f16(max_min / 63.0) if max_min > 0 else 0.0
        levels = [0] * _QK_K
        for j in range(8):
            sc, mn = _unpack_k4_scales(packed, j)
            step = d * sc
            if not step:
                continue
            offset = dmin * mn
            for i in range(32):
                levels[j * 32 + i] = max(0, min(15, round((block[j * 32 + i] + offset) / step)))
        out += _f16_bytes(max_scale / 63.0 if max_scale > 0 else 0.0)
        out += _f16_bytes(max_min / 63.0 if max_min > 0 else 0.0)
        out += packed
        for j in range(0, _QK_K, 64):
            for i in range(32):
                out.append(levels[j + i] | (levels[j + i + 32] << 4))
    return bytes(out)


def encode_q5_k(values: list[float]) -> bytes:
    out = bytearray()
    for block in _check(values):
        groups = [block[i * 32:(i + 1) * 32] for i in range(8)]
        scales_f: list[float] = []
        mins_f: list[float] = []
        for group in groups:
            lo, hi = min(group), max(group)
            if hi > lo:
                scales_f.append((hi - lo) / 31.0)
                mins_f.append(-lo)
            else:
                scales_f.append(0.0)
                mins_f.append(0.0)
        max_scale = max(scales_f)
        max_min = max(mins_f)
        inv_scale = 63.0 / max_scale if max_scale > 0 else 0.0
        inv_min = 63.0 / max_min if max_min > 0 else 0.0
        pairs = [
            (round(inv_scale * s), round(inv_min * m)) for s, m in zip(scales_f, mins_f, strict=True)
        ]
        packed = _pack_k4_scales(pairs)
        d = _f16(max_scale / 63.0) if max_scale > 0 else 0.0
        dmin = _f16(max_min / 63.0) if max_min > 0 else 0.0
        levels = [0] * _QK_K
        for j in range(8):
            sc, mn = _unpack_k4_scales(packed, j)
            step = d * sc
            if not step:
                continue
            offset = dmin * mn
            for i in range(32):
                levels[j * 32 + i] = max(0, min(31, round((block[j * 32 + i] + offset) / step)))
        out += _f16_bytes(max_scale / 63.0 if max_scale > 0 else 0.0)
        out += _f16_bytes(max_min / 63.0 if max_min > 0 else 0.0)
        out += packed
        qh = bytearray(32)
        ql = bytearray()
        m1, m2 = 1, 2
        for n in range(0, _QK_K, 64):
            for j in range(32):
                low = levels[n + j]
                if low > 15:
                    low -= 16
                    qh[j] |= m1
                high = levels[n + j + 32]
                if high > 15:
                    high -= 16
                    qh[j] |= m2
                ql.append(low | (high << 4))
            m1 <<= 2
            m2 <<= 2
        out += bytes(qh)
        out += bytes(ql)
    return bytes(out)


def encode_q6_k(values: list[float]) -> bytes:
    out = bytearray()
    for block in _check(values):
        peak = max((abs(v) for v in block), default=0.0)
        if peak <= 0.0:
            out += bytes(128 + 64 + 16 + 2)
            continue
        # Fixed mid-range scales keep every group representable; quality is
        # fixture-grade, validity is exact.
        d = _f16(peak / (64.0 * 31.0))
        step = d * 64.0
        levels = [max(0, min(63, round(v / step) + 32)) for v in block]
        ql = bytearray(128)
        qh = bytearray(64)
        for j in (0, 128):
            off_ql, off_qh = (j // 128) * 64, (j // 128) * 32
            for i in range(32):
                q1 = levels[j + i]
                q2 = levels[j + i + 32]
                q3 = levels[j + i + 64]
                q4 = levels[j + i + 96]
                ql[off_ql + i] = (q1 & 0xF) | ((q3 & 0xF) << 4)
                ql[off_ql + i + 32] = (q2 & 0xF) | ((q4 & 0xF) << 4)
                qh[off_qh + i] = (
                    (q1 >> 4) | ((q2 >> 4) << 2) | ((q3 >> 4) << 4) | ((q4 >> 4) << 6)
                )
        out += bytes(ql)
        out += bytes(qh)
        out += struct.pack("<16b", *([64] * 16))
        out += _f16_bytes(peak / (64.0 * 31.0))
    return bytes(out)


def encode_q2_k(values: list[float]) -> bytes:
    out = bytearray()
    for block in _check(values):
        groups = [block[i * 16:(i + 1) * 16] for i in range(16)]
        ranges = [max(g) - min(g) for g in groups]
        max_range = max(ranges)
        max_neg = max((-min(g) for g in groups if min(g) < 0.0), default=0.0)
        d = _f16(max_range / 45.0) if max_range > 0 else 0.0
        dmin = _f16(max_neg / 15.0) if max_neg > 0 else 0.0
        quants = bytearray(64)
        scales = bytearray()
        for g, group in enumerate(groups):
            lo = min(group)
            sc = 15 if ranges[g] > 0 else 0
            mn = round(-lo / dmin) if dmin > 0 and lo < 0.0 else 0
            mn = max(0, min(15, mn))
            scales.append((mn << 4) | sc)
            step = d * sc
            offset = dmin * mn
            half = g // 8
            shift = 2 * ((g % 8) // 2)
            base = half * 32 + (g % 2) * 16
            for i, value in enumerate(group):
                code = max(0, min(3, round((value + offset) / step))) if step else 0
                quants[base + i] |= code << shift
        out += bytes(scales)
        out += bytes(quants)
        out += _f16_bytes(max_range / 45.0 if max_range > 0 else 0.0)
        out += _f16_bytes(max_neg / 15.0 if max_neg > 0 else 0.0)
    return bytes(out)


def pack_q3_k_block(codes: list[int], levels: list[int], d: float) -> bytes:
    """Port of ``quantize_row_q3_K_ref``'s packing loops.

    ``codes`` are the 16 six-bit scale codes (0..63, the stored value is
    ``code = scale + 32``) and ``levels`` the 256 three-bit quant levels (0..7).
    """
    assert len(codes) == 16 and len(levels) == _QK_K

    scales = bytearray(12)
    for j, code in enumerate(codes):
        low = code & 0xF
        if j < 8:
            scales[j] = low
        else:
            scales[j - 8] |= low << 4
        scales[j % 4 + 8] |= (code >> 4) << (2 * (j // 4))

    hmask = bytearray(_QK_K // 8)
    quants = [level - 4 if level > 3 else level for level in levels]
    m, bit = 0, 1
    for j in range(_QK_K):
        if levels[j] > 3:
            hmask[m] |= bit
        m += 1
        if m == _QK_K // 8:
            m, bit = 0, bit << 1

    qs = bytearray(64)
    for j in range(0, _QK_K, 128):
        for i in range(32):
            qs[j // 4 + i] = (
                quants[j + i]
                | (quants[j + i + 32] << 2)
                | (quants[j + i + 64] << 4)
                | (quants[j + i + 96] << 6)
            )
    return bytes(hmask) + bytes(qs) + bytes(scales) + struct.pack("<e", d)


def encode_q3_k(values: list[float]) -> bytes:
    out = bytearray()
    for block in _check(values):
        groups = [block[i * 16:(i + 1) * 16] for i in range(16)]
        group_scales = []
        for group in groups:
            lo, hi = min(group), max(group)
            group_scales.append((hi - lo) / 7.0 if hi > lo else 0.0)
        peak = max(group_scales)
        d_all = _f16(peak / 31.0) if peak > 0 else 0.0
        codes = [
            max(0, min(63, 32 + round(s / d_all))) if d_all > 0 else 32 for s in group_scales
        ]
        levels: list[int] = []
        for group in groups:
            lo, hi = min(group), max(group)
            if hi > lo:
                levels.extend(max(0, min(7, round((v - lo) / (hi - lo) * 7.0))) for v in group)
            else:
                levels.extend([4] * 16)
        out += pack_q3_k_block(codes, levels, peak / 31.0 if peak > 0 else 0.0)
    return bytes(out)


#: ggml type id -> naive fixture encoder.
K_ENCODERS = {
    10: encode_q2_k,
    11: encode_q3_k,
    12: encode_q4_k,
    13: encode_q5_k,
    14: encode_q6_k,
}
