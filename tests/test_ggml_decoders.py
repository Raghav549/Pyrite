"""GGML quantization decoders, pinned to the reference implementations.

Two kinds of checks live here:

* a round-trip through a faithful port of llama.cpp's *reference quantizer*
  (``quantize_row_q*_ref`` in ``ggml-quants.c``), which pins the exact byte
  layout, and
* a parity check against the independent ``gguf`` Python package when it is
  installed.

``gguf`` is not a runtime dependency, so those checks skip when it is absent.
"""
from __future__ import annotations

import math
import random
import struct

import pytest

from pyrite.ggml_types import spec, tensor_size
from pyrite.tensor_ops import DECODABLE_TYPES, decode_vector


def _quants():
    """The optional decoder oracle, if it is installed."""
    module = pytest.importorskip("gguf.quants", reason="optional decoder oracle")
    return module


def _as_uint8(payload: bytes):
    """The oracle works on numpy arrays of bytes."""
    np = pytest.importorskip("numpy", reason="the gguf oracle needs numpy")
    return np.frombuffer(payload, dtype=np.uint8)


def _force_finite_fp16_words(raw: bytearray) -> None:
    """Replace NaN/Inf fp16 patterns so scale fields stay comparable.

    Real quantizers never emit these for ``d``/``dmin``; random bytes are used
    here to stress the nibble layouts, so the scale half-words are sanitized.
    """
    for offset in range(0, len(raw) - 1, 2):
        value = struct.unpack_from("<e", raw, offset)[0]
        if not math.isfinite(value):
            struct.pack_into("<e", raw, offset, 0.5)

#: (pyrite type id, name in the ``gguf`` package) pairs known to agree.
LIBRARY_PARITY = [
    (0, "F32"),
    (1, "F16"),
    (2, "Q4_0"),
    (3, "Q4_1"),
    (6, "Q5_0"),
    (7, "Q5_1"),
    (8, "Q8_0"),
    (10, "Q2_K"),
    (11, "Q3_K"),
    (12, "Q4_K"),
    (13, "Q5_K"),
    (14, "Q6_K"),
    (30, "BF16"),
]


@pytest.mark.parametrize("ggml_type, qtype_name", LIBRARY_PARITY)
def test_decoders_match_the_reference_library(ggml_type: int, qtype_name: str) -> None:
    quants = _quants()
    qtype = getattr(quants.GGMLQuantizationType, qtype_name)
    geometry = spec(ggml_type)
    assert geometry.name == qtype_name
    rng = random.Random(ggml_type * 7919)  # noqa: S311 - deterministic test data
    blocks = 3
    raw = bytearray(rng.randrange(256) for _ in range(geometry.bytes_per_block * blocks))
    _force_finite_fp16_words(raw)
    payload = bytes(raw)
    count = geometry.block_size * blocks

    expected = list(quants.dequantize(_as_uint8(payload), qtype))
    got = decode_vector(ggml_type, payload, count)
    assert len(got) == count
    worst = max(abs(a - b) for a, b in zip(got, expected, strict=True))
    assert worst <= 1e-6, f"{geometry.name}: max abs diff {worst}"


# --------------------------------------------------------------------- Q3_K
_QK_K = 256


def _pack_q3_k_reference(codes: list[int], levels: list[int], d: float) -> bytes:
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
        for l in range(32):
            qs[j // 4 + l] = (
                quants[j + l]
                | (quants[j + l + 32] << 2)
                | (quants[j + l + 64] << 4)
                | (quants[j + l + 96] << 6)
            )
    return bytes(hmask) + bytes(qs) + bytes(scales) + struct.pack("<e", d)


def test_q3_k_round_trips_through_the_reference_packing() -> None:
    codes = [32 + value for value in (-16, -12, -8, -4, 0, 4, 8, 12, 15, -15, 3, -3, 7, -7, 11, -11)]
    levels = [(index * 3) % 8 for index in range(_QK_K)]
    d = 0.03125
    payload = _pack_q3_k_reference(codes, levels, d)

    got = decode_vector(11, payload, _QK_K)
    expected = []
    for group, code in enumerate(codes):
        scale = d * (code - 32)
        for item in range(16):
            level = levels[group * 16 + item]
            expected.append(scale * (level - 4))

    assert any(level > 3 for level in levels) and any(level < 4 for level in levels)
    assert any(code >> 4 for code in codes) and any(code & 0xF for code in codes)
    assert got == pytest.approx(expected, abs=1e-6)


# ------------------------------------------------------------ hand-built blocks
def test_q2_k_layout_matches_the_struct() -> None:
    # block_q2_K: scales[16] | qs[64] | d(fp16) | dmin(fp16)
    scales = bytes([0x21] + [0] * 15)  # low nibble = scale 1, high nibble = min 2
    qs = bytes([0xE4] * 64)
    d, dmin = 0.5, 0.25
    payload = scales + qs + struct.pack("<ee", d, dmin)
    values = decode_vector(10, payload, 256)
    assert len(values) == 256
    # value = d * scale * q - dmin * min, with q = (qs >> shift) & 3.
    assert values[0] == pytest.approx(d * 1 * ((qs[0] >> 0) & 3) - dmin * 2, abs=1e-6)
    # The next 16-element group uses the following scale byte (scales[1] = 0).
    assert values[16] == pytest.approx(d * 0 * ((qs[16] >> 0) & 3) - dmin * 0, abs=1e-6)


def test_every_decodable_type_has_a_decoder() -> None:
    for ggml_type in DECODABLE_TYPES:
        geometry = spec(ggml_type)
        payload = bytes(geometry.bytes_per_block)
        values = decode_vector(ggml_type, payload, geometry.block_size)
        assert len(values) == geometry.block_size
        assert all(isinstance(value, float) for value in values)


def test_undecodable_types_are_named_clearly() -> None:
    from pyrite.ggml_types import type_name
    from pyrite.tensor_ops import supports_type

    assert not supports_type(16)  # IQ2_XXS
    assert not supports_type(35)  # TQ2_0
    assert not supports_type(39)  # MXFP4
    assert type_name(16) == "IQ2_XXS"
    assert type_name(35) == "TQ2_0"


def test_truncated_payloads_are_rejected() -> None:
    q4_0_size = tensor_size(32, 2)
    with pytest.raises(ValueError):
        decode_vector(2, bytes(q4_0_size - 1), 32)
    with pytest.raises(ValueError):
        decode_vector(2, bytes(q4_0_size), 33)  # count not block aligned
    with pytest.raises(ValueError):
        decode_vector(999, bytes(16), 4)


def test_bf16_decoder_matches_the_definition() -> None:
    # bfloat16 is the top half of a float32 bit pattern.
    words = [0x3F80, 0xBF80, 0x4048, 0x0000, 0x7F7F]
    payload = struct.pack("<5H", *words)
    expected = [struct.unpack("<f", struct.pack("<I", word << 16))[0] for word in words]
    assert decode_vector(30, payload, 5) == expected


def test_q8_1_layout_is_d_plus_quantized_values() -> None:
    # block_q8_1: { half d; half s; int8 qs[32] }
    payload = struct.pack("<e", 0.5) + struct.pack("<e", 0.25) + bytes([2] * 32)
    values = decode_vector(9, payload, 32)
    assert values == pytest.approx([1.0] * 32)


def test_float_decoders_are_exact() -> None:
    values = [0.0, -1.5, 3.25, 1e-5]
    as_f32 = list(struct.unpack("<4f", struct.pack("<4f", *values)))
    assert decode_vector(0, struct.pack("<4f", *values), 4) == as_f32
    rounded = list(struct.unpack("<4e", struct.pack("<4e", *values)))
    assert decode_vector(1, struct.pack("<4e", *values), 4) == pytest.approx(rounded, rel=1e-6)
    assert decode_vector(28, struct.pack("<4d", *values), 4) == values


def test_q4_0_uses_low_then_high_nibbles() -> None:
    # qs[i] holds the low nibble for element i and the high nibble for i + 16.
    payload = struct.pack("<e", 1.0) + bytes([0x80] * 16)  # low=0, high=8
    values = decode_vector(2, payload, 32)
    assert values == pytest.approx([-8.0] * 16 + [0.0] * 16)


def test_randomized_blocks_never_raise_for_decodable_types() -> None:
    rng = __import__("random").Random(1234)
    for ggml_type in sorted(DECODABLE_TYPES):
        geometry = spec(ggml_type)
        for _ in range(2):
            payload = bytes(rng.randrange(256) for _ in range(geometry.bytes_per_block))
            values = decode_vector(ggml_type, payload, geometry.block_size)
            assert all(math.isfinite(value) for value in values)
