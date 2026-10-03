"""The optional C kernels must agree with the Python reference for every type.

``pyrite/kernels/native.c`` reimplements each GGML dequantizer in C for speed.
A C decoder that quietly disagrees with :mod:`pyrite.tensor_ops` would produce
wrong logits with no error, so every native type is cross-checked here against
the Python decoder byte-for-byte.

Payloads are deterministic pseudo-random bytes.  Types that store fp16 block
scales need those half-words patched to finite values first, otherwise the
random bytes produce NaN/Inf scales and the comparison proves nothing.
"""
from __future__ import annotations

import math

import pytest

from pyrite.ggml_types import spec
from pyrite.kernels.native import (
    dequantize_rows,
    matvec,
    native_available,
    native_type_ids,
    sorted_native_types,
    supports_native,
)
from pyrite.tensor_ops import DECODABLE_TYPES, decode_vector
from pyrite.tensor_ops import matvec as reference_matvec

from .k_quant import K_ENCODERS

# Byte offsets inside a block that hold fp16 scale/min words.  Patching them to
# a finite half keeps the block values representable.
SCALE_OFFSETS: dict[int, tuple[int, ...]] = {
    1: (0,),
    2: (0,),
    3: (0, 2),
    6: (0,),
    7: (0, 2),
    8: (0,),
    9: (0, 2),
    10: (80, 82),
    12: (0, 2),
    13: (0, 2),
    14: (208,),
    20: (0,),
    23: (0,),
    30: (0,),
}

# A benign finite fp16: 0x3C00 == 1.0
FINITE_HALF = b"\x00\x3c"


#: Linear types whose payload *is* the IEEE representation of the values, so
#: random bytes would be NaN/Inf most of the time.  These get finite values
#: packed in directly instead.
_LINEAR_FORMATS: dict[int, str] = {0: "<f", 28: "<d"}


def _finite_linear(ggml_type: int, elements: int, seed: int = 20240607) -> bytes:
    import struct

    fmt = _LINEAR_FORMATS[ggml_type]
    state = seed
    out = bytearray()
    for _ in range(elements):
        state = (1103515245 * state + 12345) & 0x7FFFFFFF
        out += struct.pack(fmt, (state / 0x7FFFFFFF) * 2.0 - 1.0)
    return bytes(out)


def _payload(ggml_type: int, rows: int, cols: int, seed: int = 20240607) -> bytes:
    block = spec(ggml_type)
    if ggml_type in _LINEAR_FORMATS:
        return _finite_linear(ggml_type, rows * cols, seed)
    blocks_per_row = cols // block.block_size
    raw = bytearray()
    state = seed
    for _ in range(rows * blocks_per_row * block.bytes_per_block):
        state = (1103515245 * state + 12345) & 0x7FFFFFFF
        raw.append((state >> 8) & 0xFF)
    for block_index in range(rows * blocks_per_row):
        base = block_index * block.bytes_per_block
        for offset in SCALE_OFFSETS.get(ggml_type, ()):
            raw[base + offset : base + offset + 2] = FINITE_HALF
    return bytes(raw)


def _vectors(cols: int) -> list[float]:
    return [math.cos(index * 0.13) * 0.4 for index in range(cols)]


def test_native_type_table_is_consistent_with_python() -> None:
    """Every native type must also be decodable in Python, or there is no oracle."""
    ids = native_type_ids()
    assert ids, "the native type table is empty"
    assert set(ids) <= set(DECODABLE_TYPES)
    names = sorted_native_types()
    assert len(names) == len(ids)
    assert all(isinstance(name, str) and name for name in names)
    assert len(set(names)) == len(names), "native type names must be unique"


@pytest.mark.parametrize("ggml_type", sorted(set(native_type_ids())), ids=str)
def test_native_decoders_match_the_python_reference(ggml_type: int) -> None:
    if not native_available():
        pytest.skip("a C compiler is not available for the optional native kernels")
    rows, cols = 2, 512
    block = spec(ggml_type)
    if cols % block.block_size:
        pytest.skip(f"{block.name} block size {block.block_size} does not divide {cols}")

    payload = _payload(ggml_type, rows, cols)
    decoded = decode_vector(ggml_type, payload, rows * cols)
    if not all(math.isfinite(value) for value in decoded):
        pytest.skip(f"{block.name} produced non-finite values from the probe payload")

    vector = _vectors(cols)
    expected = reference_matvec(decoded, rows, cols, vector)
    got = matvec(ggml_type, payload, vector, rows, cols)
    assert got == pytest.approx(expected, rel=1e-11, abs=1e-11)

    row_bytes = block.bytes_per_block * (cols // block.block_size)
    row = dequantize_rows(ggml_type, payload[:row_bytes], rows=1, cols=cols)
    assert row == pytest.approx(decoded[:cols], rel=1e-12, abs=1e-12)


@pytest.mark.parametrize("ggml_type", [12, 14], ids=["Q4_K", "Q6_K"])
def test_native_kernels_match_the_reference_quantizer(ggml_type: int) -> None:
    """Same check, but over a payload produced by the project's own encoders."""
    if not native_available():
        pytest.skip("a C compiler is not available for the optional native kernels")

    rows, cols = 2, 512
    weights = [
        math.sin(index * 0.071) * 0.6 + math.cos(index * 0.019) * 0.2
        for index in range(rows * cols)
    ]
    vector = _vectors(cols)
    payload = K_ENCODERS[ggml_type](weights)
    decoded = decode_vector(ggml_type, payload, rows * cols)
    expected = reference_matvec(decoded, rows, cols, vector)

    got = matvec(ggml_type, payload, vector, rows, cols)
    assert got == pytest.approx(expected, rel=1e-11, abs=1e-11)

    block_bytes = 144 if ggml_type == 12 else 210
    row_bytes = block_bytes * (cols // 256)
    row = dequantize_rows(ggml_type, payload[:row_bytes], rows=1, cols=cols)
    assert row == pytest.approx(decoded[:cols], rel=1e-12, abs=1e-12)


def test_supports_native_reports_the_table() -> None:
    for ggml_type in native_type_ids():
        assert supports_native(ggml_type) is True
    # IQ4_XS is decoded in Python only; it must not be claimed as native.
    assert supports_native(23) is False
