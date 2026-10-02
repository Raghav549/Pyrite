from __future__ import annotations

import math

import pytest

from pyrite.kernels.native import dequantize_rows, matvec, native_available
from pyrite.tensor_ops import decode_vector
from pyrite.tensor_ops import matvec as reference_matvec

from .k_quant import K_ENCODERS


@pytest.mark.parametrize("ggml_type", [12, 14], ids=["Q4_K", "Q6_K"])
def test_native_streaming_kernels_match_reference(ggml_type: int) -> None:
    if not native_available():
        pytest.skip("a C compiler is not available for the optional native kernels")

    rows, cols = 2, 512
    weights = [
        math.sin(index * 0.071) * 0.6 + math.cos(index * 0.019) * 0.2
        for index in range(rows * cols)
    ]
    vector = [math.cos(index * 0.13) * 0.4 for index in range(cols)]
    payload = K_ENCODERS[ggml_type](weights)
    decoded = decode_vector(ggml_type, payload, rows * cols)
    expected = reference_matvec(decoded, rows, cols, vector)

    got = matvec(ggml_type, payload, vector, rows, cols)
    assert got == pytest.approx(expected, rel=1e-11, abs=1e-11)

    block_bytes = 144 if ggml_type == 12 else 210
    row_bytes = block_bytes * (cols // 256)
    row = dequantize_rows(ggml_type, payload[:row_bytes], rows=1, cols=cols)
    assert row == pytest.approx(decoded[:cols], rel=1e-12, abs=1e-12)
