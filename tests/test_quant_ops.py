from pyrite.quant import dequantize_int8, quantize_int8


def test_int8_roundtrip_shape():
    block = quantize_int8([0.0, 1.0, 2.0])
    out = dequantize_int8(block)
    assert len(out) == 3
    assert abs(out[-1] - 2.0) < 0.02
