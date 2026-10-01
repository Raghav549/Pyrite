from pyrite.models.quant import estimate_weight_bytes, get_quantization


def test_low_bit_estimate():
    assert estimate_weight_bytes(8, 4) == 4
    assert get_quantization("q4").bits == 4
    assert get_quantization("ternary").bits == 2
