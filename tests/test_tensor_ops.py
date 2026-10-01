import struct

from pyrite.tensor_ops import (
    bf16_vector,
    f16_vector,
    f32_vector,
    matvec,
    q4_0_vector,
    q8_0_vector,
    rms_norm,
)


def test_float_decoders():
    assert f32_vector(struct.pack("<f", 1.5), 1) == [1.5]
    assert f16_vector(struct.pack("<e", 2.5), 1) == [2.5]
    assert bf16_vector(struct.pack("<H", 0x3f80), 1) == [1.0]


def test_q8_decoder():
    payload = struct.pack("<e", 1.0) + bytes([1] * 32)
    assert q8_0_vector(payload, 32) == [1.0] * 32


def test_q4_decoder():
    payload = struct.pack("<e", 1.0) + bytes([0x88] * 16)
    assert q4_0_vector(payload, 32) == [0.0] * 32


def test_matvec_and_norm():
    assert matvec([1, 2, 3, 4], 2, 2, [1, 1]) == [3, 7]
    assert len(rms_norm([1, 2], [1, 1])) == 2
