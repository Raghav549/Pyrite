from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GGMLTypeSpec:
    type_id: int
    name: str
    block_size: int
    bytes_per_block: int


# GGML block layouts used by GGUF V2/V3 checkpoints.  The byte sizes mirror
# the upstream ggml block structs; keeping them centralized prevents parser
# and decoder drift.
_TYPES = {
    0: GGMLTypeSpec(0, "F32", 1, 4),
    1: GGMLTypeSpec(1, "F16", 1, 2),
    2: GGMLTypeSpec(2, "Q4_0", 32, 18),
    3: GGMLTypeSpec(3, "Q4_1", 32, 20),
    4: GGMLTypeSpec(4, "Q5_0", 32, 22),
    5: GGMLTypeSpec(5, "Q5_1", 32, 24),
    6: GGMLTypeSpec(6, "Q8_0", 32, 34),
    7: GGMLTypeSpec(7, "Q8_1", 32, 36),
    8: GGMLTypeSpec(8, "Q2_K", 256, 84),
    9: GGMLTypeSpec(9, "Q3_K", 256, 110),
    10: GGMLTypeSpec(10, "Q4_K", 256, 144),
    11: GGMLTypeSpec(11, "Q5_K", 256, 176),
    12: GGMLTypeSpec(12, "Q6_K", 256, 210),
    13: GGMLTypeSpec(13, "Q8_K", 256, 292),
    14: GGMLTypeSpec(14, "IQ2_XXS", 256, 66),
    15: GGMLTypeSpec(15, "IQ2_XS", 256, 74),
    16: GGMLTypeSpec(16, "IQ3_XXS", 256, 98),
    17: GGMLTypeSpec(17, "IQ1_S", 256, 50),
    18: GGMLTypeSpec(18, "IQ4_NL", 32, 18),
    19: GGMLTypeSpec(19, "IQ3_S", 256, 110),
    20: GGMLTypeSpec(20, "IQ2_S", 256, 82),
    21: GGMLTypeSpec(21, "IQ4_XS", 256, 136),
    22: GGMLTypeSpec(22, "I8", 1, 1),
    23: GGMLTypeSpec(23, "I16", 1, 2),
    24: GGMLTypeSpec(24, "I32", 1, 4),
    25: GGMLTypeSpec(25, "I64", 1, 8),
    26: GGMLTypeSpec(26, "IQ1_M", 256, 56),
    27: GGMLTypeSpec(27, "BF16", 1, 2),
}


def spec(type_id: int) -> GGMLTypeSpec:
    try:
        return _TYPES[type_id]
    except KeyError as exc:
        raise ValueError(f"unsupported GGML tensor type: {type_id}") from exc


def tensor_size(elements: int, type_id: int) -> int:
    t = spec(type_id)
    if elements < 0:
        raise ValueError("elements must be non-negative")
    if t.block_size != 1 and elements % t.block_size:
        raise ValueError(
            f"{t.name} requires element count divisible by {t.block_size}, got {elements}"
        )
    blocks = elements if t.block_size == 1 else elements // t.block_size
    return blocks * t.bytes_per_block
