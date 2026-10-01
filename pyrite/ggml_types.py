from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GGMLTypeSpec:
    type_id: int
    name: str
    block_size: int
    bytes_per_block: int


# Common GGUF/ggml tensor types used by current model checkpoints.
_TYPES = {
    0: GGMLTypeSpec(0, "F32", 1, 4),
    1: GGMLTypeSpec(1, "F16", 1, 2),
    2: GGMLTypeSpec(2, "Q4_0", 32, 18),
    3: GGMLTypeSpec(3, "Q4_1", 32, 20),
    6: GGMLTypeSpec(6, "Q5_0", 32, 22),
    7: GGMLTypeSpec(7, "Q5_1", 32, 24),
    8: GGMLTypeSpec(8, "Q8_0", 32, 34),
    9: GGMLTypeSpec(9, "Q8_1", 32, 36),
    10: GGMLTypeSpec(10, "Q2_K", 256, 84),
    11: GGMLTypeSpec(11, "Q3_K", 256, 110),
    12: GGMLTypeSpec(12, "Q4_K", 256, 144),
    13: GGMLTypeSpec(13, "Q5_K", 256, 176),
    14: GGMLTypeSpec(14, "Q6_K", 256, 210),
    15: GGMLTypeSpec(15, "Q8_K", 256, 292),
    16: GGMLTypeSpec(16, "IQ2_XXS", 256, 66),
    17: GGMLTypeSpec(17, "IQ2_XS", 256, 74),
    18: GGMLTypeSpec(18, "IQ3_XXS", 256, 90),
    19: GGMLTypeSpec(19, "IQ1_S", 256, 50),
    20: GGMLTypeSpec(20, "IQ4_NL", 32, 18),
    21: GGMLTypeSpec(21, "IQ3_S", 256, 110),
    22: GGMLTypeSpec(22, "IQ2_S", 256, 82),
    23: GGMLTypeSpec(23, "IQ4_XS", 256, 136),
    24: GGMLTypeSpec(24, "I8", 1, 1),
    25: GGMLTypeSpec(25, "I16", 1, 2),
    26: GGMLTypeSpec(26, "I32", 1, 4),
    27: GGMLTypeSpec(27, "I64", 1, 8),
    28: GGMLTypeSpec(28, "IQ1_M", 256, 56),
    29: GGMLTypeSpec(29, "BF16", 1, 2),
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
