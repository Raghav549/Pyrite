"""GGML tensor type registry.

Type ids, block sizes and per-block byte sizes mirror upstream ``ggml``
(``ggml/include/ggml.h`` for the enum and ``ggml/src/ggml-common.h`` for the
block structs).  Keeping the geometry centralized is what lets the GGUF reader
compute exact tensor extents for checkpoints written by real quantizers.

A wrong entry here is not a cosmetic bug: tensor sizes are used to validate
tensor offsets against the file length, so shifted ids make every real
``Q4_K_M``/``Q5_K_M`` checkpoint look corrupted (or worse, "valid" with the
wrong payload length).  The ids below therefore keep the two retired slots
(4 = Q4_2, 5 = Q4_3) that ``ggml`` still reserves, and the retired
``Q4_0_4_4``-family slots (31-33, 36-38).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GGMLTypeSpec:
    type_id: int
    name: str
    block_size: int
    bytes_per_block: int
    quantized: bool = False

    @property
    def bits_per_weight(self) -> float:
        return self.bytes_per_block * 8 / self.block_size


#: The full ``enum ggml_type`` layout as published by upstream ggml.
_TYPES: dict[int, GGMLTypeSpec] = {
    0: GGMLTypeSpec(0, "F32", 1, 4),
    1: GGMLTypeSpec(1, "F16", 1, 2),
    2: GGMLTypeSpec(2, "Q4_0", 32, 18, True),
    3: GGMLTypeSpec(3, "Q4_1", 32, 20, True),
    # 4 = Q4_2 and 5 = Q4_3 were removed but the ids stay reserved.
    6: GGMLTypeSpec(6, "Q5_0", 32, 22, True),
    7: GGMLTypeSpec(7, "Q5_1", 32, 24, True),
    8: GGMLTypeSpec(8, "Q8_0", 32, 34, True),
    9: GGMLTypeSpec(9, "Q8_1", 32, 36, True),
    10: GGMLTypeSpec(10, "Q2_K", 256, 84, True),
    11: GGMLTypeSpec(11, "Q3_K", 256, 110, True),
    12: GGMLTypeSpec(12, "Q4_K", 256, 144, True),
    13: GGMLTypeSpec(13, "Q5_K", 256, 176, True),
    14: GGMLTypeSpec(14, "Q6_K", 256, 210, True),
    15: GGMLTypeSpec(15, "Q8_K", 256, 292, True),
    16: GGMLTypeSpec(16, "IQ2_XXS", 256, 66, True),
    17: GGMLTypeSpec(17, "IQ2_XS", 256, 74, True),
    18: GGMLTypeSpec(18, "IQ3_XXS", 256, 98, True),
    19: GGMLTypeSpec(19, "IQ1_S", 256, 50, True),
    20: GGMLTypeSpec(20, "IQ4_NL", 32, 18, True),
    21: GGMLTypeSpec(21, "IQ3_S", 256, 110, True),
    22: GGMLTypeSpec(22, "IQ2_S", 256, 82, True),
    23: GGMLTypeSpec(23, "IQ4_XS", 256, 136, True),
    24: GGMLTypeSpec(24, "I8", 1, 1),
    25: GGMLTypeSpec(25, "I16", 1, 2),
    26: GGMLTypeSpec(26, "I32", 1, 4),
    27: GGMLTypeSpec(27, "I64", 1, 8),
    28: GGMLTypeSpec(28, "F64", 1, 8),
    29: GGMLTypeSpec(29, "IQ1_M", 256, 56, True),
    30: GGMLTypeSpec(30, "BF16", 1, 2),
    # 31-33 = retired Q4_0_4_4 / Q4_0_4_8 / Q4_0_8_8.
    34: GGMLTypeSpec(34, "TQ1_0", 256, 54, True),
    35: GGMLTypeSpec(35, "TQ2_0", 256, 66, True),
    # 36-38 = retired IQ4_NL_4_4 / IQ4_NL_4_8 / IQ4_NL_8_8.
    39: GGMLTypeSpec(39, "MXFP4", 32, 17, True),
    40: GGMLTypeSpec(40, "NVFP4", 64, 36, True),
    41: GGMLTypeSpec(41, "Q1_0", 128, 18, True),
    42: GGMLTypeSpec(42, "Q2_0", 64, 18, True),
}

#: Names for ids that upstream reserves but no longer produces.
RETIRED_TYPE_IDS: dict[int, str] = {
    4: "Q4_2 (retired)",
    5: "Q4_3 (retired)",
    31: "Q4_0_4_4 (retired)",
    32: "Q4_0_4_8 (retired)",
    33: "Q4_0_8_8 (retired)",
    36: "IQ4_NL_4_4 (retired)",
    37: "IQ4_NL_4_8 (retired)",
    38: "IQ4_NL_8_8 (retired)",
}


def spec(type_id: int) -> GGMLTypeSpec:
    """Return the geometry for ``type_id`` or raise ``ValueError``/``TypeError``."""
    if not isinstance(type_id, int):
        raise TypeError(f"GGML tensor type must be an integer, got {type_id!r}")
    found = _TYPES.get(type_id)
    if found is not None:
        return found
    if type_id in RETIRED_TYPE_IDS:
        raise ValueError(
            f"GGML tensor type {type_id} ({RETIRED_TYPE_IDS[type_id]}) "
            "is retired and cannot be read"
        )
    raise ValueError(f"unsupported GGML tensor type: {type_id}")


def type_name(type_id: int) -> str:
    """Best-effort display name, including retired ids."""
    try:
        return spec(type_id).name
    except ValueError:
        if isinstance(type_id, int) and type_id in RETIRED_TYPE_IDS:
            return RETIRED_TYPE_IDS[type_id]
        return f"UNKNOWN({type_id})"


def known_types() -> tuple[GGMLTypeSpec, ...]:
    return tuple(_TYPES[key] for key in sorted(_TYPES))


def row_size(elements: int, type_id: int) -> int:
    """Byte size of one GGML row containing ``elements`` values.

    Quantized GGML tensors are quantized row-by-row.  Checking only the total
    element count can accept an invalid shape whose first dimension is not a
    whole quantization block (the error can be hidden by multiplying by the
    number of rows).  GGUF sizes must therefore be calculated from dimension 0
    and then multiplied by the number of rows.
    """
    t = spec(type_id)
    if elements < 0:
        raise ValueError("elements must be non-negative")
    if elements == 0:
        return 0
    if t.block_size != 1 and elements % t.block_size:
        raise ValueError(
            f"{t.name} row requires element count divisible by {t.block_size}, got {elements}"
        )
    blocks = elements if t.block_size == 1 else elements // t.block_size
    return blocks * t.bytes_per_block


def tensor_size(elements: int, type_id: int) -> int:
    """Byte size of a linear stream of ``elements`` values of ``type_id``."""
    return row_size(elements, type_id)
