from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class QuantizationSpec:
    name: str
    bits: int
    symmetric: bool = True


SUPPORTED = {
    "fp16": QuantizationSpec("fp16", 16),
    "bf16": QuantizationSpec("bf16", 16),
    "int8": QuantizationSpec("int8", 8),
    "q8": QuantizationSpec("q8", 8),
    "q4": QuantizationSpec("q4", 4),
    "q3": QuantizationSpec("q3", 3),
    "q2": QuantizationSpec("q2", 2),
    "ternary": QuantizationSpec("ternary", 2),
}


def get_quantization(name: str) -> QuantizationSpec:
    key = name.strip().lower()
    if key not in SUPPORTED:
        raise ValueError(f"unsupported quantization: {name}")
    return SUPPORTED[key]


def estimate_weight_bytes(parameters: int, bits: int) -> int:
    if parameters < 0 or bits <= 0:
        raise ValueError("parameters must be non-negative and bits must be positive")
    return (parameters * bits + 7) // 8
