from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import struct
from typing import Sequence

from ..adapters.gguf import GGUFAdapter, GGUFTensor
from ..tensor_ops import bf16_vector, f16_vector, f32_vector, matvec, q4_0_vector, q8_0_vector


@dataclass(frozen=True)
class TensorPlacement:
    name: str
    shape: tuple[int, ...]
    ggml_type: int
    offset: int
    size: int


class GGUFReferenceBackend:
    """Minimal real-checkpoint tensor backend for supported dense tensor formats.

    This is an execution primitive, not a complete transformer-family implementation.
    """

    SUPPORTED = {0: f32_vector, 1: f16_vector, 8: q8_0_vector, 2: q4_0_vector, 29: bf16_vector}

    def __init__(self, path: str | Path, max_tensor_bytes: int | None = None):
        self.path = Path(path)
        self.adapter = GGUFAdapter(self.path)
        self.max_tensor_bytes = max_tensor_bytes

    def tensors(self) -> tuple[TensorPlacement, ...]:
        return tuple(
            TensorPlacement(
                t.name,
                t.dims,
                t.ggml_type,
                t.offset,
                t.size,
            )
            for t in self.adapter.reader.tensor_index()
        )

    def decode(self, tensor: GGUFTensor) -> list[float]:
        decoder = self.SUPPORTED.get(tensor.ggml_type)
        if decoder is None:
            raise ValueError(f"GGUF type {tensor.ggml_type} is not supported by reference backend")
        if self.max_tensor_bytes is not None and tensor.size > self.max_tensor_bytes:
            raise MemoryError("tensor exceeds configured reference backend budget")
        payload = self.adapter.load(
            next(
                block
                for block in self.adapter.blocks()
                if block.block_id == f"tensor:{tensor.name}"
            )
        )
        elements = math.prod(tensor.dims)
        return decoder(bytes(payload), elements)

    def tensor(self, name: str) -> list[float]:
        for tensor in self.adapter.reader.tensor_index():
            if tensor.name == name:
                return self.decode(tensor)
        raise KeyError(name)

    def matvec_tensor(self, name: str, vector: Sequence[float]) -> list[float]:
        tensor = next((t for t in self.adapter.reader.tensor_index() if t.name == name), None)
        if tensor is None:
            raise KeyError(name)
        values = self.decode(tensor)
        if len(tensor.dims) != 2:
            raise ValueError("matvec_tensor requires a rank-2 tensor")
        rows, cols = tensor.dims[0], tensor.dims[1]
        if rows * cols != len(values):
            raise ValueError("tensor payload shape does not match decoded element count")
        return matvec(values, rows, cols, vector)

    def close(self) -> None:
        return None
