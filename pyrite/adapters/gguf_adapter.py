from __future__ import annotations

from pathlib import Path

from ..ggml_types import type_name
from .base import TensorMeta, WeightBlock
from .gguf import GGUFReader


class GGUFAdapter:
    """Tensor-aware GGUF adapter exposing bounded file-range blocks."""

    name = "gguf"

    def __init__(self, path: Path, reader: GGUFReader | None = None):
        self.path = Path(path)
        self.reader = reader or GGUFReader(self.path)
        if self.reader.path != self.path:
            raise ValueError("GGUF reader belongs to a different checkpoint")

    def blocks(self) -> list[WeightBlock]:
        return [
            WeightBlock(
                block_id=f"tensor:{tensor.name}",
                path=self.path,
                offset=tensor.offset,
                size=tensor.size,
                dtype=f"ggml:{type_name(tensor.ggml_type)}",
                kind="layer",
                index=index,
            )
            for index, tensor in enumerate(self.reader.tensor_index())
        ]

    def tensors(self) -> list[TensorMeta]:
        return [
            TensorMeta(
                name=tensor.name,
                block_id=f"tensor:{tensor.name}",
                shape=tensor.dims,
                dtype=type_name(tensor.ggml_type),
                ggml_type=tensor.ggml_type,
                offset=tensor.offset,
                size=tensor.size,
            )
            for tensor in self.reader.tensor_index()
        ]

    def load(self, block: WeightBlock) -> memoryview:
        if block.path != self.path:
            raise ValueError("block belongs to a different checkpoint")
        if block.offset < 0 or block.size < 0:
            raise ValueError("invalid GGUF block range")
        with self.path.open("rb") as fh:
            fh.seek(block.offset)
            payload = fh.read(block.size)
        if len(payload) != block.size:
            raise ValueError(f"truncated GGUF tensor block: {block.block_id}")
        return memoryview(payload)
