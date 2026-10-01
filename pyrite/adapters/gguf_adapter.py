from __future__ import annotations

from pathlib import Path

from .base import WeightBlock
from .gguf import GGUFReader


class GGUFAdapter:
    """Tensor-aware GGUF adapter exposing bounded file-range blocks."""

    name = "gguf"

    def __init__(self, path: Path):
        self.path = Path(path)
        self.reader = GGUFReader(self.path)

    def blocks(self) -> list[WeightBlock]:
        result: list[WeightBlock] = []
        for index, tensor in enumerate(self.reader.tensor_index()):
            result.append(
                WeightBlock(
                    block_id=f"tensor:{tensor.name}",
                    path=self.path,
                    offset=tensor.offset,
                    size=tensor.size,
                    dtype=f"ggml:{tensor.ggml_type}",
                    kind="layer",
                    index=index,
                )
            )
        return result

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
