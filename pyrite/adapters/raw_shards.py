from __future__ import annotations

from pathlib import Path

from .base import TensorMeta, WeightBlock


class RawShardAdapter:
    """Adapter for already-sharded local tensor blobs.

    Each block is a standalone binary file. This is intentionally format-neutral
    so GGUF and Safetensors adapters can map their tensors into the same runtime.
    """

    name = "raw-shards"

    def __init__(self, root: Path):
        self.root = root

    def blocks(self) -> list[WeightBlock]:
        result: list[WeightBlock] = []
        for index, path in enumerate(sorted(self.root.glob("*.bin"))):
            result.append(
                WeightBlock(
                    block_id=path.stem,
                    path=path,
                    offset=0,
                    size=path.stat().st_size,
                    dtype="opaque",
                    index=index,
                )
            )
        return result

    def tensors(self) -> list[TensorMeta]:
        return [
            TensorMeta(
                name=block.block_id,
                block_id=block.block_id,
                shape=(block.size,),
                dtype=block.dtype,
                ggml_type=None,
                offset=block.offset,
                size=block.size,
            )
            for block in self.blocks()
        ]

    def load(self, block: WeightBlock) -> memoryview:
        with block.path.open("rb") as fh:
            fh.seek(block.offset)
            return memoryview(fh.read(block.size))
