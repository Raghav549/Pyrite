from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .checkpoint import CheckpointHandle, open_checkpoint


@dataclass(frozen=True)
class TensorInfo:
    name: str
    offset: int
    size: int
    dtype: str
    shape: tuple[int, ...]


class LocalTensorLoader:
    """Tensor-level bounded loader; only one requested tensor is materialized."""

    def __init__(self, checkpoint: str | Path, chunk_bytes: int = 64 * 1024 * 1024):
        self.handle: CheckpointHandle = open_checkpoint(checkpoint, chunk_bytes=chunk_bytes)
        self._tensors = {tensor.name: tensor for tensor in self.handle.adapter.tensors()}
        self._blocks = {block.block_id: block for block in self.handle.adapter.blocks()}

    def tensors(self) -> tuple[TensorInfo, ...]:
        return tuple(
            TensorInfo(
                name=tensor.name,
                offset=tensor.offset,
                size=tensor.size,
                dtype=tensor.dtype,
                shape=tensor.shape,
            )
            for tensor in self.handle.adapter.tensors()
        )

    def name_for(self, block_id: str) -> str:
        return block_id.removeprefix("tensor:")

    def load(self, name: str) -> bytes:
        tensor = self._tensors.get(name)
        if tensor is None:
            raise KeyError(name)
        block = self._blocks.get(tensor.block_id)
        if block is None:
            raise KeyError(f"no block for tensor {name}")
        return bytes(self.handle.adapter.load(block))

    def close(self) -> None:
        return None

    def __enter__(self) -> LocalTensorLoader:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
