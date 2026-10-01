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

    def __init__(self, checkpoint: str | Path):
        self.handle: CheckpointHandle = open_checkpoint(checkpoint)
        self._blocks = {b.block_id: b for b in self.handle.adapter.blocks()}

    def tensors(self) -> tuple[TensorInfo, ...]:
        result = []
        for block_id, block in self._blocks.items():
            name = block_id.removeprefix("tensor:")
            shape: tuple[int, ...] = ()
            dtype = block.dtype
            if self.handle.info.format == "gguf":
                tensor = next(t for t in self.handle.adapter.reader.tensor_index() if f"tensor:{t.name}" == block_id)
                shape = tensor.dims
                dtype = f"ggml:{tensor.ggml_type}"
            elif self.handle.info.format == "safetensors":
                tensor = next(t for t in self.handle.adapter._tensors if f"tensor:{t.name}" == block_id)
                shape = tensor.shape
                dtype = tensor.dtype
            result.append(TensorInfo(name, block.offset, block.size, dtype, shape))
        return tuple(result)

    def load(self, name: str) -> bytes:
        key = f"tensor:{name}"
        block = self._blocks.get(key)
        if block is None:
            raise KeyError(name)
        return bytes(self.handle.adapter.load(block))

    def close(self) -> None:
        return None

    def __enter__(self) -> "LocalTensorLoader":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
