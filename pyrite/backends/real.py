from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from ..adapters.gguf import GGUFTensor
from ..adapters.gguf_adapter import GGUFAdapter
from ..ggml_types import type_name
from ..tensor_ops import DECODABLE_TYPES, decode_vector, matvec


@dataclass(frozen=True)
class TensorPlacement:
    name: str
    shape: tuple[int, ...]
    ggml_type: int
    dtype: str
    offset: int
    size: int


class GGUFReferenceBackend:
    """Real-checkpoint tensor backend for the GGML types Pyrite can decode.

    This is an execution primitive (decode a tensor, run a matvec), not a
    transformer implementation; :class:`pyrite.executor.Qwen3MoEExecutor` is the
    model-level consumer.  Unsupported quantization types raise instead of being
    approximated.
    """

    def __init__(
        self,
        path: str | Path,
        max_tensor_bytes: int | None = None,
        allowed_types: set[int] | None = None,
    ):
        self.path = Path(path)
        self.adapter = GGUFAdapter(self.path)
        self.max_tensor_bytes = max_tensor_bytes
        self.allowed_types = allowed_types if allowed_types is not None else set(DECODABLE_TYPES)

    @property
    def reader(self):
        return self.adapter.reader

    @staticmethod
    def supported_types() -> dict[int, str]:
        return {type_id: type_name(type_id) for type_id in sorted(DECODABLE_TYPES)}

    def tensors(self) -> tuple[TensorPlacement, ...]:
        return tuple(
            TensorPlacement(
                name=meta.name,
                shape=meta.shape,
                ggml_type=meta.ggml_type if meta.ggml_type is not None else -1,
                dtype=meta.dtype,
                offset=meta.offset,
                size=meta.size,
            )
            for meta in self.adapter.tensors()
        )

    def _block_id(self, name: str) -> str:
        return f"tensor:{name}"

    def load(self, name: str) -> bytes:
        tensor = self.adapter.reader.tensor(name)
        if self.max_tensor_bytes is not None and tensor.size > self.max_tensor_bytes:
            raise MemoryError(f"tensor {name!r} exceeds configured reference backend budget")
        block = next(
            block for block in self.adapter.blocks() if block.block_id == self._block_id(name)
        )
        return bytes(self.adapter.load(block))

    def decode(self, tensor: GGUFTensor) -> list[float]:
        if tensor.ggml_type not in self.allowed_types:
            raise ValueError(
                f"GGML type {tensor.ggml_type} ({type_name(tensor.ggml_type)}) is not supported "
                "by the reference backend"
            )
        if self.max_tensor_bytes is not None and tensor.size > self.max_tensor_bytes:
            raise MemoryError("tensor exceeds configured reference backend budget")
        payload = self.load(tensor.name)
        return decode_vector(tensor.ggml_type, payload, tensor.element_count)

    def tensor(self, name: str) -> list[float]:
        return self.decode(self.adapter.reader.tensor(name))

    def matvec_tensor(self, name: str, vector: Sequence[float]) -> list[float]:
        tensor = self.adapter.reader.tensor(name)
        values = self.decode(tensor)
        if len(tensor.dims) != 2:
            raise ValueError("matvec_tensor requires a rank-2 tensor")
        cols, rows = tensor.dims
        if rows * cols != len(values):
            raise ValueError("tensor payload shape does not match decoded element count")
        return matvec(values, rows, cols, vector)

    def close(self) -> None:
        return None

    def __enter__(self) -> GGUFReferenceBackend:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
