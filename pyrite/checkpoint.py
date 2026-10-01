from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .adapters.base import ModelAdapter
from .adapters.checkpoint import ChunkedFileAdapter, CheckpointInfo, detect_checkpoint
from .adapters.gguf_adapter import GGUFAdapter
from .adapters.safetensors import SafetensorsAdapter


@dataclass(frozen=True)
class CheckpointHandle:
    info: CheckpointInfo
    adapter: ModelAdapter


def open_checkpoint(path: str | Path, chunk_bytes: int = 64 * 1024 * 1024) -> CheckpointHandle:
    resolved = Path(path).expanduser().resolve()
    info = detect_checkpoint(resolved)
    if info.format == "unknown":
        raise ValueError(f"unsupported checkpoint format: {resolved}")

    if info.format == "gguf":
        adapter: ModelAdapter = GGUFAdapter(resolved)
    elif info.format == "safetensors":
        adapter = SafetensorsAdapter(resolved)
    else:
        adapter = ChunkedFileAdapter(resolved, chunk_bytes=chunk_bytes)

    return CheckpointHandle(info=info, adapter=adapter)
