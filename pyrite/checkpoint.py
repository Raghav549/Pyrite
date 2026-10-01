from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .adapters.checkpoint import ChunkedFileAdapter, CheckpointInfo, detect_checkpoint


@dataclass(frozen=True)
class CheckpointHandle:
    info: CheckpointInfo
    adapter: ChunkedFileAdapter


def open_checkpoint(path: str | Path, chunk_bytes: int = 64 * 1024 * 1024) -> CheckpointHandle:
    resolved = Path(path).expanduser().resolve()
    info = detect_checkpoint(resolved)
    if info.format == "unknown":
        raise ValueError(f"unsupported checkpoint format: {resolved}")
    return CheckpointHandle(
        info=info,
        adapter=ChunkedFileAdapter(resolved, chunk_bytes=chunk_bytes),
    )
