from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
from typing import Iterator


@dataclass(frozen=True)
class BlockRef:
    block_id: str
    path: Path
    size_bytes: int
    sha256: str


class LocalBlockStore:
    """Content-addressed local block store for model shards or experts."""

    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def add_bytes(self, block_id: str, payload: bytes) -> BlockRef:
        path = self.root / f"{block_id}.bin"
        path.write_bytes(payload)
        digest = hashlib.sha256(payload).hexdigest()
        ref = BlockRef(block_id, path, len(payload), digest)
        (self.root / f"{block_id}.json").write_text(
            json.dumps(ref.__dict__, default=str),
            encoding="utf-8",
        )
        return ref

    def put_file(self, block_id: str, source: Path) -> BlockRef:
        target = self.root / f"{block_id}.bin"
        digest = hashlib.sha256()
        size = 0
        with source.open("rb") as src, target.open("wb") as dst:
            while chunk := src.read(1024 * 1024):
                digest.update(chunk)
                size += len(chunk)
                dst.write(chunk)

        ref = BlockRef(block_id, target, size, digest.hexdigest())
        (self.root / f"{block_id}.json").write_text(
            json.dumps(ref.__dict__, default=str),
            encoding="utf-8",
        )
        return ref

    def iter_blocks(self) -> Iterator[BlockRef]:
        for meta in sorted(self.root.glob("*.json")):
            data = json.loads(meta.read_text(encoding="utf-8"))
            yield BlockRef(
                str(data["block_id"]),
                Path(data["path"]),
                int(data["size_bytes"]),
                str(data["sha256"]),
            )
