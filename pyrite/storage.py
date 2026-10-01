from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

#: Block ids become file names, so they must not be able to escape the store.
_SAFE_BLOCK_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:@+-]{0,127}$")


@dataclass(frozen=True)
class BlockRef:
    block_id: str
    path: Path
    size_bytes: int
    sha256: str


def validate_block_id(block_id: str) -> str:
    if not isinstance(block_id, str) or not _SAFE_BLOCK_ID.match(block_id):
        raise ValueError(
            f"unsafe block id {block_id!r}: use letters, digits and . _ : @ + - only"
        )
    if ".." in block_id:
        raise ValueError(f"unsafe block id {block_id!r}: '..' is not allowed")
    return block_id


class LocalBlockStore:
    """Content-addressed local block store for model shards or experts."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _paths(self, block_id: str) -> tuple[Path, Path]:
        validate_block_id(block_id)
        return self.root / f"{block_id}.bin", self.root / f"{block_id}.json"

    def _write_meta(self, ref: BlockRef) -> None:
        _, meta_path = self._paths(ref.block_id)
        payload = json.dumps(
            {
                "block_id": ref.block_id,
                "path": str(ref.path),
                "size_bytes": ref.size_bytes,
                "sha256": ref.sha256,
            }
        )
        tmp = meta_path.with_suffix(".json.tmp")
        tmp.write_text(payload, encoding="utf-8")
        os.replace(tmp, meta_path)

    def add_bytes(self, block_id: str, payload: bytes) -> BlockRef:
        path, _ = self._paths(block_id)
        tmp = path.with_suffix(".bin.tmp")
        tmp.write_bytes(payload)
        os.replace(tmp, path)
        ref = BlockRef(block_id, path, len(payload), hashlib.sha256(payload).hexdigest())
        self._write_meta(ref)
        return ref

    def put_file(self, block_id: str, source: Path) -> BlockRef:
        target, _ = self._paths(block_id)
        digest = hashlib.sha256()
        size = 0
        tmp = target.with_suffix(".bin.tmp")
        with Path(source).open("rb") as src, tmp.open("wb") as dst:
            while chunk := src.read(1024 * 1024):
                digest.update(chunk)
                size += len(chunk)
                dst.write(chunk)
        os.replace(tmp, target)

        ref = BlockRef(block_id, target, size, digest.hexdigest())
        self._write_meta(ref)
        return ref

    def iter_blocks(self) -> Iterator[BlockRef]:
        for meta in sorted(self.root.glob("*.json")):
            try:
                data = json.loads(meta.read_text(encoding="utf-8"))
                block_id = validate_block_id(str(data["block_id"]))
                path = Path(data["path"])
                size_bytes = int(data["size_bytes"])
                sha256 = str(data["sha256"])
            except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
                continue
            if not self._inside_root(path) or not path.is_file():
                # Metadata is local state, not a trusted pointer: a path that
                # escaped the store is ignored rather than read.
                continue
            yield BlockRef(block_id, path, size_bytes, sha256)

    def _inside_root(self, path: Path) -> bool:
        try:
            return path.resolve().is_relative_to(self.root.resolve())
        except OSError:
            return False

    def get(self, block_id: str) -> BlockRef | None:
        for ref in self.iter_blocks():
            if ref.block_id == block_id:
                return ref
        return None

    def verify(self, ref: BlockRef) -> bool:
        digest = hashlib.sha256()
        try:
            with ref.path.open("rb") as fh:
                while chunk := fh.read(1024 * 1024):
                    digest.update(chunk)
        except OSError:
            return False
        return digest.hexdigest() == ref.sha256
