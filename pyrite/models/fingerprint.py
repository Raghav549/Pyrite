from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
import json


@dataclass(frozen=True)
class ModelFingerprint:
    architecture: str
    tokenizer: str
    checkpoint_sha256: str

    @property
    def cache_namespace(self) -> str:
        raw = f"{self.architecture}|{self.tokenizer}|{self.checkpoint_sha256}".encode()
        return hashlib.sha256(raw).hexdigest()


def fingerprint_file(path: Path, chunk_bytes: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        while chunk := fh.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def save_fingerprint(fp: ModelFingerprint, path: Path) -> None:
    Path(path).write_text(json.dumps(fp.__dict__, indent=2), encoding="utf-8")


def load_fingerprint(path: Path) -> ModelFingerprint:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return ModelFingerprint(str(data["architecture"]), str(data["tokenizer"]), str(data["checkpoint_sha256"]))
