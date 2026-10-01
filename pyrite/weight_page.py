from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib


@dataclass(frozen=True)
class WeightPageRef:
    digest: str
    source: Path
    offset: int
    size: int


class PageManifest:
    """Deterministic content-addressed manifest for source-file byte pages."""

    def __init__(self, page_bytes: int = 2 * 1024 * 1024):
        if page_bytes <= 0:
            raise ValueError("page_bytes must be positive")
        self.page_bytes = page_bytes

    def build(self, source: Path) -> tuple[WeightPageRef, ...]:
        refs = []
        offset = 0
        with Path(source).open("rb") as fh:
            while True:
                payload = fh.read(self.page_bytes)
                if not payload:
                    break
                digest = hashlib.sha256(payload).hexdigest()
                refs.append(WeightPageRef(digest, Path(source), offset, len(payload)))
                offset += len(payload)
        return tuple(refs)
