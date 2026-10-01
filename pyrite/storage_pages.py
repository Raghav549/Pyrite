from __future__ import annotations

import hashlib
import mmap
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class WeightPage:
    page_id: str
    path: Path
    offset: int
    size: int
    sha256: str


class ContentAddressedPager:
    """Immutable local weight pages addressed by content hash."""

    def __init__(self, root: Path, page_bytes: int = 2 * 1024 * 1024):
        if page_bytes <= 0:
            raise ValueError("page_bytes must be positive")
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.page_bytes = page_bytes

    @staticmethod
    def digest(payload: bytes) -> str:
        return hashlib.sha256(payload).hexdigest()

    def ingest(self, source: Path) -> tuple[WeightPage, ...]:
        source = Path(source)
        pages: list[WeightPage] = []
        with source.open("rb") as fh:
            index = 0
            offset = 0
            while True:
                payload = fh.read(self.page_bytes)
                if not payload:
                    break
                digest = self.digest(payload)
                target = self.root / f"{digest}.page"
                if not target.exists():
                    target.write_bytes(payload)
                pages.append(WeightPage(digest, target, offset, len(payload), digest))
                offset += len(payload)
                index += 1
        return tuple(pages)

    def verify(self, page: WeightPage) -> bool:
        """Re-hash the stored page bytes; ``False`` means corruption or loss."""
        try:
            payload = page.path.read_bytes()
        except OSError:
            return False
        return len(payload) == page.size and self.digest(payload) == page.sha256


class MMapPage:
    """Small lifetime-safe mmap view over one immutable page.

    Views handed out by :meth:`view` are tracked so ``close()`` can release any
    that are still exported; otherwise ``mmap.close()`` raises ``BufferError``
    and the mapping leaks until garbage collection.
    """

    def __init__(self, page: WeightPage):
        if page.size <= 0:
            raise ValueError(f"cannot mmap an empty page: {page.page_id!r}")
        self.page = page
        self._fh = page.path.open("rb")
        try:
            # Length 0 maps the whole file on both POSIX and Windows.
            self._map = mmap.mmap(self._fh.fileno(), 0, access=mmap.ACCESS_READ)
        except (OSError, ValueError) as exc:
            self._fh.close()
            raise ValueError(f"cannot mmap page {page.page_id!r}: {exc}") from exc
        self._views: list[memoryview] = []

    def view(self) -> memoryview:
        view = memoryview(self._map)
        self._views.append(view)
        return view

    def close(self) -> None:
        for view in self._views:
            try:
                view.release()
            except ValueError:  # already released by the caller
                continue
        self._views.clear()
        self._map.close()
        self._fh.close()

    def __enter__(self) -> MMapPage:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
