from __future__ import annotations

import hashlib
from collections import OrderedDict
from dataclasses import dataclass


@dataclass(frozen=True)
class PrefixEntry:
    key: str
    token_count: int
    byte_size: int
    payload: bytes


class PrefixKVCache:
    """Local prefix cache with bounded bytes and LRU replacement."""

    def __init__(self, max_bytes: int):
        if max_bytes <= 0:
            raise ValueError("max_bytes must be positive")
        self.max_bytes = max_bytes
        self._entries: OrderedDict[str, PrefixEntry] = OrderedDict()
        self._bytes = 0

    @staticmethod
    def key(tokens: list[int]) -> str:
        raw = ",".join(map(str, tokens)).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    def get(self, tokens: list[int]) -> PrefixEntry | None:
        key = self.key(tokens)
        entry = self._entries.get(key)
        if entry is None:
            return None
        self._entries.move_to_end(key)
        return entry

    def put(self, tokens: list[int], payload: bytes) -> PrefixEntry | None:
        size = len(payload)
        if size > self.max_bytes:
            return None
        key = self.key(tokens)
        old = self._entries.pop(key, None)
        if old is not None:
            self._bytes -= old.byte_size
        entry = PrefixEntry(key, len(tokens), size, payload)
        self._entries[key] = entry
        self._bytes += size
        while self._bytes > self.max_bytes and self._entries:
            _, victim = self._entries.popitem(last=False)
            self._bytes -= victim.byte_size
        return entry

    @property
    def bytes_used(self) -> int:
        return self._bytes

    def clear(self) -> None:
        self._entries.clear()
        self._bytes = 0
