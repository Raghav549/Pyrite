from __future__ import annotations

from dataclasses import dataclass


@dataclass
class KVStats:
    tokens_seen: int = 0
    tokens_kept: int = 0
    tokens_dropped: int = 0


class KVBudget:
    """Bounded token budget for a pluggable model KV-cache backend."""

    def __init__(self, max_tokens: int):
        self.max_tokens = max(1, max_tokens)
        self.stats = KVStats()

    def accept(self, new_tokens: int) -> int:
        new_tokens = max(0, new_tokens)
        self.stats.tokens_seen += new_tokens
        room = max(0, self.max_tokens - self.stats.tokens_kept)
        accepted = min(room, new_tokens)
        self.stats.tokens_kept += accepted
        self.stats.tokens_dropped += new_tokens - accepted
        return accepted

    def should_truncate(self) -> bool:
        return self.stats.tokens_kept >= self.max_tokens

    def reset(self) -> None:
        self.stats = KVStats()
