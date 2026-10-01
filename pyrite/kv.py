from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class KVToken:
    position: int
    precision_bits: int
    importance: float


@dataclass
class KVStats:
    tokens_seen: int = 0
    tokens_kept: int = 0
    tokens_dropped: int = 0
    estimated_bytes: int = 0


class AdaptiveKVCache:
    """Byte-aware KV token policy with adaptive 2/4/8/16-bit tiers."""

    def __init__(self, max_bytes: int, bytes_per_token_fp16: int = 1024):
        if max_bytes <= 0:
            raise ValueError("max_bytes must be positive")
        if bytes_per_token_fp16 <= 0:
            raise ValueError("bytes_per_token_fp16 must be positive")
        self.max_bytes = max_bytes
        self.bytes_per_token_fp16 = bytes_per_token_fp16
        self.tokens: list[KVToken] = []
        self.stats = KVStats()

    @staticmethod
    def choose_precision(importance: float) -> int:
        importance = max(0.0, min(1.0, importance))
        if importance >= 0.8:
            return 16
        if importance >= 0.55:
            return 8
        if importance >= 0.25:
            return 4
        return 2

    def token_bytes(self, precision_bits: int) -> int:
        return max(1, (self.bytes_per_token_fp16 * precision_bits + 15) // 16)

    def add(self, importance: float) -> KVToken | None:
        precision = self.choose_precision(importance)
        required = self.token_bytes(precision)
        if self.stats.estimated_bytes + required > self.max_bytes:
            self.evict_low_importance(required)
        if self.stats.estimated_bytes + required > self.max_bytes:
            self.stats.tokens_dropped += 1
            self.stats.tokens_seen += 1
            return None

        token = KVToken(len(self.tokens), precision, max(0.0, min(1.0, importance)))
        self.tokens.append(token)
        self.stats.tokens_seen += 1
        self.stats.tokens_kept += 1
        self.stats.estimated_bytes += required
        return token

    def evict_low_importance(self, required_bytes: int) -> None:
        del required_bytes
        while self.tokens and self.stats.estimated_bytes > max(0, self.max_bytes - 1):
            victim = min(self.tokens, key=lambda item: item.importance)
            self.tokens.remove(victim)
            self.stats.estimated_bytes -= self.token_bytes(victim.precision_bits)

    def reset(self) -> None:
        self.tokens.clear()
        self.stats = KVStats()


class KVBudget:
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
