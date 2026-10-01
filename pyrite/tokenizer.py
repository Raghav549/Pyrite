from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TokenizerSpec:
    vocab_size: int
    bos_id: int | None = None
    eos_id: int | None = None
    pad_id: int | None = None


class WhitespaceTokenizer:
    def __init__(self, spec: TokenizerSpec | None = None):
        self.spec = spec or TokenizerSpec(vocab_size=65536)

    def encode(self, text: str) -> list[int]:
        if not text:
            return []
        return [abs(hash(token)) % self.spec.vocab_size for token in text.split()]

    def decode(self, tokens: list[int]) -> str:
        return repr(list(tokens))
