from __future__ import annotations

from typing import Protocol


class InferenceBackend(Protocol):
    def generate(self, input_ids: list[int], max_new_tokens: int, temperature: float, top_p: float) -> list[int]:
        ...


class LocalInference:
    def __init__(self, tokenizer, backend: InferenceBackend):
        self.tokenizer = tokenizer
        self.backend = backend

    def generate(self, prompt: str, max_new_tokens: int = 128, temperature: float = 0.7, top_p: float = 0.9) -> str:
        ids = self.tokenizer.encode(prompt)
        out = self.backend.generate(ids, max_new_tokens, temperature, top_p)
        return self.tokenizer.decode(out)
