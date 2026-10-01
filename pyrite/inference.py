from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class GenerationRequest:
    prompt: str
    max_new_tokens: int = 128
    temperature: float = 0.7
    top_p: float = 0.9

    def __post_init__(self) -> None:
        if self.max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")
        if self.temperature < 0:
            raise ValueError("temperature must be non-negative")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")


class InferenceBackend(Protocol):
    def generate(self, input_ids: list[int], max_new_tokens: int, temperature: float, top_p: float) -> list[int]:
        ...


class LocalInference:
    def __init__(self, tokenizer, backend: InferenceBackend):
        self.tokenizer = tokenizer
        self.backend = backend

    def generate(
        self,
        request: GenerationRequest | str,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> str:
        if isinstance(request, str):
            request = GenerationRequest(request, max_new_tokens, temperature, top_p)
        ids = self.tokenizer.encode(request.prompt)
        out = self.backend.generate(
            ids,
            request.max_new_tokens,
            request.temperature,
            request.top_p,
        )
        return self.tokenizer.decode(out)
