from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class Tokenizer(Protocol):
    def encode(self, text: str) -> list[int]:
        ...

    def decode(self, tokens: list[int]) -> str:
        ...


class InferenceBackend(Protocol):
    def generate(
        self,
        input_ids: list[int],
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ) -> list[int]:
        ...


@dataclass(frozen=True)
class GenerationRequest:
    prompt: str
    max_new_tokens: int = 128
    temperature: float = 0.7
    top_p: float = 0.9


class LocalInference:
    """Model-agnostic generation facade used by the local runtime."""

    def __init__(self, tokenizer: Tokenizer, backend: InferenceBackend):
        self.tokenizer = tokenizer
        self.backend = backend

    def generate(self, request: GenerationRequest) -> str:
        input_ids = self.tokenizer.encode(request.prompt)
        output_ids = self.backend.generate(
            input_ids,
            request.max_new_tokens,
            request.temperature,
            request.top_p,
        )
        return self.tokenizer.decode(output_ids)
