from __future__ import annotations

from dataclasses import dataclass

from .engine import PyriteRuntime
from .inference import InferenceBackend
from .sampler import Sampler


@dataclass(frozen=True)
class SessionResult:
    text: str
    route: str
    tokens_in: int
    tokens_out: int


class LocalSession:
    def __init__(self, runtime: PyriteRuntime, tokenizer, backend: InferenceBackend, seed: int | None = None):
        self.runtime = runtime
        self.tokenizer = tokenizer
        self.backend = backend
        self.sampler = Sampler(seed=seed)

    def generate(self, prompt: str, max_new_tokens: int = 128, temperature: float = 0.7, top_p: float = 0.9) -> SessionResult:
        self.runtime.policy.validate()
        route = self.runtime.route(prompt)
        input_ids = self.tokenizer.encode(prompt)
        accepted = self.runtime.kv.accept(len(input_ids))
        if accepted < len(input_ids):
            raise MemoryError("input exceeds configured KV token budget")
        output_ids = self.backend.generate(input_ids, max_new_tokens, temperature, top_p)
        new_tokens = max(0, len(output_ids) - len(input_ids))
        self.runtime.kv.accept(new_tokens)
        return SessionResult(
            text=self.tokenizer.decode(output_ids),
            route=route.name,
            tokens_in=len(input_ids),
            tokens_out=new_tokens,
        )

    def close(self) -> None:
        close = getattr(self.backend, "close", None)
        if callable(close):
            close()
