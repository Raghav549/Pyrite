from __future__ import annotations

from dataclasses import dataclass

from .engine import PyriteRuntime
from .inference import InferenceBackend
from .kv import KVBudget
from .sampler import Sampler


@dataclass(frozen=True)
class SessionResult:
    text: str
    route: str
    tokens_in: int
    tokens_out: int


class LocalSession:
    """One prompt submission against an attached backend.

    The session owns routing, its KV token budget and the sampler; the backend
    owns model execution (which may be the native streaming executor).  A new
    session starts with an empty context, so repeated prompts do not exhaust a
    budget shared across calls.
    """

    def __init__(
        self,
        runtime: PyriteRuntime,
        tokenizer,
        backend: InferenceBackend,
        seed: int | None = None,
    ):
        self.runtime = runtime
        self.tokenizer = tokenizer
        self.backend = backend
        self.sampler = Sampler(seed=seed)
        self.kv = KVBudget(runtime.config.max_kv_tokens)

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop_on_budget: bool = True,
    ) -> SessionResult:
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")
        if temperature < 0:
            raise ValueError("temperature must be non-negative")
        if not 0 < top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")

        self.runtime.policy.validate()
        route = self.runtime.route(prompt)
        input_ids = self.tokenizer.encode(prompt)
        if not input_ids:
            raise ValueError("prompt encodes to zero tokens")
        accepted = self.kv.accept(len(input_ids))
        if accepted < len(input_ids):
            raise MemoryError("input exceeds configured KV token budget")
        room = max(0, self.runtime.config.max_kv_tokens - self.kv.stats.tokens_kept)
        if max_new_tokens > room:
            if not stop_on_budget:
                raise MemoryError(
                    f"{max_new_tokens} new tokens do not fit the remaining KV budget "
                    f"({room})"
                )
            max_new_tokens = room
        output_ids = self.backend.generate(input_ids, max_new_tokens, temperature, top_p)
        new_tokens = max(0, len(output_ids) - len(input_ids))
        self.kv.accept(new_tokens)
        return SessionResult(
            text=self.tokenizer.decode(output_ids),
            route=route.name,
            tokens_in=len(input_ids),
            tokens_out=new_tokens,
        )

    def close(self, close_backend: bool = False) -> None:
        """Release session state.

        The backend is owned by the caller, so it is only closed on request.
        """
        if close_backend:
            close = getattr(self.backend, "close", None)
            if callable(close):
                close()
