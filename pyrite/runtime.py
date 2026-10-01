from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .engine import PyriteRuntime, RuntimeStatus
from .inference import InferenceBackend
from .session import LocalSession


@dataclass(frozen=True)
class GenerationPolicy:
    max_new_tokens: int = 128
    temperature: float = 0.7
    top_p: float = 0.9
    seed: int | None = None
    stop_on_budget: bool = True

    def __post_init__(self) -> None:
        if self.max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")
        if self.temperature < 0:
            raise ValueError("temperature must be non-negative")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")


@dataclass(frozen=True)
class GenerationResult:
    text: str
    route: str
    tokens_in: int
    tokens_out: int


class LocalGenerationRuntime:
    """Model-agnostic shell for a local backend.

    The backend is injected so storage/scheduling policy can be tested without a
    checkpoint.  For a real GGUF checkpoint use
    :meth:`LocalGenerationRuntime.from_checkpoint`, which wires the native
    streaming executor (tokenizer included).
    """

    def __init__(
        self,
        runtime: PyriteRuntime | None = None,
        tokenizer=None,
        backend: InferenceBackend | None = None,
        policy: GenerationPolicy | None = None,
    ):
        self.runtime = runtime or PyriteRuntime()
        self.tokenizer = tokenizer
        self.backend = backend
        self.policy = policy or GenerationPolicy()

    def status(self) -> RuntimeStatus:
        return self.runtime.status()

    def ensure_offline(self) -> None:
        if not self.runtime.config.offline:
            raise RuntimeError("local generation requires PYRITE_OFFLINE=1")

    def attach(self, tokenizer, backend: InferenceBackend) -> LocalGenerationRuntime:
        """Return a copy bound to a tokenizer/backend pair."""
        return LocalGenerationRuntime(self.runtime, tokenizer, backend, self.policy)

    @property
    def ready(self) -> bool:
        return self.tokenizer is not None and self.backend is not None

    def close(self) -> None:
        """Release the backend (the executor owns its streamer and threads)."""
        close = getattr(self.backend, "close", None)
        if callable(close):
            close()

    def __enter__(self) -> LocalGenerationRuntime:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def generate(self, prompt: str, policy: GenerationPolicy | None = None) -> GenerationResult:
        self.ensure_offline()
        if not self.ready:
            raise RuntimeError(
                "no tokenizer/backend attached; use from_checkpoint() or attach() first"
            )
        active = policy or self.policy
        # The session holds no resources of its own, and the backend's lifetime
        # belongs to the caller, so the session is not closed here.
        session = LocalSession(self.runtime, self.tokenizer, self.backend, seed=active.seed)
        result = session.generate(
            prompt,
            max_new_tokens=active.max_new_tokens,
            temperature=active.temperature,
            top_p=active.top_p,
            stop_on_budget=active.stop_on_budget,
        )
        return GenerationResult(result.text, result.route, result.tokens_in, result.tokens_out)

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        config=None,
        policy: GenerationPolicy | None = None,
        **executor_kwargs,
    ) -> LocalGenerationRuntime:
        """Wire a real Qwen3-MoE GGUF checkpoint into the runtime."""
        from .executor import Qwen3MoEExecutor

        runtime = PyriteRuntime(config)
        executor = Qwen3MoEExecutor(path, runtime=runtime, **executor_kwargs)
        return cls(runtime, executor.tokenizer, executor, policy)
