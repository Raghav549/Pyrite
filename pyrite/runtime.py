from __future__ import annotations

from dataclasses import dataclass

from .engine import PyriteRuntime, RuntimeStatus


@dataclass(frozen=True)
class GenerationPolicy:
    max_new_tokens: int = 128
    temperature: float = 0.7
    top_p: float = 0.9
    stop_on_budget: bool = True


class LocalGenerationRuntime:
    """Model-agnostic shell for a local backend.

    The backend is intentionally injected so storage/scheduling policy can be
    tested without downloading a model. A backend should expose generate().
    """

    def __init__(self, runtime: PyriteRuntime | None = None):
        self.runtime = runtime or PyriteRuntime()

    def status(self) -> RuntimeStatus:
        return self.runtime.status()

    def ensure_offline(self) -> None:
        if not self.runtime.config.offline:
            raise RuntimeError("local generation requires PYRITE_OFFLINE=1")
