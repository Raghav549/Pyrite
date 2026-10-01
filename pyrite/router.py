from __future__ import annotations

import re
from dataclasses import dataclass
from typing import ClassVar


@dataclass(frozen=True)
class Route:
    name: str
    confidence: float
    preferred_blocks: tuple[str, ...]


class HeuristicRouter:
    """Dependency-free baseline router; a learned router can replace it later."""

    PATTERNS: ClassVar[dict[str, re.Pattern[str]]] = {
        "coding": re.compile(r"\b(code|python|javascript|typescript|bug|api|function|class|sql|debug)\b", re.IGNORECASE),
        "math": re.compile(r"[0-9][0-9\s+*/^().%-]*[0-9]|\b(math|equation|calculate|solve)\b", re.IGNORECASE),
        "reasoning": re.compile(r"\b(why|reason|prove|compare|analy[sz]e|derive|logic)\b", re.IGNORECASE),
    }

    def route(self, prompt: str) -> Route:
        for name, pattern in self.PATTERNS.items():
            if pattern.search(prompt):
                return Route(name, 0.72, (f"shared:{name}", "shared:general"))
        return Route("general", 0.55, ("shared:general",))
