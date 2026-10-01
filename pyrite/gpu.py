from __future__ import annotations

from dataclasses import dataclass
import os


@dataclass(frozen=True)
class AcceleratorInfo:
    name: str
    available: bool
    reason: str


def detect_accelerators() -> list[AcceleratorInfo]:
    """Detect optional accelerators without importing heavyweight runtimes."""
    cuda = bool(os.getenv("CUDA_VISIBLE_DEVICES")) and os.getenv("CUDA_VISIBLE_DEVICES") != "-1"
    return [
        AcceleratorInfo(
            name="cuda",
            available=cuda,
            reason="CUDA visibility is controlled by the host environment",
        )
    ]
