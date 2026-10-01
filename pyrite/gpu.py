from __future__ import annotations

import os
import platform
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AcceleratorInfo:
    name: str
    available: bool
    reason: str


def _cuda_probe() -> AcceleratorInfo:
    """Detect a CUDA device without importing heavyweight runtimes.

    Detection is deliberately conservative: it reports what the host exposes,
    not what Pyrite can execute (the reference kernels are CPU-side).
    """
    visible = os.getenv("CUDA_VISIBLE_DEVICES")
    if visible == "-1":
        return AcceleratorInfo("cuda", False, "CUDA_VISIBLE_DEVICES hides all devices")
    if visible:
        return AcceleratorInfo("cuda", True, "CUDA_VISIBLE_DEVICES exposes a device")
    try:
        nodes = any(Path("/dev").glob("nvidia*"))
    except OSError:  # pragma: no cover - /dev is unreadable on some hosts
        nodes = False
    if nodes:
        return AcceleratorInfo("cuda", True, "NVIDIA device nodes are present in /dev")
    return AcceleratorInfo("cuda", False, "no CUDA device nodes were detected")


def _metal_probe() -> AcceleratorInfo:
    if platform.system() != "Darwin":
        return AcceleratorInfo("metal", False, "Metal is only available on macOS")
    if platform.machine() in {"arm64", "aarch64"}:
        return AcceleratorInfo("metal", True, "Apple silicon GPU is available via Metal")
    return AcceleratorInfo("metal", False, "Metal requires Apple silicon")


def detect_accelerators() -> list[AcceleratorInfo]:
    """Return optional accelerator information for the current host."""
    return [_cuda_probe(), _metal_probe()]
