"""Reproducible Pyrite benchmarks with machine information.

Every benchmark payload includes the host it ran on (OS, Python, CPU, RAM) so
numbers are comparable across machines, plus the measured resident-memory and
I/O counters — never invented values.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from contextlib import suppress
from pathlib import Path

from .engine import PyriteRuntime
from .memory import process_memory_mb


def _cpu_model() -> str | None:
    """Best-effort CPU model name (Linux /proc/cpuinfo, else platform)."""
    if sys.platform.startswith("linux"):
        with suppress(OSError, ValueError), open("/proc/cpuinfo", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("model name"):
                    return line.partition(":")[2].strip() or None
    name = platform.processor() or None
    return name


def total_memory_mb() -> float | None:
    """Best-effort total physical RAM in MB, or ``None`` if unmeasurable."""
    if sys.platform.startswith("linux"):
        with suppress(OSError, ValueError), open("/proc/meminfo", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("MemTotal:"):
                    return float(line.split()[1]) / 1024.0
    elif os.name == "nt":
        with suppress(AttributeError, OSError, TypeError, ValueError):
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            status = MEMORYSTATUSEX()
            status.dwLength = ctypes.sizeof(status)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return status.ullTotalPhys / (1024 * 1024)
    elif sys.platform == "darwin":
        with suppress(OSError, ValueError):
            import subprocess

            out = subprocess.run(
                ["/usr/sbin/sysctl", "-n", "hw.memsize"],
                capture_output=True,
                text=True,
                timeout=5,
                check=True,
            )
            return float(out.stdout.strip()) / (1024 * 1024)
    return None


def machine_info() -> dict[str, object]:
    """Host description included in every benchmark payload."""
    return {
        "os": platform.platform(),
        "system": platform.system(),
        "machine": platform.machine(),
        "python": sys.version.split()[0],
        "cpu_count": os.cpu_count(),
        "cpu_model": _cpu_model(),
        "total_ram_mb": total_memory_mb(),
    }


def _policy_benchmark(prompt: str, blocks: int) -> dict[str, object]:
    runtime = PyriteRuntime()
    names = [f"layer:{index:04d}" for index in range(blocks)]

    rss_before = process_memory_mb()
    started = time.perf_counter()
    route = runtime.route(prompt)
    plans = [runtime.plan(names, index) for index in range(len(names))]
    elapsed = time.perf_counter() - started

    return {
        "benchmark": "policy",
        "machine": machine_info(),
        "route": route.to_dict() if hasattr(route, "to_dict") else dict(route.__dict__),
        "plans": len(plans),
        "elapsed_s": elapsed,
        "process_rss_before_mb": rss_before,
        "process_rss_after_mb": process_memory_mb(),
        "status": runtime.status().to_dict(),
    }


def _quantization_mix(checkpoint: str) -> dict[str, int]:
    """Count checkpoint tensors per GGML type name (parsed, not guessed)."""
    from .adapters.gguf import GGUFReader
    from .ggml_types import type_name

    reader = GGUFReader(Path(checkpoint))
    mix: dict[str, int] = {}
    for tensor in reader.tensor_index():
        name = type_name(tensor.ggml_type)
        mix[name] = mix.get(name, 0) + 1
    return mix


def _checkpoint_benchmark(
    checkpoint: str,
    prompt: str,
    tokens: int,
    workers: int,
    prefetch: bool,
) -> dict[str, object]:
    """Measure real streaming generation on a local GGUF checkpoint."""
    from .executor import open_executor

    rss_before = process_memory_mb()
    with open_executor(checkpoint, workers=workers, prefetch=prefetch, sampler_seed=0) as executor:
        report = executor.checkpoint_report()
        started = time.perf_counter()
        result = executor.generate_text(prompt, max_new_tokens=tokens, temperature=0.0)
        elapsed = time.perf_counter() - started
        rss_after = process_memory_mb()
        generated = int(result["new_tokens"])
        stats = executor.stats.to_dict()
        bytes_streamed = int(stats["bytes_loaded"])

    checkpoint_bytes = Path(checkpoint).stat().st_size
    return {
        "benchmark": "checkpoint-generation",
        "machine": machine_info(),
        "checkpoint": str(checkpoint),
        "architecture": report["architecture"],
        "layers": report["layers"],
        "hidden_size": report["hidden_size"],
        "experts": report.get("experts"),
        "experts_per_token": report.get("experts_per_token"),
        "intermediate_size": report.get("intermediate_size"),
        "vocab_size": report["vocab_size"],
        "tensor_count": report["tensor_count"],
        "quantization_mix": _quantization_mix(checkpoint),
        "checkpoint_bytes": checkpoint_bytes,
        "total_tensor_bytes": report["total_bytes"],
        "prompt": prompt,
        "prompt_tokens": result["prompt_tokens"],
        "new_tokens": generated,
        "elapsed_s": elapsed,
        "tokens_per_second": (generated / elapsed) if elapsed > 0 else 0.0,
        "streamed_bytes_per_second": (bytes_streamed / elapsed) if elapsed > 0 else 0.0,
        "streamed_bytes": bytes_streamed,
        "unsupported_tensor_types": report["unsupported_tensor_types"],
        "process_rss_before_mb": rss_before,
        "process_rss_after_mb": rss_after,
        "stats": stats,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pyrite runtime benchmark")
    parser.add_argument("--prompt", default="analyze this Python API bug")
    parser.add_argument(
        "--blocks",
        type=int,
        default=16,
        help="number of synthetic blocks for the scheduler benchmark",
    )
    parser.add_argument(
        "--checkpoint",
        help="benchmark real streaming generation on a GGUF checkpoint instead",
    )
    parser.add_argument("--tokens", type=int, default=8, help="tokens to generate with --checkpoint")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--no-prefetch", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.checkpoint:
        payload = _checkpoint_benchmark(
            args.checkpoint,
            args.prompt,
            max(0, args.tokens),
            max(1, args.workers),
            not args.no_prefetch,
        )
    else:
        payload = _policy_benchmark(args.prompt, max(0, args.blocks))

    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
