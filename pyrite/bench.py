from __future__ import annotations

import argparse
import json
import time

from .engine import PyriteRuntime


def _policy_benchmark(prompt: str, blocks: int) -> dict[str, object]:
    runtime = PyriteRuntime()
    names = [f"layer:{index:04d}" for index in range(blocks)]

    started = time.perf_counter()
    route = runtime.route(prompt)
    plans = [runtime.plan(names, index) for index in range(len(names))]
    elapsed = time.perf_counter() - started

    return {
        "route": route.to_dict() if hasattr(route, "to_dict") else dict(route.__dict__),
        "plans": len(plans),
        "elapsed_s": elapsed,
        "status": runtime.status().to_dict(),
    }


def _checkpoint_benchmark(
    checkpoint: str,
    prompt: str,
    tokens: int,
    workers: int,
    prefetch: bool,
) -> dict[str, object]:
    """Measure real streaming generation on a local GGUF checkpoint."""
    from .executor import Qwen3MoEExecutor

    with Qwen3MoEExecutor(checkpoint, workers=workers, prefetch=prefetch, sampler_seed=0) as executor:
        report = executor.checkpoint_report()
        started = time.perf_counter()
        result = executor.generate_text(prompt, max_new_tokens=tokens, temperature=0.0)
        elapsed = time.perf_counter() - started
        generated = int(result["new_tokens"])
        stats = executor.stats.to_dict()
        bytes_streamed = int(stats["bytes_loaded"])

    return {
        "checkpoint": str(checkpoint),
        "prompt_tokens": result["prompt_tokens"],
        "new_tokens": generated,
        "elapsed_s": elapsed,
        "tokens_per_second": (generated / elapsed) if elapsed > 0 else 0.0,
        "streamed_bytes_per_second": (bytes_streamed / elapsed) if elapsed > 0 else 0.0,
        "streamed_bytes": bytes_streamed,
        "total_checkpoint_bytes": report["total_bytes"],
        "unsupported_tensor_types": report["unsupported_tensor_types"],
        "stats": stats,
    }


def main(argv: list[str] | None = None) -> int:
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
    args = parser.parse_args(argv)

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
