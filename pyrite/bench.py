from __future__ import annotations

import argparse
import json
import time

from .engine import PyriteRuntime


def main() -> int:
    parser = argparse.ArgumentParser(description="Pyrite scheduler benchmark")
    parser.add_argument("--prompt", default="analyze this Python API bug")
    parser.add_argument("--blocks", type=int, default=16)
    args = parser.parse_args()

    runtime = PyriteRuntime()
    blocks = [f"layer:{i:04d}" for i in range(args.blocks)]

    started = time.perf_counter()
    route = runtime.route(args.prompt)
    plans = [runtime.plan(blocks, i) for i in range(len(blocks))]
    elapsed = time.perf_counter() - started

    print(
        json.dumps(
            {
                "route": route.__dict__,
                "plans": len(plans),
                "elapsed_s": elapsed,
                "status": runtime.status().__dict__,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
