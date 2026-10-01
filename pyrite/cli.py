from __future__ import annotations

import argparse
import json
import sys

from .config import RuntimeConfig
from .engine import PyriteRuntime
from .local import LocalModel


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pyrite", description="Local memory-budgeted AI runtime")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("status")

    route = sub.add_parser("route")
    route.add_argument("prompt")

    inspect = sub.add_parser("inspect")
    inspect.add_argument("checkpoint")

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = RuntimeConfig.from_env()
    runtime = PyriteRuntime(config)

    if args.command == "status":
        print(json.dumps(runtime.status().__dict__, indent=2))
        return 0

    if args.command == "route":
        print(json.dumps(runtime.route(args.prompt).__dict__, indent=2))
        return 0

    if args.command == "inspect":
        with LocalModel(args.checkpoint, config=config) as model:
            info = model.info()
            print(json.dumps({
                "path": str(info.path),
                "format": info.format,
                "block_count": info.block_count,
                "total_bytes": info.total_bytes,
            }, indent=2))
            return 0

    print("unknown command", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
