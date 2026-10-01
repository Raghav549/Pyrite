from __future__ import annotations

import argparse
import json

from .config import RuntimeConfig
from .engine import PyriteRuntime


def main() -> int:
    parser = argparse.ArgumentParser(prog="pyrite", description="Local, memory-budgeted AI runtime")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("status")
    route = sub.add_parser("route")
    route.add_argument("prompt")
    args = parser.parse_args()
    runtime = PyriteRuntime(RuntimeConfig.from_env())
    if args.command == "status":
        print(json.dumps(runtime.status().__dict__, indent=2))
        return 0
    if args.command == "route":
        print(json.dumps(runtime.route(args.prompt).__dict__, indent=2))
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
