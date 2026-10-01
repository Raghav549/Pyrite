from __future__ import annotations

import argparse
import json
import sys

from .adapters.gguf import GGUFReader
from .adapters.safetensors import SafetensorsAdapter
from .config import RuntimeConfig
from .engine import PyriteRuntime
from .local import LocalModel
from .storage_pages import ContentAddressedPager


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pyrite", description="Local memory-budgeted AI runtime")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("status")

    route = sub.add_parser("route")
    route.add_argument("prompt")

    inspect = sub.add_parser("inspect")
    inspect.add_argument("checkpoint")

    pages = sub.add_parser("pages")
    pages.add_argument("checkpoint")
    pages.add_argument("--page-bytes", type=int, default=2 * 1024 * 1024)

    return parser


def _inspect_tensor_checkpoint(path: str) -> dict:
    from pathlib import Path

    p = Path(path)
    if p.suffix.lower() in {".safetensors", ".safetensor"}:
        adapter = SafetensorsAdapter(p)
        blocks = adapter.blocks()
        return {
            "format": "safetensors",
            "tensor_count": len(blocks),
            "tensors": [
                {"name": b.block_id.removeprefix("tensor:"), "offset": b.offset, "size": b.size, "dtype": b.dtype}
                for b in blocks
            ],
        }
    with p.open("rb") as fh:
        magic = fh.read(4)
    if magic == b"GGUF":
        reader = GGUFReader(p)
        return {
            "format": "gguf",
            "version": reader.header().version,
            "tensor_count": len(reader.tensor_index()),
            "metadata": reader.metadata(),
            "tensors": [
                {"name": t.name, "shape": t.dims, "type": t.ggml_type, "offset": t.offset, "size": t.size}
                for t in reader.tensor_index()
            ],
        }
    raise ValueError("tensor inspection requires GGUF or Safetensors")


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
        try:
            result = _inspect_tensor_checkpoint(args.checkpoint)
            if result.get("format") == "gguf":
                print(json.dumps(result, indent=2, default=list))
            else:
                print(json.dumps(result, indent=2))
            return 0
        except ValueError:
            with LocalModel(args.checkpoint, config=config) as model:
                info = model.info()
                print(json.dumps({
                    "path": str(info.path),
                    "format": info.format,
                    "block_count": info.block_count,
                    "total_bytes": info.total_bytes,
                }, indent=2))
                return 0

    if args.command == "pages":
        from pathlib import Path

        source = Path(args.checkpoint)
        pager = ContentAddressedPager(config.storage_dir / "pages", page_bytes=args.page_bytes)
        pages = pager.ingest(source)
        print(json.dumps({
            "source": str(source),
            "page_count": len(pages),
            "page_bytes": args.page_bytes,
            "unique_pages": len({page.digest for page in pages}),
            "bytes": sum(page.size for page in pages),
        }, indent=2))
        return 0

    print("unknown command", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
