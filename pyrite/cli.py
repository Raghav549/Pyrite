from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .adapters.gguf import GGUFReader
from .adapters.safetensors import SafetensorsAdapter
from .config import RuntimeConfig
from .engine import PyriteRuntime
from .ggml_types import known_types
from .local import LocalModel
from .storage_pages import ContentAddressedPager
from .tensor_ops import DECODABLE_TYPES


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pyrite",
        description="Local memory-budgeted AI runtime",
    )
    parser.add_argument("--version", action="version", version=f"pyrite {__version__}")
    parser.add_argument(
        "--profile",
        help="JSON runtime profile (for example examples/four_gb.json)",
    )
    parser.add_argument(
        "--ram-mb",
        type=int,
        help="override the resident RAM budget for this command",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="force offline mode (already the default)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    status = sub.add_parser("status", help="show the runtime budget and process status")
    status.add_argument("--json", action="store_true", help="print machine-readable JSON")

    route = sub.add_parser("route", help="classify a prompt without running a model")
    route.add_argument("prompt")

    inspect = sub.add_parser("inspect", help="inspect a checkpoint's metadata and tensors")
    inspect.add_argument("checkpoint")
    inspect.add_argument(
        "--full-metadata",
        action="store_true",
        help="print every metadata value (large token arrays included)",
    )
    inspect.add_argument("--limit", type=int, default=20, help="tensor rows to print")

    qwen = sub.add_parser("qwen3-check", help="validate a Qwen3-MoE GGUF contract")
    qwen.add_argument("checkpoint")
    qwen.add_argument("--full", action="store_true", help="include the streaming report")

    pages = sub.add_parser("pages", help="content-address a checkpoint into local pages")
    pages.add_argument("checkpoint")
    pages.add_argument("--page-bytes", type=int, default=2 * 1024 * 1024)

    tokenize = sub.add_parser("tokenize", help="tokenize text with a GGUF vocabulary")
    tokenize.add_argument("checkpoint")
    tokenize.add_argument("text")
    tokenize.add_argument("--ids-only", action="store_true")

    generate = sub.add_parser("generate", help="run real local generation on a checkpoint")
    generate.add_argument("checkpoint")
    generate.add_argument("--prompt", required=True)
    generate.add_argument("--max-new-tokens", type=int, default=16)
    generate.add_argument("--temperature", type=float, default=0.0)
    generate.add_argument("--top-p", type=float, default=1.0)
    generate.add_argument("--seed", type=int, default=None)
    generate.add_argument("--workers", type=int, default=2)
    generate.add_argument("--no-prefetch", action="store_true")

    types = sub.add_parser("types", help="list GGML tensor types and decoder support")
    types.add_argument("--decodable-only", action="store_true")

    bench = sub.add_parser("bench", help="run a reproducible runtime benchmark")
    bench.add_argument("--prompt", default="analyze this Python API bug")
    bench.add_argument(
        "--blocks",
        type=int,
        default=16,
        help="number of synthetic blocks for the scheduler benchmark",
    )
    bench.add_argument(
        "--checkpoint",
        help="benchmark real streaming generation on a GGUF checkpoint instead",
    )
    bench.add_argument("--tokens", type=int, default=8, help="tokens to generate with --checkpoint")
    bench.add_argument("--workers", type=int, default=2)
    bench.add_argument("--no-prefetch", action="store_true")

    return parser


def _config(args: argparse.Namespace) -> RuntimeConfig:
    config = RuntimeConfig.from_env()
    if args.profile:
        config = RuntimeConfig.from_profile(args.profile)
    if args.ram_mb is not None:
        config = config.with_overrides(ram_budget_mb=args.ram_mb)
    if args.offline:
        config = config.with_overrides(offline=True)
    config.validate()
    return config


def _print(payload: object) -> None:
    print(json.dumps(payload, indent=2, default=str))


def _inspect_tensor_checkpoint(path: str, full_metadata: bool, limit: int) -> dict:
    p = Path(path)
    if p.suffix.lower() in {".safetensors", ".safetensor"}:
        adapter = SafetensorsAdapter(p)
        return {
            "format": "safetensors",
            "tensor_count": len(adapter.blocks()),
            "tensors": [
                {
                    "name": meta.name,
                    "shape": list(meta.shape),
                    "dtype": meta.dtype,
                    "offset": meta.offset,
                    "size": meta.size,
                }
                for meta in adapter.tensors()[: max(0, limit)]
            ],
        }
    with p.open("rb") as fh:
        magic = fh.read(4)
    if magic == b"GGUF":
        reader = GGUFReader(p)
        header = reader.header()
        tensors = reader.tensor_index()
        metadata = reader.metadata() if full_metadata else reader.metadata_summary()
        return {
            "format": "gguf",
            "version": header.version,
            "alignment": header.alignment,
            "tensor_count": len(tensors),
            "metadata": metadata,
            "tensors": [
                {
                    "name": t.name,
                    "shape": list(t.dims),
                    "type": reader.describe_type(t.ggml_type),
                    "offset": t.offset,
                    "size": t.size,
                }
                for t in tensors[: max(0, limit)]
            ],
        }
    raise ValueError("tensor inspection requires GGUF or Safetensors")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = _config(args)
    except (ValueError, FileNotFoundError) as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2

    try:
        runtime = PyriteRuntime(config)
    except (OSError, ValueError, MemoryError) as exc:
        print(f"runtime error: {exc}", file=sys.stderr)
        return 2

    if args.command == "status":
        payload = runtime.describe()
        if not args.json:
            payload["process_rss_mb"] = round(float(payload["status"]["process_rss_mb"]), 1)
        _print(payload)
        return 0

    if args.command == "route":
        _print(runtime.route(args.prompt).__dict__)
        return 0

    if args.command == "bench":
        from .bench import main as bench_main

        forwarded = [
            "--prompt", args.prompt,
            "--blocks", str(args.blocks),
            "--tokens", str(args.tokens),
            "--workers", str(args.workers),
        ]
        if args.checkpoint:
            forwarded += ["--checkpoint", args.checkpoint]
        if args.no_prefetch:
            forwarded.append("--no-prefetch")
        return bench_main(forwarded)

    if args.command == "types":
        entries = [
            {
                "id": t.type_id,
                "name": t.name,
                "block_size": t.block_size,
                "bytes_per_block": t.bytes_per_block,
                "decodable": t.type_id in DECODABLE_TYPES,
            }
            for t in known_types()
        ]
        if args.decodable_only:
            entries = [entry for entry in entries if entry["decodable"]]
        _print({"types": entries})
        return 0

    if args.command == "qwen3-check":
        try:
            from .qwen3_moe import Qwen3MoECheckpoint

            model = Qwen3MoECheckpoint(args.checkpoint)
            summary = model.routing_summary()
            payload: dict[str, object] = {
                "model": "Qwen3-MoE",
                "config": model.config.to_dict(),
                "routing": summary,
            }
            if args.full:
                payload["streaming"] = model.validate_for_streaming(
                    runtime.config.resident_byte_budget
                )
            _print(payload)
            return 0
        except (ValueError, KeyError, FileNotFoundError) as exc:
            print(f"qwen3-check: {exc}", file=sys.stderr)
            return 1

    if args.command == "inspect":
        try:
            _print(_inspect_tensor_checkpoint(args.checkpoint, args.full_metadata, args.limit))
            return 0
        except FileNotFoundError:
            print(f"inspect: no such checkpoint: {args.checkpoint}", file=sys.stderr)
            return 1
        except ValueError as exc:
            try:
                with LocalModel(args.checkpoint, config=config) as model:
                    info = model.info()
                    _print(
                        {
                            "path": str(info.path),
                            "format": info.format,
                            "block_count": info.block_count,
                            "total_bytes": info.total_bytes,
                        }
                    )
                    return 0
            except (ValueError, FileNotFoundError) as inner:
                print(f"inspect: {inner or exc}", file=sys.stderr)
                return 1

    if args.command == "pages":
        source = Path(args.checkpoint)
        if not source.is_file():
            print(f"pages: no such file: {source}", file=sys.stderr)
            return 1
        pager = ContentAddressedPager(config.storage_dir / "pages", page_bytes=args.page_bytes)
        pages = pager.ingest(source)
        _print(
            {
                "source": str(source),
                "page_count": len(pages),
                "page_bytes": args.page_bytes,
                "unique_pages": len({page.page_id for page in pages}),
                "bytes": sum(page.size for page in pages),
            }
        )
        return 0

    if args.command == "tokenize":
        try:
            from .tokenizer import load_gguf_tokenizer

            reader = GGUFReader(Path(args.checkpoint))
            tokenizer = load_gguf_tokenizer(reader)
            if tokenizer is None:
                print("tokenize: checkpoint has no byte-level BPE vocabulary", file=sys.stderr)
                return 1
            ids = tokenizer.encode(args.text)
            if args.ids_only:
                _print({"ids": ids})
            else:
                _print(
                    {
                        "ids": ids,
                        "tokens": [tokenizer.tokens[i] for i in ids],
                        "decoded": tokenizer.decode(ids),
                    }
                )
            return 0
        except FileNotFoundError:
            print(f"tokenize: no such checkpoint: {args.checkpoint}", file=sys.stderr)
            return 1
        except ValueError as exc:
            print(f"tokenize: {exc}", file=sys.stderr)
            return 1

    if args.command == "generate":
        from .executor import UnsupportedTensorType, open_executor

        try:
            with open_executor(
                args.checkpoint,
                runtime=runtime,
                workers=args.workers,
                prefetch=not args.no_prefetch,
                sampler_seed=args.seed,
            ) as executor:
                report = executor.checkpoint_report()
                if report["total_bytes"] > 1024 * 1024 * 1024:
                    print(
                        "warning: pure-Python decoding of a checkpoint this size is slow; "
                        "this path is for correctness, not throughput",
                        file=sys.stderr,
                    )
                result = executor.generate_text(
                    args.prompt,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                )
                _print({**result, "route": runtime.route(args.prompt).name, "stats": executor.stats.to_dict()})
            return 0
        except FileNotFoundError:
            print(f"generate: no such checkpoint: {args.checkpoint}", file=sys.stderr)
            return 1
        except UnsupportedTensorType as exc:
            print(f"generate: {exc}", file=sys.stderr)
            return 3
        except (ValueError, KeyError, MemoryError, RuntimeError) as exc:
            print(f"generate: {exc}", file=sys.stderr)
            return 1

    print("unknown command", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
