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


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("value must be an integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


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

    qwen = sub.add_parser("qwen3-check", help="validate Qwen3 GGUF metadata, tensors, and streaming readiness")
    qwen.add_argument("checkpoint")
    qwen.add_argument("--full", action="store_true", help="include the bounded streaming report")
    qwen.add_argument(
        "--require-canonical",
        action="store_true",
        help="require the canonical Qwen3-MoE contract: 94 layers, 128 experts, top-8",
    )

    pages = sub.add_parser("pages", help="content-address a checkpoint into local pages")
    pages.add_argument("checkpoint")
    pages.add_argument("--page-bytes", type=_positive_int, default=2 * 1024 * 1024)

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
    generate.add_argument(
        "--kv-cache-policy",
        choices=("stop", "sliding_window"),
        help="stop at KV capacity or evict oldest KV entries until the model context limit",
    )

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
    bench.add_argument(
        "--kv-cache-policy",
        choices=("stop", "sliding_window"),
        help="KV capacity policy for checkpoint inference",
    )

    return parser


def _config(args: argparse.Namespace) -> RuntimeConfig:
    config = RuntimeConfig.from_env()
    if args.profile:
        config = RuntimeConfig.from_profile(args.profile)
    if args.ram_mb is not None:
        config = config.with_overrides(ram_budget_mb=args.ram_mb)
    kv_policy = getattr(args, "kv_cache_policy", None)
    if kv_policy is not None:
        config = config.with_overrides(kv_cache_policy=kv_policy)
    if args.offline:
        config = config.with_overrides(offline=True)
    config.validate()
    return config


def _print(payload: object) -> None:
    print(json.dumps(payload, indent=2, default=str))


def _streaming_budget_bytes(config: RuntimeConfig, model_config) -> int:
    """Match the executor's KV/static-memory reservations for check reports."""
    import sys
    from array import array

    working = config.resident_byte_budget
    kv_bytes = (
        config.max_kv_tokens
        * model_config.num_hidden_layers
        * model_config.kv_cache_dim
        * 4
        + config.max_kv_tokens
        * model_config.num_hidden_layers
        * (2 * (sys.getsizeof(array("f")) + 8))
    )
    mib = 1024 * 1024
    small = min(32 * mib, max(1, working // 32))
    transient = min(64 * mib, max(16 * mib, working // 16))
    return max(0, working - kv_bytes - small - transient)


def _dense_streaming_report(model, resident_bytes: int) -> dict[str, object]:
    from .ggml_types import row_size

    tensors = model.reader.tensor_index()
    largest_tensor = max((tensor.size for tensor in tensors), default=0)
    largest_row = max(
        (row_size(tensor.dims[0], tensor.ggml_type) for tensor in tensors),
        default=0,
    )
    return {
        "resident_bytes": resident_bytes,
        "tensor_count": len(tensors),
        "total_bytes": sum(tensor.size for tensor in tensors),
        "oversized_tensors": sum(tensor.size > resident_bytes for tensor in tensors),
        "largest_tensor_bytes": largest_tensor,
        "largest_row_bytes": largest_row,
        "can_stream_storage": resident_bytes > 0 and largest_row <= resident_bytes,
        "whole_model_residency_required": False,
        "full_model_loaded": False,
    }


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
    except (OSError, ValueError, TypeError) as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2

    try:
        runtime = PyriteRuntime(config)
    except (OSError, ValueError, MemoryError, TypeError) as exc:
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
        return bench_main(forwarded, runtime_config=config)

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
            from .dense import DenseCheckpoint
            from .qwen3_moe import ARCHITECTURE as MOE_ARCH
            from .qwen3_moe import Qwen3MoECheckpoint

            reader = GGUFReader(Path(args.checkpoint))
            architecture = str(reader.metadata().get("general.architecture", ""))
            if architecture == MOE_ARCH:
                model = Qwen3MoECheckpoint(args.checkpoint, reader=reader)
                summary = model.validate_contract()
                if args.require_canonical and not model.config.is_canonical_qwen3_moe:
                    raise ValueError(
                        "checkpoint does not match canonical Qwen3-MoE: expected "
                        "94 layers, 128 experts, top-8"
                    )
                payload: dict[str, object] = {
                    "model": "Qwen3-MoE",
                    "architecture": architecture,
                    "config": model.config.to_dict(),
                    "routing": summary,
                }
                if args.full:
                    stream_budget = _streaming_budget_bytes(runtime.config, model.config)
                    payload["streaming"] = model.validate_for_streaming(stream_budget)
            elif architecture == "qwen3":
                if args.require_canonical:
                    raise ValueError("--require-canonical applies only to Qwen3-MoE checkpoints")
                model = DenseCheckpoint(args.checkpoint, reader=reader)
                summary = model.validate_contract()
                payload = {
                    "model": "Qwen3 dense",
                    "architecture": architecture,
                    "config": model.config.to_dict(),
                    "validation": summary,
                }
                if args.full:
                    stream_budget = _streaming_budget_bytes(runtime.config, model.config)
                    payload["streaming"] = _dense_streaming_report(model, stream_budget)
            else:
                raise ValueError(
                    f"qwen3-check requires a qwen3 or qwen3moe GGUF, got {architecture or 'unknown'}"
                )
            _print(payload)
            return 0
        except (OSError, ValueError, KeyError, TypeError, OverflowError, MemoryError) as exc:
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
        try:
            pager = ContentAddressedPager(
                config.storage_dir / "pages", page_bytes=args.page_bytes
            )
            pages = pager.ingest(source)
        except (OSError, ValueError, MemoryError) as exc:
            print(f"pages: {exc}", file=sys.stderr)
            return 1
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
        except (OSError, ValueError, TypeError) as exc:
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
                if report["total_bytes"] > 1024 * 1024 * 1024 and not report["native_kernel_available"]:
                    print(
                        "warning: native Q4_K/Q6_K kernels are unavailable; reference decoding "
                        "of this checkpoint will be slow",
                        file=sys.stderr,
                    )
                result = executor.generate_text(
                    args.prompt,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                )
                _print({
                    **result,
                    "route": runtime.route(args.prompt).name,
                    "backend": "cpu-native" if executor.stats.native_kernel_calls else "cpu-reference",
                    "checkpoint": report,
                    "stats": executor.stats.to_dict(),
                })
            return 0
        except FileNotFoundError:
            print(f"generate: no such checkpoint: {args.checkpoint}", file=sys.stderr)
            return 1
        except UnsupportedTensorType as exc:
            print(f"generate: {exc}", file=sys.stderr)
            return 3
        except (OSError, ValueError, KeyError, MemoryError, RuntimeError, TypeError, OverflowError) as exc:
            print(f"generate: {exc}", file=sys.stderr)
            return 1

    print("unknown command", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
