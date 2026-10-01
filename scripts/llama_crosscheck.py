#!/usr/bin/env python3
"""Cross-validate Pyrite against official llama.cpp release binaries.

This script needs ``llama-quantize`` and ``llama-perplexity`` (any recent
llama.cpp release) plus the ``dev`` extra (``gguf`` and ``numpy``) as the
decode oracle.  Pass ``--llama-bin`` pointing at the directory with the
binaries, or put them on ``PATH``.

What it checks, on real GGUF files built from scratch in a temp directory:

1. The official quantizer accepts Pyrite's fixture files, for every pure
   quantization Pyrite supports plus the ``Q4_K_M`` mix.
2. Pyrite parses each quantized file and its decoders match the independent
   ``gguf`` dequantizer to 1e-6 on official-quantizer bytes.
3. Pyrite generates deterministically on each file (finite logits, valid ids,
   expert slices actually streamed).
4. ``llama-perplexity`` and Pyrite agree on PPL of a fixed text (within 1%;
   measured agreement is ~1e-4 relative).

Exit status is 0 only if every check passes.  A JSON report is written to
``--report`` (default: ``crosscheck_report.json`` in the work directory).
"""
from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from pyrite.adapters.gguf import GGUFReader
from pyrite.config import RuntimeConfig
from pyrite.engine import PyriteRuntime
from pyrite.executor import detect_architecture, open_executor
from pyrite.ggml_types import spec as ggml_spec
from pyrite.tensor_ops import decode_vector, softmax
from tests.tiny_dense import (
    TinyDenseConfig,
    build_tiny_dense_checkpoint,
    qwen3_dense_config,
)
from tests.tiny_qwen3moe import TinyConfig, build_tiny_checkpoint

PPL_TEXT = "hello world " * 100
PPL_TOLERANCE = 0.01  # relative; measured agreement is ~1e-4
PARITY_TOLERANCE = 1e-6  # absolute per-weight decode parity

TINY_QUANTS = ["Q4_0", "Q4_1", "Q5_0", "Q5_1", "Q8_0", "Q4_K_M"]
K_QUANTS = ["Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q6_K", "Q4_K_M"]


def _find_binary(name: str, llama_bin: Path | None) -> Path:
    if llama_bin is not None:
        candidate = llama_bin / name
        if candidate.is_file():
            return candidate
        raise FileNotFoundError(f"{name} not found in {llama_bin}")
    found = shutil.which(name)
    if found is None:
        raise FileNotFoundError(
            f"{name} not on PATH; download a llama.cpp release or pass --llama-bin"
        )
    return Path(found)


def _run(argv: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603 - argv comes from our own CLI flags
        argv, capture_output=True, text=True, check=False
    )


def _runtime() -> PyriteRuntime:
    return PyriteRuntime(RuntimeConfig(ram_budget_mb=1024, reserve_mb=768))


def _k_config() -> TinyConfig:
    return TinyConfig(
        hidden_size=256,
        num_hidden_layers=1,
        num_attention_heads=8,
        num_key_value_heads=8,
        head_dim=64,
        moe_intermediate_size=256,
        num_experts=2,
        num_experts_per_tok=2,
        vocab_size=271,
    )


def decode_parity(path: Path) -> dict[str, float]:
    """Worst |Pyrite - gguf| per tensor type on ``path``."""
    import numpy as np
    from gguf import quants

    reader = GGUFReader(path)
    worst: dict[str, float] = {}
    for tensor in reader.tensor_index():
        name = ggml_spec(tensor.ggml_type).name
        payload = reader.read_tensor(tensor)
        qtype = getattr(quants.GGMLQuantizationType, name, None)
        if qtype is None:  # pragma: no cover - only for exotic fallback types
            continue
        if name == "F32":
            expected = np.frombuffer(payload, dtype=np.float32).tolist()
        elif name == "F16":
            expected = np.frombuffer(payload, dtype=np.float16).astype(np.float64).tolist()
        else:
            expected = list(quants.dequantize(np.frombuffer(payload, dtype=np.uint8).copy(), qtype))
        got = decode_vector(tensor.ggml_type, payload, tensor.element_count)
        diff = max((abs(a - b) for a, b in zip(got, expected, strict=True)), default=0.0)
        worst[name] = max(worst.get(name, 0.0), diff)
    return worst


def generation_check(path: Path) -> dict[str, object]:
    """Deterministic generation plus streaming evidence on ``path``."""
    with open_executor(path, runtime=_runtime(), sampler_seed=0) as executor:
        unsupported = list(executor.unsupported_tensor_types())
        logits = executor.logits(3, 0)
        first = executor.generate([5, 9, 11], max_new_tokens=3, temperature=0.0)
        second = executor.generate([5, 9, 11], max_new_tokens=3, temperature=0.0)
        stats = executor.stats.to_dict()
    vocab = GGUFReader(path)
    size = next(t for t in vocab.tensor_index() if t.name == "token_embd.weight").dims[1]
    return {
        "unsupported_tensor_types": unsupported,
        "logits_finite": all(math.isfinite(v) for v in logits),
        "deterministic": first == second,
        "ids_valid": all(0 <= token < size for token in first),
        "generated_ids": first,
        "layers_executed": stats["layers_executed"],
        "tensors_streamed": stats["tensors_streamed"],
        "expert_slices_streamed": stats["expert_slices_streamed"],
        "bytes_loaded": stats["bytes_loaded"],
    }


def pyrite_ppl(path: Path) -> tuple[float, int]:
    with open_executor(path, runtime=_runtime()) as executor:
        ids = executor.tokenizer.encode(PPL_TEXT)
        executor.reset_kv()
        nll = 0.0
        for position in range(len(ids) - 1):
            probs = softmax(executor.logits(ids[position], position))
            nll += -math.log(max(probs[ids[position + 1]], 1e-300))
    return math.exp(nll / (len(ids) - 1)), len(ids)


def llama_ppl(binary: Path, path: Path, text_file: Path) -> float:
    completed = _run(
        [str(binary), "-m", str(path), "-f", str(text_file), "-c", "128",
         "--no-warmup", "-t", "1"]
    )
    output = completed.stdout + completed.stderr
    match = re.search(r"Final estimate: PPL = ([0-9.]+)", output)
    if completed.returncode != 0 or match is None:
        raise RuntimeError(f"llama-perplexity failed on {path.name}:\n{output[-2000:]}")
    return float(match.group(1))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llama-bin", type=Path, default=None,
                        help="directory with llama-quantize/llama-perplexity")
    parser.add_argument("--work-dir", type=Path, default=None,
                        help="keep intermediate GGUFs here (default: temp dir)")
    parser.add_argument("--report", type=Path, default=None,
                        help="write the JSON report here")
    args = parser.parse_args(argv)

    try:
        quantize = _find_binary("llama-quantize", args.llama_bin)
        perplexity = _find_binary("llama-perplexity", args.llama_bin)
    except FileNotFoundError as exc:
        print(f"crosscheck: {exc}", file=sys.stderr)
        return 2
    try:
        import gguf  # noqa: F401
        import numpy  # noqa: F401
    except ImportError:
        print("crosscheck: the dev extra (gguf, numpy) is required", file=sys.stderr)
        return 2

    work = args.work_dir or Path(tempfile.mkdtemp(prefix="pyrite-crosscheck-"))
    work.mkdir(parents=True, exist_ok=True)
    report: dict[str, object] = {"files": [], "ppl": [], "ok": True}
    failures: list[str] = []

    def fail(message: str) -> None:
        failures.append(message)
        report["ok"] = False
        print(f"FAIL: {message}", file=sys.stderr)

    fixtures = [
        ("tiny", build_tiny_checkpoint, None, TINY_QUANTS),
        ("k_dims", build_tiny_checkpoint, _k_config(), K_QUANTS),
        ("dense_llama", build_tiny_dense_checkpoint, TinyDenseConfig(), TINY_QUANTS),
        ("dense_qwen3", build_tiny_dense_checkpoint, qwen3_dense_config(), TINY_QUANTS),
    ]
    for label, builder, config, quants in fixtures:
        f32 = work / f"{label}_f32.gguf"
        builder(f32, config) if config is not None else builder(f32)
        targets = [("F32", f32)]
        for quant in quants:
            out = work / f"{label}_{quant.lower()}.gguf"
            completed = _run([str(quantize), str(f32), str(out), quant, "2"])
            if completed.returncode != 0 or not out.is_file():
                fail(f"{label}: llama-quantize {quant} failed:\n"
                     f"{completed.stdout[-1000:]}{completed.stderr[-1000:]}")
                continue
            targets.append((quant, out))
        for quant, path in targets:
            entry: dict[str, object] = {
                "fixture": label, "quant": quant,
                "bytes": path.stat().st_size, "path": str(path),
            }
            try:
                parity = decode_parity(path)
            except Exception as exc:
                fail(f"{label}/{quant}: parity error: {exc}")
                continue
            entry["decode_parity_worst"] = parity
            for name, diff in parity.items():
                if diff > PARITY_TOLERANCE:
                    fail(f"{label}/{quant}: {name} parity {diff:.3e} exceeds {PARITY_TOLERANCE}")
            try:
                gen = generation_check(path)
            except Exception as exc:
                fail(f"{label}/{quant}: generation error: {exc}")
                continue
            entry["generation"] = gen
            if gen["unsupported_tensor_types"]:
                fail(f"{label}/{quant}: unsupported types {gen['unsupported_tensor_types']}")
            for key in ("logits_finite", "deterministic", "ids_valid"):
                if not gen[key]:
                    fail(f"{label}/{quant}: generation check {key} failed")
            if detect_architecture(path) == "qwen3moe":
                if not gen["expert_slices_streamed"]:
                    fail(f"{label}/{quant}: no expert slices were streamed")
            elif not gen["layers_executed"] or not gen["tensors_streamed"]:
                fail(f"{label}/{quant}: dense run streamed nothing")
            report["files"].append(entry)
            print(f"ok: {label}/{quant} "
                  f"parity={max(parity.values()) if parity else 0:.1e} "
                  f"gen={gen['generated_ids']} slices={gen['expert_slices_streamed']}")

    text_file = work / "ppl.txt"
    text_file.write_text(PPL_TEXT, encoding="utf-8")
    for label, quant in (
        ("tiny", "F32"),
        ("tiny", "Q4_K_M"),
        ("dense_llama", "F32"),
        ("dense_qwen3", "F32"),
        ("dense_qwen3", "Q4_K_M"),
    ):
        path = work / f"{label}_{quant.lower()}.gguf"
        if not path.is_file():
            continue
        try:
            reference = llama_ppl(perplexity, path, text_file)
            measured, tokens = pyrite_ppl(path)
        except Exception as exc:
            fail(f"{label}/{quant}: PPL error: {exc}")
            continue
        rel = abs(measured - reference) / reference
        report["ppl"].append({
            "fixture": label, "quant": quant, "tokens": tokens,
            "llama_ppl": reference, "pyrite_ppl": measured, "relative_diff": rel,
        })
        print(f"ppl: {label}/{quant} llama={reference:.4f} pyrite={measured:.4f} rel={rel:.2e}")
        if rel > PPL_TOLERANCE:
            fail(f"{label}/{quant}: PPL relative diff {rel:.3e} exceeds {PPL_TOLERANCE}")

    report_path = args.report or (work / "crosscheck_report.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"report: {report_path}")
    if failures:
        print(f"{len(failures)} check(s) failed", file=sys.stderr)
        return 1
    print(f"all crosschecks passed ({len(report['files'])} files, "
          f"{len(report['ppl'])} PPL comparisons)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
