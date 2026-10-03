"""Cross-check Pyrite against the reference ``llama.cpp`` runtime.

Run after building llama.cpp (see ``docs/validation.md``)::

    python scripts/llama_reference_compare.py \\
        --llama-bin /path/to/llama.cpp/build/bin \\
        --vocab-gguf /path/to/ggml-vocab-qwen2.gguf \\
        --model-gguf /path/to/model.gguf

Two things are compared, both on the *same* files:

* **tokenizer ids** - ``llama-tokenize`` vs :class:`pyrite.tokenizer`, over a
  fixed corpus that includes ASCII, Hindi, mixed scripts, digits, punctuation
  and Qwen chat control tokens.
* **generation** - ``llama-completion`` vs the Pyrite executor on the same
  checkpoint with greedy decoding and the same prompt.

The script prints what it actually observed.  It never reports agreement it did
not measure.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pyrite.adapters.gguf import GGUFReader
from pyrite.config import RuntimeConfig
from pyrite.engine import PyriteRuntime
from pyrite.executor import open_executor
from pyrite.tokenizer import load_gguf_tokenizer

CORPUS: tuple[tuple[str, str], ...] = (
    ("ascii", "Explain Pyrite in one short paragraph."),
    ("hindi", "नमस्ते दुनिया"),
    ("mixed", "Pyrite एक local inference engine है"),
    ("digits", "There are 12345 items at 3.14 and ½ left."),
    ("punct", "Hello, world! (Is this working?) -- yes; ok."),
    ("whitespace", "  spaced   out\ttext\nwith newline  "),
    ("empty-adjacent", "a"),
    ("chat", "<|fim_prefix|>user\nHi<|im_end|>\n<|fim_prefix|>assistant\n"),
    ("long", " ".join(f"token{i}" for i in range(120))),
)


def llama_tokenize(binary: Path, model: Path, text: str) -> list[int]:
    command = [str(binary / "llama-tokenize"), "-m", str(model), "-p", text]
    # Byte-fallback tokens are echoed as raw bytes, so stdout is not always
    # valid UTF-8; decode permissively instead of crashing on the reference.
    raw = subprocess.run(  # noqa: S603 - fixed local binary and arguments
        command, capture_output=True, timeout=300
    )
    result = subprocess.CompletedProcess(
        command,
        raw.returncode,
        stdout=raw.stdout.decode("utf-8", errors="replace"),
        stderr=raw.stderr.decode("utf-8", errors="replace"),
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"llama-tokenize failed ({result.returncode}): {result.stderr.strip()[:300]}"
        )
    # llama-tokenize prints one "  <id> -> \'<text>\'" line per token.
    ids = [int(match.group(1)) for match in re.finditer(r"^\s*(\d+) -> ", result.stdout, re.M)]
    if not ids and result.stdout.strip():
        raise RuntimeError(f"could not find token list in llama-tokenize output: {result.stdout!r}")
    return ids


def compare_tokenizer(binary: Path, vocab_gguf: Path) -> dict[str, object]:
    reader = GGUFReader(vocab_gguf)
    tokenizer = load_gguf_tokenizer(reader)
    if tokenizer is None:
        raise RuntimeError(f"{vocab_gguf} does not carry a byte-level BPE tokenizer")
    rows = []
    mismatched = 0
    for name, text in CORPUS:
        reference = llama_tokenize(binary, vocab_gguf, text)
        mine = tokenizer.encode(text)
        agree = reference == mine
        mismatched += int(not agree)
        rows.append(
            {
                "case": name,
                "reference_ids": reference,
                "pyrite_ids": mine,
                "identical": agree,
                "first_difference": next(
                    (
                        index
                        for index, (a, b) in enumerate(zip(reference, mine, strict=False))
                        if a != b
                    ),
                    None if len(reference) == len(mine) else min(len(reference), len(mine)),
                ),
            }
        )
    round_trip_ok = all(
        tokenizer.decode(row["pyrite_ids"]) == text
        for row, (_name, text) in zip(rows, CORPUS, strict=True)
        if text.strip() and "<|" not in text
    )
    return {
        "tokenizer_file": str(vocab_gguf),
        "tokenizer_pre": tokenizer.pre,
        "vocab_size": tokenizer.vocab_size,
        "cases": len(rows),
        "identical_cases": len(rows) - mismatched,
        "round_trip_ok": round_trip_ok,
        "rows": rows,
    }


def llama_completion(
    binary: Path, model: Path, prompt: str, tokens: int, seed: int
) -> dict[str, object]:
    command = [
        str(binary / "llama-completion"),
        "-m", str(model),
        "-p", prompt,
        "-n", str(tokens),
        "--temp", "0",
        "-s", str(seed),
        "--no-conversation",
        "--no-display-prompt",
        "-t", "1",
        "-ngl", "0",
    ]
    raw = subprocess.run(  # noqa: S603 - fixed local binary and arguments
        command, capture_output=True, timeout=1800
    )
    result = subprocess.CompletedProcess(
        command,
        raw.returncode,
        stdout=raw.stdout.decode("utf-8", errors="replace"),
        stderr=raw.stderr.decode("utf-8", errors="replace"),
    )
    return {
        "command": command,
        "returncode": result.returncode,
        "stdout_tail": result.stdout[-800:],
        "stderr_tail": result.stderr[-800:],
    }


def compare_generation(binary: Path, model: Path, prompt: str, tokens: int) -> dict[str, object]:
    reference = llama_completion(binary, model, prompt, tokens, 42)
    report: dict[str, object] = {"reference": reference}
    config = RuntimeConfig.from_env().with_overrides(
        max_context_tokens=max(256, tokens + 64), max_kv_tokens=max(256, tokens + 64)
    )
    runtime = PyriteRuntime(config)
    with open_executor(model, runtime=runtime, sampler_seed=42) as executor:
        result = executor.generate_text(prompt, max_new_tokens=tokens, temperature=0.0)
        report["pyrite"] = {
            "completion": result["completion"],
            "prompt_tokens": result["prompt_tokens"],
            "new_tokens": result["new_tokens"],
            "output_ids": result["output_ids"][-tokens:],
            "lm_head": executor.checkpoint_report()["lm_head"],
        }
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llama-bin", required=True, type=Path)
    parser.add_argument("--vocab-gguf", type=Path)
    parser.add_argument("--model-gguf", type=Path)
    parser.add_argument("--prompt", default="Explain Pyrite in one short paragraph.")
    parser.add_argument("--tokens", type=int, default=8)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    payload: dict[str, object] = {}
    if args.vocab_gguf:
        payload["tokenizer"] = compare_tokenizer(args.llama_bin, args.vocab_gguf)
    if args.model_gguf:
        payload["generation"] = compare_generation(
            args.llama_bin, args.model_gguf, args.prompt, args.tokens
        )
    if args.json:
        print(json.dumps(payload, indent=2, default=str))
        return 0
    tokenizer = payload.get("tokenizer")
    if tokenizer:
        print(
            f"tokenizer: {tokenizer['tokenizer_file']} pre={tokenizer['tokenizer_pre']} "
            f"vocab={tokenizer['vocab_size']}"
        )
        for row in tokenizer["rows"]:
            marker = "IDENTICAL" if row["identical"] else "DIFFERS"
            print(
                f"  {row['case']:14s} {marker:9s} n_ref={len(row['reference_ids'])} "
                f"n_pyrite={len(row['pyrite_ids'])}"
            )
            if not row["identical"]:
                print(f"      reference: {row['reference_ids'][:16]}")
                print(f"      pyrite   : {row['pyrite_ids'][:16]}")
        print(
            f"  -> {tokenizer['identical_cases']}/{tokenizer['cases']} cases identical, "
            f"round_trip_ok={tokenizer['round_trip_ok']}"
        )
    generation = payload.get("generation")
    if generation:
        reference = generation["reference"]
        print(f"\ngeneration reference returncode={reference['returncode']}")
        print("  llama.cpp output tail:")
        for line in str(reference["stdout_tail"]).splitlines()[-6:]:
            print("   ", line)
        print("  pyrite:", json.dumps(generation["pyrite"], default=str)[:400])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
