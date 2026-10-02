# Pyrite

Pyrite is a research runtime for private, local AI on memory-constrained devices.

## Core idea

Separate model capacity from resident memory. GGUF tensors and MoE expert slices are read on demand from local storage; only a byte-capped working set, compact KV state, and the current activations are kept in memory.

## Target

First-class target: laptops with about 4 GB RAM. The default profile sets a 4096 MiB process-RSS ceiling (enforced where the OS exposes RSS), reserves 768 MiB for the interpreter and transient allocations, and budgets the remaining 3328 MiB for managed runtime state. Managed cache/KV budgets are enforced independently; a missing OS RSS counter is reported as unknown, not treated as zero usage. These limits are not a promise that every model or context fits. Lower `PYRITE_KV_TOKENS` when the KV cache cannot fit.

## Architecture

- bounded, byte-accounted cache with LRU eviction and range reads
- local block and shard store, content-addressed paging, and offline-first policy
- route selection, execution planning, and predictive prefetch hooks
- bounded compact KV cache with `stop` and `sliding_window` policies
- GGUF adapters with per-row quantized-size validation
- CPU transformer execution for supported dense and MoE architectures
- optional portable C Q4_K/Q6_K streaming kernels; Python reference decoders remain available
- T-SAR-inspired ternary reference kernel (not the proposed hardware modification)

## Supported architectures

Pyrite dispatches on GGUF `general.architecture`:

- `qwen3moe` — Qwen3-MoE GQA, per-head QK norm, softmax router, top-k SwiGLU experts streamed from stacked expert tensors
- `qwen3` — dense Qwen3 with GQA, QK norm, and dense SwiGLU MLP
- `llama` — dense Llama with GQA, optional partial rotary dimensions, and dense SwiGLU MLP

Unknown architectures and unsupported tensor types are refused rather than approximated. `qwen3-check --require-canonical` additionally checks the canonical Qwen3-MoE metadata contract of 94 layers, 128 experts, and top-8 routing.

## Qwen3 checkpoint validation

```bash
python -m pyrite qwen3-check /path/to/model.gguf
python -m pyrite qwen3-check /path/to/model.gguf --full
python -m pyrite qwen3-check /path/to/model.gguf --require-canonical
```

The full report includes tensor sizes, the largest streamed row or expert slice, and whether those ranges fit the current budget. For the Qwen3-235B-A22B-Instruct-2507 GGUF family, Q4_K_M is published as five split files; use `llama-gguf-split` to merge them before passing a single GGUF file to Pyrite. The full Q4_K_M set is roughly 142 GB. A 4 GB setting bounds resident memory; it does not make the full model fit on disk or make a large KV context affordable.

## Local generation and benchmarking

```bash
python -m pyrite generate model.gguf --prompt "hello" --max-new-tokens 16 --temperature 0
python -m pyrite generate model.gguf --prompt "hello" --kv-cache-policy sliding_window
python -m pyrite tokenize model.gguf "hello world"
python -m pyrite.bench --checkpoint model.gguf --tokens 8
```

The executor streams dense projection rows, selected MoE expert slices, and vocabulary rows; it does not materialize the whole model. KV entries are stored as compact fp32 arrays. The default `stop` policy ends generation when the configured KV limit is reached; `sliding_window` evicts the oldest KV entries and continues up to the model/runtime context limit.

A C compiler (`cc`, `gcc`, or `clang`) enables the optional scalar C Q4_K/Q6_K kernels. Pyrite compiles the small, Python-header-free C source into the system temporary directory on first use, borrows quantized buffers without copying them, and fuses dequantization with matrix-vector multiplication. If compilation is unavailable, those types use the bounded Python reference path instead. This is not a throughput claim: transformer orchestration and non-Q4_K/Q6_K tensor types still use Python, and no published-checkpoint performance result is claimed here.

Reference decoders cover F32, F16, BF16, F64, I8/I16/I32/I64, Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/Q8_1, and Q2_K through Q6_K. No runtime oracle, cloud service, or network access is required for inference.

## Validation

`scripts/llama_crosscheck.py` can compare Pyrite with official `llama.cpp` binaries on locally generated GGUF fixtures (decoder parity, deterministic generation, and perplexity). The measured fixture results and remaining gaps are recorded in [`docs/validation.md`](docs/validation.md). A fixture crosscheck does not substitute for running a published checkpoint.

```bash
python scripts/llama_crosscheck.py --llama-bin /path/to/llama.cpp/build/bin
```

## Privacy

Inference defaults to offline operation. Prompts and model state are not sent to a data center by the runtime. Applications embedding Pyrite must also keep their own telemetry, cloud sync, and remote tools disabled when strict local-only operation is required.

## Development

The `dev` extra installs the optional GGUF and NumPy oracles used by decoder-parity tests; these are test-only dependencies.

```bash
python -m pip install -e ".[dev]"
python -m pyrite status
python -m pyrite route "write a Python API"
python -m pyrite.bench
python -m ruff check pyrite tests scripts setup.py
python -m compileall -q pyrite tests scripts
python -m pytest -q
```
