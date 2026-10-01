# Pyrite

Pyrite is a research runtime for private, local AI on memory-constrained devices.

## Core idea

Separate model capacity from resident memory. Model blocks or experts may live in local storage while the runtime keeps a bounded working set in RAM and schedules the next blocks.

## Target

First-class target: laptops with about 4 GB RAM. The budget is configurable because real memory and speed depend on the operating system, model, storage, context length, and CPU.

## Architecture

- bounded resident cache
- local block and shard store
- route selection
- execution planning
- predictive prefetch hooks
- KV-cache budgeting
- offline-first policy
- pluggable model adapters
- pluggable CPU kernels
- T-SAR-inspired ternary kernel extension

## Research basis

Pyrite tracks research directions in extreme quantization, structured sparsity, MoE expert offloading, predictive prefetching, KV-cache management, and CPU-only ternary inference.

- AQLM: https://arxiv.org/abs/2401.06118
- T-SAR: https://arxiv.org/abs/2511.13676
- SPICE: https://arxiv.org/abs/2608.21240
- OLED-MoE: https://arxiv.org/abs/2609.33385

## Privacy

The core runtime has no network dependency for inference and defaults to offline operation. Prompts and model state are not sent to a data center by the runtime.

Applications embedding Pyrite must keep telemetry, cloud sync, and remote tools disabled when strict local-only operation is required.

## Supported architectures

Pyrite executes three GGUF architectures and dispatches on the file's
`general.architecture` automatically:

- `qwen3moe` — Qwen3-MoE (GQA + QK-norm attention, top-k SwiGLU experts
  streamed as per-expert slices)
- `qwen3` — dense decoder-only Qwen3 (same attention core, dense SwiGLU MLP)
- `llama` — dense decoder-only Llama (GQA without QK-norm, partial rotary
  via `rope_dimension_count`, dense SwiGLU MLP)

Unknown architectures are refused with an explicit error instead of being
mis-executed.

## Qwen3-MoE checkpoint validation

Pyrite includes a strict GGUF checkpoint contract for Qwen3-MoE (`qwen3moe`), including architecture metadata parsing, expert count/top-k discovery, tensor-index validation, and bounded streaming readiness checks.

Validate a checkpoint locally:

```bash
python -m pyrite qwen3-check /path/to/Qwen3-235B-A22B-Q4_K_M.gguf
```

For the current Qwen3-235B-A22B-Instruct-2507 GGUF family, Q4_K_M is published as five split GGUF files. Merge the split files with `llama-gguf-split` before passing the resulting single GGUF path to Pyrite. The full Q4_K_M set is roughly 142 GB, while the runtime's 4 GB setting is a resident-memory budget, not a promise that the whole model fits in RAM.

## Native streaming generation

`pyrite generate` runs the checkpoint through the reference executor: GGUF tensors
are streamed (whole dense tensors, per-expert slices of the stacked MoE tensors),
decoded from their quantization with bounded chunks, and executed by a
memory-budgeted forward pass with an incremental KV cache. The KV footprint is
checked against the configured working set at startup, and generation stops at
the KV token budget instead of silently discarding context.

```bash
python -m pyrite generate model.gguf --prompt "hello" --max-new-tokens 16 --temperature 0
python -m pyrite tokenize model.gguf "hello world"
python -m pyrite.bench --checkpoint model.gguf --tokens 8
```

Reference decoders are available for F32, F16, BF16, F64, I8/I16/I32/I64,
Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/Q8_1 and Q2_K..Q6_K. Checkpoints that use other
quantizations are reported and refused rather than approximated. This path is a
correctness reference written in pure Python: it is not a throughput-optimized
kernel, and decoding a 100+ GB checkpoint token-by-token on a CPU will be slow.

## Validation

`scripts/llama_crosscheck.py` cross-validates Pyrite against official
`llama.cpp` binaries (official quantizer acceptance, exact decode parity,
deterministic generation, and PPL agreement to ~1e-5). Measured numbers,
greedy-parity token IDs, and the openly-unproven list live in
`docs/validation.md`.

```bash
python scripts/llama_crosscheck.py --llama-bin /path/to/llama.cpp/build/bin
```

## Development

The `dev` extra installs the optional GGUF and NumPy oracle used to check decoder parity; these are test-only and are not runtime dependencies.

```bash
python -m pip install -e ".[dev]"
python -m pyrite status
python -m pyrite route "write a Python API"
python -m pyrite.bench
python -m ruff check pyrite tests scripts
python -m pytest
```
