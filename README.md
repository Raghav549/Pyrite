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

## Development

```bash
python -m pyrite status
python -m pyrite route "write a Python API"
python -m pyrite.bench
python -m pytest
```