# Pyrite

Pyrite is a privacy-first local AI runtime focused on running capable open-weight language models on memory-constrained devices.

## Goals

- 4 GB RAM first-class target.
- Fully local inference by default.
- Explicit memory budgeting and graceful degradation.
- Layer/expert streaming from local storage.
- Quantization-aware execution.
- Adaptive routing and prefetch scheduling.
- KV-cache budgeting.
- CPU-first execution, with optional accelerators.
- Pluggable kernels, including a portable T-SAR-inspired ternary reference kernel.

## Status

The package contains an executable runtime, not a scaffold:

- GGUF and Safetensors adapters with a tensor index and bounded block/range reads.
- A strict Qwen3-MoE checkpoint contract (`pyrite qwen3-check`).
- A reference streaming executor with real quantized decoding and KV accounting.
- A byte-level BPE tokenizer loaded from GGUF metadata.
- Bounded caches, an I/O/compute scheduler, prefetch prediction, a content-addressed
  local block store, and KV object codecs.
- A CLI (`status`, `route`, `types`, `qwen3-check`, `inspect`, `pages`, `tokenize`,
  `generate`) plus a benchmark entry point.

The executor is a correctness reference: it runs real checkpoints with a bounded
resident budget, but its CPU kernels are not throughput-optimized. Model adapters
and kernels stay pluggable so the runtime can evolve without coupling the
scheduler to one model family.

## Privacy

Pyrite does not require a server for inference. No telemetry or remote inference is part of the core runtime. Applications embedding Pyrite must not add network logging if strict local-only operation is desired.

## Research basis

The design tracks research directions including AQLM, sparse/structure-aware compression, MoE expert offloading, predictive prefetching, KV-cache offloading, and T-SAR CPU-only ternary inference.

This is research software: measured performance and model quality must be established on each target device/model combination.
