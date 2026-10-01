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
- Pluggable kernels, including a future T-SAR-inspired ternary CPU backend.

## Status

The repository contains the initial runtime architecture and executable CLI scaffold. Model adapters and optimized kernels are intentionally pluggable so the runtime can evolve without coupling the scheduler to one model family.

## Privacy

Pyrite does not require a server for inference. No telemetry or remote inference is part of the core runtime. Applications embedding Pyrite must not add network logging if strict local-only operation is desired.

## Research basis

The design tracks research directions including AQLM, sparse/structure-aware compression, MoE expert offloading, predictive prefetching, KV-cache offloading, and T-SAR CPU-only ternary inference.

This is research software: measured performance and model quality must be established on each target device/model combination.
