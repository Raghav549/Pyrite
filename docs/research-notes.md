# Pyrite research notes

## Locked architecture

Pyrite treats model capacity and resident memory as separate resources.

- SSD or local storage is the cold model tier.
- RAM is the bounded hot working set.
- Prefetch and caching are used to overlap data movement.
- Model blocks and experts are explicit scheduling units.
- KV cache has its own memory policy.
- CPU kernels are first-class; accelerator use is optional.
- Offline inference is the default privacy mode.

## Adaptive KV precision

Recent 2026 work proposes adaptive KV-cache bit allocation based on token importance instead of one fixed precision. Pyrite now includes a byte-aware policy with 2, 4, 8 and 16-bit token tiers.

Paper: https://arxiv.org/abs/2604.04722

KVTC reports transform coding, decorrelation, adaptive quantization, and entropy coding for compact KV storage. It motivates a future compressed KV storage backend for Pyrite.

Paper: https://arxiv.org/abs/2511.01815

## Extreme weight compression

AQLM studies additive multi-codebook quantization and reports sub-3-bit compression with CPU kernels. Pyrite keeps low-bit formats and CPU execution as separate pluggable layers so format-specific kernels can be optimized independently.

Paper: https://arxiv.org/abs/2401.06118

## CPU ternary execution

T-SAR explores CPU-only ternary inference using in-register SIMD lookup-table generation. Pyrite includes a portable ternary reference kernel and a future native SIMD extension point.

Paper: https://arxiv.org/abs/2511.13676

## MoE execution

Research on MoE offloading and prefetching motivates expert-aware working sets, predictive loading, asynchronous I/O, and caching. Pyrite has an explicit expert router and block streamer; model-specific learned predictors remain a future optimization.

## Storage and I/O

Storage capacity does not equal compute capacity. Large model files can fit on disk while repeated reads become the dominant latency and energy cost. Pyrite therefore treats storage as a cold tier and prioritizes RAM locality and prefetch accuracy.

## Validation rule

No model-size, latency, RAM, energy, or quality claim is considered proven until measured on the target device and checkpoint. A large model can be resident on disk while still being impractically slow on a 4 GB machine.

## Privacy

The core runtime does not require a data center for inference and defaults to offline operation. Host applications must not add telemetry, cloud sync, or remote inference when strict local-only operation is required.
