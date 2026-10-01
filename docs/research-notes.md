# Pyrite research notes

## Architecture locks

Pyrite treats model capacity and resident memory as separate resources.

- Local storage is the cold model tier.
- RAM is the bounded hot working set.
- Prefetch and caching hide some storage latency.
- Model blocks and experts are explicit scheduling units.
- KV cache has its own memory policy.
- CPU kernels are first-class; accelerators are optional.
- Offline inference is the default privacy mode.

## Adaptive KV precision

Recent 2026 work proposes adaptive KV-cache bit allocation from token importance rather than one fixed precision. Pyrite includes a byte-aware 2, 4, 8 and 16-bit policy.

Paper: https://arxiv.org/abs/2604.04722

KVTC studies transform coding, decorrelation, adaptive quantization and entropy coding for compact reusable KV storage.

Paper: https://arxiv.org/abs/2511.01815

## Extreme weight compression

AQLM studies additive multi-codebook quantization and reports sub-3-bit compression with CPU kernels.

Paper: https://arxiv.org/abs/2401.06118

## CPU ternary execution

T-SAR studies CPU-only ternary inference with in-register SIMD lookup-table generation. Pyrite exposes a portable ternary reference kernel and native-kernel extension point.

Paper: https://arxiv.org/abs/2511.13676

## Adaptive computation depth

AdaSkip studies token/context-dependent sublayer skipping for long-context inference. FlexiDepth studies adaptive layer depth using a plug-in router and adapter, showing that different token types can need different computational depth.

Papers:
- https://arxiv.org/abs/2501.02336
- https://arxiv.org/abs/2503.23798

2026 analysis also finds diminishing early-exit returns in newer LLMs, so Pyrite treats adaptive depth as a learned, benchmark-gated policy rather than a guaranteed optimization.

Paper: https://arxiv.org/abs/2603.23701

## MoE routing

MoE research motivates expert-aware working sets, predictive loading, async I/O, and cache retention. Pyrite includes explicit expert routing plus local block streaming.

## Storage

SSD is not RAM. Large files can fit locally while repeated reads still become the latency and energy bottleneck. RAM locality and prefetch accuracy therefore remain first-class targets.

## Validation

No speed, memory, energy, model-quality, or 4 GB claim is accepted without measurement on the target device and checkpoint.
