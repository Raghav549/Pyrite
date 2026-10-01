# Pyrite research notes

## Design lock: SSD is a cold tier

Pyrite does not treat SSD as equivalent to RAM. The runtime uses storage for model capacity and cold blocks, while RAM is the hot working set. Prefetch and eviction are used to hide some storage latency, but storage bandwidth remains a hard limit.

Research supporting this design includes HeteGen, which studies heterogeneous CPU and GPU execution with asynchronous overlap to mitigate I/O bottlenecks, and SP-MoE, which uses speculative expert prefetching, cutoff policies, asynchronous prefetch threads, and batched I/O for MoE inference.

HeteGen: https://arxiv.org/abs/2403.01164
SP-MoE: https://arxiv.org/abs/2510.10302

## T-SAR

T-SAR studies CPU-only ternary LLM inference using in-register SIMD lookup-table generation. It reports large GEMM and GEMV improvements in its hardware/software co-design. Pyrite therefore keeps ternary inference as a first-class native-kernel extension point.

T-SAR: https://arxiv.org/abs/2511.13676

## KV cache

KV cache can become a dominant memory cost for long contexts. CLO studies CPU-light KV-cache offloading plus prefetching and persistent caching. Pyrite therefore has an explicit KV budget instead of allowing context growth to consume unbounded memory.

CLO: https://arxiv.org/abs/2511.14510

## MoE routing and on-demand loading

SP-MoE and OD-MoE motivate future learned expert prediction and just-in-time expert loading. Pyrite's current prefetch predictor is intentionally lightweight; a model-specific predictor can replace it later.

OD-MoE: https://arxiv.org/abs/2512.03927

## Compression and sparsity

AQLM motivates aggressive low-bit weight compression. Structured and non-uniform sparsity research motivates allocating precision and pruning unevenly across layers instead of applying one global ratio.

AQLM: https://arxiv.org/abs/2401.06118

## Engineering rule

All performance claims must come from measurements on the target device. A large model can fit on local storage while still being too slow because of compute or storage bandwidth. Pyrite therefore optimizes capacity, residency, data movement, and computation together.

## Privacy

Offline inference is the default. The runtime does not require a data center for model execution. Host applications must not add telemetry, cloud sync, or remote inference when strict local-only operation is required.