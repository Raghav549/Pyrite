# Pyrite research notes

Pyrite combines several published research directions into one experimental
runtime. These notes distinguish a research idea from an implemented feature.

## AQLM

AQLM uses additive/multi-codebook quantization to push language-model weight
compression below the usual 4-bit range while preserving model behavior.
Pyrite therefore treats quantization as a model-adapter concern rather than
hard-coding one format.

Paper: https://arxiv.org/abs/2401.06118

## MoE offloading and prefetch

MoE models separate total parameter capacity from the active experts used for
a token. Recent work studies expert retention, prediction, CPU cooperation,
and prefetching to reduce memory pressure.

SPICE: https://arxiv.org/abs/2608.21240
OLED-MoE: https://arxiv.org/abs/2609.33385

Pyrite currently contains a small locality predictor and bounded cache. It is
not yet a learned expert router.

## SSD offload caveat

SSD capacity solves storage capacity, not free compute. Recent analysis warns
that repeated SSD reads can dominate energy during decode. Therefore Pyrite
uses SSD as a cold tier and treats RAM caching and prefetch correctness as
first-class optimization targets.

Paper: https://arxiv.org/abs/2508.06978

## T-SAR

T-SAR explores CPU-only ternary inference through in-register SIMD lookup-table
construction. Its reported gains come from a hardware/software co-design.

Paper: https://arxiv.org/abs/2511.13676

Pyrite includes a portable ternary reference interface. It deliberately does
not claim to reproduce T-SAR hardware or performance. The next native backend
should target common x86-64 SIMD instructions first, then optional ARM NEON.

## What remains to validate

A real model adapter must be benchmarked for:
- peak RSS under the 4 GB profile
- storage read amplification
- tokens per second
- first-token latency
- cache hit rate
- KV growth
- quality/perplexity after quantization and sparsity
- CPU energy per generated token

The 4 GB target is a measured engineering constraint, not a guarantee that
every multi-billion-parameter model will run smoothly.
