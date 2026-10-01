# Pyrite research-derived systems directions

This document records ideas that are implemented as explicit primitives or marked as research hooks. Published ideas are not presented as novel inventions.

## Storage-resident inference

Recent work on weight streaming demonstrates that when a model exceeds RAM, random-access page faults can become the dominant cost; sequential NVMe specifications can dramatically overstate effective inference bandwidth. Pyrite therefore measures resident bytes, I/O bytes, and execution timing instead of treating SSD capacity as RAM.

Reference implementation study:
https://github.com/i-mrDed/weight-streaming

## Timing-aware cache staging

TempoKV separates knowledge that a reusable KV object exists from the moment when fast-tier residency should be committed. Pyrite's future KV staging policy can apply the same principle: reserve scarce RAM close to predicted use instead of immediately on cache discovery.

arXiv: https://arxiv.org/abs/2609.35065

## Agent-semantic cache prediction

CacheScout argues that KV reuse can depend on learned execution transitions between agents, not just LRU recency. Pyrite's transition-based prefetch predictor is a lightweight local analogue; an agent-aware policy can be layered on later without changing the storage interface.

## Cross-model KV reuse

DroidSpeak studies reuse of KV caches across architecturally compatible LLMs with selective layer recomputation. Pyrite's prefix cache is model-local today; the adapter boundary leaves room for architecture-hash and layer-subset validation before cross-model reuse.

arXiv: https://arxiv.org/abs/2411.02820

## Non-prefix KV reuse

Adaptive non-prefix KV reuse work shows that semantic reuse requires restoring cross-chunk attention relationships and jointly optimizing compute with cache I/O. Pyrite should treat arbitrary chunk reuse as a transformation with verification, not as a blind byte splice.

## Diffusion-LM scheduling

Sangam shows that diffusion language models change the execution unit from one autoregressive token to blocks of denoising positions. Pyrite's scheduler is intentionally block-oriented, so a future dLLM backend can reuse the residency/I/O scheduling layer while replacing token-step semantics.

arXiv: https://arxiv.org/abs/2607.04206

## Design rule

A Pyrite optimization is only accepted as a runtime truth after measurement on a real checkpoint and target device. Research results describe their evaluated systems and hardware; they do not automatically transfer to Pyrite's 4 GB target.


## New research leads (2026)

- DUAL-BLADE explores choosing a page-cache path versus direct NVMe path for KV residency under memory pressure. Pyrite remains local/offline and filesystem-based for portability, but its I/O scheduler is now structured so a direct-I/O backend can be added without changing model execution.
  https://arxiv.org/abs/2604.26557

- SpecPrefetch uses a small shared adapter only to predict candidate MoE experts while the frozen native router keeps the final routing decision. This is a useful design for Pyrite: prediction may be approximate, but execution correctness must remain authoritative.
  https://arxiv.org/abs/2607.24787

- CacheTune combines sparse KV transfer, selective recomputation, asynchronous I/O and hardware-aware recomputation ratios for non-prefix reuse. Pyrite's reusable KV layer is intentionally separated from its compression and staging layers so these can be composed and benchmarked.
  https://arxiv.org/abs/2605.24022

- Cross-model KV transfer work suggests model-family-compatible KV representations can sometimes be mapped rather than recomputed. Pyrite should only permit this behind explicit architecture fingerprints and a validation gate; cache bytes must never be interpreted across incompatible models.
  https://arxiv.org/abs/2608.03893

- Current storage-streaming measurements emphasize that random mmap/page-fault access can make effective disk bandwidth far below sequential SSD specifications. Pyrite therefore treats observed I/O latency/bytes as a first-class signal and never equates capacity with RAM.
