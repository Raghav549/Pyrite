# Pyrite package

Pyrite is a privacy-first local AI runtime focused on running open-weight
language models on memory-constrained devices.

## Runtime capabilities

- GGUF and Safetensors adapters with tensor indexing, strict GGUF row-size
  validation, and bounded byte-range reads.
- Streaming CPU execution for dense Qwen3, Llama, and Qwen3-MoE GGUF checkpoints.
- A canonical Qwen3-MoE contract check (94 layers, 128 experts, top-8), with
  `qwen3-check --full` streaming-readiness reporting.
- Selected-expert routing and SwiGLU execution without materializing stacked
  expert tensors.
- Compact fp32 KV tensors with a stop-at-capacity or sliding-window eviction
  policy.
- A GGUF byte-level BPE tokenizer, deterministic/sampled autoregressive
  generation, and CPU checkpoint benchmarking.
- Optional scalar C Q4_K/Q6_K matvec kernels; Python reference decoding is
  retained for other types and compiler-less installs.
- Byte-capped caches, RSS monitoring, prefetch prediction, I/O scheduling,
  content-addressed local pages, and a T-SAR-inspired ternary reference kernel.

The model format and tensor types are validated before execution. Unsupported
architectures or quantizations are refused rather than approximated. Native
Q4_K/Q6_K execution is optional and correctness-tested on quantized fixtures;
no published-checkpoint throughput or 4-GB residency result is claimed yet.
See [`../docs/validation.md`](../docs/validation.md) for the measured checks and
open validation gaps.

## Privacy

Inference defaults to offline operation. Pyrite's core does not require a
server, telemetry, or remote inference. Applications embedding Pyrite must also
avoid adding network logging when strict local-only operation is required.
