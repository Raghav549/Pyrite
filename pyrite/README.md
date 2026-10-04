# Pyrite package

Pyrite is a privacy-first local AI runtime focused on running open-weight
language models on memory-constrained devices.

## Runtime capabilities

- GGUF and Safetensors adapters with tensor indexing, strict GGUF row-size
  validation, and bounded byte-range reads.
- Streaming CPU execution for dense Qwen3, Llama, and Qwen3-MoE GGUF checkpoints.
- `qwen3-check` reporting the resolved config, tensor registry, LM-head
  resolution and streaming readiness, with `--full` adding a real memory plan
  and an opt-in `--expect-profile` check against published Qwen3 shapes.
- Selected-expert routing and SwiGLU execution without materializing stacked
  expert tensors.
- Compact fp32 KV tensors with a stop-at-capacity or sliding-window eviction
  policy.
- A GGUF byte-level BPE tokenizer, deterministic/sampled autoregressive
  generation, and CPU checkpoint benchmarking.
- Optional scalar C matvec kernels for all 21 decodable GGML types; Python
  reference decoding is retained for compiler-less installs.
- Byte-capped caches, RSS monitoring, prefetch prediction, I/O scheduling,
  content-addressed local pages, and a T-SAR-inspired ternary reference kernel.

The model format and tensor types are validated before execution. Unsupported
architectures or quantizations are refused rather than approximated. Every
native kernel is cross-checked against its Python reference decoder in
`tests/test_native_kernels.py`.

Measured against a real `llama.cpp` build on the same GGUF files: token ids are
byte-identical on 9/9 corpus cases, and greedy generation agrees for 288
consecutive tokens across the dense and Qwen3-MoE architectures. See
[`../docs/validation.md`](../docs/validation.md) for the exact commands, the
numbers, and what is still unverified.

## Privacy

Inference defaults to offline operation. Pyrite's core does not require a
server, telemetry, or remote inference. Applications embedding Pyrite must also
avoid adding network logging when strict local-only operation is required.
