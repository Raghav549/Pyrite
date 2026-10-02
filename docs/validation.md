# Pyrite validation record

What was run, what was measured, and what remains unproven. Fixture results are
not represented as published-checkpoint results.

## Current audit checks (2026-10-03)

- `.venv/bin/python -m pytest -q` — **249 passed**.
- `.venv/bin/python -m ruff check pyrite tests scripts setup.py` — passed.
- `.venv/bin/python -m compileall -q pyrite tests scripts` — passed.
- `cc -std=c99 -Wall -Wextra -Werror -fsyntax-only pyrite/kernels/native.c` — passed.
- The native Q4_K/Q6_K tests compare C matvec/dequantization against Pyrite's
  reference decoders on valid multi-block rows. The executor integration tests
  also assert native calls occur when a compiler is available.
- With `PATH` set so no C compiler could be found, the K-quant execution tests
  still passed through the Python fallback (**15 passed, 2 native-only tests
  skipped**).
- `python -m pip wheel . --no-deps -w /tmp/pyrite-wheel` built a wheel; inspection
  confirmed both `pyrite/kernels/native.py` and `native.c` are packaged. Installing
  that wheel into a temporary target and importing it reported the native kernels
  available. This runtime build needs a C compiler, not Python development headers.
- Range-cache, pending-prefetch, KV accounting/eviction, RSS-ceiling behavior,
  per-row quantized GGUF sizing, dense Qwen3 CLI checks, and the canonical MoE
  contract are covered by tests using genuine on-disk GGUF fixtures.

## Published-checkpoint attempt

The requested Qwen3-0.6B Q4_K_M file was identified at
[unsloth/Qwen3-0.6B-GGUF](https://huggingface.co/unsloth/Qwen3-0.6B-GGUF/blob/main/Qwen3-0.6B-Q4_K_M.gguf).
The file is listed as 397 MB with SHA-256
`ac2d97712095a558e31573f62f466a3f9d93990898b0ec79d7c974c1780d524a`. It was **not**
downloaded or run: direct HTTPS from this sandbox fails during TLS setup with
`OpenSSL SSL_connect: SSL_ERROR_SYSCALL` (the Hugging Face endpoint and alternate
model mirrors were both unreachable). No fixture or externally hosted inference
was substituted for the requested local end-to-end run.

Accordingly, real published-checkpoint compatibility and 4-GB residency on a
published model remain unverified. The next validation step is to provide the
GGUF file locally or run in an environment that can download it, verify its
SHA-256, then run `qwen3-check --full`, generation, and the CPU benchmark while
recording actual peak RSS and output.

## Architectures and fixture coverage

| Path | GGUF architecture | Attention | Feed-forward | Coverage |
|---|---|---|---|---|
| MoE | `qwen3moe` | GQA, QK-norm, RoPE | softmax router, normalized top-k SwiGLU experts | tiny on-disk MoE fixtures, oracle quantization checks |
| Dense | `qwen3` | GQA, QK-norm, RoPE | dense SwiGLU | tiny on-disk fixtures, dense CLI/generation tests |
| Dense | `llama` | GQA, optional partial RoPE | dense SwiGLU | tiny on-disk fixtures, reference checks |

`pyrite generate`, `pyrite bench`, and `qwen3-check` dispatch on
`general.architecture`. The canonical Qwen3-MoE metadata contract is 94 layers,
128 experts, and top-8; `qwen3-check --require-canonical` enforces those three
values. Unknown architectures and unsupported quantizations are refused.

## Historical oracle crosscheck (measured 2026-10-01)

These results were recorded before the current native-kernel and memory audit;
they were measured on generated tiny GGUF fixtures with synthetic weights and
real tokenizer/quantization layouts, not published checkpoints. The crosscheck
reported 28 fixture files and 5 perplexity comparisons passed. Decoder parity
against the independent `gguf` oracle was exactly `0.0` on the official-quantizer
bytes checked (F32, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q8_1, Q4_K_M, and pure Q2_K
through Q6_K).

Historical greedy decoding matched `llama.cpp` token-for-token on the F32
fixtures (`"hello world"`, four new tokens, temperature zero):

- tiny MoE: `[264, 260, 248, 63, 90, 209]`
- dense Llama and dense Qwen3: `[264, 260, 26, 154, 11, 190]`

Historical PPL on `"hello world " * 100` (300 tokens, `-c 128`), Pyrite vs
`llama-perplexity`:

| Fixture | Quant | llama PPL | Pyrite PPL | Relative difference |
|---|---|---:|---:|---:|
| tiny MoE | F32 | 275.6389 | 275.6347 | 1.52e-05 |
| tiny MoE | Q4_K_M | 275.6256 | 275.6077 | 6.48e-05 |
| dense Llama | F32 | 270.8377 | 270.8301 | 2.82e-05 |
| dense Qwen3 | F32 | 270.8125 | 270.8060 | 2.39e-05 |
| dense Qwen3 | Q4_K_M | 270.8555 | 270.8482 | 2.71e-05 |

Historical realistic-shape fixture: an 8-layer Qwen3-dense model (hidden 256,
GQA 8q/2kv, head dim 32, QK-norm, intermediate 512, vocab 1024, about 5.3 MB
as official Q8_0) on a 2-vCPU sandbox:

- Greedy IDs matched exactly: `[264, 260, 375, 375, 375, 375]`.
- PPL over 135 tokens: llama 1028.9051 vs Pyrite 1029.0594, relative difference
  `1.5e-04`.
- Historical benchmark: 5.50 s, 1.46 tok/s, process RSS 19 → 33 MiB, 91 tensors
  streamed, 0 unsupported types. This is a fixture-only result and is not a
  published-model or current native-kernel performance claim.

## Remaining gaps

- **Published weights:** no Qwen3-0.6B GGUF has been loaded or executed in this
  environment yet; see the download attempt above.
- **Native throughput:** Q4_K/Q6_K C kernels compile and match the reference on
  valid test blocks, but no published-checkpoint throughput has been measured.
  Attention, routing, normalization, RoPE, and other tensor formats still use
  Python.
- **Full-size RSS:** cache limits, range eviction, compact KV, and RSS monitoring
  are exercised on fixtures and tiny budgets. There is no 4-GB peak-RSS result
  for a published model.
- **Long-context output parity:** sliding-window KV eviction is regression-tested
  for correct capacity and continued generation; published-model long-context
  parity has not been measured.
