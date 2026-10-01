# Pyrite validation record

How Pyrite's execution path is checked, what the measured numbers are, and
what is still openly unproven. No claim below is accepted without the
measurement next to it.

## What runs

- `python -m pytest` — full suite (232 tests, all passing).
- `python -m ruff check pyrite tests scripts` — clean.
- `python scripts/llama_crosscheck.py --llama-bin <dir>` — cross-validates
  Pyrite against official `llama.cpp` release binaries on real GGUF files
  built from scratch: the official quantizer must accept the fixtures,
  Pyrite's decoders must match the independent `gguf` dequantizer exactly on
  official-quantizer bytes, generation must be deterministic, and
  `llama-perplexity` must agree with Pyrite on PPL within 1%.

## Architectures covered

| Path | Arch key | Attention | MLP | Notes |
|---|---|---|---|---|
| MoE | `qwen3moe` | GQA + QK-norm, full rotary | SwiGLU MoE, top-k expert slices | original path |
| Dense | `qwen3` | GQA + QK-norm, full rotary | dense SwiGLU | shares the MoE attention core |
| Dense | `llama` | GQA, no QK-norm, partial rotary (`rope_dimension_count`) | dense SwiGLU | legacy Llama rotary layout |

`pyrite generate`, `pyrite bench`, and the crosscheck script dispatch on the
file's `general.architecture` automatically; unknown architectures are
refused with an explicit error.

## Oracle agreement (measured 2026-10-01)

Crosscheck result: **28 files, 5 PPL comparisons, all passed.**
Decode parity vs the `gguf` oracle is exactly `0.0` on every
official-quantizer byte string (F32, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q4_K_M on
all four fixtures, plus pure Q2_K..Q6_K on the K-dims MoE fixture).

PPL on `"hello world " * 100` (300 tokens, `-c 128`), Pyrite vs
`llama-perplexity`:

| Fixture | Quant | llama PPL | Pyrite PPL | rel diff |
|---|---|---|---|---|
| tiny MoE | F32 | 275.6389 | 275.6347 | 1.52e-05 |
| tiny MoE | Q4_K_M | 275.6256 | 275.6077 | 6.48e-05 |
| dense llama | F32 | 270.8377 | 270.8301 | 2.82e-05 |
| dense qwen3 | F32 | 270.8125 | 270.8060 | 2.39e-05 |
| dense qwen3 | Q4_K_M | 270.8555 | 270.8482 | 2.71e-05 |

Greedy decoding matches `llama.cpp` **token-for-token** on every F32
fixture (`"hello world"`, 4 new tokens, temperature 0):

- tiny MoE: `[264, 260, 248, 63, 90, 209]`
- dense llama and dense qwen3: `[264, 260, 26, 154, 11, 190]`

## Realistic-shape run (measured 2026-10-01)

An 8-layer qwen3-dense model (hidden 256, GQA 8q/2kv, head dim 32, QK-norm,
intermediate 512, vocab 1024, ~5.3 MB as official Q8_0) on a 2-vCPU sandbox:

- Greedy parity: Pyrite `[264, 260, 375, 375, 375, 375]` — exact match.
- PPL over 135 tokens: llama 1028.9051 vs Pyrite 1029.0594, rel 1.5e-04.
  (Slightly above the 2-layer bar; consistent with fp32 rounding
  accumulating over 8 layers rather than a math error — greedy IDs agree
  exactly.)
- `pyrite bench --checkpoint <q8> --tokens 8`: 5.50 s, **1.46 tok/s**,
  process RSS 19 → 33 MB, 91 tensors streamed, 0 unsupported types.

## Openly unproven

- **Published weights.** The sandbox has no network access, so every fixture
  above uses random weights with a real tokenizer and real GGUF structure.
  Math parity against `llama.cpp` on byte-identical files is strong evidence
  the forward pass is right, but no published checkpoint (e.g. Qwen3-0.6B,
  SmolLM2) has been run end to end yet. With network, the check is:
  download a small GGUF, run `pyrite generate` and the `llama.cpp` CLI with
  temperature 0, and compare IDs.
- **Throughput.** The executor is a pure-Python correctness reference
  (~1.5 tok/s on a 5 MB model). It is not a throughput claim; real-size
  checkpoints need the native-kernel extension point before they are usable.
- **Pure Q2_K/Q3_K/Q5_K/Q6_K on dense.** K-quant decoders are shared code
  covered by unit tests plus pure-K MoE crosschecks; dense fixtures are
  crosschecked through F32, Q4_0..Q8_0, and the Q4_K_M mix.
- **4 GB residency on real models.** The resident budget, streaming, and
  eviction machinery is tested with tiny budgets on fixtures
  (`peak_resident_bytes` enforced, streamed == relaxed outputs); no
  full-size-model residency measurement exists yet.
