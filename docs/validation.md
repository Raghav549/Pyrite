# Pyrite validation record

What was actually run on this machine, what it measured, and what is still
unproven. Nothing below is a claim about a checkpoint that was not executed
here.

Machine: 2 CPU cores, 3939.9 MiB RAM (`/proc/meminfo`, no swap), no GPU (neither
CUDA nor Metal), Python 3.11.2, 20 GB free disk.

## Commands

Every result below comes from one of these, run from the repository root.

```bash
# Unit + integration + regression suite
python3 -m pytest -q

# Inspect any GGUF: metadata, arch config, tensor registry, LM head, memory plan
python3 -m pyrite qwen3-check models/qwen3-vocab-fixture.gguf --full

# Memory requirements vs what this machine has
python3 -m pyrite plan models/qwen3-0p6b-shape.gguf

# Real local generation
python3 -m pyrite generate models/qwen3-vocab-fixture.gguf \
    --prompt "Explain Pyrite in one short paragraph." --max-new-tokens 12 \
    --temperature 0.0 --seed 42

# Tokenizer ids for a prompt
python3 -m pyrite tokenize models/qwen3-vocab-fixture.gguf "Hello world"

# Supported quantization types and decoder support
python3 -m pyrite types --decodable-only

# Route-planning benchmark (planning only, not inference - see below)
python3 -m pyrite bench

# Build the reference fixtures used below
python3 scripts/build_vocab_fixture.py \
    --vocab-gguf models/ggml-vocab-qwen2.gguf --out models/qwen3-vocab-fixture.gguf

# Cross-check against llama.cpp (needs a llama.cpp build; see below)
python3 scripts/llama_reference_compare.py \
    --llama-bin /tmp/llama.cpp-master/build/bin \
    --vocab-gguf models/ggml-vocab-qwen2.gguf \
    --model-gguf models/qwen3-vocab-fixture.gguf --tokens 12
```

Reference runtime: llama.cpp `master`, built from source with

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DGGML_BLAS=OFF -DLLAMA_CURL=OFF \
      -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF \
      -DBUILD_SHARED_LIBS=OFF
cmake --build build -j2 --target llama-tokenize llama-completion
```

## Test suite

```
$ python3 -m pytest -q
345 passed in 19.73s
```

The baseline before this round of work was **249 passed**; 96 tests were added.
No test is skipped: the checkpoint-shape tests build their fixtures themselves
rather than reading files that are not in the repository.

| New file | What it pins |
| --- | --- |
| `tests/test_lm_head.py` | The exact `output.weight` regression; untied, tied, tied-by-omission and genuinely-missing LM heads; vocabulary resolution and its conflict rules. |
| `tests/test_native_kernels.py` | All 21 native C kernels cross-checked against the Python reference decoders (`dequantize_rows` and `matvec`), plus a guard that no decodable type is left without one. |
| `tests/test_gguf_errors.py` | Corrupt magic, versions, counts, string lengths, tensor names, dimension counts, unknown types, out-of-range offsets, duplicates, alignment, truncated payloads. |
| `tests/test_memory_plan.py` | KV estimate, `/proc/meminfo` units, footprint, plan acceptance/refusal, streaming note for larger-than-RAM models. |
| `tests/test_tokenizer.py` | Per-`pre` pre-tokenizer semantics; refusal of multi-regex and missing `pre`. |
| `tests/test_model_profiles.py` | The published-profile table cannot assert fields real checkpoints lack (architecture, shared experts). |

## Tokenizer vs llama.cpp

`models/ggml-vocab-qwen2.gguf` (151,936 tokens, `tokenizer.ggml.pre = qwen2`)
tokenized by both runtimes over a 9-case corpus:

```
  ascii          IDENTICAL n_ref=9   n_pyrite=9
  hindi          IDENTICAL n_ref=13  n_pyrite=13
  mixed          IDENTICAL n_ref=11  n_pyrite=11
  digits         IDENTICAL n_ref=20  n_pyrite=20
  punct          IDENTICAL n_ref=14  n_pyrite=14
  whitespace     IDENTICAL n_ref=9   n_pyrite=9
  empty-adjacent IDENTICAL n_ref=1   n_pyrite=1
  chat           IDENTICAL n_ref=21  n_pyrite=21
  long           IDENTICAL n_ref=370 n_pyrite=370
  -> 9/9 cases identical, round_trip_ok=True
```

The corpus covers ASCII, Devanagari, mixed scripts, digits and a fraction,
punctuation, runs of whitespace and a trailing newline, a single character, the
Qwen chat control tokens, and a 120-word input. The `chat` case proves control
tokens are matched verbatim rather than split.

## Generation vs llama.cpp

No published Qwen3 GGUF could be obtained in this environment (Hugging Face and
every mirror are unreachable here; Git LFS media and the GitHub releases CDN are
blocked, and no Qwen3 quant fits in GitHub's 100 MB non-LFS limit). So a
reference checkpoint was built locally instead: real Qwen3 tensor layout, real
Qwen2 vocabulary (151,936 tokens), F32 weights, random values.

```
$ python3 scripts/build_vocab_fixture.py --vocab-gguf models/ggml-vocab-qwen2.gguf \
      --out models/qwen3-vocab-fixture.gguf
wrote models/qwen3-vocab-fixture.gguf (84,017,568 bytes) arch=qwen3 vocab=151936 pre='qwen2' layers=2 hidden=64
```

Both runtimes, same file, same prompt, greedy, 12 tokens:

```
llama.cpp : 本质 {}\n聞く spectro jouer_visibility Tireizzes.simps Perform ż𝄅
pyrite    : 本质 {}\n聞く spectro jouer_visibility Tireizzes.simps Perform ż𝄅
```

12/12 generated tokens identical. Byte-identical output was not expected and is
not required; the point is that the forward pass agrees. This exercises
tokenization, RMSNorm, GQA attention, QK-Norm, RoPE, SwiGLU, the KV cache, the
LM head projection over 151,936 logits, and argmax sampling.

The generated text is meaningless because the weights are random. It says
nothing about model quality.

Saved outputs: `validation/llama-cpp-generation.txt`,
`validation/pyrite-generation.txt`, `validation/generation-qwen3-vocab.json`.

### Qwen3-MoE

The same cross-check was run on the MoE architecture, using a checkpoint built
with the same real Qwen2 vocabulary:

```
$ python3 scripts/build_vocab_fixture.py --vocab-gguf models/ggml-vocab-qwen2.gguf \
      --out models/qwen3moe-vocab-fixture.gguf --architecture qwen3moe \
      --hidden-size 64 --num-layers 2 --num-heads 4 --num-kv-heads 2 --head-dim 16 \
      --experts 4 --experts-used 2 --moe-ffn 16
wrote models/qwen3moe-vocab-fixture.gguf (83,921,760 bytes) arch=qwen3moe vocab=151936 pre='qwen2' layers=2 hidden=64
```

```
llama.cpp : emand 있는데\r\n        \r\n Ant /**\n ,-_inp(Address surprisesthèseiveringkening
pyrite    : emand 있는데\r\n        \r\n Ant /**\n ,-_inp(Address surprisesthèseiveringkening
IDENTICAL : True
```

12/12 tokens identical. This exercises the softmax router, top-k expert
selection, expert-weight normalisation, slicing of the stacked 3D
`ffn_{gate,up,down}_exps` tensors, and the MoE SwiGLU path.

### 288-step greedy divergence check

Twelve tokens from one prompt only tests argmax at twelve hidden states. So both
fixtures were run for 48 tokens from three different prompts each - prose, a
partial sentence, and Python source - and compared byte for byte:

```
qwen3 dense  prompt='Explain Pyrite in one short '   tokens= 48 IDENTICAL=True agreeing_prefix=249/249
qwen3 dense  prompt='The capital of France is'       tokens= 48 IDENTICAL=True agreeing_prefix=226/226
qwen3 dense  prompt='def fibonacci(n):\n    '        tokens= 48 IDENTICAL=True agreeing_prefix=237/237
qwen3moe     prompt='Explain Pyrite in one short '   tokens= 48 IDENTICAL=True agreeing_prefix=236/236
qwen3moe     prompt='The capital of France is'       tokens= 48 IDENTICAL=True agreeing_prefix=258/258
qwen3moe     prompt='def fibonacci(n):\n    '        tokens= 48 IDENTICAL=True agreeing_prefix=243/243

6/6 greedy runs byte-identical over 48 tokens each
```

288 consecutive argmax decisions agreed, from three unrelated starting states.
A forward pass that differed by more than float noise would diverge within a few
steps, because each step's output feeds the next. This is evidence about the
logits, not just the sampled ids, though it is not a direct float comparison.
Saved output: `validation/greedy-divergence-check.txt`.

One thing worth recording: llama.cpp master's `src/models/qwen3moe.cpp` loads
only `ffn_gate_exps` / `ffn_up_exps` / `ffn_down_exps` and never references an
`shexp` tensor - Qwen3-MoE has **no shared expert**, unlike Qwen2-MoE. Pyrite
matches that. Saved output: `validation/moe-crosscheck.txt`.

### 128-expert, top-8 routing

Scaled the MoE fixture to the published expert count and top-k (128 experts,
top-8, 4 layers, hidden 128, real 151,936-token vocabulary, 187,733,824 bytes)
and ran 24 greedy tokens from three prompts:

```
  prompt='Explain Pyrite in one short '   tokens= 24 IDENTICAL=True
  prompt='The capital of France is'       tokens= 24 IDENTICAL=True
  prompt='def fibonacci(n):\n    '        tokens= 24 IDENTICAL=True

3/3 runs byte-identical over 24 tokens each (128 experts, top-8)
```

**A correction to an earlier version of this document.** A first run of this
check reported 2/3 and blamed float noise in the router's top-8 boundary, with
measured 8th-vs-9th probability gaps of 2.0e-06 to 1.4e-05. That conclusion was
wrong. Those margins are real measurements, but they were not the cause.

The cause was the comparison harness. One of the generated tokens is 151808,
whose text is `[PAD151808]` and whose `tokenizer.ggml.token_type` is 4
(`USER_DEFINED`), i.e. a special token. Pyrite *did* generate it - it is present
in `output_ids` - but `generate_text` decodes with `skip_special=True`, which is
the right default for a user-facing string, so the token vanished from the
compared text. llama.cpp prints every sampled token. Decoding the same ids with
`skip_special=False` reproduces llama.cpp's output exactly:

```
decode(skip_special=True) : '...-positionمع vấn greens设定 Duis resumed'
decode(skip_special=False): '...-positionمع vấn[PAD151808] greens设定 Duis resumed'
```

`scripts/llama_reference_compare.py` now decodes verbatim, and
`tests/test_tokenizer.py` pins the distinction so the false conclusion cannot
come back. The router margins are recorded here only because they were measured;
they explain nothing.

Evidence: `validation/moe-128-expert-crosscheck.txt`,
`validation/moe-128e-router-margin.txt`.

## Larger-model validation

`models/qwen3-0p6b-shape.gguf` is a 1,411,777,152-byte checkpoint with the
published Qwen3-0.6B shape - 28 layers, 1024 hidden, 3072 feed-forward, 16 heads
/ 8 KV heads, head_dim 64 - written with the streaming builder in
`scripts/build_vocab_fixture.py` (peak memory is one tensor, so it can be built
on this machine at all).

```
arch  : qwen3 layers 28 hidden 1024 ffn 3072
heads : 16 kv_heads 8 head_dim 64 rotary 64
lmhead: output.weight [1024, 271] F32, not tied
types : decodable True, native_ready True, unsupported []
mem   : model 1346.4 MiB, kv 58.4 MiB, workspace 3269.6 MiB, ram 3939.9 MiB, ok True
stream: 311 tensors, 1,411,753,984 bytes, 0 oversized, streams from disk
```

It validates, loads, and generates. Real inference produced 8 new tokens with a
peak RSS of 1374.9 MB and zero evictions.

**Hardware limit, stated plainly:** a real Qwen3-0.6B Q8_0 file is 639,446,688
bytes and would run here; Qwen3-1.7B Q8_0 is 1,834,426,016 bytes and would be
tight against 3.9 GB of RAM with any useful context; Qwen3-4B and above cannot
run on this machine at any quantization that keeps a usable KV cache. No model
larger than the 1.41 GB fixture above was run, and none is claimed to have been.

## Benchmark (measured, not estimated)

`models/qwen3-0p6b-shape.gguf`, 2 cores, CPU only, saved to
`validation/benchmark-qwen3-0p6b-shape.txt`:

```
open_seconds        : 0.420
first_token_seconds : 1.421  (cold: first full pass over 28 layers)
prefill             : 0.80 tok/s
decode              : 0.685 tok/s
peak_rss_mb         : 1374.2
kv_tokens/kv_bytes  : 44 / 5,046,272   (context reached 44 of 512)
native_calls        : 8800
native_bytes        : 62,057,611,264
native_seconds      : 47.831
io_seconds          : 1.692
bytes_loaded        : 1,410,721,792
evictions           : 0    cache_hits: 8582
```

The native C kernels account for 47.8 s of the run; Python orchestration is the
remainder. Decode throughput is dominated by the fact that every weight tensor
is read from disk or cache once per token and there are only 2 cores.

Machine, as reported by `python3 -m pyrite bench`
(`validation/bench-cli.json`):

```
Intel(R) Xeon(R) Processor @ 2.60GHz | 2 cores | 3939.9 MiB RAM | Python 3.11.2
Linux-6.1.158+-x86_64-with-glibc2.36
```

Note that `pyrite bench` benchmarks *route planning* (16 plans in 0.66 ms), not
inference. The inference numbers above come from the direct measurement, which
is the script reproduced in `validation/benchmark-qwen3-0p6b-shape.txt`.

## Known limitations

- **Multi-regex pre-tokenizers are refused, not approximated.** `chameleon`,
  `deepseek-coder`, `deepseek-llm`, `falcon`, `gpt-4o`, `llama4`, `qwen35`,
  `tekken`, `default` and others apply a *sequence* of regexes in llama.cpp.
  Pyrite raises `UnsupportedPreTokenizer` rather than emit plausible-but-wrong
  ids. Supported: the qwen2, llama3/llama-bpe, gpt-2 and poro/bloom families.
- **A GGUF with no `tokenizer.ggml.pre` is refused.** llama.cpp throws for an
  unknown pre-tokenizer for the same reason.
- **`_apply_merges` is O(n^2)** in the number of symbols in a piece
  (`pyrite/tokenizer.py:401-419`), and has deliberately been left that way.
  Measured on the real 151,936-token vocabulary: a 200-word sentence encodes in
  **8.1 ms**, a 1,000-character word in **70 ms**, a pathological
  4,000-character word in **694 ms**. Real pre-tokenized pieces are short, so
  the quadratic term never dominates on real text. A heap with lazy
  invalidation would change the merge tie-breaking, putting the single most
  heavily verified property here - token ids identical to llama.cpp - at risk
  to speed up inputs that do not occur.
- **`generate_text` hides special tokens.** It decodes with
  `skip_special=True`, so a generated `USER_DEFINED` or `CONTROL` token does not
  appear in the returned string even though it is in `output_ids`. That is
  intended for user-facing text, but any comparison against another runtime must
  decode with `skip_special=False` or it will report a difference that is not
  there. This already produced one wrong conclusion, recorded above.
- **No GPU path.** Everything runs on CPU; there is no CUDA or Metal backend.
- **No published MoE checkpoint was run.** Every MoE cross-check above uses a
  locally built checkpoint with random weights. That now covers both the small
  (4 experts, top-2) and the published-scale (128 experts, top-8) routing path,
  and both agree with llama.cpp - but a real released Qwen3-MoE file has still
  never been loaded here, so weight-loading for an actual published checkpoint
  is unverified for the MoE architecture.

## Quantized checkpoints (the case this project started from)

The original failure was on `Qwen3-0.6B-Q8_0.gguf`, so quantized storage is a
first-class case rather than an afterthought.
`scripts/build_vocab_fixture.py --quant q8_0|q4_0` emits Q8_0/Q4_0 weights with
encoders that mirror `quantize_row_q8_0_ref` and `quantize_row_q4_0_ref` in
`ggml/src/ggml-quants.c`, over the same real 151,936-token Qwen2 vocabulary.
Pyrite decodes with the same arithmetic ggml does - `dequantize_row_q8_0` is
literally `y = qs[j] * d`.

Greedy generation against `llama-completion --temp 0`, 32 tokens each, on
checkpoints whose **LM head is itself quantized**:

| checkpoint | LM head | identical over 32 tokens |
| --- | --- | --- |
| `qwen3-q8_0-fixture.gguf` (26,646,944 B) | `output.weight [64, 151936] Q8_0` | 2/2 |
| `qwen3-q4_0-fixture.gguf` (16,898,464 B) | `output.weight [64, 151936] Q4_0` | 2/2 |

**4/4 byte-identical** - `validation/quantized-crosscheck.txt`.

Two constraints surfaced while building those files and are now enforced in the
fixture builder and pinned by `tests/test_quantized_checkpoint.py`:

- **RMSNorm weights must stay F32 in a quantized file.** Quantizing them makes
  ggml abort with `binary_op: unsupported types: dst: f32, src0: f32, src1:
  q8_0` - its CPU element-wise ops refuse a mixed-type operand. Reproduced
  against llama.cpp master before the rule was written down. Every real
  quantized GGUF leaves `*_norm.weight` in F32 for exactly this reason.
- **Block alignment is per row, not per tensor.** ggml quantizes block by block
  *within a row*, so `dims[0]` must be a multiple of the block size; a total
  element count that happens to divide is not enough. `attn_q_norm` (16
  elements) and `ffn_down` in a 48-wide FFN both fail that test and stay F32.

```bash
python3 scripts/build_vocab_fixture.py --vocab-gguf models/ggml-vocab-qwen2.gguf \
  --out qwen3-q8_0.gguf --quant q8_0 --architecture qwen3 \
  --hidden-size 64 --num-layers 2 --intermediate-size 64 \
  --num-heads 4 --num-kv-heads 2 --head-dim 16 --max-context 512
```

## Artifacts

| Path | Contents |
| --- | --- |
| `validation/generation-qwen3-vocab.json` | Pyrite generation on the real-vocab Qwen3 fixture. |
| `validation/quantized-crosscheck.txt` | Q8_0 / Q4_0 greedy generation vs llama.cpp: 4/4 identical. |
| `validation/pyrite-generation.txt` | Pyrite's completion text. |
| `validation/llama-cpp-generation.txt` | llama.cpp's completion text, same prompt. |
| `validation/benchmark-qwen3-0p6b-shape.txt` | The benchmark above. |
| `validation/qwen3-vocab-check.json` | Full `qwen3-check --full` on the 84 MB fixture. |
| `validation/qwen3-0p6b-shape-check.json` | Full `qwen3-check --full` on the 1.41 GB fixture. |
| `validation/moe-crosscheck.txt` | The Qwen3-MoE generation cross-check against llama.cpp. |
| `validation/greedy-divergence-check.txt` | 6/6 greedy runs, 48 tokens each, both architectures. |
| `validation/bench-cli.json` | `pyrite bench` output, including the machine description. |
| `validation/moe-128-expert-crosscheck.txt` | 128-expert top-8 cross-check: 3/3 identical. |
| `validation/moe-128e-router-margin.txt` | Router softmax margins - real, but not the cause of anything (see the correction above). |
