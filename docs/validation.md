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

# Benchmark
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
335 passed in 31.83s
```

The baseline before this round of work was **249 passed**; 86 tests were added.
No test is skipped on this machine.

| New file | What it pins |
| --- | --- |
| `tests/test_lm_head.py` | The exact `output.weight` regression; untied, tied, tied-by-omission and genuinely-missing LM heads; vocabulary resolution and its conflict rules. |
| `tests/test_native_kernels.py` | All 19 native C kernels cross-checked against the Python reference decoders (`dequantize_rows` and `matvec`). |
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

One thing worth recording: llama.cpp master's `src/models/qwen3moe.cpp` loads
only `ffn_gate_exps` / `ffn_up_exps` / `ffn_down_exps` and never references an
`shexp` tensor - Qwen3-MoE has **no shared expert**, unlike Qwen2-MoE. Pyrite
matches that. Saved output: `validation/moe-crosscheck.txt`.

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

## Known limitations

- **IQ4_XS is decoded in Python only.** The other 19 native types have C
  kernels; IQ4_XS works but is slower.
- **Multi-regex pre-tokenizers are refused, not approximated.** `chameleon`,
  `deepseek-coder`, `deepseek-llm`, `falcon`, `gpt-4o`, `llama4`, `qwen35`,
  `tekken`, `default` and others apply a *sequence* of regexes in llama.cpp.
  Pyrite raises `UnsupportedPreTokenizer` rather than emit plausible-but-wrong
  ids. Supported: the qwen2, llama3/llama-bpe, gpt-2 and poro/bloom families.
- **A GGUF with no `tokenizer.ggml.pre` is refused.** llama.cpp throws for an
  unknown pre-tokenizer for the same reason.
- **`_apply_merges` is O(n^2)** in the number of symbols in a piece. Long pieces
  in a large vocabulary are slower than they need to be.
- **No GPU path.** Everything runs on CPU; there is no CUDA or Metal backend.
- **No published MoE checkpoint was run.** The MoE cross-check above uses a
  locally built checkpoint with 4 experts, top-2 routing and random weights. The
  routing path is verified against llama.cpp, but nothing at 128-expert scale
  was executed here.

## Artifacts

| Path | Contents |
| --- | --- |
| `validation/generation-qwen3-vocab.json` | Pyrite generation on the real-vocab Qwen3 fixture. |
| `validation/pyrite-generation.txt` | Pyrite's completion text. |
| `validation/llama-cpp-generation.txt` | llama.cpp's completion text, same prompt. |
| `validation/benchmark-qwen3-0p6b-shape.txt` | The benchmark above. |
| `validation/qwen3-vocab-check.json` | Full `qwen3-check --full` on the 84 MB fixture. |
| `validation/qwen3-0p6b-shape-check.json` | Full `qwen3-check --full` on the 1.41 GB fixture. |
| `validation/moe-crosscheck.txt` | The Qwen3-MoE generation cross-check against llama.cpp. |
