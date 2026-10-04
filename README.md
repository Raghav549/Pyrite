# Pyrite

Pyrite is a research runtime for private, local AI on memory-constrained devices.

## Core idea

Separate model capacity from resident memory. GGUF tensors and MoE expert slices are read on demand from local storage; only a byte-capped working set, compact KV state, and the current activations are kept in memory.

## Target

First-class target: laptops with about 4 GB RAM. The default profile sets a 4096 MiB process-RSS ceiling (enforced where the OS exposes RSS), reserves 768 MiB for the interpreter and transient allocations, and budgets the remaining 3328 MiB for managed runtime state. Managed cache/KV budgets are enforced independently; a missing OS RSS counter is reported as unknown, not treated as zero usage. These limits are not a promise that every model or context fits. Lower `PYRITE_KV_TOKENS` when the KV cache cannot fit.

## Architecture

- bounded, byte-accounted cache with LRU eviction and range reads
- local block and shard store, content-addressed paging, and offline-first policy
- route selection, execution planning, and predictive prefetch hooks
- bounded compact KV cache with `stop` and `sliding_window` policies
- GGUF adapters with per-row quantized-size validation
- CPU transformer execution for supported dense and MoE architectures
- optional portable C kernels for 19 GGML types, each cross-checked against the Python reference decoder
- T-SAR-inspired ternary reference kernel (not the proposed hardware modification)

## Supported architectures

Pyrite dispatches on GGUF `general.architecture`:

- `qwen3moe` — Qwen3-MoE GQA, per-head QK norm, softmax router, top-k SwiGLU experts streamed from stacked expert tensors
- `qwen3` — dense Qwen3 with GQA, QK norm, and dense SwiGLU MLP
- `llama` — dense Llama with GQA, optional partial rotary dimensions, and dense SwiGLU MLP

Unknown architectures and unsupported tensor types are refused rather than approximated. Nothing in the engine requires a particular model size: every dimension comes from the GGUF metadata and is cross-validated against the actual tensor shapes. `qwen3-check --expect-profile NAME` is the opposite of a requirement - it asserts a *declared* expectation (see `pyrite/model_profiles.py` for the published Qwen3 and Qwen3-MoE shapes) and reports a field-by-field diff when the file does not match it.

## Qwen3 checkpoint validation

```bash
python -m pyrite qwen3-check /path/to/model.gguf
python -m pyrite qwen3-check /path/to/model.gguf --full
python -m pyrite qwen3-check /path/to/model.gguf --expect-profile qwen3-235b-a22b
python -m pyrite plan /path/to/model.gguf
```

`qwen3-check` reports the resolved architecture config, the full tensor
registry, which tensor is the LM head and why, and any tensor type Pyrite cannot
decode. `plan` reports the checkpoint's real byte footprint, the KV cache size
for the configured context, the streaming workspace, and the RAM this machine
actually has (from `/proc/meminfo`, falling back to `os.sysconf`), and says
plainly whether the run fits.

The full report includes tensor sizes, the largest streamed row or expert slice, and whether those ranges fit the current budget. For the Qwen3-235B-A22B-Instruct-2507 GGUF family, Q4_K_M is published as five split files; use `llama-gguf-split` to merge them before passing a single GGUF file to Pyrite. The full Q4_K_M set is roughly 142 GB. A 4 GB setting bounds resident memory; it does not make the full model fit on disk or make a large KV context affordable.

## Local generation and benchmarking

```bash
python -m pyrite generate model.gguf --prompt "hello" --max-new-tokens 16 --temperature 0
python -m pyrite generate model.gguf --prompt "hello" --kv-cache-policy sliding_window
python -m pyrite tokenize model.gguf "hello world"
python -m pyrite.bench --checkpoint model.gguf --tokens 8
```

The executor streams dense projection rows, selected MoE expert slices, and vocabulary rows; it does not materialize the whole model. KV entries are stored as compact fp32 arrays. The default `stop` policy ends generation when the configured KV limit is reached; `sliding_window` evicts the oldest KV entries and continues up to the model/runtime context limit.

A C compiler (`cc`, `gcc`, or `clang`) enables the optional scalar C kernels.
Pyrite compiles the small, Python-header-free C source into the system temporary
directory on first use, borrows quantized buffers without copying them, and fuses
dequantization with matrix-vector multiplication. If compilation is unavailable,
every type falls back to the bounded Python reference path instead.

Native kernels cover all 21 decodable GGML types - F32, F16, BF16, F64,
I8/I16/I32/I64, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q8_1, Q2_K, Q3_K, Q4_K, Q5_K,
Q6_K, IQ4_NL and IQ4_XS - each cross-checked against its Python reference
decoder in `tests/test_native_kernels.py`, which fails if any decodable type
loses its C kernel. No runtime oracle, cloud service, or network access is
required for inference.

Measured on 2 CPU cores with no GPU: a 1.41 GB, 28-layer Qwen3-shaped checkpoint
runs at about 0.8 prefill tokens/s and 0.69 decode tokens/s with a 1374 MiB peak
RSS. See [`docs/validation.md`](docs/validation.md) for the full numbers.

## Tokenizer

Byte-level BPE from GGUF metadata, with the pre-tokenizer regex selected from
`tokenizer.ggml.pre`. Supported families: qwen2 (and its aliases), llama3 /
llama-bpe / dbrx / smaug-bpe, gpt-2 / mpt / olmo / starcoder / refact /
command-r / smollm / codeshell / exaone, and poro-chat / bloom / gpt3-finnish.
Unicode property classes (`\p{L}`, `\p{N}`, `\p{P}`, `\p{M}`, `\p{S}`) are
expanded from `unicodedata`, not approximated by Python's `\w`/`\d`.

Pre-tokenizers that llama.cpp implements as a *sequence* of regexes
(`default`, `chameleon`, `deepseek-coder`, `deepseek-llm`, `falcon`, `gpt-4o`,
`llama4`, `qwen35`, `tekken` and others) raise `UnsupportedPreTokenizer` rather
than emit plausible-looking but wrong token ids. A GGUF that omits
`tokenizer.ggml.pre` is refused for the same reason llama.cpp refuses it.

Verified against llama.cpp on a real 151,936-token Qwen2 vocabulary: 9/9 corpus
cases produce byte-identical token ids, including Devanagari, mixed scripts,
digits, punctuation and the Qwen chat control tokens.

## Validation

Two scripts compare Pyrite against a real `llama.cpp` build:

```bash
python scripts/llama_reference_compare.py --llama-bin /path/to/llama.cpp/build/bin \
    --vocab-gguf models/ggml-vocab-qwen2.gguf --model-gguf models/qwen3-vocab-fixture.gguf
python scripts/llama_crosscheck.py --llama-bin /path/to/llama.cpp/build/bin
```

The measured results are in [`docs/validation.md`](docs/validation.md). On the
same GGUF, with the same prompt and greedy decoding, Pyrite and llama.cpp
produced 12 identical tokens - for both the dense and the Qwen3-MoE
architectures. Those checkpoints have random weights, so they validate the
arithmetic, not model quality; no published Qwen3 checkpoint could be downloaded
in the environment where this was measured, and none is claimed to have been
run.

## Privacy

Inference defaults to offline operation. Prompts and model state are not sent to a data center by the runtime. Applications embedding Pyrite must also keep their own telemetry, cloud sync, and remote tools disabled when strict local-only operation is required.

## Development

The `dev` extra installs the optional GGUF and NumPy oracles used by decoder-parity tests; these are test-only dependencies.

```bash
python -m pip install -e ".[dev]"
python -m pyrite status
python -m pyrite route "write a Python API"
python -m pyrite.bench
python -m ruff check pyrite tests scripts setup.py
python -m compileall -q pyrite tests scripts
python -m pytest -q
```
