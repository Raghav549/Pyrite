"""Resolve the LM output projection and the vocabulary from GGUF metadata.

The contract implemented here is the one the reference runtime uses.  In
``llama.cpp`` every decoder-only architecture loads the LM head like this
(``src/models/qwen3.cpp`` and ``src/models/qwen3moe.cpp``)::

    output = create_tensor(tn(LLM_TENSOR_OUTPUT, "weight"),
                           {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);
    // if output is NULL, init from the input tok embed
    if (output == NULL) {
        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"),
                               {n_embd, n_vocab}, TENSOR_DUPLICATED);
    }

Two consequences matter and both were previously violated:

* ``output.weight`` is **optional**.  A checkpoint with tied embeddings simply
  does not contain it, and the token-embedding matrix *is* the LM head.  The
  reference implementation never requires ``output.weight`` to exist.
* There is **no** ``tied_word_embeddings`` GGUF metadata key in the reference
  implementation (``grep -rn tied_word_embeddings`` over ``llama.cpp`` returns
  nothing).  Tying is declared by the *absence* of ``output.weight``, not by a
  flag.  A flag is honoured when a writer does emit one, but its absence must
  never be treated as "not tied".

The vocabulary is not a ``{arch}.vocab_size`` key either: the converter pads
``tokenizer.ggml.tokens`` up to the HuggingFace ``vocab_size`` with ``[PADn]``
tokens (``conversion/base.py``, "Padding vocab with N token(s)"), so
``len(tokenizer.ggml.tokens)`` equals the row count of ``token_embd.weight``.
Real checkpoints therefore carry no explicit vocab key at all, and vocab-only
files (``ggml-vocab-*.gguf``) carry no tokenizer.  :func:`resolve_vocabulary`
accepts any consistent source and reports which one it used.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .ggml_types import spec as ggml_spec
from .ggml_types import type_name

#: Metadata keys a writer may use to declare tied embeddings.  None of them is
#: required; see the module docstring.
# llama.cpp itself never reads a tie flag - it falls back to the embedding
# unconditionally - so these keys are only ever *hints* that let Pyrite refuse
# an explicitly untied checkpoint that is missing its LM head.  Both the GGUF
# "tied_word_embeddings" spelling and the Hugging Face config.json
# "tie_word_embeddings" spelling are accepted.
TIED_EMBEDDING_KEYS: tuple[str, ...] = (
    "{arch}.tied_word_embeddings",
    "{arch}.tie_word_embeddings",
    "general.tied_word_embeddings",
    "general.tie_word_embeddings",
    "tied_word_embeddings",
    "tie_word_embeddings",
)

TOKEN_EMBD = "token_embd.weight"  # noqa: S105 - a GGML tensor name, not a secret
OUTPUT = "output.weight"


class LMHeadContractError(ValueError):
    """Raised when a checkpoint cannot provide a usable LM output projection."""


class VocabularyContractError(ValueError):
    """Raised when a checkpoint's vocabulary size cannot be determined."""


@dataclass(frozen=True)
class Vocabulary:
    """Vocabulary size and how it was established.

    ``size`` is the width of the LM head / row count of ``token_embd.weight``
    and therefore the number of logits produced per token.  ``token_count`` is
    the number of vocabulary entries the tokenizer can actually name; the two
    differ only for checkpoints that pad the token list (the common case).
    """

    size: int
    token_count: int
    source: str

    @property
    def is_padded(self) -> bool:
        return self.size != self.token_count


@dataclass(frozen=True)
class LMHead:
    """The tensor that projects hidden states to vocabulary logits."""

    tensor_name: str
    tied: bool
    rows: int
    cols: int
    ggml_type: int
    declared_tied: bool | None
    source: str

    @property
    def type_name(self) -> str:
        return type_name(self.ggml_type)

    def to_dict(self) -> dict[str, object]:
        return {
            "tensor": self.tensor_name,
            "tied": self.tied,
            "declared_tied": self.declared_tied,
            "shape": [self.cols, self.rows],
            "type": self.type_name,
            "source": self.source,
        }


def declared_tied_embeddings(metadata: Mapping[str, object], arch: str) -> bool | None:
    """Read an explicit tie declaration, or ``None`` when the file makes none."""
    for template in TIED_EMBEDDING_KEYS:
        key = template.format(arch=arch)
        if key not in metadata:
            continue
        value = metadata[key]
        if isinstance(value, bool):
            return value
        if isinstance(value, int) and value in (0, 1):
            return bool(value)
        raise LMHeadContractError(
            f"{key} must be a boolean, got {value!r}; remove the key or fix its value"
        )
    return None


def resolve_vocabulary(
    metadata: Mapping[str, object],
    arch: str,
    embedding_rows: int | None,
) -> Vocabulary:
    """Determine the vocabulary width from metadata and/or the embedding tensor.

    ``embedding_rows`` is the row count of ``token_embd.weight`` (``None`` when
    the checkpoint has no embedding tensor, which is itself a contract error
    for a decoder-only model but is reported by the caller, not here).
    """
    candidates: list[tuple[str, int]] = []

    tokens = metadata.get("tokenizer.ggml.tokens")
    if isinstance(tokens, (list, tuple, Sequence)) and not isinstance(tokens, (str, bytes)):
        count = len(tokens)
        if count:
            candidates.append(("tokenizer.ggml.tokens", count))

    legacy = metadata.get("tokenizer.ggml.vocab_size")
    if isinstance(legacy, int) and not isinstance(legacy, bool) and legacy > 0:
        candidates.append(("tokenizer.ggml.vocab_size", legacy))

    explicit = metadata.get(f"{arch}.vocab_size")
    if isinstance(explicit, int) and not isinstance(explicit, bool) and explicit > 0:
        candidates.append((f"{arch}.vocab_size", explicit))

    if embedding_rows is not None and embedding_rows > 0:
        candidates.append((f"{TOKEN_EMBD} rows", embedding_rows))

    if not candidates:
        raise VocabularyContractError(
            f"cannot determine the vocabulary size: no tokenizer.ggml.tokens, no "
            f"vocab_size metadata and no {TOKEN_EMBD} tensor to measure"
        )

    declared = [(name, value) for name, value in candidates if value != embedding_rows]
    token_count = next((value for name, value in candidates if name == "tokenizer.ggml.tokens"), None)

    # Two *declared* vocab sizes that disagree mean the file contradicts itself.
    # A token list shorter than the tensor is not a contradiction: converters
    # append [PADn] entries, and llama.cpp pads the vocabulary up to the tensor
    # width (``n_vocab = ml.get_key(LLM_KV_VOCAB_SIZE, n_tokens)`` followed by
    # padding), so the tensor is authoritative for the projection width.
    distinct_declared = {value for name, value in declared if name != "tokenizer.ggml.tokens"}
    if len(distinct_declared) > 1:
        raise VocabularyContractError(
            "the checkpoint declares conflicting vocabulary sizes: "
            + ", ".join(f"{name}={value}" for name, value in declared)
            + "; refusing a self-inconsistent checkpoint"
        )

    declared_size = distinct_declared.pop() if distinct_declared else None
    if embedding_rows is not None and embedding_rows > 0:
        # The embedding matrix is the ground truth: the LM head projects onto
        # exactly this many rows, and no token id may address past it.
        size = embedding_rows
        source = f"{TOKEN_EMBD} rows"
    elif declared_size is not None:
        size = declared_size
        source = next(name for name, value in declared if value == declared_size)
    else:
        size = int(token_count)  # type: ignore[arg-type]
        source = "tokenizer.ggml.tokens"

    if declared_size is not None and declared_size > size:
        raise VocabularyContractError(
            f"the declared vocabulary size {declared_size} exceeds the {size} rows "
            f"available; the LM head could not address every token"
        )
    if token_count is None:
        token_count = size
    if token_count > size:
        raise VocabularyContractError(
            f"tokenizer.ggml.tokens has {token_count} entries but the vocabulary is "
            f"only {size} wide; token ids would index past the LM head"
        )
    return Vocabulary(size=size, token_count=token_count, source=source)


def resolve_lm_head(
    tensors: Mapping[str, object],
    metadata: Mapping[str, object],
    arch: str,
    hidden_size: int,
    vocab: Vocabulary,
    *,
    decodable: set[int] | frozenset[int] | None = None,
) -> LMHead:
    """Pick and validate the LM output projection.

    ``tensors`` maps tensor names to objects exposing ``dims`` and ``ggml_type``.
    Returns an :class:`LMHead` describing which tensor to use.  Raises
    :class:`LMHeadContractError` with an actionable message when neither a
    separate ``output.weight`` nor a legally reusable embedding matrix exists.
    """
    declared = declared_tied_embeddings(metadata, arch)
    expected = (hidden_size, vocab.size)

    def describe(tensor: object) -> str:
        return f"shape {tuple(getattr(tensor, 'dims', ()))}, type {type_name(getattr(tensor, 'ggml_type', -1))}"

    def check(name: str, tensor: object, role: str) -> None:
        dims = tuple(getattr(tensor, "dims", ()))
        if dims != expected:
            raise LMHeadContractError(
                f"{name} has {describe(tensor)}; the {role} must be "
                f"(embedding_length={hidden_size}, vocab={vocab.size}) in GGML order"
            )
        ggml_type = int(getattr(tensor, "ggml_type", -1))
        try:
            ggml_spec(ggml_type)
        except (TypeError, ValueError) as exc:
            raise LMHeadContractError(f"{name}: {exc}") from exc
        if decodable is not None and ggml_type not in decodable:
            raise LMHeadContractError(
                f"{name} uses GGML type {type_name(ggml_type)}, which Pyrite cannot "
                "decode; it cannot serve as the LM head"
            )

    output = tensors.get(OUTPUT)
    if output is not None:
        check(OUTPUT, output, "output projection")
        return LMHead(
            tensor_name=OUTPUT,
            tied=False,
            rows=vocab.size,
            cols=hidden_size,
            ggml_type=int(output.ggml_type),
            declared_tied=declared,
            source=(
                "output.weight is present, so it is the LM head"
                + (" even though the checkpoint declares tied embeddings" if declared else "")
            ),
        )

    if declared is False:
        raise LMHeadContractError(
            f"missing tensor: {OUTPUT} and the checkpoint explicitly declares "
            "tie_word_embeddings=false, so there is no LM output projection to use; "
            "re-export the checkpoint with the LM head included"
        )

    embedding = tensors.get(TOKEN_EMBD)
    if embedding is None:
        raise LMHeadContractError(
            f"missing tensor: {OUTPUT} and {TOKEN_EMBD}; neither an untied LM head nor "
            "a tied embedding matrix is available"
        )
    check(TOKEN_EMBD, embedding, "tied LM head (token_embd.weight)")
    return LMHead(
        tensor_name=TOKEN_EMBD,
        tied=True,
        rows=vocab.size,
        cols=hidden_size,
        ggml_type=int(embedding.ggml_type),
        declared_tied=declared,
        source=(
            "output.weight is absent, so token_embd.weight is reused as the LM head "
            + ("(checkpoint declares tied embeddings)" if declared else "(tied by omission)")
        ),
    )
