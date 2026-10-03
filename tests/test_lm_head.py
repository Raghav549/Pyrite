"""Regression tests for the LM-head / vocabulary contract.

The original failure was::

    missing tensor: output.weight (checkpoint does not declare tied embeddings)

raised on a *valid* GGUF.  These tests pin the three legitimate cases - untied,
tied, and genuinely missing - plus the vocab checks, so the bug cannot return
by silently assuming either shape.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from pyrite.lm_head import (
    LMHead,
    LMHeadContractError,
    Vocabulary,
    VocabularyContractError,
    declared_tied_embeddings,
    resolve_lm_head,
    resolve_vocabulary,
)


@dataclass(frozen=True)
class FakeTensor:
    dims: tuple[int, ...]
    ggml_type: int = 0
    size: int = 0


def tensors(**shapes: tuple[int, ...]) -> dict[str, FakeTensor]:
    return {name: FakeTensor(dims) for name, dims in shapes.items()}


def vocab(size: int = 64) -> Vocabulary:
    return Vocabulary(size=size, token_count=size, source="test")


# --------------------------------------------------------------------------- #
# LM head resolution
# --------------------------------------------------------------------------- #
def test_untied_checkpoint_uses_output_weight() -> None:
    """The exact regression: output.weight present, no tied declaration."""
    head = resolve_lm_head(
        tensors(**{"token_embd.weight": (64, 64), "output.weight": (64, 64)}),
        {},
        "qwen3",
        64,
        vocab(64),
    )
    assert head.tensor_name == "output.weight"
    assert head.tied is False
    assert head.declared_tied is None
    assert head.rows == 64 and head.cols == 64
    assert "output.weight is present" in head.source


def test_tied_checkpoint_reuses_the_embedding_as_lm_head() -> None:
    head = resolve_lm_head(
        tensors(**{"token_embd.weight": (64, 64)}),
        {"qwen3.tied_word_embeddings": True},
        "qwen3",
        64,
        vocab(64),
    )
    assert head.tensor_name == "token_embd.weight"
    assert head.tied is True
    assert head.declared_tied is True
    assert "tied" in head.source.lower()


def test_missing_lm_head_with_no_declaration_ties_and_says_so() -> None:
    """Absent output.weight and no tie flag: reuse the embedding, and say so.

    This is exactly what llama.cpp does - ``src/models/qwen3.cpp`` falls back to
    ``LLM_TENSOR_TOKEN_EMBD`` marked ``TENSOR_DUPLICATED`` with no flag check.
    The inference is recorded in ``source`` so it is never silent.
    """
    head = resolve_lm_head(
        tensors(**{"token_embd.weight": (64, 64)}),
        {},
        "qwen3",
        64,
        vocab(64),
    )
    assert head.tensor_name == "token_embd.weight"
    assert head.tied is True
    assert head.declared_tied is None
    assert "tied by omission" in head.source


def test_missing_both_output_and_embedding_fails_loudly() -> None:
    with pytest.raises(LMHeadContractError) as excinfo:
        resolve_lm_head(tensors(**{"attn_norm.weight": (64,)}), {}, "qwen3", 64, vocab(64))
    message = str(excinfo.value)
    assert "output.weight" in message
    assert "token_embd.weight" in message


def test_untied_declaration_with_no_output_weight_is_refused() -> None:
    with pytest.raises(LMHeadContractError):
        resolve_lm_head(
            tensors(**{"token_embd.weight": (64, 64)}),
            {"qwen3.tied_word_embeddings": False},
            "qwen3",
            64,
            vocab(64),
        )


def test_lm_head_shape_must_match_hidden_size_and_vocab() -> None:
    # rows must be hidden_size, cols must be vocab_size for the [hidden, vocab]
    # GGML convention Pyrite and llama.cpp both use.
    with pytest.raises(LMHeadContractError) as excinfo:
        resolve_lm_head(
            tensors(**{"output.weight": (32, 64)}),
            {},
            "qwen3",
            64,
            vocab(64),
        )
    assert "hidden" in str(excinfo.value).lower() or "shape" in str(excinfo.value).lower()


def test_lm_head_type_must_be_a_known_ggml_type() -> None:
    with pytest.raises(LMHeadContractError):
        resolve_lm_head(
            {"output.weight": FakeTensor((64, 64), ggml_type=9999)},
            {},
            "qwen3",
            64,
            vocab(64),
        )


def test_undecodable_lm_head_is_reported_only_when_asked() -> None:
    """Structural resolution must not depend on decoder availability.

    ``qwen3-check`` has to be able to *report* an undecodable type; refusing to
    run is the executor's job (``ensure_decodable``), not the contract's.
    """
    head = resolve_lm_head(
        {"output.weight": FakeTensor((64, 64), ggml_type=35)},
        {},
        "qwen3",
        64,
        vocab(64),
        decodable=None,
    )
    assert head.ggml_type == 35
    with pytest.raises(LMHeadContractError, match="cannot decode"):
        resolve_lm_head(
            {"output.weight": FakeTensor((64, 64), ggml_type=35)},
            {},
            "qwen3",
            64,
            vocab(64),
            decodable=frozenset({0}),
        )


def test_lm_head_dict_is_json_shaped() -> None:
    head = LMHead(
        tensor_name="output.weight",
        tied=False,
        rows=64,
        cols=64,
        ggml_type=0,
        declared_tied=None,
        source="test",
    )
    payload = head.to_dict()
    assert payload["tensor"] == "output.weight"
    assert payload["shape"] == [64, 64]
    assert payload["tied"] is False
    assert payload["declared_tied"] is None


def test_declared_tied_embeddings_reads_every_spelling() -> None:
    assert declared_tied_embeddings({"qwen3.tied_word_embeddings": True}, "qwen3") is True
    assert declared_tied_embeddings({"qwen3.tie_word_embeddings": False}, "qwen3") is False
    assert declared_tied_embeddings({}, "qwen3") is None


# --------------------------------------------------------------------------- #
# Vocabulary
# --------------------------------------------------------------------------- #
def test_vocab_token_list_wins_and_padding_is_allowed() -> None:
    """Converters append [PADn] tokens until the list reaches the tensor rows.

    So fewer *tokens* than embedding rows is normal; the token list is then the
    real vocabulary and the extra rows are padding.
    """
    resolved = resolve_vocabulary(
        {"tokenizer.ggml.tokens": ["a", "b", "c"], "qwen3.vocab_size": 3}, "qwen3", 99
    )
    assert resolved.size == 99
    assert resolved.token_count == 3
    # The projection width comes from the tensor; the token list only bounds
    # how many of those rows are real tokens.
    assert resolved.source == "token_embd.weight rows"


def test_vocab_metadata_conflict_is_refused() -> None:
    """A token list that disagrees with vocab_size contradicts the file."""
    with pytest.raises(VocabularyContractError) as excinfo:
        resolve_vocabulary(
            {"tokenizer.ggml.tokens": ["a", "b", "c"], "qwen3.vocab_size": 99}, "qwen3", 3
        )
    message = str(excinfo.value)
    assert "3" in message and "99" in message


def test_vocab_falls_back_to_embedding_rows() -> None:
    """A checkpoint with no vocab metadata still has a usable vocabulary."""
    resolved = resolve_vocabulary({}, "qwen3", 64)
    assert resolved.size == 64
    assert resolved.source == "token_embd.weight rows"


def test_vocab_zero_is_not_accepted_silently() -> None:
    with pytest.raises(VocabularyContractError):
        resolve_vocabulary({"qwen3.vocab_size": 0}, "qwen3", 0)


def test_vocab_conflict_between_metadata_and_embedding_is_refused() -> None:
    with pytest.raises(VocabularyContractError) as excinfo:
        resolve_vocabulary({"qwen3.vocab_size": 128}, "qwen3", 64)
    message = str(excinfo.value)
    assert "128" in message and "64" in message


# --------------------------------------------------------------------------- #
# End to end, on a real file
# --------------------------------------------------------------------------- #
def test_real_fixture_without_vocab_metadata_still_validates() -> None:
    """``qwen3-f32-fixture.gguf`` declares no vocab keys at all.

    It used to fail with "GGUF is missing required qwen3 metadata"; it must now
    resolve the vocabulary from the embedding rows and find its LM head.
    """
    from pyrite.dense import DenseCheckpoint

    path = Path("/home/user/models/qwen3-f32-fixture.gguf")
    if not path.is_file():
        pytest.skip("the downloaded fixture is not present on this machine")
    checkpoint = DenseCheckpoint(path)
    summary = checkpoint.validate_contract()
    assert summary["lm_head"]["tensor"] == "output.weight"
    assert summary["lm_head"]["tied"] is False
    assert checkpoint.config.vocab_source == "token_embd.weight rows"
    assert checkpoint.config.vocab_token_count == summary["vocab_token_count"]
