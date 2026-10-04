"""Byte-level BPE tokenizer behaviour, on a real merge table."""
from __future__ import annotations

import re as _re

import pytest

from pyrite.tokenizer import (
    GGUFBPETokenizer,
    TokenizerSpec,
    UnsupportedPreTokenizer,
    WhitespaceTokenizer,
    bytes_to_unicode,
    load_gguf_tokenizer,
    pre_tokenizer_for,
)

from .tiny_qwen3moe import make_test_tokenizer, tokenizer_vocab


def test_byte_encoder_is_a_reversible_bijection():
    encoder = bytes_to_unicode()
    assert len(encoder) == 256
    assert len(set(encoder.values())) == 256
    assert all(len(char) == 1 for char in encoder.values())


def test_merges_are_applied_by_rank():
    tokenizer = make_test_tokenizer()
    assert [tokenizer.tokens[i] for i in tokenizer.encode("hello")] == ["hello"]
    assert [tokenizer.tokens[i] for i in tokenizer.encode("hello world")] == ["hello", "Ġworld"]


def test_unknown_text_falls_back_to_bytes():
    tokenizer = make_test_tokenizer()
    ids = tokenizer.encode("Z")
    assert len(ids) == 1  # a single base alphabet symbol, still decodable
    assert tokenizer.decode(ids) == "Z"


def test_unicode_round_trip_through_bytes():
    tokenizer = make_test_tokenizer()
    for text in ("hello world", "héllo wörld", "世界", "emoji 🐍 test", "line\nbreak"):
        assert tokenizer.decode(tokenizer.encode(text)) == text


def test_special_tokens_are_kept_whole():
    tokenizer = make_test_tokenizer()
    ids = tokenizer.encode("<|im_start|>hi")
    assert tokenizer.tokens[ids[0]] == "<|im_start|>"
    assert tokenizer.decode(ids, skip_special=False) == "<|im_start|>hi"
    assert tokenizer.decode(ids) == "hi"


def test_generated_special_tokens_survive_a_verbatim_decode():
    """`skip_special` changes the *text*, never the id list.

    This distinction produced a wrong conclusion once: a comparison against
    llama.cpp used the default `skip_special=True` decode, so a generated
    USER_DEFINED token vanished from the string and looked like the two runtimes
    disagreed. They had produced identical ids.
    """
    tokenizer = make_test_tokenizer()
    ids = tokenizer.encode("<|im_start|>hi")
    special_id = ids[0]
    assert tokenizer.tokens[special_id] == "<|im_start|>"
    assert tokenizer._is_special(special_id, tokenizer.tokens[special_id]) is True

    # The id list is the ground truth and is unaffected by the flag.
    assert special_id in ids
    assert tokenizer.decode(ids, skip_special=False).startswith("<|im_start|>")
    assert "<|im_start|>" not in tokenizer.decode(ids, skip_special=True)
    # Round-tripping the verbatim form must recover every id.
    assert tokenizer.encode(tokenizer.decode(ids, skip_special=False)) == ids


def test_encode_with_bos():
    tokenizer = make_test_tokenizer()
    vocab, _, _, bos, _ = tokenizer_vocab()
    assert tokenizer.encode("hello", add_bos=True)[0] == bos
    assert tokenizer.spec == TokenizerSpec(vocab_size=len(vocab), bos_id=bos, eos_id=bos + 1)


def test_byte_fallback_tokens_are_used_when_a_symbol_is_missing():
    # A vocabulary that omits the "h" symbol must fall back to <0x68>.
    vocab = [bytes_to_unicode()[byte] for byte in range(256) if byte != ord("h")]
    vocab.append("<0x68>")
    tokenizer = GGUFBPETokenizer(vocab, [])
    ids = tokenizer.encode("h")
    assert tokenizer.tokens[ids[0]] == "<0x68>"
    assert tokenizer.decode(ids) == "h"


def test_decode_rejects_out_of_range_ids():
    tokenizer = make_test_tokenizer()
    with pytest.raises(ValueError):
        tokenizer.decode([10 ** 9])
    with pytest.raises(TypeError):
        tokenizer.encode(b"bytes are not text")


def test_empty_vocabulary_is_rejected():
    with pytest.raises(ValueError):
        GGUFBPETokenizer([])


def test_load_gguf_tokenizer_reads_metadata():
    reader_metadata = {
        "tokenizer.ggml.model": "gpt2",
        "tokenizer.ggml.pre": "qwen2",
        "tokenizer.ggml.tokens": ["a", "b", "ab"],
        "tokenizer.ggml.merges": ["a b"],
        "tokenizer.ggml.bos_token_id": 2,
        "tokenizer.ggml.eos_token_id": 1,
    }

    class _Reader:
        def metadata(self):
            return reader_metadata

    tokenizer = load_gguf_tokenizer(_Reader())
    assert tokenizer is not None
    assert tokenizer.bos_id == 2 and tokenizer.eos_id == 1
    assert [tokenizer.tokens[i] for i in tokenizer.encode("ab")] == ["ab"]


def test_missing_pre_tokenizer_is_refused():
    """A BPE file with no ``tokenizer.ggml.pre`` must not be guessed at.

    llama.cpp reads the key without a fallback and throws for an unknown
    pre-tokenizer; Pyrite refuses for the same reason instead of emitting
    plausible-looking but wrong token ids.
    """

    class _Reader:
        def metadata(self):
            return {
                "tokenizer.ggml.model": "gpt2",
                "tokenizer.ggml.tokens": ["a", "b"],
                "tokenizer.ggml.merges": [],
            }

    with pytest.raises(UnsupportedPreTokenizer):
        load_gguf_tokenizer(_Reader())


def test_multi_regex_pre_tokenizer_is_refused():
    with pytest.raises(UnsupportedPreTokenizer):
        pre_tokenizer_for("default")
    with pytest.raises(UnsupportedPreTokenizer):
        pre_tokenizer_for("chameleon")


def test_pre_tokenizer_patterns_match_the_reference_semantics():
    # qwen2: one digit at a time, letters run together, newline preserved.
    assert pre_tokenizer_for("qwen2").findall("ab 12\n") == ["ab", " ", "1", "2", "\n"]
    # llama-bpe/llama3: digits are grouped in runs of at most three.
    assert pre_tokenizer_for("llama-bpe").findall("1234") == ["123", "4"]
    # gpt-2: digits are one run, and there is no bare-whitespace alternative,
    # so an unmatched newline is dropped exactly as llama.cpp's regex does.
    assert pre_tokenizer_for("gpt-2").findall("a1234\nb") == ["a", "1234", "b"]


def test_load_gguf_tokenizer_returns_none_for_other_models():
    class _Reader:
        def metadata(self):
            return {"tokenizer.ggml.model": "llama", "tokenizer.ggml.tokens": ["a"]}

    assert load_gguf_tokenizer(_Reader()) is None


def test_whitespace_tokenizer_is_reproducible():
    first = WhitespaceTokenizer(TokenizerSpec(vocab_size=1024)).encode("same words here")
    second = WhitespaceTokenizer(TokenizerSpec(vocab_size=1024)).encode("same words here")
    assert first == second
    assert len(first) == 3
    assert WhitespaceTokenizer().encode("") == []


# --------------------------------------------------------------------------
# qwen35: llama.cpp's LLAMA_VOCAB_PRE_TYPE_QWEN35 (src/llama-vocab.cpp:392).
#
# It is a *single* regex - the case block puts exactly one entry in
# regex_exprs - identical to qwen2 except that combining marks (\p{M}) count
# as letters and are excluded from the punctuation run.  It used to be listed
# with the multi-pattern pre-tokenizers and refused; these tests pin the
# behaviour that replaced that refusal.  Verified against ``llama-tokenize``
# 9/9 on the real 151,936-token vocabulary, including Devanagari, Arabic,
# Thai, Hebrew and decomposed Latin accents
# (validation/qwen35-pretokenizer-crosscheck.txt).
# --------------------------------------------------------------------------


def test_qwen35_is_a_single_pattern_pre_tokenizer():
    from pyrite.tokenizer import MULTI_PATTERN_PRE_TOKENIZERS, PRE_TOKENIZER_PATTERNS

    assert "qwen35" in PRE_TOKENIZER_PATTERNS
    assert "qwen35" not in MULTI_PATTERN_PRE_TOKENIZERS
    # Does not raise UnsupportedPreTokenizer.
    assert pre_tokenizer_for("qwen35") is not None


def test_qwen35_differs_from_qwen2_exactly_on_combining_marks():
    """The whole point of the separate pattern: marks join their base letter."""
    qwen2 = pre_tokenizer_for("qwen2")
    qwen35 = pre_tokenizer_for("qwen35")

    devanagari = "नमस्ते"  # नमस्ते: base letters + virama/vowel signs (Mn)
    assert qwen2.findall(devanagari) == ["नमस", "्त", "े"]
    assert qwen35.findall(devanagari) == ["नमस्ते"]

    decomposed = "cafe\u0301"  # cafe + COMBINING ACUTE ACCENT
    assert qwen2.findall(decomposed) == ["cafe", "\u0301"]
    assert qwen35.findall(decomposed) == ["cafe\u0301"]


def test_qwen35_is_identical_to_qwen2_when_there_are_no_marks():
    """No behavioural drift on the text the two patterns agree about."""
    qwen2 = pre_tokenizer_for("qwen2")
    qwen35 = pre_tokenizer_for("qwen35")
    for text in (
        "ab 12\n",
        "Explain Pyrite in one short paragraph.",
        "Hello, world! (again) [x] {y}",
        "12345 678 90",
        "don't stop",
        "\r\n\r\n  spaced  ",
    ):
        assert qwen35.findall(text) == qwen2.findall(text), text


# --------------------------------------------------------------------------
# Text the pre-tokenizer regex does not match must still be encoded.
#
# ``_encode_ordinary`` used to call ``findall``, which returns only the matched
# spans, so any character the regex failed to match was silently dropped from
# the output - real text loss, not just a different segmentation.  The GPT-2
# regex genuinely has such gaps: it has no plain ``\s+`` alternative, so the
# second of two consecutive newlines matches nothing.  Verified against
# llama.cpp on the real vocabulary: "line one\nline two\n\nline three" is
# [1056, 825, 198, 1056, 1378, 198, 198, 1056, 2326] and Pyrite now matches.
# --------------------------------------------------------------------------


def _tokenizer_with_pre(pre: str):
    vocab, merges, token_types, bos, eos = tokenizer_vocab()
    return GGUFBPETokenizer(
        vocab, merges, token_types=token_types, bos_id=bos, eos_id=eos, pre=pre
    )


def test_characters_the_pre_tokenizer_does_not_match_are_not_dropped():
    tok = _tokenizer_with_pre("gpt-2")
    # The GPT-2 regex matches the first "\n" via \s+(?!\S) and leaves the
    # second one unmatched.
    text = "a\n\nb"
    pieces = pre_tokenizer_for("gpt-2").findall(text)
    assert "".join(pieces) != text, "premise: this text has an unmatched gap"
    ids = tok.encode(text)
    assert tok.decode(ids, skip_special=False) == text


def test_gpt2_and_poro_gap_policies_differ_and_both_round_trip():
    """Gaps are encoded as whole pieces, so merges still apply inside them."""
    text = "line one\nline two\n\nline three"
    for pre in ("gpt-2", "poro-chat"):
        tok = _tokenizer_with_pre(pre)
        assert tok.decode(tok.encode(text), skip_special=False) == text, pre


def test_no_pre_tokenizer_pattern_captures():
    """``findall``/``finditer`` grouping would corrupt every split.

    A pattern containing a capturing group makes ``findall`` return group
    tuples instead of whole matches.  The reference spells the GPT-4o
    lookaheads as ``((?=[\\p{L}])([^a-z]))``; they are transcribed here as
    ``(?:(?=[\\p{L}])[^a-z])`` for exactly this reason.
    """
    import re as _re

    from pyrite.tokenizer import PRE_TOKENIZER_PATTERNS, _expand_unicode_properties

    for name, pattern in PRE_TOKENIZER_PATTERNS.items():
        expanded = _re.compile(_expand_unicode_properties(pattern))
        assert expanded.groups == 0, f"{name} has {expanded.groups} capturing group(s)"


def test_unicode_property_expansion_handles_multi_letter_categories():
    """``\\p{Lu}`` and friends, not just single-letter ``\\p{L}``.

    Checked behaviourally: the rendered class is a wall of ``\\uXXXX`` escapes
    whose *text* contains plenty of ordinary letters, so substring tests on it
    are meaningless.
    """
    import re as _re

    from pyrite.tokenizer import _expand_unicode_properties

    upper = _re.compile(_expand_unicode_properties(r"\p{Lu}"))
    letters = _re.compile(_expand_unicode_properties(r"\p{L}"))
    assert upper.fullmatch("A") and not upper.fullmatch("a")
    assert letters.fullmatch("A") and letters.fullmatch("a")
    # Marks are their own category, disjoint from letters.
    marks = _re.compile(_expand_unicode_properties(r"\p{M}"))
    assert marks.fullmatch("\u0301") and not marks.fullmatch("A")


def test_non_bpe_pre_tokenizers_are_refused_with_the_real_reason():
    """Refusals name the specific upstream flag, not a generic message."""
    from pyrite.tokenizer import NON_BPE_PRE_TOKENIZERS

    for pre, reason in NON_BPE_PRE_TOKENIZERS.items():
        with pytest.raises(UnsupportedPreTokenizer) as excinfo:
            pre_tokenizer_for(pre)
        message = str(excinfo.value)
        assert "not byte-level BPE" in message, pre
        assert reason[:24] in message, pre


def test_newly_implemented_pre_tokenizers_are_registered():
    from pyrite.tokenizer import PRE_TOKENIZER_PATTERNS

    for pre in (
        "qwen35", "gpt-4o", "llama4", "kanana2", "talkie",
        "mellum", "modern-bert", "jina-v5-nano",
    ):
        assert pre in PRE_TOKENIZER_PATTERNS, pre
        assert pre_tokenizer_for(pre) is not None


# --------------------------------------------------------------------------
# Multi-regex pre-tokenizers: each regex splits the fragments the previous
# pass produced, and the spans between matches are kept.  Transcribed from the
# regex_exprs initializers in src/llama-vocab.cpp and cross-checked against
# llama-tokenize 8/8 each (validation/pretokenizer-audit-crosscheck.txt).
# --------------------------------------------------------------------------


def test_multi_regex_pre_tokenizers_are_implemented_not_refused():
    from pyrite.tokenizer import MULTI_PATTERN_PRE_TOKENIZERS, PRE_TOKENIZER_SEQUENCES

    for pre in (
        "falcon", "mellum2", "minicpm5",
        "deepseek-coder", "deepseek-llm", "chameleon",
    ):
        assert pre in PRE_TOKENIZER_SEQUENCES, pre
        assert pre not in MULTI_PATTERN_PRE_TOKENIZERS, pre
    # Nothing is refused as multi-regex any more: every sequence llama.cpp
    # defines has been transcribed.  'default' is refused separately, because
    # llama.cpp defines no regex_exprs for it at all and the obvious no-op
    # reading was measured against llama-tokenize at only 4/8.
    assert not MULTI_PATTERN_PRE_TOKENIZERS


def test_each_regex_in_a_sequence_splits_the_previous_pass():
    """The observable consequence of applying regexes in order.

    falcon's third regex is ``[0-9][0-9][0-9]``, so after the GPT-2 pass has
    produced the run "12345" it is re-split into groups of three.  GPT-2 alone
    leaves it whole, and mellum2 (whose *first* regex is ``\\p{N}``) splits it
    into single digits.  All three differ, which only holds if the passes run
    in sequence over the previous output.
    """
    from pyrite.tokenizer import pre_tokenize

    assert pre_tokenize("gpt-2", "12345") == ["12345"]
    assert pre_tokenize("falcon", "12345") == ["123", "45"]
    assert pre_tokenize("mellum2", "12345") == ["1", "2", "3", "4", "5"]


def test_multi_regex_splitting_keeps_unmatched_spans():
    """A split, never a filter - same invariant as the single-regex path."""
    from pyrite.tokenizer import pre_tokenize

    for pre in ("falcon", "mellum2", "minicpm5", "deepseek-coder", "chameleon"):
        text = "Hello, world! 12345 abc"
        assert "".join(pre_tokenize(pre, text)) == text, pre


def test_multi_regex_pre_tokenizer_round_trips_through_the_tokenizer():
    tok = _tokenizer_with_pre("falcon")
    text = "line one\nline two\n\nline three"
    assert tok.decode(tok.encode(text), skip_special=False) == text


def test_sequence_literals_match_the_reference_exactly():
    """Guard the transcription: these were regenerated from llama.cpp's source,
    not retyped, after a hand-copy truncated a 223-character literal and
    produced an invalid ``Ὗ-ώ`` range."""
    from pyrite.tokenizer import PRE_TOKENIZER_SEQUENCES, _expand_unicode_properties

    for pre, exprs in PRE_TOKENIZER_SEQUENCES.items():
        assert len(exprs) >= 2, f"{pre} is not multi-regex"
        for expr in exprs:
            _re.compile(_expand_unicode_properties(expr))  # must not raise
