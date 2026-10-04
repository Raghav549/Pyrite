"""Byte-level BPE tokenizer behaviour, on a real merge table."""
from __future__ import annotations

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
