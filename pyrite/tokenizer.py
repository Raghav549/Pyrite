"""Tokenizers.

:class:`GGUFBPETokenizer` implements the byte-level BPE stored in a GGUF file
(``tokenizer.ggml.tokens`` / ``tokenizer.ggml.merges``) so Pyrite can tokenize a
real prompt exactly the way the model was trained.  :class:`WhitespaceTokenizer`
is a deterministic stand-in used by scheduler/session tests; it is not a model
tokenizer.
"""
from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

#: ``tokenizer.ggml.token_type`` values, from ``gguf.constants.TokenType``.
TOKEN_TYPE_NORMAL = 1
TOKEN_TYPE_UNKNOWN = 2
TOKEN_TYPE_CONTROL = 3
TOKEN_TYPE_USER_DEFINED = 4
TOKEN_TYPE_UNUSED = 5
TOKEN_TYPE_BYTE = 6

#: llama.cpp treats control, user-defined and unknown tokens as "special": they
#: are matched verbatim in the prompt and are not produced by the BPE pass
#: (``llama-vocab.cpp``: ``attr & (CONTROL | USER_DEFINED | UNKNOWN)``).
SPECIAL_TOKEN_TYPES = frozenset(
    {TOKEN_TYPE_CONTROL, TOKEN_TYPE_USER_DEFINED, TOKEN_TYPE_UNKNOWN}
)


class UnsupportedPreTokenizer(ValueError):
    """Raised for a ``tokenizer.ggml.pre`` Pyrite does not implement exactly."""


# ---------------------------------------------------------------------------
# Pre-tokenizer patterns
#
# ``tokenizer.ggml.pre`` selects the regex llama.cpp uses to split text before
# BPE (``LLAMA_VOCAB_PRE_TYPE_*`` in ``src/llama-vocab.cpp``).  Using the wrong
# one produces silently wrong token ids, so Pyrite implements only the
# single-pattern pre-tokenizers it can reproduce exactly and refuses the rest
# instead of guessing.  The patterns below are transcribed from that file;
# ``\\p{...}`` classes are expanded from ``unicodedata`` at first use.
#
# The ``(?:'[sS]|...)`` spelling is deliberate: ``(?i:...)`` scoped inline flags
# only exist from Python 3.11 and Pyrite supports 3.10.
# ---------------------------------------------------------------------------
_QWEN2_PATTERN = (
    r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])"
    r"|[^\r\n\p{L}\p{N}]?\p{L}+"
    r"|\p{N}"
    r"| ?[^\s\p{L}\p{N}]+[\r\n]*"
    r"|\s*[\r\n]+"
    r"|\s+(?!\S)"
    r"|\s+"
)
# llama.cpp: LLAMA_VOCAB_PRE_TYPE_QWEN35.  Identical to QWEN2 except that
# combining marks (\p{M}) count as letters and are excluded from the
# punctuation run.  It is a *single* regex - src/llama-vocab.cpp:392-397 puts
# exactly one entry in regex_exprs - so it is reproducible exactly and must not
# be lumped in with the multi-pattern pre-tokenizers.
_QWEN35_PATTERN = (
    r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])"
    r"|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+"
    r"|\p{N}"
    r"| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*"
    r"|\s*[\r\n]+"
    r"|\s+(?!\S)"
    r"|\s+"
)
_LLAMA_BPE_PATTERN = (
    r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])"
    r"|[^\r\n\p{L}\p{N}]?\p{L}+"
    r"|\p{N}{1,3}"
    r"| ?[^\s\p{L}\p{N}]+[\r\n]*"
    r"|\s*[\r\n]+"
    r"|\s+(?!\S)"
    r"|\s+"
)
_GPT2_PATTERN = (
    r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)"
)
# llama.cpp: LLAMA_VOCAB_PRE_TYPE_GPT4O / MINIMAX_M2 (llama-vocab.cpp:438).
# The ``\p{Lu}``-heavy spelling above it in that file is a *comment* quoting the
# original tokenizer.json; the literal actually compiled uses only \p{L}/\p{N}
# plus lookaheads, which is why this is reproducible at all.
#
# The reference writes its lookaheads as ``((?=[\p{L}])([^a-z]))``.  The capture
# groups are rewritten as ``(?:(?=[\p{L}])[^a-z])`` because this module splits
# with ``findall``, which returns *group tuples* rather than whole matches for
# any pattern containing a capturing group - see
# ``test_no_pre_tokenizer_pattern_captures``.
_GPT4O_PATTERN = (
    r"[^\r\n\p{L}\p{N}]?(?:(?=[\p{L}])[^a-z])*(?:(?=[\p{L}])[^A-Z])+"
    r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?"
    r"|[^\r\n\p{L}\p{N}]?(?:(?=[\p{L}])[^a-z])+(?:(?=[\p{L}])[^A-Z])*"
    r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?"
    r"|\p{N}{1,3}"
    r"| ?[^\s\p{L}\p{N}]+[\r\n/]*"
    r"|\s*[\r\n]+"
    r"|\s+(?!\S)"
    r"|\s+"
)
# llama.cpp: LLAMA_VOCAB_PRE_TYPE_TEKKEN - the same case-splitting lookaheads
# without the contraction suffix, and digits one at a time.
_TEKKEN_PATTERN = (
    r"[^\r\n\p{L}\p{N}]?(?:(?=[\p{L}])[^a-z])*(?:(?=[\p{L}])[^A-Z])+"
    r"|[^\r\n\p{L}\p{N}]?(?:(?=[\p{L}])[^a-z])+(?:(?=[\p{L}])[^A-Z])*"
    r"|\p{N}"
    r"| ?[^\s\p{L}\p{N}]+[\r\n/]*"
    r"|\s*[\r\n]+"
    r"|\s+(?!\S)"
    r"|\s+"
)
_GEMMA4_PATTERN = r"[^\n]+|[\n]+"  # llama.cpp: LLAMA_VOCAB_PRE_TYPE_GEMMA4
_WHITESPACE_PATTERN = r"\S+"  # llama.cpp: LLAMA_VOCAB_PRE_TYPE_WHITESPACE
_PORO_PATTERN = r" ?[^\(\s|.,!?…。，、।۔،)]+"  # noqa: RUF001 - these code points are in the reference regex

#: ``tokenizer.ggml.pre`` value -> single-regex pre-tokenizer.
PRE_TOKENIZER_PATTERNS: dict[str, str] = {
    # llama.cpp: LLAMA_VOCAB_PRE_TYPE_QWEN2 / STABLELM2 / HUNYUAN / SOLAR_OPEN
    "qwen2": _QWEN2_PATTERN,
    "qwen35": _QWEN35_PATTERN,
    # llama.cpp: LLAMA_VOCAB_PRE_TYPE_GPT4O also backs llama4, kanana2, talkie.
    "gpt-4o": _GPT4O_PATTERN,
    "llama4": _GPT4O_PATTERN,
    "kanana2": _GPT4O_PATTERN,
    "talkie": _GPT4O_PATTERN,
    # llama.cpp: LLAMA_VOCAB_PRE_TYPE_GPT2 also backs mellum and modern-bert.
    "mellum": _GPT2_PATTERN,
    "modern-bert": _GPT2_PATTERN,
    # llama.cpp: LLAMA_VOCAB_PRE_TYPE_LLAMA3 also backs jina-v5-nano.
    "jina-v5-nano": _LLAMA_BPE_PATTERN,
    "deepseek-r1-qwen": _QWEN2_PATTERN,
    "kormo": _QWEN2_PATTERN,
    "f2llmv2": _QWEN2_PATTERN,
    "hunyuan": _QWEN2_PATTERN,
    "solar-open": _QWEN2_PATTERN,
    "stablelm2": _QWEN2_PATTERN,
    # LLAMA_VOCAB_PRE_TYPE_LLAMA3 / LLAMA_BPE / DBRX / SMAUG
    "llama3": _LLAMA_BPE_PATTERN,
    "llama-v3": _LLAMA_BPE_PATTERN,
    "llama-bpe": _LLAMA_BPE_PATTERN,
    "dbrx": _LLAMA_BPE_PATTERN,
    "smaug-bpe": _LLAMA_BPE_PATTERN,
    # LLAMA_VOCAB_PRE_TYPE_GPT2 / MPT / OLMO / JAIS / TRILLION
    "gpt-2": _GPT2_PATTERN,
    "mpt": _GPT2_PATTERN,
    "olmo": _GPT2_PATTERN,
    "jais": _GPT2_PATTERN,
    "trillion": _GPT2_PATTERN,
    # LLAMA_VOCAB_PRE_TYPE_STARCODER / REFACT / COMMAND_R / SMOLLM / EXAONE
    "starcoder": _GPT2_PATTERN,
    "refact": _GPT2_PATTERN,
    "command-r": _GPT2_PATTERN,
    "smollm": _GPT2_PATTERN,
    "codeshell": _GPT2_PATTERN,
    "exaone": _GPT2_PATTERN,
    # LLAMA_VOCAB_PRE_TYPE_PORO / BLOOM / GPT3_FINNISH
    "poro-chat": _PORO_PATTERN,
    "bloom": _PORO_PATTERN,
    "gpt3-finnish": _PORO_PATTERN,
}

#: ``tokenizer.ggml.pre`` values whose llama.cpp pre-tokenizer is a *list* of
#: regexes applied in sequence.  A joined alternation is not equivalent, so
#: Pyrite refuses them rather than emitting plausible-looking wrong ids.
# ---------------------------------------------------------------------------
# Multi-regex pre-tokenizers.  llama.cpp applies these *in sequence*: each
# regex splits every fragment the previous pass produced, and the text between
# matches is kept, exactly as for the single-regex case.  Transcribed from the
# ``regex_exprs`` initializers in ``src/llama-vocab.cpp``.
# ---------------------------------------------------------------------------
_DEEPSEEK_CODER_SEQUENCE = (
    '[\r\n]',
    '\\s?\\p{L}+',
    '\\s?\\p{P}+',
    '[一-龥ࠀ-一가-\ud7ff]+',
    '\\p{N}',
)

_DEEPSEEK_LLM_SEQUENCE = (
    '[\r\n]',
    '\\s?[A-Za-zµÀ-ÖØ-öø-ƺƼ-ƿǄ-ʓʕ-ʯͰ-ͳͶͷͻ-ͽͿΆΈ-ΊΌΎ-ΡΣ-ϵϷ-ҁҊ-ԯԱ-ՖႠ-ჅᎠ-Ᏽᏸ-ᏽᲐ-ᲺᲽ-Ჿᴀ-ᴫᵫ-ᵷᵹ-ᶚḀ-ἕἘ-Ἕἠ-ὅὈ-Ὅὐ-ὗὙὛὝὟ-ώᾀ-ᾴᾶ-ᾼιῂ-ῄῆ-ῌῐ-ΐῖ-Ίῠ-Ῥῲ-ῴῶ-ῼℂℇℊ-ℓℕℙ-ℝℤΩℨK-ℭℯ-ℴℹℼ-ℿⅅ-ⅉⅎↃↄⰀ-ⱻⱾ-ⳤⳫ-ⳮⳲⳳꙀ-ꙭꚀ-ꚛꜢ-ꝯꝱ-ꞇꞋ-ꞎꭰ-ꮿﬀ-ﬆﬓ-ﬗＡ-Ｚａ-ｚ𐐀-𐑏𐒰-𐓓𐓘-𐓻𐲀-𐲲𐳀-𐳲𑢠-𑣟𞤀-𞥃]+',  # noqa: RUF001
    '\\s?[!-/:-~！-／：-～‘-‟\u3000-。]+',  # noqa: RUF001
    '\\s+$',
    '[一-龥ࠀ-一가-\ud7ff]+',
    '\\p{N}+',
)

_CHAMELEON_SEQUENCE = (
    '<sentinel:[0-9]+>',
    '(IMGIMG)((A|B|C|D|E|F|G|H|I){1,4})Z',
    '([\\t\\n]|    |  )',
    '\\p{N}',
    '[\\p{P}!-/:-@\\[-`{-~]',
    _GPT2_PATTERN,
)

_FALCON_SEQUENCE = (
    '[\\p{P}\\$\\+<=>\\^~\\|`]+',
    _GPT2_PATTERN,
    '[0-9][0-9][0-9]',
)

_MELLUM2_SEQUENCE = (
    '\\p{N}',
    _GPT2_PATTERN,
)

_MINICPM5_SEQUENCE = (
    '\\p{N}{1,3}',
    "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}+| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
)

#: ``tokenizer.ggml.pre`` value -> ordered regex sequence.
PRE_TOKENIZER_SEQUENCES: dict[str, tuple[str, ...]] = {
    "falcon": _FALCON_SEQUENCE,
    "mellum2": _MELLUM2_SEQUENCE,
    "minicpm5": _MINICPM5_SEQUENCE,
    "deepseek-coder": _DEEPSEEK_CODER_SEQUENCE,
    "deepseek-llm": _DEEPSEEK_LLM_SEQUENCE,
    "chameleon": _CHAMELEON_SEQUENCE,
}

#: ``pre`` values whose regex Pyrite could transcribe but whose *surrounding*
#: algorithm differs from byte-level BPE, so a correct-looking regex would still
#: produce wrong ids.  Each carries the specific reason, read off
#: ``src/llama-vocab.cpp`` rather than guessed.
NON_BPE_PRE_TOKENIZERS: dict[str, str] = {
    "tekken": (
        "sets ignore_merges=true, so a pre-token piece that exists verbatim in "
        "the vocabulary is emitted directly instead of being BPE-merged, and "
        "add_bos=true, which prepends a token Pyrite does not add"
    ),
    "gemma4": (
        "uses SPM-style normalization (spaces become U+2581 before BPE) with "
        "byte_encode=false, i.e. raw UTF-8 rather than GPT-2 byte encoding"
    ),
    "granite-embed-multi-311m": (
        "uses the GEMMA4 path: SPM-style normalization with byte_encode=false"
    ),
    "granite-embed-multi-97m": (
        "sets ignore_merges=true, so whole pieces bypass BPE merging"
    ),
    "default": (
        "has no regex_exprs in llama.cpp at all - the vector is declared empty "
        "and the switch has no case LLAMA_VOCAB_PRE_TYPE_DEFAULT - so what it "
        "should do is undefined. The obvious reading, a no-op split that hands "
        "the whole text to BPE, was implemented and measured: it matched "
        "llama-tokenize on only 4 of 8 cases, merging 't where the reference "
        "does not and \\n\\n where the reference emits two \\n. Refused "
        "rather than shipped wrong"
    ),
    "whitespace": (
        "discards the text its \\S+ regex leaves unmatched: measured on a real "
        "vocabulary, llama.cpp tokenizes 'line one\\nline two' to "
        "[1056, 603, 1056, 19789] while encoding the unmatched whitespace "
        "gives [1056, 220, 603, 198, ...]. The reason is not established, so "
        "Pyrite refuses rather than guess"
    ),
}

MULTI_PATTERN_PRE_TOKENIZERS: frozenset[str] = frozenset(
    {
    }
)

_UNICODE_CLASSES: dict[str, str] = {}


def _unicode_ranges(prefix: str) -> str:
    """Exact ``\\p{X}`` member list (no brackets) built from :mod:`unicodedata`.

    Returned without surrounding brackets so the same ranges can be embedded
    inside a larger ``[...]`` class as well as used standalone.
    """
    cached = _UNICODE_CLASSES.get(prefix)
    if cached is not None:
        return cached
    import unicodedata

    codepoints: list[int] = []
    for codepoint in range(0x110000):
        if 0xD800 <= codepoint <= 0xDFFF:
            continue
        if unicodedata.category(chr(codepoint)).startswith(prefix):
            codepoints.append(codepoint)
    if not codepoints:
        raise UnsupportedPreTokenizer(f"no Unicode {prefix}* characters found")
    parts: list[str] = []
    start = previous = codepoints[0]
    for codepoint in [*codepoints[1:], -1]:
        if codepoint == previous + 1:
            previous = codepoint
            continue
        parts.append(_escape_range(start, previous))
        start = previous = codepoint
    rendered = "".join(parts)
    _UNICODE_CLASSES[prefix] = rendered
    return rendered


def _escape_range(start: int, end: int) -> str:
    def one(codepoint: int) -> str:
        char = chr(codepoint)
        if char in "[]\\-^":
            return "\\" + char
        if 0x20 <= codepoint < 0x7F:
            return char
        if codepoint < 0x10000:
            return f"\\u{codepoint:04x}"
        return f"\\U{codepoint:08x}"

    return one(start) if start == end else f"{one(start)}-{one(end)}"


def _expand_unicode_properties(pattern: str) -> str:
    """Replace ``\\p{L}``-style classes with exact Python ``re`` classes.

    Inside a ``[...]`` class the ranges are spliced in bare; outside they are
    wrapped in brackets.  Getting this wrong silently changes the split.
    """
    out: list[str] = []
    index = 0
    in_class = False
    length = len(pattern)
    while index < length:
        char = pattern[index]
        if char == "\\":
            if pattern.startswith("\\p{", index):
                close = pattern.find("}", index)
                category = pattern[index + 3: close] if close != -1 else ""
                if category:
                    # ``_unicode_ranges`` matches by category prefix, so both
                    # ``\\p{L}`` (Lu|Ll|Lt|Lm|Lo) and ``\\p{Lu}`` work.
                    ranges = _unicode_ranges(category)
                    out.append(ranges if in_class else "[" + ranges + "]")
                    index = close + 1
                    continue
            out.append(pattern[index: index + 2])
            index += 2
            continue
        if char == "[":
            in_class = True
        elif char == "]":
            in_class = False
        out.append(char)
        index += 1
    return "".join(out)


_COMPILED_PRE_TOKENIZERS: dict[str, re.Pattern[str]] = {}


_COMPILED_SEQUENCES: dict[str, tuple[re.Pattern[str], ...]] = {}


def _compiled_sequence(pre: str) -> tuple[re.Pattern[str], ...]:
    cached = _COMPILED_SEQUENCES.get(pre)
    if cached is None:
        cached = tuple(
            re.compile(_expand_unicode_properties(expr))
            for expr in PRE_TOKENIZER_SEQUENCES[pre]
        )
        _COMPILED_SEQUENCES[pre] = cached
    return cached


def _split_with(pattern: re.Pattern[str], text: str) -> list[str]:
    """Split ``text`` on ``pattern``, keeping the spans between matches.

    A split, never a filter: dropping the unmatched spans would silently lose
    input, which is the bug ``test_characters_the_pre_tokenizer_does_not_match
    _are_not_dropped`` guards against.
    """
    pieces: list[str] = []
    position = 0
    for match in pattern.finditer(text):
        if match.start() > position:
            pieces.append(text[position: match.start()])
        pieces.append(match.group(0))
        position = match.end()
    if position < len(text):
        pieces.append(text[position:])
    return pieces


def pre_tokenize(pre: str, text: str) -> list[str]:
    """Pre-tokenize ``text`` for a ``tokenizer.ggml.pre`` value.

    Single-regex values are one pass; multi-regex values apply each regex in
    turn to the fragments the previous pass produced, which is what
    ``unicode_regex_split`` does in llama.cpp.
    """
    if not text:
        return []
    if pre in PRE_TOKENIZER_SEQUENCES:
        fragments = [text]
        for pattern in _compiled_sequence(pre):
            fragments = [piece for fragment in fragments for piece in _split_with(pattern, fragment)]
        return fragments
    return _split_with(pre_tokenizer_for(pre), text)


def pre_tokenizer_for(pre: str) -> re.Pattern[str]:
    """Compiled pre-tokenizer for a single-regex ``tokenizer.ggml.pre`` value.

    Multi-regex values have no single pattern; use :func:`pre_tokenize`.
    """
    cached = _COMPILED_PRE_TOKENIZERS.get(pre)
    if cached is not None:
        return cached
    if pre in NON_BPE_PRE_TOKENIZERS:
        raise UnsupportedPreTokenizer(
            f"tokenizer.ggml.pre={pre!r} is not byte-level BPE: it "
            f"{NON_BPE_PRE_TOKENIZERS[pre]}. Pyrite's tokenizer applies GPT-2 "
            "byte encoding and BPE merges, so it would emit plausible-looking "
            "but wrong ids; refusing instead"
        )
    if pre in MULTI_PATTERN_PRE_TOKENIZERS:
        raise UnsupportedPreTokenizer(
            f"tokenizer.ggml.pre={pre!r} uses a multi-regex pre-tokenizer that Pyrite "
            "does not approximate; refusing to emit token ids it cannot match exactly"
        )
    pattern = PRE_TOKENIZER_PATTERNS.get(pre)
    if pattern is None:
        raise UnsupportedPreTokenizer(
            f"tokenizer.ggml.pre={pre!r} has no Pyrite implementation "
            "(llama.cpp rejects an unknown pre-tokenizer for the same reason); "
            "supported values: " + ", ".join(sorted(PRE_TOKENIZER_PATTERNS))
        )
    compiled = re.compile(_expand_unicode_properties(pattern))
    _COMPILED_PRE_TOKENIZERS[pre] = compiled
    return compiled


_BYTE_FALLBACK = re.compile(r"^<0x([0-9A-Fa-f]{2})>$")


def bytes_to_unicode() -> dict[int, str]:
    """GPT-2 byte encoder: reversible mapping from byte values to characters."""
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("\xa1"), ord("\xac") + 1))
        + list(range(ord("\xae"), ord("\xff") + 1))
    )
    cs = bs[:]
    n = 0
    for byte in range(256):
        if byte not in bs:
            bs.append(byte)
            cs.append(256 + n)
            n += 1
    return {byte: chr(code) for byte, code in zip(bs, cs, strict=True)}


_BYTE_ENCODER = bytes_to_unicode()
_BYTE_DECODER = {char: byte for byte, char in _BYTE_ENCODER.items()}


@dataclass(frozen=True)
class TokenizerSpec:
    vocab_size: int
    bos_id: int | None = None
    eos_id: int | None = None
    pad_id: int | None = None


class WhitespaceTokenizer:
    """Deterministic word-level stand-in tokenizer (not an LLM tokenizer)."""

    def __init__(self, spec: TokenizerSpec | None = None):
        self.spec = spec or TokenizerSpec(vocab_size=65536)

    def encode(self, text: str) -> list[int]:
        if not text:
            return []
        return [self._token_id(token) for token in text.split()]

    def _token_id(self, token: str) -> int:
        # Python's hash() is randomized per process; a stable digest keeps token
        # ids reproducible across runs and machines.
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(digest, "big") % self.spec.vocab_size

    def decode(self, tokens: list[int]) -> str:
        return repr(list(tokens))


class GGUFBPETokenizer:
    """Byte-level BPE tokenizer backed by GGUF tokenizer metadata."""

    def __init__(
        self,
        tokens: Sequence[str],
        merges: Iterable[str] = (),
        *,
        bos_id: int | None = None,
        eos_id: int | None = None,
        pad_id: int | None = None,
        token_types: Sequence[int] | None = None,
        model: str = "gpt2",
        pre: str = "gpt-2",
        add_bos: bool = False,
    ):
        if not tokens:
            raise ValueError("tokenizer vocabulary is empty")
        self.model = model
        self.tokens = list(tokens)
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.pad_id = pad_id
        self.token_types = list(token_types) if token_types else None
        self.pre = pre
        self.add_bos = bool(add_bos)
        # Resolved eagerly so an unsupported pre-tokenizer is reported when the
        # tokenizer is built, not halfway through encoding a prompt.
        # Multi-regex pre-tokenizers have no single pattern; the split is done
        # by pre_tokenize(), which handles both shapes.
        self.pre_tokenizer = (
            None if pre in PRE_TOKENIZER_SEQUENCES else pre_tokenizer_for(pre)
        )
        self._token_to_id: dict[str, int] = {}
        for index, token in enumerate(self.tokens):
            self._token_to_id.setdefault(token, index)

        self.ranks: dict[tuple[str, str], int] = {}
        for rank, merge in enumerate(merges):
            parts = merge.split(" ", 1)
            if len(parts) != 2 or not parts[0] or not parts[1]:
                continue
            self.ranks.setdefault((parts[0], parts[1]), rank)

        self.special_tokens: tuple[str, ...] = tuple(
            sorted(
                (
                    token
                    for index, token in enumerate(self.tokens)
                    if self._is_special(index, token)
                ),
                key=len,
                reverse=True,
            )
        )

    def _is_special(self, index: int, token: str) -> bool:
        """Whether ``token`` is matched verbatim instead of produced by BPE.

        ``gguf.constants.TokenType`` is 1 NORMAL, 2 UNKNOWN, 3 CONTROL,
        4 USER_DEFINED, 5 UNUSED, 6 BYTE.  llama.cpp treats CONTROL,
        USER_DEFINED and UNKNOWN as special.  Byte tokens (6) are *not*
        special: they are the ``<0xXX>`` fallback alphabet.  When a file
        carries no token-type array, the ``<|...|>`` heuristic is used.
        """
        if self.token_types is not None and index < len(self.token_types):
            return int(self.token_types[index]) in SPECIAL_TOKEN_TYPES
        return token.startswith("<|") and token.endswith("|>")

    @property
    def vocab_size(self) -> int:
        return len(self.tokens)

    @property
    def spec(self) -> TokenizerSpec:
        return TokenizerSpec(len(self.tokens), self.bos_id, self.eos_id, self.pad_id)

    # ------------------------------------------------------------------ encode
    def encode(
        self,
        text: str,
        *,
        add_bos: bool | None = None,
        allow_special: bool = True,
    ) -> list[int]:
        """Encode ``text``; ``add_bos=None`` honours ``tokenizer.ggml.add_bos_token``."""
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        ids: list[int] = []
        emit_bos = self.add_bos if add_bos is None else add_bos
        if emit_bos and self.bos_id is not None:
            ids.append(self.bos_id)
        for chunk, is_special in self._split_special(text, allow_special):
            if is_special:
                token_id = self._token_to_id.get(chunk)
                if token_id is not None:
                    ids.append(token_id)
                continue
            ids.extend(self._encode_ordinary(chunk))
        return ids

    def _split_special(self, text: str, allow_special: bool) -> list[tuple[str, bool]]:
        if not allow_special or not self.special_tokens or not text:
            return [(text, False)] if text else []
        pattern = re.compile("|".join(re.escape(token) for token in self.special_tokens))
        result: list[tuple[str, bool]] = []
        position = 0
        for match in pattern.finditer(text):
            if match.start() > position:
                result.append((text[position: match.start()], False))
            result.append((match.group(0), True))
            position = match.end()
        if position < len(text):
            result.append((text[position:], False))
        return result

    def _encode_ordinary(self, text: str) -> list[int]:
        """Encode text the pre-tokenizer regex covers, *and* the text it does not.

        ``findall`` returns only the matched spans, so any character the
        pre-tokenizer does not match would be silently dropped - losing input.
        The GPT-2 regex really does have gaps: it has no plain ``\\s+``
        alternative, so a lone newline before a letter matches nothing, and
        llama.cpp encodes that newline (token 198) rather than discarding it.
        The spans between matches are therefore encoded too - as a single piece
        each, so BPE merges still apply inside them.  That detail is
        observable: for the poro/bloom regex (which excludes whitespace
        entirely) the gap ``"\\n\\n"`` encodes to the single merged token 271,
        while the GPT-2 regex matches the first of two newlines via
        ``\\s+(?!\\S)`` and leaves the second as a one-character gap that
        encodes to 198.  Both agree with llama.cpp only with whole-piece gaps.
        """
        return [
            token_id
            for piece in pre_tokenize(self.pre, text)
            for token_id in self._encode_piece(piece)
        ]



    def _encode_piece(self, piece: str) -> list[int]:
        result: list[int] = []
        symbols = [_BYTE_ENCODER[byte] for byte in piece.encode("utf-8")]
        for symbol in self._apply_merges(symbols):
            token_id = self._lookup(symbol)
            if token_id is None:
                result.extend(self._byte_fallback(symbol))
            else:
                result.append(token_id)
        return result

    def _apply_merges(self, symbols: list[str]) -> list[str]:
        if len(symbols) < 2:
            return symbols
        while len(symbols) > 1:
            best_rank: int | None = None
            best_index = -1
            for index in range(len(symbols) - 1):
                rank = self.ranks.get((symbols[index], symbols[index + 1]))
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank = rank
                    best_index = index
            if best_rank is None:
                break
            symbols = [
                *symbols[:best_index],
                symbols[best_index] + symbols[best_index + 1],
                *symbols[best_index + 2:],
            ]
        return symbols

    def _lookup(self, symbol: str) -> int | None:
        return self._token_to_id.get(symbol)

    def _byte_fallback(self, symbol: str) -> list[int]:
        """Fall back to ``<0xXX>`` byte tokens, then to unknown bytes."""
        result: list[int] = []
        for char in symbol:
            byte = _BYTE_DECODER.get(char)
            if byte is None:
                continue
            fallback = self._token_to_id.get(f"<0x{byte:02X}>")
            if fallback is not None:
                result.append(fallback)
        return result

    # ------------------------------------------------------------------ decode
    def decode(self, tokens: Sequence[int], *, skip_special: bool = True) -> str:
        data = bytearray()
        for token in tokens:
            if not isinstance(token, int) or not 0 <= token < len(self.tokens):
                raise ValueError(f"token id out of range: {token!r}")
            text = self.tokens[token]
            if skip_special and self._is_special(token, text):
                continue
            fallback = _BYTE_FALLBACK.match(text)
            if fallback:
                data.append(int(fallback.group(1), 16))
                continue
            for char in text:
                byte = _BYTE_DECODER.get(char)
                if byte is None:
                    data.extend(char.encode("utf-8"))
                else:
                    data.append(byte)
        return data.decode("utf-8", errors="replace")


def load_gguf_tokenizer(reader) -> GGUFBPETokenizer | None:
    """Build a :class:`GGUFBPETokenizer` from a parsed :class:`GGUFReader`.

    Returns ``None`` when the checkpoint does not carry a byte-level BPE
    tokenizer (for example SentencePiece vocabularies) instead of guessing.
    """
    metadata: Mapping[str, object] = reader.metadata()
    model = str(metadata.get("tokenizer.ggml.model", ""))
    tokens = metadata.get("tokenizer.ggml.tokens")
    if model != "gpt2" or not isinstance(tokens, (list, tuple)) or not tokens:
        return None
    merges = metadata.get("tokenizer.ggml.merges", ())
    if not isinstance(merges, (list, tuple)):
        merges = ()
    token_types = metadata.get("tokenizer.ggml.token_type")
    if not isinstance(token_types, (list, tuple)):
        token_types = None

    def optional_id(key: str) -> int | None:
        value = metadata.get(key)
        return int(value) if isinstance(value, int) else None

    add_bos = metadata.get("tokenizer.ggml.add_bos_token")
    return GGUFBPETokenizer(
        [str(token) for token in tokens],
        [str(merge) for merge in merges],
        bos_id=optional_id("tokenizer.ggml.bos_token_id"),
        eos_id=optional_id("tokenizer.ggml.eos_token_id"),
        pad_id=optional_id("tokenizer.ggml.padding_token_id"),
        token_types=[int(value) for value in token_types] if token_types else None,
        model=model,
        # llama.cpp reads this key without a fallback and throws
        # "unknown pre-tokenizer type: ''" when a BPE file omits it; Pyrite
        # refuses for the same reason rather than guessing a split rule.
        pre=str(metadata.get("tokenizer.ggml.pre", "")),
        add_bos=bool(add_bos) if isinstance(add_bos, (bool, int)) else False,
    )
