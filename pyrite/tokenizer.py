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

# Qwen2/Qwen3 style pre-tokenization, written with Python's ``re`` (which is
# Unicode-aware for ``\\w``/``\\d``) instead of requiring the third-party
# ``regex`` module.  ``\\p{L}`` becomes ``[^\\W\\d_]`` and ``\\p{N}`` becomes ``\\d``.
PRE_TOKEN_PATTERN = re.compile(
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)"
    r"|(?:[^\w\r\n]|_)?[^\W\d_]+"
    r"|\d"
    r"| ?[^\s\w]+[\r\n]*"
    r"|\s*[\r\n]+"
    r"|\s+(?!\S)"
    r"|\s+"
)

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
    ):
        if not tokens:
            raise ValueError("tokenizer vocabulary is empty")
        self.model = model
        self.tokens = list(tokens)
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.pad_id = pad_id
        self.token_types = list(token_types) if token_types else None
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
        # GGUF token types: 1 normal, 2 unknown, 3 control/special, 4 unused.
        if (
            self.token_types is not None
            and index < len(self.token_types)
            and int(self.token_types[index]) == 3
        ):
            return True
        return token.startswith("<|") and token.endswith("|>")

    @property
    def vocab_size(self) -> int:
        return len(self.tokens)

    @property
    def spec(self) -> TokenizerSpec:
        return TokenizerSpec(len(self.tokens), self.bos_id, self.eos_id, self.pad_id)

    # ------------------------------------------------------------------ encode
    def encode(self, text: str, *, add_bos: bool = False, allow_special: bool = True) -> list[int]:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        ids: list[int] = []
        if add_bos and self.bos_id is not None:
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
        result: list[int] = []
        for piece in PRE_TOKEN_PATTERN.findall(text):
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

    return GGUFBPETokenizer(
        [str(token) for token in tokens],
        [str(merge) for merge in merges],
        bos_id=optional_id("tokenizer.ggml.bos_token_id"),
        eos_id=optional_id("tokenizer.ggml.eos_token_id"),
        pad_id=optional_id("tokenizer.ggml.padding_token_id"),
        token_types=[int(value) for value in token_types] if token_types else None,
        model=model,
    )
