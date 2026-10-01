from __future__ import annotations

from dataclasses import dataclass
import hashlib
import struct


@dataclass(frozen=True)
class CompressedKVObject:
    key: str
    token_count: int
    precision_bits: int
    payload: bytes
    checksum: str

    @property
    def size_bytes(self) -> int:
        return len(self.payload)


class KVObjectCodec:
    """Dependency-free quantizer for reusable KV byte objects."""

    @staticmethod
    def encode(values: list[float], bits: int) -> bytes:
        if bits not in (2, 4, 8, 16):
            raise ValueError("bits must be one of 2, 4, 8, 16")
        if not values:
            return b""
        if bits == 16:
            return b"".join(struct.pack("<e", max(-65504.0, min(65504.0, v))) for v in values)

        levels = (1 << bits) - 1
        scale = max(1e-12, max(abs(v) for v in values))
        out = bytearray()
        for value in values:
            norm = (max(-scale, min(scale, value)) + scale) / (2 * scale)
            code = min(levels, max(0, round(norm * levels)))
            out.append(code)
        return bytes(out)

    @staticmethod
    def decode(payload: bytes, bits: int, scale: float) -> list[float]:
        if bits == 16:
            if len(payload) % 2:
                raise ValueError("invalid fp16 payload")
            return [struct.unpack("<e", payload[i:i + 2])[0] for i in range(0, len(payload), 2)]
        levels = (1 << bits) - 1
        return [((code / levels) * 2.0 - 1.0) * scale for code in payload]


class ReusableKVStore:
    """Checksum-addressed KV objects suitable for local persistence and reuse."""

    def __init__(self):
        self._objects: dict[str, CompressedKVObject] = {}

    @staticmethod
    def make_key(model_fingerprint: str, tokenizer_fingerprint: str, tokens: list[int]) -> str:
        raw = f"{model_fingerprint}|{tokenizer_fingerprint}|{','.join(map(str, tokens))}".encode()
        return hashlib.sha256(raw).hexdigest()

    def put(self, key: str, token_count: int, precision_bits: int, payload: bytes) -> CompressedKVObject:
        checksum = hashlib.sha256(payload).hexdigest()
        obj = CompressedKVObject(key, token_count, precision_bits, payload, checksum)
        self._objects[key] = obj
        return obj

    def get(self, key: str) -> CompressedKVObject | None:
        return self._objects.get(key)

    def verify(self, obj: CompressedKVObject) -> bool:
        return hashlib.sha256(obj.payload).hexdigest() == obj.checksum

    def clear(self) -> None:
        self._objects.clear()
