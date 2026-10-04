"""Malformed, truncated and hostile GGUF inputs must be rejected, not read.

Each case builds the offending bytes explicitly with the small writer below, so
the test says exactly which field is corrupt rather than relying on offsets into
a file another module produced.  Every assertion is that a :class:`GGUFError`
names the offending field - a corrupt file must never cause an out-of-bounds
read, an unbounded allocation or a misleading error.

``GGUFReader`` is lazy: nothing is parsed until an accessor is called, which is
why :func:`parse` forces one.
"""
from __future__ import annotations

import struct
from collections.abc import Sequence
from pathlib import Path

import pytest

from pyrite.adapters.gguf import GGUFError, GGUFReader, _Reader

# Limits live on the classes that enforce them so the test cannot drift.
MAX_DIMS = GGUFReader.MAX_DIMS
MAX_NAME = GGUFReader.MAX_NAME
MAX_STRING_BYTES = _Reader.MAX_STRING_BYTES
MAX_VERSION = GGUFReader.MAX_VERSION
MIN_VERSION = GGUFReader.MIN_VERSION
DEFAULT_ALIGNMENT = GGUFReader.DEFAULT_ALIGNMENT

TYPE_UINT32, TYPE_STRING = 4, 8


class RawGGUF:
    """Minimal GGUF writer, for building files the real builder would refuse."""

    def __init__(self, architecture: str = "qwen3") -> None:
        self.metadata: list[bytes] = []
        self.tensors: list[bytes] = []
        self.architecture = architecture
        self.add_string("general.architecture", architecture)
        self.add_u32("general.alignment", DEFAULT_ALIGNMENT)

    # -- values ------------------------------------------------------------
    def add_string(self, key: str, value: str) -> None:
        encoded = value.encode()
        self.metadata.append(
            _str(key) + struct.pack("<I", TYPE_STRING) + struct.pack("<Q", len(encoded)) + encoded
        )

    def add_u32(self, key: str, value: int) -> None:
        self.metadata.append(_str(key) + struct.pack("<I", TYPE_UINT32) + struct.pack("<I", value))

    def add_raw(self, key: str, value_type: int, payload: bytes) -> None:
        self.metadata.append(_str(key) + struct.pack("<I", value_type) + payload)

    # -- tensors -----------------------------------------------------------
    def add_tensor(
        self,
        name: str,
        dims: Sequence[int],
        ggml_type: int,
        payload: bytes,
        *,
        name_length: int | None = None,
        offset: int | None = None,
    ) -> None:
        raw_name = name.encode()
        self.tensors.append(
            struct.pack("<Q", len(raw_name) if name_length is None else name_length)
            + raw_name
            + struct.pack("<I", len(dims))
            + b"".join(struct.pack("<Q", dim) for dim in dims)
            + struct.pack("<I", ggml_type)
            + struct.pack("<Q", offset if offset is not None else 0)
        )
        self._payloads = getattr(self, "_payloads", [])
        self._payloads.append(payload)

    def patch_tensor(self, index: int, field: str, value: int) -> None:
        """Rewrite one field of an already-encoded tensor header."""
        raw = bytearray(self.tensors[index])
        name_len = struct.unpack_from("<Q", raw, 0)[0]
        n_dims_at = 8 + name_len
        if field == "name_length":
            struct.pack_into("<Q", raw, 0, value)
        elif field == "n_dims":
            struct.pack_into("<I", raw, n_dims_at, value)
        elif field == "ggml_type":
            struct.pack_into("<I", raw, n_dims_at + 4 + 8 * struct.unpack_from("<I", raw, n_dims_at)[0], value)
        elif field == "offset":
            n_dims = struct.unpack_from("<I", raw, n_dims_at)[0]
            struct.pack_into("<Q", raw, n_dims_at + 4 + 8 * n_dims + 4, value)
        else:  # pragma: no cover - guarded by the tests below
            raise AssertionError(field)
        self.tensors[index] = bytes(raw)

    # -- serialisation -----------------------------------------------------
    def bytes(
        self,
        *,
        magic: bytes = b"GGUF",
        version: int = MAX_VERSION,
        tensor_count: int | None = None,
        metadata_count: int | None = None,
        alignment: int = DEFAULT_ALIGNMENT,
        truncate: int = 0,
    ) -> bytes:
        metadata = b"".join(self.metadata)
        index = b"".join(self.tensors)
        payloads = getattr(self, "_payloads", [])
        header = (
            magic
            + struct.pack("<I", version)
            + struct.pack("<q", len(self.tensors) if tensor_count is None else tensor_count)
            + struct.pack("<q", len(self.metadata) if metadata_count is None else metadata_count)
        )
        prefix = header + metadata + index
        start = (len(prefix) + alignment - 1) // alignment * alignment
        body = b"".join(payloads)
        out = prefix + b"\x00" * (start - len(prefix)) + body
        return out[: len(out) - truncate] if truncate else out


def _str(value: str) -> bytes:
    encoded = value.encode()
    return struct.pack("<Q", len(encoded)) + encoded


def parse(tmp_path: Path, data: bytes, name: str = "candidate.gguf") -> GGUFReader:
    """Write the bytes and force the lazy reader to parse them."""
    path = tmp_path / name
    path.write_bytes(data)
    reader = GGUFReader(path)
    reader.header()
    reader.metadata()
    reader.tensor_index()
    return reader


def good() -> RawGGUF:
    doc = RawGGUF()
    doc.add_u32("qwen3.embedding_length", 8)
    doc.add_tensor("blk.0.attn_norm.weight", (8,), 0, struct.pack("<8f", *range(8)))
    return doc


# --------------------------------------------------------------------------- #
# Header
# --------------------------------------------------------------------------- #
def test_a_valid_file_parses(tmp_path: Path) -> None:
    reader = parse(tmp_path, good().bytes())
    header = reader.header()
    assert header.version == MAX_VERSION
    assert header.tensor_count == 1
    assert header.metadata_count == 3
    assert header.alignment == DEFAULT_ALIGNMENT
    assert header.data_offset % DEFAULT_ALIGNMENT == 0


def test_wrong_magic_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(GGUFError, match="magic"):
        parse(tmp_path, good().bytes(magic=b"GGML"))


def test_empty_file_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(GGUFError):
        parse(tmp_path, b"")


def test_truncated_header_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(GGUFError):
        parse(tmp_path, good().bytes()[:12])


def test_unsupported_versions_are_rejected(tmp_path: Path) -> None:
    assert (MIN_VERSION, MAX_VERSION) == (2, 3)
    for version in (1, MAX_VERSION + 1):
        with pytest.raises(GGUFError, match="version"):
            parse(tmp_path, good().bytes(version=version), f"v{version}.gguf")


def test_version_two_is_accepted(tmp_path: Path) -> None:
    assert parse(tmp_path, good().bytes(version=2)).header().version == 2


# --------------------------------------------------------------------------- #
# Counts
# --------------------------------------------------------------------------- #
def test_negative_tensor_count_is_rejected(tmp_path: Path) -> None:
    """Counts are unsigned in GGUF, so -1 reads as 2**64-1 and trips the limit."""
    with pytest.raises(GGUFError, match="safety limits"):
        parse(tmp_path, good().bytes(tensor_count=-1))


def test_absurd_counts_are_rejected_by_the_safety_limits(tmp_path: Path) -> None:
    with pytest.raises(GGUFError, match="safety limits"):
        parse(tmp_path, good().bytes(tensor_count=GGUFReader.MAX_TENSORS + 1))
    with pytest.raises(GGUFError, match="safety limits"):
        parse(tmp_path, good().bytes(metadata_count=GGUFReader.MAX_METADATA + 1))


def test_truncated_data_section_is_rejected(tmp_path: Path) -> None:
    """The index must not be trusted when the payload behind it is missing."""
    with pytest.raises(GGUFError, match="file ends at"):
        parse(tmp_path, good().bytes(truncate=16), "truncated.gguf")


# --------------------------------------------------------------------------- #
# Strings and names
# --------------------------------------------------------------------------- #
def test_oversized_string_length_is_refused_without_allocating(tmp_path: Path) -> None:
    """A bogus 64-bit string length must not trigger a giant allocation."""
    doc = good()
    doc.add_raw("general.name", TYPE_STRING, struct.pack("<Q", MAX_STRING_BYTES + 1) + b"x")
    with pytest.raises(GGUFError, match="remain in the file"):
        parse(tmp_path, doc.bytes())


def test_string_longer_than_the_file_is_refused(tmp_path: Path) -> None:
    doc = good()
    doc.add_raw("general.name", TYPE_STRING, struct.pack("<Q", 1 << 40))
    with pytest.raises(GGUFError, match="remain in the file"):
        parse(tmp_path, doc.bytes())


def test_oversized_tensor_name_is_rejected(tmp_path: Path) -> None:
    doc = good()
    doc.patch_tensor(0, "name_length", MAX_NAME + 1)
    with pytest.raises(GGUFError, match="name"):
        parse(tmp_path, doc.bytes())


def test_duplicate_metadata_keys_are_rejected(tmp_path: Path) -> None:
    doc = good()
    doc.add_u32("general.alignment", 64)
    with pytest.raises(GGUFError, match="duplicate"):
        parse(tmp_path, doc.bytes())


# --------------------------------------------------------------------------- #
# Tensors
# --------------------------------------------------------------------------- #
def test_too_many_dimensions_is_rejected(tmp_path: Path) -> None:
    doc = good()
    doc.patch_tensor(0, "n_dims", MAX_DIMS + 1)
    with pytest.raises(GGUFError, match="dimension"):
        parse(tmp_path, doc.bytes())


def test_zero_dimensions_is_a_legal_scalar(tmp_path: Path) -> None:
    """GGML allows ``n_dims == 0``; the remaining dims default to 1."""
    doc = RawGGUF()
    doc.add_tensor("output_norm.weight", (), 0, struct.pack("<f", 1.5))
    reader = parse(tmp_path, doc.bytes())
    tensor = reader.tensor_index()[0]
    assert tensor.dims == ()
    assert tensor.element_count == 1
    assert reader.read_tensor(tensor) == struct.pack("<f", 1.5)


def test_unknown_tensor_type_is_rejected(tmp_path: Path) -> None:
    doc = good()
    doc.patch_tensor(0, "ggml_type", 4242)
    with pytest.raises(GGUFError, match="type"):
        parse(tmp_path, doc.bytes())


def test_tensor_offset_beyond_the_data_section_is_rejected(tmp_path: Path) -> None:
    doc = good()
    doc.patch_tensor(0, "offset", 1 << 40)
    with pytest.raises(GGUFError, match="beyond the file"):
        parse(tmp_path, doc.bytes(), "badoff.gguf")


def test_duplicate_tensor_names_are_rejected(tmp_path: Path) -> None:
    payload = struct.pack("<8f", *range(8))
    doc = RawGGUF()
    doc.add_tensor("blk.0.attn_norm.weight", (8,), 0, payload)
    doc.add_tensor("blk.0.attn_norm.weight", (8,), 0, payload)
    with pytest.raises(GGUFError, match="duplicate"):
        parse(tmp_path, doc.bytes())


def test_alignment_metadata_is_honoured(tmp_path: Path) -> None:
    doc = RawGGUF()
    doc.metadata = [entry for entry in doc.metadata]  # keep the default alignment key
    doc.add_tensor("blk.0.attn_norm.weight", (8,), 0, struct.pack("<8f", *range(8)))
    doc.metadata[1] = _str("general.alignment") + struct.pack("<II", TYPE_UINT32, 64)
    reader = parse(tmp_path, doc.bytes(alignment=64))
    header = reader.header()
    assert header.alignment == 64
    assert header.data_offset % 64 == 0
