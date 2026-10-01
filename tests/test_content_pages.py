"""Content-addressed weight pages: identity, dedup, corruption, mmap access."""
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from pyrite.storage_pages import ContentAddressedPager, MMapPage


def test_digest_is_stable_and_content_addressed():
    payload = b"weights" * 100
    first = ContentAddressedPager.digest(payload)
    assert first == hashlib.sha256(payload).hexdigest()
    assert first == ContentAddressedPager.digest(bytes(payload))
    assert first != ContentAddressedPager.digest(payload + b"x")


def test_ingest_round_trips_bytes_with_offsets(tmp_path: Path):
    payload = bytes(range(256)) * 4
    source = tmp_path / "weights.bin"
    source.write_bytes(payload)
    pager = ContentAddressedPager(tmp_path / "pages", page_bytes=300)
    pages = pager.ingest(source)
    assert [page.offset for page in pages] == [0, 300, 600, 900]
    assert sum(page.size for page in pages) == len(payload)
    for page in pages:
        assert page.path.read_bytes() == payload[page.offset:page.offset + page.size]
        assert pager.verify(page)


def test_identical_content_shares_one_stored_page(tmp_path: Path):
    source = tmp_path / "weights.bin"
    source.write_bytes(b"q" * 2048)
    pager = ContentAddressedPager(tmp_path / "pages", page_bytes=512)
    pages = pager.ingest(source)
    assert len(pages) == 4
    assert len({page.page_id for page in pages}) == 1
    assert len(list((tmp_path / "pages").glob("*.page"))) == 1


def test_verify_detects_corruption_truncation_and_loss(tmp_path: Path):
    source = tmp_path / "weights.bin"
    source.write_bytes(b"data" * 64)
    pager = ContentAddressedPager(tmp_path / "pages", page_bytes=64)
    page = pager.ingest(source)[0]
    assert pager.verify(page)
    page.path.write_bytes(b"DATA" + page.path.read_bytes()[4:])
    assert not pager.verify(page)
    page.path.write_bytes(b"short")
    assert not pager.verify(page)
    page.path.unlink()
    assert not pager.verify(page)


def test_mmap_view_reconstructs_the_page_bytes(tmp_path: Path):
    payload = bytes((i * 7) % 251 for i in range(500))
    source = tmp_path / "weights.bin"
    source.write_bytes(payload)
    pager = ContentAddressedPager(tmp_path / "pages", page_bytes=200)
    pages = pager.ingest(source)
    with MMapPage(pages[1]) as view:
        assert bytes(view.view()) == payload[200:400]
    # Views stay usable until close, then release cleanly even if exported.
    mapping = MMapPage(pages[0])
    exported = mapping.view()
    assert bytes(exported) == payload[:200]
    mapping.close()


def test_pager_rejects_bad_inputs(tmp_path: Path):
    with pytest.raises(ValueError):
        ContentAddressedPager(tmp_path / "pages", page_bytes=0)
    with pytest.raises(FileNotFoundError):
        ContentAddressedPager(tmp_path / "pages").ingest(tmp_path / "missing.bin")
