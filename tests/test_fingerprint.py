from pathlib import Path

from pyrite.models.fingerprint import ModelFingerprint, fingerprint_file


def test_fingerprint_stable(tmp_path: Path):
    path = tmp_path / "x"
    path.write_bytes(b"abc")
    first = fingerprint_file(path)
    second = fingerprint_file(path)
    assert first == second


def test_namespace_changes_with_model_identity():
    a = ModelFingerprint("a", "t", "x").cache_namespace
    b = ModelFingerprint("b", "t", "x").cache_namespace
    assert a != b
