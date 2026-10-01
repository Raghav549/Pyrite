"""CLI smoke tests: every documented subcommand, against a real checkpoint."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from pyrite.cli import main

from .tiny_qwen3moe import TinyConfig, build_tiny_checkpoint


def _run(capsys, argv: list[str]) -> tuple[int, object]:
    code = main(argv)
    captured = capsys.readouterr()
    payload = None
    if captured.out.strip():
        payload = json.loads(captured.out)
    return code, payload


def test_version_flag(capsys):
    with pytest.raises(SystemExit) as info:
        main(["--version"])
    assert info.value.code == 0
    assert "pyrite" in capsys.readouterr().out


def test_status_and_route(capsys, tmp_path: Path, monkeypatch):
    monkeypatch.setenv("PYRITE_MODEL_DIR", str(tmp_path / "store"))
    code, payload = _run(capsys, ["status", "--json"])
    assert code == 0
    assert payload["status"]["ram_budget_mb"] == 4096
    assert payload["status"]["offline"] is True

    code, payload = _run(capsys, ["route", "write a Python API"])
    assert code == 0
    assert payload["name"] == "coding"


def test_types_command(capsys, tmp_path: Path, monkeypatch):
    monkeypatch.setenv("PYRITE_MODEL_DIR", str(tmp_path / "store"))
    code, payload = _run(capsys, ["types", "--decodable-only"])
    assert code == 0
    names = {entry["name"] for entry in payload["types"]}
    assert {"F32", "Q4_0", "Q4_K", "Q6_K", "BF16"} <= names
    assert all(entry["decodable"] is True for entry in payload["types"])


def test_qwen3_check_reports_the_contract(capsys, tmp_path: Path, monkeypatch):
    monkeypatch.setenv("PYRITE_MODEL_DIR", str(tmp_path / "store"))
    path, cfg, _ = build_tiny_checkpoint(tmp_path / "tiny.gguf")
    code, payload = _run(capsys, ["qwen3-check", str(path), "--full"])
    assert code == 0
    assert payload["config"]["num_hidden_layers"] == cfg.num_hidden_layers
    assert payload["routing"]["native_generation_ready"] is True
    assert payload["streaming"]["can_stream_storage"] is True


def test_inspect_gguf(capsys, tmp_path: Path, monkeypatch):
    monkeypatch.setenv("PYRITE_MODEL_DIR", str(tmp_path / "store"))
    path, _cfg, _ = build_tiny_checkpoint(tmp_path / "tiny.gguf")
    code, payload = _run(capsys, ["inspect", str(path), "--limit", "3"])
    assert code == 0
    assert payload["format"] == "gguf"
    assert payload["tensor_count"] == 27
    assert len(payload["tensors"]) == 3


def test_tokenize_and_generate(capsys, tmp_path: Path, monkeypatch):
    monkeypatch.setenv("PYRITE_MODEL_DIR", str(tmp_path / "store"))
    path, _cfg, _ = build_tiny_checkpoint(tmp_path / "tiny.gguf")

    code, payload = _run(capsys, ["tokenize", str(path), "hello world"])
    assert code == 0
    assert payload["decoded"] == "hello world"

    code, payload = _run(
        capsys,
        ["generate", str(path), "--prompt", "hello world", "--max-new-tokens", "2", "--temperature", "0"],
    )
    assert code == 0
    assert payload["prompt_tokens"] == 2
    assert payload["new_tokens"] == 2
    assert payload["stats"]["layers_executed"] > 0
    assert "route" in payload


def test_generate_refuses_unsupported_quantization(capsys, tmp_path: Path, monkeypatch):
    monkeypatch.setenv("PYRITE_MODEL_DIR", str(tmp_path / "store"))
    config = TinyConfig(
        hidden_size=256,
        num_hidden_layers=1,
        num_attention_heads=8,
        num_key_value_heads=8,
        head_dim=64,
        moe_intermediate_size=256,
        num_experts=2,
        num_experts_per_tok=2,
        vocab_size=271,
    )
    path, _cfg, _ = build_tiny_checkpoint(tmp_path / "unsupported.gguf", config, ggml_type=35)
    code = main(["generate", str(path), "--prompt", "hello", "--max-new-tokens", "1"])
    captured = capsys.readouterr()
    assert code == 3
    assert "without a reference decoder" in captured.err


def test_qwen3_check_rejects_a_non_checkpoint(capsys, tmp_path: Path, monkeypatch):
    monkeypatch.setenv("PYRITE_MODEL_DIR", str(tmp_path / "store"))
    bogus = tmp_path / "nope.gguf"
    bogus.write_bytes(b"not a gguf file")
    code = main(["qwen3-check", str(bogus)])
    assert code == 1
    assert "qwen3-check" in capsys.readouterr().err


def test_profile_and_ram_override(capsys, tmp_path: Path, monkeypatch):
    monkeypatch.setenv("PYRITE_MODEL_DIR", str(tmp_path / "store"))
    profile = Path(__file__).resolve().parents[1] / "examples" / "four_gb.json"
    code, payload = _run(capsys, ["--profile", str(profile), "status", "--json"])
    assert code == 0
    assert payload["config"]["ram_budget_mb"] == 4096

    code, payload = _run(capsys, ["--ram-mb", "2048", "status", "--json"])
    assert code == 0
    assert payload["config"]["ram_budget_mb"] == 2048


def test_pages_command_ingests_a_checkpoint(capsys, tmp_path: Path, monkeypatch):
    monkeypatch.setenv("PYRITE_MODEL_DIR", str(tmp_path / "store"))
    path, _cfg, _ = build_tiny_checkpoint(tmp_path / "tiny.gguf")
    code, payload = _run(capsys, ["pages", str(path), "--page-bytes", "65536"])
    assert code == 0
    assert payload["page_count"] >= 1
    assert payload["bytes"] == path.stat().st_size


def test_bench_runs_both_modes(capsys, tmp_path: Path, monkeypatch):
    from pyrite import bench

    monkeypatch.setenv("PYRITE_MODEL_DIR", str(tmp_path / "store"))
    assert bench.main([]) == 0
    policy = json.loads(capsys.readouterr().out)
    assert policy["plans"] == 16

    path, _cfg, _ = build_tiny_checkpoint(tmp_path / "tiny.gguf")
    assert bench.main(["--checkpoint", str(path), "--tokens", "2"]) == 0
    checkpoint = json.loads(capsys.readouterr().out)
    assert checkpoint["new_tokens"] == 2
    assert checkpoint["tokens_per_second"] > 0
    assert checkpoint["streamed_bytes"] > 0
    assert checkpoint["stats"]["kv_tokens"] == checkpoint["prompt_tokens"] + 2


def test_bench_subcommand_reports_machine_info(capsys, tmp_path: Path, monkeypatch):
    monkeypatch.setenv("PYRITE_MODEL_DIR", str(tmp_path / "store"))
    code, payload = _run(capsys, ["bench", "--blocks", "2"])
    assert code == 0
    assert payload["benchmark"] == "policy"
    assert payload["machine"]["python"]
    assert payload["machine"]["cpu_count"] >= 1
    assert payload["process_rss_before_mb"] >= 0.0


def test_bench_checkpoint_reports_quantization_mix(capsys, tmp_path: Path, monkeypatch):
    monkeypatch.setenv("PYRITE_MODEL_DIR", str(tmp_path / "store"))
    path, _cfg, _ = build_tiny_checkpoint(tmp_path / "tiny.gguf")
    code, payload = _run(capsys, ["bench", "--checkpoint", str(path), "--tokens", "1"])
    assert code == 0
    assert payload["benchmark"] == "checkpoint-generation"
    assert payload["quantization_mix"] == {"F32": 27}
    assert payload["checkpoint_bytes"] == path.stat().st_size
    assert payload["new_tokens"] == 1
