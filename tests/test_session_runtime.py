"""Session budgets and the checkpoint-backed generation runtime."""
from __future__ import annotations

from pathlib import Path

import pytest

from pyrite.backends.reference import ReferenceBackend
from pyrite.config import RuntimeConfig
from pyrite.engine import PyriteRuntime
from pyrite.runtime import GenerationPolicy, LocalGenerationRuntime
from pyrite.session import LocalSession
from pyrite.tokenizer import WhitespaceTokenizer

from .tiny_qwen3moe import build_tiny_checkpoint


def test_two_sessions_do_not_share_a_kv_budget():
    runtime = PyriteRuntime(RuntimeConfig(max_kv_tokens=8))
    tokenizer = WhitespaceTokenizer()
    for _ in range(3):
        session = LocalSession(runtime, tokenizer, ReferenceBackend())
        result = session.generate("python api", max_new_tokens=2)
        assert result.tokens_out == 2
    # The runtime's own planning budget is untouched by sessions.
    assert runtime.kv.stats.tokens_seen == 0


def test_session_clamps_or_refuses_new_tokens_at_the_budget():
    runtime = PyriteRuntime(RuntimeConfig(max_kv_tokens=4))
    tokenizer = WhitespaceTokenizer()

    clamped = LocalSession(runtime, tokenizer, ReferenceBackend())
    result = clamped.generate("python api", max_new_tokens=100, stop_on_budget=True)
    assert result.tokens_in == 2 and result.tokens_out == 2

    strict = LocalSession(runtime, tokenizer, ReferenceBackend())
    with pytest.raises(MemoryError):
        strict.generate("python api", max_new_tokens=100, stop_on_budget=False)


def test_session_rejects_an_oversized_prompt():
    runtime = PyriteRuntime(RuntimeConfig(max_kv_tokens=2))
    session = LocalSession(runtime, WhitespaceTokenizer(), ReferenceBackend())
    with pytest.raises(MemoryError):
        session.generate("three words here", max_new_tokens=1)


def test_generation_policy_validation():
    assert GenerationPolicy(max_new_tokens=0).max_new_tokens == 0
    with pytest.raises(ValueError):
        GenerationPolicy(max_new_tokens=-1)
    with pytest.raises(ValueError):
        GenerationPolicy(top_p=0.0)
    with pytest.raises(ValueError):
        GenerationPolicy(temperature=-0.1)


def test_local_generation_runtime_requires_a_backend():
    runtime = LocalGenerationRuntime()
    assert runtime.ready is False
    with pytest.raises(RuntimeError):
        runtime.generate("hello")


def test_local_generation_runtime_rejects_network_mode():
    config = RuntimeConfig(offline=False)
    runtime = LocalGenerationRuntime(PyriteRuntime(config), WhitespaceTokenizer(), ReferenceBackend())
    with pytest.raises(RuntimeError):
        runtime.generate("hello")


def test_checkpoint_runtime_generates_text(tmp_path: Path):
    path, _cfg, _ = build_tiny_checkpoint(tmp_path / "tiny.gguf")
    config = RuntimeConfig(ram_budget_mb=1024, reserve_mb=768)
    with LocalGenerationRuntime.from_checkpoint(path, config=config) as runtime:
        result = runtime.generate(
            "hello world",
            GenerationPolicy(max_new_tokens=3, temperature=0.0, top_p=1.0, seed=5),
        )
        assert result.tokens_in > 0
        assert result.tokens_out == 3
        assert result.text.startswith("hello world")
        assert result.route in {"general", "coding", "math", "reasoning"}

        # A second call must produce the same tokens for the same seed: the KV
        # cache is reset per sequence, so nothing leaks across prompts.
        again = runtime.generate(
            "hello world",
            GenerationPolicy(max_new_tokens=3, temperature=0.0, top_p=1.0, seed=5),
        )
        assert again.text == result.text


def test_checkpoint_runtime_stops_at_the_kv_budget(tmp_path: Path):
    path, _cfg, _ = build_tiny_checkpoint(tmp_path / "tiny.gguf")
    config = RuntimeConfig(ram_budget_mb=1024, reserve_mb=768, max_kv_tokens=4, max_context_tokens=8)
    with LocalGenerationRuntime.from_checkpoint(path, config=config) as runtime:
        result = runtime.generate("hello", GenerationPolicy(max_new_tokens=32, temperature=0.0))
        assert result.tokens_out <= 4
