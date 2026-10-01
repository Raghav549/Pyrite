"""Memory safety: budgets hold under cache, expert, KV and prefetch pressure.

Every test here runs against a real checkpoint file on disk and asserts the
resident-memory promise: bounded caches, real eviction, backpressure instead
of silent overuse, and identical results after eviction storms.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from pyrite.config import RuntimeConfig
from pyrite.engine import PyriteRuntime
from pyrite.executor import Qwen3MoEExecutor

from .tiny_qwen3moe import TinyConfig, build_tiny_checkpoint


def _runtime(**overrides) -> PyriteRuntime:
    config = RuntimeConfig(ram_budget_mb=1024, reserve_mb=768, **overrides)
    return PyriteRuntime(config)


def _k_config() -> TinyConfig:
    return TinyConfig(
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


def test_peak_residency_holds_under_a_tiny_byte_budget(tmp_path: Path):
    path, _, _ = build_tiny_checkpoint(tmp_path / "tiny.gguf", _k_config())
    with Qwen3MoEExecutor(path, runtime=_runtime(), resident_bytes=8192,
                           max_tensor_bytes=1 << 20) as tight:
        tight_output = tight.generate([7, 21], max_new_tokens=2, temperature=0.0)
        assert tight.stats.peak_resident_bytes <= 8192
        assert tight.streamer.stats().evictions > 0
    with Qwen3MoEExecutor(path, runtime=_runtime()) as roomy:
        relaxed = roomy.generate([7, 21], max_new_tokens=2, temperature=0.0)
    # Streaming changes residency, never results.
    assert tight_output == relaxed


def test_only_selected_experts_become_resident(tmp_path: Path):
    """With top-2 of 4 experts, expert I/O stays a fraction of all experts."""
    path, cfg, _ = build_tiny_checkpoint(tmp_path / "tiny.gguf")
    with Qwen3MoEExecutor(path, runtime=_runtime(), sampler_seed=0) as executor:
        executor.generate([5, 9], max_new_tokens=2, temperature=0.0)
        stats = executor.stats
    # 4 tokens x 2 layers x top-2 experts x 3 slices each.
    expected_slices = 4 * cfg.num_hidden_layers * cfg.num_experts_per_tok * 3
    assert stats.expert_slices_streamed == expected_slices
    # Loading every expert of every layer would need twice the slices.
    assert stats.expert_slices_streamed < 4 * cfg.num_hidden_layers * cfg.num_experts * 3


def test_prefetch_storm_cannot_exceed_the_pending_budget(tmp_path: Path):
    path, _, _ = build_tiny_checkpoint(tmp_path / "tiny.gguf")
    with Qwen3MoEExecutor(path, runtime=_runtime(), resident_bytes=4096,
                           max_tensor_bytes=1 << 20, prefetch=True) as executor:
        streamer = executor.streamer
        ids = [block.block_id for block in streamer.blocks]
        for _ in range(5):  # far more prefetch demand than budget
            streamer.prefetch(ids)
        stats = streamer.stats()
        assert stats.resident_bytes <= 4096
        assert streamer.pending_bytes <= 4096
        assert stats.wasted_prefetch_bytes > 0  # refused demand is accounted
        streamer.clear_prefetch()
        assert streamer.pending_bytes == 0


def test_cancelled_prefetch_releases_pending_bytes(tmp_path: Path):
    path, _, _ = build_tiny_checkpoint(tmp_path / "tiny.gguf")
    with Qwen3MoEExecutor(path, runtime=_runtime()) as executor:
        streamer = executor.streamer
        ids = [block.block_id for block in streamer.blocks[:4]]
        streamer.prefetch(ids)
        assert streamer.pending_bytes > 0
        streamer.cancel_prefetch(ids)
        assert streamer.pending_bytes == 0


def test_low_confidence_prefetch_is_counted_as_waste(tmp_path: Path):
    path, _, _ = build_tiny_checkpoint(tmp_path / "tiny.gguf")
    with Qwen3MoEExecutor(path, runtime=_runtime()) as executor:
        streamer = executor.streamer
        target = streamer.blocks[0].block_id
        before = streamer.stats().wasted_prefetch_bytes
        streamer.prefetch([target], confidence={target: 0.1}, min_confidence=0.9)
        after = streamer.stats()
        assert after.wasted_prefetch_bytes == before + streamer.blocks[0].size
        assert target not in streamer.pending


def test_oversized_tensor_is_rejected_not_silently_paged(tmp_path: Path):
    path, _, _ = build_tiny_checkpoint(tmp_path / "tiny.gguf")
    with Qwen3MoEExecutor(path, runtime=_runtime(), max_tensor_bytes=16) as executor, pytest.raises(
        MemoryError
    ):
        executor.logits(3, 0)


def test_kv_pressure_stops_cleanly_at_the_budget(tmp_path: Path):
    path, _, _ = build_tiny_checkpoint(tmp_path / "tiny.gguf")
    runtime = PyriteRuntime(
        RuntimeConfig(ram_budget_mb=1024, reserve_mb=768, max_kv_tokens=3, max_context_tokens=8)
    )
    with Qwen3MoEExecutor(path, runtime=runtime) as executor:
        output = executor.generate([1], max_new_tokens=10, temperature=0.0)
        assert len(output) == 3  # prompt + room for 2, then a clean stop
        assert executor.kv_tokens() == 3
        assert executor.stats.kv_bytes > 0
        with pytest.raises(MemoryError):
            executor.generate([1, 2, 3, 4], max_new_tokens=1, temperature=0.0)


def test_recovery_after_eviction_storm_matches_reference(tmp_path: Path):
    from .tiny_reference import NaiveQwen3MoE

    path, _, _ = build_tiny_checkpoint(tmp_path / "tiny.gguf")
    reference = NaiveQwen3MoE(path)
    prompt = [5, 9, 11]
    with Qwen3MoEExecutor(path, runtime=_runtime(), resident_bytes=2048,
                           max_tensor_bytes=1 << 20) as executor:
        got = executor.generate(prompt, max_new_tokens=4, temperature=0.0)
        assert executor.streamer.stats().evictions > 0
    assert got == reference.greedy_tokens(prompt, 4)


def test_io_and_decode_time_are_measured_not_invented(tmp_path: Path):
    path, _, _ = build_tiny_checkpoint(tmp_path / "tiny.gguf")
    with Qwen3MoEExecutor(path, runtime=_runtime()) as executor:
        executor.generate([5, 9], max_new_tokens=2, temperature=0.0)
        stats = executor.stats
    assert stats.io_seconds >= 0.0
    assert stats.decode_seconds > 0.0  # pure-Python decode always takes time
    assert stats.bytes_loaded > 0
    assert stats.prefetch_hits >= 0
    assert executor.streamer.stats().io_seconds == stats.io_seconds
