"""End-to-end: a real GGUF checkpoint file, read from disk, decoded and executed.

These tests build an actual Qwen3-MoE checkpoint with the published layout, run
the streaming executor on it, and compare the result with an independent naive
implementation of the same architecture.
"""
from __future__ import annotations

from itertools import pairwise
from pathlib import Path

import pytest

from pyrite.config import RuntimeConfig
from pyrite.engine import PyriteRuntime
from pyrite.executor import Qwen3MoEExecutor, UnsupportedTensorType
from pyrite.qwen3_moe import Qwen3MoECheckpoint, Qwen3MoEContractError
from pyrite.tensor_ops import decode_vector

from .gguf_builder import GGUFFileBuilder
from .tiny_qwen3moe import TinyConfig, build_tiny_checkpoint, make_test_tokenizer
from .tiny_reference import NaiveQwen3MoE

CLOSE = 1e-4


def _runtime(**overrides) -> PyriteRuntime:
    config = RuntimeConfig(ram_budget_mb=1024, reserve_mb=768, **overrides)
    return PyriteRuntime(config)


def _close(a: list[float], b: list[float], tolerance: float = CLOSE) -> bool:
    return len(a) == len(b) and all(abs(x - y) <= tolerance for x, y in zip(a, b, strict=True))


# --------------------------------------------------------------------- contract
def test_contract_and_report_on_real_file(tmp_path: Path):
    path, cfg, _ = build_tiny_checkpoint(tmp_path / "tiny.gguf")
    checkpoint = Qwen3MoECheckpoint(path)
    summary = checkpoint.validate_contract()
    assert summary["layers"] == cfg.num_hidden_layers
    assert summary["experts"] == cfg.num_experts
    assert summary["native_generation_ready"] is True
    assert checkpoint.config.vocab_size == cfg.vocab_size
    assert checkpoint.tensor_count == 3 + cfg.num_hidden_layers * 12

    report = checkpoint.validate_for_streaming(4096)
    assert report["can_stream_storage"] is True
    assert report["can_decode_full_model_in_python"] is False
    assert report["largest_expert_slice_bytes"] > 0


def test_contract_rejects_broken_checkpoint(tmp_path: Path):
    path, cfg, tensors = build_tiny_checkpoint(tmp_path / "tiny.gguf")
    builder = GGUFFileBuilder("qwen3moe")
    for key, value in Qwen3MoECheckpoint(path).metadata.items():
        if key not in {"general.architecture", "general.alignment"}:
            builder.add(key, value)
    builder.add_tensor("token_embd.weight", (cfg.hidden_size, cfg.vocab_size), 0, tensors["token_embd.weight"])
    builder.add_tensor("output_norm.weight", (cfg.hidden_size,), 0, tensors["output_norm.weight"])
    broken = builder.write(tmp_path / "broken.gguf")
    with pytest.raises(Qwen3MoEContractError):
        Qwen3MoECheckpoint(broken).validate_contract()


def test_expert_slices_are_contiguous_and_disjoint(tmp_path: Path):
    path, cfg, tensors = build_tiny_checkpoint(tmp_path / "tiny.gguf")
    checkpoint = Qwen3MoECheckpoint(path)
    raw = path.read_bytes()
    seen: list[tuple[int, int]] = []
    for expert in range(cfg.num_experts):
        slice_ = checkpoint.expert_slice(0, expert, "gate")
        payload = raw[slice_.byte_offset: slice_.byte_offset + slice_.byte_length]
        expected_inner = cfg.hidden_size * cfg.moe_intermediate_size * 4
        assert len(payload) == expected_inner
        tensor_payload = tensors["blk.0.ffn_gate_exps.weight"]
        start = expert * expected_inner
        assert payload == tensor_payload[start:start + expected_inner]
        seen.append((slice_.byte_offset, slice_.byte_offset + slice_.byte_length))
    # slices must not overlap each other
    for (_, end_a), (start_b, _) in pairwise(seen):
        assert end_a <= start_b


# -------------------------------------------------------------------- inference
def test_logits_match_independent_reference(tmp_path: Path):
    path, _cfg, _ = build_tiny_checkpoint(tmp_path / "tiny.gguf")
    reference = NaiveQwen3MoE(path)
    tokens = [3, 17, 42, 8]
    with Qwen3MoEExecutor(path, runtime=_runtime()) as executor:
        for position, token in enumerate(tokens):
            incremental = executor.logits(token, position)
            expected = reference.logits(tokens[: position + 1])
            assert _close(incremental, expected, 1e-3), f"mismatch at position {position}"


def test_generation_is_deterministic_and_matches_reference(tmp_path: Path):
    path, _cfg, _ = build_tiny_checkpoint(tmp_path / "tiny.gguf")
    reference = NaiveQwen3MoE(path)
    prompt = [5, 9, 11]

    with Qwen3MoEExecutor(path, runtime=_runtime(), sampler_seed=0) as executor:
        first = executor.generate(prompt, max_new_tokens=4, temperature=0.0)
        second = executor.generate(prompt, max_new_tokens=4, temperature=0.0)

    expected = reference.greedy_tokens(prompt, 4)
    assert first == second
    assert first == expected


def test_generation_survives_a_tiny_resident_budget(tmp_path: Path):
    """Streaming must not change results, only how much stays resident."""
    path, _cfg, _ = build_tiny_checkpoint(tmp_path / "tiny.gguf")
    prompt = [7, 21]

    with Qwen3MoEExecutor(path, runtime=_runtime(), resident_bytes=4096, max_tensor_bytes=1 << 20) as executor:
        streamed = executor.generate(prompt, max_new_tokens=3, temperature=0.0)
        stats = executor.stats
        streamer_stats = executor.streamer.stats()

    with Qwen3MoEExecutor(path, runtime=_runtime()) as roomy:
        unrestricted = roomy.generate(prompt, max_new_tokens=3, temperature=0.0)

    assert streamed == unrestricted
    assert streamer_stats.evictions > 0, "the tiny budget should force evictions"
    assert stats.peak_resident_bytes <= 4096
    assert stats.expert_slices_streamed > 0


def test_expert_prefetch_is_accounted(tmp_path: Path):
    path, _cfg, _ = build_tiny_checkpoint(tmp_path / "tiny.gguf")
    with Qwen3MoEExecutor(path, runtime=_runtime(), prefetch=True, sampler_seed=1) as executor:
        executor.generate([2, 4], max_new_tokens=2, temperature=0.0)
        streamer = executor.streamer.stats()
        stats = executor.stats
    assert streamer.prefetched > 0
    assert stats.cache_hits > 0
    assert stats.wasted_prefetch_bytes >= 0


def test_text_generation_round_trips_through_the_gguf_tokenizer(tmp_path: Path):
    path, cfg, _ = build_tiny_checkpoint(tmp_path / "tiny.gguf")
    tokenizer = make_test_tokenizer()
    with Qwen3MoEExecutor(path, runtime=_runtime(), sampler_seed=3) as executor:
        prompt_ids = executor.tokenizer.encode("hello world")
        assert prompt_ids == tokenizer.encode("hello world")
        for token in prompt_ids:
            assert 0 <= token < cfg.vocab_size
        result = executor.generate_text("hello world", max_new_tokens=3, temperature=0.0)
    assert result["prompt_tokens"] == len(prompt_ids)
    assert result["new_tokens"] == 3
    assert result["text"].startswith("hello world")
    assert isinstance(result["output_ids"], list)


def test_tied_embeddings_fallback(tmp_path: Path):
    path, _cfg, _ = build_tiny_checkpoint(tmp_path / "tied.gguf", write_output_weight=False)
    reference = NaiveQwen3MoE(path)
    with Qwen3MoEExecutor(path, runtime=_runtime()) as executor:
        logits = executor.logits(6, 0)
    assert _close(logits, reference.logits([6]), 1e-3)


def test_temperature_sampling_is_reproducible(tmp_path: Path):
    path, _cfg, _ = build_tiny_checkpoint(tmp_path / "tiny.gguf")
    with Qwen3MoEExecutor(path, runtime=_runtime(), sampler_seed=11) as a:
        first = a.generate([1, 2, 3], max_new_tokens=3, temperature=0.8, top_p=0.9)
    with Qwen3MoEExecutor(path, runtime=_runtime(), sampler_seed=11) as b:
        second = b.generate([1, 2, 3], max_new_tokens=3, temperature=0.8, top_p=0.9)
    assert first == second


def test_k_quant_expert_slicing_is_block_aligned(tmp_path: Path):
    """Q4_K expert tensors (the published Qwen3-MoE format) slice on block edges."""
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
    path, cfg, tensors = build_tiny_checkpoint(tmp_path / "q4k.gguf", config, ggml_type=12)
    checkpoint = Qwen3MoECheckpoint(path)
    assert checkpoint.validate_contract()["native_generation_ready"] is True

    expert_elements = cfg.hidden_size * cfg.moe_intermediate_size
    block_bytes = 144  # Q4_K block, 256 values
    for expert in range(cfg.num_experts):
        slices = [checkpoint.expert_slice(0, expert, name) for name in ("gate", "up", "down")]
        for slice_ in slices:
            assert slice_.byte_length % block_bytes == 0
            assert slice_.byte_length == expert_elements // 256 * block_bytes
            assert slice_.element_offset == expert * expert_elements
            assert slice_.byte_offset == checkpoint.tensor(slice_.tensor_name).offset + slice_.block_offset
        # Decoding a single expert slice must match the full-tensor decode.
        gate = slices[0]
        payload = path.read_bytes()[gate.byte_offset:gate.byte_offset + gate.byte_length]
        values = decode_vector(gate.ggml_type, payload, gate.element_count)
        assert len(values) == expert_elements
        full = tensors["blk.0.ffn_gate_exps.weight"]
        assert payload == full[gate.block_offset:gate.block_offset + gate.byte_length]


def test_forward_pass_on_a_q4_k_checkpoint(tmp_path: Path):
    """Real Q4_K tensors stream and decode through the executor."""
    import math

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
    path, cfg, _ = build_tiny_checkpoint(tmp_path / "q4k.gguf", config, ggml_type=12)
    with Qwen3MoEExecutor(path, runtime=_runtime()) as executor:
        assert executor.unsupported_tensor_types() == ()
        logits = executor.logits(3, 0)
        assert len(logits) == cfg.vocab_size
        assert all(math.isfinite(value) for value in logits)
        assert executor.stats.bytes_loaded > 0
        assert executor.stats.kv_tokens == 1


def test_unsupported_quantization_is_refused(tmp_path: Path):
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
    with Qwen3MoEExecutor(path, runtime=_runtime()) as executor:
        assert executor.unsupported_tensor_types() == ("TQ2_0",)
        with pytest.raises(UnsupportedTensorType):
            executor.ensure_decodable()


def test_generate_resets_kv_between_sequences(tmp_path: Path):
    path, _cfg, _ = build_tiny_checkpoint(tmp_path / "tiny.gguf")
    prompt = [4, 6, 8]
    with Qwen3MoEExecutor(path, runtime=_runtime(), sampler_seed=0) as executor:
        first = executor.generate(prompt, max_new_tokens=3, temperature=0.0)
        assert executor.kv_tokens() == len(first)
        second = executor.generate(prompt, max_new_tokens=3, temperature=0.0)
        assert executor.kv_tokens() == len(second) == len(first)
        assert second == first


def test_logits_refuses_positions_past_the_kv_budget(tmp_path: Path):
    path, _cfg, _ = build_tiny_checkpoint(tmp_path / "tiny.gguf")
    runtime = PyriteRuntime(RuntimeConfig(ram_budget_mb=1024, reserve_mb=768, max_kv_tokens=4, max_context_tokens=8))
    with Qwen3MoEExecutor(path, runtime=runtime) as executor:
        executor.logits(1, 3)  # last legal position
        with pytest.raises(MemoryError):
            executor.logits(1, 4)


def test_kv_trace_larger_than_the_working_set_is_rejected(tmp_path: Path):
    path, _cfg, _ = build_tiny_checkpoint(tmp_path / "tiny.gguf")
    runtime = PyriteRuntime(
        RuntimeConfig(
            ram_budget_mb=1024,
            reserve_mb=768,
            max_context_tokens=2_000_000,
            max_kv_tokens=2_000_000,
        )
    )
    with pytest.raises(ValueError, match="working set"):
        Qwen3MoEExecutor(path, runtime=runtime)


def test_kv_budget_is_enforced(tmp_path: Path):
    path, _cfg, _ = build_tiny_checkpoint(tmp_path / "tiny.gguf")
    runtime = PyriteRuntime(RuntimeConfig(ram_budget_mb=1024, reserve_mb=768, max_kv_tokens=3, max_context_tokens=8))
    with Qwen3MoEExecutor(path, runtime=runtime) as executor:
        # KV capacity counts positions, so a 1-token prompt fits 2 generated tokens.
        assert len(executor.generate([1], max_new_tokens=10, temperature=0.0)) == 3
        with pytest.raises(MemoryError):
            executor.generate([1, 2, 3, 4], max_new_tokens=1, temperature=0.0)
