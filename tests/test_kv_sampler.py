"""KV-cache behaviour and sampler behaviour, with real accounting.

Covers allocation, reuse, growth, truncation, sequence isolation, position
handling, dtype/shape correctness and memory accounting for the KV cache, plus
greedy/temperature/top-p sampling, EOS handling, seeding and token limits.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from pyrite.cache_objects import KVObjectCodec, ReusableKVStore
from pyrite.config import RuntimeConfig
from pyrite.engine import PyriteRuntime
from pyrite.executor import LayerKV, Qwen3MoEExecutor
from pyrite.sampler import Sampler

from .tiny_qwen3moe import build_tiny_checkpoint


def _runtime(**overrides) -> PyriteRuntime:
    config = RuntimeConfig(ram_budget_mb=1024, reserve_mb=768, **overrides)
    return PyriteRuntime(config)


# ------------------------------------------------------------------ LayerKV
def test_layer_kv_tracks_tokens_shapes_and_bytes():
    cache = LayerKV()
    assert cache.length == 0
    assert cache.byte_size == 0
    cache.append([0.1] * 16, [0.2] * 16)
    cache.append([0.3] * 16, [0.4] * 16)
    assert cache.length == 2
    assert cache.byte_size == 2 * 2 * 16 * 4  # tokens x KV x dim x fp32
    cache.truncate(1)
    assert cache.length == 1
    assert cache.byte_size == 2 * 16 * 4
    # Truncating above the length keeps everything.
    cache.truncate(10)
    assert cache.length == 1


def test_executor_kv_grows_per_position_and_matches_stats(tmp_path: Path):
    path, cfg, _ = build_tiny_checkpoint(tmp_path / "tiny.gguf")
    with Qwen3MoEExecutor(path, runtime=_runtime()) as executor:
        assert executor.kv_tokens() == 0
        executor.logits(3, 0)
        executor.logits(17, 1)
        assert executor.kv_tokens() == 2
        expected = cfg.num_hidden_layers * 2 * 2 * cfg.kv_proj_dim * 4
        assert executor.kv_bytes() == expected
        assert executor.stats.kv_tokens == 2
        assert executor.stats.kv_bytes == expected
        assert executor.stats.kv_budget_bytes == executor.kv_budget_bytes > 0


def test_sequences_are_isolated_by_kv_reset(tmp_path: Path):
    path, _, _ = build_tiny_checkpoint(tmp_path / "tiny.gguf")
    with Qwen3MoEExecutor(path, runtime=_runtime(), sampler_seed=0) as executor:
        first = executor.generate([5, 9, 11], max_new_tokens=3, temperature=0.0)
        # Without isolation the second call would attend to stale entries.
        second = executor.generate([5, 9, 11], max_new_tokens=3, temperature=0.0)
        assert first == second
        assert executor.kv_tokens() == len(second)
        executor.reset_kv()
        assert executor.kv_tokens() == 0
        assert executor.kv_bytes() == 0


def test_negative_positions_are_rejected(tmp_path: Path):
    path, _, _ = build_tiny_checkpoint(tmp_path / "tiny.gguf")
    with Qwen3MoEExecutor(path, runtime=_runtime()) as executor, pytest.raises(ValueError):
        executor.logits(3, -1)


# ------------------------------------------------------- compressed KV objects
@pytest.mark.parametrize("bits", [2, 4, 8, 16])
def test_kv_codec_round_trip_accuracy_scales_with_bits(bits: int):
    values = [round(-1.0 + i * 0.25, 6) for i in range(9)]
    payload = KVObjectCodec.encode(values, bits)
    scale = max(abs(v) for v in values)
    restored = KVObjectCodec.decode(payload, bits, scale)
    assert len(restored) == len(values)
    worst = max(abs(a - b) for a, b in zip(restored, values, strict=True))
    # Naive uniform quantization: error shrinks as levels grow.
    bound = {2: 0.6, 4: 0.15, 8: 0.02, 16: 1e-2}[bits]
    assert worst <= bound, f"{bits}-bit worst error {worst}"


def test_kv_codec_rejects_bad_shapes():
    with pytest.raises(ValueError):
        KVObjectCodec.encode([1.0], 3)
    with pytest.raises(ValueError):
        KVObjectCodec.decode(b"\x00", 16, 1.0)
    assert KVObjectCodec.encode([], 8) == b""


def test_reusable_kv_store_keys_corruption_and_clear():
    import dataclasses

    store = ReusableKVStore()
    first = store.make_key("model", "tok", [1, 2, 3])
    assert first == store.make_key("model", "tok", [1, 2, 3])
    assert first != store.make_key("model", "tok", [1, 2, 4])
    obj = store.put(first, 3, 8, b"payload")
    assert store.verify(obj)
    tampered = dataclasses.replace(obj, payload=b"payloaX")
    assert not store.verify(tampered)
    assert store.get("missing") is None
    store.clear()
    assert store.get(first) is None


# ------------------------------------------------------------------- sampler
def test_greedy_picks_the_maximum():
    assert Sampler(seed=0).greedy([0.1, 2.0, 0.2]) == 1
    with pytest.raises(ValueError):
        Sampler().greedy([])


def test_temperature_zero_is_deterministic_greedy():
    logits = [0.5, -1.0, 3.0, 0.0]
    first = Sampler(seed=123).sample(logits, temperature=0.0)
    second = Sampler(seed=999).sample(logits, temperature=0.0)
    assert first == second == 2


def test_temperature_sampling_is_seeded():
    logits = [1.0, 1.0, 1.0, 1.0]
    first = [Sampler(seed=7).sample(logits, temperature=1.0) for _ in range(5)]
    second = [Sampler(seed=7).sample(logits, temperature=1.0) for _ in range(5)]
    assert first == second
    assert all(token in {0, 1, 2, 3} for token in first)


def test_top_p_filters_the_tail():
    # With top_p tiny, only the argmax survives nucleus filtering.
    logits = [5.0, 0.0, 0.0, 0.0]
    for seed in range(5):
        assert Sampler(seed=seed).sample(logits, temperature=1.0, top_p=0.01) == 0


def test_sampler_rejects_bad_arguments():
    sampler = Sampler(seed=0)
    with pytest.raises(ValueError):
        sampler.sample([], temperature=1.0)
    with pytest.raises(ValueError):
        sampler.sample([1.0], temperature=-1.0)
    with pytest.raises(ValueError):
        sampler.sample([1.0], temperature=1.0, top_p=0.0)
    with pytest.raises(ValueError):
        sampler.sample([1.0], temperature=1.0, top_p=1.5)


def test_generate_stops_at_eos_and_honours_max_tokens(tmp_path: Path):
    path, _, _ = build_tiny_checkpoint(tmp_path / "tiny.gguf")
    with Qwen3MoEExecutor(path, runtime=_runtime(), sampler_seed=0) as executor:
        prompt = [5, 9, 11]
        full = executor.generate(prompt, max_new_tokens=4, temperature=0.0)
        assert len(full) == len(prompt) + 4
        # No new tokens requested: the prompt comes back unchanged.
        assert executor.generate(prompt, max_new_tokens=0, temperature=0.0) == prompt
        # Stopping on the known greedy continuation halts after one token.
        stopped = executor.generate(prompt, max_new_tokens=4, temperature=0.0,
                                    stop_ids={full[len(prompt)]})
        assert stopped == full[: len(prompt) + 1]
        with pytest.raises(ValueError):
            executor.generate([], max_new_tokens=1)
        with pytest.raises(ValueError):
            executor.generate(prompt, max_new_tokens=-1)
