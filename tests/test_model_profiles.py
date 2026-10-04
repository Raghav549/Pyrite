"""The published-profile table must not assert things the reference contradicts."""
from __future__ import annotations

import dataclasses

import pytest

from pyrite.model_profiles import (
    PROFILE_NAMES,
    ModelProfile,
    check_profile,
    profile_for,
)


@dataclasses.dataclass
class FakeConfig:
    num_hidden_layers: int = 94
    hidden_size: int = 4096
    num_attention_heads: int = 64
    num_key_value_heads: int = 4
    head_dim: int = 128
    intermediate_size: int = 12288
    vocab_size: int = 151936
    num_experts: int = 128
    num_experts_per_tok: int = 8
    moe_intermediate_size: int = 1536


def test_profile_names_are_unique_and_lookupable() -> None:
    assert len(set(PROFILE_NAMES)) == len(PROFILE_NAMES)
    for name in PROFILE_NAMES:
        assert profile_for(name).name == name


def test_unknown_profile_lists_the_known_ones() -> None:
    with pytest.raises(ValueError, match="known profiles") as excinfo:
        profile_for("qwen3-9999b")
    assert "qwen3-235b-a22b" in str(excinfo.value)


def test_a_matching_config_produces_no_mismatches() -> None:
    result = check_profile(FakeConfig(), profile_for("qwen3-235b-a22b"))
    assert result["matches"] is True
    assert result["mismatches"] == []


def test_a_mismatch_names_the_field_and_both_values() -> None:
    config = FakeConfig(num_hidden_layers=48)
    result = check_profile(config, profile_for("qwen3-235b-a22b"))
    assert result["matches"] is False
    fields = {item["field"]: item for item in result["mismatches"]}
    assert fields["num_hidden_layers"]["expected"] == 94
    assert fields["num_hidden_layers"]["actual"] == 48


def test_a_missing_attribute_is_reported_not_ignored() -> None:
    result = check_profile(object(), profile_for("qwen3-0.6b"))
    assert result["matches"] is False
    assert all(item["actual"] == "not present" for item in result["mismatches"])


def test_no_profile_asserts_a_shared_expert() -> None:
    """Qwen3-MoE has no shared expert; llama.cpp master loads none.

    ``src/models/qwen3moe.cpp`` loads only ``ffn_gate_exps`` / ``ffn_up_exps`` /
    ``ffn_down_exps`` and never references an ``shexp`` tensor.
    """
    for profile in (profile_for(name) for name in PROFILE_NAMES):
        assert not hasattr(profile, "num_shared_experts")
        assert "num_shared_experts" not in profile.expectations()


def test_no_profile_asserts_architecture_as_a_config_field() -> None:
    """``architecture`` is a GGUF key, not a config attribute, for every arch.

    Asserting it here would report "not present" against real checkpoints.
    """
    for profile in (profile_for(name) for name in PROFILE_NAMES):
        assert "architecture" not in profile.expectations()
        # ...but the profile still records which architecture it belongs to.
        assert isinstance(profile.architecture, str) and profile.architecture


def test_dense_profiles_use_the_published_head_dim() -> None:
    """head_dim is 128 for published dense Qwen3, not hidden_size / n_heads."""
    profile: ModelProfile = profile_for("qwen3-0.6b")
    assert profile.head_dim == 128
    assert profile.hidden_size // profile.num_attention_heads != profile.head_dim


def test_moe_profiles_declare_routing() -> None:
    for name in ("qwen3-30b-a3b", "qwen3-235b-a22b"):
        profile = profile_for(name)
        assert profile.architecture == "qwen3moe"
        assert profile.num_experts == 128
        assert profile.num_experts_per_tok == 8
        assert profile.moe_intermediate_size is not None
