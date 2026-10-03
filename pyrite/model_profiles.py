"""Published Qwen3 model profiles, for *optional* expectation checks.

Pyrite never needs these to run: every dimension it uses comes from the GGUF
metadata and is cross-validated against the actual tensor shapes.  llama.cpp
master does the same - it has no hardcoded Qwen3 size table at all (verified:
``grep -rn "n_layer == 28\\|n_layer == 94" src/`` in llama.cpp master returns
nothing).

They exist for one purpose: letting an operator assert "this file should be the
235B-A22B checkpoint" and get a precise diff when it is not.  That is a
*declared expectation*, never a loading requirement.

Values are the published Qwen3 / Qwen3-MoE configurations.  A profile only
lists the fields that identify a size; anything absent is taken from the file.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelProfile:
    """A published checkpoint configuration."""

    name: str
    architecture: str
    num_hidden_layers: int
    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    intermediate_size: int | None = None
    vocab_size: int | None = None
    num_experts: int | None = None
    num_experts_per_tok: int | None = None
    moe_intermediate_size: int | None = None
    num_shared_experts: int | None = None

    def expectations(self) -> dict[str, object]:
        """Every non-``None`` field as a name -> expected value mapping."""
        return {
            key: value
            for key, value in {
                "architecture": self.architecture,
                "num_hidden_layers": self.num_hidden_layers,
                "hidden_size": self.hidden_size,
                "num_attention_heads": self.num_attention_heads,
                "num_key_value_heads": self.num_key_value_heads,
                "head_dim": self.head_dim,
                "intermediate_size": self.intermediate_size,
                "vocab_size": self.vocab_size,
                "num_experts": self.num_experts,
                "num_experts_per_tok": self.num_experts_per_tok,
                "moe_intermediate_size": self.moe_intermediate_size,
                "num_shared_experts": self.num_shared_experts,
            }.items()
            if value is not None
        }


# Dense Qwen3.  head_dim is 128 for every published dense Qwen3, which is *not*
# hidden_size / num_attention_heads for the smaller sizes - the trap that makes
# a silent default wrong.
PROFILES: tuple[ModelProfile, ...] = (
    ModelProfile(
        name="qwen3-0.6b",
        architecture="qwen3",
        num_hidden_layers=28,
        hidden_size=1024,
        num_attention_heads=16,
        num_key_value_heads=8,
        head_dim=128,
        intermediate_size=3072,
        vocab_size=151936,
    ),
    ModelProfile(
        name="qwen3-1.7b",
        architecture="qwen3",
        num_hidden_layers=28,
        hidden_size=2048,
        num_attention_heads=16,
        num_key_value_heads=8,
        head_dim=128,
        intermediate_size=6144,
        vocab_size=151936,
    ),
    ModelProfile(
        name="qwen3-4b",
        architecture="qwen3",
        num_hidden_layers=36,
        hidden_size=2560,
        num_attention_heads=32,
        num_key_value_heads=8,
        head_dim=128,
        intermediate_size=9728,
        vocab_size=151936,
    ),
    ModelProfile(
        name="qwen3-8b",
        architecture="qwen3",
        num_hidden_layers=36,
        hidden_size=4096,
        num_attention_heads=32,
        num_key_value_heads=8,
        head_dim=128,
        intermediate_size=12288,
        vocab_size=151936,
    ),
    ModelProfile(
        name="qwen3-14b",
        architecture="qwen3",
        num_hidden_layers=40,
        hidden_size=5120,
        num_attention_heads=40,
        num_key_value_heads=8,
        head_dim=128,
        intermediate_size=17408,
        vocab_size=151936,
    ),
    ModelProfile(
        name="qwen3-32b",
        architecture="qwen3",
        num_hidden_layers=64,
        hidden_size=5120,
        num_attention_heads=64,
        num_key_value_heads=8,
        head_dim=128,
        intermediate_size=25600,
        vocab_size=151936,
    ),
    ModelProfile(
        name="qwen3-30b-a3b",
        architecture="qwen3moe",
        num_hidden_layers=48,
        hidden_size=2048,
        num_attention_heads=32,
        num_key_value_heads=4,
        head_dim=128,
        intermediate_size=6144,
        vocab_size=151936,
        num_experts=128,
        num_experts_per_tok=8,
        moe_intermediate_size=768,
        num_shared_experts=1,
    ),
    ModelProfile(
        name="qwen3-235b-a22b",
        architecture="qwen3moe",
        num_hidden_layers=94,
        hidden_size=4096,
        num_attention_heads=64,
        num_key_value_heads=4,
        head_dim=128,
        intermediate_size=12288,
        vocab_size=151936,
        num_experts=128,
        num_experts_per_tok=8,
        moe_intermediate_size=1536,
        num_shared_experts=1,
    ),
)

BY_NAME: Mapping[str, ModelProfile] = {profile.name: profile for profile in PROFILES}

PROFILE_NAMES: tuple[str, ...] = tuple(BY_NAME)


def profile_for(name: str) -> ModelProfile:
    """Look up a profile by name with an actionable error."""
    profile = BY_NAME.get(name)
    if profile is None:
        raise ValueError(
            f"unknown model profile {name!r}; known profiles: {', '.join(PROFILE_NAMES)}"
        )
    return profile


def check_profile(config: object, profile: ModelProfile) -> dict[str, object]:
    """Compare a loaded config against a profile.

    Returns ``{"matches": bool, "mismatches": [...]}``.  Fields the config does
    not define are reported as such rather than silently ignored.
    """
    mismatches: list[dict[str, object]] = []
    for field, expected in profile.expectations().items():
        actual = getattr(config, field, None)
        if actual is None:
            mismatches.append({"field": field, "expected": expected, "actual": "not present"})
        elif actual != expected:
            mismatches.append({"field": field, "expected": expected, "actual": actual})
    return {"profile": profile.name, "matches": not mismatches, "mismatches": mismatches}
