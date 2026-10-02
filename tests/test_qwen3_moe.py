import pytest

from pyrite.qwen3_moe import Qwen3MoEConfig


def test_qwen3_moe_config_contract():
    cfg = Qwen3MoEConfig.from_metadata({
        "general.architecture": "qwen3moe",
        "qwen3moe.embedding_length": 4096,
        "qwen3moe.block_count": 94,
        "qwen3moe.attention.head_count": 64,
        "qwen3moe.attention.head_count_kv": 4,
        "qwen3moe.feed_forward_length": 12288,
        "qwen3moe.expert_feed_forward_length": 1536,
        "qwen3moe.expert_count": 128,
        "qwen3moe.expert_used_count": 8,
        "qwen3moe.attention.key_length": 128,
        "qwen3moe.context_length": 262144,
        "qwen3moe.attention.layer_norm_rms_epsilon": 1e-6,
        "qwen3moe.rope.freq_base": 1_000_000.0,
        "tokenizer.ggml.vocab_size": 151936,
    })
    assert cfg.hidden_size == 4096
    assert cfg.num_hidden_layers == 94
    assert cfg.num_experts == 128
    assert cfg.num_experts_per_tok == 8

    # Qwen3's published RoPE default is 1,000,000, not the generic LLaMA 10,000.
    metadata = {
        "general.architecture": "qwen3moe",
        "qwen3moe.embedding_length": 4096,
        "qwen3moe.block_count": 94,
        "qwen3moe.attention.head_count": 64,
        "qwen3moe.attention.head_count_kv": 4,
        "qwen3moe.feed_forward_length": 12288,
        "qwen3moe.expert_feed_forward_length": 1536,
        "qwen3moe.expert_count": 128,
        "qwen3moe.expert_used_count": 8,
        "qwen3moe.attention.key_length": 128,
        "tokenizer.ggml.vocab_size": 151936,
    }
    assert Qwen3MoEConfig.from_metadata(metadata).rope_theta == 1_000_000.0


def test_vocab_size_must_agree_with_the_tokenizer(tmp_path):
    """A checkpoint whose vocab key disagrees with its tokenizer is refused."""
    from pyrite.qwen3_moe import Qwen3MoECheckpoint, Qwen3MoEContractError
    from tests.gguf_builder import GGUFFileBuilder
    from tests.tiny_qwen3moe import build_tiny_checkpoint, tokenizer_vocab

    path, _, _ = build_tiny_checkpoint(tmp_path / "tiny.gguf")
    reader_meta = Qwen3MoECheckpoint(path).metadata
    builder = GGUFFileBuilder("qwen3moe")
    for key, value in reader_meta.items():
        if key not in {"general.architecture", "general.alignment", "qwen3moe.vocab_size"}:
            builder.add(key, value)
    vocab, _, _, _, _ = tokenizer_vocab()
    builder.add("qwen3moe.vocab_size", len(vocab) + 100)
    bad = builder.write(tmp_path / "bad.gguf")
    with pytest.raises(Qwen3MoEContractError, match=r"self-inconsistent checkpoint"):
        Qwen3MoECheckpoint(bad)
