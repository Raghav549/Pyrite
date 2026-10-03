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


def _metadata_only_copy(tmp_path, vocab_size: int):
    """Rebuild a metadata-only GGUF with a different ``{arch}.vocab_size``."""
    from pyrite.qwen3_moe import Qwen3MoECheckpoint
    from tests.gguf_builder import GGUFFileBuilder
    from tests.tiny_qwen3moe import build_tiny_checkpoint

    path, _, _ = build_tiny_checkpoint(tmp_path / "tiny.gguf")
    reader_meta = Qwen3MoECheckpoint(path).metadata
    builder = GGUFFileBuilder("qwen3moe")
    for key, value in reader_meta.items():
        if key not in {"general.architecture", "general.alignment", "qwen3moe.vocab_size"}:
            builder.add(key, value)
    builder.add("qwen3moe.vocab_size", vocab_size)
    return builder.write(tmp_path / f"vocab{vocab_size}.gguf")


def test_larger_declared_vocab_is_treated_as_padding(tmp_path):
    """``vocab_size`` above the token count means padding, exactly as upstream.

    llama.cpp does ``n_vocab = get_key(VOCAB_SIZE, n_tokens)`` and then pads the
    vocabulary with dummy tokens, so a declared size above the token count is
    legitimate - refusing it would reject real checkpoints.
    """
    from pyrite.qwen3_moe import Qwen3MoECheckpoint
    from tests.tiny_qwen3moe import tokenizer_vocab

    vocab, _, _, _, _ = tokenizer_vocab()
    config = Qwen3MoECheckpoint(_metadata_only_copy(tmp_path, len(vocab) + 100)).config
    assert config.vocab_size == len(vocab) + 100
    assert config.vocab_token_count == len(vocab)
    assert config.vocab_source == "qwen3moe.vocab_size"


def test_vocab_size_must_agree_with_the_tokenizer(tmp_path):
    """A vocab key *below* the token count is refused: ids would run off the end."""
    from pyrite.qwen3_moe import Qwen3MoECheckpoint, Qwen3MoEContractError
    from tests.tiny_qwen3moe import tokenizer_vocab


    vocab, _, _, _, _ = tokenizer_vocab()
    bad = _metadata_only_copy(tmp_path, len(vocab) - 50)
    with pytest.raises(Qwen3MoEContractError, match=r"index past the LM head"):
        Qwen3MoECheckpoint(bad)
