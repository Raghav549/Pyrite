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
        "qwen3moe.rope.freq_base": 5_000_000.0,
        "tokenizer.ggml.vocab_size": 151936,
    })
    assert cfg.hidden_size == 4096
    assert cfg.num_hidden_layers == 94
    assert cfg.num_experts == 128
    assert cfg.num_experts_per_tok == 8
