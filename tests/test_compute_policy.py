from pyrite.compute_policy import AdaptiveLayerPolicy, TokenDifficulty


def test_adaptive_layer_policy_prefers_more_layers_for_hard_tokens():
    policy = AdaptiveLayerPolicy(min_layers=2)
    easy = policy.plan(10, TokenDifficulty(0.05, 0.9, 0.1))
    hard = policy.plan(10, TokenDifficulty(0.9, 0.0, 0.9))
    assert len(hard.layers) >= len(easy.layers)
    assert len(easy.layers) >= 2
