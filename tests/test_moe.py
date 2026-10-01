from pyrite.moe import ExpertRouter


def test_top_k_experts():
    out = ExpertRouter(top_k=2).route({"a": 0.1, "b": 0.9, "c": 0.5})
    assert [x.expert_id for x in out] == ["b", "c"]


def test_expert_ids():
    assert ExpertRouter(1).ids({"x": 0.2, "y": 0.8}) == ("y",)
