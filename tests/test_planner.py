from pyrite.models.planner import ModelExecutionPlanner
from pyrite.models.spec import BlockKind, ModelBlock, ModelManifest


def make_planner():
    manifest = ModelManifest(
        name="test",
        architecture="toy",
        parameter_count=1000,
        quantization="int4",
        blocks=(
            ModelBlock("a", BlockKind.LAYER, 3, 0),
            ModelBlock("b", BlockKind.LAYER, 3, 1),
            ModelBlock("c", BlockKind.LAYER, 2, 2),
        ),
    )
    return ModelExecutionPlanner(manifest, max_unit_bytes=5)


def test_execution_units_are_bounded():
    units = make_planner().units()
    assert [u.block_ids for u in units] == [("a",), ("b", "c")]


def test_plan_feedback_changes_score():
    planner = make_planner()
    units = planner.units()
    planner.observe(units[0], 0.01, 0.01, True)
    planner.observe(units[1], 1.0, 1.0, False)
    assert planner.score(units[0]) > planner.score(units[1])
