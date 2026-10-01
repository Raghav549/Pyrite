from pyrite.models.planner import ModelExecutionPlanner
from pyrite.models.spec import BlockKind, ModelBlock, ModelManifest


def test_execution_units_are_bounded():
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
    units = ModelExecutionPlanner(manifest, max_unit_bytes=5).units()
    assert [u.block_ids for u in units] == [("a",), ("b", "c")]
