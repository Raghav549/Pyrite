from __future__ import annotations

import json
from pathlib import Path

from .spec import ModelManifest, blocks_from_rows


def load_manifest(path: Path) -> ModelManifest:
    data = json.loads(path.read_text(encoding="utf-8"))
    return ModelManifest(
        name=str(data["name"]),
        architecture=str(data["architecture"]),
        parameter_count=int(data["parameter_count"]),
        quantization=str(data["quantization"]),
        blocks=blocks_from_rows(data["blocks"]),
    )


def save_manifest(manifest: ModelManifest, path: Path) -> None:
    data = {
        "name": manifest.name,
        "architecture": manifest.architecture,
        "parameter_count": manifest.parameter_count,
        "quantization": manifest.quantization,
        "blocks": [
            {
                "block_id": block.block_id,
                "kind": block.kind.value,
                "size_bytes": block.size_bytes,
                "index": block.index,
                "dependencies": list(block.dependencies),
            }
            for block in manifest.ordered_blocks()
        ],
    }
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
