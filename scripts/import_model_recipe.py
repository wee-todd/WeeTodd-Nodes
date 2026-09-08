#!/usr/bin/env python3
"""Import a headless recipe into a shared local library without copying/downloading weights."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--model-library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output exists; select a new recipe filename")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from render_h3_headless import NoComfyImports, assert_isolated

    sys.meta_path.insert(0, NoComfyImports())
    from wee_todd_mlx.asset_registry import AssetRegistry, import_recipe
    from wee_todd_mlx.headless_preflight import preflight_recipe

    recipe = json.loads(args.recipe.read_text())
    if recipe.get("format") != "weetodd-headless-v2":
        parser.error("Expected a weetodd-headless-v2 recipe")
    preflight = preflight_recipe(recipe)
    registry = AssetRegistry(args.model_library)
    imported = import_recipe(recipe, registry)
    for asset in registry.data["assets"].values():
        if asset["descriptor"]["kind"] == "directory" and not asset["descriptor"]["manifest_only"]:
            if any(
                args.output.resolve().is_relative_to(Path(path).resolve())
                for path in asset["paths"]
            ):
                parser.error("Keep the output recipe outside selected component directories")
    assert_isolated()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Reserve the output exclusively before updating the registry; do not overwrite user files.
    with args.output.open("x") as stream:
        try:
            registry.save()
            json.dump(imported, stream, indent=2)
            stream.write("\n")
        except BaseException:
            args.output.unlink(missing_ok=True)
            raise
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "status": preflight["status"],
                "registered_assets": len(registry.data["assets"]),
                "copied_weight_bytes": 0,
                "downloaded_bytes": 0,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
