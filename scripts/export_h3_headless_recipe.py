#!/usr/bin/env python3
"""One-time Comfy-side export of the exact milestone control; never renders."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

CONTROL_SHA = "67bf49b12f8626113a025b5c2c76c06c9adca1b5da01175618cef7e6f877950b"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workflow", type=Path, required=True)
    parser.add_argument("--comfy-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    workflow = args.workflow.resolve()
    output = args.output.resolve()
    if hashlib.sha256(workflow.read_bytes()).hexdigest() != CONTROL_SHA:
        parser.error("This milestone exporter only accepts the verified off_api.json control.")
    from preflight_h3_workflow import load_api_workflow, runtime_preflight

    graph = load_api_workflow(workflow)
    project = Path(__file__).resolve().parents[1]
    preflight = runtime_preflight(
        graph=graph, workflow_path=workflow, project=project, comfy_root=args.comfy_root.resolve()
    )
    from wee_todd_nodes.conditioning_cache import default_cache_directory
    from wee_todd_nodes.nodes import (
        WeeToddH3ComponentLoader,
        WeeToddH3FastH3ProductionProfile,
        WeeToddH3GenerationConfig,
        WeeToddH3PreviewOverride,
    )

    components = WeeToddH3ComponentLoader().specify(**graph["1"]["inputs"])[0]
    preview = {k: v for k, v in graph["2"]["inputs"].items() if k != "components"}
    components = WeeToddH3PreviewOverride().apply(components, **preview)[0]
    config = WeeToddH3GenerationConfig().configure(**graph["3"]["inputs"])[0]
    selection = {k: v for k, v in graph["4"]["inputs"].items() if k not in {"components", "config"}}
    components, config, attention, info, fastvideo = WeeToddH3FastH3ProductionProfile().apply(
        components, config, **selection
    )
    WeeToddH3FastH3ProductionProfile.validate_sampling_inputs(
        info, components, config, attention, fastvideo
    )
    # Pin the model interpretation metadata without hashing tens of GB of unchanged weights.
    root = components.resolved_paths()["transformer"]
    pins = {
        str(root / name): hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in ("paged_manifest.json", "quant_config.json")
    }
    recipe = {
        "format": "weetodd-h3-headless-recipe-v1",
        "workflow_sha256": CONTROL_SHA,
        "components": asdict(components),
        "config": asdict(config),
        "attention": asdict(attention),
        "fastvideo": asdict(fastvideo),
        "production_profile": json.loads(info),
        "model_metadata_sha256": pins,
        "prompt": graph["6"]["inputs"]["prompt"],
        "cache_directory": str(default_cache_directory()),
        "publication": {
            k: graph["8"]["inputs"][k]
            for k in ("crf", "max_av_drift_seconds", "generation_metadata")
        },
        "preflight": preflight,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(recipe, stream, indent=2)
        stream.write("\n")
    print(output)


if __name__ == "__main__":
    main()
