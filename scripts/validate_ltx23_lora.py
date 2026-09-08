#!/usr/bin/env python3
"""Real checkpoint smoke: zero-strength parity, nonzero effect, rename invariance.

Uses a tiny synthetic diagnostic LoRA, not a trained style/quality certificate.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev-recipe", type=Path, required=True)
    parser.add_argument("--distilled-recipe", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_directory.resolve()
    output.mkdir(parents=True, exist_ok=False)
    import mlx.core as mx

    mx.random.seed(312)
    adapter = output / "diagnostic.safetensors"
    tensors = {}
    for target, width, height in (
        ("proj_out", 4096, 128),
        ("transformer_blocks.0.attn1.to_q", 4096, 4096),
    ):
        tensors[f"diffusion_model.{target}.lora_down.weight"] = mx.random.normal((2, width)) * 0.01
        tensors[f"diffusion_model.{target}.lora_up.weight"] = mx.random.normal((height, 2)) * 0.01
        tensors[f"diffusion_model.{target}.alpha"] = mx.array(2.0)
    mx.save_safetensors(str(adapter), tensors, metadata={"model_version": "2.3"})
    renamed = output / "renamed-community-download.safetensors"
    renamed.hardlink_to(adapter)
    scripts = Path(__file__).resolve().parent
    records = []
    for mode in ("distilled", "one_stage", "two_stage"):
        recipe = json.loads(
            (args.distilled_recipe if mode == "distilled" else args.dev_recipe).read_text()
        )
        recipe["config"].update(pipeline_mode=mode, stage1_steps=2, stage2_steps=1)
        hashes = {}
        for label, source, strength in (
            ("base", None, 0),
            ("zero", adapter, 0),
            ("active", adapter, 0.75),
            ("renamed", renamed, 0.75),
        ):
            case = copy.deepcopy(recipe)
            if source:
                case["loras"] = {"adapters": [{"path": str(source), "strength": strength}]}
            name = mode + "-" + label
            file = output / (name + ".json")
            file.write_text(json.dumps(case, indent=2) + "\n")
            target = output / name
            for preflight in (True, False):
                command = [
                    sys.executable,
                    str(scripts / "render_headless.py"),
                    "--recipe",
                    str(file),
                    "--output-directory",
                    str(output / (name + "-preflight") if preflight else target),
                ]
                if preflight:
                    command.append("--preflight-only")
                with (output / (name + ("-preflight.log" if preflight else ".log"))).open(
                    "w"
                ) as log:
                    subprocess.run(
                        command, stdout=log, stderr=subprocess.STDOUT, check=True, cwd="/"
                    )
            result = json.loads((target / "result.json").read_text())
            if result["status"] != "success" or any(result["runtime_loaded"]):
                raise ValueError("Render failed or runtime remained resident")
            applied = result["metadata"].get("loras", [])
            if source and (len(applied) != 1 or applied[0]["targets"] != 2):
                raise ValueError("Expected two actually applied adapter targets")
            hashes[label] = hashlib.sha256((target / "render.mp4").read_bytes()).hexdigest()
            record = {"case": name, "mp4_sha256": hashes[label], "loras": applied}
            records.append(record)
            (output / "results.json").write_text(json.dumps(records, indent=2) + "\n")
            print(json.dumps(record), flush=True)
        if not (
            hashes["base"] == hashes["zero"]
            and hashes["active"] == hashes["renamed"]
            and hashes["active"] != hashes["base"]
        ):
            raise ValueError(f"LoRA parity/effect/rename check failed: {mode}")
    print("All three pipeline modes passed zero/effect/rename checks", flush=True)


if __name__ == "__main__":
    main()
