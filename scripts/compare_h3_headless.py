"""Verify the headless milestone against fresh saved-graph controls and summarize timings."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path

from summarize_fasth3_profile import validate_trace


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def timing_summary(rows):
    times = [row["seconds"] for row in rows]
    return {
        "count": len(times),
        "median_seconds": statistics.median(times),
        "mean_seconds": statistics.mean(times),
        "min_seconds": min(times),
        "max_seconds": max(times),
    }


def compare(root, headless_directories):
    recipe = root / "recipe.json"
    recipe_hash = digest(recipe)
    source = json.loads(recipe.read_text())
    history = json.loads((root / "comfy/history.json").read_text())
    if history["workflow_sha256"] != source["workflow_sha256"]:
        raise ValueError("Control workflow changed")
    traces = [json.loads(p.read_text()) for p in (root / "comfy/raw").glob("*.json")]
    by_prompt = {trace["prompt_id"]: trace for trace in traces}
    groups = {"comfy": [], "headless": []}
    all_traces = []
    for run in history["runs"]:
        trace = by_prompt[run["prompt_id"]]
        all_traces.append(trace)
        media = run["history"]["outputs"]["8"]["gifs"][0]
        video = root / "comfy/output" / media["subfolder"] / media["filename"]
        sidecar = json.loads(video.with_suffix(".json").read_text())
        attention = sidecar["sampling"]["sol_attention"]
        if attention["executed_calls"] != 160 or attention["fallback_calls"] != 0:
            raise ValueError("Control dispatch proof failed")
        groups["comfy"].append(
            {
                "seconds": run["server_seconds"],
                "sampler_seconds": sidecar["sampling"]["total_seconds"],
                "phase_memory": sidecar["phase_memory"],
                "video": str(video),
                "mp4_sha256": digest(video),
            }
        )
    for directory in headless_directories:
        results = json.loads((directory / "results.json").read_text())
        if results["recipe_sha256"] != recipe_hash:
            raise ValueError("Headless recipe changed")
        raw = [json.loads(p.read_text()) for p in (directory / "raw").glob("*.json")]
        if len(raw) != len(results["runs"]):
            raise ValueError("Unmatched headless traces")
        all_traces.extend(raw)
        for run in results["runs"]:
            if (
                run["isolation"]["comfy_modules_loaded"]
                or run["isolation"]["node_catalog_loaded"]
                or any(run["runtime_loaded_after_render"])
            ):
                raise ValueError("Headless isolation or stage release failed")
            groups["headless"].append(
                {
                    "seconds": run["render_seconds"],
                    "sampler_seconds": run["sampler_seconds"],
                    "phase_memory": run["phase_memory"],
                    "video": run["output"],
                    "mp4_sha256": digest(run["output"]),
                }
            )
    for trace in all_traces:
        validate_trace(trace)
        if trace["mode"] != "off" or trace["status"] != "success":
            raise ValueError("Invalid timing trace")
    latent_hashes = {digest(trace["latents"]) for trace in all_traces}
    media_hashes = {run["mp4_sha256"] for rows in groups.values() for run in rows}
    if len(latent_hashes) != 1 or len(media_hashes) != 1:
        raise ValueError("Exact latent or MP4 parity failed")
    summaries = {key: timing_summary(rows) for key, rows in groups.items()}
    saving = summaries["comfy"]["median_seconds"] - summaries["headless"]["median_seconds"]
    for rows in groups.values():
        for row in rows:
            row["outside_phases_seconds"] = row["seconds"] - sum(
                phase["seconds"] for phase in row["phase_memory"]["phases"]
            )
    return {
        "summary": summaries,
        "median_saving_seconds": saving,
        "median_saving_percent": saving / summaries["comfy"]["median_seconds"] * 100,
        "latent_file_sha256": next(iter(latent_hashes)),
        "mp4_sha256": next(iter(media_hashes)),
        "all_exact": True,
        "workflow_sha256": source["workflow_sha256"],
        "recipe_sha256": recipe_hash,
        "runs": groups,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--headless", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new output filename")
    report = compare(args.root, args.headless)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps({k: v for k, v in report.items() if k != "runs"}, indent=2))


if __name__ == "__main__":
    main()
