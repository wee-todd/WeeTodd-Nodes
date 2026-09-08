"""Join normal-path traces to saved-workflow timings and exact latent/media parity."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path


def validate_trace(trace):
    blocks = trace["blocks"]
    by_evaluation = {}
    for block in blocks:
        by_evaluation.setdefault(block["evaluation"], []).append(block["block"])
    if len(by_evaluation) != 4 or any(len(v) != 40 for v in by_evaluation.values()):
        raise ValueError("Trace did not execute four real 40-layer evaluations")
    expected = [0, 1, 2, 3, 5, 6, 7, 9, *range(18, 50)]
    if any(v != expected for v in by_evaluation.values()):
        raise ValueError("Trace changed the verified layer selection")
    if any(b["attention_backend"] != "metal_indexed" for b in blocks):
        raise ValueError("Trace did not use the indexed Metal VSA path")
    if trace["mode"] != "off":
        roots = [event for event in trace["events"] if event["name"] == "sampler"]
        if len(roots) != 1:
            raise ValueError("Trace needs exactly one sampler root")
        accounted = sum(e["exclusive_seconds"] + e["input_ready_seconds"] for e in trace["events"])
        if abs(accounted - roots[0]["wall_seconds"]) > 1e-5:
            raise ValueError("Profile timing contains unaccounted or double-counted regions")


def summarize(root, reference_video):
    import mlx.core as mx

    reference_hash = hashlib.sha256(reference_video.read_bytes()).hexdigest()
    reference_latents = None
    result = {}
    for mode in ("off", "coarse", "detailed"):
        history = json.loads((root / f"{mode}_history.json").read_text())
        traces = [json.loads(p.read_text()) for p in (root / mode).glob("*.json")]
        by_prompt = {trace["prompt_id"]: trace for trace in traces}
        runs = []
        for entry in history["runs"]:
            trace = by_prompt[entry["prompt_id"]]
            validate_trace(trace)
            media = entry["history"]["outputs"]["8"]["gifs"][0]
            video = root / "output" / media["subfolder"] / media["filename"]
            sidecar = json.loads(video.with_suffix(".json").read_text())
            sampling = sidecar["sampling"]
            attention = sampling["sol_attention"]
            if attention["executed_calls"] != 160 or attention["fallback_calls"] != 0:
                raise ValueError("Saved graph lost its production dispatch proof")
            latents = mx.load(trace["latents"])
            if reference_latents is None:
                reference_latents = latents
            parity = {
                key: bool(mx.array_equal(reference_latents[key], value).item())
                for key, value in latents.items()
            }
            record = {
                "prompt_id": entry["prompt_id"],
                "server_seconds": entry["server_seconds"],
                "sampling_seconds": sampling["total_seconds"],
                "phase_memory": sidecar["phase_memory"],
                "same_latents_as_control": parity,
                "same_mp4_as_verified_baseline": (
                    hashlib.sha256(video.read_bytes()).hexdigest() == reference_hash
                ),
                "video": str(video),
                "trace": trace,
            }
            runs.append(record)
        result[mode] = {
            "workflow_sha256": history["workflow_sha256"],
            "median_server_seconds": statistics.median(r["server_seconds"] for r in runs),
            "median_sampling_seconds": statistics.median(r["sampling_seconds"] for r in runs),
            "runs": runs,
        }
    if len({record["workflow_sha256"] for record in result.values()}) != 1:
        raise ValueError("Compared workflows are not byte-identical")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--reference-video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new output filename")
    result = summarize(args.root, args.reference_video)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    for mode, record in result.items():
        print(
            mode,
            record["median_server_seconds"],
            record["median_sampling_seconds"],
            "exact outputs:",
            all(
                r["same_mp4_as_verified_baseline"] and all(r["same_latents_as_control"].values())
                for r in record["runs"]
            ),
        )


if __name__ == "__main__":
    main()
