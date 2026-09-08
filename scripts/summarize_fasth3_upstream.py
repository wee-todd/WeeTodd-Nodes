"""Summarize exact saved-workflow runs, phase peaks and final MP4 byte parity."""

import argparse
import hashlib
import json
import statistics
from pathlib import Path


def summarize(root):
    result = {}
    reference_hash = None
    for policy in ("off", "compiled_adaln", "transient_q8", "combined"):
        source = json.loads((root / f"{policy}_cached.json").read_text())
        runs = []
        for run in source["runs"]:
            media = run["history"]["outputs"]["8"]["gifs"][0]
            path = root / "output" / media["subfolder"] / media["filename"]
            metadata = json.loads(path.with_suffix(".json").read_text())
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if reference_hash is None:
                reference_hash = digest
            phases = metadata["phase_memory"]["phases"]
            runs.append(
                {
                    "prompt_id": run["prompt_id"],
                    "server_seconds": run["server_seconds"],
                    "sampling_seconds": metadata["sampling"]["total_seconds"],
                    "phase_memory": metadata["phase_memory"],
                    "prompt_seconds": next(
                        p["seconds"] for p in phases if p["phase"] == "text_encoder"
                    ),
                    "video": str(path),
                    "sha256": digest,
                    "same_mp4_as_baseline": digest == reference_hash,
                    "policy": metadata["generation"]["inference_optimization"],
                    "evaluations": metadata["sampling"]["transformer_evaluations"],
                }
            )
        if any(run["policy"] != policy or run["evaluations"] != 4 for run in runs):
            raise ValueError("Benchmark policy/evaluation mismatch")
        result[policy] = {
            "workflow_sha256": source["workflow_sha256"],
            "runs": runs,
            "median_server_seconds": statistics.median(r["server_seconds"] for r in runs),
            "median_sampling_seconds": statistics.median(r["sampling_seconds"] for r in runs),
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new output path")
    report = summarize(args.root)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    for name, record in report.items():
        print(
            name,
            record["median_server_seconds"],
            record["median_sampling_seconds"],
            "same MP4:",
            all(r["same_mp4_as_baseline"] for r in record["runs"]),
        )


if __name__ == "__main__":
    main()
