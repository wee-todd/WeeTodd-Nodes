#!/usr/bin/env python3
"""Aggregate multiple FastH3 acceptance metric files into one promotion report."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

VARIANTS = (
    "grouped_vsa",
    "indexed_metal",
    "indexed_metal_fused_qkv",
    "indexed_metal_layer40",
    "dense_control",
    "dense_token_pair",
)
COMPARISONS = (
    ("grouped_vsa", "indexed_metal"),
    ("indexed_metal", "indexed_metal_fused_qkv"),
    ("indexed_metal", "indexed_metal_layer40"),
    ("dense_control", "dense_token_pair"),
)


def _scenario(value: str) -> tuple[str, Path]:
    name, separator, raw_path = value.partition("=")
    if not separator or not name or not raw_path:
        raise argparse.ArgumentTypeError("scenario must be NAME=/absolute/metrics.json")
    path = Path(raw_path).expanduser().resolve()
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"scenario metrics do not exist: {path}")
    return name, path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", action="append", type=_scenario, required=True)
    parser.add_argument("--transcripts", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    scenarios = {
        name: json.loads(path.read_text(encoding="utf-8"))
        for name, path in args.scenario
    }
    if len(scenarios) != len(args.scenario):
        raise ValueError("scenario names must be unique")

    aggregates = {}
    for name in VARIANTS:
        evaluations = [
            scenario["timings"][name]["seconds_per_evaluation"]
            for scenario in scenarios.values()
        ]
        peaks = [
            scenario["timings"][name]["sample_peak_memory_bytes"]
            for scenario in scenarios.values()
        ]
        aggregates[name] = {
            "mean_seconds_per_evaluation": statistics.mean(evaluations),
            "minimum_seconds_per_evaluation": min(evaluations),
            "maximum_seconds_per_evaluation": max(evaluations),
            "mean_sample_peak_memory_bytes": statistics.mean(peaks),
        }

    speedups = {}
    for reference, candidate in COMPARISONS:
        values = []
        for scenario_name, scenario in scenarios.items():
            baseline = scenario["timings"][reference]["seconds_per_evaluation"]
            candidate_time = scenario["timings"][candidate]["seconds_per_evaluation"]
            values.append(
                {
                    "scenario": scenario_name,
                    "percent": (baseline - candidate_time) / baseline * 100.0,
                }
            )
        speedups[f"{candidate}__vs__{reference}"] = {
            "per_scenario": values,
            "mean_percent": statistics.mean(item["percent"] for item in values),
        }

    transcript_data = None
    if args.transcripts is not None:
        transcript_path = args.transcripts.expanduser().resolve()
        transcript_data = json.loads(transcript_path.read_text(encoding="utf-8"))

    output = args.output.expanduser().resolve()
    output.write_text(
        json.dumps(
            {
                "scenario_count": len(scenarios),
                "scenarios": scenarios,
                "aggregates": aggregates,
                "speedups": speedups,
                "dialogue_transcripts": transcript_data,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
