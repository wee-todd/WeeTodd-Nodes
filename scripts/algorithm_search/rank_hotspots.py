#!/usr/bin/env python3
"""Rank synchronized diagnostic regions from one or more metadata captures."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def classify(name: str) -> str:
    if ".sdpa" in name:
        return "compute-heavy / bandwidth-heavy"
    if any(part in name for part in (".qkv_proj", ".out_proj", ".fc1", ".fc2")):
        return "compute-heavy"
    if "residual" in name or "norm" in name or "rotary" in name or "swiglu" in name:
        return "bandwidth-heavy / allocation-heavy"
    return "unclear/mixed"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("metadata", nargs="+", type=Path)
    parser.add_argument(
        "--output", type=Path, default=Path("benchmarks/algorithm_search/hotspots.md")
    )
    args = parser.parse_args()
    grouped: dict[str, list[float]] = defaultdict(list)
    shapes: dict[str, list[list[int]]] = {}
    for path in args.metadata:
        payload = json.loads(path.read_text())
        for item in payload.get("measurements", []):
            grouped[item["name"]].append(float(item["duration_seconds"]))
            shapes[item["name"]] = item.get("output_shapes", [])
    ranked = sorted(
        ((name, sum(values), len(values)) for name, values in grouped.items()),
        key=lambda item: item[1],
        reverse=True,
    )
    total = sum(duration for _, duration, _ in ranked)
    lines = [
        "# H3 synchronized region hotspots",
        "",
        "These diagnostic timings deliberately materialize each named region. They identify",
        "expensive regions but do not equal an uninstrumented evaluation because synchronization",
        "changes MLX fusion.",
        "",
        "| Rank | Region | Time | Profile share | Calls | Character | Output shapes |",
        "| ---: | --- | ---: | ---: | ---: | --- | --- |",
    ]
    for rank, (name, duration, calls) in enumerate(ranked[:10], 1):
        share = 100.0 * duration / max(total, 1e-12)
        lines.append(
            f"| {rank} | `{name}` | {duration:.6f} s | {share:.2f}% | {calls} | "
            f"{classify(name)} | `{shapes.get(name, [])}` |"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
