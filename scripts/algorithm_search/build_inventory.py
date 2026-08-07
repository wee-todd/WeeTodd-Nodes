#!/usr/bin/env python3
"""Generate the Phase 1 H3 structural operation inventory."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from minimax_h3_mlx.algorithm_search.inventory import InventoryCase, build_operation_inventory
from minimax_h3_mlx.config import DiTConfig


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("benchmarks/algorithm_search"))
    args = parser.parse_args()
    cases = (
        InventoryCase("smoke_640x384_5s", 640, 384, 5.0),
        InventoryCase("native_1344x768_5s", 1344, 768, 5.0),
    )
    records = build_operation_inventory(cases)
    config = DiTConfig()
    args.output.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "cases": [{**case.__dict__, "geometry": case.geometry(config)} for case in cases],
        "operations": [record.to_dict() for record in records],
    }
    (args.output / "h3_operation_inventory.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )

    lines = [
        "# H3 operation inventory",
        "",
        "Generated from the MLX engine configuration and exact packed-sequence geometry. FLOPs and",
        "temporary bytes are estimates; fused MLX kernels may not materialize conceptual tensors.",
        "",
    ]
    for case in cases:
        geometry = case.geometry(config)
        subset = [record for record in records if record.case == case.name]
        by_type = Counter(record.operation_type for record in subset)
        lines.extend(
            [
                f"## {case.name}",
                "",
                f"- Canvas: {case.width} × {case.height}",
                f"- Frames: {geometry['frames']}",
                f"- Packed rows: {geometry['sequence_rows']:,}",
                f"- Inventory records: {len(subset)}",
                "",
                "| Operation type | Invocations per evaluation | Estimated total FLOPs |",
                "| --- | ---: | ---: |",
            ]
        )
        for operation_type, count in sorted(by_type.items()):
            flops = sum(
                record.approximate_flops or 0
                for record in subset
                if record.operation_type == operation_type
            )
            lines.append(f"| {operation_type} | {count} | {flops:,} |")
        lines.append("")
    (args.output / "h3_operation_inventory.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {len(records)} records to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
