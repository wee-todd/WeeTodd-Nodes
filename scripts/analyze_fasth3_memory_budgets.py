#!/usr/bin/env python3
"""Evaluate FastH3 advisory thresholds across planning memory budgets."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from wee_todd_nodes.nodes import WeeToddH3FastH3ProductionProfile
from wee_todd_nodes.preflight import H3ComponentSetSpec
from wee_todd_nodes.runtime import H3GenerationConfig

PROFILE = "Balanced — compact indexed Metal (recommended)"


def parse_budgets(value: str) -> list[float]:
    budgets = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not budgets or any(budget <= 0.0 for budget in budgets):
        raise argparse.ArgumentTypeError("budgets must be a comma-separated list above zero")
    return budgets


def evaluate(budgets: list[float]) -> dict:
    profile = WeeToddH3FastH3ProductionProfile()
    components = H3ComponentSetSpec(
        checkpoint="/portable-validation/MiniMax-H3/FL2VA",
        task="t2va",
        transformer=(
            "/portable-validation/MiniMax-H3/transformers/weetodd-fasth3-vsa-datafree-q8-paged"
        ),
    )
    source = H3GenerationConfig(duration_seconds=4.0, width=768, height=448)
    rows = []
    for budget_gib in budgets:
        for resolution_preset in profile._RESOLUTION_MATRIX:
            _, configured, _, raw, _ = profile.apply(
                components,
                source,
                PROFILE,
                resolution_preset,
                4096,
                budget_gib,
            )
            info = json.loads(raw)
            measurement = info["measurement"]
            hardware = info["hardware"]
            rows.append(
                {
                    "budget_gib": budget_gib,
                    "resolution": f"{configured.width}×{configured.height}",
                    "measured_peak_gb": round(
                        measurement["complete_peak_memory_bytes"] / 1_000_000_000, 3
                    ),
                    "advisory_minimum_gib": round(hardware["advisory_minimum_gib"], 3),
                    "headroom_status": hardware["headroom_status"],
                    "warning_policy_pass": hardware["headroom_status"] == "comfortable",
                    "validation_scope": hardware["validation_scope"],
                }
            )
    return {
        "status": "simulated_budget_policy_validation",
        "real_lower_memory_render_validated": False,
        "policy": (
            "comfortable when budget >= max(1.35 × measured MLX peak, "
            "measured MLX peak + 4.0 decimal GB)"
        ),
        "budgets_gib": budgets,
        "rows": rows,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budgets", type=parse_budgets, default=parse_budgets("8,16,24,32,48,64"))
    parser.add_argument("--json", type=Path)
    parser.add_argument("--csv", type=Path)
    args = parser.parse_args()

    report = evaluate(args.budgets)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.json is not None:
        args.json.write_text(rendered)
    if args.csv is not None:
        write_csv(args.csv, report["rows"])
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
