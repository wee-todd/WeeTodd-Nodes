#!/usr/bin/env python3
"""Validate that saved benchmark metadata contains real optimized execution."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from wee_todd_mlx.execution_evidence import require_executed


def _field(value: object, path: str) -> object:
    current = value
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            raise KeyError(f"Metadata has no field {path!r}.")
        current = current[part]
    return current


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metadata", type=Path)
    parser.add_argument("--field", default="stage_timings.sol_attention")
    args = parser.parse_args()

    raw = json.loads(args.metadata.read_text(encoding="utf-8"))
    report = _field(raw, args.field)
    if not isinstance(report, dict):
        raise TypeError(f"Metadata field {args.field!r} is not an object.")
    require_executed(report)
    requested = report.get(
        "requested_backend", "sol_attention" if report.get("enabled") else "disabled"
    )
    executed = report.get("executed_calls", report.get("sparse_kernel_calls", 0))
    print(
        json.dumps(
            {
                "metadata": str(args.metadata),
                "field": args.field,
                "requested_backend": requested,
                "resolved_backend": report.get("resolved_backend", requested),
                "executed_calls": executed,
                "fallback_counts": report.get("fallback_counts", {}),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
