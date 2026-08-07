#!/usr/bin/env python3
"""Report complete macOS ComfyUI process memory, including Metal allocations.

Run against a freshly started ComfyUI process when comparing workflows because the kernel's
physical-footprint peak is process-lifetime monotonic.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess

_PEAK = re.compile(r"phys_footprint_peak:\s*([0-9]+)")
_CURRENT = re.compile(r"phys_footprint:\s*([0-9]+)")


def parse_footprint(text: str) -> dict[str, int]:
    peak = _PEAK.search(text)
    current = _CURRENT.search(text)
    if peak is None:
        raise ValueError("footprint output did not contain phys_footprint_peak")
    result = {"process_peak_bytes": int(peak.group(1))}
    if current is not None:
        result["process_current_bytes"] = int(current.group(1))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("pid", type=int)
    parser.add_argument("--baseline-bytes", type=int, default=0)
    args = parser.parse_args()
    completed = subprocess.run(
        [
            "/usr/bin/footprint",
            "--pid",
            str(args.pid),
            "--sample",
            "0.1",
            "--sample-duration",
            "0.1",
            "--noCategories",
            "--format",
            "bytes",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    report = parse_footprint(completed.stdout + completed.stderr)
    report["baseline_bytes"] = args.baseline_bytes
    report["incremental_peak_bytes"] = max(
        report["process_peak_bytes"] - args.baseline_bytes, 0
    )
    report["scope"] = "complete_comfy_process_lifetime"
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
