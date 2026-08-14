"""Best-effort complete-process memory telemetry for macOS ComfyUI runs."""

from __future__ import annotations

import os
import re
import subprocess
from typing import Any

_PEAK = re.compile(r"phys_footprint_peak:\s*([0-9]+)")
_CURRENT = re.compile(r"phys_footprint:\s*([0-9]+)")


def parse_macos_footprint(text: str) -> dict[str, int]:
    """Parse byte-valued physical-footprint fields without assuming line order."""
    peak = _PEAK.search(text)
    if peak is None:
        raise ValueError("footprint output did not contain phys_footprint_peak")
    result = {"complete_process_peak_bytes": int(peak.group(1))}
    current = _CURRENT.search(text)
    if current is not None:
        result["complete_process_current_bytes"] = int(current.group(1))
    return result


def complete_process_memory() -> dict[str, Any]:
    """Return macOS process-footprint telemetry, or an explicit unavailable report.

    Metal allocations are part of the physical footprint but are not represented
    reliably by RSS. Failure is deliberately non-fatal so publication cannot be
    lost merely because the host does not provide Apple's ``footprint`` tool.
    """
    command = "/usr/bin/footprint"
    if not os.path.isfile(command):
        return {
            "complete_process_memory_scope": "unavailable",
            "complete_process_memory_error": "macOS footprint tool is unavailable",
        }
    try:
        completed = subprocess.run(
            [
                command,
                "--pid",
                str(os.getpid()),
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
            timeout=5.0,
        )
        report: dict[str, Any] = parse_macos_footprint(
            completed.stdout + completed.stderr
        )
        report["complete_process_memory_scope"] = "complete_comfy_process_lifetime"
        return report
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        return {
            "complete_process_memory_scope": "unavailable",
            "complete_process_memory_error": str(exc),
        }
