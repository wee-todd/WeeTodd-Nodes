from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("operator", ["sdpa", "qkv_projection"])
def test_metal_kernel_probe_writes_portable_result(tmp_path: Path, operator: str) -> None:
    output = tmp_path / f"{operator}.json"
    ready = tmp_path / "ready"
    start = tmp_path / "start"
    start.touch()
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/algorithm_search/metal_kernel_probe.py",
            "--operator",
            operator,
            "--rows",
            "16",
            "--heads",
            "2",
            "--head-dim",
            "32",
            "--hidden-size",
            "64",
            "--warmups",
            "1",
            "--repetitions",
            "2",
            "--ready-file",
            str(ready),
            "--start-file",
            str(start),
            "--output",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert ready.exists()
    payload = json.loads(output.read_text())
    assert payload["operator"] == operator
    assert payload["geometry"]["rows"] == 16
    assert payload["measurement"]["median_seconds"] > 0
    assert len(payload["measurement"]["samples_seconds"]) == 2
