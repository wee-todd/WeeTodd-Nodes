"""Run the matrix's saved Comfy controls sequentially on an existing dedicated server."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--comfy-root", type=Path, required=True)
    parser.add_argument("--url", default="http://127.0.0.1:8199")
    parser.add_argument("--only", nargs="*")
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    for row in json.loads(args.matrix.read_text()):
        if args.only and row["candidate"] not in args.only:
            continue
        directory = Path(row["recipe"]).parent
        report = directory / "control.json"
        result = directory / "headless/result.json"
        if report.exists() or not result.exists():
            continue
        if json.loads(result.read_text())["status"] != "success":
            continue
        print(f"Starting control {row['candidate']}", flush=True)
        if row["engine"] == "h3":
            subprocess.run(
                [
                    sys.executable,
                    str(project / "scripts/preflight_h3_workflow.py"),
                    "--project",
                    str(project),
                    "--workflow",
                    row["api"],
                    "--comfy-root",
                    str(args.comfy_root),
                ],
                cwd="/",
                check=True,
            )
        subprocess.run(
            [
                sys.executable,
                str(project / "scripts/benchmark_saved_h3_workflow.py"),
                "--workflow",
                row["api"],
                "--url",
                args.url,
                "--runs",
                "1",
                "--output",
                str(report),
            ],
            cwd="/",
            check=True,
        )


if __name__ == "__main__":
    main()
