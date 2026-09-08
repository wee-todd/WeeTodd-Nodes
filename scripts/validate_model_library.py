#!/usr/bin/env python3
"""Check saved candidates through zero-copy import, resolution, and headless preflight."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_directory.resolve()
    output.mkdir(parents=True, exist_ok=False)
    scripts = Path(__file__).resolve().parent
    library = output / "library.json"
    records = []
    for candidate in json.loads(args.matrix.read_text()):
        name = candidate["candidate"]
        if Path(name).name != name or name in {".", ".."}:
            raise ValueError("Candidate name must be a simple directory name")
        imported = output / (name + ".json")
        record = {"candidate": name, "status": "failed"}
        records.append(record)
        try:
            for label, command in (
                (
                    "import",
                    [
                        str(scripts / "import_model_recipe.py"),
                        "--recipe",
                        candidate["recipe"],
                        "--model-library",
                        str(library),
                        "--output",
                        str(imported),
                    ],
                ),
                (
                    "preflight",
                    [
                        str(scripts / "render_headless.py"),
                        "--recipe",
                        str(imported),
                        "--model-library",
                        str(library),
                        "--output-directory",
                        str(output / name),
                        "--preflight-only",
                    ],
                ),
            ):
                run = subprocess.run(
                    [sys.executable, *command],
                    cwd="/",
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                )
                (output / (name + "." + label + ".log")).write_text(run.stdout)
                if run.returncode:
                    raise ValueError(f"{label} failed; see {name}.{label}.log")
            original = json.loads(Path(candidate["recipe"]).read_text())
            resolved = json.loads((output / name / "resolved-recipe.json").read_text())
            if original != resolved:
                raise ValueError("Resolved recipe differs from the tested direct-path recipe")
            result = json.loads((output / name / "result.json").read_text())
            if result["status"] != "preflight_passed":
                raise ValueError("Headless preflight did not pass")
            record.update(
                status="passed",
                resolved_recipe_equal=True,
                assets=len(result["asset_resolution"]["assets"]),
                copied_weight_bytes=0,
                downloaded_bytes=0,
            )
        except Exception as exc:
            record["error"] = str(exc)
        finally:
            (output / "validation.json").write_text(json.dumps(records, indent=2) + "\n")
            print(json.dumps(record), flush=True)
        if record["status"] != "passed":
            raise SystemExit(1)


if __name__ == "__main__":
    main()
