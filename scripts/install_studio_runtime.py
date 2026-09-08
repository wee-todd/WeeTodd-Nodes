#!/usr/bin/env python3
"""Create an isolated, versioned native renderer; never replace a user's active environment."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import sys
from pathlib import Path

from studio_bridge import run as run_child


def run(command):
    print(json.dumps({"event": "setup", "step": str(command[1])}), flush=True)
    run_child([str(p) for p in command])


def install(source, destination, uv):
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise ValueError("WeeTodd's native runtime requires Apple Silicon and macOS.")
    if destination.exists():
        raise ValueError("Choose a new runtime folder. Existing runtimes are preserved.")
    preflight = source / ".agents/skills/python-environment-preflight/scripts/preflight.py"
    run(
        [
            sys.executable,
            preflight,
            "--project",
            source,
            "--python",
            sys.executable,
            "--require-architecture",
            "arm64",
        ]
    )
    destination.mkdir(parents=True)
    runtime_source = destination / "Source"
    runtime_source.mkdir()
    for name in ("src", "scripts"):
        shutil.copytree(
            source / name,
            runtime_source / name,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "._*"),
        )
    for name in ("pyproject.toml", "LICENSE", "README.md"):
        shutil.copy2(source / name, runtime_source / name)
    lock = destination / "requirements.lock"
    shutil.copy2(source / "studio/runtime/requirements.lock", lock)
    python = destination / "environment/bin/python"
    run([uv, "venv", "--python", sys.executable, destination / "environment"])
    run(
        [
            sys.executable,
            preflight,
            "--project",
            source,
            "--python",
            python,
            "--require-architecture",
            "arm64",
        ]
    )
    run([uv, "pip", "install", "--python", python, "--require-hashes", "-r", lock])
    run([uv, "pip", "check", "--python", python])
    code = (
        "import sys;sys.path.insert(0,sys.argv[1]);"
        "import mlx.core as mx;"
        "assert mx.metal.is_available(), 'Metal GPU unavailable';"
        "import ltx_core_mlx, ltx_pipelines_mlx, wee_todd_mlx, minimax_h3_mlx, ltx25_mlx;"
        "print('Native MLX runtime verified')"
    )
    run([python, "-c", code, runtime_source / "src"])
    result = {
        "root": str(runtime_source),
        "pythonPath": str(python),
        "format": "weetodd-runtime-v1",
    }
    receipt = destination / "runtime.json"
    temporary = receipt.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    os.replace(temporary, receipt)
    print(json.dumps({"status": "success", "result": result}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--uv", type=Path, required=True)
    args = parser.parse_args()
    install(args.source.resolve(), args.destination.resolve(), args.uv.resolve())


if __name__ == "__main__":
    main()
