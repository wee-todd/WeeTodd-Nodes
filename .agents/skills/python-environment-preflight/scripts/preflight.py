#!/usr/bin/env python3
"""Check a candidate interpreter before Python environment mutation."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path


def _requires_python(pyproject: Path) -> str:
    text = pyproject.read_text(encoding="utf-8")
    match = re.search(r'^requires-python\s*=\s*["\']([^"\']+)["\']', text, re.MULTILINE)
    if not match:
        raise ValueError(f"project.requires-python is missing: {pyproject}")
    return match.group(1).strip()


def _minimum_version(constraint: str) -> tuple[int, ...]:
    match = re.search(r"(?:^|,)\s*>?=\s*(\d+(?:\.\d+)+)", constraint)
    if not match:
        raise ValueError(
            "Preflight supports a requires-python lower bound such as >=3.11; "
            f"got {constraint!r}."
        )
    return tuple(int(part) for part in match.group(1).split("."))


def _candidate(python: str) -> dict[str, object]:
    probe = (
        "import json,platform,sys; "
        "print(json.dumps({'executable':sys.executable,'version':list(sys.version_info[:3]),"
        "'architecture':platform.machine(),'platform':platform.platform(),"
        "'prefix':sys.prefix,'base_prefix':sys.base_prefix}))"
    )
    result = subprocess.run(
        [python, "-c", probe], check=False, capture_output=True, text=True
    )
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or f"Cannot execute candidate Python: {python}")
    candidate = json.loads(result.stdout)
    pip_result = subprocess.run(
        [python, "-m", "pip", "--version"], check=False, capture_output=True, text=True
    )
    candidate["pip"] = pip_result.stdout.strip() if pip_result.returncode == 0 else None
    return candidate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default=".")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--require-architecture")
    args = parser.parse_args()

    project = Path(args.project).resolve()
    constraint = _requires_python(project / "pyproject.toml")
    minimum = _minimum_version(constraint)
    candidate = _candidate(args.python)
    version = tuple(candidate["version"])
    version_compatible = version >= minimum
    architecture_compatible = (
        args.require_architecture is None
        or candidate["architecture"] == args.require_architecture
    )
    compatible = version_compatible and architecture_compatible
    report = {
        "project": str(project),
        "requires_python": constraint,
        "candidate": candidate,
        "compatible": compatible,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if not compatible:
        if not version_compatible:
            print(
                f"Candidate Python {'.'.join(map(str, version))} does not satisfy {constraint}.",
                file=sys.stderr,
            )
        if not architecture_compatible:
            print(
                f"Candidate architecture {candidate['architecture']} does not satisfy "
                f"{args.require_architecture}.",
                file=sys.stderr,
            )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
