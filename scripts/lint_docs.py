#!/usr/bin/env python3
"""Check repository Markdown hygiene without rewriting documents."""

from __future__ import annotations

import re
import sys
from pathlib import Path

EXCLUDED_PARTS = {".git", ".venv", ".pytest_cache", ".ruff_cache"}
HEADING = re.compile(r"^(#{1,6}) (.+)$")


def lint(path: Path) -> list[str]:
    errors: list[str] = []
    raw = path.read_bytes()
    if b"\r\n" in raw:
        errors.append(f"{path}: use LF line endings")
    if raw and not raw.endswith(b"\n"):
        errors.append(f"{path}: missing final newline")
    text = raw.decode("utf-8")
    in_fence = False
    previous_level = 0
    for number, line in enumerate(text.splitlines(), start=1):
        if line.rstrip(" \t") != line:
            errors.append(f"{path}:{number}: trailing whitespace")
        if line.startswith("```") or line.startswith("~~~"):
            in_fence = not in_fence
            continue
        if in_fence or line == "---":
            continue
        if line.startswith("#"):
            match = HEADING.match(line)
            if match is None:
                errors.append(f"{path}:{number}: malformed ATX heading")
                continue
            level = len(match.group(1))
            if previous_level and level > previous_level + 1:
                errors.append(
                    f"{path}:{number}: heading level jumps from {previous_level} to {level}"
                )
            previous_level = level
    return errors


def main() -> int:
    errors: list[str] = []
    for path in sorted(Path(".").rglob("*.md")):
        if EXCLUDED_PARTS.intersection(path.parts):
            continue
        errors.extend(lint(path))
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    print("Markdown lint passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
