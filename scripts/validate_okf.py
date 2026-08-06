#!/usr/bin/env python3
"""Validate a Google Open Knowledge Format v0.2 bundle."""

from __future__ import annotations

import argparse
import datetime as dt
import re
import sys
from pathlib import Path

try:
    import yaml
except ImportError as exc:  # pragma: no cover - exercised by clean-install usage
    raise SystemExit("PyYAML is required: install the project with the 'docs' extra") from exc

FRONTMATTER = re.compile(r"\A---\n(.*?)\n---(?:\n|\Z)", re.DOTALL)
LOG_DATE = re.compile(r"^## (\d{4}-\d{2}-\d{2})$")
STATUSES = {"draft", "stable", "deprecated"}


def _mapping(value: object) -> bool:
    return isinstance(value, dict)


def _iso_datetime(value: object) -> bool:
    if isinstance(value, (dt.date, dt.datetime)):
        return True
    if not isinstance(value, str):
        return False
    try:
        dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    return "T" in value


def _iso_date(value: object) -> bool:
    if isinstance(value, dt.datetime):
        return False
    if isinstance(value, dt.date):
        return True
    if not isinstance(value, str):
        return False
    try:
        dt.date.fromisoformat(value)
    except ValueError:
        return False
    return True


def _load_frontmatter(path: Path, text: str, errors: list[str]) -> dict | None:
    match = FRONTMATTER.match(text)
    if match is None:
        errors.append(f"{path}: missing YAML frontmatter")
        return None
    try:
        data = yaml.safe_load(match.group(1))
    except yaml.YAMLError as exc:
        errors.append(f"{path}: invalid YAML frontmatter: {exc}")
        return None
    if not _mapping(data):
        errors.append(f"{path}: frontmatter must be a mapping")
        return None
    return data


def _validate_actor(path: Path, field: str, value: object, errors: list[str]) -> None:
    if not isinstance(value, str) or not value.strip():
        errors.append(f"{path}: {field} must be a non-empty actor string")
    elif "/" not in value and not value.startswith(("human:", "process:")):
        errors.append(f"{path}: {field} does not follow the OKF actor convention")


def _validate_concept(path: Path, data: dict, errors: list[str]) -> None:
    concept_type = data.get("type")
    if not isinstance(concept_type, str) or not concept_type.strip():
        errors.append(f"{path}: frontmatter requires a non-empty type")
    status = data.get("status")
    if status is not None and status not in STATUSES:
        errors.append(f"{path}: status must be one of {sorted(STATUSES)}")
    if "stale_after" in data and not _iso_date(data["stale_after"]):
        errors.append(f"{path}: stale_after must be an ISO 8601 date")

    generated = data.get("generated")
    if generated is not None:
        if not _mapping(generated) or "by" not in generated:
            errors.append(f"{path}: generated must be a mapping with by")
        else:
            _validate_actor(path, "generated.by", generated["by"], errors)
            if "at" in generated and not _iso_datetime(generated["at"]):
                errors.append(f"{path}: generated.at must be an ISO 8601 datetime")

    verified = data.get("verified")
    events = [verified] if _mapping(verified) else verified
    if events is not None:
        if not isinstance(events, list):
            errors.append(f"{path}: verified must be a mapping or list of mappings")
        else:
            for index, event in enumerate(events):
                if not _mapping(event) or "by" not in event or "at" not in event:
                    errors.append(f"{path}: verified[{index}] requires by and at")
                    continue
                _validate_actor(path, f"verified[{index}].by", event["by"], errors)
                if not _iso_datetime(event["at"]):
                    errors.append(f"{path}: verified[{index}].at must be an ISO datetime")

    sources = data.get("sources")
    if sources is not None:
        if not isinstance(sources, list):
            errors.append(f"{path}: sources must be a list")
        else:
            ids: set[str] = set()
            for index, source in enumerate(sources):
                if not _mapping(source) or not source.get("resource"):
                    errors.append(f"{path}: sources[{index}] requires resource")
                source_id = source.get("id") if _mapping(source) else None
                if source_id and source_id in ids:
                    errors.append(f"{path}: duplicate source id {source_id!r}")
                if source_id:
                    ids.add(source_id)

    if concept_type == "Attested Computation" and not data.get("runtime"):
        errors.append(f"{path}: Attested Computation requires runtime")


def _body_after_frontmatter(text: str) -> str:
    match = FRONTMATTER.match(text)
    return text[match.end() :] if match else text


def _validate_index(path: Path, text: str, errors: list[str]) -> None:
    body = _body_after_frontmatter(text)
    lines = [line for line in body.splitlines() if line.strip()]
    if not lines or not lines[0].startswith("# "):
        errors.append(f"{path}: index body must start with a level-one section heading")
    for number, line in enumerate(body.splitlines(), start=1):
        if (
            not line.strip()
            or re.match(r"^#{1,6} ", line)
            or line.startswith(("- [", "* ["))
        ):
            continue
        errors.append(f"{path}:{number}: index entries must be Markdown links in a list")


def _validate_log(path: Path, text: str, errors: list[str]) -> None:
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines or not lines[0].startswith("# "):
        errors.append(f"{path}: log must start with a level-one heading")
    dates = [match.group(1) for line in lines if (match := LOG_DATE.match(line))]
    if not dates:
        errors.append(f"{path}: log requires at least one ISO date heading")
    elif dates != sorted(dates, reverse=True):
        errors.append(f"{path}: log date headings must be newest first")


def validate_bundle(root: Path) -> list[str]:
    errors: list[str] = []
    if not root.is_dir():
        return [f"{root}: bundle directory does not exist"]
    markdown = sorted(root.rglob("*.md"))
    if not markdown:
        return [f"{root}: bundle contains no Markdown files"]
    if root / "index.md" not in markdown:
        errors.append(f"{root}: project profile requires a root index.md")

    for path in markdown:
        text = path.read_text(encoding="utf-8")
        if path.name == "index.md":
            if path == root / "index.md" and text.startswith("---\n"):
                data = _load_frontmatter(path, text, errors)
                if data is not None and str(data.get("okf_version")) != "0.2":
                    errors.append(f"{path}: root index must declare okf_version: '0.2'")
            elif text.startswith("---\n"):
                errors.append(f"{path}: only the bundle-root index may have frontmatter")
            _validate_index(path, text, errors)
            continue
        if path.name == "log.md":
            _validate_log(path, text, errors)
            continue
        data = _load_frontmatter(path, text, errors)
        if data is not None:
            _validate_concept(path, data, errors)
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", nargs="?", default="knowledge", type=Path)
    args = parser.parse_args()
    errors = validate_bundle(args.bundle)
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    print(f"OKF v0.2 validation passed: {args.bundle}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
