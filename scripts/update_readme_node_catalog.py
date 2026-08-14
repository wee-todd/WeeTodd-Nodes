#!/usr/bin/env python3
"""Generate or verify the registered-node catalog in README.md."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from wee_todd_nodes.nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS  # noqa: E402

START = "<!-- BEGIN GENERATED NODE CATALOG -->"
END = "<!-- END GENERATED NODE CATALOG -->"

CATEGORY_NAMES = {
    "WeeTodd/H3": "H3 — Core and convenience",
    "WeeTodd/H3/loaders": "H3 — Loaders",
    "WeeTodd/H3/conditioning": "H3 — Conditioning",
    "WeeTodd/H3/sampling": "H3 — Sampling and acceleration",
    "WeeTodd/H3/continuation": "H3 — Continuation",
    "WeeTodd/H3/decoding": "H3 — Decoding",
    "WeeTodd/H3/output": "H3 — Output",
    "WeeTodd/LTX 2.3": "LTX 2.3 — Core",
    "WeeTodd/LTX 2.3/loaders": "LTX 2.3 — Loaders",
    "WeeTodd/LTX 2.3/upscale": "LTX 2.3 — Upscaling",
    "WeeTodd/LTX 2.5": "LTX 2.5 — Core",
    "WeeTodd/LTX 2.5/loaders": "LTX 2.5 — Loaders",
}

RECOMMENDED = {
    "WeeToddH3ComponentLoader",
    "WeeToddH3Preflight",
    "WeeToddH3GenerationConfig",
    "WeeToddH3TextEncode",
    "WeeToddH3Sample",
    "WeeToddH3ValidatedSamplingPreset",
    "WeeToddH3DirectPublishLatents",
    "WeeToddLTX23Preflight",
}
EXPERIMENTAL = {
    "WeeToddH3PreviewOverride",
    "WeeToddH3QuantizedTransformerLoader",
    "WeeToddH3ChainedTimeline",
    "WeeToddH3TimedKeyframe",
    "WeeToddH3ReferenceVideo",
    "WeeToddH3TimedKeyframeEncode",
    "WeeToddH3ReferenceEncode",
    "WeeToddH3ReferenceStrength",
    "WeeToddH3ContinuationContext",
    "WeeToddH3ChainAppend",
    "WeeToddH3LatentHiresFix",
    "WeeToddH3EasyCache",
    "WeeToddH3TrajectoryForecast",
    "WeeToddH3BlockCache",
    "WeeToddH3HierarchicalBlockCache",
    "WeeToddH3TrimContinuation",
    "WeeToddH3DirectPublishChain",
    "WeeToddLTX23Generate",
    "WeeToddLTX23UpscalerLoader",
    "WeeToddLTX23UpscalePublish",
    "WeeToddLTX25ComponentLoader",
    "WeeToddLTX25GenerationConfig",
    "WeeToddLTX25Preflight",
    "WeeToddLTX25Generate",
    "WeeToddLTX25GenerateChained",
    "WeeToddLTX25VideoUpscale",
}
CONVENIENCE = {
    "WeeToddH3ModelLoader",
    "WeeToddH3Generate",
    "WeeToddH3Unload",
}
FOUNDATION = set()
NOT_READY = set()

NOTE_OVERRIDES = {
    "WeeToddH3Unload": "Release state held by the monolithic H3 runtime.",
    "WeeToddLTX23GenerationConfig": (
        "Configure LTX 2.3 mode, canvas, duration, steps, guidance, and memory policy."
    ),
    "WeeToddLTX23Preflight": (
        "Validate the selected LTX 2.3 bundle and mode-specific components before allocation."
    ),
    "WeeToddLTX23Unload": "Release the process-local LTX 2.3 pipeline.",
    "WeeToddLTX25Preflight": (
        "Validate LTX 2.5 component metadata and architecture requirements before allocation."
    ),
    "WeeToddLTX25Unload": "Release process-local LTX 2.5 state.",
}


def _status(node_id: str) -> str:
    if node_id in RECOMMENDED:
        return "Recommended"
    if node_id in EXPERIMENTAL:
        return "Experimental"
    if node_id in CONVENIENCE:
        return "Legacy/convenience"
    if node_id in FOUNDATION:
        return "Foundation"
    if node_id in NOT_READY:
        return "Not production-ready"
    return "Supported"


def _note(node_id: str, node_class: type) -> str:
    note = NOTE_OVERRIDES.get(node_id) or getattr(node_class, "DESCRIPTION", "")
    if not note:
        note = (node_class.__doc__ or "").strip().splitlines()[0]
    if not note:
        raise ValueError(f"Registered node {node_id!r} requires a catalog note.")
    return " ".join(str(note).split()).replace("|", "\\|")


def render_catalog() -> str:
    rows = []
    for node_id, node_class in NODE_CLASS_MAPPINGS.items():
        category = getattr(node_class, "CATEGORY", "")
        if category not in CATEGORY_NAMES:
            raise ValueError(
                f"Registered node {node_id!r} has an undocumented category {category!r}."
            )
        name = NODE_DISPLAY_NAME_MAPPINGS.get(node_id, node_id).replace("WeeTodd ", "", 1)
        rows.append((name, _note(node_id, node_class), CATEGORY_NAMES[category], _status(node_id)))

    lines = [
        START,
        "| Node | Notes | Category | Status |",
        "| --- | --- | --- | --- |",
    ]
    lines.extend(
        f"| {name} | {note} | {category} | {status} |" for name, note, category, status in rows
    )
    lines.append(END)
    return "\n".join(lines)


def update_readme(readme: Path, *, check: bool) -> bool:
    current = readme.read_text()
    if current.count(START) != 1 or current.count(END) != 1:
        raise ValueError("README must contain exactly one generated node-catalog marker pair.")
    before, remainder = current.split(START, 1)
    _, after = remainder.split(END, 1)
    expected = before + render_catalog() + after
    if expected == current:
        return False
    if check:
        return True
    readme.write_text(expected)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=ROOT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    readme = args.project.resolve() / "README.md"
    changed = update_readme(readme, check=args.check)
    if args.check and changed:
        print("README node catalog is stale. Run scripts/update_readme_node_catalog.py.")
        return 1
    print("README node catalog is current." if not changed else "Updated README node catalog.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
