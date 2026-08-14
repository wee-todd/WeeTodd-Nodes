#!/usr/bin/env python3
"""Audit the public workflow catalog for portability and preview coverage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

PROFILES = {"speed", "balance", "performance"}
TASKS = {"t2v", "i2v", "fflf2va", "ref2va", "continuation", "video-upscale"}
MEDIA_NODE_WIDGET = {"LoadImage": 0, "LoadVideo": 0, "LoadAudio": 0}
MEDIA_NODE_INPUT = {"LoadImage": "image", "LoadVideo": "file", "LoadAudio": "audio"}
H3_SAMPLERS = {"WeeToddH3Sample", "WeeToddH3LatentHiresFix"}
H3_PREVIEW = "WeeToddH3PreviewOverride"


def _load(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{path}: invalid workflow JSON: {exc}") from exc


def _audit_media(path: Path, nodes: list[dict]) -> list[str]:
    errors = []
    for node in nodes:
        index = MEDIA_NODE_WIDGET.get(node.get("type"))
        if index is None:
            continue
        values = node.get("widgets_values", [])
        if len(values) > index and values[index]:
            errors.append(
                f"{path}: {node['type']} node {node.get('id')} must ship with an empty media field"
            )
    return errors


def _audit_preview(path: Path, workflow: dict, nodes: list[dict]) -> list[str]:
    if not any(node.get("type") in H3_SAMPLERS for node in nodes):
        return []
    previews = [node for node in nodes if node.get("type") == H3_PREVIEW]
    if len(previews) != 1:
        return [f"{path}: H3 sampler workflow must contain exactly one {H3_PREVIEW} node"]

    preview_id = previews[0]["id"]
    adjacency: dict[int, set[int]] = {}
    for _, origin_id, _, target_id, _, link_type in workflow.get("links", []):
        if link_type == "WEETODD_H3_COMPONENTS":
            adjacency.setdefault(origin_id, set()).add(target_id)
    reachable = {preview_id}
    pending = [preview_id]
    while pending:
        for target_id in adjacency.get(pending.pop(), set()):
            if target_id not in reachable:
                reachable.add(target_id)
                pending.append(target_id)
    sampler_ids = {node["id"] for node in nodes if node.get("type") in H3_SAMPLERS}
    if not sampler_ids <= reachable:
        return [f"{path}: H3 sampler components must come from the preview override"]
    return []


def audit(project: Path) -> list[str]:
    root = project / "workflows"
    errors: list[str] = []
    for path in sorted(root.rglob("*.json")):
        relative = path.relative_to(root)
        if len(relative.parts) != 3:
            errors.append(f"{path}: expected workflows/<profile>/<task>/<file>.json")
            continue
        profile, task, _ = relative.parts
        if profile not in PROFILES:
            errors.append(f"{path}: unknown workflow profile {profile!r}")
        if task not in TASKS:
            errors.append(f"{path}: unknown workflow task {task!r}")

        workflow = _load(path)
        nodes = workflow.get("nodes")
        if not isinstance(nodes, list):
            errors.append(f"{path}: expected a ComfyUI UI workflow with a nodes list")
            continue
        errors.extend(_audit_media(path, nodes))
        errors.extend(_audit_preview(path, workflow, nodes))
    for path in sorted((project / "examples").glob("*_api.json")):
        prompt = _load(path)
        for node_id, node in prompt.items():
            field = MEDIA_NODE_INPUT.get(node.get("class_type"))
            if field is not None and node.get("inputs", {}).get(field):
                errors.append(
                    f"{path}: {node['class_type']} node {node_id} must ship with an empty "
                    "media field"
                )
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path.cwd())
    args = parser.parse_args()
    errors = audit(args.project.resolve())
    if errors:
        print("\n".join(errors))
        return 1
    print("Public workflow catalog is portable, media-empty, and preview-complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
