#!/usr/bin/env python3
"""Validate portable H3 workflow paths and optionally resolve them in a live ComfyUI install."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

COMPONENT_NODE = "WeeToddH3ComponentLoader"
CONFIG_NODE = "WeeToddH3GenerationConfig"
PREFLIGHT_NODE = "WeeToddH3Preflight"
PRESET_NODE = "WeeToddH3ValidatedSamplingPreset"
PATH_FIELDS = (
    "checkpoint",
    "transformer",
    "text_encoder",
    "processor",
    "tokenizer",
    "video_vae",
    "audio_vae",
)
MEDIA_INPUT_FIELDS = {
    "LoadImage": "image",
    "LoadVideo": "file",
    "LoadAudio": "audio",
}


def load_api_workflow(path: Path) -> dict[str, dict]:
    """Load one ComfyUI API prompt and reject UI-workflow or malformed JSON shapes."""

    try:
        document = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read workflow JSON {path}: {exc}") from exc
    if not isinstance(document, dict) or not document:
        raise ValueError(f"Workflow must be a non-empty ComfyUI API prompt: {path}")
    for node_id, node in document.items():
        if not isinstance(node, dict) or not isinstance(node.get("class_type"), str):
            raise ValueError(
                f"Workflow node {node_id!r} has no class_type. Select the matching *_api.json file."
            )
        if not isinstance(node.get("inputs"), dict):
            raise ValueError(f"Workflow node {node_id!r} has no input mapping.")
    return document


def unique_node(graph: dict[str, dict], class_type: str, *, required: bool = True):
    matches = [
        (node_id, node)
        for node_id, node in graph.items()
        if node["class_type"] == class_type
    ]
    if not matches and not required:
        return None
    if len(matches) != 1:
        raise ValueError(
            f"Workflow must contain exactly one {class_type} node; found {len(matches)}."
        )
    return matches[0]


def portable_component_paths(graph: dict[str, dict]) -> dict[str, str]:
    """Validate path syntax without claiming that files exist in a ComfyUI installation."""

    _, node = unique_node(graph, COMPONENT_NODE)
    values = {}
    for field in PATH_FIELDS:
        value = node["inputs"].get(field, "")
        if not value:
            continue
        if not isinstance(value, str):
            raise ValueError(f"Component path {field!r} must be text, got {type(value).__name__}.")
        path = Path(value).expanduser()
        if path.is_absolute():
            raise ValueError(
                f"Component path {field!r} must be relative to a ComfyUI model root: {value!r}."
            )
        if ".." in path.parts:
            raise ValueError(f"Component path {field!r} cannot contain '..': {value!r}.")
        values[field] = value
    return values


def portable_media_inputs(graph: dict[str, dict]) -> dict[str, str]:
    """Return literal Comfy input names and reject machine-specific media paths."""

    values = {}
    for node_id, node in graph.items():
        field = MEDIA_INPUT_FIELDS.get(node["class_type"])
        if field is None:
            continue
        value = node["inputs"].get(field)
        if not isinstance(value, str):
            raise ValueError(
                f"Workflow media node {node_id!r} ({node['class_type']}) must use a literal "
                f"{field!r} value."
            )
        if not value:
            continue
        path = Path(value).expanduser()
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(
                f"Workflow media input must be relative to ComfyUI's input directory: {value!r}."
            )
        values[node_id] = value
    return values


def unselected_media_inputs(graph: dict[str, dict]) -> dict[str, str]:
    """Return media nodes that require a user selection before live execution."""

    return {
        node_id: node["class_type"]
        for node_id, node in graph.items()
        if (field := MEDIA_INPUT_FIELDS.get(node["class_type"])) is not None
        and node.get("inputs", {}).get(field) == ""
    }


def missing_media_inputs(graph: dict[str, dict], folder_paths) -> dict[str, str]:
    """Resolve every saved LoadImage/LoadVideo/LoadAudio input in live ComfyUI."""

    missing = {}
    for node_id, value in portable_media_inputs(graph).items():
        try:
            resolved = Path(folder_paths.get_annotated_filepath(value))
        except (AttributeError, KeyError, TypeError, ValueError):
            resolved = Path(folder_paths.get_input_directory()) / value
        if not resolved.is_file():
            missing[node_id] = str(resolved)
    return missing


def missing_component_paths(components) -> dict[str, str]:
    """Return every unresolved component path instead of stopping at the first missing path."""

    candidates = {"checkpoint": Path(components.checkpoint).expanduser()}
    candidates.update(components.resolved_paths())
    return {name: str(path) for name, path in candidates.items() if not path.exists()}


def runtime_preflight(
    *,
    graph: dict[str, dict],
    workflow_path: Path,
    project: Path,
    comfy_root: Path,
) -> dict:
    """Resolve and header-validate one saved API prompt in the selected ComfyUI installation."""

    if not (comfy_root / "folder_paths.py").is_file() or not (comfy_root / "main.py").is_file():
        raise ValueError(f"ComfyUI root is invalid: {comfy_root}")
    sys.path.insert(0, str(project / "src"))
    sys.path.insert(0, str(project))
    sys.path.insert(0, str(comfy_root))
    os.chdir(comfy_root)

    import folder_paths

    from wee_todd_nodes.nodes import (
        WeeToddH3ComponentLoader,
        WeeToddH3GenerationConfig,
        WeeToddH3ValidatedSamplingPreset,
    )
    from wee_todd_nodes.preflight import H3PreflightRequest, preflight_components

    _, component_node = unique_node(graph, COMPONENT_NODE)
    _, config_node = unique_node(graph, CONFIG_NODE)
    preflight_match = unique_node(graph, PREFLIGHT_NODE, required=False)
    preset_match = unique_node(graph, PRESET_NODE, required=False)

    components = WeeToddH3ComponentLoader().specify(**component_node["inputs"])[0]
    config = WeeToddH3GenerationConfig().configure(**config_node["inputs"])[0]
    unselected_media = unselected_media_inputs(graph)
    if unselected_media:
        details = "; ".join(
            f"node {node_id} ({node_type})" for node_id, node_type in unselected_media.items()
        )
        raise ValueError(
            "Select every required workflow image, video, or audio input before runtime "
            f"preflight. Unselected inputs: {details}"
        )
    missing_media = missing_media_inputs(graph, folder_paths)
    if missing_media:
        details = "; ".join(f"node {node_id}={path}" for node_id, path in missing_media.items())
        raise FileNotFoundError(
            "Saved workflow media inputs do not resolve in the selected ComfyUI installation. "
            f"Missing inputs: {details}"
        )
    missing = missing_component_paths(components)
    if missing:
        details = "; ".join(f"{name}={path}" for name, path in missing.items())
        raise FileNotFoundError(
            "Saved workflow paths do not resolve in the selected ComfyUI installation. "
            f"Missing components: {details}"
        )
    prompt_tokens = 512
    available_memory_gb = 0.0
    if preflight_match is not None:
        preflight_inputs = preflight_match[1]["inputs"]
        prompt_tokens = int(preflight_inputs.get("prompt_tokens", prompt_tokens))
        available_memory_gb = float(
            preflight_inputs.get("available_memory_gb", available_memory_gb)
        )

    preset_info = None
    if preset_match is not None:
        preset_name = preset_match[1]["inputs"].get("preset")
        if not isinstance(preset_name, str):
            raise ValueError("Validated Sampling Preset has no literal preset name.")
        config, _, _, preset_info_raw = WeeToddH3ValidatedSamplingPreset().apply(
            config, preset_name
        )
        preset_info = json.loads(preset_info_raw)

    report = preflight_components(
        components,
        H3PreflightRequest(
            duration_seconds=config.duration_seconds,
            steps=config.steps,
            width=config.width,
            height=config.height,
            prompt_tokens=prompt_tokens,
            available_memory_gb=available_memory_gb,
        ),
    )
    resolved = {name: str(path) for name, path in components.resolved_paths().items()}
    return {
        "runtime_ready": True,
        "workflow": str(workflow_path),
        "workflow_sha256": hashlib.sha256(workflow_path.read_bytes()).hexdigest(),
        "comfy_root": str(comfy_root),
        "models_dir": str(folder_paths.models_dir),
        "resolved_paths": resolved,
        "preset": preset_info,
        "task": report.task,
        "frames": report.frames,
        "estimated_staged_peak_bytes": report.staged_peak_bytes,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, default=Path.cwd())
    parser.add_argument("--workflow", type=Path, action="append")
    parser.add_argument("--all-api", action="store_true")
    parser.add_argument("--comfy-root", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project = args.project.resolve()
    paths = [path.resolve() for path in (args.workflow or [])]
    if args.all_api:
        paths.extend(sorted((project / "examples").glob("*_api.json")))
    paths = list(dict.fromkeys(paths))
    if not paths:
        raise SystemExit("Select at least one --workflow or use --all-api.")

    reports = []
    for path in paths:
        graph = load_api_workflow(path)
        if not any(node["class_type"] == COMPONENT_NODE for node in graph.values()):
            continue
        portable = portable_component_paths(graph)
        media = portable_media_inputs(graph)
        report = {
            "workflow": str(path),
            "portable_paths_valid": True,
            "component_paths": portable,
            "media_inputs": media,
            "runtime_ready": None,
        }
        if args.comfy_root is not None:
            try:
                report.update(
                    runtime_preflight(
                        graph=graph,
                        workflow_path=path,
                        project=project,
                        comfy_root=args.comfy_root.resolve(),
                    )
                )
            except (FileNotFoundError, ValueError) as exc:
                raise SystemExit(f"Runtime workflow preflight failed for {path}: {exc}") from exc
        reports.append(report)
    print(json.dumps(reports, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
