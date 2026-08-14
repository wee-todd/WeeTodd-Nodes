#!/usr/bin/env python3
"""Build the compact profiled H3 workflow matrix from three maintained templates."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from textwrap import dedent

from add_workflow_notes import NOTE_TITLE, _model_note

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / "workflows"
PROFILE = {
    "speed": {
        "steps": 5,
        "tier": "384 px short edge — fast smoke",
        "short_edge": 384,
        "width": 640,
        "height": 384,
        "memory": "low_memory_bf16",
        "preset": "Turbo — drbaph v4 step-600 — 384p low-memory — 5 points / 4 evaluations",
    },
    "balance": {
        "steps": 7,
        "tier": "512 px short edge — balanced preview",
        "short_edge": 512,
        "width": 896,
        "height": 512,
        "memory": "normal",
        "preset": "Staged Turbo — drbaph v4 step-600 — 2 base + 4 Turbo evaluations",
    },
    "performance": {
        "steps": 20,
        "tier": "512 px short edge — balanced preview",
        "short_edge": 512,
        "width": 896,
        "height": 512,
        "memory": "normal",
        "preset": "Dense baseline — 20 points / 19 evaluations",
    },
}
MEDIA_NODES = {"LoadImage", "LoadVideo", "LoadAudio"}
PROMPTS = {
    "t2v": dedent(
        """\
        integrated_multimodal_description: [Shot 1] Photorealistic single continuous shot in a
        sunlit craft studio. An adult ceramic artist centers a wet clay bowl on a spinning wheel,
        shapes the rim with both hands, pauses to inspect the curve, then looks toward the camera
        with a satisfied restrained smile. The camera makes one slow shoulder-height push-in.
        Preserve natural hand contact, stable tools, consistent daylight, and realistic wet clay
        texture.

        overall_soundscape: Quiet studio room tone, steady wheel motor, wet clay rubbing under the
        artist's fingers, a small tool placed on the wooden table, natural breathing, and clothing
        movement. No dialogue.

        non_diegetic_music: N/A"""
    ),
    "i2v": dedent(
        """\
        <Picture 1> defines the subject, clothing, environment, lighting, and opening composition.

        integrated_multimodal_description: [Shot 1] Begin exactly from <Picture 1>. Preserve the
        subject's identity, proportions, clothing, scene layout, and light direction. The subject
        takes three natural steps toward the camera, stops, turns slightly toward the strongest
        light, and makes one small deliberate hand gesture. The camera performs a restrained
        handheld push-in without a cut. Maintain physical contact, stable anatomy, and continuous
        background geometry.

        overall_soundscape: Preserve the visible environment as one acoustic space. Add
        synchronized footsteps, clothing rustle, natural breathing, and quiet location ambience.
        No dialogue.

        non_diegetic_music: N/A"""
    ),
    "fflf2va": dedent(
        """\
        <Picture 1> defines the exact opening frame. <Picture 2> defines the exact final frame.
        Preserve the same subject identity, clothing, environment, and lighting across both
        pictures.

        integrated_multimodal_description: [Shot 1] Begin exactly from <Picture 1>. The subject
        performs one simple physically continuous action that naturally reaches the pose and
        composition in <Picture 2>. Keep screen direction, camera height, background geometry, and
        light direction stable. Use one continuous restrained camera move with no cut, fade,
        teleport, duplicate subject, or reset. End exactly on <Picture 2>.

        overall_soundscape: Maintain one continuous acoustic space. Synchronize footsteps, contact
        sounds, clothing movement, and breathing with the visible action. No dialogue.

        non_diegetic_music: N/A"""
    ),
    "ref2va": dedent(
        """\
        <Picture 1> defines the first subject's identity, proportions, clothing, and texture.
        <Picture 2> defines the second subject's distinct identity, proportions, clothing, and
        texture. <Video 1> defines the target motion rhythm, camera behavior, and spatial staging.
        Preserve both subjects as separate individuals without merging their faces, bodies,
        colors, or clothing.

        integrated_multimodal_description: [Shot 1] Create a photorealistic single continuous shot
        that follows the motion and camera logic from <Video 1> while replacing its visible
        subjects with the two subjects from <Picture 1> and <Picture 2>. Keep their identities
        readable throughout. Preserve physical contact, stable screen positions, natural motion,
        consistent lighting, and continuous background geometry. Do not duplicate, swap, merge,
        teleport, or reset either subject.

        overall_soundscape: Use the connected reference audio as the timing and sound-performance
        guide. Preserve one continuous acoustic space and keep visible impacts, footsteps,
        breathing, and dialogue synchronized.

        non_diegetic_music: Follow the connected reference audio."""
    ),
}


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _execution_nodes(workflow: dict) -> list[dict]:
    return [node for node in workflow["nodes"] if node.get("type") not in {"MarkdownNote", "Note"}]


def _apply_profile(workflow: dict, profile: str) -> None:
    policy = PROFILE[profile]
    for node in _execution_nodes(workflow):
        if node["type"] == "WeeToddH3GenerationConfig":
            values = node["widgets_values"]
            values[1] = policy["steps"]
            values[4] = policy["tier"]
            values[5] = "16:9 — widescreen landscape"
            values[6] = policy["short_edge"]
            values[7] = policy["width"]
            values[8] = policy["height"]
            values[10] = policy["memory"]
        elif node["type"] == "WeeToddH3ValidatedSamplingPreset":
            node["widgets_values"] = [policy["preset"]]
        elif node["type"] == "WeeToddH3ComponentLoader" and node["widgets_values"][1] == "ref2va":
            node["widgets_values"][8] = False


def _empty_media(workflow: dict) -> None:
    for node in workflow["nodes"]:
        if node.get("type") in MEDIA_NODES and node.get("widgets_values"):
            node["widgets_values"][0] = ""


def _apply_task_prompt(workflow: dict, task: str) -> None:
    node_type = {
        "t2v": "WeeToddH3TextEncode",
        "i2v": "WeeToddH3KeyframeEncode",
        "fflf2va": "WeeToddH3KeyframeEncode",
        "ref2va": "WeeToddH3ReferenceEncode",
    }[task]
    node = next(node for node in workflow["nodes"] if node.get("type") == node_type)
    node["widgets_values"][0] = PROMPTS[task]


def _apply_output_metadata(workflow: dict, profile: str, task: str) -> None:
    publisher = next(
        node for node in workflow["nodes"] if node.get("type") == "WeeToddH3DirectPublishLatents"
    )
    config = next(
        node for node in workflow["nodes"] if node.get("type") == "WeeToddH3GenerationConfig"
    )
    publisher["widgets_values"][0] = f"WeeTodd/H3_{task}_{profile}"
    publisher["widgets_values"][3] = json.dumps(
        {
            "workflow": f"h3_{task}_{profile}",
            "profile": profile,
            "task": task,
            "seed": config["widgets_values"][2],
            "real_transformer_evaluations": PROFILE[profile]["steps"] - 1,
            "cache": "disabled",
            "preview": "enabled",
            "reference_sizing": (
                "output-matched pixel area" if task == "ref2va" else "not applicable"
            ),
        },
        separators=(",", ":"),
    )


def _apply_group_titles(workflow: dict, profile: str, task: str) -> None:
    label = {
        "t2v": "T2V + audio",
        "i2v": "I2V + audio",
        "fflf2va": "first / last frame + audio",
        "ref2va": "Ref2VA",
    }[task]
    for index, group in enumerate(workflow.get("groups", []), start=1):
        suffix = "" if index == 1 else f" — section {index}"
        group["title"] = f"H3 {label} — {profile} profile{suffix}"


def _add_preview(workflow: dict) -> None:
    nodes = workflow["nodes"]
    if any(node.get("type") == "WeeToddH3PreviewOverride" for node in nodes):
        return
    loader = next(node for node in nodes if node.get("type") == "WeeToddH3ComponentLoader")
    old_links = list(loader["outputs"][0].get("links") or [])
    node_id = max(int(node["id"]) for node in nodes) + 1
    link_id = max((int(link[0]) for link in workflow.get("links", [])), default=0) + 1
    for link in workflow["links"]:
        if link[0] in old_links and link[1] == loader["id"] and link[2] == 0:
            link[1] = node_id
    loader["outputs"][0]["links"] = [link_id]
    workflow["links"].append(
        [link_id, loader["id"], 0, node_id, 0, "WEETODD_H3_COMPONENTS"]
    )
    nodes.append(
        {
            "id": node_id,
            "type": "WeeToddH3PreviewOverride",
            "pos": [loader["pos"][0] + loader["size"][0] + 40, loader["pos"][1]],
            "size": [390, 270],
            "flags": {},
            "order": loader.get("order", 0) + 1,
            "mode": 0,
            "inputs": [
                {"name": "components", "type": "WEETODD_H3_COMPONENTS", "link": link_id}
            ],
            "outputs": [
                {
                    "name": "components",
                    "type": "WEETODD_H3_COMPONENTS",
                    "links": old_links,
                    "slot_index": 0,
                }
            ],
            "properties": {"Node name for S&R": "WeeToddH3PreviewOverride"},
            "widgets_values": [
                "taeh3.safetensors",
                "auto",
                "taeh3_coreml_256.mlpackage",
                1,
                6,
                256,
                "conservative collapse guard",
            ],
        }
    )
    workflow["last_node_id"] = max(int(workflow.get("last_node_id", 0)), node_id)
    workflow["last_link_id"] = max(int(workflow.get("last_link_id", 0)), link_id)


def _first_last(workflow: dict) -> None:
    nodes = workflow["nodes"]
    first_loader = next(node for node in nodes if node.get("type") == "LoadImage")
    endpoint = next(node for node in nodes if node.get("type") == "WeeToddH3FirstFrame")
    node_id = max(int(node["id"]) for node in nodes) + 1
    link_id = max((int(link[0]) for link in workflow["links"]), default=0) + 1
    last_loader = copy.deepcopy(first_loader)
    last_loader["id"] = node_id
    last_loader["pos"] = [first_loader["pos"][0], first_loader["pos"][1] + 350]
    last_loader["outputs"][0]["links"] = [link_id]
    last_loader["widgets_values"][0] = ""
    endpoint["type"] = "WeeToddH3FirstLastFrame"
    endpoint["properties"]["Node name for S&R"] = "WeeToddH3FirstLastFrame"
    endpoint["inputs"].append({"name": "last_frame", "type": "IMAGE", "link": link_id})
    workflow["links"].append([link_id, node_id, 0, endpoint["id"], 1, "IMAGE"])
    nodes.append(last_loader)
    workflow["last_node_id"] = max(int(workflow.get("last_node_id", 0)), node_id)
    workflow["last_link_id"] = max(int(workflow.get("last_link_id", 0)), link_id)


def _write(profile: str, task: str, workflow: dict) -> None:
    target = WORKFLOWS / profile / task / f"h3_{task}_{profile}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    note = next(
        node
        for node in workflow["nodes"]
        if node.get("type") == "MarkdownNote" and node.get("title") == NOTE_TITLE
    )
    note["widgets_values"] = [_model_note(target, workflow)]
    target.write_text(json.dumps(workflow, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    templates = {
        "t2v": _read(WORKFLOWS / "speed/t2v/h3_t2v_speed.json"),
        "i2v": _read(WORKFLOWS / "balance/i2v/h3_i2v_balance.json"),
        "ref2va": _read(WORKFLOWS / "speed/ref2va/h3_ref2va_speed.json"),
    }
    for profile in PROFILE:
        for task in ("t2v", "i2v", "fflf2va", "ref2va"):
            source_task = "i2v" if task == "fflf2va" else task
            workflow = copy.deepcopy(templates[source_task])
            if task == "fflf2va":
                _first_last(workflow)
            _apply_profile(workflow, profile)
            _apply_task_prompt(workflow, task)
            _apply_output_metadata(workflow, profile, task)
            _apply_group_titles(workflow, profile, task)
            _empty_media(workflow)
            _add_preview(workflow)
            _write(profile, task, workflow)


if __name__ == "__main__":
    main()
