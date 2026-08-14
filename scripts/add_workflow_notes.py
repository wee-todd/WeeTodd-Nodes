#!/usr/bin/env python3
"""Add or refresh the setup note in every shipped ComfyUI UI workflow."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTE_TITLE = "Setup and model downloads"

H3_OFFICIAL = "https://huggingface.co/MiniMaxAI/MiniMax-H3"
H3_Q8_TRANSFORMER = "https://huggingface.co/Vayden/MiniMax-H3-MLX-q8-extended-paged"
H3_Q8_QWEN = "https://huggingface.co/Vayden/Qwen3-VL-32B-H3-MLX-q8-paged"
H3_Q8_VAE = "https://huggingface.co/Vayden/MiniMax-H3-Video-VAE-MLX-Q8"
DRBAPH_V4 = (
    "https://huggingface.co/drbaph/MiniMax-H3-Turbo-Lora-ComfyUI/blob/main/"
    "minimax_h3_turbo_v4_step600_ema_pruned_comfyui.safetensors"
)
LARRY_TURBO = "https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora"
LTX23 = "https://huggingface.co/Lightricks/LTX-2.3"
GEMMA3_MLX = "https://huggingface.co/mlx-community/gemma-3-12b-it-4bit"
LTX25 = "https://huggingface.co/Lightricks/LTX-2.5"
LTX25_UPSCALER = "https://huggingface.co/Lightricks/LTX-2.5-22b-IC-LoRA-Pixel-Spatial-Upscaler"


def _workflow_paths(project: Path) -> list[Path]:
    return sorted((project / "workflows").rglob("*.json"))


def _first_string_widget(node: dict) -> str | None:
    for value in node.get("widgets_values", []):
        if isinstance(value, str) and value:
            return value
        if isinstance(value, dict):
            filename = value.get("filename")
            if isinstance(filename, str) and filename:
                return filename
    return None


def _model_note(path: Path, workflow: dict) -> str:
    nodes = workflow.get("nodes", [])
    types = {node.get("type") for node in nodes}
    lines = [
        f"## {path.stem.replace('_', ' ')}",
        "",
        "1. Install WeeTodd Nodes and restart ComfyUI.",
    ]

    if any(str(node_type).startswith("WeeToddLTX23") for node_type in types):
        lines.extend(
            [
                "2. Download the licensed LTX 2.3 files and the Gemma encoder.",
                "3. Assemble the README model bundle under `ComfyUI/models/LTX-2.3/q8`.",
                "4. Queue the workflow after **LTX 2.3 Preflight** succeeds.",
                "",
                "### Downloads",
                "",
                f"- [LTX 2.3 model files]({LTX23})",
                f"- [Gemma 3 MLX encoder]({GEMMA3_MLX})",
            ]
        )
        return "\n".join(lines)

    if any(str(node_type).startswith("WeeToddLTX25") for node_type in types):
        upscale = "WeeToddLTX25VideoUpscale" in types
        lines.extend(
            [
                "2. Accept the LTX 2.5 license and download the split model components.",
                "3. Use `ComfyUI/models/` or shared roots from `extra_model_paths.yaml`.",
                (
                    "4. Select a source movie. Queue the workflow."
                    if upscale
                    else "4. Queue the workflow after **LTX 2.5 Preflight** succeeds."
                ),
                "",
                "### Downloads",
                "",
                f"- [LTX 2.5 model components]({LTX25})",
            ]
        )
        if upscale:
            lines.append(f"- [Pixel-spatial upscaler IC-LoRA]({LTX25_UPSCALER})")
        return "\n".join(lines)

    component_nodes = [node for node in nodes if node.get("type") == "WeeToddH3ComponentLoader"]
    component_values = component_nodes[0].get("widgets_values", []) if component_nodes else []
    component_text = " ".join(str(value) for value in component_values)
    preset_text = " ".join(
        str(value)
        for node in nodes
        if node.get("type") == "WeeToddH3ValidatedSamplingPreset"
        for value in node.get("widgets_values", [])
    )

    media = []
    media_types = []
    for node in nodes:
        if node.get("type") in {"LoadImage", "LoadVideo"}:
            media_types.append(node.get("type"))
            value = _first_string_widget(node)
            if value and value not in media:
                media.append(value)

    lines.extend(
        [
            "2. Download the H3 components listed below.",
            "3. Keep the relative Component Loader paths unchanged.",
        ]
    )
    if media_types:
        inputs = []
        if "LoadImage" in media_types:
            inputs.append("required images")
        if "LoadVideo" in media_types:
            inputs.append("required video or audio source")
        lines.append("4. Select the " + " and ".join(inputs) + " in ComfyUI.")
        queue_step = 5
    else:
        queue_step = 4
    if "WeeToddH3Preflight" in types:
        lines.append(f"{queue_step}. Queue the workflow after **H3 Preflight** succeeds.")
    else:
        lines.append(f"{queue_step}. Queue the workflow after all required inputs are selected.")
    lines.extend(
        [
            "",
            "### H3 downloads",
            "",
            "- [Official MiniMax H3 partition, processor, tokenizer, and audio VAE]"
            f"({H3_OFFICIAL})",
            "  Save the repository under `ComfyUI/models/MiniMax-H3`.",
        ]
    )

    if "q8_extended_paged" in component_text:
        lines.extend(
            [
                f"- [Q8-extended paged transformer]({H3_Q8_TRANSFORMER})",
                "  Save it under `ComfyUI/models/MiniMax-H3/transformers/q8_extended_paged`.",
                f"- [Qwen3-VL Q8 paged conditioner]({H3_Q8_QWEN})",
                "  Save it under `ComfyUI/models/MiniMax-H3/text_encoders/q8-paged`.",
                f"- [MiniMax H3 video VAE MLX Q8]({H3_Q8_VAE})",
                "  Save it under `ComfyUI/models/MiniMax-H3/vae/q8`.",
            ]
        )

    if "drbaph" in preset_text:
        lines.extend(
            [
                "",
                "### Required LoRA",
                "",
                f"- [drbaph v4 step-600 EMA adapter]({DRBAPH_V4})",
                "- Save it as `ComfyUI/models/loras/"
                "minimax_h3_turbo_v4_step600_ema_pruned_comfyui.safetensors`.",
            ]
        )
    elif "Larry" in preset_text:
        lora_name = (
            "minimax_h3_turbo_4step_ema_ckpt850.safetensors"
            if "EMA-850" in preset_text
            else "minimax_h3_turbo_v4_step600_ema.safetensors"
        )
        lines.extend(
            [
                "",
                "### Required LoRA",
                "",
                f"- [Larry Turbo LoRA repository]({LARRY_TURBO})",
                f"- Save `{lora_name}` under `ComfyUI/models/loras`.",
            ]
        )
    elif "LightX2V" in preset_text:
        lora_name = (
            "minimax_h3_fl2v_lightx2v_turbo_4step_v0.1_comfy_resized_avg_rank_21_bf16.safetensors"
            if "rank 21" in preset_text
            else "minimax_h3_fl2v_lightx2v_turbo_4step_v0.1_comfy.safetensors"
        )
        lines.extend(
            [
                "",
                "### Required LoRA",
                "",
                f"- Save the existing `{lora_name}` under `ComfyUI/models/loras`.",
                "- This legacy comparison does not publish an adapter download link.",
                "- Use a drbaph workflow when you need a documented adapter source.",
            ]
        )

    if "ref2va" in component_text.lower():
        lines.extend(
            [
                "",
                "Use a genuine Ref2VA partition for strict Ref2VA workflows.",
                "Do not substitute FL2VA weights unless the workflow enables compatibility mode.",
            ]
        )
    return "\n".join(lines)


def _node_spans(raw: str) -> tuple[list[tuple[dict, int, int]], int]:
    match = re.search(r'"nodes"\s*:\s*\[', raw)
    if match is None:
        raise ValueError("Workflow JSON does not contain a nodes array.")
    decoder = json.JSONDecoder()
    cursor = match.end()
    spans = []
    while True:
        while cursor < len(raw) and (raw[cursor].isspace() or raw[cursor] == ","):
            cursor += 1
        if cursor >= len(raw):
            raise ValueError("Workflow nodes array is not terminated.")
        if raw[cursor] == "]":
            return spans, cursor
        node, end = decoder.raw_decode(raw, cursor)
        if not isinstance(node, dict):
            raise ValueError("Workflow nodes array contains a non-object value.")
        spans.append((node, cursor, end))
        cursor = end


def _render_node(node: dict, prefix: str) -> str:
    rendered = json.dumps(node, indent=2, ensure_ascii=False)
    return rendered.replace("\n", "\n" + prefix)


def _upsert_note(path: Path) -> bool:
    raw = path.read_text()
    workflow = json.loads(raw)
    nodes = workflow.get("nodes", [])
    existing = next(
        (
            node
            for node in nodes
            if node.get("type") == "MarkdownNote" and node.get("title") == NOTE_TITLE
        ),
        None,
    )
    note_text = _model_note(path, workflow)
    min_x = min((float(node.get("pos", [0, 0])[0]) for node in nodes), default=0.0)
    min_y = min((float(node.get("pos", [0, 0])[1]) for node in nodes), default=0.0)
    node_id = max((int(node.get("id", 0)) for node in nodes), default=0) + 1
    order = max((int(node.get("order", 0)) for node in nodes), default=-1) + 1
    note = {
        "id": node_id,
        "type": "MarkdownNote",
        "pos": [min_x, min_y - 560.0],
        "size": [760, 500],
        "flags": {},
        "order": order,
        "mode": 0,
        "inputs": [],
        "outputs": [],
        "title": NOTE_TITLE,
        "properties": {},
        "widgets_values": [note_text],
        "color": "#243447",
        "bgcolor": "#15202b",
    }
    spans, _ = _node_spans(raw)
    if existing is not None:
        note["id"] = existing["id"]
        note["pos"] = existing.get("pos", note["pos"])
        note["size"] = existing.get("size", note["size"])
        note["order"] = existing.get("order", note["order"])
        _, start, end = next(
            span
            for span in spans
            if span[0].get("id") == existing["id"] and span[0].get("type") == "MarkdownNote"
        )
        line_start = raw.rfind("\n", 0, start) + 1
        prefix = raw[line_start:start]
        rendered = raw[:start] + _render_node(note, prefix) + raw[end:]
    else:
        if not spans:
            raise ValueError("Workflow nodes array must contain at least one node.")
        _, last_start, last_end = spans[-1]
        line_start = raw.rfind("\n", 0, last_start) + 1
        prefix = raw[line_start:last_start]
        insertion = ",\n" + prefix + _render_node(note, prefix)
        rendered = raw[:last_end] + insertion + raw[last_end:]
        rendered, substitutions = re.subn(
            r'("last_node_id"\s*:\s*)\d+',
            rf"\g<1>{max(int(workflow.get('last_node_id', 0)), node_id)}",
            rendered,
            count=1,
        )
        if substitutions != 1:
            raise ValueError("Workflow JSON does not contain last_node_id.")

    if rendered == raw:
        return False
    path.write_text(rendered)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=ROOT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    changed = []
    for path in _workflow_paths(args.project.resolve()):
        original = path.read_text()
        would_change = _upsert_note(path)
        if args.check and would_change:
            path.write_text(original)
        if would_change:
            changed.append(path)
    if args.check and changed:
        for path in changed:
            print(path.relative_to(args.project.resolve()))
        return 1
    print(f"Updated {len(changed)} workflow note(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
