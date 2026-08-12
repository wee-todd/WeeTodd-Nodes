#!/usr/bin/env python3
"""Build portable 15-second one-shot and chained H3 comparison workflows."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from build_h3_chained_workflows import PROMPTS

from wee_todd_nodes.nodes import NODE_CLASS_MAPPINGS

ROOT = Path(__file__).parents[1]
LORA = "minimax_h3_turbo_v4_step600_ema_pruned_comfyui.safetensors"
ONE_SHOT_PRESET = (
    "One-shot staged Turbo — drbaph v4 step-600 — 15-second quality baseline"
)
CHAIN_PRESET = (
    "Chained staged Turbo — drbaph v4 step-600 — 4 windows / 22-frame context"
)
ONE_SHOT_SEED = 20_260_811
CHAIN_SEED = 20_260_812
PORTABLE_COMPONENTS = {
    "checkpoint": "MiniMax-H3/FL2VA",
    "task": "t2va",
    "transformer": "MiniMax-H3/transformers/q8_extended_paged",
    "text_encoder": "MiniMax-H3/text_encoders/q8-paged",
    "processor": "MiniMax-H3/FL2VA/processor",
    "tokenizer": "MiniMax-H3/FL2VA/tokenizer",
    "video_vae": "MiniMax-H3/vae/q8/video_vae_affine_q8.safetensors",
    "audio_vae": "MiniMax-H3/FL2VA/audio_vae",
    "allow_fl2va_weights_for_ref2va": False,
}
ONE_SHOT_PROMPT = (
    "integrated_multimodal_description: [Shot 1] Live-action mockumentary crossover "
    "parody with realistic skin, clothing, soil, and physically natural movement, presented "
    "as one continuous handheld shot at a barren desert excavation site in harsh afternoon "
    "sunlight. Jesse, a lean young white man with a shaved head, dusty dark hoodie, work "
    "gloves, and black work trousers, stands inside a deep grave-sized hole. During the "
    "opening four seconds, Jesse angrily throws several heavy shovelfuls of dirt over the "
    "rim; the shovel strikes compact soil, loose dirt scatters, and his breathing grows "
    "heavier. Dwight, a rigid middle-aged white man with neatly parted brown hair, wire-frame "
    "glasses, and a mustard-yellow short-sleeve shirt, walks into frame above the hole, "
    "studies it suspiciously, adjusts his glasses, looks down with a smug expression, and "
    "says <d>[English] Hello, little man. Digging your own grave? </d> Jesse immediately "
    "stops digging, slowly turns his head and shoulders toward Dwight, plants his boots, "
    "grips the shovel tightly with both hands, and replies with a cold threatening expression "
    "<d>[English] No, no, no... I'm digging it for you. Just like I did for Michael. </d> "
    "Dwight's confidence collapses. His eyes widen behind his glasses, his face fills with "
    "horror, and he begins crying and panicking. He drops to his knees at the rim, grabs Jesse "
    "by both shoulders, shakes him once, and screams <d>[English] No! What did you do to "
    "Michael?! Where is he? Tell me now! </d> Jesse remains completely emotionless and "
    "silent, holding the shovel upright. The handheld camera makes restrained documentary "
    "corrections, pushes closer as Dwight panics, and ends abruptly on a tight view of "
    "Dwight's terrified face with Jesse still visible in the shallow background. Do not cut, "
    "fade, teleport, duplicate either man, change clothing, reverse screen positions, or "
    "reset the action.\n\n"
    "overall_soundscape: Continuous dry desert wind and quiet outdoor ambience; synchronized "
    "shovel impacts, scraping soil, scattering dirt, Jesse's heavy breathing, Dwight's "
    "footsteps on gravel, clothing rustle as he adjusts his glasses, the final shovel scrape, "
    "knees striking loose soil, one brief shoulder shake, Dwight's crying and panicked "
    "breathing, and clean location-recorded dialogue synchronized to each speaker's mouth. "
    "Preserve one continuous acoustic space without an ambience reset.\n\n"
    "non_diegetic_music: N/A"
)


@dataclass(frozen=True)
class Ref:
    node: int
    slot: int = 0


class Workflow:
    def __init__(self, title: str) -> None:
        self.title = title
        self.nodes: dict[int, dict] = {}

    def add(
        self,
        node_id: int,
        node_type: str,
        values: dict[str, object],
        *,
        pos: tuple[int, int],
        size: tuple[int, int] = (390, 260),
    ) -> None:
        self.nodes[node_id] = {
            "id": node_id,
            "type": node_type,
            "values": values,
            "pos": list(pos),
            "size": list(size),
        }

    @staticmethod
    def _schema(node_type: str) -> list[tuple[str, tuple]]:
        raw = NODE_CLASS_MAPPINGS[node_type].INPUT_TYPES()
        return [
            (name, specification)
            for group in ("required", "optional")
            for name, specification in raw.get(group, {}).items()
        ]

    def api(self) -> dict[str, dict]:
        payload = {}
        for node_id, node in self.nodes.items():
            inputs = {}
            for name, value in node["values"].items():
                inputs[name] = [str(value.node), value.slot] if isinstance(value, Ref) else value
            payload[str(node_id)] = {"class_type": node["type"], "inputs": inputs}
        return payload

    def ui(self) -> dict:
        links: list[list] = []
        rendered = {}
        next_link = 1
        for order, (node_id, node) in enumerate(self.nodes.items()):
            node_class = NODE_CLASS_MAPPINGS[node["type"]]
            schema = self._schema(node["type"])
            connected = []
            widgets = []
            for name, specification in schema:
                value = node["values"].get(name)
                if isinstance(value, Ref):
                    source_class = NODE_CLASS_MAPPINGS[self.nodes[value.node]["type"]]
                    value_type = source_class.RETURN_TYPES[value.slot]
                    connected.append(
                        {"name": name, "type": value_type, "link": next_link}
                    )
                    links.append(
                        [next_link, value.node, value.slot, node_id, len(connected) - 1, value_type]
                    )
                    next_link += 1
                    continue
                input_type = specification[0]
                options = specification[1] if len(specification) > 1 else {}
                if isinstance(input_type, list) or input_type in {
                    "BOOLEAN",
                    "FLOAT",
                    "INT",
                    "STRING",
                }:
                    if name not in node["values"]:
                        value = options.get("default")
                    widgets.append((name, value))
            if node["type"] == "WeeToddH3GenerationConfig":
                short_edge = next(item for item in widgets if item[0] == "short_edge")
                widgets.remove(short_edge)
                ratio_index = next(i for i, item in enumerate(widgets) if item[0] == "aspect_ratio")
                widgets.insert(ratio_index + 1, short_edge)
            rendered[node_id] = {
                "id": node_id,
                "type": node["type"],
                "pos": node["pos"],
                "size": node["size"],
                "flags": {},
                "order": order,
                "mode": 0,
                "inputs": connected,
                "outputs": [
                    {
                        "name": name,
                        "type": value_type,
                        "links": [],
                        "slot_index": slot,
                    }
                    for slot, (name, value_type) in enumerate(
                        zip(node_class.RETURN_NAMES, node_class.RETURN_TYPES, strict=True)
                    )
                ],
                "properties": {"Node name for S&R": node["type"]},
                "widgets_values": [value for _, value in widgets],
            }
        for link_id, source, source_slot, *_ in links:
            rendered[source]["outputs"][source_slot]["links"].append(link_id)
        for node in rendered.values():
            for output in node["outputs"]:
                if not output["links"]:
                    output["links"] = None
        return {
            "last_node_id": max(rendered),
            "last_link_id": next_link - 1,
            "nodes": list(rendered.values()),
            "links": links,
            "groups": [
                {
                    "title": self.title,
                    "bounding": [20, 20, 4100, 2250],
                    "color": "#3f789e",
                    "font_size": 24,
                    "flags": {},
                }
            ],
            "config": {},
            "extra": {"ds": {"scale": 0.5, "offset": [40, 70]}},
            "version": 0.4,
        }


def generation_values(
    duration: float, seed: int, *, exact_dimensions: bool = False
) -> dict[str, object]:
    return {
        "duration_seconds": duration,
        "steps": 7,
        "seed": seed,
        "resolution_mode": "exact dimensions" if exact_dimensions else "ratio + size",
        "resolution_tier": "768 px short edge — native",
        "aspect_ratio": "16:9 — widescreen landscape",
        "short_edge": 768,
        "custom_width": 1344,
        "custom_height": 768,
        "drop_adaln": True,
        "memory_mode": "normal",
        "attention_chunk_size": "automatic",
        "projection_backend": "mlx",
        "sampling_method": "euler",
    }


def common_prefix(
    workflow: Workflow,
    duration: float,
    preset: str,
    seed: int,
    *,
    exact_dimensions: bool = False,
) -> None:
    workflow.add(
        1,
        "WeeToddH3ComponentLoader",
        PORTABLE_COMPONENTS,
        pos=(40, 60),
        size=(400, 340),
    )
    workflow.add(
        2,
        "WeeToddH3GenerationConfig",
        generation_values(duration, seed, exact_dimensions=exact_dimensions),
        pos=(40, 450),
        size=(410, 420),
    )
    workflow.add(
        3,
        "WeeToddH3ValidatedSamplingPreset",
        {"config": Ref(2), "preset": preset},
        pos=(500, 470),
        size=(500, 230),
    )
    workflow.add(
        4,
        "WeeToddH3Preflight",
        {
            "components": Ref(1),
            "config": Ref(3),
            "prompt_tokens": 1024,
            "available_memory_gb": 0.0,
            "ffmpeg_path": "",
        },
        pos=(500, 80),
    )


def build_one_shot() -> Workflow:
    workflow = Workflow("15-second one-shot staged-Turbo quality baseline")
    common_prefix(
        workflow,
        15.0,
        ONE_SHOT_PRESET,
        ONE_SHOT_SEED,
        exact_dimensions=True,
    )
    workflow.add(
        5,
        "WeeToddH3TextEncode",
        {
            "components": Ref(4),
            "prompt": ONE_SHOT_PROMPT,
            "unload_after_encode": True,
            "config": Ref(3),
        },
        pos=(1080, 80),
        size=(650, 540),
    )
    workflow.add(
        6,
        "WeeToddH3Sample",
        {
            "components": Ref(4),
            "conditioning": Ref(5),
            "config": Ref(3),
            "unload_after_sample": True,
            "loras": Ref(3, 1),
        },
        pos=(1800, 130),
    )
    workflow.add(
        7,
        "WeeToddH3DirectPublishLatents",
        {
            "components": Ref(4),
            "latents": Ref(6),
            "filename_prefix": "WeeTodd/H3_768p_15s_OneShot_Staged_Turbo",
            "crf": 18,
            "max_av_drift_seconds": 0.025,
            "generation_metadata": json.dumps(
                {
                    "workflow": "h3_768p_15s_one_clip_staged_turbo",
                    "seed": ONE_SHOT_SEED,
                    "lora": LORA,
                }
            ),
            "sampling_info": Ref(6, 1),
            "ffmpeg_path": "",
        },
        pos=(2270, 120),
        size=(470, 320),
    )
    return workflow


def build_chain() -> Workflow:
    workflow = Workflow("15-second four-window H3 chain with repaired audiovisual joins")
    common_prefix(workflow, 4.0, CHAIN_PRESET, CHAIN_SEED)
    workflow.add(
        5,
        "WeeToddH3ChainedTimeline",
        {
            "window_duration_seconds": 4.0,
            "window_count": 4,
            "context_frames": "22",
            "target_duration_seconds": 15.0,
        },
        pos=(500, 760),
        size=(390, 250),
    )
    text_ids = (6, 10, 14, 18)
    sample_ids = (7, 11, 15, 19)
    append_ids = (8, 12, 16, 20)
    continuation_ids = (9, 13, 17)
    for index in range(4):
        y = 60 + index * 500
        text_values = {
            "components": Ref(4),
            "prompt": PROMPTS[index],
            "unload_after_encode": index == 3,
            "config": Ref(3),
        }
        workflow.add(
            text_ids[index],
            "WeeToddH3TextEncode",
            text_values,
            pos=(1080, y),
            size=(650, 410),
        )
        sample_values = {
            "components": Ref(4),
            "conditioning": Ref(text_ids[index]),
            "config": Ref(3),
            "unload_after_sample": index == 3,
            "loras": Ref(3, 1),
        }
        if index:
            sample_values["continuation"] = Ref(continuation_ids[index - 1])
        workflow.add(
            sample_ids[index],
            "WeeToddH3Sample",
            sample_values,
            pos=(1800, y + 40),
        )
        append_values = {"timeline": Ref(5), "latents": Ref(sample_ids[index])}
        if index:
            append_values["previous_chain"] = Ref(append_ids[index - 1])
        workflow.add(
            append_ids[index],
            "WeeToddH3ChainAppend",
            append_values,
            pos=(2250, y + 30),
        )
        if index < 3:
            workflow.add(
                continuation_ids[index],
                "WeeToddH3ContinuationContext",
                {"latents": Ref(sample_ids[index]), "context_frames": "22"},
                pos=(2250, y + 300),
                size=(360, 190),
            )
    workflow.add(
        21,
        "WeeToddH3DirectPublishChain",
        {
            "components": Ref(4),
            "chain": Ref(20),
            "filename_prefix": "WeeTodd/H3_768p_15s_Chain_JoinRepair",
            "crf": 18,
            "max_av_drift_seconds": 0.025,
            "generation_metadata": json.dumps(
                {
                    "workflow": "h3_768p_15s_four_window_join_repair",
                    "seed": CHAIN_SEED,
                    "lora": LORA,
                    "join_policy": (
                        "motion-matched overlap + 4-frame cosine blend + "
                        "50-ms audio crossfade"
                    ),
                }
            ),
            "ffmpeg_path": "",
        },
        pos=(2790, 760),
        size=(500, 330),
    )
    return workflow


def write(name: str, workflow: Workflow) -> None:
    (ROOT / "workflows" / f"{name}.json").write_text(
        json.dumps(workflow.ui(), indent=2) + "\n"
    )
    (ROOT / "examples" / f"{name}_api.json").write_text(
        json.dumps(workflow.api(), indent=2) + "\n"
    )


if __name__ == "__main__":
    write("h3_768p_15s_one_clip_staged_turbo", build_one_shot())
    write("h3_768p_15s_four_window_join_repair", build_chain())
