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
PORTABLE_REF2VA_COMPONENTS = {
    "checkpoint": "MiniMax-H3/Ref2VA",
    "task": "ref2va",
    "transformer": "MiniMax-H3/Ref2VA/transformer",
    "text_encoder": "MiniMax-H3/Ref2VA/text_encoder",
    "processor": "MiniMax-H3/Ref2VA/processor",
    "tokenizer": "MiniMax-H3/Ref2VA/tokenizer",
    "video_vae": "MiniMax-H3/Ref2VA/video_vae",
    "audio_vae": "MiniMax-H3/Ref2VA/audio_vae",
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
ALIEN_REF2VA_PROMPT = (
    "<Picture 1> defines the single tall white extraterrestrial's elongated face, pale white "
    "skin, large blue-gray eyes, fine swept-back white hair, very tall slender proportions, "
    "hands, feet, and fitted white clothing from all shown angles. <Picture 2> defines the "
    "single short grey extraterrestrial's oversized smooth head, enormous glossy black eyes, "
    "small mouth, gray-green skin texture, lean ribbed torso, long fingers, and compact body "
    "from all shown angles. Preserve them as two distinct individuals without merging their "
    "faces, eye colors, heights, skin colors, or body proportions.\n\n"
    "integrated_multimodal_description: [Shot 1] Live-action mockumentary crossover parody "
    "with photorealistic skin, clothing, soil, and physically natural movement, presented as "
    "one continuous handheld shot at a barren desert excavation site in harsh afternoon "
    "sunlight. The short grey extraterrestrial from <Picture 2> stands inside a deep "
    "grave-sized hole. During the opening four seconds, the grey angrily throws several heavy "
    "shovelfuls of dirt over the rim; the shovel strikes compact soil, loose dirt scatters, "
    "and its breathing grows heavier. The tall white extraterrestrial from <Picture 1> walks "
    "into frame above the hole, studies it suspiciously, brushes fine white hair behind one "
    "ear, looks down with a smug expression, and says <d>[English] Hello, little man. Digging "
    "your own grave? </d> The grey immediately stops digging, slowly turns its head and "
    "shoulders toward the tall white, plants its feet, grips the shovel tightly with both "
    "hands, and replies with a cold threatening expression <d>[English] No, no, no... I'm "
    "digging it for you. Just like I did for Michael. </d> The tall white's confidence "
    "collapses. Its large blue-gray eyes widen, its face fills with horror, and it begins "
    "crying and panicking. It drops to its knees at the rim, grabs the grey by both shoulders, "
    "shakes it once, and screams <d>[English] No! What did you do to Michael?! Where is he? "
    "Tell me now! </d> The grey remains completely emotionless and silent, holding the shovel "
    "upright. The handheld camera makes restrained documentary corrections, pushes closer as "
    "the tall white panics, and ends abruptly on a tight view of the tall white's terrified "
    "face with the grey still visible in the shallow background. Do not cut, fade, teleport, "
    "duplicate either alien, swap their features, reverse screen positions, or reset the "
    "action.\n\n"
    "overall_soundscape: Continuous dry desert wind and quiet outdoor ambience; synchronized "
    "shovel impacts, scraping soil, scattering dirt, the grey's heavy breathing, the tall "
    "white's footsteps on gravel, clothing rustle, the final shovel scrape, knees striking "
    "loose soil, one brief shoulder shake, crying and panicked breathing, and clean "
    "location-recorded dialogue synchronized to each speaker's mouth. Preserve one continuous "
    "acoustic space without an ambience reset.\n\n"
    "non_diegetic_music: N/A"
)

ALIEN_REFERENCE_PREFIX = (
    "<Picture 1> defines the single tall white extraterrestrial's elongated face, pale white "
    "skin, large blue-gray eyes, fine swept-back white hair, very tall slender proportions, "
    "hands, feet, and fitted white clothing from all shown angles. <Picture 2> defines the "
    "single short grey extraterrestrial's oversized smooth head, enormous glossy black eyes, "
    "small mouth, gray-green skin texture, lean ribbed torso, long fingers, and compact body "
    "from all shown angles. <Video 1> supplies the corresponding segment's live-action blocking, "
    "camera movement, physical timing, dialogue pacing, and continuous desert sound. Replace the "
    "two people in <Video 1> with the two extraterrestrials while preserving the characters from "
    "<Picture 1> and <Picture 2> as distinct individuals. Do not inherit either person's face, "
    "hair color, eye color, clothing, height, skin color, or body proportions from <Video 1>.\n\n"
)
ALIEN_CHAIN_PROMPTS = (
    ALIEN_REFERENCE_PREFIX
    + "integrated_multimodal_description: [Shot 1] Begin one continuous photorealistic "
    "mockumentary shot at the barren desert excavation site shown in <Video 1>, in harsh "
    "afternoon sunlight. The short grey from <Picture 2> stands inside the grave-sized hole and "
    "angrily throws several heavy shovelfuls of dirt over the rim. The shovel strikes compact "
    "soil, dirt scatters, and the grey breathes heavily. The tall white from <Picture 1> walks "
    "into frame above the hole, studies it suspiciously, brushes fine white hair behind one ear, "
    "looks down smugly, and says <d>[English] Hello, little man. Digging your own grave? </d> "
    "End with the grey frozen mid-dig below the tall white. Preserve the blocking and restrained "
    "handheld corrections from <Video 1>; do not cut, duplicate, teleport, or merge the aliens.\n\n"
    "overall_soundscape: Dry desert wind, shovel impacts, scattering dirt, the grey's heavy "
    "breathing, the tall white's gravel footsteps, clothing rustle, and clean synchronized "
    "dialogue. Preserve the acoustic pacing and continuous location ambience from <Video 1>.\n\n"
    "non_diegetic_music: N/A",
    ALIEN_REFERENCE_PREFIX
    + "integrated_multimodal_description: [Shot 1] Continue directly from the inherited latent "
    "frames with no cut or reset. Preserve the same grave, sunlight, screen positions, shovel, "
    "and both extraterrestrials. The short grey from <Picture 2> immediately stops digging, "
    "slowly turns its head and shoulders toward the tall white, plants its feet, and grips the "
    "shovel tightly with both hands. With a cold threatening expression it replies "
    "<d>[English] No, no, no... I'm digging it for you. Just like I did for Michael. </d> The "
    "tall white's smug expression begins to fail. Follow <Video 1>'s action timing and subtle "
    "handheld push while retaining the aliens' reference-defined anatomy.\n\n"
    "overall_soundscape: Continue the same dry wind without an ambience reset. One final shovel "
    "scrape stops, loose dirt settles, the grey's breathing slows, long fingers tighten on the "
    "handle, and its threatening dialogue remains synchronized. Use <Video 1> for pacing.\n\n"
    "non_diegetic_music: N/A",
    ALIEN_REFERENCE_PREFIX
    + "integrated_multimodal_description: [Shot 1] Continue directly from the inherited latent "
    "frames in the same unbroken handheld shot. The tall white from <Picture 1> loses all "
    "confidence; its large blue-gray eyes widen, its pale elongated face fills with horror, and "
    "it begins crying and panicking. It drops to its knees at the rim, reaches down, grabs the "
    "short grey from <Picture 2> by both shoulders, shakes it once, and screams "
    "<d>[English] No! What did you do to Michael?! Where is he? Tell me now! </d> The grey remains "
    "emotionless, holding the shovel. Follow <Video 1>'s blocking and camera jolt without "
    "transferring human features or changing either alien's identity.\n\n"
    "overall_soundscape: Continue the same desert wind. Knees strike loose soil, clothing pulls, "
    "one shoulder shake rustles both bodies, and the tall white cries and screams with "
    "synchronized mouth movement. Preserve <Video 1>'s acoustic timing without a reset.\n\n"
    "non_diegetic_music: N/A",
    ALIEN_REFERENCE_PREFIX
    + "integrated_multimodal_description: [Shot 1] Continue directly from the inherited latent "
    "frames. The handheld camera completes its push into a tight close-up of the tall white from "
    "<Picture 1>. Tears cross its pale skin, its blue-gray eyes remain wide, and its breathing is "
    "rapid and uneven. It looks from the short grey to the grave and back. The short grey from "
    "<Picture 2> remains silent and emotionless in the shallow background, holding the shovel "
    "upright. Hold the dreadful silence briefly and end abruptly on the tall white's terrified "
    "face. Match <Video 1>'s terminal framing while preserving both alien identities. No cut, "
    "fade, duplication, feature swap, or action reset.\n\n"
    "overall_soundscape: Continue the same wind and outdoor ambience. The tall white's panicked "
    "breathing and quiet sobbing dominate; the grey is silent; a few grains of dirt fall into the "
    "hole. End hard with no musical sting.\n\n"
    "non_diegetic_music: N/A",
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
        if node_type == "LoadImage":
            return [("image", ("STRING", {"default": "reference.png"}))]
        if node_type == "LoadVideo":
            return [("file", ("STRING", {"default": "reference.mp4"}))]
        if node_type == "Video Slice":
            return [
                ("video", ("VIDEO",)),
                ("start_time", ("FLOAT", {"default": 0.0})),
                ("duration", ("FLOAT", {"default": 0.0})),
                ("strict_duration", ("BOOLEAN", {"default": False})),
            ]
        if node_type == "GetVideoComponents":
            return [("video", ("VIDEO",))]
        raw = NODE_CLASS_MAPPINGS[node_type].INPUT_TYPES()
        return [
            (name, specification)
            for group in ("required", "optional")
            for name, specification in raw.get(group, {}).items()
        ]

    @staticmethod
    def _returns(node_type: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
        if node_type == "LoadImage":
            return ("IMAGE", "MASK"), ("IMAGE", "MASK")
        if node_type in {"LoadVideo", "Video Slice"}:
            return ("VIDEO",), ("video",)
        if node_type == "GetVideoComponents":
            return ("IMAGE", "AUDIO", "FLOAT", "INT"), (
                "images",
                "audio",
                "fps",
                "frame_count",
            )
        node_class = NODE_CLASS_MAPPINGS[node_type]
        return node_class.RETURN_TYPES, node_class.RETURN_NAMES

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
            schema = self._schema(node["type"])
            connected = []
            widgets = []
            for name, specification in schema:
                value = node["values"].get(name)
                if isinstance(value, Ref):
                    source_types, _ = self._returns(self.nodes[value.node]["type"])
                    value_type = source_types[value.slot]
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
            if node["type"] == "LoadImage":
                widgets = [("image", node["values"]["image"]), ("upload", "image")]
            elif node["type"] == "LoadVideo":
                widgets = [("file", node["values"]["file"])]
            return_types, return_names = self._returns(node["type"])
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
                        zip(return_names, return_types, strict=True)
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
    components: dict[str, object] = PORTABLE_COMPONENTS,
) -> None:
    workflow.add(
        1,
        "WeeToddH3ComponentLoader",
        components,
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


def build_alien_ref2va() -> Workflow:
    workflow = Workflow("15-second two-reference Ref2VA staged-Turbo alien scene")
    common_prefix(
        workflow,
        15.0,
        ONE_SHOT_PRESET,
        ONE_SHOT_SEED,
        exact_dimensions=True,
        components=PORTABLE_REF2VA_COMPONENTS,
    )
    workflow.add(
        5,
        "LoadImage",
        {"image": "tall_white_reference_sheet.png"},
        pos=(40, 930),
        size=(330, 330),
    )
    workflow.add(
        6,
        "WeeToddH3ReferenceImage",
        {"image": Ref(5), "pixel_budget_percent": 100},
        pos=(420, 980),
        size=(340, 190),
    )
    workflow.add(
        7,
        "LoadImage",
        {"image": "grey_alien_reference_sheet.png"},
        pos=(810, 930),
        size=(330, 330),
    )
    workflow.add(
        8,
        "WeeToddH3ReferenceImage",
        {
            "image": Ref(7),
            "pixel_budget_percent": 100,
            "previous_references": Ref(6),
        },
        pos=(1190, 980),
        size=(360, 210),
    )
    workflow.add(
        9,
        "WeeToddH3ReferenceEncode",
        {
            "components": Ref(4),
            "config": Ref(3),
            "references": Ref(8),
            "prompt": ALIEN_REF2VA_PROMPT,
        },
        pos=(1080, 60),
        size=(720, 600),
    )
    workflow.add(
        10,
        "WeeToddH3ReferenceStrength",
        {
            "conditioning": Ref(9),
            "visual_strength": 0.999,
            "audio_strength": 1.0,
        },
        pos=(1860, 110),
        size=(390, 220),
    )
    workflow.add(
        11,
        "WeeToddH3Sample",
        {
            "components": Ref(4),
            "conditioning": Ref(10),
            "config": Ref(3),
            "unload_after_sample": True,
            "loras": Ref(3, 1),
        },
        pos=(2320, 100),
        size=(410, 300),
    )
    workflow.add(
        12,
        "WeeToddH3DirectPublishLatents",
        {
            "components": Ref(4),
            "latents": Ref(11),
            "filename_prefix": "WeeTodd/H3_768p_15s_Ref2VA_Aliens_Staged_Turbo",
            "crf": 18,
            "max_av_drift_seconds": 0.025,
            "generation_metadata": json.dumps(
                {
                    "workflow": "h3_768p_15s_ref2va_aliens_staged_turbo",
                    "seed": ONE_SHOT_SEED,
                    "lora": LORA,
                    "reference_order": ["tall white alien", "grey alien"],
                    "reference_strength": {"visual": 0.999, "audio": 1.0},
                }
            ),
            "sampling_info": Ref(11, 1),
            "ffmpeg_path": "",
        },
        pos=(2800, 90),
        size=(510, 340),
    )
    return workflow


def build_alien_ref2va_chain() -> Workflow:
    workflow = Workflow(
        "15-second four-window Ref2VA alien chain with image, source-video, and latent context"
    )
    common_prefix(
        workflow,
        4.0,
        CHAIN_PRESET,
        CHAIN_SEED,
        exact_dimensions=True,
        components=PORTABLE_REF2VA_COMPONENTS,
    )
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
    workflow.add(
        6,
        "LoadImage",
        {"image": "tall_white_reference_sheet.png"},
        pos=(40, 1030),
        size=(330, 300),
    )
    workflow.add(
        7,
        "WeeToddH3ReferenceImage",
        {"image": Ref(6), "pixel_budget_percent": 100},
        pos=(410, 1080),
        size=(340, 190),
    )
    workflow.add(
        8,
        "LoadImage",
        {"image": "grey_alien_reference_sheet.png"},
        pos=(40, 1390),
        size=(330, 300),
    )
    workflow.add(
        9,
        "WeeToddH3ReferenceImage",
        {
            "image": Ref(8),
            "pixel_budget_percent": 100,
            "previous_references": Ref(7),
        },
        pos=(410, 1440),
        size=(350, 210),
    )
    workflow.add(
        10,
        "LoadVideo",
        {"file": "H3_768p_15s_One_Clip_Staged_Turbo_Benchmark_20260811.mp4"},
        pos=(40, 1760),
        size=(410, 250),
    )

    starts = (0.0, 85 / 24, 170 / 24, 255 / 24)
    slice_ids = (11, 20, 29, 38)
    component_ids = (12, 21, 30, 39)
    reference_ids = (13, 22, 31, 40)
    encode_ids = (14, 23, 32, 41)
    strength_ids = (15, 24, 33, 42)
    sample_ids = (16, 25, 34, 43)
    append_ids = (17, 26, 35, 44)
    continuation_ids = (18, 27, 36)
    for index in range(4):
        y = 60 + index * 560
        workflow.add(
            slice_ids[index],
            "Video Slice",
            {
                "video": Ref(10),
                "start_time": starts[index],
                "duration": 107 / 24,
                "strict_duration": False,
            },
            pos=(920, y),
            size=(360, 230),
        )
        workflow.add(
            component_ids[index],
            "GetVideoComponents",
            {"video": Ref(slice_ids[index])},
            pos=(1320, y),
            size=(340, 170),
        )
        workflow.add(
            reference_ids[index],
            "WeeToddH3ReferenceVideo",
            {
                "video_frames": Ref(component_ids[index], 0),
                "fps": Ref(component_ids[index], 2),
                "soundtrack": Ref(component_ids[index], 1),
                "previous_references": Ref(9),
            },
            pos=(1700, y),
            size=(390, 250),
        )
        workflow.add(
            encode_ids[index],
            "WeeToddH3ReferenceEncode",
            {
                "components": Ref(4),
                "config": Ref(3),
                "references": Ref(reference_ids[index]),
                "prompt": ALIEN_CHAIN_PROMPTS[index],
            },
            pos=(2140, y),
            size=(700, 470),
        )
        workflow.add(
            strength_ids[index],
            "WeeToddH3ReferenceStrength",
            {
                "conditioning": Ref(encode_ids[index]),
                "visual_strength": 0.999,
                "audio_strength": 1.0,
            },
            pos=(2890, y + 30),
            size=(360, 210),
        )
        sample_values = {
            "components": Ref(4),
            "conditioning": Ref(strength_ids[index]),
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
            pos=(3300, y + 20),
            size=(410, 310),
        )
        append_values = {"timeline": Ref(5), "latents": Ref(sample_ids[index])}
        if index:
            append_values["previous_chain"] = Ref(append_ids[index - 1])
        workflow.add(
            append_ids[index],
            "WeeToddH3ChainAppend",
            append_values,
            pos=(3760, y + 20),
            size=(350, 230),
        )
        if index < 3:
            workflow.add(
                continuation_ids[index],
                "WeeToddH3ContinuationContext",
                {"latents": Ref(sample_ids[index]), "context_frames": "22"},
                pos=(3760, y + 290),
                size=(360, 190),
            )
    workflow.add(
        45,
        "WeeToddH3DirectPublishChain",
        {
            "components": Ref(4),
            "chain": Ref(44),
            "filename_prefix": "WeeTodd/H3_768p_15s_Ref2VA_Aliens_Chained_Staged_Turbo",
            "crf": 18,
            "max_av_drift_seconds": 0.025,
            "generation_metadata": json.dumps(
                {
                    "workflow": "h3_768p_15s_ref2va_aliens_chained_staged_turbo",
                    "seed": CHAIN_SEED,
                    "lora": LORA,
                    "references": [
                        "tall white character sheet",
                        "grey alien character sheet",
                        "matching source-video segment with soundtrack",
                    ],
                    "context_frames": 22,
                    "join_policy": (
                        "latent overlap + motion-matched 4-frame cosine blend + "
                        "50-ms audio crossfade"
                    ),
                }
            ),
            "ffmpeg_path": "",
        },
        pos=(3760, 2260),
        size=(520, 350),
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
    write("h3_768p_15s_ref2va_aliens_staged_turbo", build_alien_ref2va())
    write(
        "h3_768p_15s_ref2va_aliens_chained_staged_turbo",
        build_alien_ref2va_chain(),
    )
