#!/usr/bin/env python3
# ruff: noqa: E501
"""Build the two portable four-window H3 continuation workflows."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
OUTPUT = ROOT / "workflows"
SEED = 54_420_260_810
PROMPTS = (
    """integrated_multimodal_description: [Shot 1] Live-action mockumentary crossover parody with realistic skin, clothing, soil, and natural handheld camera movement. In a barren desert excavation site under harsh afternoon sunlight, Jesse, a lean young white man with a shaved head wearing a dusty dark hoodie and work gloves, stands inside a deep grave-sized hole. He angrily throws several heavy shovelfuls of dirt over the edge. Each shovel strikes compact soil, dirt scatters realistically, and his breathing grows heavier. Dwight, a rigid middle-aged white man with neatly parted brown hair, wire-frame glasses, and a mustard-yellow short-sleeve shirt, suddenly walks into frame above the hole. Dwight studies the hole suspiciously, adjusts his glasses, looks down with a smug expression, and says <d>[English] Hello, little man. Digging your own grave? </d> The handheld camera makes small documentary-style corrections and ends with Jesse frozen mid-dig below Dwight.

overall_soundscape: Dry desert wind, heavy breathing, shovel impacts against compact soil, loose dirt scattering, footsteps approaching on gravel, a faint clothing rustle as Dwight adjusts his glasses, and clean location-recorded dialogue synchronized to Dwight's mouth. No crowd and no machinery.

non_diegetic_music: N/A""",
    """integrated_multimodal_description: [Shot 1] Continue the same unbroken live-action mockumentary crossover shot at the same desert excavation site. Preserve Jesse's shaved head, dusty dark hoodie, gloves, shovel, position inside the deep hole, and Dwight's mustard-yellow shirt, brown parted hair, glasses, and position at the rim. Jesse immediately stops digging. He slowly turns his head and shoulders toward Dwight, plants his boots, and tightens both hands around the shovel handle. His face becomes cold and threatening. Looking directly up at Dwight, Jesse says <d>[English] No, no, no... I'm digging it for you. Just like I did for Michael. </d> Dwight's smug expression begins to fail as the sentence ends. The handheld camera subtly pushes closer without a cut and ends on both men holding tense eye contact.

overall_soundscape: Continue the same dry wind and quiet desert ambience without an audible reset. The shovel movement stops with one final scrape, loose soil settles, Jesse's heavy breathing slows, his gloves tighten audibly against the wooden handle, and his clean threatening dialogue remains synchronized to his mouth.

non_diegetic_music: N/A""",
    """integrated_multimodal_description: [Shot 1] Continue the same unbroken realistic handheld shot with the same Jesse and Dwight, unchanged clothing, excavation site, shovel, grave, sunlight, and screen positions. Dwight's confidence collapses. His eyes widen behind his glasses, his face fills with horror, and he begins crying and panicking dramatically. Dwight drops to his knees at the edge of the hole, reaches down, grabs Jesse by both shoulders, shakes him once, and screams <d>[English] No! What did you do to Michael?! Where is he? Tell me now! </d> Jesse remains completely emotionless and keeps one hand firmly around the shovel. The camera jolts with Dwight's movement, then pushes toward his terrified face while retaining Jesse in the background.

overall_soundscape: Continue the same desert wind and roomless outdoor ambience. Dwight's knees hit loose soil, fabric pulls as he grabs Jesse, one brief shoulder shake rustles both men's clothing, Dwight cries and screams with synchronized mouth movement, and Jesse remains silent. The shovel gives one small wooden creak.

non_diegetic_music: N/A""",
    """integrated_multimodal_description: [Shot 1] Continue the same unbroken live-action mockumentary shot at the same desert grave. Preserve both men's faces, clothing, positions, lighting, and the shovel. The handheld camera completes its push into a tight close-up of Dwight's terrified face. Tears run down his cheeks, his eyes remain wide behind his glasses, and his breathing becomes rapid and uneven. Dwight looks from Jesse to the grave and back again. Jesse remains silent and completely emotionless in the shallow background, staring at Dwight while holding the shovel upright. Hold the dreadful silence for a beat, then end abruptly on Dwight's horrified face without a fade.

overall_soundscape: Continue the same wind and location ambience without a reset. Dwight's panicked breathing and quiet sobbing dominate the close-up. Jesse makes no sound. A few grains of dirt fall into the hole, followed by an abrupt hard ending with no musical sting.

non_diegetic_music: N/A""",
)


class Workflow:
    def __init__(self) -> None:
        self.nodes: dict[int, dict] = {}
        self.links: list[list] = []
        self._next_link = 1

    def add(
        self,
        node_id: int,
        node_type: str,
        pos: tuple[int, int],
        size: tuple[int, int],
        order: int,
        outputs: list[tuple[str, str]],
        widgets: list | None = None,
    ) -> None:
        self.nodes[node_id] = {
            "id": node_id,
            "type": node_type,
            "pos": list(pos),
            "size": list(size),
            "flags": {},
            "order": order,
            "mode": 0,
            "inputs": [],
            "outputs": [
                {"name": name, "type": value_type, "links": None, "slot_index": index}
                for index, (name, value_type) in enumerate(outputs)
            ],
            "properties": {"Node name for S&R": node_type},
            "widgets_values": widgets or [],
        }

    def connect(
        self,
        source: int,
        source_slot: int,
        target: int,
        input_name: str,
        value_type: str,
    ) -> None:
        link_id = self._next_link
        self._next_link += 1
        link = [link_id, source, source_slot, target, len(self.nodes[target]["inputs"]), value_type]
        self.links.append(link)
        output = self.nodes[source]["outputs"][source_slot]
        output["links"] = [*(output["links"] or []), link_id]
        self.nodes[target]["inputs"].append(
            {"name": input_name, "type": value_type, "link": link_id}
        )

    def payload(self) -> dict:
        return {
            "last_node_id": max(self.nodes),
            "last_link_id": self._next_link - 1,
            "nodes": [self.nodes[node_id] for node_id in sorted(self.nodes)],
            "links": self.links,
            "groups": [
                {
                    "title": "Four-window latent-native H3 continuation",
                    "bounding": [20, 20, 4240, 2110],
                    "color": "#3f789e",
                    "font_size": 24,
                    "flags": {},
                }
            ],
            "config": {},
            "extra": {"ds": {"scale": 0.42, "offset": [35, 60]}},
            "version": 0.4,
        }


def build(filename: str, preset: str, label: str) -> None:
    workflow = Workflow()
    workflow.add(
        1,
        "WeeToddH3ComponentLoader",
        (40, 80),
        (360, 330),
        0,
        [("components", "WEETODD_H3_COMPONENTS")],
        [
            "MiniMax-H3/FL2VA",
            "t2va",
            "MiniMax-H3/transformers/q8_extended_paged",
            "MiniMax-H3/text_encoders/q8-paged",
            "MiniMax-H3/FL2VA/processor",
            "MiniMax-H3/FL2VA/tokenizer",
            "MiniMax-H3/vae/q8/video_vae_affine_q8.safetensors",
            "MiniMax-H3/FL2VA/audio_vae",
            False,
        ],
    )
    workflow.add(
        2,
        "WeeToddH3GenerationConfig",
        (40, 470),
        (380, 330),
        1,
        [("config", "WEETODD_H3_CONFIG"), ("resolved_resolution", "STRING")],
        [
            5.17,
            20,
            SEED,
            "ratio + size",
            "Use size slider — 32 px steps",
            "16:9 — widescreen landscape",
            544,
            960,
            544,
            False,
            "normal",
            "automatic",
            "mlx",
            "euler",
        ],
    )
    workflow.add(
        3,
        "WeeToddH3ValidatedSamplingPreset",
        (470, 470),
        (460, 230),
        2,
        [
            ("config", "WEETODD_H3_CONFIG"),
            ("loras", "WEETODD_H3_LORAS"),
            ("trajectory_forecast", "WEETODD_H3_TRAJECTORY_FORECAST"),
            ("preset_info", "STRING"),
        ],
        [preset],
    )
    workflow.add(
        4,
        "WeeToddH3Preflight",
        (970, 80),
        (350, 300),
        3,
        [("components", "WEETODD_H3_COMPONENTS"), ("preflight_report", "STRING")],
        [512, 0.0, ""],
    )
    workflow.connect(1, 0, 4, "components", "WEETODD_H3_COMPONENTS")
    workflow.connect(2, 0, 3, "config", "WEETODD_H3_CONFIG")
    workflow.connect(3, 0, 4, "config", "WEETODD_H3_CONFIG")

    text_ids = (5, 6, 7, 8)
    sample_ids = (9, 10, 11, 12)
    context_ids = (13, 14, 15)
    video_ids = (16, 17, 18, 19)
    audio_ids = (20, 21, 22, 23)
    trim_ids = (24, 25, 26)
    publish_ids = (27, 28, 29, 30)

    for index, (text_id, sample_id) in enumerate(zip(text_ids, sample_ids, strict=True)):
        y = 80 + index * 490
        workflow.add(
            text_id,
            "WeeToddH3TextEncode",
            (1380, y),
            (520, 360),
            4 + index * 2,
            [("conditioning", "WEETODD_H3_CONDITIONING"), ("conditioning_info", "STRING")],
            [PROMPTS[index], True],
        )
        workflow.add(
            sample_id,
            "WeeToddH3Sample",
            (1960, y + 40),
            (410, 330),
            5 + index * 2,
            [("latents", "WEETODD_H3_LATENTS"), ("sampling_info", "STRING")],
            [index == 3],
        )
        workflow.connect(4, 0, text_id, "components", "WEETODD_H3_COMPONENTS")
        workflow.connect(3, 0, text_id, "config", "WEETODD_H3_CONFIG")
        workflow.connect(4, 0, sample_id, "components", "WEETODD_H3_COMPONENTS")
        workflow.connect(text_id, 0, sample_id, "conditioning", "WEETODD_H3_CONDITIONING")
        workflow.connect(3, 0, sample_id, "config", "WEETODD_H3_CONFIG")
        workflow.connect(
            3,
            2,
            sample_id,
            "trajectory_forecast",
            "WEETODD_H3_TRAJECTORY_FORECAST",
        )
        if index == 0:
            workflow.connect(3, 1, sample_id, "loras", "WEETODD_H3_LORAS")
    for index, context_id in enumerate(context_ids):
        workflow.add(
            context_id,
            "WeeToddH3ContinuationContext",
            (2430, 190 + index * 490),
            (330, 190),
            12 + index,
            [("continuation", "WEETODD_H3_CONTINUATION"), ("continuation_info", "STRING")],
            ["22"],
        )
        workflow.connect(sample_ids[index], 0, context_id, "latents", "WEETODD_H3_LATENTS")
        workflow.connect(
            context_id,
            0,
            sample_ids[index + 1],
            "continuation",
            "WEETODD_H3_CONTINUATION",
        )
        workflow.connect(
            3,
            1,
            sample_ids[index + 1],
            "loras",
            "WEETODD_H3_LORAS",
        )

    for index, (video_id, audio_id, publish_id) in enumerate(
        zip(video_ids, audio_ids, publish_ids, strict=True)
    ):
        y = 80 + index * 490
        workflow.add(
            video_id,
            "WeeToddH3VideoVAEDecode",
            (2820, y),
            (330, 190),
            15 + index * 4,
            [("frames", "IMAGE"), ("decode_info", "STRING")],
            [True],
        )
        workflow.add(
            audio_id,
            "WeeToddH3AudioVAEDecode",
            (2820, y + 220),
            (330, 190),
            16 + index * 4,
            [("audio", "AUDIO"), ("decode_info", "STRING")],
            [True],
        )
        workflow.add(
            publish_id,
            "WeeToddH3PublishVideoAudio",
            (3780, y + 70),
            (420, 340),
            18 + index * 4,
            [("video_path", "STRING"), ("generation_info", "STRING")],
            [
                f"WeeTodd/H3_544p_{label}_Chained_Clip{index + 1}",
                18,
                0.025,
                json.dumps(
                    {
                        "workflow": filename.removesuffix(".json"),
                        "preset": preset,
                        "clip": index + 1,
                        "context_frames": 0 if index == 0 else 22,
                    }
                ),
                "",
                *([""] if index == 0 else []),
            ],
        )
        workflow.connect(4, 0, video_id, "components", "WEETODD_H3_COMPONENTS")
        workflow.connect(sample_ids[index], 0, video_id, "latents", "WEETODD_H3_LATENTS")
        workflow.connect(4, 0, audio_id, "components", "WEETODD_H3_COMPONENTS")
        workflow.connect(sample_ids[index], 0, audio_id, "latents", "WEETODD_H3_LATENTS")

        if index == 0:
            workflow.connect(video_id, 0, publish_id, "images", "IMAGE")
            workflow.connect(audio_id, 0, publish_id, "audio", "AUDIO")
        else:
            trim_id = trim_ids[index - 1]
            workflow.add(
                trim_id,
                "WeeToddH3TrimContinuation",
                (3240, y + 90),
                (360, 250),
                17 + index * 4,
                [("images", "IMAGE"), ("audio", "AUDIO"), ("trim_info", "STRING")],
            )
            workflow.connect(video_id, 0, trim_id, "images", "IMAGE")
            workflow.connect(audio_id, 0, trim_id, "audio", "AUDIO")
            workflow.connect(
                context_ids[index - 1],
                0,
                trim_id,
                "continuation",
                "WEETODD_H3_CONTINUATION",
            )
            workflow.connect(trim_id, 0, publish_id, "images", "IMAGE")
            workflow.connect(trim_id, 1, publish_id, "audio", "AUDIO")

        workflow.connect(4, 0, publish_id, "components", "WEETODD_H3_COMPONENTS")
        workflow.connect(3, 0, publish_id, "config", "WEETODD_H3_CONFIG")
        workflow.connect(sample_ids[index], 1, publish_id, "sampling_info", "STRING")
        if index > 0:
            workflow.connect(
                trim_ids[index - 1],
                2,
                publish_id,
                "media_timing_info",
                "STRING",
            )

    target = OUTPUT / filename
    target.write_text(json.dumps(workflow.payload(), indent=2) + "\n", encoding="utf-8")


def main() -> None:
    build(
        "h3_544p_chained_dense_turbo_rank21.json",
        "Chained context — Dense Turbo LightX2V rank 21 — 5 points / 4 evaluations",
        "Dense_Turbo_Rank21",
    )
    build(
        "h3_544p_chained_trajectory_replay.json",
        (
            "Chained context — Trajectory target-only replay — "
            "20 points / up to 11 evaluations"
        ),
        "Trajectory_Replay",
    )


if __name__ == "__main__":
    main()
