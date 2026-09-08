#!/usr/bin/env python3
"""Prepare the shared experimental H3 reference recipe for Studio and headless execution."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

COMPONENTS = {
    "checkpoint": "MiniMax-H3/Ref2VA",
    "task": "ref2va",
    "transformer": "MiniMax-H3/transformers/ref2va-q8-extended-paged",
    "text_encoder": "MiniMax-H3/text_encoders/q8-vision-paged",
    "processor": "MiniMax-H3/Ref2VA/processor",
    "tokenizer": "MiniMax-H3/Ref2VA/tokenizer",
    "video_vae": "MiniMax-H3/vae/q8/video_vae_affine_q8.safetensors",
    "audio_vae": "MiniMax-H3/FL2VA/audio_vae/audio_vae.safetensors",
    "allow_fl2va_weights_for_ref2va": False,
}
DEFAULT_PROMPT = (
    "integrated_multimodal_description: Live-action cinematic realism. Use <Picture 1> "
    "as the subject and appearance reference. Describe a clear action, camera motion, "
    "lighting, and stable final state.\n\n"
    "overall_soundscape: Describe synchronized environmental sound and movement impacts."
    "\n\nnon_diegetic_music: N/A"
)

PUBLICATION_METADATA = (
    '{"workflow": "h3_ref2va_q8_paged", "status": "experimental", "evaluations": 19}'
)


def build_recipe(models: Path, references: list[Path], prompt: str | None = None) -> dict:
    """Resolve the shipped Comfy profile into an equivalent host-independent recipe."""
    from dataclasses import asdict

    from wee_todd_nodes.runtime import H3GenerationConfig

    components = dict(COMPONENTS)
    for key in (
        "checkpoint",
        "transformer",
        "text_encoder",
        "processor",
        "tokenizer",
        "video_vae",
        "audio_vae",
    ):
        components[key] = str((models.expanduser().resolve() / components[key]).resolve())
    config = H3GenerationConfig(
        duration_seconds=5.0,
        steps=20,
        seed=20260908,
        width=640,
        height=384,
        resolution_mode="exact dimensions",
        memory_mode="low_memory_bf16",
        drop_adaln=True,
        projection_backend="mlx",
        sampling_method="euler",
        inference_optimization="off",
    )
    return {
        "format": "weetodd-headless-v2",
        "candidate": "h3-reference-q8-paged",
        "engine": "h3",
        "ffmpeg": "ffmpeg",
        "components": components,
        "config": asdict(config),
        "prompt": prompt or DEFAULT_PROMPT,
        "reference_images": [str(p.expanduser().resolve()) for p in references],
        "publication": {
            "crf": 18,
            "max_av_drift_seconds": 0.025,
            "generation_metadata": PUBLICATION_METADATA,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        type=Path,
        required=True,
        help="ComfyUI models folder or equivalent shared layout",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        action="append",
        required=True,
        help="Reference image; repeat for more images",
    )
    parser.add_argument("--prompt-file", type=Path)
    parser.add_argument("--ffmpeg", help="Optional FFmpeg executable override")
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New JSON file to import in Studio Runtime Settings",
    )
    args = parser.parse_args()
    from wee_todd_mlx.headless_preflight import preflight_recipe

    recipe = build_recipe(
        args.models, args.reference, args.prompt_file.read_text() if args.prompt_file else None
    )
    from minimax_h3_mlx.media import resolve_ffmpeg

    recipe["ffmpeg"] = str(resolve_ffmpeg(args.ffmpeg).path)
    report = preflight_recipe(recipe)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(recipe, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"recipe": str(args.output.resolve()), "preflight": report}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
