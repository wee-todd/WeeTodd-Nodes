#!/usr/bin/env python3
"""Decode saved synchronized H3 latents directly into an MP4 file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlx.core as mx

from wee_todd_nodes.direct_publishing import publish_latents_direct
from wee_todd_nodes.preflight import H3ComponentSetSpec
from wee_todd_nodes.runtime import H3GenerationConfig
from wee_todd_nodes.sampling import H3Latents, H3TransformerSpec


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--latents", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--transformer", type=Path, required=True)
    parser.add_argument("--video-vae", type=Path, required=True)
    parser.add_argument("--audio-vae", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--frames", type=int, required=True)
    parser.add_argument("--crf", type=int, default=18)
    args = parser.parse_args()

    stored = mx.load(str(args.latents))
    components = H3ComponentSetSpec(
        checkpoint=str(args.checkpoint),
        transformer=str(args.transformer),
        video_vae=str(args.video_vae),
        audio_vae=str(args.audio_vae),
    )
    transformer_spec = H3TransformerSpec.from_components(components)
    latents = H3Latents(
        video=stored["video_latents"],
        audio=stored["audio_latents"],
        num_frames=args.frames,
        width=args.width,
        height=args.height,
        fps=24,
        sample_rate=32000,
        transformer_evaluations=0,
        seconds_per_evaluation=0.0,
        total_seconds=0.0,
        transformer_spec=transformer_spec,
        generation_config=H3GenerationConfig(
            duration_seconds=args.frames / 24,
            steps=2,
            width=args.width,
            height=args.height,
            memory_mode="low_memory_bf16",
        ),
    )
    mx.reset_peak_memory()
    result = publish_latents_direct(
        args.output,
        components,
        latents,
        crf=args.crf,
        max_av_drift_seconds=0.025,
        generation_metadata=json.dumps(
            {
                "source_latents": args.latents.name,
                "purpose": "direct_mlx_publication_validation",
            }
        ),
    )
    print(
        json.dumps(
            {
                "video": str(result.video_path),
                "metadata": str(result.metadata_path),
                "peak_memory_bytes": int(mx.get_peak_memory()),
                **result.metadata,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
