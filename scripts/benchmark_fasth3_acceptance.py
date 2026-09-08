#!/usr/bin/env python3
"""Render the matched FastH3 optimization acceptance matrix on Apple Silicon."""

from __future__ import annotations

import argparse
import gc
import json
import time
from dataclasses import asdict
from pathlib import Path

import mlx.core as mx

from minimax_h3_mlx.fasth3_approx import FastH3ApproximationConfig
from wee_todd_nodes.direct_publishing import publish_latents_direct
from wee_todd_nodes.nodes import WeeToddH3SolAttention, WeeToddH3TextEncode
from wee_todd_nodes.preflight import H3ComponentSetSpec
from wee_todd_nodes.runtime import H3GenerationConfig
from wee_todd_nodes.sampling import TRANSFORMER_RUNTIME, H3TransformerSpec

DEFAULT_PROMPT = (
    "integrated_multimodal_description: [Shot 1] Live-action cinematic realism. A small "
    "red wind-up robot walks from left to right across a rain-wet workbench, turns toward the "
    "camera, raises one metal hand, and gives a clear friendly wave. The camera makes one slow "
    "lateral tracking move and settles without a cut. Warm workshop lamps reflect in small "
    "puddles and on the robot's painted shell.\n\n"
    "overall_soundscape: Soft rain taps a nearby window throughout. Each robot step makes a "
    "synchronized light metallic click, its winding key ticks steadily, and the raised hand "
    "makes two quiet servo whirs during the wave.\n\n"
    "non_diegetic_music: N/A"
)


VARIANTS = {
    "grouped_vsa": {
        "checkpoint": "vsa",
        "attention": "fasth3_vsa_90",
        "approximation": None,
    },
    "indexed_metal": {
        "checkpoint": "vsa",
        "attention": "fasth3_vsa_90_metal",
        "approximation": None,
    },
    "indexed_metal_fused_qkv": {
        "checkpoint": "vsa",
        "attention": "fasth3_vsa_90_metal_fused_qkv",
        "approximation": None,
    },
    "indexed_metal_layer40": {
        "checkpoint": "vsa",
        "attention": "fasth3_vsa_90_metal",
        "approximation": FastH3ApproximationConfig(active_layers=40),
    },
    "dense_control": {
        "checkpoint": "dense",
        "attention": None,
        "approximation": None,
    },
    "dense_token_pair": {
        "checkpoint": "dense",
        "attention": None,
        "approximation": FastH3ApproximationConfig(
            pair_target_video=True,
            pair_start_layer=4,
            pair_end_layer=30,
        ),
    },
}


def _components(model_root: Path, checkpoint: str) -> H3ComponentSetSpec:
    transformer = {
        "vsa": "weetodd-fasth3-vsa-datafree-q8-paged",
        "dense": "weetodd-fasth3-dense-q8-paged",
    }[checkpoint]
    return H3ComponentSetSpec(
        checkpoint=str(model_root / "FL2VA"),
        task="t2va",
        transformer=str(model_root / "transformers" / transformer),
        text_encoder=str(model_root / "text_encoders" / "q8-paged"),
        processor=str(model_root / "FL2VA" / "processor"),
        tokenizer=str(model_root / "FL2VA" / "tokenizer"),
        video_vae=str(model_root / "vae" / "q8" / "video_vae_affine_q8.safetensors"),
        audio_vae=str(model_root / "FL2VA" / "audio_vae" / "audio_vae.safetensors"),
    )


def _preflight(components: H3ComponentSetSpec, output: Path, ffmpeg: Path) -> None:
    if not ffmpeg.is_file():
        raise FileNotFoundError(f"ffmpeg not found: {ffmpeg}")
    if not (Path(components.checkpoint) / "model_index.json").is_file():
        raise FileNotFoundError("FastH3 acceptance model_index.json is missing")
    for name, path in components.resolved_paths().items():
        if not path.exists():
            raise FileNotFoundError(f"FastH3 acceptance {name} is missing: {path}")
    output.mkdir(parents=True, exist_ok=True)
    probe = output / ".write-probe"
    probe.write_text("ok\n", encoding="utf-8")
    probe.unlink()


def _sampling_report(latents) -> dict[str, object]:
    return {
        "transformer_evaluations": latents.transformer_evaluations,
        "seconds_per_evaluation": latents.seconds_per_evaluation,
        "total_seconds": latents.total_seconds,
        "video_shape": list(latents.video.shape),
        "audio_shape": list(latents.audio.shape),
        "paging": latents.paging_report,
        "attention": latents.sol_attention_report,
        "fastvideo": latents.fast_h3_approximation_report,
        "projection_backend": latents.projection_backend_report,
        "projection_runtime": latents.projection_backend_runtime,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--ffmpeg",
        type=Path,
        default=Path("/opt/homebrew/bin/ffmpeg"),
    )
    parser.add_argument("--variant", action="append", choices=tuple(VARIANTS))
    prompt_group = parser.add_mutually_exclusive_group()
    prompt_group.add_argument("--prompt")
    prompt_group.add_argument("--prompt-file", type=Path)
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument(
        "--purpose",
        default="FastH3 optimization 1-5 audiovisual acceptance",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    prompt = DEFAULT_PROMPT
    if args.prompt_file is not None:
        prompt = args.prompt_file.expanduser().resolve().read_text(encoding="utf-8").strip()
    elif args.prompt is not None:
        prompt = args.prompt.strip()
    if not prompt:
        raise ValueError("FastH3 acceptance prompt must not be empty")

    selected = tuple(args.variant or VARIANTS)
    output = args.output.expanduser().resolve()
    model_root = args.model_root.expanduser().resolve()
    checkpoint_kinds = {VARIANTS[name]["checkpoint"] for name in selected}
    component_sets = {kind: _components(model_root, kind) for kind in checkpoint_kinds}
    for components in component_sets.values():
        _preflight(components, output, args.ffmpeg.expanduser().resolve())
    print(
        json.dumps(
            {
                "status": "preflight_ok",
                "variants": list(selected),
                "output": str(output),
                "ffmpeg": str(args.ffmpeg),
                "seed": args.seed,
                "purpose": args.purpose,
                "prompt_file": str(args.prompt_file) if args.prompt_file else None,
            },
            indent=2,
        ),
        flush=True,
    )
    if args.dry_run:
        return 0

    config = H3GenerationConfig(
        duration_seconds=5.0,
        steps=5,
        seed=args.seed,
        width=640,
        height=384,
        drop_adaln=True,
        memory_mode="low_memory_bf16",
        projection_backend="mlx",
        sampling_method="euler",
    )
    conditioning_components = component_sets[VARIANTS[selected[0]]["checkpoint"]]
    conditioning = WeeToddH3TextEncode().encode(
        conditioning_components,
        prompt,
        True,
        config,
    )[0]
    (output / "scenario.json").write_text(
        json.dumps(
            {
                "purpose": args.purpose,
                "prompt": prompt,
                "seed": config.seed,
                "generation": asdict(config),
                "variants": list(selected),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    summary_path = output / "matrix.json"
    matrix = json.loads(summary_path.read_text()) if args.resume and summary_path.is_file() else {}

    for ordinal, name in enumerate(selected, start=1):
        definition = VARIANTS[name]
        variant_dir = output / name
        variant_dir.mkdir(parents=True, exist_ok=True)
        video_path = variant_dir / f"{name}.mp4"
        latent_path = variant_dir / f"{name}-latents.safetensors"
        if args.resume and video_path.is_file() and latent_path.is_file():
            print(f"[{ordinal}/{len(selected)}] {name}: verified files exist; skipping", flush=True)
            continue
        components = component_sets[definition["checkpoint"]]
        attention = None
        if definition["attention"] is not None:
            attention = WeeToddH3SolAttention().configure(
                definition["attention"],
                0.75,
                0.2,
                1.0,
                2,
                8192,
            )[0]
        print(f"[{ordinal}/{len(selected)}] {name}: sampling", flush=True)
        mx.reset_peak_memory()
        wall_started = time.perf_counter()
        latents = TRANSFORMER_RUNTIME.sample(
            H3TransformerSpec.from_components(components),
            conditioning,
            config,
            unload_after=True,
            sol_attention=attention,
            fastvideo=definition["approximation"],
        )
        mx.eval(latents.video, latents.audio)
        sample_wall = time.perf_counter() - wall_started
        sample_peak = int(mx.get_peak_memory())
        mx.save_safetensors(
            str(latent_path),
            {"video_latents": latents.video, "audio_latents": latents.audio},
        )
        sampling = _sampling_report(latents)
        sampling.update(
            {
                "sample_wall_seconds": sample_wall,
                "sample_peak_memory_bytes": sample_peak,
            }
        )
        print(f"[{ordinal}/{len(selected)}] {name}: decoding and publishing", flush=True)
        publication = publish_latents_direct(
            video_path,
            components,
            latents,
            crf=18,
            max_av_drift_seconds=0.025,
            generation_metadata=json.dumps(
                {
                    "purpose": args.purpose,
                    "variant": name,
                    "prompt": prompt,
                    "seed": config.seed,
                    "generation": asdict(config),
                    "sampling": sampling,
                }
            ),
            ffmpeg_path=args.ffmpeg,
        )
        matrix[name] = {
            "video": str(publication.video_path),
            "metadata": str(publication.metadata_path),
            "latents": str(latent_path),
            "sampling": sampling,
            "publication": publication.metadata,
        }
        summary_path.write_text(json.dumps(matrix, indent=2, sort_keys=True) + "\n")
        del latents
        gc.collect()
        mx.clear_cache()
        print(
            f"[{ordinal}/{len(selected)}] {name}: published {publication.video_path}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
