#!/usr/bin/env python3
"""Render a resumable FastH3 resolution scaling matrix on Apple Silicon."""

from __future__ import annotations

import argparse
import gc
import json
import platform
import shutil
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

import mlx.core as mx

from minimax_h3_mlx.packing import align_num_frames
from wee_todd_nodes.direct_publishing import publish_latents_direct
from wee_todd_nodes.nodes import WeeToddH3FastH3ProductionProfile, WeeToddH3TextEncode
from wee_todd_nodes.preflight import H3ComponentSetSpec
from wee_todd_nodes.runtime import H3GenerationConfig
from wee_todd_nodes.sampling import TRANSFORMER_RUNTIME, H3TransformerSpec

RESOLUTIONS = (
    (512, 256),
    (768, 448),
    (1024, 576),
    (1280, 704),
    (1536, 832),
    (1920, 1088),
)
PROFILE = "Balanced — compact indexed Metal (recommended)"
TRANSFORMER = "weetodd-fasth3-vsa-datafree-q8-paged"
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


def _hardware_report() -> dict[str, object]:
    report: dict[str, object] = {"architecture": platform.machine()}
    try:
        result = subprocess.run(
            ["system_profiler", "SPHardwareDataType", "-json"],
            check=True,
            capture_output=True,
            text=True,
        )
        hardware = json.loads(result.stdout)["SPHardwareDataType"][0]
        report.update(
            {
                "machine_name": hardware.get("machine_name"),
                "machine_model": hardware.get("machine_model"),
                "chip": hardware.get("chip_type"),
                "cpu_topology": hardware.get("number_processors"),
                "memory": hardware.get("physical_memory"),
            }
        )
    except (KeyError, OSError, subprocess.CalledProcessError, json.JSONDecodeError):
        report["discovery"] = "system_profiler unavailable"
    return report


def _key(width: int, height: int) -> str:
    return f"{width}x{height}"


def _components(model_root: Path) -> H3ComponentSetSpec:
    return H3ComponentSetSpec(
        checkpoint=str(model_root / "FL2VA"),
        task="t2va",
        transformer=str(model_root / "transformers" / TRANSFORMER),
        text_encoder=str(model_root / "text_encoders" / "q8-paged"),
        processor=str(model_root / "FL2VA" / "processor"),
        tokenizer=str(model_root / "FL2VA" / "tokenizer"),
        video_vae=str(model_root / "vae" / "q8" / "video_vae_affine_q8.safetensors"),
        audio_vae=str(model_root / "FL2VA" / "audio_vae" / "audio_vae.safetensors"),
    )


def _preflight(
    components: H3ComponentSetSpec,
    output: Path,
    ffmpeg: Path,
    selected: tuple[tuple[int, int], ...],
) -> dict[str, object]:
    if not ffmpeg.is_file():
        raise FileNotFoundError(f"ffmpeg not found: {ffmpeg}")
    if not (Path(components.checkpoint) / "model_index.json").is_file():
        raise FileNotFoundError("MiniMax H3 model_index.json is missing")
    missing = {
        name: str(path) for name, path in components.resolved_paths().items() if not path.exists()
    }
    if missing:
        raise FileNotFoundError(f"FastH3 scaling components are missing: {missing}")
    for width, height in selected:
        H3GenerationConfig(
            duration_seconds=4.0,
            steps=5,
            width=width,
            height=height,
        ).validate()
    output.mkdir(parents=True, exist_ok=True)
    probe = output / ".write-probe"
    probe.write_text("ok\n", encoding="utf-8")
    probe.unlink()
    free_bytes = shutil.disk_usage(output).free
    if free_bytes < 100 * 1024**3:
        raise OSError(
            f"FastH3 scaling requires at least 100 GiB free for safe checkpoints; "
            f"found {free_bytes / 1024**3:.1f} GiB."
        )
    return {
        "status": "preflight_ok",
        "model_root": str(Path(components.checkpoint).parent),
        "transformer": Path(components.resolved_paths()["transformer"]).name,
        "output": str(output),
        "ffmpeg": str(ffmpeg),
        "free_disk_gib": free_bytes / 1024**3,
        "resolutions": [_key(*resolution) for resolution in selected],
    }


def _sampling_report(latents) -> dict[str, object]:
    return {
        "transformer_evaluations": latents.transformer_evaluations,
        "seconds_per_evaluation": latents.seconds_per_evaluation,
        "transformer_seconds": latents.total_seconds,
        "video_latent_shape": list(latents.video.shape),
        "audio_latent_shape": list(latents.audio.shape),
        "paging": latents.paging_report,
        "attention": latents.sol_attention_report,
        "fastvideo": latents.fast_h3_approximation_report,
        "projection_backend": latents.projection_backend_report,
        "projection_runtime": latents.projection_backend_runtime,
    }


def _read_prompt(args) -> str:
    if args.prompt_file is not None:
        prompt = args.prompt_file.expanduser().resolve().read_text(encoding="utf-8").strip()
    elif args.prompt is not None:
        prompt = args.prompt.strip()
    else:
        prompt = DEFAULT_PROMPT
    if not prompt:
        raise ValueError("FastH3 scaling prompt must not be empty")
    return prompt


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
    parser.add_argument("--ffmpeg", type=Path, default=Path("/opt/homebrew/bin/ffmpeg"))
    parser.add_argument(
        "--resolution",
        action="append",
        choices=tuple(_key(*resolution) for resolution in RESOLUTIONS),
        help="Render only selected rows; repeat the option to select multiple rows.",
    )
    prompt_group = parser.add_mutually_exclusive_group()
    prompt_group.add_argument("--prompt")
    prompt_group.add_argument("--prompt-file", type=Path)
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    selected_keys = set(args.resolution or ())
    selected = tuple(
        resolution
        for resolution in RESOLUTIONS
        if not selected_keys or _key(*resolution) in selected_keys
    )
    output = args.output.expanduser().resolve()
    model_root = args.model_root.expanduser().resolve()
    ffmpeg = args.ffmpeg.expanduser().resolve()
    components = _components(model_root)
    prompt = _read_prompt(args)
    preflight = _preflight(components, output, ffmpeg, selected)

    requested_frames = round(4.0 * 24)
    aligned_frames = align_num_frames(requested_frames)
    preflight.update(
        {
            "requested_duration_seconds": 4.0,
            "requested_frames_at_24fps": requested_frames,
            "aligned_frames_17n_plus_5": aligned_frames,
            "delivered_duration_seconds": aligned_frames / 24,
            "seed": args.seed,
            "profile": PROFILE,
            "parameter_scope": "~35.05B FastH3 DiT; Q8 core with BF16 VSA gates",
            "hardware": _hardware_report(),
        }
    )
    print(json.dumps(preflight, indent=2), flush=True)
    if args.dry_run:
        return 0

    scenario_path = output / "scenario.json"
    scenario_path.write_text(
        json.dumps(
            {
                **preflight,
                "prompt": prompt,
                "resolution_order": [_key(*resolution) for resolution in RESOLUTIONS],
                "timing_scope": (
                    "Per-row complete wall includes transformer sampling plus direct video/audio "
                    "decode and mux. Shared text conditioning is reported separately."
                ),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    matrix_path = output / "matrix.json"
    matrix = json.loads(matrix_path.read_text()) if args.resume and matrix_path.is_file() else {}

    conditioning_config = H3GenerationConfig(
        duration_seconds=4.0,
        steps=5,
        seed=args.seed,
        width=selected[0][0],
        height=selected[0][1],
        drop_adaln=True,
        memory_mode="low_memory_bf16",
        projection_backend="mlx",
        sampling_method="euler",
    )
    print("Encoding the shared prompt", flush=True)
    conditioning_started = time.perf_counter()
    conditioning = WeeToddH3TextEncode().encode(
        components,
        prompt,
        True,
        conditioning_config,
    )[0]
    conditioning_seconds = time.perf_counter() - conditioning_started
    print(f"Shared prompt encoded in {conditioning_seconds:.3f}s", flush=True)

    for ordinal, (width, height) in enumerate(selected, start=1):
        name = _key(width, height)
        row_dir = output / name
        row_dir.mkdir(parents=True, exist_ok=True)
        video_path = row_dir / f"fasth3-{name}.mp4"
        latent_path = row_dir / f"fasth3-{name}-latents.safetensors"
        existing = matrix.get(name)
        if args.resume and existing is not None and video_path.is_file() and latent_path.is_file():
            print(f"[{ordinal}/{len(selected)}] {name}: complete row exists; skipping", flush=True)
            continue

        config = H3GenerationConfig(
            duration_seconds=4.0,
            steps=5,
            seed=args.seed,
            width=width,
            height=height,
            drop_adaln=True,
            memory_mode="low_memory_bf16",
            projection_backend="mlx",
            sampling_method="euler",
        )
        _, configured, attention, profile_raw, _ = WeeToddH3FastH3ProductionProfile().apply(
            components,
            config,
            PROFILE,
            resolution_preset=WeeToddH3FastH3ProductionProfile._KEEP_RESOLUTION,
            min_tokens=4096,
        )
        print(f"[{ordinal}/{len(selected)}] {name}: sampling {aligned_frames} frames", flush=True)
        mx.reset_peak_memory()
        complete_started = time.perf_counter()
        sampling_started = time.perf_counter()
        latents = TRANSFORMER_RUNTIME.sample(
            H3TransformerSpec.from_components(components),
            conditioning,
            configured,
            unload_after=True,
            sol_attention=attention,
        )
        mx.eval(latents.video, latents.audio)
        sampling_wall = time.perf_counter() - sampling_started
        sampling_peak = int(mx.get_peak_memory())
        mx.save_safetensors(
            str(latent_path),
            {"video_latents": latents.video, "audio_latents": latents.audio},
        )
        sampling = _sampling_report(latents)
        sampling.update(
            {
                "sampling_wall_seconds": sampling_wall,
                "sampling_peak_memory_bytes": sampling_peak,
            }
        )
        print(f"[{ordinal}/{len(selected)}] {name}: direct decode and publish", flush=True)
        publication_started = time.perf_counter()
        publication = publish_latents_direct(
            video_path,
            components,
            latents,
            crf=18,
            max_av_drift_seconds=0.025,
            generation_metadata=json.dumps(
                {
                    "workflow": "fasth3_scaling_matrix",
                    "parameter_scope": "~35.05B FastH3 DiT; Q8 core with BF16 VSA gates",
                    "resolution": [width, height],
                    "megapixels": width * height / 1_000_000,
                    "prompt": prompt,
                    "seed": configured.seed,
                    "generation": asdict(configured),
                    "frame_alignment": {
                        "requested": requested_frames,
                        "aligned": aligned_frames,
                        "rule": "17n+5",
                    },
                    "profile": json.loads(profile_raw),
                    "sampling": sampling,
                }
            ),
            ffmpeg_path=ffmpeg,
        )
        publication_seconds = time.perf_counter() - publication_started
        complete_wall = time.perf_counter() - complete_started
        complete_peak = int(mx.get_peak_memory())
        matrix[name] = {
            "ordinal": RESOLUTIONS.index((width, height)) + 1,
            "width": width,
            "height": height,
            "pixels": width * height,
            "megapixels": width * height / 1_000_000,
            "requested_duration_seconds": configured.duration_seconds,
            "requested_frames_at_24fps": requested_frames,
            "aligned_frames": aligned_frames,
            "delivered_duration_seconds": aligned_frames / 24,
            "conditioning_seconds_shared": conditioning_seconds,
            "sampling": sampling,
            "publication_seconds": publication_seconds,
            "complete_wall_seconds": complete_wall,
            "complete_peak_memory_bytes": complete_peak,
            "profile": json.loads(profile_raw),
            "video": str(publication.video_path),
            "metadata": str(publication.metadata_path),
            "latents": str(latent_path),
            "publication": publication.metadata,
        }
        matrix_path.write_text(json.dumps(matrix, indent=2, sort_keys=True) + "\n")
        del latents
        gc.collect()
        mx.clear_cache()
        print(
            f"[{ordinal}/{len(selected)}] {name}: {complete_wall:.3f}s, "
            f"{complete_peak / 1_000_000_000:.3f} GB peak, published {publication.video_path}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
