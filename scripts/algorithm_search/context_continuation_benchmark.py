#!/usr/bin/env python3
"""Benchmark matched dense and latent-context H3 transformer evaluations."""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from safetensors import safe_open

from minimax_h3_mlx.algorithm_search.capture import CaptureConfig, DiagnosticSession
from minimax_h3_mlx.config import PipelineConfig
from minimax_h3_mlx.continuation_cost import estimate_continuation_cost
from minimax_h3_mlx.load import load_dit
from minimax_h3_mlx.packing import (
    FPS,
    align_num_frames,
    audio_latent_num_frames,
    build_packed_sequence,
    video_latent_num_frames,
)
from minimax_h3_mlx.pipeline import MiniMaxH3Pipeline
from minimax_h3_mlx.text_encoder import MiniMaxH3TextEncoder


def _checkpoint_identity(path: Path) -> dict:
    if path.is_file():
        with safe_open(str(path), framework="np") as handle:
            metadata = handle.metadata() or {}
        return {
            "name": path.name,
            "precision": metadata.get("precision"),
            "partition": metadata.get("partition"),
            "source_revision": metadata.get("source_revision"),
            "adaln_curve_grid": metadata.get("adaln_curve_grid"),
            "adaln_curve_rank": metadata.get("adaln_curve_rank"),
        }
    manifest = path / "paged_manifest.json"
    if manifest.is_file():
        raw = json.loads(manifest.read_text())
        return {
            "name": path.name,
            "format": raw.get("format"),
            "source": Path(str(raw.get("source", ""))).name or None,
        }
    return {"name": path.name}


def _context_latents(dit, width: int, height: int, context_frames: int):
    if context_frames == 0:
        return None, None
    latent_height, latent_width = height // 16, width // 16
    video = mx.random.normal(
        (
            1,
            dit.config.latents_dim,
            video_latent_num_frames(context_frames),
            latent_height,
            latent_width,
        ),
        key=mx.random.key(10_000 + context_frames),
    ).astype(mx.float32)
    audio = mx.random.normal(
        (
            2,
            dit.config.audio_latents_dim,
            audio_latent_num_frames(context_frames),
        ),
        key=mx.random.key(20_000 + context_frames),
    ).astype(mx.float32)
    mx.eval(video, audio)
    return video, audio


def _packing_median_seconds(
    *,
    tags: np.ndarray,
    width: int,
    height: int,
    num_frames: int,
    context_frames: int,
    patch_size: tuple[int, int, int],
    repeats: int = 25,
) -> float:
    values = []
    for _ in range(repeats):
        started = time.perf_counter()
        layout = build_packed_sequence(
            tags,
            video_latent_num_frames(num_frames),
            height // 16,
            width // 16,
            audio_latent_num_frames(num_frames),
            patch_size,
            continuation_video_frames=(
                video_latent_num_frames(context_frames) if context_frames else 0
            ),
            continuation_audio_latents=(
                audio_latent_num_frames(context_frames) if context_frames else 0
            ),
        )
        mx.eval(layout.position_ids, layout.token_tags)
        values.append(time.perf_counter() - started)
    return statistics.median(values)


def _region_group(name: str) -> str:
    if name.endswith(".sdpa"):
        return "attention_sdpa"
    if ".attn.qkv_proj" in name or ".attn.out_proj" in name:
        return "attention_projections"
    if ".attn.qk_norm" in name or ".attn.rotary" in name:
        return "attention_preparation"
    if ".mlp." in name:
        return "mlp"
    if name.startswith("input."):
        return "input_and_rope"
    if name.startswith("output."):
        return "output_heads"
    if name.startswith("scheduler."):
        return "scheduler"
    if name.startswith("packing."):
        return "packing"
    if ".norm" in name or name.endswith("_residual"):
        return "normalization_and_residual"
    return "other"


def _region_summary(session: DiagnosticSession) -> dict[str, float]:
    grouped: dict[str, float] = {}
    for item in session.measurements:
        group = _region_group(item.name)
        grouped[group] = grouped.get(group, 0.0) + item.duration_seconds
    return dict(sorted(grouped.items()))


def _run_case(
    *,
    pipeline,
    embeddings,
    tags,
    width: int,
    height: int,
    duration: float,
    steps: int,
    seed: int,
    context_frames: int,
    repeats: int,
    output: Path,
    profile_blocks: tuple[int, ...],
    profile: bool,
    terminal_target_only: bool,
) -> tuple[dict, object, object]:
    video_context, audio_context = _context_latents(
        pipeline.dit, width, height, context_frames
    )
    timings = []
    peaks = []
    active = []
    result = None
    for _ in range(repeats):
        mx.reset_peak_memory()
        active.append(int(mx.get_active_memory()))
        result = pipeline.sample_latents(
            embeddings,
            tags,
            duration_seconds=duration,
            num_inference_steps=steps,
            seed=seed,
            height=height,
            width=width,
            drop_adaln=False,
            verbose=False,
            continuation_video_latents=video_context,
            continuation_audio_latents=audio_context,
            continuation_frames=context_frames,
            terminal_target_only=terminal_target_only,
        )
        mx.eval(result.video_latents, result.audio_latents)
        timings.append(float(result.seconds_per_evaluation))
        peaks.append(int(mx.get_peak_memory()))

    profile_summary = None
    if profile:
        profile_output = output / f"profile_context_{context_frames}"
        session = DiagnosticSession(
            CaptureConfig(
                enabled=False,
                output_directory=str(profile_output),
                profile_blocks=profile_blocks,
                profile_regions=True,
            )
        )
        pipeline.sample_latents(
            embeddings,
            tags,
            duration_seconds=duration,
            num_inference_steps=steps,
            seed=seed,
            height=height,
            width=width,
            drop_adaln=False,
            verbose=False,
            continuation_video_latents=video_context,
            continuation_audio_latents=audio_context,
            continuation_frames=context_frames,
            terminal_target_only=terminal_target_only,
            diagnostics=session,
        )
        profile_summary = {
            "blocks": list(profile_blocks),
            "region_seconds": _region_summary(session),
            "metadata": str(profile_output.relative_to(output) / "metadata.json"),
        }

    num_frames = align_num_frames(int(round(duration * FPS)))
    geometry = estimate_continuation_cost(
        width=width,
        height=height,
        num_frames=num_frames,
        text_rows=int(embeddings.shape[1]),
        context_frames=context_frames,
        config=pipeline.dit.config,
    )
    record = {
        "context_frames": context_frames,
        "terminal_target_only": terminal_target_only,
        "geometry": geometry.to_dict(),
        "packing_median_seconds": _packing_median_seconds(
            tags=tags,
            width=width,
            height=height,
            num_frames=num_frames,
            context_frames=context_frames,
            patch_size=pipeline.dit.config.patch_size,
        ),
        "evaluation_seconds": timings,
        "evaluation_seconds_min": min(timings),
        "evaluation_seconds_median": statistics.median(timings),
        "active_memory_bytes": active,
        "peak_memory_bytes": peaks,
        "incremental_peak_bytes": [peak - base for peak, base in zip(peaks, active, strict=True)],
        "transformer_evaluations": result.transformer_evaluations,
        "profile": profile_summary,
    }
    return record, result.video_latents, result.audio_latents


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-index", type=Path, required=True)
    parser.add_argument("--transformer", type=Path, required=True)
    parser.add_argument("--text-encoder", type=Path, required=True)
    parser.add_argument("--text-config", type=Path)
    parser.add_argument("--processor", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=544)
    parser.add_argument("--duration", type=float, default=5.17)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--seed", type=int, default=54_420_260_810)
    parser.add_argument(
        "--context-frames",
        type=int,
        action="append",
        choices=(0, 5, 22, 39, 56),
    )
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--profile-context", type=int, action="append", default=[])
    parser.add_argument("--profile-block", type=int, action="append", default=[])
    parser.add_argument("--terminal-target-only", action="store_true")
    parser.add_argument("--compare-terminal-target-only", action="store_true")
    args = parser.parse_args()

    if args.steps < 2:
        raise ValueError("Context benchmark requires at least two requested schedule points.")
    contexts = tuple(args.context_frames or (0, 5, 22, 39, 56))
    profile_contexts = set(args.profile_context)
    profile_blocks = tuple(args.profile_block or (0, 25, 49))
    args.output.mkdir(parents=True, exist_ok=True)

    prompt = args.prompt_file.read_text().strip()
    encoder = MiniMaxH3TextEncoder(
        args.text_encoder,
        load_vision=False,
        config_path=args.text_config,
        processor_dir=args.processor,
        tokenizer_dir=args.tokenizer,
    )
    embeddings, tags = encoder.encode(prompt)
    tags = np.asarray(tags, dtype=np.int32)
    mx.eval(embeddings)
    del encoder
    gc.collect()
    mx.clear_cache()

    dit = load_dit(args.transformer)
    pipeline = MiniMaxH3Pipeline(
        dit,
        None,
        None,
        None,
        PipelineConfig.from_model_index(args.model_index),
    )
    cases = []
    reference_latents = {}
    terminal_modes = (
        (False, True, True, False)
        if args.compare_terminal_target_only
        else (args.terminal_target_only,)
    )
    for context_frames in contexts:
        for terminal_target_only in terminal_modes:
            print(
                f"benchmarking context={context_frames} frames "
                f"terminal_target_only={terminal_target_only}",
                flush=True,
            )
            case, video_latents, audio_latents = _run_case(
                pipeline=pipeline,
                embeddings=embeddings,
                tags=tags,
                width=args.width,
                height=args.height,
                duration=args.duration,
                steps=args.steps,
                seed=args.seed,
                context_frames=context_frames,
                repeats=args.repeats,
                output=args.output,
                profile_blocks=profile_blocks,
                profile=context_frames in profile_contexts,
                terminal_target_only=terminal_target_only,
            )
            if terminal_target_only and context_frames in reference_latents:
                reference_video, reference_audio = reference_latents[context_frames]
                video_delta = video_latents.astype(mx.float32) - reference_video.astype(
                    mx.float32
                )
                audio_delta = audio_latents.astype(mx.float32) - reference_audio.astype(
                    mx.float32
                )
                mx.eval(video_delta, audio_delta)
                case["reference_comparison"] = {
                    "video_array_equal": bool(
                        mx.array_equal(video_latents, reference_video).item()
                    ),
                    "audio_array_equal": bool(
                        mx.array_equal(audio_latents, reference_audio).item()
                    ),
                    "video_max_abs": float(mx.max(mx.abs(video_delta)).item()),
                    "audio_max_abs": float(mx.max(mx.abs(audio_delta)).item()),
                    "video_rmse": float(mx.sqrt(mx.mean(video_delta * video_delta)).item()),
                    "audio_rmse": float(mx.sqrt(mx.mean(audio_delta * audio_delta)).item()),
                }
            elif not terminal_target_only:
                reference_latents[context_frames] = (
                    video_latents + mx.zeros((), dtype=video_latents.dtype),
                    audio_latents + mx.zeros((), dtype=audio_latents.dtype),
                )
                mx.eval(*reference_latents[context_frames])
            cases.append(case)

    baseline = next(
        (
            case
            for case in cases
            if case["context_frames"] == 0 and not case["terminal_target_only"]
        ),
        None,
    )
    if baseline is not None:
        baseline_seconds = baseline["evaluation_seconds_min"]
        baseline_peak = min(baseline["peak_memory_bytes"])
        for case in cases:
            case["evaluation_overhead_percent_vs_zero"] = (
                case["evaluation_seconds_min"] / baseline_seconds - 1.0
            ) * 100.0
            case["peak_memory_delta_bytes_vs_zero"] = (
                min(case["peak_memory_bytes"]) - baseline_peak
            )

    report = {
        "schema": "weetodd-h3-context-cost-v1",
        "checkpoint": _checkpoint_identity(args.transformer),
        "canvas": [args.width, args.height],
        "duration_seconds": args.duration,
        "requested_schedule_points": args.steps,
        "seed": args.seed,
        "text_rows": int(embeddings.shape[1]),
        "drop_adaln": False,
        "terminal_target_only": args.terminal_target_only,
        "profiling_note": (
            "Profiled region timings contain synchronization overhead. "
            "Use unprofiled evaluation_seconds for production comparisons."
        ),
        "cases": cases,
    }
    target = args.output / "context_continuation_benchmark.json"
    target.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
