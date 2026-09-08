#!/usr/bin/env python3
"""Refine one native H3 movie through temporal expansion and frame recovery."""

from __future__ import annotations

import argparse
import copy
import json
import signal
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def preflight(request):
    import studio_bridge as bridge

    from wee_todd_mlx.motion_fidelity import MotionSettings, aligned_frames, validate_recipe

    settings = MotionSettings(**request["settings"])
    settings.validate()
    validate_recipe(request["recipe"])
    clip = request["clip"]
    info = bridge.inspect_media(clip["sourcePath"], request["runtime"])
    start, duration = clip.get("sourceIn", 0), clip["duration"]
    import math

    if not math.isfinite(start + duration) or start < 0 or duration < 2.5:
        raise ValueError("Motion Fidelity needs a finite trim of at least 2.5 seconds.")
    if abs(info["fps"] - 24) > 0.0001 or info["width"] % 32 or info["height"] % 32:
        raise ValueError("Use the native 24 fps H3 source with dimensions on the 32-pixel grid.")
    if (
        abs(start * 24 - round(start * 24)) > 0.001
        or abs(duration * 24 - round(duration * 24)) > 0.001
    ):
        raise ValueError("Motion Fidelity trims must land on native 24 fps frame boundaries.")
    count = round(duration * 24)
    if count > 345 or start + duration > info["duration"] + 0.025:
        raise ValueError(
            "Split the source into clips of at most 345 frames within the source movie."
        )
    # The current VAE input conversion materializes float RGB. Bound that allocation up front.
    budget = (
        aligned_frames(count * settings.maxHold)
        if settings.mode == "uniform"
        else settings.maxFrames
    )
    if budget > settings.maxFrames:
        raise ValueError("Expansion exceeds the frame budget; reduce expansion or split the clip.")
    if budget * info["width"] * info["height"] > 160_000_000:
        raise ValueError(
            "Motion Fidelity exceeds the current RGB memory budget. "
            "Split the clip or lower its frame budget."
        )
    return settings, info, count


def _execute(request, target, analyze_only=False):
    import numpy as np
    import studio_bridge as bridge
    from studio_job import atomic_json, digest, file_hash

    from wee_todd_mlx.motion_fidelity import (
        aligned_frames,
        audio_filter,
        expansion_indices,
        plan_motion,
    )

    settings, info, count = preflight(request)
    target.mkdir(parents=True, exist_ok=False)
    atomic_json(target / "request.json", request)
    recipe = copy.deepcopy(request["recipe"])
    clip, runtime = request["clip"], request["runtime"]
    source_hash = file_hash(clip["sourcePath"])
    ffmpeg = bridge.executable("ffmpeg", runtime)
    probe = bridge.executable("ffprobe", runtime)
    # Check timestamps, not just advertised average rate. VFR would break recovery alignment.
    frames = json.loads(
        bridge.run(
            [
                probe,
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-read_intervals",
                f"{clip.get('sourceIn', 0)}%+{clip['duration'] + 1}",
                "-show_frames",
                "-show_entries",
                "frame=best_effort_timestamp_time",
                "-of",
                "json",
                clip["sourcePath"],
            ],
            capture=True,
        )
    )["frames"]
    times = np.array([float(f["best_effort_timestamp_time"]) for f in frames])
    if len(times) < count or not np.allclose(np.diff(times), 1 / 24, atol=0.0001):
        raise ValueError(
            "Motion Fidelity requires a constant 24 fps source without timestamp gaps."
        )
    base = [
        ffmpeg,
        "-v",
        "error",
        "-nostdin",
        "-ss",
        str(clip.get("sourceIn", 0)),
        "-i",
        clip["sourcePath"],
    ]
    raw = target / "source.rgb"
    bridge.run(
        base
        + [
            "-map",
            "0:v:0",
            "-frames:v",
            str(count),
            "-pix_fmt",
            "rgb24",
            "-f",
            "rawvideo",
            str(raw),
        ]
    )
    if raw.stat().st_size != count * info["width"] * info["height"] * 3:
        raise ValueError("Decoded source frame count does not match the requested trim.")
    pixels = np.memmap(
        raw, dtype=np.uint8, mode="r", shape=(count, info["height"], info["width"], 3)
    )
    source_audio = target / "source.wav"
    samples = round(count * 32000 / 24)
    if info["hasAudio"]:
        bridge.run(
            base
            + [
                "-map",
                "0:a:0",
                "-af",
                f"aresample=32000,apad,atrim=end_sample={samples}",
                "-ac",
                "2",
                "-c:a",
                "pcm_f32le",
                str(source_audio),
            ]
        )
    else:
        bridge.run(
            [
                ffmpeg,
                "-v",
                "error",
                "-f",
                "lavfi",
                "-i",
                "anullsrc=r=32000:cl=stereo",
                "-af",
                f"atrim=end_sample={samples}",
                "-c:a",
                "pcm_f32le",
                str(source_audio),
            ]
        )
    from minimax_h3_mlx.media import resolve_ffmpeg
    from wee_todd_nodes.conditioning import TEXT_ENCODER_RUNTIME, H3TextEncoderSpec
    from wee_todd_nodes.decoding import (
        AUDIO_VAE_RUNTIME,
        VIDEO_VAE_RUNTIME,
        H3AudioVAESpec,
        H3VideoVAESpec,
    )
    from wee_todd_nodes.direct_publishing import RawVideoEncoder, _mux_audio
    from wee_todd_nodes.preflight import H3ComponentSetSpec
    from wee_todd_nodes.runtime import H3GenerationConfig
    from wee_todd_nodes.sampling import TRANSFORMER_RUNTIME, H3Latents, H3TransformerSpec

    components = H3ComponentSetSpec(**dict(recipe["components"], preview_override=None))
    config = H3GenerationConfig(**recipe["config"])
    video_spec = H3VideoVAESpec.from_components(components)
    audio_spec = H3AudioVAESpec.from_components(components)
    transformer_spec = H3TransformerSpec.from_components(components)
    runtimes = (TEXT_ENCODER_RUNTIME, VIDEO_VAE_RUNTIME, AUDIO_VAE_RUNTIME, TRANSFORMER_RUNTIME)
    encoder = None
    try:
        source_latents = None
        if settings.mode == "adaptive":
            bridge.emit(event="progress", message="Analyzing H3 motion latents")
            source_latents = VIDEO_VAE_RUNTIME.encode_continuation(
                video_spec,
                pixels[np.minimum(np.arange(aligned_frames(count)), count - 1)],
                unload_after=True,
            )
        plan = plan_motion(count, settings, source_latents)
        del source_latents
        plan["sourceSHA256"] = source_hash
        atomic_json(target / "plan.json", plan)
        if analyze_only or plan["noop"]:
            result = {
                "status": "success",
                "analyzed": True,
                "sourceSHA256": plan["sourceSHA256"],
                "noop": plan["noop"],
                "plan": plan,
                "video": clip["sourcePath"],
                "sourceIn": clip.get("sourceIn", 0),
                "report": str(target / "plan.json"),
            }
        else:
            bridge.emit(
                event="progress", message="Encoding expanded video and pitch-preserved audio"
            )
            expanded = np.lib.format.open_memmap(
                target / "expanded.npy",
                mode="w+",
                dtype=np.uint8,
                shape=(plan["paddedFrames"], info["height"], info["width"], 3),
            )
            for i, source_index in enumerate(expansion_indices(plan)):
                expanded[i] = pixels[source_index]
            video_latents = VIDEO_VAE_RUNTIME.encode_continuation(
                video_spec, expanded, unload_after=True
            )
            del expanded
            pcm = target / "expanded.f32"
            bridge.run(
                [
                    ffmpeg,
                    "-v",
                    "error",
                    "-nostdin",
                    "-i",
                    str(source_audio),
                    "-filter_complex",
                    audio_filter(plan),
                    "-map",
                    "[out]",
                    "-ac",
                    "2",
                    "-ar",
                    "32000",
                    "-f",
                    "f32le",
                    str(pcm),
                ]
            )
            waveform = np.fromfile(pcm, dtype=np.float32).reshape(-1, 2).T.copy()
            audio_latents = AUDIO_VAE_RUNTIME.encode_continuation(
                audio_spec, waveform, num_frames=plan["paddedFrames"], unload_after=True
            )
            config = replace(
                config,
                duration_seconds=plan["paddedFrames"] / 24,
                width=info["width"],
                height=info["height"],
                seed=settings.seed,
            )
            initial = H3Latents(
                video_latents,
                audio_latents,
                plan["paddedFrames"],
                config.width,
                config.height,
                24,
                32000,
                0,
                0,
                0,
                transformer_spec,
                config,
            )
            bridge.emit(event="progress", message="Refining expanded H3 clip")
            conditioning = TEXT_ENCODER_RUNTIME.encode(
                H3TextEncoderSpec.from_components(components, load_vision=False),
                recipe["prompt"],
                task="t2va",
                unload_after=True,
            )
            refined = TRANSFORMER_RUNTIME.sample(
                transformer_spec,
                conditioning,
                config,
                refinement_source=initial,
                refinement_mode="motion",
                refinement_strength=settings.strength,
                refinement_evaluations=settings.evaluations,
                step_callback=lambda completed, total: bridge.emit(
                    event="progress",
                    message=f"Refining H3 motion: {completed}/{total} evaluations",
                ),
                unload_after=True,
            )
            del initial, video_latents, audio_latents, waveform, conditioning
            bridge.emit(event="progress", message="Recovering original frame timing")
            silent = target / "recovered.mp4"
            encoder = RawVideoEncoder(
                silent, config.width, config.height, 24, 18, resolve_ffmpeg(ffmpeg)
            )
            recovery = np.asarray(plan["recovery"])
            cursor = 0

            def write_chunk(chunk):
                nonlocal cursor
                selection = (
                    recovery[(recovery >= cursor) & (recovery < cursor + len(chunk))] - cursor
                )
                if len(selection):
                    encoder.write(chunk[selection])
                cursor += len(chunk)

            VIDEO_VAE_RUNTIME.decode_stream(video_spec, refined, write_chunk, unload_after=True)
            encoder.close()
            if encoder.frames != count:
                raise ValueError("Recovered frame count differs from the source; output rejected.")
            output = target / "enhanced.mp4"
            _mux_audio(silent, source_audio, output, resolve_ffmpeg(ffmpeg))
            result = {
                "status": "success",
                "video": str(output),
                "sourceIn": 0,
                "sha256": file_hash(output),
                "sourceSHA256": plan["sourceSHA256"],
                "planSHA256": digest(plan),
                "plan": plan,
                "report": str(target / "plan.json"),
                "evaluations": refined.transformer_evaluations,
                "experimental": True,
            }
        if file_hash(clip["sourcePath"]) != source_hash:
            raise ValueError("Source movie changed during refinement; result was not accepted.")
        atomic_json(target / "result.json", result)
        return result
    finally:
        if encoder is not None:
            encoder.abort()
        for cache in runtimes:
            cache.unload()
        # Intermediates can be recreated deterministically; keep the recipe, plan and result.
        del pixels
        for name in ("source.rgb", "expanded.npy", "expanded.f32", "source.wav", "recovered.mp4"):
            (target / name).unlink(missing_ok=True)


def execute(request, target, analyze_only=False):
    if target.exists():
        raise ValueError(
            "Choose a new Motion Fidelity output folder; originals are never overwritten."
        )
    try:
        return _execute(request, target, analyze_only)
    finally:
        for name in ("source.rgb", "expanded.npy", "expanded.f32", "source.wav", "recovered.mp4"):
            (target / name).unlink(missing_ok=True)


def main():
    from render_h3_headless import NoComfyImports, assert_isolated

    sys.meta_path.insert(0, NoComfyImports())
    assert_isolated()
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--analyze-only", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    if args.preflight_only:
        preflight(request)
    else:
        execute(request, args.output_directory, args.analyze_only)


if __name__ == "__main__":
    main()
