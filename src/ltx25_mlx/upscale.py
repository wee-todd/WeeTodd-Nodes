"""Upscale decoded ComfyUI video through LTX 2.5 latent refinement."""

from __future__ import annotations

import gc
import json
import math
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .components import (
    LTX25AudioConditioner,
    LTX25ImageConditioner,
    LTX25LatentNormalizer,
    LTX25VideoDecoder,
    load_ltx25_spatial_upsampler,
)
from .gemma_encoder import LTX25Gemma4Conditioner, resolve_prompt_context_length
from .runtime import LTX25_STAGE2_SIGMAS, LTX25ComponentSpec
from .sampling import euler_ancestral_denoise_loop
from .transformer import inspect_ltx25_ic_lora, load_ltx25_transformer
from .upscale_contracts import (
    LTX25_INPUT_SIZE_POLICIES,
    LTX25_PIXEL_SPATIAL_MODE,
    LTX25_SOURCE_FRAME_ANCHORS,
    LTX25_UPSCALE_MODES,
)


@dataclass(frozen=True)
class LTX25UpscaleResult:
    video_path: Path
    metadata_path: Path
    metadata: dict[str, Any]


def _requires_refinement(mode: str) -> bool:
    return mode != LTX25_UPSCALE_MODES[0]


def _release(*objects: Any) -> None:
    for obj in objects:
        free = getattr(obj, "free", None)
        if free is not None:
            free()
    gc.collect()
    try:
        import mlx.core as mx

        mx.clear_cache()
    except (ImportError, AttributeError):
        pass


def _host_video(images: Any):
    import numpy as np

    detach = getattr(images, "detach", None)
    if detach is not None:
        images = detach()
    cpu = getattr(images, "cpu", None)
    if cpu is not None:
        images = cpu()
    video = np.asarray(images, dtype=np.float32)
    if video.ndim != 4 or video.shape[-1] != 3 or video.shape[0] < 1:
        raise ValueError("ComfyUI IMAGE must have shape (frames, height, width, 3).")
    if not np.isfinite(video).all():
        raise ValueError("Input video contains non-finite pixel values.")
    return np.ascontiguousarray(np.clip(video, 0.0, 1.0))


def _prepare_video_size(video: Any, policy: str):
    """Apply the selected, deterministic LTX VAE grid policy."""
    import numpy as np

    if policy not in LTX25_INPUT_SIZE_POLICIES:
        raise ValueError(f"Unsupported LTX 2.5 input size policy: {policy!r}.")
    frames, height, width, _channels = video.shape
    target_height = height - height % 32
    target_width = width - width % 32
    if target_height < 32 or target_width < 32:
        raise ValueError("LTX 2.5 input video must be at least 32 by 32 pixels.")
    if target_height == height and target_width == width:
        return video, {
            "policy": policy,
            "source": {"width": width, "height": height},
            "processed": {"width": width, "height": height},
            "crop": {"left": 0, "top": 0, "right": 0, "bottom": 0},
        }
    if policy == LTX25_INPUT_SIZE_POLICIES[1]:
        raise ValueError(
            "LTX 2.5 VAE input width and height must be divisible by 32; "
            "select the center-crop policy for arbitrary movie dimensions."
        )
    top = (height - target_height) // 2
    left = (width - target_width) // 2
    bottom = height - target_height - top
    right = width - target_width - left
    cropped = video[:, top : top + target_height, left : left + target_width, :]
    return np.ascontiguousarray(cropped), {
        "policy": policy,
        "source": {"width": width, "height": height},
        "processed": {"width": target_width, "height": target_height},
        "crop": {"left": left, "top": top, "right": right, "bottom": bottom},
        "frames": frames,
    }


def _host_audio(audio: Any):
    import numpy as np

    if not isinstance(audio, dict) or "waveform" not in audio or "sample_rate" not in audio:
        raise ValueError("ComfyUI AUDIO must contain waveform and sample_rate fields.")
    waveform = audio["waveform"]
    detach = getattr(waveform, "detach", None)
    if detach is not None:
        waveform = detach()
    cpu = getattr(waveform, "cpu", None)
    if cpu is not None:
        waveform = cpu()
    waveform = np.asarray(waveform, dtype=np.float32)
    if waveform.ndim != 3 or waveform.shape[0] != 1 or waveform.shape[1] not in {1, 2}:
        raise ValueError("ComfyUI AUDIO waveform must have shape (1, channels, samples).")
    waveform = waveform[0]
    if waveform.shape[0] == 1:
        waveform = np.repeat(waveform, 2, axis=0)
    if not np.isfinite(waveform).all():
        raise ValueError("Input audio contains non-finite samples.")
    sample_rate = int(audio["sample_rate"])
    if sample_rate < 1:
        raise ValueError("Audio sample rate must be positive.")
    return np.ascontiguousarray(waveform), sample_rate


def _host_audio_or_silence(audio: Any, duration_seconds: float):
    if audio is not None:
        waveform, sample_rate = _host_audio(audio)
        return waveform, sample_rate, True
    import numpy as np

    sample_rate = 48000
    waveform = np.zeros((2, max(1, round(duration_seconds * sample_rate))), dtype=np.float32)
    return waveform, sample_rate, False


def _probe(path: Path, ffmpeg_path: Path) -> dict[str, Any] | None:
    sibling = ffmpeg_path.with_name("ffprobe")
    ffprobe = sibling if sibling.is_file() else shutil.which("ffprobe")
    if ffprobe is None:
        return None
    completed = subprocess.run(
        [
            str(ffprobe),
            "-v",
            "error",
            "-count_frames",
            "-show_entries",
            "stream=codec_type,width,height,nb_read_frames,sample_rate,channels",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
    )
    if completed.returncode:
        raise RuntimeError(
            f"ffprobe could not verify LTX 2.5 output: {completed.stderr.decode()[:500]}"
        )
    return json.loads(completed.stdout)


def _mux_command(ffmpeg: Path, silent: Path, audio: Path, partial: Path, frames: int) -> list[str]:
    return [
        str(ffmpeg),
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(silent),
        "-i",
        str(audio),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-frames:v",
        str(frames),
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        str(partial),
    ]


def upscale_video_to_file(
    spec: LTX25ComponentSpec,
    images: Any,
    audio: Any,
    target: str | Path,
    *,
    mode: str,
    prompt: str = "",
    seed: int = 0,
    fps: float = 24.0,
    input_size_policy: str = LTX25_INPUT_SIZE_POLICIES[0],
    refinement_strength: float = 0.35,
    source_frame_anchors: str = "first frame",
    first_reference_path: str | None = None,
    last_reference_path: str | None = None,
    reference_strength: float = 0.7,
    max_av_drift_seconds: float = 0.05,
    low_ram_streaming: bool = False,
    prompt_context: str = "official_1024",
    pixel_spatial_lora_path: str | None = None,
    pixel_spatial_lora_strength: float = 1.0,
    generation_metadata: dict[str, Any] | None = None,
    check_interrupted=None,
    step_callback=None,
) -> LTX25UpscaleResult:
    """Encode movie frames, upscale in LTX latent space, optionally refine, and publish."""
    if mode not in LTX25_UPSCALE_MODES:
        raise ValueError(f"Unsupported LTX 2.5 upscale mode: {mode!r}.")
    if fps <= 0:
        raise ValueError("Video fps must be positive.")
    refinement_enabled = _requires_refinement(mode)
    pixel_spatial_enabled = mode == LTX25_PIXEL_SPATIAL_MODE
    if refinement_enabled and not prompt.strip():
        raise ValueError("Stage-two LTX 2.5 refinement requires a non-empty prompt.")
    if not 0.05 <= refinement_strength <= LTX25_STAGE2_SIGMAS[0]:
        raise ValueError("LTX 2.5 cross-model refinement_strength must be between 0.05 and 0.85.")
    if not 0.0 <= reference_strength <= 1.0:
        raise ValueError("LTX 2.5 reference_strength must be between 0 and 1.")
    if source_frame_anchors not in LTX25_SOURCE_FRAME_ANCHORS:
        raise ValueError(f"Unsupported LTX 2.5 source_frame_anchors: {source_frame_anchors!r}.")
    lora_report = None
    if pixel_spatial_enabled:
        if not pixel_spatial_lora_path:
            raise ValueError("Pixel spatial IC-LoRA mode requires its LTX 2.5 LoRA checkpoint.")
        if pixel_spatial_lora_strength <= 0:
            raise ValueError("Pixel spatial IC-LoRA strength must be positive.")
        lora_report = inspect_ltx25_ic_lora(pixel_spatial_lora_path)
        if lora_report["reference_downscale_factor"] != 2:
            raise ValueError("Pixel spatial IC-LoRA mode requires reference_downscale_factor=2.")
    spec.validate()
    video = _host_video(images)
    video, size_report = _prepare_video_size(video, input_size_policy)
    frames, height, width, _channels = video.shape
    video_seconds = frames / fps
    waveform, sample_rate, source_audio_supplied = _host_audio_or_silence(audio, video_seconds)
    audio_seconds = waveform.shape[1] / sample_rate
    drift = abs(video_seconds - audio_seconds)
    if drift > max_av_drift_seconds + 1e-9:
        raise ValueError(
            f"Input audio and video differ by {drift:.6f} seconds, above the allowed "
            f"{max_av_drift_seconds:.6f} seconds."
        )

    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(f".{target.stem}.partial{target.suffix}")
    silent = target.with_name(f".{target.stem}.video.partial{target.suffix}")
    audio_path = target.with_name(f".{target.stem}.audio.partial.wav")
    metadata_path = target.with_suffix(".json")
    partial_metadata = target.with_name(f".{target.stem}.metadata.partial.json")
    image_block = audio_block = video_decoder = prompt_encoder = None
    upsampler = transformer = None
    started = time.perf_counter()
    padded_frames = 1 + 8 * math.ceil((frames - 1) / 8)
    timings: dict[str, Any] = {"stage2_evaluations": []}
    source_reference_paths: list[Path] = []

    try:
        import mlx.core as mx
        import numpy as np
        from ltx_core_mlx.components.patchifiers import AudioPatchifier, VideoLatentPatchifier
        from ltx_core_mlx.conditioning.types.latent_cond import LatentState
        from ltx_core_mlx.model.audio_vae import encode_audio
        from ltx_core_mlx.model.transformer.model import X0Model
        from ltx_core_mlx.utils.audio import load_audio
        from ltx_core_mlx.utils.positions import (
            compute_audio_positions,
            compute_audio_token_count,
            compute_video_positions,
        )
        from ltx_pipelines_mlx.utils.helpers import create_noised_state

        from minimax_h3_mlx.media import resolve_ffmpeg, save_wav

        mx.reset_peak_memory()
        if check_interrupted is not None:
            check_interrupted()
        effective_first_reference = first_reference_path
        effective_last_reference = last_reference_path
        if refinement_enabled and source_frame_anchors != "none":
            from PIL import Image

            if effective_first_reference is None:
                source_first = target.with_name(f".{target.stem}.source-first.partial.png")
                Image.fromarray((video[0] * 255).astype(np.uint8)).save(source_first)
                source_reference_paths.append(source_first)
                effective_first_reference = str(source_first)
            if source_frame_anchors == "first + last frames" and effective_last_reference is None:
                source_last = target.with_name(f".{target.stem}.source-last.partial.png")
                Image.fromarray((video[-1] * 255).astype(np.uint8)).save(source_last)
                source_reference_paths.append(source_last)
                effective_last_reference = str(source_last)
        if padded_frames != frames:
            tail = np.repeat(video[-1:, ...], padded_frames - frames, axis=0)
            video = np.concatenate((video, tail), axis=0)

        encode_started = time.perf_counter()
        pixels = mx.array(video * 2.0 - 1.0).transpose(3, 0, 1, 2)[None].astype(mx.bfloat16)
        image_block = LTX25ImageConditioner(spec.video_vae_path)
        encoder = image_block.load()
        latent = encoder.encode(pixels)
        mx.eval(latent)
        timings["video_encode_seconds"] = time.perf_counter() - encode_started
        del pixels, video

        upscale_started = time.perf_counter()
        normalizer = LTX25LatentNormalizer(spec.video_vae_path)
        denormalized = normalizer.denormalize_latent(latent.transpose(0, 2, 3, 4, 1)).transpose(
            0, 4, 1, 2, 3
        )
        upsampler = load_ltx25_spatial_upsampler(spec.spatial_upscaler_path)
        upscaled = upsampler(denormalized)
        upscaled = normalizer.normalize_latent(upscaled.transpose(0, 2, 3, 4, 1)).transpose(
            0, 4, 1, 2, 3
        )
        mx.eval(upscaled)
        timings["latent_upscale_seconds"] = time.perf_counter() - upscale_started
        del denormalized, normalizer

        refinement_conditionings = []
        if pixel_spatial_enabled:
            from ltx_core_mlx.conditioning.types.reference_video_cond import (
                VideoConditionByReferenceLatent,
            )

            reference_patchifier = VideoLatentPatchifier()
            reference_tokens, reference_spatial = reference_patchifier.patchify(latent)
            ref_f, ref_h, ref_w = reference_spatial
            refinement_conditionings.append(
                VideoConditionByReferenceLatent(
                    reference_latent=reference_tokens,
                    reference_positions=compute_video_positions(
                        ref_f,
                        ref_h,
                        ref_w,
                        frame_rate=fps,
                    ),
                    downscale_factor=2,
                    strength=1.0,
                )
            )
            mx.eval(reference_tokens)
        del latent

        if refinement_enabled and (effective_first_reference or effective_last_reference):
            from ltx_pipelines_mlx.utils._orchestration import combined_image_conditionings
            from ltx_pipelines_mlx.utils.args import ImageConditioningInput

            reference_inputs = []
            if effective_first_reference:
                reference_inputs.append(
                    ImageConditioningInput(effective_first_reference, 0, reference_strength)
                )
            if effective_last_reference:
                reference_inputs.append(
                    ImageConditioningInput(effective_last_reference, frames - 1, reference_strength)
                )
            refinement_conditionings = combined_image_conditionings(
                reference_inputs,
                enc_h=height * 2,
                enc_w=width * 2,
                spatial_dims=tuple(int(value) for value in upscaled.shape[2:]),
                video_encoder=encoder,
                frame_rate=fps,
            )
            reference_latents = []
            for conditioning in refinement_conditionings:
                for name in ("clean_latent", "keyframe_latent"):
                    value = getattr(conditioning, name, None)
                    if value is not None:
                        reference_latents.append(value)
            if reference_latents:
                mx.eval(*reference_latents)
        image_block.free()
        image_block = None
        upsampler = None
        _release()

        if refinement_enabled:
            if check_interrupted is not None:
                check_interrupted()
            prompt_started = time.perf_counter()
            prompt_encoder = LTX25Gemma4Conditioner(
                spec.text_encoder_path, connector_path=spec.transformer_path
            )
            prompt_encoder.load()
            resolved_context = resolve_prompt_context_length(
                prompt_encoder.tokenizer, prompt, prompt_context
            )
            video_embeds, audio_embeds, _mask = prompt_encoder.encode(
                prompt, max_length=resolved_context
            )
            mx.eval(video_embeds, audio_embeds)
            timings["prompt_encode_seconds"] = time.perf_counter() - prompt_started
            prompt_encoder.free()
            prompt_encoder = None
            _release()

            save_wav(audio_path, waveform, sample_rate)
            audio_started = time.perf_counter()
            audio_data = load_audio(
                str(audio_path), target_sample_rate=16000, max_duration=video_seconds
            )
            if audio_data is None:
                raise ValueError("LTX 2.5 could not read the supplied movie audio.")
            audio_block = LTX25AudioConditioner(spec.audio_vae_path)
            audio_encoder, audio_processor = audio_block.load()
            audio_latent = encode_audio(
                audio_data.waveform,
                audio_data.sample_rate,
                audio_encoder,
                audio_processor,
            )
            target_audio_tokens = compute_audio_token_count(padded_frames, frame_rate=fps)
            if audio_latent.shape[2] < target_audio_tokens:
                pad = mx.repeat(
                    audio_latent[:, :, -1:, :], target_audio_tokens - audio_latent.shape[2], axis=2
                )
                audio_latent = mx.concatenate((audio_latent, pad), axis=2)
            audio_latent = audio_latent[:, :, :target_audio_tokens, :]
            audio_patchifier = AudioPatchifier()
            audio_tokens, _ = audio_patchifier.patchify(audio_latent)
            mx.eval(audio_tokens)
            timings["audio_context_encode_seconds"] = time.perf_counter() - audio_started
            audio_block.free()
            audio_block = None
            del audio_latent, audio_data
            _release()

            video_patchifier = VideoLatentPatchifier()
            video_tokens, spatial = video_patchifier.patchify(upscaled)
            latent_f, full_h, full_w = spatial
            stage2_scale = refinement_strength / LTX25_STAGE2_SIGMAS[0]
            stage2_sigmas = tuple(
                value * stage2_scale if value else 0.0 for value in LTX25_STAGE2_SIGMAS
            )
            video_state = create_noised_state(
                base_shape=video_tokens.shape,
                conditionings=refinement_conditionings,
                spatial_dims=(latent_f, full_h, full_w),
                positions=compute_video_positions(latent_f, full_h, full_w, frame_rate=fps),
                seed=seed + 2,
                sigma=stage2_sigmas[0],
                initial_latent=video_tokens,
                legacy_scalar_blend=True,
            )
            audio_state = LatentState(
                latent=audio_tokens,
                clean_latent=audio_tokens,
                denoise_mask=mx.zeros((1, audio_tokens.shape[1], 1), dtype=mx.bfloat16),
                positions=compute_audio_positions(audio_tokens.shape[1]),
            )
            mx.eval(video_state.latent, video_state.clean_latent, audio_state.latent)
            del upscaled, video_tokens, audio_tokens
            _release()

            transformer_load_started = time.perf_counter()
            transformer = load_ltx25_transformer(
                spec.transformer_path,
                low_ram_streaming=low_ram_streaming,
                feed_forward_backend="reference_fp32",
                loras=(
                    ((pixel_spatial_lora_path, pixel_spatial_lora_strength),)
                    if pixel_spatial_enabled
                    else ()
                ),
            )
            timings["transformer_load_seconds"] = time.perf_counter() - transformer_load_started
            refine_started = time.perf_counter()
            output = euler_ancestral_denoise_loop(
                X0Model(transformer),
                video_state,
                audio_state,
                video_embeds,
                audio_embeds,
                sigmas=list(stage2_sigmas),
                noise_seed=seed + 2,
                eta=0.0,
                s_noise=1.0,
                check_interrupted=check_interrupted,
                step_callback=step_callback,
                evaluation_timing_callback=lambda index, elapsed: timings[
                    "stage2_evaluations"
                ].append({"evaluation": index, "seconds": elapsed}),
            )
            mx.eval(output.video_latent)
            timings["stage2_refine_seconds"] = time.perf_counter() - refine_started
            upscaled = video_patchifier.unpatchify(
                output.video_latent[:, : latent_f * full_h * full_w, :],
                (latent_f, full_h, full_w),
            )
            mx.eval(upscaled)
            transformer = None
            _release()
        else:
            save_wav(audio_path, waveform, sample_rate)

        if check_interrupted is not None:
            check_interrupted()
        decode_started = time.perf_counter()
        video_decoder = LTX25VideoDecoder(spec.video_vae_path, verbose=False)
        video_decoder.decode_and_stream(upscaled, str(silent), frame_rate=fps)
        video_decoder.free()
        video_decoder = None
        timings["video_decode_seconds"] = time.perf_counter() - decode_started
        del upscaled
        _release()

        if check_interrupted is not None:
            check_interrupted()
        ffmpeg = resolve_ffmpeg()
        mux_started = time.perf_counter()
        completed = subprocess.run(
            _mux_command(ffmpeg.path, silent, audio_path, partial, frames), capture_output=True
        )
        if completed.returncode:
            raise RuntimeError(f"ffmpeg LTX 2.5 mux failed: {completed.stderr.decode()[:500]}")
        timings["mux_seconds"] = time.perf_counter() - mux_started
        if not partial.is_file() or partial.stat().st_size == 0:
            raise RuntimeError("LTX 2.5 video upscaler did not produce a video file.")

        probe = _probe(partial, ffmpeg.path)
        if probe is not None:
            video_streams = [
                stream for stream in probe.get("streams", []) if stream.get("codec_type") == "video"
            ]
            audio_streams = [
                stream for stream in probe.get("streams", []) if stream.get("codec_type") == "audio"
            ]
            if len(video_streams) != 1 or len(audio_streams) != 1:
                raise RuntimeError(
                    "Published LTX 2.5 bridge output must contain one video and one audio stream."
                )
            actual = {
                "frames": int(video_streams[0]["nb_read_frames"]),
                "width": int(video_streams[0]["width"]),
                "height": int(video_streams[0]["height"]),
            }
            expected = {"frames": frames, "width": width * 2, "height": height * 2}
            if actual != expected:
                raise RuntimeError(
                    "Published LTX 2.5 bridge output violates its frame contract: "
                    f"{actual} != {expected}."
                )
        metadata = {
            **(generation_metadata or {}),
            "pipeline": "ltx2.5_video_upscale",
            "mode": mode,
            "prompt": prompt if refinement_enabled else None,
            "seed": seed,
            "input": {"frames": frames, "width": width, "height": height, "fps": fps},
            "output": {
                "frames": frames,
                "width": width * 2,
                "height": height * 2,
                "fps": fps,
            },
            "original_audio_preserved": source_audio_supplied,
            "audio_policy": "preserve source" if source_audio_supplied else "synthesize silence",
            "audio_used_as_frozen_refinement_context": refinement_enabled,
            "pixel_spatial_ic_lora": (
                {
                    "file": Path(pixel_spatial_lora_path).name,
                    "strength": pixel_spatial_lora_strength,
                    "model_version": lora_report["model_version"],
                    "reference_downscale_factor": lora_report["reference_downscale_factor"],
                    "adapter_pairs": lora_report["adapter_pairs"],
                    "full_source_video_reference": True,
                }
                if pixel_spatial_enabled and lora_report is not None
                else None
            ),
            "image_references": {
                "source_frame_anchors": source_frame_anchors,
                "first": bool(effective_first_reference),
                "last": bool(effective_last_reference),
                "external_first": bool(first_reference_path),
                "external_last": bool(last_reference_path),
                "strength": reference_strength,
            },
            "audio_sample_rate": sample_rate,
            "av_drift_seconds": drift,
            "vae_padded_frames": padded_frames - frames,
            "refinement_strength": refinement_strength if refinement_enabled else None,
            "stage2_sigmas": list(stage2_sigmas) if refinement_enabled else [],
            "timings": timings,
            "mlx_peak_bytes": int(mx.get_peak_memory()),
            "total_seconds": time.perf_counter() - started,
            "publication_probe": probe,
            "frame_policy": "causal_tail_pad_then_crop_to_input_frame_count",
            "input_size": size_report,
        }
        partial_metadata.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
        os.replace(partial, target)
        os.replace(partial_metadata, metadata_path)
        return LTX25UpscaleResult(target, metadata_path, metadata)
    except BaseException:
        partial.unlink(missing_ok=True)
        partial_metadata.unlink(missing_ok=True)
        raise
    finally:
        _release(image_block, audio_block, video_decoder, prompt_encoder)
        transformer = upsampler = None
        _release()
        audio_path.unlink(missing_ok=True)
        silent.unlink(missing_ok=True)
        for path in source_reference_paths:
            path.unlink(missing_ok=True)


upscale_h3_video_to_file = upscale_video_to_file


__all__ = [
    "LTX25_INPUT_SIZE_POLICIES",
    "LTX25_UPSCALE_MODES",
    "LTX25_PIXEL_SPATIAL_MODE",
    "LTX25_SOURCE_FRAME_ANCHORS",
    "LTX25UpscaleResult",
    "upscale_video_to_file",
    "upscale_h3_video_to_file",
]
