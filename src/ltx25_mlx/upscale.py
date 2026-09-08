"""Upscale decoded ComfyUI video through LTX 2.5 latent refinement."""

from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import shutil
import subprocess
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from wee_todd_mlx.numpy_import import adopt_numpy_array

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


_PROMPT_CACHE_LIMIT = 2
_PROMPT_CONDITIONING_CACHE: OrderedDict[tuple[Any, ...], tuple[Any, Any, int]] = OrderedDict()


def _checkpoint_identity(path: str | Path) -> tuple[str, int, int]:
    """Return a cheap process-local identity for a file or paged checkpoint directory."""
    source = Path(path).expanduser().resolve()
    identity_source = source / "paged_manifest.json" if source.is_dir() else source
    stat = identity_source.stat()
    return str(source), int(stat.st_size), int(stat.st_mtime_ns)


def _prompt_cache_key(
    spec: LTX25ComponentSpec,
    prompt: str,
    prompt_context: str,
) -> tuple[Any, ...]:
    return (
        _checkpoint_identity(spec.text_encoder_path),
        _checkpoint_identity(spec.transformer_path),
        prompt,
        prompt_context,
    )


def _cached_prompt_conditioning(key: tuple[Any, ...]):
    value = _PROMPT_CONDITIONING_CACHE.get(key)
    if value is not None:
        _PROMPT_CONDITIONING_CACHE.move_to_end(key)
    return value


def _remember_prompt_conditioning(key: tuple[Any, ...], value: tuple[Any, Any, int]) -> None:
    _PROMPT_CONDITIONING_CACHE[key] = value
    _PROMPT_CONDITIONING_CACHE.move_to_end(key)
    while len(_PROMPT_CONDITIONING_CACHE) > _PROMPT_CACHE_LIMIT:
        _PROMPT_CONDITIONING_CACHE.popitem(last=False)


def clear_upscale_prompt_cache() -> int:
    """Release reusable prompt outputs retained by the any-video refinement path."""
    count = len(_PROMPT_CONDITIONING_CACHE)
    _PROMPT_CONDITIONING_CACHE.clear()
    _release()
    return count


def _requires_refinement(mode: str) -> bool:
    return mode != LTX25_UPSCALE_MODES[0]


def _merge_refinement_conditionings(*groups: Any) -> list[Any]:
    """Preserve every independent conditioning group in deterministic order."""
    return [conditioning for group in groups for conditioning in group]


def _upscale_sol_exact_suffix_rows(video_state: Any, target_token_count: int) -> int:
    """Count appended Pixel Spatial and endpoint rows that Sol must keep exact."""
    return max(0, int(video_state.latent.shape[1]) - int(target_token_count))


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
    if policy == LTX25_INPUT_SIZE_POLICIES[2]:
        raise ValueError(
            "LTX 2.5 VAE input width and height must be divisible by 32; "
            "select the center-crop policy for arbitrary movie dimensions."
        )
    if policy == LTX25_INPUT_SIZE_POLICIES[0]:
        fitted = _nearest_aspect_grid_size(width, height, grid=32)
        if fitted is not None and fitted != (width, height):
            from PIL import Image

            fitted_width, fitted_height = fitted
            resized = np.empty(
                (frames, fitted_height, fitted_width, 3),
                dtype=np.float32,
            )
            for index, frame in enumerate(video):
                # Pillow does not support three-channel float resize consistently. Use RGB8
                # Lanczos and retain only one bounded float output buffer.
                source = Image.fromarray((np.clip(frame, 0.0, 1.0) * 255).astype(np.uint8), "RGB")
                resized[index] = (
                    np.asarray(
                        source.resize((fitted_width, fitted_height), Image.Resampling.LANCZOS),
                        dtype=np.float32,
                    )
                    / 255.0
                )
            return resized, {
                "policy": policy,
                "source": {"width": width, "height": height},
                "processed": {"width": fitted_width, "height": fitted_height},
                "operation": "lanczos_resize",
                "aspect_error_fraction": abs(
                    (fitted_width / fitted_height) / (width / height) - 1.0
                ),
                "crop": {"left": 0, "top": 0, "right": 0, "bottom": 0},
                "frames": frames,
            }
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


def _nearest_aspect_grid_size(
    width: int,
    height: int,
    *,
    grid: int,
    max_scale_delta: float = 0.35,
    max_aspect_error: float = 0.005,
) -> tuple[int, int] | None:
    """Find a nearby grid-aligned size without visibly stretching the source."""
    if width % grid == 0 and height % grid == 0:
        return width, height
    aspect = width / height
    lower = max(grid, math.floor(height * (1.0 - max_scale_delta) / grid) * grid)
    upper = max(grid, math.ceil(height * (1.0 + max_scale_delta) / grid) * grid)
    candidates = []
    for candidate_height in range(lower, upper + grid, grid):
        candidate_width = max(grid, round(candidate_height * aspect / grid) * grid)
        aspect_error = abs((candidate_width / candidate_height) / aspect - 1.0)
        if aspect_error > max_aspect_error:
            continue
        scale = candidate_height / height
        candidates.append(
            (
                abs(math.log(scale)),
                0 if scale >= 1.0 else 1,
                aspect_error,
                candidate_width,
                candidate_height,
            )
        )
    if not candidates:
        return None
    best = min(candidates)
    return best[3], best[4]


def _output_frame_megapixels(frames: int, width: int, height: int) -> float:
    """Return model-output frame megapixels, a hardware-neutral workload measure."""
    return float(frames) * float(width * 2) * float(height * 2) / 1_000_000.0


def _validate_output_workload(
    frames: int,
    width: int,
    height: int,
    limit: float,
) -> float:
    workload = _output_frame_megapixels(frames, width, height)
    if limit > 0.0 and workload > limit:
        raise ValueError(
            "LTX 2.5 upscale output workload is "
            f"{workload:.1f} frame-megapixels, above the configured {limit:.1f} limit. "
            "Shorten the clip, reduce its source dimensions, or raise the limit deliberately. "
            "This metric is a workload guard, not a cross-hardware memory prediction."
        )
    return workload


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


def _upscale_video_to_file_single(
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
    sol_attention_profile: str = "disabled",
    prompt_context: str = "official_1024",
    reuse_prompt_conditioning: bool = True,
    max_output_frame_megapixels: float = 0.0,
    pixel_spatial_lora_path: str | None = None,
    pixel_spatial_lora_strength: float = 1.0,
    ffmpeg_path: str | Path | None = None,
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
    if sol_attention_profile not in {"disabled", "paged_speed"}:
        raise ValueError("LTX 2.5 upscale Sol Attention must be disabled or paged_speed.")
    if sol_attention_profile == "paged_speed" and not low_ram_streaming:
        raise ValueError("The paged_speed upscale Sol profile requires low_ram_streaming=true.")
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
    output_frame_megapixels = _validate_output_workload(
        frames,
        width,
        height,
        max_output_frame_megapixels,
    )
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
        encode_started = time.perf_counter()
        pixels = adopt_numpy_array(video).transpose(3, 0, 1, 2)[None].astype(mx.bfloat16)
        if padded_frames != frames:
            pixels = mx.concatenate(
                (pixels, mx.repeat(pixels[:, :, -1:, ...], padded_frames - frames, axis=2)),
                axis=2,
            )
        pixels = pixels * 2.0 - 1.0
        del video
        image_block = LTX25ImageConditioner(spec.video_vae_path)
        encoder = image_block.load()
        latent = encoder.encode(pixels)
        mx.eval(latent)
        timings["video_encode_seconds"] = time.perf_counter() - encode_started
        del pixels

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
            refinement_conditionings = _merge_refinement_conditionings(
                refinement_conditionings,
                combined_image_conditionings(
                    reference_inputs,
                    enc_h=height * 2,
                    enc_w=width * 2,
                    spatial_dims=tuple(int(value) for value in upscaled.shape[2:]),
                    video_encoder=encoder,
                    frame_rate=fps,
                ),
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
            prompt_key = _prompt_cache_key(spec, prompt, prompt_context)
            cached_prompt = (
                _cached_prompt_conditioning(prompt_key) if reuse_prompt_conditioning else None
            )
            if cached_prompt is None:
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
                if reuse_prompt_conditioning:
                    _remember_prompt_conditioning(
                        prompt_key,
                        (video_embeds, audio_embeds, resolved_context),
                    )
                prompt_encoder.free()
                prompt_encoder = None
                _release()
            else:
                video_embeds, audio_embeds, resolved_context = cached_prompt
            timings["prompt_encode_seconds"] = time.perf_counter() - prompt_started
            timings["prompt_conditioning_cache"] = {
                "enabled": bool(reuse_prompt_conditioning),
                "hit": cached_prompt is not None,
                "scope": "process_local",
                "entries": len(_PROMPT_CONDITIONING_CACHE),
                "resolved_context": int(resolved_context),
            }

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
            sol_policy = {"enabled": False, "patched_video_self_attention": 0}
            if sol_attention_profile == "paged_speed":
                from wee_todd_mlx.sol_attention import SolAttentionConfig

                from .sol_attention import configure_ltx25_sol_attention

                sol_policy = configure_ltx25_sol_attention(
                    transformer,
                    SolAttentionConfig(
                        enabled=True,
                        tau=1.25,
                        min_tokens=16000,
                        start_percent=0.0,
                        dense_blocks=0,
                    ),
                )
            timings["transformer_load_seconds"] = time.perf_counter() - transformer_load_started
            refine_started = time.perf_counter()
            model = X0Model(transformer)
            if sol_attention_profile == "paged_speed":
                from .sol_attention import set_ltx25_sol_context

                set_ltx25_sol_context(
                    model,
                    step_index=0,
                    total_steps=len(stage2_sigmas) - 1,
                    exact_suffix_rows=_upscale_sol_exact_suffix_rows(
                        video_state,
                        latent_f * full_h * full_w,
                    ),
                )
            output = euler_ancestral_denoise_loop(
                model,
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
            if sol_attention_profile == "paged_speed":
                from .sol_attention import ltx25_sol_attention_report

                timings["sol_attention"] = ltx25_sol_attention_report(transformer, sol_policy)
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
        ffmpeg = resolve_ffmpeg(ffmpeg_path)
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
            "refinement_contract": (
                "generative_repaint_not_identity_safe_restoration"
                if refinement_enabled
                else "latent_resize_without_transformer_refinement"
            ),
            "sol_attention_profile": sol_attention_profile,
            "stage2_sigmas": list(stage2_sigmas) if refinement_enabled else [],
            "timings": timings,
            "mlx_peak_bytes": int(mx.get_peak_memory()),
            "total_seconds": time.perf_counter() - started,
            "publication_probe": probe,
            "frame_policy": "causal_tail_pad_then_crop_to_input_frame_count",
            "input_size": size_report,
            "output_frame_megapixels": output_frame_megapixels,
            "output_frame_megapixel_limit": max_output_frame_megapixels,
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


def _redetail_source_fingerprint(
    video: Any,
    waveform: Any,
    *,
    sample_rate: int,
    settings: dict[str, Any],
) -> str:
    """Identify resumable chunk output without retaining a second media copy."""
    digest = hashlib.sha256()
    digest.update(memoryview(video).cast("B"))
    digest.update(memoryview(waveform).cast("B"))
    digest.update(str(sample_rate).encode())
    digest.update(json.dumps(settings, sort_keys=True, separators=(",", ":")).encode())
    return digest.hexdigest()


def _chunk_file_is_valid(
    path: Path,
    *,
    ffmpeg: Path,
    frames: int,
    width: int,
    height: int,
) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        report = _probe(path, ffmpeg)
    except (OSError, RuntimeError, ValueError):
        return False
    if report is None:
        return False
    videos = [stream for stream in report.get("streams", []) if stream.get("codec_type") == "video"]
    if len(videos) != 1:
        return False
    stream = videos[0]
    return (
        int(stream.get("nb_read_frames", -1)) == frames
        and int(stream.get("width", -1)) == width
        and int(stream.get("height", -1)) == height
    )


def _upscale_video_to_file_chunked(
    spec: LTX25ComponentSpec,
    images: Any,
    audio: Any,
    target: str | Path,
    *,
    chunk_frame_megapixels: float,
    keep_chunks: bool,
    progress_callback_factory=None,
    **kwargs,
) -> LTX25UpscaleResult:
    """Run independently resumable visual chunks and remux the untouched full audio once."""
    from minimax_h3_mlx.media import resolve_ffmpeg, save_wav

    from .redetail import (
        audio_sample_bounds,
        detect_scene_cut_scores,
        plan_redetail_chunks,
        scene_cut_candidates,
    )
    fps = float(kwargs.get("fps", 24.0))
    if fps <= 0:
        raise ValueError("Video fps must be positive.")
    policy = str(kwargs.get("input_size_policy", LTX25_INPUT_SIZE_POLICIES[0]))
    video = _host_video(images)
    video, size_report = _prepare_video_size(video, policy)
    frames, height, width, _channels = video.shape
    video_seconds = frames / fps
    waveform, sample_rate, source_audio_supplied = _host_audio_or_silence(audio, video_seconds)
    drift = abs(video_seconds - waveform.shape[1] / sample_rate)
    allowed_drift = float(kwargs.get("max_av_drift_seconds", 0.05))
    if drift > allowed_drift + 1e-9:
        raise ValueError(
            f"Input audio and video differ by {drift:.6f} seconds, above the allowed "
            f"{allowed_drift:.6f} seconds."
        )
    total_workload = _validate_output_workload(
        frames,
        width,
        height,
        float(kwargs.get("max_output_frame_megapixels", 0.0)),
    )
    scores = detect_scene_cut_scores(video)
    chunks = plan_redetail_chunks(
        frames,
        fps=fps,
        output_width=width * 2,
        output_height=height * 2,
        frame_megapixel_budget=chunk_frame_megapixels,
        cut_frames=scene_cut_candidates(scores),
    )
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = resolve_ffmpeg(kwargs.get("ffmpeg_path"))
    settings = {
        "mode": kwargs.get("mode"),
        "prompt": kwargs.get("prompt", ""),
        "seed": int(kwargs.get("seed", 0)),
        "fps": fps,
        "width": width,
        "height": height,
        "chunk_frame_megapixels": chunk_frame_megapixels,
        "refinement_strength": kwargs.get("refinement_strength", 0.35),
        "source_frame_anchors": kwargs.get("source_frame_anchors", "first frame"),
        "prompt_context": kwargs.get("prompt_context", "official_1024"),
        "text_encoder": list(_checkpoint_identity(spec.text_encoder_path)),
        "transformer": list(_checkpoint_identity(spec.transformer_path)),
    }
    fingerprint = _redetail_source_fingerprint(
        video,
        waveform,
        sample_rate=sample_rate,
        settings=settings,
    )
    # ComfyUI publication chooses a fresh target name instead of overwriting an existing output.
    # Keep resumable weighted work independent from that presentation-only suffix.
    work = target.parent / f".weetodd-ltx25-redetail-{fingerprint[:16]}"
    work.mkdir(parents=True, exist_ok=True)
    manifest_path = work / "manifest.json"
    manifest = {
        "format": "weetodd-ltx25-redetail-chunks-v1",
        "fingerprint": fingerprint,
        "source": {
            "frames": frames,
            "width": width,
            "height": height,
            "fps": fps,
            "sample_rate": sample_rate,
        },
        "output": {"width": width * 2, "height": height * 2},
        "chunks": [chunk.as_dict() for chunk in chunks],
        "settings": settings,
    }
    if manifest_path.is_file():
        prior = json.loads(manifest_path.read_text())
        if prior != manifest:
            raise RuntimeError(
                "Existing LTX 2.5 chunk manifest does not match this request. "
                "Change the filename prefix or remove the stale internal chunk directory."
            )
    else:
        temporary_manifest = work / ".manifest.partial.json"
        temporary_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        os.replace(temporary_manifest, manifest_path)

    global_step_callback = kwargs.pop("step_callback", None)
    if progress_callback_factory is not None:
        global_step_callback = progress_callback_factory(len(chunks) * 3)
    completed_paths: list[Path] = []
    chunk_reports: list[dict[str, Any]] = []
    reused_chunks = 0
    started = time.perf_counter()
    for chunk in chunks:
        if kwargs.get("check_interrupted") is not None:
            kwargs["check_interrupted"]()
        chunk_target = work / f"chunk-{chunk.index:04d}.mp4"
        if _chunk_file_is_valid(
            chunk_target,
            ffmpeg=ffmpeg.path,
            frames=chunk.input_frames,
            width=width * 2,
            height=height * 2,
        ):
            reused_chunks += 1
            completed_paths.append(chunk_target)
            metadata_file = chunk_target.with_suffix(".json")
            chunk_reports.append(
                json.loads(metadata_file.read_text()) if metadata_file.is_file() else {}
            )
            if global_step_callback is not None:
                global_step_callback((chunk.index + 1) * 3, len(chunks) * 3)
            continue
        sample_start, sample_end = audio_sample_bounds(
            chunk.start_frame,
            chunk.end_frame,
            fps=fps,
            sample_rate=sample_rate,
            total_samples=waveform.shape[1],
        )
        chunk_audio = {
            "waveform": waveform[None, :, sample_start:sample_end],
            "sample_rate": sample_rate,
        }
        local_kwargs = dict(kwargs)
        local_kwargs["input_size_policy"] = LTX25_INPUT_SIZE_POLICIES[2]
        local_kwargs["max_output_frame_megapixels"] = 0.0
        local_kwargs["first_reference_path"] = (
            kwargs.get("first_reference_path") if chunk.index == 0 else None
        )
        local_kwargs["last_reference_path"] = (
            kwargs.get("last_reference_path") if chunk.index == len(chunks) - 1 else None
        )
        local_kwargs["generation_metadata"] = {
            **(kwargs.get("generation_metadata") or {}),
            "redetail_chunk": chunk.as_dict(),
            "redetail_fingerprint": fingerprint,
        }
        if global_step_callback is not None:
            offset = chunk.index * 3
            total_steps = len(chunks) * 3

            def local_step(completed, _reported_total, *, offset=offset, total=total_steps):
                global_step_callback(offset + completed, total)

            local_kwargs["step_callback"] = local_step
        result = _upscale_video_to_file_single(
            spec,
            video[chunk.start_frame : chunk.end_frame],
            chunk_audio,
            chunk_target,
            **local_kwargs,
        )
        completed_paths.append(result.video_path)
        chunk_reports.append(result.metadata)

    full_audio = work / "source-audio.wav"
    joined_silent = work / "joined-video.mp4"
    concat_list = work / "concat.txt"
    partial = target.with_name(f".{target.stem}.partial{target.suffix}")
    metadata_path = target.with_suffix(".json")
    partial_metadata = target.with_name(f".{target.stem}.metadata.partial.json")

    def discard_partial_publication() -> None:
        partial.unlink(missing_ok=True)
        partial_metadata.unlink(missing_ok=True)

    save_wav(full_audio, waveform, sample_rate)
    lines = []
    for path in completed_paths:
        escaped = str(path.resolve()).replace("'", "'\\''")
        lines.append(f"file '{escaped}'")
    concat_list.write_text("\n".join(lines) + "\n")
    joined = subprocess.run(
        [
            str(ffmpeg.path),
            "-y",
            "-loglevel",
            "error",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_list),
            "-map",
            "0:v:0",
            "-an",
            "-c:v",
            "copy",
            str(joined_silent),
        ],
        capture_output=True,
    )
    if joined.returncode:
        discard_partial_publication()
        raise RuntimeError(f"LTX 2.5 chunk join failed: {joined.stderr.decode()[:500]}")
    published = subprocess.run(
        _mux_command(ffmpeg.path, joined_silent, full_audio, partial, frames),
        capture_output=True,
    )
    if published.returncode:
        discard_partial_publication()
        raise RuntimeError(f"LTX 2.5 chunk audio mux failed: {published.stderr.decode()[:500]}")
    try:
        probe = _probe(partial, ffmpeg.path)
    except Exception:
        discard_partial_publication()
        raise
    videos = [
        stream
        for stream in (probe or {}).get("streams", [])
        if stream.get("codec_type") == "video"
    ]
    audios = [
        stream
        for stream in (probe or {}).get("streams", [])
        if stream.get("codec_type") == "audio"
    ]
    if len(videos) != 1 or len(audios) != 1:
        discard_partial_publication()
        raise RuntimeError("Chunked LTX 2.5 output must contain one video and one audio stream.")
    actual = {
        "frames": int(videos[0]["nb_read_frames"]),
        "width": int(videos[0]["width"]),
        "height": int(videos[0]["height"]),
    }
    expected = {"frames": frames, "width": width * 2, "height": height * 2}
    if actual != expected:
        discard_partial_publication()
        raise RuntimeError(f"Chunked LTX 2.5 output frame contract failed: {actual} != {expected}.")
    metadata = {
        **(kwargs.get("generation_metadata") or {}),
        "pipeline": "ltx2.5_video_upscale_chunked",
        "mode": kwargs.get("mode"),
        "prompt": kwargs.get("prompt", ""),
        "seed": int(kwargs.get("seed", 0)),
        "refinement_contract": "generative_repaint_not_identity_safe_restoration",
        "input": {"frames": frames, "width": width, "height": height, "fps": fps},
        "output": {"frames": frames, "width": width * 2, "height": height * 2, "fps": fps},
        "input_size": size_report,
        "original_audio_preserved": source_audio_supplied,
        "audio_policy": "preserve source" if source_audio_supplied else "synthesize silence",
        "av_drift_seconds": drift,
        "temporal_chunking": {
            "enabled": True,
            "frame_megapixel_budget": chunk_frame_megapixels,
            "total_frame_megapixels": total_workload,
            "fingerprint": fingerprint,
            "chunks": [chunk.as_dict() for chunk in chunks],
            "reused_chunks": reused_chunks,
            "kept": keep_chunks,
        },
        "chunk_reports": chunk_reports,
        "mlx_peak_bytes": max(
            (int(report.get("mlx_peak_bytes", 0)) for report in chunk_reports),
            default=0,
        ),
        "total_seconds": time.perf_counter() - started,
        "publication_probe": probe,
    }
    partial_metadata.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    os.replace(partial, target)
    os.replace(partial_metadata, metadata_path)
    if not keep_chunks:
        shutil.rmtree(work)
    return LTX25UpscaleResult(target, metadata_path, metadata)


def upscale_video_to_file(
    spec: LTX25ComponentSpec,
    images: Any,
    audio: Any,
    target: str | Path,
    *,
    temporal_chunking: str = "disabled",
    chunk_frame_megapixels: float = 260.0,
    keep_chunks: bool = False,
    progress_callback_factory=None,
    **kwargs,
) -> LTX25UpscaleResult:
    """Upscale one clip, or partition a long clip into resumable scene-aware chunks."""
    if temporal_chunking not in {"disabled", "auto scene-aware"}:
        raise ValueError("Temporal chunking must be disabled or auto scene-aware.")
    if temporal_chunking == "auto scene-aware":
        return _upscale_video_to_file_chunked(
            spec,
            images,
            audio,
            target,
            chunk_frame_megapixels=chunk_frame_megapixels,
            keep_chunks=keep_chunks,
            progress_callback_factory=progress_callback_factory,
            **kwargs,
        )
    if progress_callback_factory is not None and kwargs.get("step_callback") is None:
        kwargs["step_callback"] = progress_callback_factory(3)
    return _upscale_video_to_file_single(spec, images, audio, target, **kwargs)


upscale_h3_video_to_file = upscale_video_to_file


__all__ = [
    "LTX25_INPUT_SIZE_POLICIES",
    "LTX25_UPSCALE_MODES",
    "LTX25_PIXEL_SPATIAL_MODE",
    "LTX25_SOURCE_FRAME_ANCHORS",
    "LTX25UpscaleResult",
    "clear_upscale_prompt_cache",
    "upscale_video_to_file",
    "upscale_h3_video_to_file",
]
