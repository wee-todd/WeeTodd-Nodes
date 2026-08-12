"""Latent-native H3 motion continuation and synchronized overlap trimming."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

SUPPORTED_CONTEXT_FRAMES = (5, 22, 39, 56)
FPS = 24


@dataclass(frozen=True)
class H3ContinuationContext:
    """A copied tail of synchronized H3 video and audio latents."""

    video: Any
    audio: Any
    context_frames: int
    width: int
    height: int
    fps: int
    sample_rate: int
    transformer_checkpoint: str
    transformer_path: str

    @property
    def video_latent_frames(self) -> int:
        from minimax_h3_mlx.packing import video_latent_num_frames

        return video_latent_num_frames(self.context_frames)

    @property
    def audio_latent_frames(self) -> int:
        from minimax_h3_mlx.packing import audio_latent_num_frames

        return audio_latent_num_frames(self.context_frames)


def continuation_context_from_latents(latents, context_frames: int) -> H3ContinuationContext:
    """Copy a legal H3 temporal tail from synchronized undecoded latents."""
    from minimax_h3_mlx.packing import audio_latent_num_frames, video_latent_num_frames

    if context_frames not in SUPPORTED_CONTEXT_FRAMES:
        raise ValueError(
            f"H3 continuation context must be one of {SUPPORTED_CONTEXT_FRAMES}, "
            f"got {context_frames}."
        )
    if latents.fps != FPS or latents.sample_rate != 32000:
        raise ValueError("H3 continuation requires 24 fps video and 32 kHz audio.")
    if context_frames >= latents.num_frames:
        raise ValueError("Continuation context must be shorter than the source clip.")

    video_count = video_latent_num_frames(context_frames)
    audio_count = audio_latent_num_frames(context_frames)
    if len(latents.video.shape) != 5 or int(latents.video.shape[0]) != 1:
        raise ValueError("H3 video latents must have shape (1, channels, frames, height, width).")
    if len(latents.audio.shape) != 3 or int(latents.audio.shape[0]) != 2:
        raise ValueError("H3 audio latents must have shape (2, channels, frames).")
    if int(latents.video.shape[2]) < video_count or int(latents.audio.shape[2]) < audio_count:
        raise ValueError("The source latent streams are shorter than the requested context.")

    video = latents.video[:, :, -video_count:, :, :]
    audio = latents.audio[:, :, -audio_count:]
    try:
        import mlx.core as mx

        video = video + mx.zeros((), dtype=video.dtype)
        audio = audio + mx.zeros((), dtype=audio.dtype)
        mx.eval(video, audio)
    except (ImportError, AttributeError, TypeError):
        video = video.copy()
        audio = audio.copy()

    spec = latents.transformer_spec
    return H3ContinuationContext(
        video=video,
        audio=audio,
        context_frames=context_frames,
        width=latents.width,
        height=latents.height,
        fps=latents.fps,
        sample_rate=latents.sample_rate,
        transformer_checkpoint=spec.checkpoint,
        transformer_path=spec.transformer,
    )


def validate_continuation_for_sample(context: H3ContinuationContext, spec, config) -> None:
    """Reject geometry or provenance mismatches before transformer execution."""
    if spec.task not in {"t2va", "fl2va", "ref2va"}:
        raise ValueError("H3 motion continuation supports T2VA, FL2VA, and Ref2VA checkpoints.")
    if context.width != config.width or context.height != config.height:
        raise ValueError(
            "Continuation canvas must match the new generation: "
            f"context is {context.width}x{context.height}, request is "
            f"{config.width}x{config.height}."
        )
    if context.fps != FPS or context.sample_rate != 32000:
        raise ValueError("Continuation context timing must be 24 fps and 32 kHz.")
    if (
        context.transformer_checkpoint != spec.checkpoint
        or context.transformer_path != spec.transformer
    ):
        raise ValueError(
            "Continuation context and sampler must use the same transformer checkpoint."
        )


def trim_continuation_overlap(images, audio: dict[str, Any], context_frames: int, fps: int = FPS):
    """Remove the repeated head and force audio to the exact remaining video duration."""
    if context_frames not in SUPPORTED_CONTEXT_FRAMES:
        raise ValueError(f"Unsupported H3 continuation context: {context_frames} frames.")
    if len(images.shape) != 4 or int(images.shape[0]) <= context_frames:
        raise ValueError("IMAGE must contain more frames than the continuation overlap.")
    if not isinstance(audio, dict) or "waveform" not in audio or "sample_rate" not in audio:
        raise ValueError("AUDIO must contain waveform and sample_rate fields.")
    waveform = audio["waveform"]
    sample_rate = int(audio["sample_rate"])
    if sample_rate != 32000:
        raise ValueError(f"H3 continuation audio must be 32 kHz, got {sample_rate} Hz.")
    if len(waveform.shape) not in {2, 3} or int(waveform.shape[-2]) != 2:
        raise ValueError("H3 continuation audio must be stereo.")

    trimmed_images = images[context_frames:]
    drop_samples = round(context_frames / fps * sample_rate)
    if int(waveform.shape[-1]) <= drop_samples:
        raise ValueError("AUDIO is shorter than the continuation overlap.")
    trimmed_waveform = waveform[..., drop_samples:]
    target_samples = round(int(trimmed_images.shape[0]) / fps * sample_rate)
    available = int(trimmed_waveform.shape[-1])
    adjustment = "none"
    if available > target_samples:
        trimmed_waveform = trimmed_waveform[..., :target_samples]
        adjustment = "truncated"
    elif available < target_samples:
        missing = target_samples - available
        try:
            import torch

            padding = torch.zeros(
                (*trimmed_waveform.shape[:-1], missing),
                dtype=trimmed_waveform.dtype,
                device=trimmed_waveform.device,
            )
            trimmed_waveform = torch.cat((trimmed_waveform, padding), dim=-1)
        except (ImportError, AttributeError, TypeError):
            import numpy as np

            padding = np.zeros(
                (*trimmed_waveform.shape[:-1], missing), dtype=trimmed_waveform.dtype
            )
            trimmed_waveform = np.concatenate((trimmed_waveform, padding), axis=-1)
        adjustment = "zero_padded"

    info = {
        "context_frames_removed": context_frames,
        "context_samples_removed": drop_samples,
        "output_frames": int(trimmed_images.shape[0]),
        "output_samples": target_samples,
        "fps": fps,
        "sample_rate": sample_rate,
        "audio_adjustment": adjustment,
    }
    return trimmed_images, {"waveform": trimmed_waveform, "sample_rate": sample_rate}, info
