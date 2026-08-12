"""Learned LTX 2.3 latent upscaling for decoded ComfyUI video frames."""

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


@dataclass(frozen=True)
class LTX23UpscalerSpec:
    """A complete LTX VAE plus one learned latent upscaler."""

    model_dir: str
    upscaler_name: str = "spatial_upscaler_x2_v1_1"

    @property
    def root(self) -> Path:
        return Path(self.model_dir).expanduser()

    @property
    def config_path(self) -> Path:
        return self.root / f"{self.upscaler_name}_config.json"

    @property
    def weights_path(self) -> Path:
        return self.root / f"{self.upscaler_name}.safetensors"

    def validate(self) -> dict[str, Any]:
        if not self.root.is_dir():
            raise FileNotFoundError(f"LTX 2.3 model directory not found: {self.root}")
        required = (
            self.root / "vae_encoder.safetensors",
            self.root / "vae_decoder.safetensors",
            self.config_path,
            self.weights_path,
        )
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError("LTX 2.3 upscale bundle is incomplete: " + ", ".join(missing))
        config = json.loads(self.config_path.read_text(encoding="utf-8")).get("config", {})
        if not config.get("spatial_upsample") or config.get("temporal_upsample"):
            raise ValueError("H3 output upscaling currently requires a spatial-only LTX upscaler.")
        scale = float(config.get("spatial_scale", 2.0))
        if scale not in {1.5, 2.0}:
            raise ValueError(f"Unsupported LTX spatial upscale factor: {scale}.")
        return config


@dataclass(frozen=True)
class LTX23UpscaleResult:
    video_path: Path
    metadata_path: Path
    metadata: dict[str, Any]


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
        raise ValueError(
            "ComfyUI IMAGE must contain video frames with shape (frames, height, width, 3)."
        )
    if not np.isfinite(video).all():
        raise ValueError("Input video contains non-finite pixel values.")
    if video.shape[1] % 32 or video.shape[2] % 32:
        raise ValueError("LTX VAE input width and height must be divisible by 32.")
    return np.ascontiguousarray(np.clip(video, 0.0, 1.0))


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
        raise ValueError("ComfyUI AUDIO waveform must have shape (1, mono-or-stereo, samples).")
    waveform = waveform[0]
    if waveform.shape[0] == 1:
        waveform = np.repeat(waveform, 2, axis=0)
    if not np.isfinite(waveform).all():
        raise ValueError("Input audio contains non-finite samples.")
    sample_rate = int(audio["sample_rate"])
    if sample_rate < 1:
        raise ValueError("Audio sample rate must be positive.")
    return np.ascontiguousarray(waveform), sample_rate


def _mux_command(
    ffmpeg_path: Path,
    silent_video: Path,
    audio_path: Path,
    partial: Path,
    frames: int,
) -> list[str]:
    """Build an exact-frame mux command without shortest-stream truncation."""
    return [
        str(ffmpeg_path),
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(silent_video),
        "-i",
        str(audio_path),
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


def _probe_video_stream(video_path: Path, ffmpeg_path: Path) -> dict[str, int] | None:
    """Return exact video dimensions/frame count when ffprobe is available."""
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
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,nb_read_frames",
            "-of",
            "json",
            str(video_path),
        ],
        capture_output=True,
    )
    if completed.returncode:
        error = completed.stderr.decode()[:500]
        raise RuntimeError(f"ffprobe could not verify LTX output: {error}")
    streams = json.loads(completed.stdout).get("streams", [])
    if len(streams) != 1:
        raise RuntimeError("ffprobe did not find exactly one video stream in LTX output.")
    stream = streams[0]
    return {
        "frames": int(stream["nb_read_frames"]),
        "width": int(stream["width"]),
        "height": int(stream["height"]),
    }


def upscale_video_to_file(
    spec: LTX23UpscalerSpec,
    images: Any,
    audio: Any,
    target: str | Path,
    *,
    fps: float = 24.0,
    max_av_drift_seconds: float = 0.05,
    generation_metadata: dict[str, Any] | None = None,
    check_interrupted=None,
) -> LTX23UpscaleResult:
    """Encode, spatially upscale, decode, and atomically publish one video."""
    if fps <= 0:
        raise ValueError("Video fps must be positive.")
    config = spec.validate()
    video = _host_video(images)
    waveform, sample_rate = _host_audio(audio)
    frames, height, width, _ = video.shape
    video_seconds = frames / fps
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
    silent_video = target.with_name(f".{target.stem}.video.partial{target.suffix}")
    audio_path = target.with_name(f".{target.stem}.audio.partial.wav")
    metadata_path = target.with_suffix(".json")
    partial_metadata = target.with_name(f".{target.stem}.metadata.partial.json")
    encoder_block = upsampler_block = decoder_block = None
    started = time.perf_counter()
    padded_frames = 1 + 8 * math.ceil((frames - 1) / 8)

    try:
        if check_interrupted is not None:
            check_interrupted()
        try:
            import mlx.core as mx
            from ltx_pipelines_mlx import ImageConditioner, VideoDecoder, VideoUpsampler
        except ImportError as exc:
            raise ImportError(
                "LTX 2.3 support is optional. Install this project with its 'ltx' extra."
            ) from exc
        from minimax_h3_mlx.media import resolve_ffmpeg, save_wav

        mx.reset_peak_memory()
        if padded_frames != frames:
            import numpy as np

            tail = np.repeat(video[-1:, ...], padded_frames - frames, axis=0)
            video = np.concatenate((video, tail), axis=0)
        pixels = mx.array(video * 2.0 - 1.0).transpose(3, 0, 1, 2)[None].astype(mx.bfloat16)
        encoder_block = ImageConditioner(spec.root)
        encoder = encoder_block.load()
        latent = encoder.encode(pixels)
        mx.eval(latent)
        del pixels, video

        latent_bfhwc = latent.transpose(0, 2, 3, 4, 1)
        denormalized = encoder.denormalize_latent(latent_bfhwc).transpose(0, 4, 1, 2, 3)
        upsampler_block = VideoUpsampler(spec.root, name=spec.upscaler_name)
        upscaled = upsampler_block(denormalized)
        normalized = encoder.normalize_latent(upscaled.transpose(0, 2, 3, 4, 1))
        normalized = normalized.transpose(0, 4, 1, 2, 3)
        mx.eval(normalized)
        del latent, latent_bfhwc, denormalized, upscaled
        encoder_block.free()
        upsampler_block.free()
        encoder_block = upsampler_block = None
        _release()

        if check_interrupted is not None:
            check_interrupted()
        save_wav(audio_path, waveform, sample_rate)
        decoder_block = VideoDecoder(spec.root, verbose=False)
        decoder_block.decode_and_stream(normalized, str(silent_video), frame_rate=fps)
        decoder_block.free()
        decoder_block = None
        if check_interrupted is not None:
            check_interrupted()
        ffmpeg = resolve_ffmpeg()
        completed = subprocess.run(
            _mux_command(ffmpeg.path, silent_video, audio_path, partial, frames),
            capture_output=True,
        )
        if completed.returncode:
            raise RuntimeError(f"ffmpeg LTX upscale mux failed: {completed.stderr.decode()[:500]}")
        if not partial.is_file() or partial.stat().st_size == 0:
            raise RuntimeError("LTX 2.3 upscaler did not produce a video file.")

        scale = float(config.get("spatial_scale", 2.0))
        expected_output = {
            "frames": frames,
            "width": round(width * scale),
            "height": round(height * scale),
        }
        publication_probe = _probe_video_stream(partial, ffmpeg.path)
        if publication_probe is not None and publication_probe != expected_output:
            raise RuntimeError(
                "Published LTX output does not match its frame contract: "
                f"{publication_probe} != {expected_output}."
            )
        metadata = {
            **(generation_metadata or {}),
            "pipeline": "ltx2.3_spatial_upscale",
            "upscaler": spec.upscaler_name,
            "spatial_scale": scale,
            "input": {"frames": frames, "width": width, "height": height, "fps": fps},
            "output": {
                **expected_output,
                "fps": fps,
            },
            "publication_probe": publication_probe,
            "vae_padded_frames": padded_frames - frames,
            "audio_sample_rate": sample_rate,
            "av_drift_seconds": drift,
            "mlx_peak_bytes": int(mx.get_peak_memory()),
            "total_seconds": time.perf_counter() - started,
            "frame_policy": "causal_tail_pad_then_crop_to_input_frame_count",
        }
        partial_metadata.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
        os.replace(partial, target)
        os.replace(partial_metadata, metadata_path)
        return LTX23UpscaleResult(target, metadata_path, metadata)
    except BaseException:
        partial.unlink(missing_ok=True)
        partial_metadata.unlink(missing_ok=True)
        raise
    finally:
        _release(encoder_block, upsampler_block, decoder_block)
        audio_path.unlink(missing_ok=True)
        silent_video.unlink(missing_ok=True)
