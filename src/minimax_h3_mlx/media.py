"""Writing MiniMax-H3 output to files.

Deliberately dependency-free: stereo WAV goes through the standard library's ``wave`` module and
video is piped as raw RGB frames into the ``ffmpeg`` binary, so the port needs no imageio,
soundfile or torchvision. If ``ffmpeg`` is absent the frames can still be written as a PNG
sequence.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class FFmpegExecutable:
    """Resolved ffmpeg executable and the portable mechanism that selected it."""

    path: Path
    source: str

    def to_dict(self) -> dict[str, str]:
        return {"path": str(self.path), "source": self.source}


def _executable_path(value: str, search_path: str | None = None) -> Path | None:
    candidate = Path(value).expanduser()
    if candidate.parent != Path(".") or candidate.is_absolute():
        resolved = candidate.resolve()
        return resolved if resolved.is_file() and os.access(resolved, os.X_OK) else None
    located = shutil.which(value, path=search_path)
    return Path(located).resolve() if located else None


def resolve_ffmpeg(
    explicit_path: str | Path | None = None,
    *,
    environ: dict[str, str] | None = None,
) -> FFmpegExecutable:
    """Find ffmpeg without assuming that ComfyUI inherited an interactive shell PATH."""
    environment = os.environ if environ is None else environ
    configured = []
    if explicit_path is not None and str(explicit_path).strip():
        configured.append(("node override", str(explicit_path).strip()))
    for variable in ("WEETODD_FFMPEG", "FFMPEG_BINARY", "IMAGEIO_FFMPEG_EXE"):
        value = environment.get(variable, "").strip()
        if value:
            configured.append((f"environment variable {variable}", value))

    for source, value in configured:
        resolved = _executable_path(value, environment.get("PATH"))
        if resolved is None:
            raise RuntimeError(f"The {source} does not name an executable ffmpeg file.")
        return FFmpegExecutable(resolved, source)

    located = shutil.which("ffmpeg", path=environment.get("PATH"))
    if located:
        return FFmpegExecutable(Path(located).resolve(), "process PATH")

    try:
        import imageio_ffmpeg

        packaged = _executable_path(imageio_ffmpeg.get_ffmpeg_exe())
        if packaged is not None:
            return FFmpegExecutable(packaged, "imageio-ffmpeg package")
    except (ImportError, RuntimeError, OSError):
        pass

    raise RuntimeError(
        "ffmpeg is unavailable to the ComfyUI Python process. Set the node's ffmpeg_path, "
        "set WEETODD_FFMPEG, or install ffmpeg on the process PATH."
    )


def ffmpeg_status(explicit_path: str | Path | None = None) -> dict[str, str | bool]:
    """Return safe preflight information without exposing the process PATH."""
    try:
        executable = resolve_ffmpeg(explicit_path)
    except RuntimeError as exc:
        return {"available": False, "error": str(exc)}
    return {"available": True, **executable.to_dict()}


def save_wav(path: str | Path, audio: np.ndarray, sample_rate: int) -> Path:
    """Write ``(channels, samples)`` float audio in ``[-1, 1]`` as 16-bit PCM."""
    path = Path(path)
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 1:
        audio = audio[None]
    clipped = np.clip(audio, -1.0, 1.0)
    # Interleave channels, which is what WAV expects.
    pcm = (clipped.T * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as fh:
        fh.setnchannels(audio.shape[0])
        fh.setsampwidth(2)
        fh.setframerate(int(sample_rate))
        fh.writeframes(pcm.tobytes())
    return path


def save_mp4(
    path: str | Path,
    video: np.ndarray,
    fps: float,
    audio: np.ndarray | None = None,
    sample_rate: int = 32000,
    crf: int = 18,
    audio_tempo: float = 1.0,
    ffmpeg_path: str | Path | None = None,
) -> Path:
    """Encode ``(frames, height, width, 3)`` uint8 video, muxing audio when given.

    ``audio_tempo`` below 1.0 stretches the generated audio to cover a clip written at a reduced
    frame rate. ``atempo`` preserves pitch, so speech stays intelligible where a plain resample
    would drop it an octave — but it is a time-stretch artefact, not a model output, and long
    stretches audibly smear transients.

    Resolves an explicit override, environment configuration, process PATH, or an optional
    imageio-ffmpeg installation. Use :func:`save_frames` when no encoder is available.
    """
    ffmpeg = str(resolve_ffmpeg(ffmpeg_path).path)

    path = Path(path)
    video = np.ascontiguousarray(video, dtype=np.uint8)
    frames, height, width, _ = video.shape

    audio_path = None
    if audio is not None:
        audio_path = path.with_suffix(".wav")
        save_wav(audio_path, audio, sample_rate)

    cmd = [
        ffmpeg,
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "pipe:0",
    ]
    if audio_path is not None:
        cmd += ["-i", str(audio_path)]
        if abs(audio_tempo - 1.0) > 1e-6:
            # atempo is only defined on [0.5, 100]; chain factors for anything slower.
            factors, remaining = [], float(audio_tempo)
            while remaining < 0.5:
                factors.append(0.5)
                remaining /= 0.5
            factors.append(remaining)
            cmd += ["-filter:a", ",".join(f"atempo={f:.6f}" for f in factors)]
        cmd += ["-c:a", "aac", "-b:a", "192k", "-shortest"]
    cmd += ["-c:v", "libx264", "-crf", str(crf), "-pix_fmt", "yuv420p", str(path)]

    process = subprocess.run(cmd, input=video.tobytes(), capture_output=True)
    if process.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {process.stderr.decode()[:500]}")
    return path


def save_frames(directory: str | Path, video: np.ndarray, limit: int | None = None) -> Path:
    """Write frames as ``frame_00000.png`` — the fallback when ffmpeg is unavailable."""
    from PIL import Image

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    count = len(video) if limit is None else min(limit, len(video))
    for index in range(count):
        Image.fromarray(video[index]).save(directory / f"frame_{index:05d}.png")
    return directory
