"""Validate request-owned Draw Things media and hand verified frames to FFmpeg."""

from __future__ import annotations

import math
import os
import signal
import struct
import subprocess
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

from PIL import Image

MEDIA_SCHEMA = "weetodd-drawthings-media-v1"


def _integer(value: Any, name: str, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer of at least {minimum}")
    return value


def _confined_path(root: Path, value: Any, name: str, *, directory: bool = False) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty relative path")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{name} must be confined to the request root")
    root = root.resolve(strict=True)
    try:
        resolved = (root / relative).resolve(strict=True)
    except (FileNotFoundError, RuntimeError) as error:
        raise ValueError(f"{name} does not exist inside the request root") from error
    if not resolved.is_relative_to(root):
        raise ValueError(f"{name} must be confined to the request root")
    if directory != resolved.is_dir():
        expected = "directory" if directory else "file"
        raise ValueError(f"{name} must identify a {expected}")
    return resolved


def _png_size(path: Path, name: str) -> tuple[int, int]:
    try:
        with Image.open(path) as image:
            if image.format != "PNG":
                raise ValueError(f"{name} must be a PNG image")
            size = image.size
            image.verify()
    except (OSError, SyntaxError) as error:
        raise ValueError(f"{name} is not a valid PNG image") from error
    return size


def _measure_wav(path: Path) -> tuple[int, int, int]:
    file_size = path.stat().st_size
    fmt = None
    pcm_offset = None
    pcm_size = None
    with path.open("rb") as stream:
        header = stream.read(12)
        if len(header) != 12 or header[:4] != b"RIFF" or header[8:] != b"WAVE":
            raise ValueError("audioPath must contain a RIFF WAVE file")
        if struct.unpack_from("<I", header, 4)[0] + 8 != file_size:
            raise ValueError("audioPath RIFF size does not match the file")
        while stream.tell() + 8 <= file_size:
            chunk_header = stream.read(8)
            chunk_id = chunk_header[:4]
            size = struct.unpack_from("<I", chunk_header, 4)[0]
            start = stream.tell()
            end = start + size
            if end > file_size:
                raise ValueError("audioPath contains an incomplete WAV chunk")
            if chunk_id == b"fmt " and fmt is None:
                if size < 16:
                    raise ValueError("audioPath contains an incomplete WAV format chunk")
                fmt = stream.read(16)
            elif chunk_id == b"data" and pcm_offset is None:
                pcm_offset, pcm_size = start, size
            stream.seek(end + (size & 1))
    if fmt is None or pcm_offset is None or pcm_size is None:
        raise ValueError("audioPath must contain WAV format and data chunks")
    audio_format, channels, rate, byte_rate, block_align, bits = struct.unpack("<HHIIHH", fmt)
    if audio_format != 3 or bits != 32 or channels < 1 or rate < 1:
        raise ValueError("audioPath must contain interleaved IEEE Float32 WAV audio")
    if block_align != channels * 4 or byte_rate != rate * block_align or pcm_size % block_align:
        raise ValueError("audioPath has invalid interleaved Float32 sample data")
    with path.open("rb") as stream:
        stream.seek(pcm_offset)
        remaining = pcm_size
        while remaining:
            chunk = stream.read(min(remaining, 1024 * 1024))
            if not chunk:
                raise ValueError("audioPath contains incomplete sample data")
            if any(not math.isfinite(value[0]) for value in struct.iter_unpack("<f", chunk)):
                raise ValueError("audioPath samples must be finite Float32 values")
            remaining -= len(chunk)
    return rate, pcm_size // block_align, channels


def _check_dimensions(paths: list[Path], configuration: dict[str, Any]) -> tuple[int, int]:
    sizes = [_png_size(path, path.name) for path in paths]
    if not sizes or any(size != sizes[0] for size in sizes[1:]):
        raise ValueError("all PNG frames must have matching dimensions")
    width, height = sizes[0]
    for key, measured in (("width", width), ("height", height)):
        declared = configuration.get(key)
        if declared is not None and declared != measured:
            raise ValueError(f"measured {key} does not match configuration")
    return width, height


def validate_media(
    manifest: dict[str, Any],
    *,
    root: Path,
    expected_request_id: str,
    expected_operation: str,
    expected_frames: int,
    requires_audio: bool = False,
) -> dict[str, Any]:
    """Return a normalized manifest only after measuring every declared artifact."""
    if not isinstance(manifest, dict):
        raise ValueError("media manifest must be an object")
    result = dict(manifest)
    if result.get("schema") != MEDIA_SCHEMA:
        raise ValueError(f"schema must be {MEDIA_SCHEMA}")
    if result.get("requestID") != expected_request_id:
        raise ValueError("requestID does not match the submitted request")
    if (
        not isinstance(expected_operation, str)
        or expected_operation not in {"image", "video"}
        or result.get("operation") != expected_operation
    ):
        raise ValueError("operation does not match the submitted request")
    expected_frames = _integer(expected_frames, "expected_frames")
    frame_count = _integer(result.get("frameCount"), "frameCount")
    if frame_count != expected_frames:
        raise ValueError("frameCount does not match the requested frame count")
    configuration = result.get("configuration")
    if not isinstance(configuration, dict):
        raise ValueError("configuration must be an object")

    if expected_operation == "image":
        image_paths = result.get("imagePaths")
        if not isinstance(image_paths, list) or len(image_paths) != frame_count:
            raise ValueError("imagePaths must contain exactly frameCount paths")
        paths = [
            _confined_path(root, value, f"imagePaths[{index}]")
            for index, value in enumerate(image_paths)
        ]
        expected_names = [f"{index:08d}.png" for index in range(frame_count)]
        if [path.name for path in paths] != expected_names:
            raise ValueError("imagePaths must be sequential eight-digit zero-based PNG names")
        result["imagePaths"] = [str(path) for path in paths]
    else:
        frames_directory = _confined_path(
            root, result.get("framesDirectory"), "framesDirectory", directory=True
        )
        paths = sorted(frames_directory.glob("*.png"))
        expected_names = [f"{index:08d}.png" for index in range(frame_count)]
        if len(paths) != frame_count or [path.name for path in paths] != expected_names:
            raise ValueError("video PNG frames must be sequential eight-digit zero-based files")
        paths = [
            _confined_path(root, str(path.relative_to(root.resolve())), f"frame {path.name}")
            for path in paths
        ]
        result["framesDirectory"] = str(frames_directory)
        numerator = _integer(result.get("fpsNumerator"), "fpsNumerator")
        denominator = _integer(result.get("fpsDenominator"), "fpsDenominator")
        declared_requires_audio = result.get("requiresAudio", False)
        if not isinstance(declared_requires_audio, bool):
            raise ValueError("requiresAudio must be a boolean")
        result["requiresAudio"] = requires_audio or declared_requires_audio
        audio_path = result.get("audioPath")
        if (requires_audio or declared_requires_audio) and audio_path is None:
            raise ValueError("audio is required for this synchronized AV result")
        if audio_path is not None:
            resolved_audio = _confined_path(root, audio_path, "audioPath")
            rate, samples, channels = _measure_wav(resolved_audio)
            declared_rate = _integer(result.get("sampleRate"), "sampleRate")
            declared_samples = _integer(result.get("sampleCount"), "sampleCount", minimum=0)
            declared_channels = _integer(result.get("channels"), "channels")
            if (rate, samples, channels) != (declared_rate, declared_samples, declared_channels):
                raise ValueError("measured audio format or sample count does not match manifest")
            video_duration = frame_count * denominator / numerator
            audio_duration = samples / rate
            timing = result.get("audioTiming")
            if timing is not None:
                # Draw Things' LTX decoder has a fixed causal audio clock;
                # playback FPS does not change the number of decoded samples.
                # Remeasure the exact contract, never trust a duration waiver.
                if (
                    timing != "ltx-causal-v1" or rate not in (24000, 48000)
                    or (frame_count - 1) % 8 != 0
                    or samples != (4 * (frame_count - 1) + 1) * (rate // 100)
                ):
                    raise ValueError("audio does not match the declared LTX causal sample count")
                if audio_duration > video_duration:
                    raise ValueError(
                        "LTX audio exceeds video duration; use 24 or 25 FPS to preserve its tail"
                    )
            elif abs(video_duration - audio_duration) > denominator / numerator + 0.01:
                raise ValueError("audio duration differs from video by more than one frame")
            result["audioPath"] = str(resolved_audio)

    width, height = _check_dimensions(paths, configuration)
    result["width"] = width
    result["height"] = height
    return result


def finish_video(
    manifest: dict[str, Any],
    output: Path,
    ffmpeg: Path,
    *,
    cancelled: Callable[[], bool] = lambda: False,
) -> Path:
    """Mux an already validated frame manifest using its exact rational frame rate."""
    if manifest.get("operation") != "video":
        raise ValueError("finish_video requires a video media manifest")
    executable = Path(ffmpeg).resolve(strict=True)
    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise ValueError("ffmpeg must be an executable file")
    frames_directory = Path(manifest.get("framesDirectory", ""))
    if not frames_directory.is_absolute() or not frames_directory.is_dir():
        raise ValueError("framesDirectory must come from validate_media")
    numerator = _integer(manifest.get("fpsNumerator"), "fpsNumerator")
    denominator = _integer(manifest.get("fpsDenominator"), "fpsDenominator")
    frame_count = _integer(manifest.get("frameCount"), "frameCount")
    audio_value = manifest.get("audioPath")
    if manifest.get("requiresAudio", False) and audio_value is None:
        raise ValueError("audio is required; silent fallback is prohibited")
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"finished video already exists: {output}")
    temporary = output.with_name(f".{output.stem}-{uuid.uuid4().hex}.partial{output.suffix}")
    fps = f"{numerator}/{denominator}"
    command = [
        str(executable), "-v", "error", "-y", "-framerate", fps, "-start_number", "0",
        "-i", str(frames_directory / "%08d.png"),
    ]
    if audio_value is not None:
        audio_path = Path(audio_value)
        if not audio_path.is_absolute() or not audio_path.is_file():
            raise ValueError("audioPath must come from validate_media")
        command += ["-i", str(audio_path)]
    command += ["-frames:v", str(frame_count), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", fps]
    if audio_value is not None:
        command += ["-c:a", "aac"]
    command.append(str(temporary))
    process = None
    try:
        if cancelled():
            raise InterruptedError("video finishing cancelled")
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        while process.poll() is None:
            if cancelled():
                raise InterruptedError("video finishing cancelled")
            time.sleep(0.05)
        status = process.returncode
        if status != 0 or not temporary.is_file():
            raise RuntimeError(f"FFmpeg failed to finish video (status {status})")
        os.link(temporary, output)
    finally:
        if process is not None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=0.5)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
        temporary.unlink(missing_ok=True)
    return output
