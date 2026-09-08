"""Bounded local media boundary for task-conditioned headless renders."""

from __future__ import annotations

import json
import math
import shutil
import subprocess
from fractions import Fraction
from pathlib import Path

from .task_conditioning import frame_geometry


def media_binary(recipe, name):
    supplied = recipe.get(name)
    sibling = Path(recipe["ffmpeg"]).with_name(name) if recipe.get("ffmpeg") else None
    value = supplied or (str(sibling) if sibling and sibling.is_file() else shutil.which(name))
    if not value:
        raise FileNotFoundError(f"Install {name} or supply its explicit path")
    return str(value)


def assemble_external_extension(
    source,
    generated,
    target,
    *,
    source_frames,
    context_frames,
    fps,
    sample_rate,
    ffmpeg,
):
    """Preserve a source AV prefix and append only newly generated continuation."""

    source_duration = source_frames / fps
    context_duration = context_frames / fps
    command = [
        str(ffmpeg),
        "-v",
        "error",
        "-nostdin",
        "-y",
        "-i",
        str(source),
        "-i",
        str(generated),
        "-filter_complex",
        (
            f"[0:v]fps={fps},setpts=PTS-STARTPTS[srcv];"
            f"[1:v]trim=start_frame={context_frames},setpts=PTS-STARTPTS[newv];"
            "[srcv][newv]concat=n=2:v=1:a=0[v];"
            f"[0:a]aresample={sample_rate},apad,atrim=duration={source_duration:.9f},"
            "asetpts=PTS-STARTPTS[srca];"
            f"[1:a]aresample={sample_rate},atrim=start={context_duration:.9f},"
            "asetpts=PTS-STARTPTS,apad[newa];"
            "[srca][newa]concat=n=2:v=0:a=1[a]"
        ),
        "-map",
        "[v]",
        "-map",
        "[a]",
        "-c:v",
        "libx264",
        "-crf",
        "18",
        "-preset",
        "medium",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-movflags",
        "+faststart",
        "-shortest",
        str(target),
    ]
    subprocess.run(command, check=True, timeout=300)
    if not Path(target).is_file() or not Path(target).stat().st_size:
        raise RuntimeError("External extension assembly did not produce a video")


def inspect_media(recipe, contract):
    """Inspect actual media types/geometry before any weighted runtime is acquired."""
    reports = []
    for item in contract["inputs"]:
        if item["kind"] == "image":
            from PIL import Image

            with Image.open(item["path"]) as image:
                if getattr(image, "n_frames", 1) != 1:
                    raise ValueError(
                        f"{item['id']}: animated images require explicit video transport"
                    )
                width, height = image.size
                image.verify()
            if width * height > 32_000_000:
                raise ValueError(f"{item['id']}: image exceeds the 32 megapixel input limit")
            report = {"width": width, "height": height}
        else:
            result = subprocess.run(
                [
                    media_binary(recipe, "ffprobe"),
                    "-v",
                    "error",
                    "-show_streams",
                    "-show_format",
                    "-of",
                    "json",
                    item["path"],
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=30,
            )
            metadata = json.loads(result.stdout)
            streams = [
                s for s in metadata.get("streams", []) if s.get("codec_type") == item["kind"]
            ]
            if len(streams) != 1:
                raise ValueError(f"{item['id']}: expected exactly one {item['kind']} stream")
            stream = streams[0]
            duration = float(
                stream.get("duration") or metadata.get("format", {}).get("duration") or 0
            )
            maximum = 15 if recipe["engine"] == "h3" else 30
            if not math.isfinite(duration) or not 0 < duration <= maximum + 0.05:
                raise ValueError(
                    f"{item['id']}: input duration must be positive and at most {maximum}s"
                )
            report = {"duration_seconds": duration}
            if item["kind"] == "video":
                if any(
                    float(side.get("rotation", 0)) != 0 for side in stream.get("side_data_list", [])
                ):
                    raise ValueError(f"{item['id']}: bake video rotation into pixels before import")
                fps = float(Fraction(stream.get("avg_frame_rate", "0/1")))
                nominal = float(Fraction(stream.get("r_frame_rate", "0/1")))
                if abs(fps - nominal) > 0.01:
                    raise ValueError(
                        f"{item['id']}: normalize variable-frame-rate video before import"
                    )
                width, height = int(stream["width"]), int(stream["height"])
                num_frames = int(stream.get("nb_frames") or round(duration * fps))
                if not 0 < fps <= 120 or min(width, height) < 1:
                    raise ValueError(f"{item['id']}: unsupported video geometry/fps")
                # The transport does not select/trim a source interval implicitly.
                if width * height * math.ceil(duration * fps) * 3 > 512 * 1024**2:
                    raise ValueError(
                        f"{item['id']}: decoded reference exceeds 512 MiB; resize the input first"
                    )
                if recipe["engine"] != "h3" and abs(fps - frame_geometry(recipe)[1]) > 0.001:
                    raise ValueError(f"{item['id']}: LTX control video fps must match output fps")
                report.update(width=width, height=height, fps=fps, num_frames=num_frames)
                if contract.get("task") == "extension":
                    report["embedded_audio_policy"] = (
                        contract["audio_policy"]
                    )
                    config = recipe["config"]
                    if (width, height) != (config["width"], config["height"]):
                        raise ValueError(
                            f"{item['id']}: extension source geometry must match the output config"
                        )
                    if recipe["engine"] == "h3":
                        window_frames, window_fps = frame_geometry(recipe)
                        if abs(fps - window_fps) > 0.001:
                            raise ValueError("H3 extension source must be constant 24 fps")
                        if duration < 2.0:
                            raise ValueError(
                                "H3 Ref2VA extension source must be at least 2 seconds long"
                            )
                        if window_frames != contract["extension"]["additional_frames"]:
                            raise ValueError(
                                "H3 aligned Ref2VA generation window must equal "
                                "extension.additional_frames"
                            )
                        audio_streams = [
                            value
                            for value in metadata.get("streams", [])
                            if value.get("codec_type") == "audio"
                        ]
                        if len(audio_streams) != 1:
                            raise ValueError(
                                "H3 extension requires exactly one embedded source audio stream"
                            )
                        audio_stream = audio_streams[0]
                        channels = int(audio_stream["channels"])
                        sample_rate = int(audio_stream["sample_rate"])
                        if channels not in {1, 2} or not 0 < sample_rate <= 192000:
                            raise ValueError("H3 extension source audio must be mono or stereo")
                        report.update(
                            audio_channels=channels,
                            audio_sample_rate=sample_rate,
                        )
                    elif recipe["engine"] == "ltx23":
                        expected_frames, expected_fps = frame_geometry(recipe)
                        if abs(fps - expected_fps) > 0.001 or num_frames != expected_frames:
                            raise ValueError(
                                "LTX 2.3 extension source fps and frame count must match the config"
                            )
                    else:
                        context = contract["extension"]["context_frames"]
                        window_frames, window_fps = frame_geometry(recipe)
                        if abs(fps - window_fps) > 0.001:
                            raise ValueError("LTX 2.5 extension source fps must match output fps")
                        if num_frames <= context:
                            raise ValueError(
                                "LTX 2.5 extension source must be longer than its context"
                            )
                        if window_frames - context != contract["extension"]["additional_frames"]:
                            raise ValueError(
                                "LTX 2.5 generation window minus context_frames must equal "
                                "extension.additional_frames"
                            )
                        audio_streams = [
                            value
                            for value in metadata.get("streams", [])
                            if value.get("codec_type") == "audio"
                        ]
                        if len(audio_streams) != 1:
                            raise ValueError(
                                "LTX 2.5 extension requires exactly one embedded source "
                                "audio stream"
                            )
                        audio_stream = audio_streams[0]
                        channels = int(audio_stream["channels"])
                        sample_rate = int(audio_stream["sample_rate"])
                        if channels not in {1, 2} or not 0 < sample_rate <= 192000:
                            raise ValueError(
                                "LTX 2.5 extension source audio must be mono or stereo"
                            )
                        report.update(audio_channels=channels, audio_sample_rate=sample_rate)
                else:
                    report["embedded_audio_policy"] = (
                        "explicit_soundtrack"
                        if item.get("soundtrack_path")
                        else "not_conditioned; supply soundtrack_path or a separate audio reference"
                    )
            else:
                rate, channels = int(stream["sample_rate"]), int(stream["channels"])
                if not 0 < rate <= 192000 or channels not in {1, 2}:
                    raise ValueError(f"{item['id']}: audio requires mono/stereo at up to 192 kHz")
                report.update(sample_rate=rate, channels=channels)
        if item.get("soundtrack_path"):
            audio_item = {
                "id": item["id"] + "-soundtrack",
                "kind": "audio",
                "path": item["soundtrack_path"],
            }
            report["soundtrack"] = inspect_media(recipe, {"inputs": [audio_item]})[0]
        reports.append({"id": item["id"], "kind": item["kind"], "path": item["path"], **report})
    if recipe["engine"] == "h3":
        for kind in ("video", "audio"):
            seconds = sum(r.get("duration_seconds", 0) for r in reports if r["kind"] == kind)
            if kind == "audio":
                seconds += sum(r.get("soundtrack", {}).get("duration_seconds", 0) for r in reports)
            if seconds > 15.05:
                raise ValueError(f"H3 references exceed the combined 15-second {kind} budget")
    return reports


def read_media(recipe, item, report):
    import numpy as np

    if item["kind"] == "image":
        from PIL import Image

        with Image.open(item["path"]) as image:
            return np.asarray(image.convert("RGB"), dtype=np.uint8)[None]
    command = [media_binary(recipe, "ffmpeg"), "-v", "error", "-nostdin", "-i", item["path"]]
    if item["kind"] == "audio":
        rate = 32000 if recipe["engine"] == "h3" else report["sample_rate"]
        command += [
            "-map",
            "0:a:0",
            "-t",
            str(report["duration_seconds"]),
            "-ac",
            "2",
            "-ar",
            str(rate),
            "-f",
            "f32le",
            "pipe:1",
        ]
        raw = subprocess.run(command, check=True, capture_output=True, timeout=60).stdout
        waveform = np.frombuffer(raw, dtype="<f4").reshape(-1, 2).T.copy()[None]
        if not waveform.size or not np.isfinite(waveform).all():
            raise ValueError(f"{item['id']}: decoded audio is empty or non-finite")
        return {"waveform": waveform, "sample_rate": rate}
    maximum_frames = math.ceil(report["duration_seconds"] * report["fps"]) + 1
    command += [
        "-map",
        "0:v:0",
        "-an",
        "-frames:v",
        str(maximum_frames),
        "-fps_mode",
        "passthrough",
        "-pix_fmt",
        "rgb24",
        "-f",
        "rawvideo",
        "pipe:1",
    ]
    raw = subprocess.run(command, check=True, capture_output=True, timeout=60).stdout
    frames = np.frombuffer(raw, dtype=np.uint8).reshape(-1, report["height"], report["width"], 3)
    if not len(frames):
        raise ValueError(f"{item['id']}: video decoded no frames")
    return frames


def read_embedded_video_audio(recipe, item, report, *, sample_rate=32000):
    """Decode one video's embedded soundtrack as channel-first float PCM."""

    import numpy as np

    command = [
        media_binary(recipe, "ffmpeg"),
        "-v",
        "error",
        "-nostdin",
        "-i",
        item["path"],
        "-map",
        "0:a:0",
        "-ac",
        "2",
        "-ar",
        str(sample_rate),
        "-f",
        "f32le",
        "pipe:1",
    ]
    raw = subprocess.run(command, check=True, capture_output=True, timeout=60).stdout
    waveform = np.frombuffer(raw, dtype="<f4").reshape(-1, 2).T.copy()
    if not waveform.size or not np.isfinite(waveform).all():
        raise ValueError(f"{item['id']}: decoded embedded audio is empty or non-finite")
    expected_minimum = round(report["duration_seconds"] * sample_rate * 0.95)
    if waveform.shape[1] < expected_minimum:
        raise ValueError(f"{item['id']}: embedded audio is unexpectedly shorter than video")
    return waveform


def ltx_conditioning_kwargs(recipe, contract, reports):
    """Translate an already validated contract into consumed runtime arguments."""
    frames, _fps = frame_geometry(recipe)
    kwargs = {}
    images = []
    references = []
    msr_references = []
    for item, report in zip(contract["inputs"], reports, strict=True):
        if item["role"] == "keyframe":
            images.append(
                {
                    "path": item["path"],
                    "frame_index": item["frame_index"],
                    "strength": item["strength"],
                }
            )
        elif item["role"] == "audio_driver":
            if recipe["engine"] == "ltx23":
                kwargs["audio_path"] = item["path"]
            else:
                kwargs["audio_reference"] = read_media(recipe, item, report)
        elif item["role"] == "reference" and contract["task"] == "extension":
            extension_input = {"path": item["path"], **contract["extension"]}
            if recipe["engine"] == "ltx25":
                context = contract["extension"]["context_frames"]
                video = read_media(recipe, item, report)
                audio = read_embedded_video_audio(recipe, item, report, sample_rate=16000)
                context_samples = round(context / report["fps"] * 16000)
                if audio.shape[1] < context_samples:
                    raise ValueError("LTX 2.5 extension source audio is shorter than its context")
                extension_input.update(
                    video=video[-context:],
                    audio={
                        "waveform": audio[:, -context_samples:][None],
                        "sample_rate": 16000,
                    },
                )
            kwargs["extension_input"] = extension_input
        elif (
            item["role"] == "reference"
            and recipe["engine"] == "ltx25"
            and contract["task"] == "ref2va"
        ):
            import numpy as np

            msr_references.append(
                {
                    "image": read_media(recipe, item, report).astype(np.float32) / 255.0,
                    "role": item["reference_role"],
                    "description": item["description"],
                    "strength": item["strength"],
                    "attention_strength": item["attention_strength"],
                    "reference_frames": item["reference_frames"],
                    "reference_size_policy": item["reference_size_policy"],
                    "reference_priority": item["reference_priority"],
                }
            )
        elif item["role"] == "control":
            if recipe["engine"] == "ltx23":
                value = {key: item[key] for key in ("path", "kind", "strength", "control_type")}
                if item["kind"] == "image":
                    kwargs["reference_sheet_input"] = value
                else:
                    kwargs.setdefault("control_inputs", []).append(value)
                continue
            import numpy as np

            references.append(
                {
                    "images": read_media(recipe, item, report).astype(np.float32) / 255.0,
                    "start_frame": 0,
                    "end_frame": frames - 1,
                    "strength": item["strength"],
                    "attention_strength": 1.0,
                    "mask": None,
                    "control_type": item["control_type"],
                    "reference_size_policy": "quality",
                    "reference_role": "",
                }
            )
        else:
            raise ValueError(f"No LTX transport for role {item['role']}")
    if images:
        kwargs["image_inputs"] = images
    if references:
        kwargs["video_references"] = references
    if msr_references:
        kwargs["msr_references"] = msr_references
    return kwargs


def materialize_ingredients_video(item, config, directory, *, ffmpeg=None):
    """Create the full-length, black-padded static guide required by Ingredients."""
    binary = ffmpeg or shutil.which("ffmpeg")
    if not binary:
        raise FileNotFoundError("Ingredients requires ffmpeg")
    target = Path(directory) / "ingredients-reference.mp4"
    fit = (
        f"scale={config.width}:{config.height}:force_original_aspect_ratio=decrease,"
        f"pad={config.width}:{config.height}:(ow-iw)/2:(oh-ih)/2:black"
    )
    subprocess.run(
        [
            str(binary),
            "-v",
            "error",
            "-nostdin",
            "-y",
            "-loop",
            "1",
            "-i",
            item["path"],
            "-vf",
            fit,
            "-r",
            str(config.frame_rate),
            "-frames:v",
            str(config.num_frames),
            "-pix_fmt",
            "yuv420p",
            str(target),
        ],
        check=True,
        timeout=120,
    )
    # Preserve semantic kind=image for the task validator; path is now the
    # private static-video transport consumed by ICLoraPipeline.
    return {**item, "path": str(target)}
