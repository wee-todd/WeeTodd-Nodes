"""Lightweight ComfyUI input contracts for H3 keyframes and Ref2VA references."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

REFERENCE_PIXEL_BUDGET_MIN = 50
REFERENCE_PIXEL_BUDGET_MAX = 400
REFERENCE_CANVAS_MULTIPLE = 32


def resolve_reference_image_canvas(
    source_width: int,
    source_height: int,
    target_width: int,
    target_height: int,
    pixel_budget_percent: int,
) -> tuple[int, int]:
    """Preserve source aspect while scaling its area relative to the output canvas."""
    values = (source_width, source_height, target_width, target_height)
    if any(value < 1 for value in values):
        raise ValueError("Reference and target image dimensions must be positive.")
    if not REFERENCE_PIXEL_BUDGET_MIN <= pixel_budget_percent <= REFERENCE_PIXEL_BUDGET_MAX:
        raise ValueError(
            "Reference image pixel budget must be between "
            f"{REFERENCE_PIXEL_BUDGET_MIN}% and {REFERENCE_PIXEL_BUDGET_MAX}%."
        )
    aspect = source_width / source_height
    if not 0.25 <= aspect <= 4.0:
        raise ValueError("H3 reference image aspect ratio must be between 1:4 and 4:1.")
    pixel_budget = target_width * target_height * pixel_budget_percent / 100.0
    # Current ComfyUI Ref2VA conditioning is explicitly down-only. Enlarging a
    # small identity sheet creates pixels but also creates persistent reference
    # tokens that ride through every transformer evaluation.
    scale = min(1.0, math.sqrt(pixel_budget / (source_width * source_height)))
    width = source_width * scale
    height = source_height * scale
    multiple = REFERENCE_CANVAS_MULTIPLE
    resolved_width = max(multiple, round(width / multiple) * multiple)
    resolved_height = max(multiple, round(height / multiple) * multiple)
    return resolved_width, resolved_height


def resolve_reference_video_canvas(
    source_width: int,
    source_height: int,
    target_width: int,
    target_height: int,
    size_mode: str,
) -> tuple[int, int]:
    """Resolve a down-only output-matched or native H3 reference-video canvas."""
    if size_mode == "match output (recommended)":
        scale = min(
            1.0,
            math.sqrt((target_width * target_height) / (source_width * source_height)),
        )
        multiple = REFERENCE_CANVAS_MULTIPLE
        return (
            max(multiple, round(source_height * scale / multiple) * multiple),
            max(multiple, round(source_width * scale / multiple) * multiple),
        )
    if size_mode != "native H3 reference canvas (high detail / slow)":
        raise ValueError(f"Unknown H3 reference-video size mode: {size_mode!r}.")
    from minimax_h3_mlx.packing import resolve_canvas_size

    target_height, target_width = resolve_canvas_size(source_width, source_height)
    if source_width * source_height < target_width * target_height:
        multiple = REFERENCE_CANVAS_MULTIPLE
        target_width = max(multiple, round(source_width / multiple) * multiple)
        target_height = max(multiple, round(source_height / multiple) * multiple)
    return target_height, target_width


def _shape(value: Any, subject: str) -> tuple[int, ...]:
    shape = getattr(value, "shape", None)
    if shape is None:
        raise ValueError(f"{subject} must be a ComfyUI tensor.")
    return tuple(int(dimension) for dimension in shape)


def _validate_image(value: Any, subject: str, *, video: bool = False) -> tuple[int, ...]:
    shape = _shape(value, subject)
    if len(shape) != 4 or shape[-1] < 3:
        raise ValueError(
            f"{subject} must have ComfyUI IMAGE shape (frames, height, width, channels)."
        )
    if shape[0] < (5 if video else 1) or shape[1] < 1 or shape[2] < 1:
        minimum = "at least five frames" if video else "one image"
        raise ValueError(f"{subject} must contain {minimum} with positive dimensions.")
    return shape


def _validate_audio(value: Any, subject: str) -> tuple[int, int, int, int]:
    if not isinstance(value, dict):
        raise ValueError(f"{subject} must be a ComfyUI AUDIO value.")
    waveform = value.get("waveform")
    sample_rate = value.get("sample_rate")
    shape = _shape(waveform, f"{subject} waveform")
    if len(shape) != 3 or shape[0] < 1 or shape[1] not in {1, 2} or shape[2] < 1:
        raise ValueError(
            f"{subject} waveform must have shape (batch, mono-or-stereo channels, samples)."
        )
    if not isinstance(sample_rate, int) or sample_rate < 1:
        raise ValueError(f"{subject} sample rate must be a positive integer.")
    return shape[0], shape[1], shape[2], sample_rate


def comfy_image_to_pil(value: Any, subject: str):
    """Convert the first image of a ComfyUI IMAGE batch at the explicit host boundary."""
    _validate_image(value, subject)
    import numpy as np
    from PIL import Image

    frame = value[0]
    detach = getattr(frame, "detach", None)
    if detach is not None:
        frame = detach()
    cpu = getattr(frame, "cpu", None)
    if cpu is not None:
        frame = cpu()
    pixels = np.asarray(frame)
    if pixels.dtype != np.uint8:
        pixels = np.rint(np.clip(pixels, 0.0, 1.0) * 255.0).astype(np.uint8)
    return Image.fromarray(pixels[..., :3], mode="RGB")


def _comfy_frames_to_uint8(value: Any, subject: str):
    """Convert one ComfyUI IMAGE batch to channels-last uint8 frames."""
    _validate_image(value, subject, video=True)
    import numpy as np

    frames = value
    detach = getattr(frames, "detach", None)
    if detach is not None:
        frames = detach()
    cpu = getattr(frames, "cpu", None)
    if cpu is not None:
        frames = cpu()
    frames = np.asarray(frames)
    if frames.dtype != np.uint8:
        frames = np.rint(np.clip(frames, 0.0, 1.0) * 255.0).astype(np.uint8)
    return np.ascontiguousarray(frames[..., :3])


def _comfy_audio_to_numpy(value: Any, subject: str):
    """Convert the first ComfyUI audio batch to stereo float32 samples."""
    import numpy as np

    _, channels, _, sample_rate = _validate_audio(value, subject)
    waveform = value["waveform"][0]
    detach = getattr(waveform, "detach", None)
    if detach is not None:
        waveform = detach()
    cpu = getattr(waveform, "cpu", None)
    if cpu is not None:
        waveform = cpu()
    waveform = np.asarray(waveform, dtype=np.float32)
    if channels == 1:
        waveform = np.repeat(waveform, 2, axis=0)
    return np.ascontiguousarray(waveform), sample_rate


def _resample_waveform(waveform, source_rate: int, target_rate: int):
    """Resample a short reference waveform at the host boundary."""
    if source_rate == target_rate:
        return waveform
    try:
        import torch
        import torchaudio.functional

        result = torchaudio.functional.resample(
            torch.from_numpy(waveform), source_rate, target_rate
        )
        return result.numpy()
    except (ImportError, OSError) as error:
        raise ImportError(
            "Ref2VA audio resampling requires torchaudio. Supply 32 kHz reference audio or "
            "install a torchaudio build compatible with the ComfyUI Python environment."
        ) from error


@dataclass(frozen=True)
class H3KeyframeConditioning:
    """Optional first and last frames for the FL2VA checkpoint."""

    first_frame: Any | None = None
    last_frame: Any | None = None

    def validate(self) -> None:
        if self.first_frame is None and self.last_frame is None:
            raise ValueError(
                "H3 keyframe conditioning requires a first frame, a last frame, or both."
            )
        if self.first_frame is not None:
            _validate_image(self.first_frame, "H3 first frame")
        if self.last_frame is not None:
            _validate_image(self.last_frame, "H3 last frame")

    @property
    def anchors(self) -> tuple[str, ...]:
        return tuple(
            anchor
            for anchor, value in (
                ("first", self.first_frame),
                ("last", self.last_frame),
            )
            if value is not None
        )

    def metadata(self) -> dict[str, object]:
        self.validate()
        return {
            "task": "fl2va",
            "anchors": list(self.anchors),
            "first_frame_shape": (
                list(_shape(self.first_frame, "H3 first frame"))
                if self.first_frame is not None
                else None
            ),
            "last_frame_shape": (
                list(_shape(self.last_frame, "H3 last frame"))
                if self.last_frame is not None
                else None
            ),
        }

    def images(self) -> list[Any]:
        self.validate()
        return [
            comfy_image_to_pil(value, f"H3 {anchor} frame")
            for anchor, value in (
                ("first", self.first_frame),
                ("last", self.last_frame),
            )
            if value is not None
        ]


@dataclass(frozen=True)
class H3TimedKeyframe:
    """One image anchored to a requested local or chained-timeline timestamp."""

    image: Any
    timestamp_seconds: float
    global_timestamp_seconds: float | None = None

    def validate(self) -> None:
        _validate_image(self.image, "H3 timed keyframe")
        if self.timestamp_seconds < 0:
            raise ValueError("A timed H3 keyframe timestamp cannot be negative.")


@dataclass(frozen=True)
class H3TimedKeyframeStack:
    """Ordered, composable timed keyframes for one FL2VA generation window."""

    keyframes: tuple[H3TimedKeyframe, ...] = ()

    def append(self, keyframe: H3TimedKeyframe) -> H3TimedKeyframeStack:
        keyframe.validate()
        return H3TimedKeyframeStack((*self.keyframes, keyframe))

    def _resolved(self, num_frames: int, fps: int) -> list[tuple[int, H3TimedKeyframe]]:
        if not self.keyframes:
            raise ValueError("Timed FL2VA conditioning requires at least one keyframe.")
        if len(self.keyframes) > 8:
            raise ValueError("Timed FL2VA supports at most eight keyframes per window.")
        resolved = []
        for keyframe in self.keyframes:
            keyframe.validate()
            frame = round(keyframe.timestamp_seconds * fps)
            if not 0 <= frame < num_frames:
                raise ValueError(
                    f"Timed keyframe at {keyframe.timestamp_seconds:g}s resolves to frame "
                    f"{frame}, outside this window's 0..{num_frames - 1} frame range."
                )
            resolved.append((frame, keyframe))
        resolved.sort(key=lambda item: item[0])
        frames = [item[0] for item in resolved]
        if len(set(frames)) != len(frames):
            raise ValueError("Two timed keyframes resolve to the same 24 fps frame.")
        return resolved

    def resolve(self, num_frames: int, fps: int = 24) -> tuple[tuple[int, ...], list[Any]]:
        resolved = self._resolved(num_frames, fps)
        frames = [item[0] for item in resolved]
        images = [comfy_image_to_pil(item[1].image, "H3 timed keyframe") for item in resolved]
        return tuple(frames), images

    def metadata(self, num_frames: int | None = None, fps: int = 24) -> dict[str, object]:
        payload: dict[str, object] = {
            "task": "fl2va",
            "keyframes": [
                {
                    "local_timestamp_seconds": item.timestamp_seconds,
                    "global_timestamp_seconds": item.global_timestamp_seconds,
                    "shape": list(_shape(item.image, "H3 timed keyframe")),
                }
                for item in self.keyframes
            ],
        }
        if num_frames is not None:
            anchors = tuple(item[0] for item in self._resolved(num_frames, fps))
            payload["resolved_frames"] = list(anchors)
            payload["rope_times"] = [frame * (5.0 / 3.0) for frame in anchors]
        return payload


@dataclass(frozen=True)
class H3ReferenceInput:
    """One ordered image, video, or audio reference for Ref2VA."""

    kind: Literal["image", "video", "audio"]
    media: Any
    fps: float | None = None
    soundtrack: Any | None = None
    image_pixel_budget_percent: int = 100
    video_size_mode: str = "match output (recommended)"
    temporal_density: str = "all frames (recommended)"
    target_frame: int | None = None

    def validate(self) -> None:
        if self.kind == "image":
            _validate_image(self.media, "H3 reference image")
            if self.soundtrack is not None:
                raise ValueError("An H3 image reference cannot carry a soundtrack.")
            if not (
                REFERENCE_PIXEL_BUDGET_MIN
                <= self.image_pixel_budget_percent
                <= REFERENCE_PIXEL_BUDGET_MAX
            ):
                raise ValueError(
                    "H3 reference image pixel budget must be between "
                    f"{REFERENCE_PIXEL_BUDGET_MIN}% and {REFERENCE_PIXEL_BUDGET_MAX}%."
                )
            return
        if self.kind == "video":
            _validate_image(self.media, "H3 reference video", video=True)
            if self.fps is None or self.fps <= 0:
                raise ValueError("H3 reference video fps must be positive.")
            if self.soundtrack is not None:
                _validate_audio(self.soundtrack, "H3 reference video soundtrack")
            if self.video_size_mode not in {
                "match output (recommended)",
                "native H3 reference canvas (high detail / slow)",
            }:
                raise ValueError(f"Unknown H3 reference-video size mode: {self.video_size_mode!r}.")
            if self.temporal_density not in {
                "all frames (recommended)",
                "automatic (conservative, experimental)",
                "uniform 50% (experimental)",
                "uniform 25% (experimental)",
            }:
                raise ValueError(
                    f"Unknown H3 reference-video temporal density: {self.temporal_density!r}."
                )
            return
        if self.kind == "audio":
            _validate_audio(self.media, "H3 reference audio")
            if self.soundtrack is not None:
                raise ValueError(
                    "A standalone H3 audio reference cannot carry a second soundtrack."
                )
            return
        raise ValueError("H3 reference kind must be image, video, or audio.")

    @property
    def has_audio(self) -> bool:
        return self.kind == "audio" or self.soundtrack is not None

    def metadata(self) -> dict[str, object]:
        self.validate()
        payload: dict[str, object] = {
            "kind": self.kind,
            "has_audio": self.has_audio,
            "target_frame": self.target_frame,
        }
        if self.kind == "image":
            payload["shape"] = list(_shape(self.media, "H3 reference image"))
            payload["image_pixel_budget_percent"] = self.image_pixel_budget_percent
        elif self.kind == "video":
            payload["shape"] = list(_shape(self.media, "H3 reference video"))
            payload["fps"] = self.fps
            payload["video_size_mode"] = self.video_size_mode
            payload["temporal_density"] = self.temporal_density
        if self.has_audio:
            audio = self.media if self.kind == "audio" else self.soundtrack
            _, channels, samples, sample_rate = _validate_audio(audio, "H3 reference audio")
            payload["audio_channels"] = channels
            payload["audio_samples"] = samples
            payload["audio_sample_rate"] = sample_rate
        return payload


@dataclass(frozen=True)
class H3ReferenceStack:
    """Ordered Ref2VA reference inputs; order controls labels and rotary placement."""

    references: tuple[H3ReferenceInput, ...] = ()

    def append(self, reference: H3ReferenceInput) -> H3ReferenceStack:
        reference.validate()
        result = H3ReferenceStack(self.references + (reference,))
        result._validate_limits()
        return result

    def _validate_limits(self) -> None:
        if len(self.references) > 12:
            raise ValueError("Ref2VA supports at most 12 references.")
        images = sum(reference.kind == "image" for reference in self.references)
        videos = sum(reference.kind == "video" for reference in self.references)
        audios = sum(reference.has_audio for reference in self.references)
        if images > 9:
            raise ValueError("Ref2VA supports at most 9 image references.")
        if videos > 3:
            raise ValueError("Ref2VA supports at most 3 video references.")
        if audios > 3:
            raise ValueError("Ref2VA supports at most 3 audio references, including soundtracks.")

    def validate_request(self) -> None:
        self._validate_limits()
        if not self.references:
            raise ValueError("Ref2VA requires at least one reference.")
        for reference in self.references:
            reference.validate()
        has_visual_reference = any(
            reference.kind in {"image", "video"} for reference in self.references
        )
        all_references_are_timed = all(
            reference.target_frame is not None for reference in self.references
        )
        if not has_visual_reference and not all_references_are_timed:
            raise ValueError(
                "Untimed Ref2VA audio references require at least one image or video reference."
            )

    def metadata(self) -> dict[str, object]:
        self._validate_limits()
        counts = {"image": 0, "video": 0, "audio": 0}
        items = []
        for position, reference in enumerate(self.references, start=1):
            item = reference.metadata()
            labels = []
            if reference.has_audio:
                counts["audio"] += 1
                labels.append(f"<Audio {counts['audio']}>")
            if reference.kind in {"image", "video"}:
                counts[reference.kind] += 1
                noun = "Picture" if reference.kind == "image" else "Video"
                labels.append(f"<{noun} {counts[reference.kind]}>")
            item.update({"position": position, "prompt_labels": labels})
            items.append(item)
        return {"task": "ref2va", "count": len(items), "references": items}

    def prepare(
        self,
        *,
        target_width: int,
        target_height: int,
        target_num_frames: int,
        target_sample_rate: int = 32000,
    ):
        """Prepare ordered in-memory media for staged Ref2VA encoders."""
        import numpy as np
        from PIL import Image

        from minimax_h3_mlx.ref2va import (
            PreparedReference,
            reduce_reference_video_frames,
            resample_reference_frames,
            resolve_reference_video_density,
            sample_reference_video_frames,
            trim_reference_num_frames,
        )

        self.validate_request()
        prepared = []
        for index, reference in enumerate(self.references, start=1):
            item = PreparedReference(kind=reference.kind)
            if reference.target_frame is not None:
                resolved_frame = (
                    reference.target_frame
                    if reference.target_frame >= 0
                    else target_num_frames + reference.target_frame
                )
                if not 0 <= resolved_frame < target_num_frames:
                    raise ValueError(
                        f"H3 timeline guide {index} frame {reference.target_frame} resolves "
                        f"outside 0..{target_num_frames - 1}."
                    )
                item.target_frame = resolved_frame
            if reference.kind == "image":
                image = comfy_image_to_pil(reference.media, f"H3 reference image {index}")
                width, height = resolve_reference_image_canvas(
                    image.width,
                    image.height,
                    target_width,
                    target_height,
                    reference.image_pixel_budget_percent,
                )
                item.image = image.resize((width, height), Image.Resampling.LANCZOS)
            elif reference.kind == "video":
                frames = _comfy_frames_to_uint8(reference.media, f"H3 reference video {index}")
                frames = resample_reference_frames(frames, float(reference.fps))
                if frames.shape[0] < 5:
                    raise ValueError(
                        f"H3 reference video {index} has fewer than five frames at 24 fps."
                    )
                remaining_frames = target_num_frames - (item.target_frame or 0)
                frames = frames[:remaining_frames]
                if frames.shape[0] < 5:
                    raise ValueError(
                        f"H3 timeline video guide {index} has fewer than five frames before "
                        "the end of the target window."
                    )
                # Use one aligned sequence for Qwen timestamps and the video
                # VAE. Previously Qwen could inspect tail frames that the VAE
                # silently discarded later.
                frames = frames[: trim_reference_num_frames(int(frames.shape[0]))]
                height, width = resolve_reference_video_canvas(
                    frames.shape[2],
                    frames.shape[1],
                    target_width,
                    target_height,
                    reference.video_size_mode,
                )
                if frames.shape[1:3] != (height, width):
                    frames = np.stack(
                        [
                            np.asarray(
                                Image.fromarray(frame).resize(
                                    (width, height), Image.Resampling.LANCZOS
                                )
                            )
                            for frame in frames
                        ]
                    )
                item.qwen_frames = frames
                density_policy = {
                    "all frames (recommended)": "full",
                    "automatic (conservative, experimental)": "automatic",
                    "uniform 50% (experimental)": "half",
                    "uniform 25% (experimental)": "quarter",
                }[reference.temporal_density]
                decision = resolve_reference_video_density(frames, density_policy)
                item.frames, item.source_num_latent_frames = reduce_reference_video_frames(
                    frames, decision.density
                )
                item.temporal_density_requested = density_policy
                item.temporal_density_resolved = decision.density
                item.temporal_activity_mean = decision.activity_mean
                item.temporal_activity_p95 = decision.activity_p95
                item.temporal_density_reason = decision.reason
                _, item.block_timestamps = sample_reference_video_frames(frames)
            audio = reference.media if reference.kind == "audio" else reference.soundtrack
            if audio is not None:
                waveform, sample_rate = _comfy_audio_to_numpy(audio, f"H3 reference audio {index}")
                # Truncate at the source rate, then resample once. This avoids
                # filtering an unused tail and matches the reference pipeline.
                remaining_frames = target_num_frames - (item.target_frame or 0)
                source_max_samples = round(remaining_frames / 24.0 * sample_rate)
                waveform = waveform[:, :source_max_samples]
                item.waveform = np.ascontiguousarray(
                    _resample_waveform(waveform, sample_rate, target_sample_rate)[
                        :, : round(remaining_frames / 24.0 * target_sample_rate)
                    ],
                    dtype=np.float32,
                )
            prepared.append(item)
        return prepared
