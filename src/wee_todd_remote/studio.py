"""Pure Studio composition and shared Draw Things video finishing."""

from __future__ import annotations

import copy
import math
import uuid
from pathlib import Path
from typing import Any

from .conditioning import canonical_inputs, canonical_loras
from .contracts import validate_request
from .media import finish_video

_REMOTE_SETTINGS = frozenset(
    {"steps", "guidanceScale", "strength", "shift", "audioShift", "sampler", "numFrames", "fps"}
)
_LTX_FRAME_CLOCK_FAMILIES = frozenset({"ltx2", "ltx2.3", "ltx23", "ltx2_3"})


def _object(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def _nonempty(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be selected")
    return value


def _integer(value: Any, name: str, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer of at least {minimum}")
    return value


def _fps(value: Any) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 1
        or value != int(value)
    ):
        raise ValueError("fps must be a positive finite integer; fractional fps is not rounded")
    return int(value)


def _duration(value: Any) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError("duration must be positive and finite")
    return float(value)


def _ltx_frames(duration: float, fps: int) -> int:
    required = math.ceil(duration * fps)
    return max(1, math.ceil((required - 1) / 8) * 8 + 1)


def _validate_generation_selection(clip: dict[str, Any]) -> None:
    selection = clip.get("generationSelection")
    if selection is None:
        return
    selection = _object(selection, "Draw Things generation selection")
    numeric = {"steps", "refinementSteps", "cfg", "shift", "memoryPolicy", "projectionBackend"}
    unknown = set(selection) - {"task", "preset"} - numeric
    submitted = {key for key in numeric if selection.get(key) is not None}
    if unknown or submitted:
        name = sorted(unknown | submitted)[0]
        raise ValueError(
            f"Unsupported Draw Things generation selection control: {name}; "
            "edit the Draw Things configuration controls instead"
        )
    if selection.get("preset") != "custom":
        raise ValueError("Draw Things generation selection currently requires the Custom preset")
    task = selection.get("task")
    if task not in {"t2v", "i2v", "fflf"}:
        raise ValueError(
            "Draw Things supports Text to video, Image to video and H3 First/last frames"
        )
    attachments = clip.get("attachments", [])
    if not isinstance(attachments, list):
        raise ValueError("Draw Things attachments must be an array")
    if task == "t2v" and attachments:
        raise ValueError(
            "Draw Things Text to video conflicts with attached media; "
            "remove the attachments or select Image to video"
        )
    if task == "i2v":
        roles = [item.get("role") if isinstance(item, dict) else None for item in attachments]
        conflicts = [str(role) for role in roles if role != "first"]
        if conflicts:
            raise ValueError(
                "Draw Things Image to video conflicts with these attachment roles: "
                + ", ".join(conflicts)
            )
        if roles != ["first"]:
            raise ValueError("Draw Things Image to video requires exactly one first-frame image")
    if task == "fflf":
        roles = [item.get("role") if isinstance(item, dict) else None for item in attachments]
        if sorted(str(role) for role in roles) != ["first", "last"]:
            raise ValueError(
                "Draw Things First/last frames requires exactly one first and one last image"
            )


def compose_drawthings_request(
    project: dict[str, Any],
    clip_id: str,
    assets: Any,
    *,
    request_id: str | None = None,
) -> dict[str, Any]:
    """Compose one secret-free remote video request from an exact Studio clip.

    Resolve first-frame images and H3 first/last pairs from Studio assets.
    """
    project = _object(project, "project")
    clips = project.get("clips")
    if not isinstance(clips, list):
        raise ValueError("project.clips must be an array")
    matches = [
        clip for clip in clips if isinstance(clip, dict) and str(clip.get("id")) == str(clip_id)
    ]
    if len(matches) != 1:
        raise ValueError("clip_id must identify exactly one Studio clip")
    clip = matches[0]
    if clip.get("engine") != "drawThings":
        raise ValueError("The selected clip is not a Draw Things clip")
    selection = clip.get("drawThings")
    if not isinstance(selection, dict):
        raise ValueError("The Draw Things selection is missing")
    profile_id = _nonempty(selection.get("profileID"), "profileID")
    model_id = _nonempty(selection.get("modelID"), "modelID")
    model_family = _nonempty(selection.get("modelFamily"), "modelFamily")
    is_h3 = model_family.lower() == "minimaxh3"
    if not is_h3 and model_family.lower() not in _LTX_FRAME_CLOCK_FAMILIES:
        raise ValueError("modelFamily is not supported by the Draw Things video helper")

    _validate_generation_selection(clip)
    if clip.get("extensionDirection") or clip.get("extensionSource") or clip.get("extensionClipID"):
        raise ValueError(
            "Draw Things clip extension is not supported yet; remove extension settings"
        )
    motion_fidelity = clip.get("motionFidelity")
    if isinstance(motion_fidelity, dict) and motion_fidelity.get("enabled") is True:
        raise ValueError(
            "Draw Things motion fidelity generation is not supported yet; disable motion fidelity"
        )

    supplied = selection.get("configuration", {})
    if not isinstance(supplied, dict):
        raise ValueError("drawThings.configuration must be an object")
    unknown = set(supplied) - (_REMOTE_SETTINGS | {"width", "height", "seed"})
    if unknown:
        raise ValueError(f"Unsupported Draw Things configuration setting: {sorted(unknown)[0]}")
    configuration = {
        key: copy.deepcopy(value) for key, value in supplied.items() if key in _REMOTE_SETTINGS
    }

    width = _integer(clip.get("generationWidth"), "generationWidth", minimum=64)
    height = _integer(clip.get("generationHeight"), "generationHeight", minimum=64)
    if width % 64 or height % 64:
        raise ValueError("Draw Things generation dimensions must use the 64-pixel grid")
    seed = clip.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    configuration.update(width=width, height=height, seed=seed)

    override = clip.get("settingsOverride")
    override_fps = override.get("fps") if isinstance(override, dict) else None
    project_settings = _object(project.get("settings"), "project.settings")
    fallback_fps = override_fps if override_fps is not None else project_settings.get("fps")
    fps = _fps(configuration.get("fps", 24 if is_h3 else fallback_fps))
    configuration["fps"] = fps
    if is_h3 and fps != 24:
        raise ValueError("Draw Things H3 generates at 24 FPS; apply movie FPS during finishing")
    if not is_h3 and "audioShift" in configuration:
        raise ValueError("Draw Things Audio Shift is supported only for H3")
    duration = _duration(clip.get("duration"))
    if "numFrames" in configuration:
        frames = _integer(configuration["numFrames"], "numFrames")
        if is_h3 and (frames < 5 or (frames - 5) % 17):
            raise ValueError("Draw Things H3 numFrames must use the 17n+5 frame clock")
        if not is_h3 and (frames - 1) % 8:
            raise ValueError("numFrames must use the LTX 8n+1 frame clock")
        configuration["numFrames"] = frames
    else:
        configuration["numFrames"] = (
            max(5, math.ceil((math.ceil(duration * fps - 1e-9) - 5) / 17) * 17 + 5)
            if is_h3 else _ltx_frames(duration, fps)
        )
    configuration.setdefault("steps", 50 if is_h3 else 8)
    configuration.setdefault("guidanceScale", 1)
    if is_h3:
        configuration.setdefault("shift", 12)
        configuration.setdefault("audioShift", 3)
    inputs = canonical_inputs(clip.get("attachments", []), assets,
                              model_family=model_family, num_frames=configuration["numFrames"])

    result = {
        "schema": "weetodd-drawthings-request-v1",
        "requestID": request_id or str(uuid.uuid4()),
        "operation": "video",
        "profileID": profile_id,
        "modelID": model_id,
        "prompt": clip.get("prompt"),
        "negativePrompt": clip.get("negativePrompt", ""),
        "configuration": configuration,
        "inputs": inputs,
        "loras": canonical_loras(selection.get("loras", [])),
        "billingPolicy": "freeOnly",
    }
    return validate_request(result)


def endpoint_duration(request: dict[str, Any]) -> float | None:
    """Keep the last conditioned frame when a validated request rounds its duration."""
    if not any(item.get("role") == "last" for item in request.get("inputs", [])):
        return None
    configuration = request["configuration"]
    return _integer(configuration["numFrames"], "numFrames") / _fps(configuration["fps"])


def render_drawthings_clip(
    request: dict[str, Any],
    adapter: Any,
    output: Path,
    ffmpeg: Path,
    cancelled: Any,
    progress: Any,
) -> dict[str, Any]:
    """Render a composed remote video and finish its validated frames once."""
    request = validate_request(request)
    if request["operation"] != "video":
        raise ValueError("render_drawthings_clip requires a video request")
    output = Path(output).resolve()
    media_root = output / "media"
    video = output / "clip.mp4"
    if media_root.exists() or video.exists():
        raise ValueError("Draw Things render output must be new")
    result = None
    for event in adapter.generate(request, media_root, cancelled):
        if event.get("type") == "result":
            result = event.get("value")
        else:
            progress(event)
    if not isinstance(result, dict) or not isinstance(result.get("media"), dict):
        raise RuntimeError("Draw Things generation returned no validated video media")
    finished = finish_video(result["media"], output=video, ffmpeg=Path(ffmpeg), cancelled=cancelled)
    return {
        "video": str(finished),
        "manifestPath": result.get("manifestPath"),
        "fingerprint": result.get("fingerprint"),
        "normalizedRequest": copy.deepcopy(result.get("normalizedRequest")),
        "endpointDuration": endpoint_duration(result.get("normalizedRequest") or request),
    }
