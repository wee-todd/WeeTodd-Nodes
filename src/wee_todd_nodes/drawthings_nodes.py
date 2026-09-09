"""Lightweight ComfyUI wrappers for the shared Draw Things adapter."""

from __future__ import annotations

import json
import os
import re
import shutil
import time
import uuid
from pathlib import Path
from typing import Any

PROFILE_ID = "comfy-drawthings"
_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _adapter_class():
    from wee_todd_remote.adapter import DrawThingsAdapter

    return DrawThingsAdapter


def _finish_video(manifest, output, ffmpeg, cancelled):
    from wee_todd_remote.media import finish_video

    return finish_video(
        manifest, output=output, ffmpeg=Path(ffmpeg), cancelled=cancelled
    )


def _throw_if_interrupted() -> None:
    try:
        from comfy.model_management import throw_exception_if_processing_interrupted
    except ImportError:
        return
    throw_exception_if_processing_interrupted()


def _cancelled() -> bool:
    _throw_if_interrupted()
    return False


def _output_root() -> Path:
    try:
        import folder_paths

        root = Path(folder_paths.get_output_directory())
    except ImportError:
        root = Path.cwd() / "output"
    return root.resolve() / "WeeTodd" / "DrawThings"


def _video_local_paths(ffmpeg_path: str, filename_prefix: str) -> tuple[Path, Path]:
    prefix = Path(filename_prefix)
    if not filename_prefix.strip() or prefix.is_absolute() or ".." in prefix.parts:
        raise ValueError("filename_prefix must be a non-empty relative path")
    executable_value = ffmpeg_path.strip() or "ffmpeg"
    executable = shutil.which(executable_value)
    if executable is None:
        raise ValueError("ffmpeg must name an available executable")
    executable_path = Path(executable).resolve(strict=True)
    if not executable_path.is_file() or not os.access(executable_path, os.X_OK):
        raise ValueError("ffmpeg must name an available executable")

    comfy_output = _output_root().parents[1].resolve()
    movie = comfy_output / prefix.with_name(f"{prefix.name}-{uuid.uuid4().hex}.mp4")
    movie.parent.mkdir(parents=True, exist_ok=True)
    resolved_movie = movie.resolve(strict=False)
    if not resolved_movie.is_relative_to(comfy_output):
        raise ValueError("filename_prefix must remain confined to the ComfyUI output directory")
    return executable_path, resolved_movie


def _adapter(connection: dict[str, Any]):
    return _adapter_class()(
        helper=connection["helper"],
        profiles={PROFILE_ID: connection["profile"]},
        credential_provider=connection["credential_provider"],
        now=time.time,
    )


def _request_for_connection(request: dict[str, Any]) -> dict[str, Any]:
    value = dict(request)
    value["profileID"] = PROFILE_ID
    return value


def _json_object(raw: str, name: str) -> dict[str, Any]:
    try:
        value = json.loads(raw or "{}")
    except json.JSONDecodeError as error:
        raise ValueError(f"{name} must be valid JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _json_array(raw: str, name: str) -> list[Any]:
    try:
        value = json.loads(raw or "[]")
    except json.JSONDecodeError as error:
        raise ValueError(f"{name} must be valid JSON") from error
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a JSON array")
    return value


class WeeToddDrawThingsConnection:
    """Configure a nonsecret Draw Things endpoint and runtime credential reference."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "route": (["grpc", "dtBridge", "dtCloud"],),
            "host": ("STRING", {"default": "127.0.0.1"}),
            "port": ("INT", {"default": 7859, "min": 1, "max": 65535}),
            "use_tls": ("BOOLEAN", {"default": False}),
            "helper_path": (
                "STRING",
                {"default": "integrations/drawthings-client/.build/release/WeeToddDrawThings"},
            ),
            "credential_env": ("STRING", {"default": ""}),
            "self_hosted_confirmed": ("BOOLEAN", {"default": False}),
        }}

    RETURN_TYPES = ("DT_CONNECTION",)
    FUNCTION = "connect"
    CATEGORY = "WeeTodd/Draw Things"
    DESCRIPTION = (
        "Configure route, endpoint, helper, and an environment-variable credential reference; "
        "no connection occurs here."
    )

    def connect(
        self, route, host, port, use_tls, helper_path, credential_env, self_hosted_confirmed
    ):
        from wee_todd_remote.profiles import DrawThingsProfile

        credential_env = credential_env.strip()
        if credential_env and not _ENV_NAME.fullmatch(credential_env):
            raise ValueError("credential_env must be an environment variable name, never a secret")
        profile = DrawThingsProfile(
            id=PROFILE_ID, name="ComfyUI Draw Things", route=route, host=host, port=port,
            useTLS=use_tls, credentialRef=credential_env or None,
            selfHostedConfirmed=self_hosted_confirmed,
        )

        def credentials(current_profile):
            name = current_profile.credentialRef
            if not name:
                return {}
            value = os.environ.get(name)
            if not value:
                raise RuntimeError(f"Draw Things credential environment variable {name} is unset")
            key = "apiKey" if current_profile.route == "dtCloud" else "sharedSecret"
            return {key: value}

        return (
            {
                "profile": profile,
                "helper": Path(helper_path),
                "credential_provider": credentials,
            },
        )


class WeeToddDrawThingsDiscover:
    """Explicitly query the endpoint catalog; copy exact model IDs into Request."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"connection": ("DT_CONNECTION",)}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("catalog_json",)
    OUTPUT_NODE = True
    FUNCTION = "discover"
    CATEGORY = "WeeTodd/Draw Things"
    DESCRIPTION = (
        "Explicitly refresh the server catalog without generating media; INPUT_TYPES never "
        "contacts the endpoint."
    )

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def discover(self, connection):
        report = json.dumps(
            _adapter(connection).discover(PROFILE_ID), indent=2, sort_keys=True
        )
        return {"ui": {"text": [report]}, "result": (report,)}


class WeeToddDrawThingsRequest:
    """Build the canonical portable Draw Things request used by every host."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "operation": (["image", "video"],),
            "model_id": ("STRING", {"default": "paste-exact-server-model-id"}),
            "prompt": ("STRING", {"multiline": True, "default": ""}),
            "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
            "width": ("INT", {"default": 512, "min": 64}),
            "height": ("INT", {"default": 512, "min": 64}),
            "steps": ("INT", {"default": 4, "min": 1}),
            "seed": ("INT", {"default": 0, "min": 0}),
            "num_frames": ("INT", {"default": 1, "min": 1}),
            "fps": ("FLOAT", {"default": 24.0, "min": 1.0}),
            "settings_json": ("STRING", {"multiline": True, "default": "{}"}),
            "inputs_json": ("STRING", {"multiline": True, "default": "[]"}),
            "loras_json": ("STRING", {"multiline": True, "default": "[]"}),
        }}

    RETURN_TYPES = ("DT_REQUEST", "STRING")
    RETURN_NAMES = ("request", "canonical_json")
    FUNCTION = "build"
    CATEGORY = "WeeTodd/Draw Things"
    DESCRIPTION = (
        "Build a free-only canonical request; model IDs must be copied exactly from explicit "
        "discovery."
    )

    def build(self, operation, model_id, prompt, negative_prompt, width, height, steps, seed,
              num_frames, fps, settings_json, inputs_json, loras_json):
        from wee_todd_remote.contracts import validate_request

        configuration = {"width": width, "height": height, "steps": steps, "seed": seed}
        configuration.update(_json_object(settings_json, "settings_json"))
        if operation == "video":
            configuration.update(numFrames=num_frames, fps=fps)
        request = validate_request({
            "schema": "weetodd-drawthings-request-v1", "requestID": uuid.uuid4().hex,
            "operation": operation, "profileID": PROFILE_ID, "modelID": model_id,
            "prompt": prompt, "negativePrompt": negative_prompt, "configuration": configuration,
            "inputs": _json_array(inputs_json, "inputs_json"),
            "loras": _json_array(loras_json, "loras_json"), "billingPolicy": "freeOnly",
        })
        return request, json.dumps(request, sort_keys=True, separators=(",", ":"))


class WeeToddDrawThingsEstimate:
    """Refresh capability, account policy, and CU without submitting generation."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"connection": ("DT_CONNECTION",), "request": ("DT_REQUEST",)}}

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("estimate_cu", "limit_cu", "status", "report_json")
    OUTPUT_NODE = True
    FUNCTION = "estimate"
    CATEGORY = "WeeTodd/Draw Things"
    DESCRIPTION = (
        "Estimate CU and refresh eligibility only. Use the estimate-only workflow to avoid "
        "generation."
    )

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def estimate(self, connection, request):
        result = _adapter(connection).prepare(_request_for_connection(request))
        estimate = result.get("estimateCU")
        limit = result.get("limitCU")
        estimate_text = "unknown" if estimate is None else str(estimate)
        limit_text = "unknown" if limit is None else str(limit)
        report = json.dumps(result, indent=2, sort_keys=True)
        return {"ui": {"text": [report]}, "result": (
            estimate_text,
            limit_text,
            str(result.get("eligibility", "unknown")),
            report,
        )}


def _result_event(adapter, request, output):
    result = None
    for event in adapter.generate(_request_for_connection(request), output, _cancelled):
        if event.get("type") == "result":
            result = event.get("value")
    if not isinstance(result, dict) or not isinstance(result.get("media"), dict):
        raise RuntimeError("Draw Things generation returned no media result")
    return result["media"]


class WeeToddDrawThingsGenerateImage:
    """Generate and materialize one image tensor plus its durable file path."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"connection": ("DT_CONNECTION",), "request": ("DT_REQUEST",)}}

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "asset_path")
    FUNCTION = "generate"
    CATEGORY = "WeeTodd/Draw Things"
    DESCRIPTION = (
        "Generate an image through the shared adapter and load only the returned image into an "
        "IMAGE tensor."
    )

    def generate(self, connection, request):
        if request.get("operation") != "image":
            raise ValueError("Generate Image requires an image request")
        output = _output_root() / f"image-{uuid.uuid4().hex}"
        media = _result_event(_adapter(connection), request, output)
        path = Path(media["imagePaths"][0])
        import numpy as np
        import torch
        from PIL import Image

        with Image.open(path) as image:
            array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        return torch.from_numpy(array).unsqueeze(0), str(path)


class WeeToddDrawThingsGenerateVideo:
    """Generate synchronized media and return files without loading all frames into memory."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "connection": ("DT_CONNECTION",), "request": ("DT_REQUEST",),
            "ffmpeg_path": ("STRING", {"default": "ffmpeg"}),
            "filename_prefix": ("STRING", {"default": "WeeTodd/DrawThings"}),
        }}

    RETURN_TYPES = ("STRING", "FLOAT", "STRING", "STRING")
    RETURN_NAMES = ("movie_path", "fps", "audio_path", "frames_directory")
    OUTPUT_NODE = True
    FUNCTION = "generate"
    CATEGORY = "WeeTodd/Draw Things"
    DESCRIPTION = (
        "Generate video and finish it to a file; frames stay on disk for low-memory graphs."
    )

    def generate(self, connection, request, ffmpeg_path, filename_prefix):
        if request.get("operation") != "video":
            raise ValueError("Generate Video requires a video request")
        executable, movie = _video_local_paths(ffmpeg_path, filename_prefix)
        output = _output_root() / f"video-{uuid.uuid4().hex}"
        media = _result_event(_adapter(connection), request, output)
        movie = _finish_video(media, movie, executable, _cancelled)
        fps = media["fpsNumerator"] / media["fpsDenominator"]
        return (
            str(movie),
            float(fps),
            str(media.get("audioPath", "")),
            str(media["framesDirectory"]),
        )


NODE_CLASS_MAPPINGS = {
    "WeeToddDrawThingsConnection": WeeToddDrawThingsConnection,
    "WeeToddDrawThingsDiscover": WeeToddDrawThingsDiscover,
    "WeeToddDrawThingsRequest": WeeToddDrawThingsRequest,
    "WeeToddDrawThingsEstimate": WeeToddDrawThingsEstimate,
    "WeeToddDrawThingsGenerateImage": WeeToddDrawThingsGenerateImage,
    "WeeToddDrawThingsGenerateVideo": WeeToddDrawThingsGenerateVideo,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    key: key.replace("WeeToddDrawThings", "WeeTodd Draw Things ") for key in NODE_CLASS_MAPPINGS
}
