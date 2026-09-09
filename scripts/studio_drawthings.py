"""Draw Things bridge actions shared with the native editor; imports never load weights."""

from __future__ import annotations

import os
import re
import shutil
import time
import uuid
from pathlib import Path
from typing import Any

from wee_todd_remote.adapter import DrawThingsAdapter
from wee_todd_remote.contracts import validate_request
from wee_todd_remote.profiles import DrawThingsProfile
from wee_todd_remote.studio import compose_drawthings_request, render_drawthings_clip


def adapter_for(request: dict[str, Any]) -> DrawThingsAdapter:
    connection = dict(request["connection"])
    connection["selfHostedConfirmed"] = connection.get("selfHostedConfirmed") is True
    profile = DrawThingsProfile(**connection)
    runtime = request.get("runtime", {})
    helper_value = runtime.get("drawThingsHelperPath") or os.environ.get("WEETODD_DT_HELPER")
    if not helper_value or not Path(helper_value).is_file() or not os.access(helper_value, os.X_OK):
        raise ValueError(
            "Import or build the WeeToddDrawThings transport helper in Draw Things Connections."
        )

    def credentials(selected):
        variable = "WEETODD_DT_CREDENTIAL"
        reference = selected.credentialRef or ""
        if reference.startswith("env:"):
            variable = reference[4:]
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", variable):
                raise ValueError("invalid credential environment reference")
        value = os.environ.get(variable)
        return {"apiKey" if selected.route == "dtCloud" else "sharedSecret": value} if value else {}

    return DrawThingsAdapter(
        helper=Path(helper_value),
        profiles={profile.id: profile},
        credential_provider=credentials,
        now=time.time,
    )


def bridge_generate_image(
    request: dict[str, Any],
    project: dict[str, Any],
    *,
    adapter,
    output: Path,
    progress=lambda event: None,
) -> dict[str, Any]:
    canonical = validate_request(request["drawThingsRequest"])
    if canonical["operation"] != "image":
        raise ValueError("image generation requires an image request")
    scope = request.get("scope")
    if not isinstance(scope, str) or scope not in {"global", "project", "clip"}:
        raise ValueError("select a Global, Project, or Clip asset store")
    owner = request.get("owner") if scope == "clip" else None
    if scope == "clip" and not any(clip.get("id") == owner for clip in project.get("clips", [])):
        raise ValueError("the destination clip no longer exists")
    result = None
    for event in adapter.generate(canonical, output_directory=output, cancelled=lambda: False):
        if event["type"] == "result":
            result = event["value"]
        else:
            progress(event)
    if not isinstance(result, dict):
        raise RuntimeError("Draw Things returned no completed image")
    media = result["media"]
    paths = media.get("imagePaths")
    if not isinstance(paths, list) or len(paths) != 1:
        raise ValueError("the image action requires exactly one completed image")
    asset = {
        "id": str(uuid.uuid4()),
        "name": request.get("name") or "Generated image",
        "kind": "image",
        "path": paths[0],
        "scope": scope,
        "duration": 0,
        "width": media["width"],
        "height": media["height"],
        "fps": 0,
        "thumbnail": "",
        "text": "",
    }
    if owner is not None:
        asset["owner"] = owner
    return {
        "asset": asset,
        "fingerprint": result["fingerprint"],
        "normalizedRequest": result["normalizedRequest"],
    }


def dispatch(
    command: str, request: dict[str, Any], output: Path | None, *, progress=lambda event: None
):
    adapter = adapter_for(request)
    if command == "dt-discover":
        return adapter.discover(request["connection"]["id"])
    if command == "dt-estimate":
        return adapter.prepare(request["drawThingsRequest"])
    if command in {"dt-prepare-clip", "dt-generate-clip"}:
        canonical = compose_drawthings_request(
            request["project"], request["clipID"],
            request.get("globalAssets", []) + request["project"].get("assets", []),
        )
        if canonical["profileID"] != request["connection"]["id"]:
            raise ValueError("The connection does not match the selected clip")
        if command == "dt-prepare-clip":
            return adapter.prepare(canonical)
        if output is None:
            raise ValueError("clip generation requires an output directory")
        ffmpeg = request.get("runtime", {}).get("ffmpegPath") or shutil.which("ffmpeg")
        if not ffmpeg or not os.access(ffmpeg, os.X_OK):
            raise ValueError("Connect FFmpeg in Runtime Settings before generating a clip")
        return render_drawthings_clip(
            canonical, adapter, output, Path(ffmpeg), lambda: False, progress
        )
    if command == "dt-generate-image":
        if output is None:
            raise ValueError("image generation requires an output directory")
        return bridge_generate_image(
            request, request["project"], adapter=adapter, output=output, progress=progress
        )
    raise ValueError("unsupported Draw Things action")
