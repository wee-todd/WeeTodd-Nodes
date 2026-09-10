import hashlib
import math
import subprocess
import sys
import time
from pathlib import Path

import pytest
from PIL import Image

from wee_todd_remote.studio import compose_drawthings_request, render_drawthings_clip


def project():
    return {
        "settings": {"fps": 24},
        "clips": [
            {
                "id": "clip-uuid",
                "engine": "drawThings",
                "prompt": "A paper bird flies",
                "negativePrompt": "blur",
                "duration": 5,
                "seed": 7,
                "generationWidth": 768,
                "generationHeight": 448,
                "attachments": [],
                "extensionDirection": "",
                "extensionSource": "",
                "drawThings": {
                    "profileID": "connection-1",
                    "modelID": "ltx-model",
                    "modelFamily": "ltx2.3",
                    "configuration": {},
                },
            }
        ],
    }


def test_compose_builds_video_request_with_ltx_frame_clock():
    result = compose_drawthings_request(project(), "clip-uuid", [], request_id="request-1")
    assert result["operation"] == "video"
    assert result["configuration"] == {
        "width": 768,
        "height": 448,
        "seed": 7,
        "fps": 24,
        "numFrames": 121,
        "steps": 8,
        "guidanceScale": 1,
    }
    assert result["profileID"] == "connection-1"
    assert result["modelID"] == "ltx-model"


def test_explicit_remote_settings_preserved_but_clip_geometry_and_seed_win():
    value = project()
    value["clips"][0]["drawThings"]["configuration"] = {
        "width": 64,
        "height": 64,
        "seed": 99,
        "fps": 25,
        "numFrames": 81,
        "steps": 12,
        "guidanceScale": 2.5,
        "strength": 0.8,
        "shift": 1.2,
        "sampler": 4,
    }
    result = compose_drawthings_request(value, "clip-uuid", [], request_id="request-1")
    assert result["configuration"] == {
        "width": 768,
        "height": 448,
        "seed": 7,
        "fps": 25,
        "numFrames": 81,
        "steps": 12,
        "guidanceScale": 2.5,
        "strength": 0.8,
        "shift": 1.2,
        "sampler": 4,
    }


def test_fps_inherits_clip_override_then_project():
    value = project()
    value["clips"][0]["settingsOverride"] = {"fps": 30}
    assert (
        compose_drawthings_request(value, "clip-uuid", [], request_id="x")["configuration"]["fps"]
        == 30
    )
    value["clips"][0]["drawThings"]["configuration"]["fps"] = 20
    assert (
        compose_drawthings_request(value, "clip-uuid", [], request_id="x")["configuration"]["fps"]
        == 20
    )


@pytest.mark.parametrize("fps", [23.976, math.nan, math.inf, True, "24"])
def test_fps_must_be_finite_integer_without_rounding(fps):
    value = project()
    value["settings"]["fps"] = fps
    with pytest.raises(ValueError, match="fps"):
        compose_drawthings_request(value, "clip-uuid", [], request_id="x")


@pytest.mark.parametrize("duration", [0, -1, math.nan, math.inf, True])
def test_duration_must_be_positive_and_finite(duration):
    value = project()
    value["clips"][0]["duration"] = duration
    with pytest.raises(ValueError, match="duration"):
        compose_drawthings_request(value, "clip-uuid", [], request_id="x")


def test_dimensions_use_remote_64_grid_not_native_32_grid():
    value = project()
    value["clips"][0]["generationWidth"] = 736
    with pytest.raises(ValueError, match="64"):
        compose_drawthings_request(value, "clip-uuid", [], request_id="x")


@pytest.mark.parametrize("change", [{"engine": "ltx23"}, {"drawThings": None}])
def test_native_or_unselected_clip_is_rejected(change):
    value = project()
    value["clips"][0].update(change)
    with pytest.raises(ValueError, match="Draw Things|selection"):
        compose_drawthings_request(value, "clip-uuid", [], request_id="x")


@pytest.mark.parametrize("field", ["profileID", "modelID"])
def test_missing_connection_or_model_is_actionable(field):
    value = project()
    value["clips"][0]["drawThings"][field] = ""
    with pytest.raises(ValueError, match=field):
        compose_drawthings_request(value, "clip-uuid", [], request_id="x")


@pytest.mark.parametrize("unsupported", ["attachment", "extension"])
def test_unsupported_conditioning_is_refused_without_fallback(unsupported):
    value = project()
    if unsupported == "attachment":
        value["clips"][0]["attachments"] = [{"role": "last", "assetID": "asset-1"}]
    else:
        value["clips"][0]["extensionDirection"] = "after"
    with pytest.raises(ValueError, match="attachment|extension"):
        compose_drawthings_request(value, "clip-uuid", [], request_id="x")


def test_compose_includes_verified_first_frame_and_server_loras(tmp_path):
    image = tmp_path / "first.png"
    image.write_bytes(b"first-frame")
    value = project()
    value["clips"][0]["attachments"] = [{"role": "first", "assetID": "asset-1"}]
    value["clips"][0]["drawThings"]["loras"] = [{"modelID": "style-a", "weight": 0.75}]
    result = compose_drawthings_request(
        value,
        "clip-uuid",
        [{"id": "asset-1", "kind": "image", "path": str(image)}],
        request_id="x",
    )
    assert result["inputs"][0]["sha256"] == hashlib.sha256(b"first-frame").hexdigest()
    assert result["inputs"][0]["path"] == str(image.resolve())
    assert result["loras"] == [{"modelID": "style-a", "weight": 0.75}]


def test_active_motion_fidelity_is_rejected_before_remote_generation():
    value = project()
    value["clips"][0]["motionFidelity"] = {"enabled": True}
    with pytest.raises(ValueError, match="motion fidelity"):
        compose_drawthings_request(value, "clip-uuid", [], request_id="x")


def test_disabled_motion_history_and_finishing_guides_do_not_block_composition():
    value = project()
    value["clips"][0].update(
        motionFidelity={"enabled": False},
        motionResult={"score": 0.8},
        motionRecipeID="prior-recipe",
        motionPrompt="prior prompt",
        depthDirectory="depth-for-finishing",
        motionDirectory="motion-for-finishing",
    )
    result = compose_drawthings_request(value, "clip-uuid", [], request_id="x")
    assert result["operation"] == "video"


def test_sdk_ltx2_3_family_uses_ltx_frame_clock():
    value = project()
    value["clips"][0]["drawThings"]["modelFamily"] = "ltx2_3"
    assert (
        compose_drawthings_request(value, "clip-uuid", [], request_id="x")["configuration"][
            "numFrames"
        ]
        == 121
    )


def test_composed_request_is_secret_free_and_ignores_native_paths():
    value = project()
    value["clips"][0].update(profileID="native-profile", sourcePath="/local/checkpoint")
    result = compose_drawthings_request(value, "clip-uuid", [], request_id="x")
    assert "sourcePath" not in result
    assert "credentials" not in result


def test_studio_composition_imports_no_native_generation_engine():
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; sys.path.insert(0, 'src'); import wee_todd_remote.studio; "
            "print(any(name.startswith(('ltx25_mlx', 'minimax_h3_mlx', 'wee_todd_mlx')) "
            "for name in sys.modules))",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stdout.strip() == "False"


def test_render_rejects_nonvideo_request(tmp_path):
    request = compose_drawthings_request(project(), "clip-uuid", [], request_id="x")
    request["operation"] = "image"
    with pytest.raises(ValueError, match="video"):
        render_drawthings_clip(
            request, object(), tmp_path, Path("ffmpeg"), lambda: False, lambda _: None
        )


def test_render_streams_progress_and_finishes_real_tiny_video_when_available(tmp_path):
    ffmpeg = Path("/opt/homebrew/bin/ffmpeg")
    if not ffmpeg.is_file():
        pytest.skip("FFmpeg unavailable")
    output = tmp_path / "render"
    output.mkdir()
    media_root = output / "media"
    normalized = compose_drawthings_request(project(), "clip-uuid", [], request_id="request-1")
    normalized["configuration"]["numFrames"] = 9

    class Adapter:
        def generate(self, request, destination, cancelled):
            assert destination == media_root
            frames = media_root / "frames"
            frames.mkdir(parents=True)
            for index in range(9):
                Image.new("RGB", (64, 64), (index * 10, 0, 0)).save(frames / f"{index:08d}.png")
            media = {
                "operation": "video",
                "framesDirectory": str(frames),
                "fpsNumerator": 24,
                "fpsDenominator": 1,
                "frameCount": 9,
                "requiresAudio": False,
            }
            yield {"type": "progress", "requestID": "request-1", "value": {"fraction": 0.5}}
            yield {
                "type": "result",
                "requestID": "request-1",
                "value": {
                    "media": media,
                    "manifestPath": str(media_root / "manifest.json"),
                    "fingerprint": "fingerprint",
                    "normalizedRequest": normalized,
                },
            }

    progress = []
    result = render_drawthings_clip(
        normalized, Adapter(), output, ffmpeg, lambda: False, progress.append
    )
    assert Path(result["video"]).is_file()
    assert result["fingerprint"] == "fingerprint"
    assert progress[0]["type"] == "progress"


def test_render_forwards_cancellation_to_finisher_and_cleans_partial(tmp_path):
    ffmpeg = tmp_path / "slow-ffmpeg"
    ffmpeg.write_text(
        "#!/usr/bin/env python3\n"
        "import pathlib, sys, time\n"
        "pathlib.Path(sys.argv[-1]).write_bytes(b'partial')\n"
        "time.sleep(30)\n"
    )
    ffmpeg.chmod(0o755)
    output = tmp_path / "render"
    output.mkdir()
    media_root = output / "media"
    state = {"calls": 0}

    def cancelled():
        state["calls"] += 1
        return state["calls"] > 3

    class Adapter:
        def generate(self, request, destination, callback):
            assert callback is cancelled
            frames = media_root / "frames"
            frames.mkdir(parents=True)
            Image.new("RGB", (64, 64), "red").save(frames / "00000000.png")
            yield {
                "type": "result",
                "requestID": "request-1",
                "value": {
                    "media": {
                        "operation": "video",
                        "framesDirectory": str(frames),
                        "fpsNumerator": 24,
                        "fpsDenominator": 1,
                        "frameCount": 1,
                        "requiresAudio": False,
                    },
                    "manifestPath": str(media_root / "manifest.json"),
                    "fingerprint": "fingerprint",
                    "normalizedRequest": request,
                },
            }

    request_value = compose_drawthings_request(project(), "clip-uuid", [], request_id="request-1")
    with pytest.raises(InterruptedError, match="cancel"):
        render_drawthings_clip(request_value, Adapter(), output, ffmpeg, cancelled, lambda _: None)
    deadline = time.monotonic() + 2
    while list(output.glob(".*.partial*")) and time.monotonic() < deadline:
        time.sleep(0.01)
    assert not (output / "clip.mp4").exists()
    assert not list(output.glob(".*.partial*"))


@pytest.mark.parametrize("task,attachments,message", [
    ("t2v", [{"role": "first", "assetID": "image"}], "Text to video.*attachment"),
    ("i2v", [], "Image to video.*first"),
    ("i2v", [{"role": "last", "assetID": "image"}], "Image to video.*last"),
])
def test_explicit_task_conflicts_preserve_attached_media(task, attachments, message):
    import copy

    value = project()
    value["clips"][0].update(generationSelection={"task": task, "preset": "custom"},
                              attachments=attachments)
    original = copy.deepcopy(value)
    with pytest.raises(ValueError, match=message):
        compose_drawthings_request(value, "clip-uuid", [], request_id="x")
    assert value == original


@pytest.mark.parametrize("field,value", [
    ("steps", 12), ("cfg", 2.0), ("shift", 3.0), ("refinementSteps", 2),
    ("memoryPolicy", "normal"), ("projectionBackend", "auto"), ("unexpected", False),
    ("preset", "speed"), ("task", "fflf"),
])
def test_explicit_drawthings_unsupported_selection_controls_fail(field, value):
    data = project()
    selection = {"task": "t2v", "preset": "custom", field: value}
    data["clips"][0]["generationSelection"] = selection
    with pytest.raises(ValueError, match="Draw Things"):
        compose_drawthings_request(data, "clip-uuid", [], request_id="x")


def test_explicit_drawthings_image_task_keeps_remote_controls(tmp_path):
    data = project()
    image = tmp_path / "first.png"
    image.write_bytes(b"first-frame")
    data["clips"][0].update(generationSelection={"task": "i2v", "preset": "custom"},
                              attachments=[{"role": "first", "assetID": "image"}])
    data["clips"][0]["drawThings"]["configuration"] = {"steps": 12, "shift": 2.0}
    result = compose_drawthings_request(data, "clip-uuid",
                                       [{"id": "image", "kind": "image", "path": str(image)}])
    assert len(result["inputs"]) == 1
    assert result["configuration"]["steps"] == 12
    assert result["configuration"]["shift"] == 2.0
