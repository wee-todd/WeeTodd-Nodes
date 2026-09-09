from __future__ import annotations

import importlib
import json
import shutil
import struct
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image


def test_import_is_passive(monkeypatch):
    before = set(sys.modules)
    module = importlib.import_module("wee_todd_nodes.drawthings_nodes")
    assert "mlx.core" not in set(sys.modules) - before
    assert module.NODE_CLASS_MAPPINGS["WeeToddDrawThingsConnection"]
    helper_default = module.WeeToddDrawThingsConnection.INPUT_TYPES()["required"][
        "helper_path"
    ][1]["default"]
    assert helper_default.endswith("/.build/release/WeeToddDrawThings")


def test_connection_rejects_secret_values_and_builds_profile(monkeypatch, tmp_path):
    from wee_todd_nodes.drawthings_nodes import WeeToddDrawThingsConnection

    monkeypatch.setenv("DRAW_THINGS_TOKEN", "private-value")
    connection = WeeToddDrawThingsConnection().connect(
        "grpc", "127.0.0.1", 7859, False, str(tmp_path / "helper"),
        "DRAW_THINGS_TOKEN", True,
    )[0]
    assert connection["profile"].credentialRef == "DRAW_THINGS_TOKEN"
    assert connection["credential_provider"](connection["profile"]) == {
        "sharedSecret": "private-value"
    }
    cloud = WeeToddDrawThingsConnection().connect(
        "dtCloud", "example.invalid", 443, True, str(tmp_path / "helper"),
        "DRAW_THINGS_TOKEN", False,
    )[0]
    assert cloud["credential_provider"](cloud["profile"]) == {"apiKey": "private-value"}
    assert "private-value" not in repr(connection)
    with pytest.raises(ValueError, match="environment variable name"):
        WeeToddDrawThingsConnection().connect(
            "grpc", "localhost", 7859, False, str(tmp_path / "helper"), "actual-secret!", True
        )


def test_request_emits_canonical_json():
    from wee_todd_nodes.drawthings_nodes import WeeToddDrawThingsRequest

    request, raw = WeeToddDrawThingsRequest().build(
        "image", "exact-model-id", "a red cube", "", 64, 48, 4, 42, 1, 24.0,
        '{"sampler":"Euler A"}', "[]", "[]",
    )
    assert request == json.loads(raw)
    assert request["configuration"] == {
        "width": 64, "height": 48, "steps": 4, "seed": 42, "sampler": "Euler A"
    }
    assert request["profileID"] == "comfy-drawthings"


class _Adapter:
    prepare_result = {}
    events = []
    cancelled = None

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def discover(self, profile_id):
        return {"files": ["exact-model-id"], "profileID": profile_id}

    def prepare(self, request):
        return self.prepare_result

    def generate(self, request, output_directory, cancelled):
        type(self).cancelled = cancelled
        yield from self.events


def _connection(tmp_path):
    from wee_todd_nodes.drawthings_nodes import WeeToddDrawThingsConnection

    return WeeToddDrawThingsConnection().connect(
        "grpc", "localhost", 7859, False, str(tmp_path / "helper"), "", True
    )[0]


def test_discovery_and_estimate_use_shared_adapter(monkeypatch, tmp_path):
    import wee_todd_nodes.drawthings_nodes as nodes

    monkeypatch.setattr(nodes, "_adapter_class", lambda: _Adapter)
    connection = _connection(tmp_path)
    discovery = nodes.WeeToddDrawThingsDiscover().discover(connection)
    assert "exact-model-id" in discovery["result"][0]
    assert discovery["ui"]["text"]

    _Adapter.prepare_result = {
        "estimateCU": 99, "limitCU": 1000, "eligibility": "allowed", "issues": []
    }
    estimate = nodes.WeeToddDrawThingsEstimate().estimate(connection, {"profileID": "ignored"})
    assert estimate["result"][:3] == ("99", "1000", "allowed")
    _Adapter.prepare_result = {
        "estimateCU": 1000, "limitCU": 1000, "eligibility": "blocked",
        "issues": [{"message": "free CU limit reached"}],
    }
    blocked = nodes.WeeToddDrawThingsEstimate().estimate(connection, {})
    assert blocked["result"][:3] == ("1000", "1000", "blocked")
    assert "free CU limit reached" in blocked["ui"]["text"][0]

    _Adapter.prepare_result = {
        "estimateCU": None, "limitCU": None, "eligibility": "unknown", "issues": []
    }
    unknown = nodes.WeeToddDrawThingsEstimate().estimate(connection, {})
    assert unknown["result"][:3] == ("unknown", "unknown", "unknown")


def test_generate_image_loads_result_and_bridges_cancellation(monkeypatch, tmp_path):
    import wee_todd_nodes.drawthings_nodes as nodes

    monkeypatch.setattr(nodes, "_adapter_class", lambda: _Adapter)
    monkeypatch.setattr(nodes, "_output_root", lambda: tmp_path)
    image_path = tmp_path / "result.png"
    Image.new("RGB", (3, 2), "red").save(image_path)
    _Adapter.events = [{"type": "result", "value": {"media": {"imagePaths": [str(image_path)]}}}]
    monkeypatch.setattr(nodes, "_throw_if_interrupted", lambda: None)

    class Tensor:
        def __init__(self, value):
            self.value = value

        def unsqueeze(self, axis):
            return Tensor(np.expand_dims(self.value, axis))

        @property
        def shape(self):
            return self.value.shape

        def __array__(self, dtype=None):
            return np.asarray(self.value, dtype=dtype)

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(from_numpy=Tensor))
    tensor, path = nodes.WeeToddDrawThingsGenerateImage().generate(
        _connection(tmp_path), {"operation": "image"}
    )
    assert tuple(tensor.shape) == (1, 2, 3, 3)
    assert np.asarray(tensor)[0, 0, 0, 0] == pytest.approx(1.0)
    assert path == str(image_path)
    _Adapter.cancelled()


def test_generate_video_validates_local_paths_before_remote_submission(monkeypatch, tmp_path):
    import wee_todd_nodes.drawthings_nodes as nodes

    monkeypatch.setattr(nodes, "_adapter_class", lambda: _Adapter)
    monkeypatch.setattr(nodes, "_output_root", lambda: tmp_path)
    calls = []
    monkeypatch.setattr(
        _Adapter, "generate", lambda *args, **kwargs: calls.append(args) or iter(())
    )
    with pytest.raises(ValueError, match="filename_prefix"):
        nodes.WeeToddDrawThingsGenerateVideo().generate(
            _connection(tmp_path), {"operation": "video"}, "ffmpeg", "../escape"
        )
    with pytest.raises(ValueError, match="ffmpeg"):
        nodes.WeeToddDrawThingsGenerateVideo().generate(
            _connection(tmp_path), {"operation": "video"}, "missing-ffmpeg", "safe/name"
        )
    assert calls == []


def test_generate_video_rejects_symlink_output_escape_before_submission(
    monkeypatch, tmp_path
):
    import wee_todd_nodes.drawthings_nodes as nodes

    output = tmp_path / "output" / "WeeTodd" / "DrawThings"
    output.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    (tmp_path / "output" / "escape").symlink_to(outside, target_is_directory=True)
    monkeypatch.setattr(nodes, "_output_root", lambda: output)
    calls = []
    monkeypatch.setattr(
        _Adapter, "generate", lambda *args, **kwargs: calls.append(args) or iter(())
    )
    with pytest.raises(ValueError, match="confined"):
        nodes.WeeToddDrawThingsGenerateVideo().generate(
            _connection(tmp_path), {"operation": "video"}, "ffmpeg", "escape/movie"
        )
    assert calls == []


def test_generate_video_uses_real_keyword_only_ffmpeg_finish(monkeypatch, tmp_path):
    import wee_todd_nodes.drawthings_nodes as nodes
    from wee_todd_remote.media import validate_media

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        pytest.skip("ffmpeg unavailable")
    frames = tmp_path / "frames"
    frames.mkdir()
    for index in range(2):
        Image.new("RGB", (16, 16), (index * 100, 0, 0)).save(frames / f"{index:08d}.png")
    samples = [0.0] * 800
    pcm = struct.pack(f"<{len(samples)}f", *samples)
    wav = tmp_path / "audio.wav"
    wav.write_bytes(
        b"RIFF" + struct.pack("<I", 36 + len(pcm)) + b"WAVEfmt "
        + struct.pack("<IHHIIHH", 16, 3, 1, 8000, 32000, 4, 32)
        + b"data" + struct.pack("<I", len(pcm)) + pcm
    )
    media = validate_media(
        {"schema": "weetodd-drawthings-media-v1", "requestID": "r", "operation": "video",
         "frameCount": 2, "configuration": {"width": 16, "height": 16},
         "framesDirectory": "frames", "fpsNumerator": 20, "fpsDenominator": 1,
         "requiresAudio": True, "audioPath": "audio.wav", "sampleRate": 8000,
         "sampleCount": 800, "channels": 1},
        root=tmp_path, expected_request_id="r", expected_operation="video", expected_frames=2,
        requires_audio=True,
    )
    output_root = tmp_path / "output" / "WeeTodd" / "DrawThings"
    monkeypatch.setattr(nodes, "_output_root", lambda: output_root)
    monkeypatch.setattr(nodes, "_adapter_class", lambda: _Adapter)
    _Adapter.events = [{"type": "result", "value": {"media": media}}]
    movie, fps, audio_path, frames_path = nodes.WeeToddDrawThingsGenerateVideo().generate(
        _connection(tmp_path), {"operation": "video"}, ffmpeg, "WeeTodd/tiny-av"
    )
    assert Path(movie).is_file() and Path(movie).stat().st_size > 0
    assert fps == 20.0
    assert audio_path == str(wav)
    assert frames_path == str(frames)


def test_catalog_registers_drawthings_nodes():
    from wee_todd_nodes.nodes import NODE_CLASS_MAPPINGS

    expected = {
        "WeeToddDrawThingsConnection", "WeeToddDrawThingsDiscover", "WeeToddDrawThingsRequest",
        "WeeToddDrawThingsEstimate", "WeeToddDrawThingsGenerateImage",
        "WeeToddDrawThingsGenerateVideo",
    }
    assert expected <= NODE_CLASS_MAPPINGS.keys()
    assert NODE_CLASS_MAPPINGS["WeeToddDrawThingsDiscover"].OUTPUT_NODE is True
    assert NODE_CLASS_MAPPINGS["WeeToddDrawThingsEstimate"].OUTPUT_NODE is True
    assert str(NODE_CLASS_MAPPINGS["WeeToddDrawThingsDiscover"].IS_CHANGED()) == "nan"
    assert str(NODE_CLASS_MAPPINGS["WeeToddDrawThingsEstimate"].IS_CHANGED()) == "nan"


def test_shipped_image_workflows_connect_generation_to_save_image_output():
    root = Path(__file__).resolve().parents[1]
    api = json.loads((root / "examples/drawthings_image_api.json").read_text())
    ui = json.loads((root / "workflows/balance/t2v/drawthings_image.json").read_text())
    # ComfyUI's built-in SaveImage is an OUTPUT_NODE; an unconsumed image return
    # does not make the generator itself an execution terminal.
    terminals = [(key, value) for key, value in api.items() if value["class_type"] == "SaveImage"]
    assert len(terminals) == 1, "The image API prompt must have a SaveImage output node"
    terminal_id, terminal = terminals[0]
    generator_id, output_slot = terminal["inputs"]["images"]
    assert api[generator_id]["class_type"] == "WeeToddDrawThingsGenerateImage"
    assert output_slot == 0
    prefix = terminal["inputs"]["filename_prefix"]
    assert prefix and not Path(prefix).is_absolute() and ".." not in Path(prefix).parts

    nodes = {str(node["id"]): node for node in ui["nodes"]}
    saved = nodes[terminal_id]
    assert saved["type"] == "SaveImage"
    assert saved["mode"] == 0
    assert saved["widgets_values"] == [prefix]
    image_socket = next(item for item in saved["inputs"] if item["name"] == "images")
    link_id = image_socket["link"]
    assert image_socket["type"] == "IMAGE"
    assert [link_id, int(generator_id), 0, int(terminal_id), 0, "IMAGE"] in ui["links"]
    assert link_id in nodes[generator_id]["outputs"][0]["links"]
