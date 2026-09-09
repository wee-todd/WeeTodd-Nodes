import copy
import importlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
remote_bridge = importlib.import_module("studio_drawthings")


def request():
    return {
        "drawThingsRequest": {
            "schema": "weetodd-drawthings-request-v1",
            "requestID": "image-fixture",
            "operation": "image",
            "profileID": "local",
            "modelID": "fixture",
            "prompt": "Red cube",
            "negativePrompt": "",
            "configuration": {"width": 64, "height": 64, "steps": 4, "seed": 42},
            "inputs": [],
            "loras": [],
            "billingPolicy": "freeOnly",
        },
        "scope": "clip",
        "owner": "clip-a",
        "name": "Cube",
    }


class Adapter:
    def __init__(self):
        self.calls = []

    def generate(self, request, output_directory, cancelled):
        self.calls.append((request, output_directory))
        yield {
            "type": "result",
            "requestID": request["requestID"],
            "value": {
                "fingerprint": "fixture-hash",
                "normalizedRequest": request,
                "media": {
                    "imagePaths": [str(output_directory / "frames/00000000.png")],
                    "width": 64,
                    "height": 64,
                },
            },
        }


def test_image_result_uses_captured_owner_without_mutating_timeline(tmp_path):
    project = {"clips": [{"id": "clip-a"}, {"id": "clip-b"}], "assets": []}
    before = copy.deepcopy(project)
    adapter = Adapter()
    result = remote_bridge.bridge_generate_image(
        request(), project, adapter=adapter, output=tmp_path
    )
    assert project == before
    assert result["asset"]["owner"] == "clip-a"
    assert result["asset"]["scope"] == "clip"
    assert result["fingerprint"] == "fixture-hash"
    assert adapter.calls[0][0]["operation"] == "image"


@pytest.mark.parametrize("change", [{"owner": "missing"}, {"scope": "unsupported"}])
def test_invalid_destination_rejected_before_generation(tmp_path, change):
    job = request() | change
    adapter = Adapter()
    with pytest.raises(ValueError):
        remote_bridge.bridge_generate_image(
            job, {"clips": [{"id": "clip-a"}]}, adapter=adapter, output=tmp_path
        )
    assert not adapter.calls


def test_image_action_cannot_submit_video(tmp_path):
    job = request()
    job["drawThingsRequest"]["operation"] = "video"
    adapter = Adapter()
    with pytest.raises(ValueError, match="image"):
        remote_bridge.bridge_generate_image(
            job, {"clips": [{"id": "clip-a"}]}, adapter=adapter, output=tmp_path
        )
    assert not adapter.calls
