"""Studio catalog provenance and renderer transport, without allocating model tensors."""

import importlib
import json
import math
import struct
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
lora = importlib.import_module("studio_lora")


def adapter(tmp_path, metadata, name="renamed.safetensors"):
    header = {"__metadata__": metadata}
    offset = 0
    for key, shape in {"block.lora_A.weight": [1, 2], "block.lora_B.weight": [2, 1]}.items():
        stop = offset + math.prod(shape) * 4
        header[key] = dict(dtype="F32", shape=shape, data_offsets=[offset, stop])
        offset = stop
    encoded = json.dumps(header).encode()
    source = tmp_path / name
    source.write_bytes(struct.pack("<Q", len(encoded)) + encoded + bytes(offset))
    return source


@pytest.mark.parametrize(
    "metadata,expected",
    [
        ({"model_version": "2.3.0"}, "ltx23"),
        ({"base_model": "Lightricks/LTX-2.5"}, "ltx25"),
        ({"base_model": "MiniMax-H3"}, "h3"),
        ({}, None),
    ],
)
def test_inspection_uses_metadata_not_filename(tmp_path, metadata, expected):
    result = lora.inspect_lora(adapter(tmp_path, metadata, "H3-LTX-2.5.safetensors"))
    assert result.get("loraModel") == expected
    assert result["kind"] == "lora"


def test_import_rejects_non_adapters_and_conflicting_metadata(tmp_path):
    broken = tmp_path / "model.safetensors"
    broken.write_bytes(b"not a checkpoint")
    with pytest.raises((ValueError, OSError)):
        lora.inspect_lora(broken)
    with pytest.raises(ValueError, match="conflicting"):
        lora.inspect_lora(adapter(tmp_path, {"base_model": "MiniMax-H3", "model_version": "2.5.0"}))


@pytest.mark.parametrize(
    "engine,model,accepted",
    [
        ("h3", "h3", True),
        ("h3", "ltx23", False),
        ("ltx23", "ltx25", False),
        ("ltx25", "ltx23", True),
        ("ltx25", "ltx25", True),
        ("movie", "ltx25", False),
    ],
)
def test_model_filter_rechecked_at_recipe_boundary(tmp_path, engine, model, accepted):
    source = adapter(tmp_path, {})
    asset = dict(kind="lora", path=str(source), loraModel=model)
    if accepted:
        assert lora.clip_lora(asset, {"strength": 0.7}, engine)["strength"] == 0.7
    else:
        with pytest.raises(ValueError, match="compatible"):
            lora.clip_lora(asset, {}, engine)


def test_declaration_cannot_override_checkpoint_metadata(tmp_path):
    source = adapter(tmp_path, {"model_version": "2.5.0"})
    with pytest.raises(ValueError, match="declares"):
        lora.clip_lora(dict(kind="lora", path=str(source), loraModel="h3"), {}, "h3")


@pytest.mark.parametrize("strength", [float("nan"), float("inf"), -1, 2.1, True, "no"])
def test_reject_invalid_strengths(tmp_path, strength):
    asset = dict(kind="lora", path=str(adapter(tmp_path, {})), loraModel="ltx25")
    with pytest.raises(ValueError, match="strength"):
        lora.clip_lora(asset, {"strength": strength}, "ltx25")


def recipe_request(tmp_path, engine="ltx25", embedded=False):
    from studio_bridge import compose_recipe

    first = adapter(
        tmp_path, {"model_version": "2.3.0"} if engine != "h3" else {"base_model": "MiniMax-H3"}
    )
    second = adapter(tmp_path, {}, "second.safetensors")
    base = dict(
        format="weetodd-headless-v2", engine=engine, components={}, config={"frame_rate": 24}
    )
    if engine == "h3":
        base["components"]["task"] = "t2va"
    if embedded:
        base["components"]["loras"] = [{"path": "/tmp/profile-lora.safetensors", "strength": 0.4}]
    profile = tmp_path / "recipe.json"
    profile.write_text(json.dumps(base))
    model = "h3" if engine == "h3" else "ltx23"
    clip = dict(
        id="clip",
        name="Group shot",
        engine=engine,
        profileID=str(profile),
        prompt="A cyclist rides through a sunlit street.",
        generationWidth=768,
        generationHeight=512,
        seed=42,
        duration=5,
        attachments=[
            dict(
                id="a",
                assetID="one",
                role="lora",
                strength=0.65,
                loraGroupName="Mixed",
                loraGroupID="group",
            ),
            dict(
                id="b",
                assetID="two",
                role="lora",
                strength=1.2,
                loraGroupName="Mixed",
                loraGroupID="group",
            ),
        ],
    )
    assets = [
        dict(id="one", kind="lora", path=str(first), loraModel=model),
        dict(
            id="two",
            kind="lora",
            path=str(second),
            loraModel="ltx25" if engine == "ltx25" else model,
        ),
    ]
    request = dict(
        clipID="clip",
        project=dict(settings={}, clips=[clip], assets=assets),
        runtime=dict(
            profilesDirectory=str(tmp_path), ffmpegPath="/usr/bin/true", ffprobePath="/usr/bin/true"
        ),
    )
    return request, compose_recipe


@pytest.mark.parametrize(
    "engine,embedded", [("h3", False), ("ltx23", False), ("ltx23", True), ("ltx25", False)]
)
def test_composed_groups_use_existing_renderer_transport(tmp_path, engine, embedded):
    request, compose = recipe_request(tmp_path, engine, embedded)
    result, report = compose(request)
    stack = (
        result["components"]["loras"]
        if engine == "ltx25" or embedded
        else result["loras"]["adapters"]
    )
    strengths = (
        [item[1] for item in stack] if engine == "ltx25" else [item["strength"] for item in stack]
    )
    assert strengths == ([0.4] if embedded else []) + [0.65, 1.2]
    assert report["task"] == "t2v"
    assert result["conditioning"]["inputs"] == []
    if embedded:
        assert "loras" not in result


def test_duplicate_clip_files_rejected_before_render(tmp_path):
    request, compose = recipe_request(tmp_path)
    request["project"]["assets"][1].update(request["project"]["assets"][0], id="two")
    with pytest.raises(ValueError, match="only once"):
        compose(request)


@pytest.mark.parametrize("clip_only", [False, True])
def test_exported_jobs_embed_group_strengths_without_library_dependency(tmp_path, clip_only):
    import studio_job

    request, _ = recipe_request(tmp_path)
    request.update(generateIDs=["clip"], clipOnly=clip_only)
    target = tmp_path / "export.weetodd-job.json"
    studio_job.export_job(request, target)
    job = json.loads(target.read_text())
    assert job["scope"] == ("clip" if clip_only else "movie")
    assert [entry[1] for entry in job["recipes"]["clip"]["recipe"]["components"]["loras"]] == [
        0.65,
        1.2,
    ]
    request["project"]["clips"][0]["attachments"][0]["strength"] = 1.9
    assert job["project"]["clips"][0]["attachments"][0]["strength"] == 0.65
    assert "loraGroups" not in job
