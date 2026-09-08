import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]


def module():
    spec = importlib.util.spec_from_file_location(
        "prepare_reference", ROOT / "scripts/prepare_h3_reference_recipe.py"
    )
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_reference_recipe_preserves_paged_components_and_dense_schedule(tmp_path):
    recipe = module().build_recipe(tmp_path, [tmp_path / "hero.png"])
    assert recipe["engine"] == "h3"
    assert recipe["components"]["task"] == "ref2va"
    assert recipe["components"]["allow_fl2va_weights_for_ref2va"] is False
    assert recipe["components"]["transformer"] == str(
        tmp_path / "MiniMax-H3/transformers/ref2va-q8-extended-paged"
    )
    assert recipe["components"]["text_encoder"].endswith("/q8-vision-paged")
    assert recipe["config"]["steps"] == 20
    assert recipe["config"]["projection_backend"] == "mlx"
    assert recipe["config"]["width"] == 640
    assert recipe["config"]["height"] == 384
    assert recipe["config"]["memory_mode"] == "low_memory_bf16"
    assert recipe["reference_images"] == [str(tmp_path / "hero.png")]
    assert recipe["format"] == "weetodd-headless-v2"
    assert recipe["ffmpeg"] == "ffmpeg"
    assert recipe["candidate"] == "h3-reference-q8-paged"
    assert recipe.get("loras") is None


def test_ui_and_api_reference_profile_match():
    api = json.loads((ROOT / "examples/h3_ref2va_q8_paged_api.json").read_text())
    ui = json.loads(
        (ROOT / "workflows/performance/ref2va/h3_ref2va_q8_paged_experimental.json").read_text()
    )
    loader = next(n for n in ui["nodes"] if n["type"] == "WeeToddH3ComponentLoader")
    assert module().COMPONENTS == api["1"]["inputs"]
    assert module().DEFAULT_PROMPT == api["5"]["inputs"]["prompt"]
    assert loader["widgets_values"][2:4] == [
        api["1"]["inputs"]["transformer"],
        api["1"]["inputs"]["text_encoder"],
    ]
    assert len([n for n in ui["nodes"] if n["type"] == "LoadImage"]) == 1
    assert not any(n["type"] in {"LoadVideo", "LoadAudio"} for n in ui["nodes"])


def test_studio_composes_reference_profile_without_changing_model_contract(tmp_path):
    import sys

    sys.path.insert(0, str(ROOT / "scripts"))
    import studio_bridge
    from PIL import Image

    Image.new("RGB", (64, 64)).save(tmp_path / "hero.png")

    recipe = module().build_recipe(tmp_path, [tmp_path / "old.png"])
    profile = tmp_path / "H3_Reference_Q8_Paged.json"
    profile.write_text(json.dumps(recipe))
    clip = dict(
        id="clip",
        engine="h3",
        profileID=str(profile),
        prompt=recipe["prompt"],
        generationWidth=640,
        generationHeight=384,
        seed=42,
        duration=5.0,
        attachments=[dict(id="attachment", assetID="hero", role="reference", strength=1.0)],
    )
    result, _ = studio_bridge.compose_recipe(
        dict(
            clipID="clip",
            project=dict(
                settings={},
                clips=[clip],
                assets=[dict(id="hero", kind="image", path=str(tmp_path / "hero.png"))],
            ),
            runtime=dict(
                profilesDirectory=str(tmp_path), ffmpeg="/usr/bin/true", ffprobe="/usr/bin/true"
            ),
        )
    )
    assert result["components"] == recipe["components"]
    assert result["config"]["steps"] == 20
    assert result["conditioning"]["task"] == "ref2va"
    assert result["conditioning"]["inputs"][0]["path"] == str(tmp_path / "hero.png")
    assert "reference_images" not in result


def test_recipe_preparation_works_in_packaged_runtime_without_examples(tmp_path):
    prepare = module()
    prepare.ROOT = tmp_path
    assert prepare.build_recipe(tmp_path, [])["components"]["task"] == "ref2va"
