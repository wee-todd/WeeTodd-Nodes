"""Explicit distilled T2V must not silently use Dev weights or hidden guidance."""

import json
from dataclasses import replace
from pathlib import Path

import pytest
from test_model_setup import ltx23_components

from ltx23_mlx.runtime import LTX23GenerationConfig, LTX23ModelSpec, LTX23RuntimeCache
from wee_todd_mlx.generation_selection import generation_descriptor, resolve_generation_selection
from wee_todd_mlx.model_setup import prepare_recipe
from wee_todd_nodes.ltx_nodes import WeeToddLTX23GenerationConfig

MODE = "distilled_single_stage"


def single_bundle(root):
    components = ltx23_components(root)
    root = Path(components["model_dir"])
    (root / "transformer-distilled.safetensors").rename(
        root / "transformer-distilled-1.1.safetensors"
    )
    for path in root.glob("spatial_upscaler*"):
        path.unlink()
    return components


def test_setup_single_pass_uses_versioned_checkpoint_without_dev_or_upscaler(tmp_path):
    components = single_bundle(tmp_path)
    result = prepare_recipe("ltx23-text-single-pass", components, tmp_path / "recipes")
    recipe = json.loads(Path(result["recipePath"]).read_text())
    config = recipe["config"]
    assert config["pipeline_mode"] == MODE
    assert (config["stage1_steps"], config["stage2_steps"]) == (8, 0)
    assert (config["cfg_scale"], config["stg_scale"], config["shift"]) == (1, 0, 5)
    assert config["low_memory"] and config["low_ram_streaming"]
    spec = LTX23ModelSpec(**components)
    spec.validate(MODE)
    inventory = spec.inventory(MODE)
    names = {name for entry in inventory["components"] for name in entry["files"]}
    assert "transformer-distilled-1.1.safetensors" in names
    assert not any("dev" in n or "upscaler" in n for n in names)
    (Path(components["model_dir"]) / "transformer-distilled-1.1.safetensors").rename(
        Path(components["model_dir"]) / "transformer-dev.safetensors"
    )
    with pytest.raises(FileNotFoundError, match="distilled-1.1"):
        spec.validate(MODE)


def test_node_and_studio_controls_keep_real_step_count_and_fixed_guidance(tmp_path):
    config, _ = WeeToddLTX23GenerationConfig().configure(
        MODE, 768, 448, 4.8, 25, 123, 0, 0, 3, 1, True, True
    )
    assert (config.stage1_steps, config.stage2_steps, config.cfg_scale, config.stg_scale) == (
        8,
        0,
        1,
        0,
    )
    components = single_bundle(tmp_path)
    result = prepare_recipe("ltx23-text-single-pass", components, tmp_path / "recipes")
    recipe = json.loads(Path(result["recipePath"]).read_text())
    descriptor = generation_descriptor(recipe)
    assert descriptor["supportedTasks"] == ["t2v"]
    controls = descriptor["controls"]
    assert controls["evaluations"] == 8 and controls["refinementSteps"] is None
    assert controls["stepsEditable"] and controls["shiftEditable"]
    assert not controls["cfgEditable"]
    # Exercise the same selection resolver used by Studio and exported jobs.
    resolved = resolve_generation_selection(
        {"task": "t2v", "steps": 10, "shift": 4},
        {"engine": "ltx23", "profileID": "test"},
        [{"id": "test", "recipe": recipe}],
        {},
    )
    assert resolved["recipe"]["config"]["stage1_steps"] == 10
    assert resolved["recipe"]["config"]["shift"] == 4


@pytest.mark.parametrize(
    "changes",
    [
        {"cfg_scale": 3},
        {"stg_scale": 1},
        {"stage2_steps": 3},
        {"shift": float("nan")},
        {"shift": 0},
    ],
)
def test_single_pass_rejects_hidden_guidance_refinement_and_invalid_shift(changes):
    config = LTX23GenerationConfig(
        pipeline_mode=MODE, stage1_steps=8, stage2_steps=0, cfg_scale=1, stg_scale=0
    )
    with pytest.raises(ValueError):
        replace(config, **changes).validate()


def test_single_pass_rejects_conditioning_before_loading_weights(tmp_path):
    config = LTX23GenerationConfig(
        pipeline_mode=MODE, stage1_steps=8, stage2_steps=0, cfg_scale=1, stg_scale=0
    )
    runtime = LTX23RuntimeCache()
    with pytest.raises(ValueError, match="text.to.video"):
        runtime.generate_to_file(
            LTX23ModelSpec("missing"), config, "test", tmp_path / "out.mp4", image_path="image.png"
        )
    assert not runtime.loaded


@pytest.mark.parametrize("outcome", ["success", "failure", "cancel"])
def test_shared_sampler_counts_evaluations_and_stages_release(tmp_path, monkeypatch, outcome):
    import mlx.core as mx

    from ltx23_mlx.runtime import _comfy_sampler_progress
    from ltx23_mlx.single_stage import LTX23SingleStageDistilledPipeline, trailing_sigmas

    assert trailing_sigmas(8, 5) == pytest.approx(
        [1, 0.9722222222, 0.9375, 0.8928571429, 0.8333333333, 0.75, 0.625, 0.4166666667, 0]
    )
    pipeline = LTX23SingleStageDistilledPipeline(str(tmp_path), low_memory=True)
    events, steps = [], []

    class TinyTransformer:
        def __call__(self, video_latent, audio_latent, **kwargs):
            events.append("evaluation")
            if outcome == "failure":
                raise RuntimeError("sampling failed")
            return mx.zeros_like(video_latent), mx.zeros_like(audio_latent)

    def encode(prompt):
        events.append("encode:" + prompt)
        return mx.zeros((1, 2, 4)), mx.zeros((1, 2, 4))

    def load_transformer(path):
        assert path.name == "transformer-distilled-1.1.safetensors"
        assert events[-1] == "free_text"
        events.append("load_transformer")
        return TinyTransformer()

    def decode(video, audio, output_path, *, frame_rate):
        assert pipeline.dit is None
        assert video.shape == (1, 128, 2, 2, 2)
        assert audio.ndim == 4
        events.append("decode")
        return output_path

    monkeypatch.setattr(pipeline, "_load_text_encoder", lambda: None)
    monkeypatch.setattr(pipeline, "_encode_text", encode)
    monkeypatch.setattr(pipeline.prompt_encoder, "free", lambda: events.append("free_text"))
    monkeypatch.setattr(pipeline, "_load_transformer_with_optional_streaming", load_transformer)
    monkeypatch.setattr(pipeline, "_decode_and_save_video", decode)

    def interrupted():
        if outcome == "cancel":
            raise KeyboardInterrupt()

    def generate():
        with _comfy_sampler_progress(
            interrupted, lambda done, total: steps.append((done, total)), 8
        ):
            return pipeline.generate_and_save(
                "positive", str(tmp_path / "out.mp4"), 64, 64, 9, frame_rate=24
            )

    if outcome == "success":
        generate()
        assert events.count("evaluation") == 8
        assert steps == [(i, 8) for i in range(1, 9)]
        assert events.count("encode:positive") == 1
        assert "decode" in events
    else:
        with pytest.raises(RuntimeError if outcome == "failure" else KeyboardInterrupt):
            generate()
        assert "decode" not in events
    assert pipeline.dit is None
    assert pipeline.vae_decoder is None and pipeline.audio_decoder is None


def test_single_pass_lora_validation_uses_selected_11_weights(tmp_path):
    import mlx.core as mx
    from test_ltx23_lora import save_adapter

    from ltx23_mlx.lora import validate_stack

    adapter = save_adapter(tmp_path / "adapter.safetensors", mx.ones((2, 4)), mx.ones((6, 2)))
    mx.save_safetensors(
        str(tmp_path / "transformer-distilled-1.1.safetensors"), {"proj.weight": mx.ones((6, 4))}
    )
    # A co-located Dev transformer with an incompatible projection cannot redirect validation.
    mx.save_safetensors(
        str(tmp_path / "transformer-dev.safetensors"), {"proj.weight": mx.ones((3, 4))}
    )
    report = validate_stack((adapter,), tmp_path, MODE)
    assert report[0]["pairs"][0]["logical_shape"] == [6, 4]
