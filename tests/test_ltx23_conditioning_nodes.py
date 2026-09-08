from pathlib import Path

import numpy as np
import pytest

from ltx23_mlx.ic_lora import LTX23ICLoRASpec
from ltx23_mlx.runtime import LTX23GenerationConfig, LTX23ModelSpec
from wee_todd_nodes.ltx23_conditioning import (
    WeeToddLTX23ControlFrames,
    WeeToddLTX23ControlVideo,
    WeeToddLTX23IngredientsReferenceSheet,
    WeeToddLTX23Keyframe,
    WeeToddLTX23VideoExtension,
    prepared_conditioning,
)


def test_keyframe_node_chains_immutable_timed_images():
    node = WeeToddLTX23Keyframe()
    image = np.zeros((1, 64, 64, 3), dtype=np.float32)
    first = node.append(image, 0, 1.0)[0]
    chain = node.append(image, -1, 0.75, first)[0]
    assert [item["frame_index"] for item in chain] == [0, -1]
    with pytest.raises(ValueError, match="unique"):
        node.append(image, 0, 1.0, chain)


def test_prepared_keyframes_use_contract_and_always_remove_owned_files():
    model = LTX23ModelSpec("/not-loaded", "/not-loaded")
    config = LTX23GenerationConfig(width=384, height=256, duration_seconds=1)
    image = np.zeros((1, 64, 64, 3), dtype=np.float32)
    chain = (
        {"image": image, "frame_index": 0, "strength": 1.0},
        {"image": image, "frame_index": -1, "strength": 0.8},
    )
    with prepared_conditioning(model, config, keyframes=chain) as (kwargs, contract):
        paths = [Path(item["path"]) for item in kwargs["image_inputs"]]
        assert all(path.is_file() for path in paths)
        assert [item["frame_index"] for item in kwargs["image_inputs"]] == [0, 24]
        assert contract["task"] == "fflf"
    assert not any(path.exists() for path in paths)


def test_control_node_requires_an_existing_local_video(tmp_path):
    node = WeeToddLTX23ControlVideo()
    with pytest.raises(FileNotFoundError):
        node.specify(str(tmp_path / "missing.mp4"), "canny_edges", 1.0)
    video = tmp_path / "guide.mp4"
    video.touch()
    control = node.specify(str(video), "motion_track", 0.5)[0]
    assert control["control_type"] == "motion_track"
    assert control["strength"] == 0.5


def test_extension_node_requires_exact_frame_groups(tmp_path):
    video = tmp_path / "source.mp4"
    video.touch()
    node = WeeToddLTX23VideoExtension()
    extension = node.specify(str(video), "after", 24)[0]
    assert extension == {
        "path": str(video.resolve()),
        "direction": "after",
        "additional_frames": 24,
    }
    with pytest.raises(ValueError, match="multiple of 8"):
        node.specify(str(video), "after", 25)


def test_control_frames_bridge_preserves_preprocessor_batch():
    images = np.zeros((25, 256, 384, 3), dtype=np.float32)
    control = WeeToddLTX23ControlFrames().prepare(images, "canny_edges", 0.8)[0]
    assert control["images"] is images
    assert control["kind"] == "frame_batch"
    assert control["strength"] == 0.8


def test_generation_config_rejects_unknown_ic_lora_topology():
    with pytest.raises(ValueError, match="IC-LoRA topology"):
        LTX23GenerationConfig(ic_lora_topology="mystery").validate()


def test_control_frames_materialize_exact_video_and_cleanup():
    images = np.zeros((25, 256, 384, 3), dtype=np.float32)
    control = WeeToddLTX23ControlFrames().prepare(images, "canny_edges", 0.8)[0]
    model = LTX23ModelSpec(
        "/not-loaded",
        "/not-loaded",
        ic_loras=(LTX23ICLoRASpec("/not-loaded/adapter.safetensors", "union_control"),),
    )
    config = LTX23GenerationConfig(
        pipeline_mode="distilled", width=384, height=256, duration_seconds=1
    )
    with prepared_conditioning(model, config, control=control) as (kwargs, contract):
        path = Path(kwargs["control_inputs"][0]["path"])
        assert path.is_file() and path.stat().st_size > 0
        assert contract["task"] == "control"
    assert not path.exists()


def test_prepared_audio_uses_dependency_free_stereo_wav():
    import wave

    model = LTX23ModelSpec("/not-loaded", "/not-loaded")
    config = LTX23GenerationConfig(width=384, height=256, duration_seconds=1)
    audio = {"waveform": np.zeros((1, 1, 32000), dtype=np.float32), "sample_rate": 32000}
    with prepared_conditioning(model, config, audio=audio) as (kwargs, contract):
        path = Path(kwargs["audio_path"])
        with wave.open(str(path), "rb") as handle:
            actual = (handle.getnchannels(), handle.getsampwidth(), handle.getframerate())
            assert actual == (2, 2, 32000)
        assert contract["task"] == "a2v"
    assert not path.exists()


def test_ingredients_node_formats_the_trained_prompt():
    image = np.zeros((1, 448, 768, 3), dtype=np.float32)
    control, prompt = WeeToddLTX23IngredientsReferenceSheet().prepare(
        image, "a red robot turnaround on black", "the robot waves", 1.0
    )
    assert control["kind"] == "image"
    assert control["control_type"] == "ingredients_reference_sheet"
    assert prompt == (
        "Reference sheet: a red robot turnaround on black\n\nGenerated video: the robot waves"
    )
    with pytest.raises(ValueError, match="reference strength"):
        WeeToddLTX23IngredientsReferenceSheet().prepare(image, "robot", "waves", 1.4)


def test_prepared_ingredients_uses_ref2va_contract_and_static_video(tmp_path):
    image = np.zeros((1, 448, 768, 3), dtype=np.float32)
    control, _prompt = WeeToddLTX23IngredientsReferenceSheet().prepare(
        image, "a red robot turnaround on black", "the robot waves", 1.0
    )
    model = LTX23ModelSpec(
        "/not-loaded",
        "/not-loaded",
        ic_loras=(
            LTX23ICLoRASpec(
                "/not-loaded/ingredients.safetensors", "ingredients_reference_sheet"
            ),
        ),
    )
    config = LTX23GenerationConfig(
        pipeline_mode="two_stage",
        width=768,
        height=448,
        duration_seconds=5,
        frame_rate=24,
    )
    with prepared_conditioning(
        model,
        config,
        prompt=(
            "Reference sheet: a red robot turnaround on black\n\n"
            "Generated video: the robot waves"
        ),
        control=control,
    ) as (kwargs, contract):
        path = Path(kwargs["control_inputs"][0]["path"])
        assert contract["task"] == "ref2va"
        assert path.is_file() and path.stat().st_size > 0
    assert not path.exists()
