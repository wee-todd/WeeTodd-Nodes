import copy
import inspect
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from wee_todd_mlx.conditioning_media import inspect_media, ltx_conditioning_kwargs, read_media
from wee_todd_mlx.task_conditioning import (
    TASKS,
    apply_ltx25_msr_prompt_guide,
    frame_geometry,
    ltx25_msr_prompt_guide,
    normalize_conditioning,
    validate_conditioning,
    validate_ltx25_control_families,
)


def recipe(engine="h3", task="t2v", inputs=()):
    component_task = {"t2v": "t2va", "fflf": "fl2va", "control": "t2va"}.get(task, task)
    if engine == "h3" and task == "a2v":
        component_task = "ref2va"
    return {
        "engine": engine,
        "components": {"task": component_task},
        "config": {"duration_seconds": 5.0, "frame_rate": 24, "pipeline_mode": "two_stage"},
        "conditioning": {"version": 1, "task": task, "inputs": list(inputs)},
    }


def media(role="keyframe", kind="image", **kwargs):
    result = {"id": "one", "kind": kind, "role": role, "path": "fixture.png"}
    if role == "keyframe":
        result["frame_index"] = 0
    result.update(kwargs)
    return result


@pytest.mark.parametrize("engine", ["h3", "ltx23", "ltx25"])
def test_legacy_t2v_normalizes_without_mutation(engine):
    r = recipe(engine)
    del r["conditioning"]
    before = copy.deepcopy(r)
    result = validate_conditioning(r)
    assert result["contract"]["task"] == "t2v"
    assert result["input_ids"] == []
    assert r == before


def test_legacy_ref2va_preserves_reference_order():
    r = recipe("h3", "ref2va")
    del r["conditioning"]
    r["reference_images"] = ["b.png", "a.png"]
    assert [i["path"] for i in normalize_conditioning(r, check_files=False)["inputs"]] == r[
        "reference_images"
    ]


@pytest.mark.parametrize("engine", ["h3", "ltx23", "ltx25"])
def test_last_frame_matches_engine_geometry(engine):
    r = recipe(engine, "fflf", [media(frame_index="last")])
    report = validate_conditioning(r, check_files=False)
    assert report["contract"]["inputs"][0]["frame_index"] == frame_geometry(r)[0] - 1


@pytest.mark.parametrize("engine", ["h3", "ltx23", "ltx25"])
@pytest.mark.parametrize("task", TASKS)
def test_all_tasks_have_explicit_empty_input_outcome(engine, task):
    r = recipe(engine, task)
    if task == "t2v":
        assert validate_conditioning(r)["transport"] == "implemented"
    else:
        with pytest.raises(ValueError):
            validate_conditioning(r)


@pytest.mark.parametrize(
    "field,value",
    [
        ("version", True),
        ("version", 2),
        ("inputs", {}),
        ("task", "typo"),
        ("audio_policy", "copy"),
        ("unknown", True),
    ],
)
def test_unknown_schema_rejected(field, value):
    r = recipe()
    r["conditioning"][field] = value
    with pytest.raises(ValueError):
        validate_conditioning(r)


@pytest.mark.parametrize(
    "change",
    [
        {"strength": float("nan")},
        {"strength": True},
        {"strength": 1.1},
        {"frame_index": True},
        {"frame_index": -1},
        {"frame_index": 999},
        {"path": "https://example.test/image.png"},
        {"kind": "mask"},
        {"role": "typo"},
        {"source_start": 1},
        {"id": ""},
        {"control_type": "canny_edges"},
    ],
)
def test_invalid_input_rejected(change):
    with pytest.raises(ValueError):
        validate_conditioning(recipe("ltx25", "fflf", [media(**change)]), check_files=False)


def test_duplicate_ids_and_duplicate_frames_rejected():
    for second in (media(), media(id="two")):
        with pytest.raises(ValueError):
            validate_conditioning(recipe("ltx25", "fflf", [media(), second]), check_files=False)


@pytest.mark.parametrize("engine", ["h3", "ltx23", "ltx25"])
def test_t2v_never_silently_drops_media(engine):
    with pytest.raises(ValueError):
        validate_conditioning(recipe(engine, "t2v", [media()]), check_files=False)


def test_legacy_conflict_and_misplaced_media_fail():
    for field in ("reference_images", "image_path", "audio_reference", "task", "conditoning"):
        r = recipe()
        r[field] = []
        with pytest.raises(ValueError):
            validate_conditioning(r)


def test_h3_paired_soundtrack_counts_against_audio_limit():
    items = [media("reference", "video", id=str(i), soundtrack_path="sound.wav") for i in range(3)]
    r = recipe("h3", "ref2va", items)
    assert validate_conditioning(r, check_files=False)["transport"] == "implemented"
    r["conditioning"]["inputs"].append(media("reference", "audio", id="fourth"))
    with pytest.raises(ValueError, match="three|3 audio"):
        validate_conditioning(r, check_files=False)
    with pytest.raises(ValueError, match="soundtrack_path"):
        validate_conditioning(
            recipe("ltx25", "fflf", [media(soundtrack_path="sound.wav")]), check_files=False
        )


@pytest.mark.parametrize("accelerator", ["attention", "fastvideo", "vdn"])
def test_accelerated_h3_conditioning_does_not_inherit_t2v_qualification(accelerator):
    r = recipe("h3", "fflf", [media()])
    r[accelerator] = {"enabled": True}
    with pytest.raises(ValueError, match="Accelerated"):
        validate_conditioning(r, check_files=False)


def test_h3_task_mismatch_audio_roles_and_strength():
    r = recipe("h3", "fflf", [media()])
    r["components"]["task"] = "ref2va"
    with pytest.raises(ValueError, match="components.task"):
        validate_conditioning(r, check_files=False)
    with pytest.raises(ValueError, match="strength"):
        validate_conditioning(recipe("h3", "fflf", [media(strength=0.5)]), check_files=False)
    r = recipe("h3", "ref2va", [media("reference", "audio")])
    with pytest.raises(ValueError, match="Untimed"):
        validate_conditioning(r, check_files=False)
    r["conditioning"]["inputs"][0]["frame_index"] = 0
    assert validate_conditioning(r, check_files=False)["transport"] == "implemented"
    report = validate_conditioning(
        recipe("h3", "a2v", [media("audio_driver", "audio")]), check_files=False
    )
    assert report["contract"]["audio_policy"] == "generated"
    assert "not copied" in report["audio_semantics"]
    wrong_partition = recipe("h3", "a2v", [media("audio_driver", "audio")])
    wrong_partition["components"]["task"] = "fl2va"
    with pytest.raises(ValueError, match="components.task=ref2va"):
        validate_conditioning(wrong_partition, check_files=False)


def test_h3_fun_controlnet_contract_is_explicit_and_fail_closed():
    r = recipe(
        "h3",
        "control",
        [media("control", "video", control_type="canny_edges", strength=0.65)],
    )
    r["components"]["fun_controlnet"] = "MiniMax-H3-Fun-Controlnet-Union.safetensors"
    report = validate_conditioning(r, check_files=False)
    assert report["transport"] == "implemented"
    assert report["contract"]["inputs"][0]["strength"] == 0.65

    del r["components"]["fun_controlnet"]
    with pytest.raises(ValueError, match="components.fun_controlnet"):
        validate_conditioning(r, check_files=False)
    r["components"]["fun_controlnet"] = "control.safetensors"
    r["conditioning"]["inputs"][0]["control_type"] = "motion_track"
    with pytest.raises(ValueError, match="Canny, depth, HED"):
        validate_conditioning(r, check_files=False)


@pytest.mark.parametrize("control_type", ["hed_edges", "mlsd_lines"])
def test_h3_fun_controlnet_accepts_external_hed_and_mlsd_guides(control_type):
    r = recipe(
        "h3",
        "control",
        [media("control", "video", control_type=control_type)],
    )
    r["components"]["fun_controlnet"] = "MiniMax-H3-Fun-Controlnet-Union.safetensors"
    assert validate_conditioning(r, check_files=False)["transport"] == "implemented"


def test_ltx23_distilled_fflf_rejected_and_audio_policy_explicit():
    r = recipe("ltx23", "fflf", [media()])
    r["config"]["pipeline_mode"] = "distilled"
    with pytest.raises(ValueError, match="Dev two_stage"):
        validate_conditioning(r, check_files=False)
    r = recipe("ltx23", "a2v", [media("audio_driver", "audio")])
    assert validate_conditioning(r, check_files=False)["contract"]["audio_policy"] == "source"
    r["conditioning"]["audio_policy"] = "generated"
    with pytest.raises(ValueError, match="audio_policy=source"):
        validate_conditioning(r, check_files=False)


def test_ltx23_conditioned_generic_lora_and_streaming_transport_is_accepted():
    r = recipe("ltx23", "fflf", [media()])
    r["config"].update(pipeline_mode="two_stage", low_ram_streaming=True)
    r["loras"] = {
        "adapters": [{"path": "downloaded-anywhere.safetensors", "strength": 0.8}]
    }
    assert validate_conditioning(r, check_files=False)["transport"] == "implemented"


@pytest.mark.parametrize(
    ("control_type", "family"),
    [("canny_edges", "union_control"), ("motion_track", "motion_track")],
)
def test_ltx23_ic_control_requires_explicit_matching_adapter(control_type, family):
    r = recipe(
        "ltx23",
        "control",
        [media("control", "video", control_type=control_type)],
    )
    r["config"].update(pipeline_mode="distilled", width=384, height=256)
    r["components"]["ic_loras"] = [
        {"path": "adapter.safetensors", "family": family}
    ]
    assert validate_conditioning(r, check_files=False)["transport"] == "implemented"
    r["components"]["ic_loras"][0]["family"] = (
        "motion_track" if family == "union_control" else "union_control"
    )
    with pytest.raises(ValueError, match="family"):
        validate_conditioning(r, check_files=False)


def test_ltx23_ic_control_rejects_streaming_and_generic_lora_stacks():
    r = recipe(
        "ltx23",
        "control",
        [media("control", "video", control_type="canny_edges")],
    )
    r["config"].update(pipeline_mode="distilled", width=384, height=256)
    r["components"]["ic_loras"] = [
        {"path": "control.safetensors", "family": "union_control"}
    ]
    r["config"]["low_ram_streaming"] = True
    with pytest.raises(ValueError, match="resident loading"):
        validate_conditioning(r, check_files=False)
    r["config"]["low_ram_streaming"] = False
    r["loras"] = {"adapters": [{"path": "style.safetensors"}]}
    with pytest.raises(ValueError, match="without generic LoRAs"):
        validate_conditioning(r, check_files=False)


def test_h3_only_control_types_fail_closed_for_ltx():
    r = recipe(
        "ltx23",
        "control",
        [media("control", "video", control_type="hed_edges")],
    )
    r["config"].update(pipeline_mode="distilled", width=384, height=256)
    r["components"]["ic_loras"] = [
        {"path": "control.safetensors", "family": "h3_fun_union"}
    ]
    with pytest.raises(ValueError, match="specific to H3"):
        validate_conditioning(r, check_files=False)


def test_ltx23_ic_ingredients_and_bad_grid_fail_closed():
    r = recipe(
        "ltx23",
        "control",
        [media("control", "image", control_type="ingredients_reference_sheet")],
    )
    r["config"].update(pipeline_mode="two_stage", width=384, height=256)
    r["components"]["ic_loras"] = [
        {"path": "adapter.safetensors", "family": "ingredients_reference_sheet"}
    ]
    with pytest.raises(ValueError):
        validate_conditioning(r, check_files=False)
    r["conditioning"]["inputs"] = [media("control", "video", control_type="canny_edges")]
    r["components"]["ic_loras"][0]["family"] = "union_control"
    r["config"]["pipeline_mode"] = "distilled"
    r["config"]["width"] = 448
    with pytest.raises(ValueError, match="reference scale"):
        validate_conditioning(r, check_files=False)


def test_ltx23_ingredients_ref2va_requires_trained_bucket_and_prompt():
    r = recipe(
        "ltx23",
        "ref2va",
        [media("control", "image", control_type="ingredients_reference_sheet")],
    )
    r["config"].update(
        pipeline_mode="two_stage", width=768, height=448, duration_seconds=5, frame_rate=24
    )
    r["components"]["ic_loras"] = [
        {"path": "adapter.safetensors", "family": "ingredients_reference_sheet"}
    ]
    r["prompt"] = "Reference sheet: one character.\nGenerated video: the character waves."
    assert validate_conditioning(r, check_files=False)["transport"] == "implemented"
    r["prompt"] = "the character waves"
    with pytest.raises(ValueError, match="Reference sheet"):
        validate_conditioning(r, check_files=False)


def test_ltx23_extension_contract_is_explicit_and_engine_scoped():
    r = recipe(
        "ltx23",
        "extension",
        [media("reference", "video", path="source.mp4")],
    )
    r["config"].update(
        pipeline_mode="one_stage", width=448, height=768, duration_seconds=3, frame_rate=24
    )
    r["conditioning"].update(
        audio_policy="source_reencoded_and_generated_extension",
        extension={"direction": "after", "additional_frames": 24},
    )
    report = validate_conditioning(r, check_files=False)
    assert report["contract"]["extension"] == {
        "direction": "after",
        "additional_frames": 24,
    }
    r["config"]["low_ram_streaming"] = True
    r["loras"] = {
        "adapters": [{"path": "downloaded-extension-style.safetensors", "strength": 0.6}]
    }
    assert validate_conditioning(r, check_files=False)["transport"] == "implemented"
    del r["config"]["low_ram_streaming"]
    del r["loras"]
    kwargs = ltx_conditioning_kwargs(
        r,
        report["contract"],
        [{"id": "one", "kind": "video", "path": "source.mp4"}],
    )
    assert kwargs["extension_input"]["additional_frames"] == 24
    r["conditioning"]["extension"]["additional_frames"] = 25
    with pytest.raises(ValueError, match="multiple of 8"):
        validate_conditioning(r, check_files=False)
    r["engine"] = "h3"
    r["components"]["task"] = "ref2va"
    r["config"]["duration_seconds"] = 5
    r["prompt"] = (
        "subject_definitions:\n<Video 1> is the source video for continuation.\n"
        "<Picture 1> is the first frame of the target video.\n\n"
        "summary:\n[video continuation + keyframe completion] Continue <Video 1>.\n\n"
        "retention_analysis:\n<Video 1>: fully_preserved - continue its motion.\n"
        "<Picture 1>: fully_preserved - use it as the opening frame.\n\n"
        "detailed_description:\n[Shot 1] Continue the scene.\n\n"
        "overall_soundscape:\nContinue the ambience.\n\nnon_diegetic_music:\nN/A"
    )
    r["conditioning"].update(audio_policy="source_reencoded_and_generated_extension")
    r["conditioning"]["extension"] = {
        "direction": "after",
        "additional_frames": 124,
    }
    assert validate_conditioning(r, check_files=False)["transport"] == "implemented"
    r["conditioning"]["extension"]["direction"] = "before"
    with pytest.raises(ValueError, match="after"):
        validate_conditioning(r, check_files=False)
    r["engine"] = "ltx25"
    r["components"] = {}
    r["config"].update(pipeline_mode="distilled", ic_lora_single_stage=False)
    r["conditioning"]["extension"] = {
        "direction": "after",
        "additional_frames": 24,
        "context_frames": 25,
    }
    r["loras"] = {
        "adapters": [{"path": "downloaded-extension-style.safetensors", "strength": 0.6}]
    }
    assert validate_conditioning(r, check_files=False)["transport"] == "implemented"
    r["conditioning"]["extension"]["context_frames"] = 24
    with pytest.raises(ValueError, match=r"8n\+1"):
        validate_conditioning(r, check_files=False)


def test_ltx23_extension_runtime_uses_retake_and_records_accounting(tmp_path, monkeypatch):
    import ltx_core_mlx.utils.ffmpeg as backend_ffmpeg

    import ltx23_mlx.runtime as module

    source = tmp_path / "source.mp4"
    source.touch()
    calls = {}

    class Pipeline:
        def __init__(self, **kwargs):
            calls["constructor"] = kwargs
            self.dit = None

        def extend_from_video(self, **kwargs):
            calls["extend"] = kwargs
            return "video-latent", "audio-latent"

        def _load_decoders(self):
            calls["decoders"] = True

        def _decode_and_save_video(self, video, audio, output, *, frame_rate):
            calls["decode"] = (video, audio, output, frame_rate)
            return output

    monkeypatch.setattr(module, "_pipeline_class", lambda _mode: Pipeline)
    monkeypatch.setattr(module.LTX23ModelSpec, "validate", lambda *_args: None)
    monkeypatch.setattr(module.LTX23ModelSpec, "gemma_root", lambda *_args: tmp_path)
    monkeypatch.setattr(
        backend_ffmpeg,
        "probe_video_info",
        lambda _path: SimpleNamespace(width=448, height=768, fps=24.0, num_frames=73),
    )
    config = module.LTX23GenerationConfig(
        pipeline_mode="one_stage",
        width=448,
        height=768,
        duration_seconds=3,
        frame_rate=24,
        stage1_steps=8,
    )
    info = module.LTX23RuntimeCache().generate_to_file(
        module.LTX23ModelSpec(str(tmp_path), str(tmp_path)),
        config,
        "continue brushing",
        tmp_path / "out.mp4",
        extension_input={"path": str(source), "direction": "after", "additional_frames": 24},
    )
    assert calls["constructor"]["dev_transformer"] == "transformer-dev.safetensors"
    assert calls["extend"]["extend_frames"] == 3
    assert calls["decode"][3] == 24
    assert info["extension"] == {
        "direction": "after",
        "source_frames": 73,
        "additional_frames": 24,
        "output_frames": 97,
    }
    assert info["audio_policy"] == "source_latent_reconstructed_and_generated_extension"
    assert info["ic_lora_topology_effective"] == "none"


def test_ltx23_distilled_extension_contract_requires_full_schedule():
    r = recipe(
        "ltx23",
        "extension",
        [media("reference", "video", path="source.mp4")],
    )
    r["config"].update(
        pipeline_mode="distilled",
        width=448,
        height=768,
        duration_seconds=3,
        frame_rate=24,
        stage1_steps=8,
    )
    r["conditioning"].update(
        audio_policy="source_reencoded_and_generated_extension",
        extension={"direction": "after", "additional_frames": 24},
    )
    assert validate_conditioning(r, check_files=False)["transport"] == "implemented"
    r["config"]["stage1_steps"] = 7
    with pytest.raises(ValueError, match="exactly eight"):
        validate_conditioning(r, check_files=False)


def test_control_family_mismatch_not_silent():
    r = recipe("ltx25", "control", [media("control", "video", control_type="depth_map")])
    c = validate_conditioning(r, check_files=False)["contract"]
    with pytest.raises(ValueError, match="union_control"):
        validate_ltx25_control_families(c, {"components": []})
    validate_ltx25_control_families(
        c, {"components": [{"component": "ic_lora_0", "adapter_family": "union_control"}]}
    )


def test_ltx25_msr_contract_orders_slots_and_builds_prompt_guide():
    adapter = "/models/LTX-2.5-Licon-MSR-V1.safetensors"
    r = recipe(
        "ltx25",
        "ref2va",
        [
            media(
                "reference",
                "image",
                id="scene",
                reference_role="background",
                description="the tiled bathroom",
            ),
            media(
                "reference",
                "image",
                id="monkey",
                reference_role="subject",
                description="the wet macaque",
            ),
        ],
    )
    r["components"].update(msr_lora_path=adapter, ic_loras=[[adapter, 1.0]])
    r["config"].update(pipeline_mode="distilled", ic_lora_single_stage=True)
    report = validate_conditioning(r, check_files=False)
    contract = report["contract"]
    assert [item["id"] for item in contract["inputs"]] == ["monkey", "scene"]
    assert [item["reference_priority"] for item in contract["inputs"]] == [
        "primary",
        "background",
    ]
    assert report["prompt_guide"] == (
        "Image 1 provides the subject: the wet macaque\n"
        "Image 2 provides the background: the tiled bathroom"
    )
    assert ltx25_msr_prompt_guide(contract) == report["prompt_guide"]
    prompted = apply_ltx25_msr_prompt_guide("The monkey brushes its teeth.", report["prompt_guide"])
    assert prompted.startswith(report["prompt_guide"])
    assert apply_ltx25_msr_prompt_guide(prompted, report["prompt_guide"]) == prompted


def test_ltx25_msr_contract_and_adapter_fail_closed():
    item = media(
        "reference",
        "image",
        reference_role="subject",
        description="one subject",
    )
    r = recipe("ltx25", "ref2va", [item])
    r["config"].update(pipeline_mode="distilled", ic_lora_single_stage=True)
    with pytest.raises(ValueError, match="msr_lora_path"):
        validate_conditioning(r, check_files=False)
    r["components"].update(
        msr_lora_path="/models/msr.safetensors",
        ic_loras=[["/models/other.safetensors", 1.0]],
    )
    with pytest.raises(ValueError, match="single components.ic_loras"):
        validate_conditioning(r, check_files=False)
    r["components"]["ic_loras"] = [["/models/msr.safetensors", 1.0]]
    r["config"]["ic_lora_single_stage"] = False
    with pytest.raises(ValueError, match="full-resolution single-stage"):
        validate_conditioning(r, check_files=False)
    del r["conditioning"]["inputs"][0]["description"]
    with pytest.raises(ValueError, match="description"):
        validate_conditioning(r, check_files=False)


def test_ltx25_msr_media_reaches_runtime_as_independent_stills(tmp_path):
    adapter = str(tmp_path / "msr.safetensors")
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    Image.new("RGB", (48, 32), "red").save(first)
    Image.new("RGB", (32, 48), "blue").save(second)
    r = recipe(
        "ltx25",
        "ref2va",
        [
            media(
                "reference",
                "image",
                id="first",
                path=str(first),
                reference_role="subject",
                description="the red subject",
            ),
            media(
                "reference",
                "image",
                id="second",
                path=str(second),
                reference_role="object",
                description="the blue prop",
                reference_frames="25",
                reference_size_policy="balanced",
                reference_priority="supporting",
            ),
        ],
    )
    r["components"].update(msr_lora_path=adapter, ic_loras=[[adapter, 1.0]])
    r["config"].update(pipeline_mode="distilled", ic_lora_single_stage=True)
    contract = validate_conditioning(r)["contract"]
    kwargs = ltx_conditioning_kwargs(r, contract, inspect_media(r, contract))
    references = kwargs["msr_references"]
    assert len(references) == 2
    assert references[0]["image"].shape == (1, 32, 48, 3)
    assert references[0]["image"].dtype.name == "float32"
    assert references[0]["image"].max() == pytest.approx(1.0)
    assert references[1]["reference_frames"] == "25"
    assert references[1]["reference_priority"] == "supporting"


def test_real_image_inspection_and_ltx_transport(tmp_path):
    filename = tmp_path / "renamed.dat"
    Image.new("RGB", (64, 32), "red").save(filename, format="PNG")
    r = recipe("ltx25", "fflf", [media(path=str(filename), frame_index="last")])
    c = validate_conditioning(r)["contract"]
    reports = inspect_media(r, c)
    assert reports[0]["width"] == 64
    assert read_media(r, c["inputs"][0], reports[0]).shape == (1, 32, 64, 3)
    kwargs = ltx_conditioning_kwargs(r, c, reports)
    assert kwargs["image_inputs"] == [{"path": str(filename), "frame_index": 120, "strength": 1}]


def test_wrong_media_type_rejected(monkeypatch):
    monkeypatch.setattr(
        subprocess, "run", lambda *a, **k: SimpleNamespace(stdout=json.dumps({"streams": []}))
    )
    r = recipe("ltx25", "a2v", [media("audio_driver", "audio")])
    with pytest.raises(ValueError, match="exactly one audio"):
        inspect_media(r, validate_conditioning(r, check_files=False)["contract"])


def test_pure_contract_does_not_import_mlx():
    run = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import wee_todd_mlx.task_conditioning; "
            "assert 'mlx.core' not in sys.modules",
        ],
        capture_output=True,
        text=True,
    )
    assert run.returncode == 0, run.stderr


@pytest.mark.parametrize("task", ["keyframe", "a2v"])
def test_ltx23_actual_consuming_signature(task):
    from ltx23_mlx.runtime import _pipeline_class

    cls = _pipeline_class(task)
    fields = inspect.signature(cls.generate_and_save).parameters
    assert ("audio_path" if task == "a2v" else "keyframe_images") in fields


@pytest.mark.parametrize("failure", [False, True])
def test_ltx23_keyframes_reach_selected_pipeline_and_cleanup(tmp_path, monkeypatch, failure):
    from ltx23_mlx import runtime as module

    filename = tmp_path / "input.png"
    Image.new("RGB", (64, 64)).save(filename)
    captured = {}

    class Pipeline:
        def __init__(self, **kwargs):
            assert kwargs["dev_transformer"] == "transformer-dev.safetensors"
            assert kwargs["distilled_lora"] == "ltx-2.3-22b-distilled-lora-384.safetensors"

        def generate_and_save(
            self,
            prompt,
            output_path,
            keyframe_images,
            keyframe_indices,
            keyframe_strengths,
            video_guider_params,
        ):
            captured.update(
                images=keyframe_images, indices=keyframe_indices, strengths=keyframe_strengths
            )
            if failure:
                raise RuntimeError("cancel fixture")
            return output_path

    monkeypatch.setattr(module, "_pipeline_class", lambda mode: Pipeline)
    monkeypatch.setattr(module.LTX23ModelSpec, "validate", lambda *a: None)
    monkeypatch.setattr(module.LTX23ModelSpec, "gemma_root", lambda self: tmp_path)
    cache = module.LTX23RuntimeCache()

    def call():
        return cache.generate_to_file(
            module.LTX23ModelSpec(str(tmp_path)),
            module.LTX23GenerationConfig(),
            "fixture",
            tmp_path / "out.mp4",
            image_inputs=[{"path": str(filename), "frame_index": 120, "strength": 0.7}],
        )

    if failure:
        with pytest.raises(RuntimeError, match="cancel fixture"):
            call()
    else:
        assert call()["conditioning_task"] == "keyframe"
    assert captured == {"images": [str(filename)], "indices": [120], "strengths": [0.7]}
    assert not cache.loaded


def test_preflight_rejects_task_before_engine_inventory(monkeypatch):
    from ltx23_mlx.runtime import LTX23ModelSpec
    from wee_todd_mlx.headless_preflight import preflight_recipe

    monkeypatch.setattr(
        LTX23ModelSpec, "validate", lambda *a: pytest.fail("No model inspection expected")
    )
    with pytest.raises(ValueError, match="extension"):
        preflight_recipe(recipe("ltx23", "extension"))


def test_control_transport_uses_unit_range_pixels(tmp_path):
    filename = tmp_path / "sheet.png"
    Image.new("RGB", (64, 32), (128, 64, 32)).save(filename)
    r = recipe(
        "ltx25",
        "ref2va",
        [
            media(
                "control",
                "image",
                path=str(filename),
                control_type="ingredients_reference_sheet",
            )
        ],
    )
    c = validate_conditioning(r)["contract"]
    kwargs = ltx_conditioning_kwargs(r, c, inspect_media(r, c))
    pixels = kwargs["video_references"][0]["images"]
    assert float(pixels.max()) == pytest.approx(128 / 255)
    assert kwargs["video_references"][0]["end_frame"] == 120


def test_a2v_temporary_waveform_removed_on_failed_mux():
    from ltx23_mlx.runtime import _audio_temporary_cleanup

    class Pipeline:
        @staticmethod
        def _save_waveform(waveform, path):
            Path(path).write_bytes(b"owned fixture")

    pipeline = Pipeline()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
        filename = Path(handle.name)
    with pytest.raises(RuntimeError, match="mux failed"):
        with _audio_temporary_cleanup(pipeline, True):
            pipeline._save_waveform(None, str(filename))
            raise RuntimeError("mux failed")
    assert not filename.exists()
    assert "_save_waveform" not in pipeline.__dict__


def test_ltx25_runtime_rejects_lipdub_before_loading(monkeypatch, tmp_path):
    from ltx25_mlx.runtime import LTX25ComponentSpec, LTX25GenerationConfig, LTX25RuntimeCache

    monkeypatch.setattr(
        LTX25ComponentSpec,
        "validate",
        lambda *a, **kw: {
            "video_scale_factors": (8, 32, 32),
        },
    )
    runtime = LTX25RuntimeCache()
    monkeypatch.setattr(runtime, "get", lambda *a: pytest.fail("No runtime should load"))
    spec = LTX25ComponentSpec("transformer", "text", "video", "audio", "upscale")
    with pytest.raises(ValueError, match="LipDub"):
        runtime.generate_to_file(
            spec,
            LTX25GenerationConfig(),
            "fixture",
            tmp_path / "out.mp4",
            video_references=[{}],
            audio_reference={},
        )
