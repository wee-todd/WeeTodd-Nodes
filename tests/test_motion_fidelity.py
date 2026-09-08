"""Motion timing, budget and shared adapter contracts; no model weights required."""

import importlib
import json
import shutil
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from wee_todd_mlx.motion_fidelity import (
    MotionSettings,
    aligned_frames,
    audio_filter,
    expansion_indices,
    latent_index,
    motion_lora_stack,
    plan_motion,
    validate_recipe,
)
from wee_todd_nodes.lora import H3LoRASpec, H3LoRAStack

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
bridge = importlib.import_module("studio_bridge")
jobs = importlib.import_module("studio_job")
runner = importlib.import_module("motion_fidelity")


def test_h3_phase_clock_and_exact_recovery():
    assert [latent_index(i) for i in range(18)] == [0] + [1] * 4 + [2] * 4 + [3] * 4 + [4] * 4 + [5]
    for n in range(60, 87):
        for hold in range(2, 5):
            plan = plan_motion(n, MotionSettings(mode="uniform", maxHold=hold))
            expanded = expansion_indices(plan)
            assert len(expanded) % 17 == 5
            assert len(expanded) == aligned_frames(n * hold)
            assert np.array_equal(expanded[plan["recovery"]], np.arange(n))
            assert expanded[-1] == n - 1


def test_adaptive_quiet_clip_is_noop_and_burst_is_local():
    latent = np.zeros((1, 24, 22, 2, 2), np.float32)
    settings = MotionSettings(maxHold=4)
    quiet = plan_motion(73, settings, latent)
    assert quiet["noop"] and max(quiet["holds"]) == 1
    latent[:, :, 10] = 10
    burst = plan_motion(73, settings, latent)
    assert max(burst["holds"]) == 4
    assert burst["holds"][0] == burst["holds"][-1] == 1
    assert np.max(np.abs(np.diff(burst["holds"]))) <= 1
    assert burst == plan_motion(73, settings, latent)


def test_explicit_refinement_evaluations_validate_and_survive_plan():
    settings = MotionSettings(mode="uniform", evaluations=14)
    assert plan_motion(73, settings)["settings"]["evaluations"] == 14
    for invalid in (0, -1, 65, True, 14.5):
        with pytest.raises(ValueError, match="evaluations"):
            MotionSettings(evaluations=invalid).validate()
    MotionSettings().validate()


@pytest.mark.parametrize(
    "settings",
    [
        MotionSettings(strength=float("nan")),
        MotionSettings(maxHold=5),
        MotionSettings(maxFrames=500),
        MotionSettings(seed=-1),
        MotionSettings(mode="guess"),
    ],
)
def test_invalid_settings_rejected(settings):
    with pytest.raises(ValueError):
        settings.validate()


def test_expansion_budget_rejects_before_render():
    with pytest.raises(ValueError, match="budget"):
        plan_motion(100, MotionSettings(mode="uniform", maxHold=4))
    with pytest.raises(ValueError, match="complete"):
        plan_motion(73, MotionSettings(), np.zeros((1, 24, 20, 2, 2)))


def test_recipe_rejects_unsupported_engines_and_distilled_metadata(tmp_path):
    for engine in ("ltx25", "ltx23", "movie"):
        with pytest.raises(ValueError, match="plain H3"):
            validate_recipe({"engine": engine})
    recipe = {
        "engine": "h3",
        "components": {"task": "t2va", "transformer": str(tmp_path)},
        "config": {"steps": 20},
    }
    (tmp_path / "paged_manifest.json").write_text(json.dumps({"source": "FastVideo/FastH3"}))
    with pytest.raises(ValueError, match="Distilled"):
        validate_recipe(recipe)
    with pytest.raises(ValueError, match="accelerated"):
        validate_recipe(dict(recipe, attention={"enabled": True}))


def _motion_lora(path, *, metadata=None):
    mx.save_safetensors(
        path,
        {
            "blocks.0.attn.out_proj.lora_A.weight": mx.ones((2, 64)),
            "blocks.0.attn.out_proj.lora_B.weight": mx.ones((64, 2)),
        },
        metadata={"base_model": "MiniMax-H3", **(metadata or {})},
    )
    return path


def _motion_recipe(tmp_path, adapter):
    return {
        "engine": "h3",
        "components": {"task": "t2va", "transformer": str(tmp_path)},
        "config": {"steps": 20},
        "loras": {"adapters": [adapter]},
    }


def test_standard_full_schedule_motion_lora_recipe_is_accepted(tmp_path, monkeypatch):
    path = _motion_lora(tmp_path / "motion.safetensors")
    recipe = _motion_recipe(tmp_path, {"path": str(path), "strength": 0.7})
    expected = {"status": "preflight_passed"}
    monkeypatch.setattr("wee_todd_mlx.headless_preflight.preflight_recipe", lambda _: expected)

    assert validate_recipe(recipe) is expected
    stack = motion_lora_stack(recipe)
    assert stack.metadata()[0]["profile"] == "standard"
    assert stack.metadata()[0]["start_after_evaluations"] == 0


@pytest.mark.parametrize(
    "adapter, error",
    [
        ({"strength": 1.0}, "Malformed"),
        ({"path": "ADAPTER", "profile": "turbo"}, "standard"),
        ({"path": "ADAPTER", "start_after_evaluations": 1}, "full schedule"),
    ],
)
def test_motion_lora_recipe_rejects_malformed_turbo_and_staged_before_preflight(
    tmp_path, monkeypatch, adapter, error
):
    path = _motion_lora(tmp_path / "motion.safetensors")
    adapter = {key: (str(path) if value == "ADAPTER" else value) for key, value in adapter.items()}
    called = False

    def preflight(_):
        nonlocal called
        called = True

    monkeypatch.setattr("wee_todd_mlx.headless_preflight.preflight_recipe", preflight)

    with pytest.raises(ValueError, match=error):
        validate_recipe(_motion_recipe(tmp_path, adapter))
    assert called is False


@pytest.mark.parametrize(
    "field, value",
    [
        ("path", False),
        ("path", ["adapter.safetensors"]),
        ("strength", True),
        ("strength", "0.7"),
        ("profile", ["standard"]),
        ("qkv_layout", {"layout": "native_interleaved"}),
        ("adaln_input_grid", 42),
        ("start_after_evaluations", False),
        ("start_after_evaluations", "1"),
    ],
)
def test_motion_lora_recipe_rejects_non_json_contract_field_types(tmp_path, field, value):
    path = _motion_lora(tmp_path / "motion.safetensors")
    adapter = {"path": str(path), field: value}

    with pytest.raises(ValueError, match="Malformed MiniMax H3 LoRA adapter"):
        motion_lora_stack(_motion_recipe(tmp_path, adapter))


def test_motion_lora_recipe_preserves_file_and_tensor_contract_errors(tmp_path):
    missing = tmp_path / "missing.safetensors"
    with pytest.raises(FileNotFoundError, match="LoRA file not found"):
        motion_lora_stack(_motion_recipe(tmp_path, {"path": str(missing)}))

    malformed = tmp_path / "bad-shape.safetensors"
    mx.save_safetensors(
        malformed,
        {
            "blocks.0.attn.out_proj.lora_A.weight": mx.ones((2, 64)),
            "blocks.0.attn.out_proj.lora_B.weight": mx.ones((64, 3)),
        },
    )
    with pytest.raises(ValueError, match="Invalid adapter rank or shape"):
        motion_lora_stack(_motion_recipe(tmp_path, {"path": str(malformed)}))


def test_motion_node_appends_optional_lora_socket_and_serializes_existing_stack(tmp_path):
    from wee_todd_nodes.motion_nodes import WeeToddH3MotionRefine, motion_recipe

    path = _motion_lora(tmp_path / "motion.safetensors")
    stack = H3LoRAStack().append(H3LoRASpec(str(path), strength=0.65, profile="standard"))
    required = list(WeeToddH3MotionRefine.INPUT_TYPES()["required"])

    recipe = motion_recipe({"task": "t2va"}, {"steps": 20}, "repair", stack)

    assert required == [
        "components", "config", "motion_settings", "source_video", "prompt", "source_in",
        "duration", "analyze_only",
    ]
    assert WeeToddH3MotionRefine.INPUT_TYPES()["optional"]["loras"] == ("WEETODD_H3_LORAS",)
    assert recipe["loras"]["adapters"] == [asdict(stack.adapters[0])]
    assert "loras" not in motion_recipe({"task": "t2va"}, {"steps": 20}, "repair")


def test_audio_retime_preserves_pitch_and_exact_sample_count(tmp_path):
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        pytest.skip("ffmpeg required")
    plan = plan_motion(73, MotionSettings(mode="uniform", maxHold=4))
    # Exercise mixed hold run boundaries as well as 4x atempo chaining.
    plan["holds"] = [1] * 12 + [2] * 18 + [3] * 19 + [4] * 24
    plan["paddedFrames"] = aligned_frames(sum(plan["holds"]))
    output = tmp_path / "audio.f32"
    subprocess.run(
        [
            ffmpeg,
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=32000:duration=3.041666667",
            "-filter_complex",
            audio_filter(plan),
            "-map",
            "[out]",
            "-ac",
            "2",
            "-f",
            "f32le",
            str(output),
        ],
        check=True,
    )
    samples = np.fromfile(output, dtype=np.float32).reshape(-1, 2)
    assert len(samples) == round(plan["paddedFrames"] * 32000 / 24)
    spectrum = np.abs(np.fft.rfft(samples[:32000, 0]))
    assert abs(np.argmax(spectrum) - 440) <= 2


def test_export_blocks_stale_enhancement_and_off_is_identity(tmp_path):
    source, enhanced = tmp_path / "source.mp4", tmp_path / "enhanced.mp4"
    source.write_bytes(b"source")
    enhanced.write_bytes(b"enhanced")
    settings = {"enabled": True}
    clip = dict(sourcePath=str(source), sourceIn=0, duration=3, motionFidelity=settings)
    with pytest.raises(ValueError, match="pending"):
        bridge.resolve_motion_output(clip)
    clip["motionResult"] = dict(
        path=str(enhanced),
        sourcePath=str(source),
        sourceIn=0,
        duration=3,
        recipeID="",
        settings=settings,
        sourceSHA256=jobs.file_hash(source),
        sha256=jobs.file_hash(enhanced),
    )
    bridge.resolve_motion_output(clip)
    assert clip["sourcePath"] == str(enhanced)
    off = dict(sourcePath=str(source), motionFidelity={"enabled": False})
    bridge.resolve_motion_output(off)
    assert off["sourcePath"] == str(source)


def _motion_editor_request(tmp_path, motion_prompt=None):
    recipe_file = tmp_path / "repair.json"
    recipe_file.write_text(
        json.dumps(
            {
                "engine": "h3",
                "prompt": "Original repair prompt",
                "components": {"task": "t2va", "transformer": str(tmp_path)},
                "config": {"steps": 20},
            }
        )
    )
    clip = {
        "id": "clip",
        "name": "Shot",
        "engine": "h3",
        "sourcePath": str(tmp_path / "source.mp4"),
        "sourceIn": 0,
        "duration": 3,
        "motionFidelity": {"enabled": True},
        "motionRecipeID": str(recipe_file),
    }
    if motion_prompt is not None:
        clip["motionPrompt"] = motion_prompt
    return {
        "project": {"clips": [clip], "settings": {}},
        "runtime": {},
        "clipID": "clip",
    }, recipe_file


def test_motion_request_preserves_recipe_and_applies_exact_override(tmp_path, monkeypatch):
    request, recipe_file = _motion_editor_request(tmp_path, "  Exact repair direction.\n  ")
    monkeypatch.setattr("wee_todd_mlx.motion_fidelity.validate_recipe", lambda _: None)

    resolved = bridge.motion_request(request)

    assert resolved["recipe"]["prompt"] == "  Exact repair direction.\n  "
    assert resolved["recipePrompt"] == "Original repair prompt"
    assert json.loads(recipe_file.read_text())["prompt"] == "Original repair prompt"
    assert bridge.motion_prepare(request) == {
        "prompt": "  Exact repair direction.\n  ",
        "recipePrompt": "Original repair prompt",
        "usingOverride": True,
    }


def test_motion_request_defaults_to_recipe_prompt_and_rejects_invalid_overrides(
    tmp_path, monkeypatch
):
    request, _ = _motion_editor_request(tmp_path)
    monkeypatch.setattr("wee_todd_mlx.motion_fidelity.validate_recipe", lambda _: None)
    resolved = bridge.motion_request(request)
    assert resolved["recipe"]["prompt"] == "Original repair prompt"
    assert resolved["recipePrompt"] == "Original repair prompt"
    assert bridge.motion_prepare(request)["usingOverride"] is False

    for invalid in ("", " \n\t ", 42, True, ["prompt"]):
        request["project"]["clips"][0]["motionPrompt"] = invalid
        with pytest.raises(ValueError, match="repair prompt"):
            bridge.motion_request(request)
    request["project"]["clips"][0]["motionPrompt"] = None
    reset = bridge.motion_prepare(request)
    assert reset["prompt"] == "Original repair prompt"
    assert reset["usingOverride"] is False


def test_motion_enhancement_records_prompt_and_changed_override_is_stale(tmp_path, monkeypatch):
    request, _ = _motion_editor_request(tmp_path, "Original override")
    source = Path(request["project"]["clips"][0]["sourcePath"])
    source.write_bytes(b"source")
    destination = tmp_path / "enhancement"
    recipe = {
        "engine": "h3",
        "prompt": "Original override",
        "components": {"task": "t2va"},
        "config": {"steps": 20},
    }
    monkeypatch.setattr(
        bridge,
        "motion_request",
        lambda _: {
            "recipe": recipe,
            "recipePrompt": "Original repair prompt",
            "clip": request["project"]["clips"][0],
            "settings": {"enabled": True},
            "runtime": {},
            "recipePath": "",
            "recipeEvidence": {},
        },
    )

    def run(_):
        destination.mkdir()
        enhanced = destination / "enhanced.mp4"
        enhanced.write_bytes(b"enhanced")
        (destination / "result.json").write_text(
            json.dumps(
                {
                    "video": str(enhanced),
                    "sourceIn": 0,
                    "sourceSHA256": jobs.file_hash(source),
                    "report": str(destination / "plan.json"),
                }
            )
        )

    monkeypatch.setattr(bridge, "run", run)
    result = bridge.motion_enhance(request, destination)
    evidence = result["motionResult"]
    assert evidence["motionPrompt"] == "Original override"

    accepted_clip = dict(request["project"]["clips"][0])
    accepted_clip["motionResult"] = evidence
    bridge.resolve_motion_output(accepted_clip)
    assert accepted_clip["sourcePath"] == evidence["path"]

    changed_clip = dict(request["project"]["clips"][0])
    changed_clip["motionPrompt"] = "A different repair prompt"
    changed_clip["motionResult"] = evidence
    with pytest.raises(ValueError, match="pending|stale"):
        bridge.resolve_motion_output(changed_clip)
    changed_clip.pop("motionPrompt")
    with pytest.raises(ValueError, match="pending|stale"):
        bridge.resolve_motion_output(changed_clip)


def test_exported_job_embeds_override_and_prompt_change_invalidates_resume(tmp_path, monkeypatch):
    request, _ = _motion_editor_request(tmp_path, "Headless override")
    Path(request["project"]["clips"][0]["sourcePath"]).write_bytes(b"source")
    monkeypatch.setattr("wee_todd_mlx.motion_fidelity.validate_recipe", lambda _: None)
    monkeypatch.setattr(jobs, "renderer_fingerprint", lambda: "renderer")
    job_file = tmp_path / "movie.json"

    jobs.export_job(request, job_file)

    exported = json.loads(job_file.read_text())
    assert exported["motionRecipes"]["clip"]["prompt"] == "Headless override"
    output = tmp_path / "job-output"
    output.mkdir()
    (output / "job-state.json").write_text(
        json.dumps(
            {
                "manifestSHA256": exported["manifestSHA256"],
                "inputsFingerprint": jobs.inputs_fingerprint(exported),
                "completed": {},
                "status": "running",
            }
        )
    )
    changed = json.loads(json.dumps(exported))
    changed["motionRecipes"]["clip"]["prompt"] = "Changed headless override"
    body = dict(changed)
    body.pop("manifestSHA256")
    changed["manifestSHA256"] = jobs.digest(body)
    with pytest.raises(ValueError, match="Job or source inputs changed"):
        jobs.execute(changed, output, resume=True)


@pytest.mark.parametrize("changed_dependency", ["adapter", "adaln_input_grid"])
def test_exported_motion_job_embeds_lora_and_dependency_change_invalidates_resume(
    tmp_path, monkeypatch, changed_dependency
):
    adapter = _motion_lora(tmp_path / "repair-motion.safetensors")
    grid = tmp_path / "repair-adaln-grid.safetensors"
    mx.save_safetensors(grid, {"adaln_input_grid": mx.ones((2, 64))})
    request, recipe_file = _motion_editor_request(tmp_path)
    recipe = json.loads(recipe_file.read_text())
    transformer = tmp_path / "transformer"
    transformer.mkdir()
    recipe["components"]["transformer"] = str(transformer)
    expected_loras = {
        "adapters": [
            {
                "path": str(adapter),
                "strength": 0.55,
                "adaln_input_grid": str(grid),
            }
        ]
    }
    recipe["loras"] = expected_loras
    recipe_file.write_text(json.dumps(recipe))
    Path(request["project"]["clips"][0]["sourcePath"]).write_bytes(b"source")
    monkeypatch.setattr("wee_todd_mlx.motion_fidelity.validate_recipe", lambda _: None)
    monkeypatch.setattr(jobs, "renderer_fingerprint", lambda: "renderer")
    job_file = tmp_path / "movie.json"

    jobs.export_job(request, job_file)

    exported = json.loads(job_file.read_text())
    embedded = exported["motionRecipes"]["clip"]
    assert embedded["loras"] == expected_loras
    stack = motion_lora_stack(embedded)
    assert stack.adapters == (
        H3LoRASpec(str(adapter), strength=0.55, adaln_input_grid=str(grid)),
    )
    before = jobs.inputs_fingerprint(exported)
    output = tmp_path / "job-output"
    output.mkdir()
    (output / "job-state.json").write_text(
        json.dumps(
            {
                "manifestSHA256": exported["manifestSHA256"],
                "inputsFingerprint": before,
                "completed": {},
                "status": "running",
            }
        )
    )
    dependency = adapter if changed_dependency == "adapter" else grid
    dependency.write_bytes(f"changed {changed_dependency} bytes".encode())

    assert jobs.inputs_fingerprint(exported) != before
    with pytest.raises(ValueError, match="Job or source inputs changed"):
        jobs.execute(exported, output, resume=True)


def test_node_settings_contract_and_validation():
    from wee_todd_nodes.motion_nodes import WeeToddH3MotionRefine, WeeToddH3MotionSettings

    settings = WeeToddH3MotionSettings().configure("adaptive", 0.5, 2, 0.5, 42, 345)[0]
    assert MotionSettings(**settings) == replace(MotionSettings(), enabled=True)
    fixed = WeeToddH3MotionSettings().configure("uniform", 0.92, 4, 0.5, 42, 345, 14)[0]
    assert fixed["evaluations"] == 14
    assert WeeToddH3MotionRefine.INPUT_TYPES()["required"]["config"] == ("WEETODD_H3_CONFIG",)
    with pytest.raises(ValueError):
        WeeToddH3MotionSettings().configure("uniform", 2, 2, 0.5, 42, 345)
    for invalid in (-1, 65, True, 14.5):
        with pytest.raises(ValueError, match="evaluations"):
            WeeToddH3MotionSettings().configure("uniform", 0.5, 2, 0.5, 42, 345, invalid)


def test_failed_worker_removes_intermediates_without_removing_report(tmp_path, monkeypatch):
    output = tmp_path / "attempt"

    def fail(*args):
        output.mkdir()
        (output / "source.rgb").write_bytes(b"rgb")
        (output / "plan.json").write_text("{}")
        raise KeyboardInterrupt()

    monkeypatch.setattr(runner, "_execute", fail)
    with pytest.raises(KeyboardInterrupt):
        runner.execute({}, output)
    assert not (output / "source.rgb").exists()
    assert (output / "plan.json").exists()


@pytest.mark.parametrize("generate_source", [False, True])
def test_job_v2_runs_enhancement_before_finishing_and_resumes(
    tmp_path, monkeypatch, generate_source
):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"original")
    clip = dict(
        id="clip",
        name="Shot",
        engine="h3",
        sourcePath="" if generate_source else str(source),
        sourceIn=0,
        duration=3,
        motionFidelity={"enabled": True},
    )
    request = dict(
        project={"clips": [clip], "settings": {}},
        runtime={},
        generateIDs=["clip"] if generate_source else [],
    )
    recipe = {"engine": "h3", "marker": "embedded repair"}
    monkeypatch.setattr(bridge, "motion_request", lambda _: {"recipe": recipe})
    monkeypatch.setattr(bridge, "compose_recipe", lambda _: (recipe, {}))
    job_file = tmp_path / "movie.json"
    jobs.export_job(request, job_file)
    job = json.loads(job_file.read_text())
    assert job["format"] == "weetodd-studio-job-v2"
    assert list(job["recipes"]) == (["clip"] if generate_source else [])
    assert job["motionRecipes"]["clip"] == recipe
    monkeypatch.setattr(jobs, "preflight", lambda *_: {})
    calls = []

    def generate(recipe_path, folder):
        calls.append("generate")
        return {"video": str(source)}

    monkeypatch.setattr(bridge, "render", generate)
    monkeypatch.setattr(bridge, "inspect_media", lambda *_: {"duration": 3})

    def run(command):
        folder = Path(command[command.index("--output-directory") + 1])
        body = json.loads(Path(command[command.index("--request") + 1]).read_text())
        assert body["recipe"] == recipe
        assert body["clip"]["sourcePath"] == str(source)
        folder.mkdir()
        movie = folder / "enhanced.mp4"
        movie.write_bytes(b"enhanced")
        (folder / "result.json").write_text(
            json.dumps({"video": str(movie), "sourceIn": 0, "sourceSHA256": jobs.file_hash(source)})
        )
        calls.append("enhance")

    def finish(request, output, **kwargs):
        resolved = request["project"]["clips"][0]
        assert Path(resolved["sourcePath"]).read_bytes() == b"enhanced"
        assert not resolved["motionFidelity"]["enabled"]
        output.write_bytes(b"finished")
        calls.append("finish")
        return {"video": str(output)}

    monkeypatch.setattr(bridge, "run", run)
    monkeypatch.setattr(bridge, "export_movie", finish)
    output = tmp_path / "job-output"
    first = jobs.execute(job, output, resume=False)
    second = jobs.execute(job, output, resume=True)
    assert first == second
    expected = (["generate"] if generate_source else []) + ["enhance", "finish"]
    assert calls == expected
    assert source.read_bytes() == b"original"
    state = json.loads((output / "job-state.json").read_text())
    Path(state["completed"]["motion-clip"]["video"]).write_bytes(b"corrupt")
    jobs.execute(job, output, resume=True)
    assert calls == expected + ["enhance"]
