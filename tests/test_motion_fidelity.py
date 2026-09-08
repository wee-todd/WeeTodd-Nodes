"""Motion timing, budget and shared adapter contracts; no model weights required."""

import importlib
import json
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from wee_todd_mlx.motion_fidelity import (
    MotionSettings,
    aligned_frames,
    audio_filter,
    expansion_indices,
    latent_index,
    plan_motion,
    validate_recipe,
)

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
        validate_recipe(dict(recipe, loras={"adapters": [{}]}))


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


def test_node_settings_contract_and_validation():
    from wee_todd_nodes.motion_nodes import WeeToddH3MotionRefine, WeeToddH3MotionSettings

    settings = WeeToddH3MotionSettings().configure("adaptive", 0.5, 2, 0.5, 42, 345)[0]
    assert MotionSettings(**settings) == replace(MotionSettings(), enabled=True)
    assert WeeToddH3MotionRefine.INPUT_TYPES()["required"]["config"] == ("WEETODD_H3_CONFIG",)
    with pytest.raises(ValueError):
        WeeToddH3MotionSettings().configure("uniform", 2, 2, 0.5, 42, 345)


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
