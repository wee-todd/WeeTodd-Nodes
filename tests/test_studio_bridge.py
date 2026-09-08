import copy
import importlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
bridge = importlib.import_module("studio_bridge")
jobs = importlib.import_module("studio_job")


def movie_settings():
    return dict(
        fps=24,
        interpolatedFPS=48,
        width=192,
        height=128,
        upscaleWidth=384,
        upscaleHeight=256,
        interpolation="off",
        upscaling="off",
        format="mp4",
        rifeScale=1.0,
        fit="fit",
        quality=20,
    )


def fixture_project(sources):
    clips = [
        dict(
            id=str(i),
            name=f"Clip {i}",
            engine="movie",
            sourcePath=str(p),
            duration=1.0,
            sourceIn=0,
            transition="cut" if i == 0 else "dissolve",
            transitionDuration=0.25,
            volume=1,
        )
        for i, p in enumerate(sources)
    ]
    return dict(
        name="Test edit",
        settings=movie_settings(),
        clips=clips,
        assets=[],
        titles=[
            dict(
                id="title",
                text="Literal: 100% 'title'",
                start=0.1,
                duration=0.7,
                position="lower",
                fontSize=14,
            )
        ],
        audio=[],
        audioTracks=[],
    )


def test_inheritance_does_not_mutate_project():
    p = fixture_project([])
    before = copy.deepcopy(p)
    own = dict(movie_settings(), width=640)
    assert bridge.resolved_settings(p, dict(settingsOverride=own))["width"] == 640
    assert bridge.resolved_settings(p, {})["width"] == 192
    assert p == before


def test_invalid_interpolation_rejected():
    p = fixture_project([])
    p["settings"].update(interpolation="rife", interpolatedFPS=60)
    with pytest.raises(ValueError, match="2×"):
        bridge.resolved_settings(p, {})


def test_clip_job_trims_audio_and_titles():
    p = fixture_project(["a.mov", "b.mov"])
    p["audio"] = [dict(id="audio", path="audio.wav", start=0.5, sourceIn=2, duration=2)]
    result = jobs.clip_project(p, "1")
    assert len(result["clips"]) == 1
    assert result["audio"][0]["start"] == 0
    assert result["audio"][0]["sourceIn"] == 2.25
    assert result["audio"][0]["duration"] == 1
    assert result["titles"][0]["start"] == 0
    assert result["titles"][0]["duration"] == pytest.approx(0.05)
    assert p["clips"][1]["transition"] == "dissolve"


def test_export_job_embeds_immutable_movie_request(tmp_path):
    p = fixture_project(["input.mov"])
    request = dict(project=p, runtime={}, generateIDs=[])
    target = tmp_path / "movie.weetodd-job.json"
    result = jobs.export_job(request, target)
    job = json.loads(target.read_text())
    expected = job.pop("manifestSHA256")
    assert jobs.digest(job) == expected
    assert job["execution"]["parallelGenerations"] == 1
    assert result["generations"] == 0
    with pytest.raises(ValueError, match="new job"):
        jobs.export_job(request, target)


def test_task_roles_are_not_silently_dropped():
    clip = dict(attachments=[dict(role="lora"), dict(role="last")])
    assert bridge.infer_task(clip) == "fflf"
    clip["attachments"].append(dict(role="audioDriver"))
    assert bridge.infer_task(clip) == "a2v"


def test_profile_catalog_rejects_arbitrary_json(tmp_path):
    (tmp_path / "not-recipe.json").write_text('{"command":"ignored"}')
    (tmp_path / "broken.json").write_text("[")
    assert bridge.profiles(tmp_path) == []


@pytest.mark.skipif(
    not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="FFmpeg required"
)
def test_real_export_and_resume_preserve_timing_and_audio(tmp_path):
    ffmpeg = shutil.which("ffmpeg")
    sources = []
    for i, color in enumerate(("red", "blue")):
        target = tmp_path / f"source-{i}.mp4"
        subprocess.run(
            [
                ffmpeg,
                "-v",
                "error",
                "-f",
                "lavfi",
                "-i",
                f"color=c={color}:size=192x128:rate=24",
                "-f",
                "lavfi",
                "-i",
                f"sine=frequency={330 + i * 220}:sample_rate=48000",
                "-t",
                "1",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                str(target),
            ],
            check=True,
        )
        sources.append(target)
    p = fixture_project(sources)
    p["audioTracks"] = [
        dict(id="music", muted=False, solo=False, replacesSource=True),
        dict(id="muted", muted=True, solo=False, replacesSource=False),
    ]
    p["audio"] = [
        dict(
            id="music",
            trackID="music",
            path=str(sources[1]),
            start=0.2,
            sourceIn=0,
            duration=0.5,
            volume=0.4,
            fade=0.05,
        ),
        dict(
            id="muted",
            trackID="muted",
            path=str(sources[0]),
            start=0,
            sourceIn=0,
            duration=1,
            volume=1,
            fade=0,
        ),
    ]
    runtime = dict(ffmpegPath=ffmpeg, ffprobePath=shutil.which("ffprobe"))
    job_path = tmp_path / "edit.json"
    jobs.export_job(dict(project=p, runtime=runtime, generateIDs=[]), job_path)
    job = json.loads(job_path.read_text())
    result = jobs.execute(job, tmp_path / "result", resume=False)
    info = bridge.inspect_media(result["video"], runtime)
    assert info["hasAudio"] and info["fps"] == 24
    assert info["duration"] == pytest.approx(1.75, abs=0.05)
    assert info["width"] == 192 and info["height"] == 128
    digest = jobs.file_hash(result["video"])
    resumed = jobs.execute(job, tmp_path / "result", resume=True)
    assert resumed["sha256"] == digest
    assert len(list((tmp_path / "result/finished-clips").glob("*.mp4"))) == 2
    sources[0].touch()
    with pytest.raises(ValueError, match="source inputs changed"):
        jobs.execute(job, tmp_path / "result", resume=True)


def test_finishing_preflight_blocks_before_generation(tmp_path, monkeypatch):
    p = fixture_project(["missing.mov"])
    p["settings"].update(interpolation="rife")
    monkeypatch.setattr(bridge, "executable", lambda *args: "ffmpeg")
    monkeypatch.setattr(
        bridge, "run", lambda *args, **kwargs: pytest.fail("generation must not start")
    )
    with pytest.raises(ValueError, match="RIFE"):
        jobs.preflight(dict(project=p, runtime={}, recipes={"0": {"recipe": {}}}), tmp_path)


def test_resume_fingerprint_tracks_adapter_and_guides(tmp_path):
    adapter = tmp_path / "adapter.safetensors"
    adapter.write_bytes(b"adapter")
    job = dict(
        project={}, recipes={"x": {"components": {"ic_loras": [[str(adapter), 1]]}}}, runtime={}
    )
    before = jobs.inputs_fingerprint(job)
    adapter.write_bytes(b"updated adapter")
    assert jobs.inputs_fingerprint(job) != before


def test_directory_artifact_verification_detects_missing_frame(tmp_path):
    (tmp_path / "frame.png").write_bytes(b"frame")
    (tmp_path / "audio.wav").write_bytes(b"audio")
    before = jobs.artifact_hash(tmp_path)
    (tmp_path / "frame.png").unlink()
    assert jobs.artifact_hash(tmp_path) != before


def test_job_output_has_exclusive_lock(tmp_path, monkeypatch):
    import fcntl

    monkeypatch.setattr(jobs, "_execute_locked", lambda *args: pytest.fail("lock bypassed"))
    with (tmp_path / ".job.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(ValueError, match="Another process"):
            jobs.execute({}, tmp_path, False)


def test_changed_renderer_rejected_before_inputs(tmp_path):
    with pytest.raises(ValueError, match="renderer version changed"):
        jobs.preflight(dict(execution={"rendererSHA256": "old"}), tmp_path)


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="FFmpeg required")
def test_sequence_import_and_bridge_extract_actual_endpoints(tmp_path):
    from PIL import Image

    frames = tmp_path / "frames"
    frames.mkdir()
    for number, color in ((1, "red"), (2, "blue")):
        Image.new("RGB", (192, 128), color).save(frames / f"frame-{number}.png")
    result = bridge.import_sequence(
        dict(path=str(frames), fps=2, runtime={}), tmp_path / "sequence"
    )
    info = bridge.inspect_media(result["video"], {})
    assert info["fps"] == 2 and info["duration"] == 1
    p = fixture_project([result["video"], result["video"]])
    p["settings"]["fps"] = 2
    anchors = bridge.bridge_frames(dict(project=p, clipID="0", runtime={}), tmp_path / "anchors")
    with Image.open(anchors["first"]) as first, Image.open(anchors["last"]) as last:
        r, _, b = first.getpixel((96, 64))[:3]
        assert b > r + 150
        r, _, b = last.getpixel((96, 64))[:3]
        assert r > b + 150
