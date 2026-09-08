import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]


def load_script(name, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("engine", ["ltx23", "ltx25"])
def test_ltx_backend_releases_runtime_on_error(engine, monkeypatch, tmp_path):
    module = load_script("render_headless", monkeypatch)
    runtime = __import__(f"{engine.replace('ltx', 'ltx')}_mlx.runtime", fromlist=["RUNTIME"])
    released = []
    monkeypatch.setattr(runtime.RUNTIME, "unload", lambda: released.append(True))
    monkeypatch.setattr(
        runtime.RUNTIME,
        "generate_to_file",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("cancel")),
    )
    components = (
        {"model_dir": str(tmp_path)}
        if engine == "ltx23"
        else dict(
            transformer_path="a",
            text_encoder_path="b",
            video_vae_path="c",
            audio_vae_path="d",
            spatial_upscaler_path="e",
        )
    )
    with pytest.raises(RuntimeError, match="cancel"):
        module.render_ltx(
            {"engine": engine, "components": components, "config": {}, "prompt": "test"},
            tmp_path / "out.mp4",
        )
    assert released == [True]


def test_both_ltx_runtimes_import_without_comfy_or_node_catalog():
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src") + os.pathsep + str(ROOT / "scripts")}
    run = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from render_h3_headless import NoComfyImports,assert_isolated; "
            "sys.meta_path.insert(0,NoComfyImports()); "
            "import ltx23_mlx.runtime,ltx25_mlx.runtime; assert_isolated()",
        ],
        cwd="/",
        env=env,
        capture_output=True,
        text=True,
    )
    assert run.returncode == 0, run.stderr


def test_unknown_recipe_does_not_start_render(tmp_path):
    recipe = tmp_path / "bad.json"
    recipe.write_text(json.dumps({"format": "weetodd-headless-v2", "engine": "unknown"}))
    output = tmp_path / "render"
    run = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/render_headless.py"),
            "--recipe",
            str(recipe),
            "--output-directory",
            str(output),
        ],
        capture_output=True,
        text=True,
    )
    assert run.returncode != 0
    assert not output.exists()


def test_foreign_h3_adapter_does_not_silently_drop_dora_fields(monkeypatch):
    module = load_script("render_headless", monkeypatch)
    module.validate_h3_adapter_keys(
        ["any_publisher.any_target.lora_A.weight", "any_publisher.any_target.lora_B.weight"]
    )
    with pytest.raises(ValueError, match="refusing partial application"):
        module.validate_h3_adapter_keys(
            ["block.lora_A.weight", "block.lora_B.weight", "block.lora_magnitude_vector"]
        )


def test_h3_extension_assembly_trims_repeated_context(monkeypatch, tmp_path):
    module = load_script("render_headless", monkeypatch)
    import wee_todd_mlx.conditioning_media as media_module

    target = tmp_path / "extended.mp4"
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        target.write_bytes(b"video")

    monkeypatch.setattr(media_module.subprocess, "run", run)
    module.assemble_h3_extension(
        tmp_path / "source.mp4",
        tmp_path / "generated.mp4",
        target,
        source_frames=73,
        context_frames=22,
        fps=24,
        ffmpeg="/usr/bin/ffmpeg",
    )
    command, kwargs = calls[0]
    graph = command[command.index("-filter_complex") + 1]
    assert command[0] == "/usr/bin/ffmpeg"
    assert "trim=start_frame=22" in graph
    assert "atrim=start=0.916666667" in graph
    assert kwargs["check"] is True
    assert target.read_bytes() == b"video"


def test_ltx25_extension_assembly_selects_48khz_audio(monkeypatch, tmp_path):
    module = load_script("render_headless", monkeypatch)
    import wee_todd_mlx.conditioning_media as media_module

    target = tmp_path / "extended.mp4"
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        target.write_bytes(b"video")

    monkeypatch.setattr(media_module.subprocess, "run", run)
    module.assemble_external_extension(
        tmp_path / "source.mp4",
        tmp_path / "generated.mp4",
        target,
        source_frames=73,
        context_frames=25,
        fps=24,
        sample_rate=48000,
        ffmpeg="/usr/bin/ffmpeg",
    )
    graph = calls[0][0][calls[0][0].index("-filter_complex") + 1]
    assert "trim=start_frame=25" in graph
    assert "aresample=48000" in graph


@pytest.mark.parametrize("engine", ["ltx23", "ltx25"])
def test_ltx_rejects_unsupported_top_level_lora_stack_before_loading(monkeypatch, tmp_path, engine):
    module = load_script("render_headless", monkeypatch)
    with pytest.raises(ValueError, match="refusing to render without"):
        module.render_ltx(
            {"engine": engine, "loras": [{"path": "community.safetensors"}]},
            tmp_path / "out.mp4",
        )


@pytest.mark.parametrize("engine,frames", [("h3", 73), ("ltx23", 65), ("ltx25", 65)])
def test_media_validation_checks_recipe_not_just_stream_presence(monkeypatch, engine, frames):
    module = load_script("validate_headless_matrix", monkeypatch)
    recipe = {
        "engine": engine,
        "config": {"width": 384, "height": 256, "duration_seconds": 2.5, "frame_rate": 24},
    }
    video = dict(
        codec_type="video",
        width=384,
        height=256,
        nb_read_frames=str(frames),
        avg_frame_rate="24/1",
        duration=str(frames / 24),
    )
    audio = dict(codec_type="audio", duration=str(frames / 24))
    media = {"streams": [video, audio]}
    assert module.validate_media(recipe, media)["expected_frames"] == frames
    video["width"] = 192
    with pytest.raises(ValueError, match="dimensions/frame count"):
        module.validate_media(recipe, media)
    video["width"] = 384
    audio["duration"] = "0.1"
    with pytest.raises(ValueError, match="duration drift"):
        module.validate_media(recipe, media)


def test_matrix_compare_rejects_changed_control_recipe(monkeypatch, tmp_path):
    module = load_script("compare_headless_matrix", monkeypatch)
    api = tmp_path / "api.json"
    recipe = tmp_path / "recipe.json"
    api.write_text("{}")
    recipe.write_text("{}")
    (tmp_path / "headless").mkdir()
    (tmp_path / "headless/result.json").write_text(
        json.dumps({"status": "success", "recipe_sha256": module.digest(recipe)})
    )
    (tmp_path / "control.json").write_text(json.dumps({"workflow_sha256": "old_hash"}))
    matrix = tmp_path / "matrix.json"
    matrix.write_text(
        json.dumps(
            [{"candidate": "fixture", "engine": "h3", "api": str(api), "recipe": str(recipe)}]
        )
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compare",
            "--matrix",
            str(matrix),
            "--control-output",
            str(tmp_path),
            "--output",
            str(tmp_path / "comparison.json"),
        ],
    )
    with pytest.raises(ValueError, match="Control graph changed"):
        module.main()


def test_incomplete_output_blocks_matrix_advance_and_persists_status(monkeypatch, tmp_path):
    module = load_script("validate_headless_matrix", monkeypatch)
    (tmp_path / "headless").mkdir()
    matrix = tmp_path / "matrix.json"
    matrix.write_text(
        json.dumps(
            [
                {
                    "candidate": "active",
                    "engine": "h3",
                    "status": "ready",
                    "recipe": str(tmp_path / "recipe.json"),
                    "api": "unused",
                },
                {
                    "candidate": "must_not_start",
                    "engine": "h3",
                    "status": "ready",
                    "recipe": str(tmp_path / "next/recipe.json"),
                    "api": "unused",
                },
            ]
        )
    )
    monkeypatch.setattr(
        sys, "argv", ["validate", "--matrix", str(matrix), "--comfy-root", str(tmp_path)]
    )
    monkeypatch.setattr(module.subprocess, "run", lambda *a, **k: pytest.fail("Must not render"))
    module.main()
    report = json.loads((tmp_path / "headless-validation.json").read_text())
    assert len(report) == 1
    assert report[0]["status"] == "running_or_incomplete"
