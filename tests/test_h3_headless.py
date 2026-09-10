import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).parents[1]


def runner():
    spec = importlib.util.spec_from_file_location(
        "headless_test", ROOT / "scripts/render_h3_headless.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def proof():
    return SimpleNamespace(
        transformer_evaluations=4,
        fast_h3_approximation_report={
            "active_layer_indices": list(range(38)) + [48, 49],
            "skipped_layer_indices": list(range(38, 48)),
            "executed_layers": 40,
            "skipped_layers": 10,
        },
        sol_attention_report={
            "executed_calls": 160,
            "fallback_calls": 0,
            "storage_layout": "compact_preordered",
        },
    )


def test_execution_proof_rejects_fallback_or_wrong_layers():
    module = runner()
    latents = proof()
    module.validate_execution(latents)
    latents.sol_attention_report["fallback_calls"] = 1
    with pytest.raises(RuntimeError, match="refusing publication"):
        module.validate_execution(latents)
    latents = proof()
    latents.fast_h3_approximation_report["active_layer_indices"][-1] = 48
    with pytest.raises(RuntimeError):
        module.validate_execution(latents)


def test_optional_comfy_imports_are_blocked():
    module = runner()
    blocker = module.NoComfyImports()
    for name in (
        "comfy",
        "comfy.utils",
        "comfy_execution.utils",
        "folder_paths",
        "server",
        "nodes",
    ):
        with pytest.raises(ModuleNotFoundError):
            blocker.find_spec(name)
    assert blocker.find_spec("wee_todd_nodes.sampling") is None


def test_metadata_drift_is_rejected(tmp_path):
    module = runner()
    manifest = tmp_path / "manifest.json"
    manifest.write_text("changed")
    recipe = tmp_path / "recipe.json"
    recipe.write_text(
        json.dumps(
            {
                "format": "weetodd-h3-headless-recipe-v1",
                "model_metadata_sha256": {str(manifest): "wrong"},
            }
        )
    )
    with pytest.raises(ValueError, match="Model metadata changed"):
        module.load_recipe(recipe)


def test_core_package_import_does_not_load_node_catalog():
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import wee_todd_nodes; import wee_todd_nodes.runtime; "
            "assert 'wee_todd_nodes.nodes' not in sys.modules; "
            "assert not any(n.split('.')[0] in {'comfy','folder_paths'} for n in sys.modules)",
        ],
        cwd="/",
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_unknown_package_attribute_raises():
    import wee_todd_nodes

    with pytest.raises(AttributeError):
        _ = wee_todd_nodes.unknown_headless_attribute


def test_lazy_node_exports_keep_existing_registry():
    import wee_todd_nodes
    from wee_todd_nodes import nodes

    assert wee_todd_nodes.NODE_CLASS_MAPPINGS is nodes.NODE_CLASS_MAPPINGS
    assert wee_todd_nodes.NODE_DISPLAY_NAME_MAPPINGS is nodes.NODE_DISPLAY_NAME_MAPPINGS


def test_comparison_verifies_files_and_rejects_changed_media(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    import compare_h3_headless as module

    def save(file, data):
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text(json.dumps(data))

    recipe = tmp_path / "recipe.json"
    save(recipe, {"workflow_sha256": "control"})
    video = tmp_path / "comfy/output/control.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"same media")
    phase = {"phases": [{"seconds": 8}], "run_peak_bytes": 123}
    save(
        video.with_suffix(".json"),
        {
            "sampling": {
                "sol_attention": {"executed_calls": 160, "fallback_calls": 0},
                "total_seconds": 7,
            },
            "phase_memory": phase,
        },
    )
    save(
        tmp_path / "comfy/history.json",
        {
            "workflow_sha256": "control",
            "runs": [
                {
                    "prompt_id": "one",
                    "server_seconds": 10,
                    "history": {
                        "outputs": {"8": {"gifs": [{"subfolder": "", "filename": video.name}]}}
                    },
                }
            ],
        },
    )
    latent = tmp_path / "latents.safetensors"
    latent.write_bytes(b"same latent tensors")
    selected = [0, 1, 2, 3, 5, 6, 7, 9, *range(18, 50)]
    trace = {
        "mode": "off",
        "status": "success",
        "prompt_id": "one",
        "latents": str(latent),
        "blocks": [
            {"evaluation": step, "block": block, "attention_backend": "metal_indexed"}
            for step in range(4)
            for block in selected
        ],
    }
    save(tmp_path / "comfy/raw/one.json", trace)
    headless = tmp_path / "headless"
    save(headless / "raw/one.json", trace)
    output = headless / "headless.mp4"
    output.write_bytes(video.read_bytes())
    save(
        headless / "results.json",
        {
            "recipe_sha256": module.digest(recipe),
            "runs": [
                {
                    "render_seconds": 9,
                    "sampler_seconds": 7,
                    "phase_memory": phase,
                    "output": str(output),
                    "runtime_loaded_after_render": [False] * 5,
                    "isolation": {"comfy_modules_loaded": [], "node_catalog_loaded": False},
                }
            ],
        },
    )
    report = module.compare(tmp_path, [headless])
    assert report["all_exact"]
    assert report["median_saving_seconds"] == 1
    assert report["median_saving_percent"] == 10
    output.write_bytes(b"changed media")
    with pytest.raises(ValueError, match="parity failed"):
        module.compare(tmp_path, [headless])


@pytest.mark.parametrize("mode,memory,cache", [
    ("invalid", "normal", 0),
    ("resident", "low_memory_bf16", 0),
    ("resident", "normal", 1),
])
def test_headless_residency_policy_rejected_before_weight_loading(tmp_path, mode, memory, cache):
    from wee_todd_nodes.runtime import H3GenerationConfig

    config = H3GenerationConfig(memory_mode=memory, paging_cache_gb=cache)
    with pytest.raises(ValueError, match="(block_residency|normal memory|block residency)"):
        config.validate_paging(tmp_path, mode)


@pytest.mark.parametrize("mode", ["checkpoint_default", "resident"])
@pytest.mark.parametrize("failure", [None, RuntimeError, KeyboardInterrupt])
def test_headless_forwards_policy_and_releases_before_decode(tmp_path, monkeypatch, mode, failure):
    from contextlib import nullcontext

    from wee_todd_mlx import conditioning_media, task_conditioning
    from wee_todd_nodes import (
        conditioning,
        decoding,
        direct_publishing,
        preflight,
        runtime,
        sampling,
    )

    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    import profile_fasth3_server
    import render_headless as module

    monkeypatch.setattr(task_conditioning, "validate_conditioning", lambda r: {
        "contract": {"task": "t2v"}, "input_ids": [],
    })
    monkeypatch.setattr(conditioning_media, "inspect_media", lambda *a: {})
    monkeypatch.setattr(preflight, "preflight_components", lambda *a: None)
    monkeypatch.setattr(profile_fasth3_server, "install_profiling", lambda *a: nullcontext())
    encoded = SimpleNamespace(cache_report={}, condition_video_rows=None, condition_audio_rows=None)
    monkeypatch.setattr(conditioning.TEXT_ENCODER_RUNTIME, "encode", lambda *a, **k: encoded)
    released = []
    runtimes = [conditioning.TEXT_ENCODER_RUNTIME, sampling.TRANSFORMER_RUNTIME,
                decoding.VIDEO_VAE_RUNTIME, decoding.AUDIO_VAE_RUNTIME, runtime.RUNTIME]
    for index, active in enumerate(runtimes):
        monkeypatch.setattr(active, "unload", lambda i=index: released.append(i))

    def sample(*args, **kwargs):
        assert kwargs["block_residency"] == mode
        assert kwargs["unload_after"] is True
        if failure:
            raise failure("test interruption")
        return SimpleNamespace(transformer_evaluations=2, total_seconds=1,
                               paging_report={}, sol_attention_report={}, vdn_report={},
                               preview_report={}, projection_backend_report={"requested": "auto"},
                               projection_backend_runtime={"fallback_calls": 1},
                               block_residency_report={"requested": mode})

    monkeypatch.setattr(sampling.TRANSFORMER_RUNTIME, "sample", sample)

    def publish(*args, **kwargs):
        assert 1 in released, "transformer must release before decoding"
        return SimpleNamespace(video_path=tmp_path / "render.mp4", metadata={})

    monkeypatch.setattr(direct_publishing, "publish_latents_direct", publish)
    recipe = {"engine": "h3", "components": {"checkpoint": str(tmp_path), "task": "t2va"},
              "config": {"steps": 3}, "prompt": "test", "ffmpeg": "ffmpeg",
              "block_residency": mode}
    if failure:
        with pytest.raises(failure):
            module.render_h3(recipe, tmp_path / "render.mp4")
    else:
        result = module.render_h3(recipe, tmp_path / "render.mp4")
        assert result["block_residency"]["requested"] == mode
        assert result["projection_backend_runtime"]["fallback_calls"] == 1
    assert set(released) == set(range(5))


def test_conditioning_schema_accepts_explicit_h3_residency():
    from wee_todd_mlx.task_conditioning import normalize_conditioning

    recipe = {"engine": "h3", "components": {"task": "t2va"}, "config": {},
              "conditioning": {"version": 1, "task": "t2v", "inputs": []},
              "block_residency": "resident"}
    assert normalize_conditioning(recipe)["task"] == "t2v"


def test_headless_preflight_forwards_residency_before_component_loading(monkeypatch, tmp_path):
    from wee_todd_mlx import conditioning_media, task_conditioning
    from wee_todd_mlx.headless_preflight import preflight_recipe
    from wee_todd_nodes import preflight

    monkeypatch.setattr(task_conditioning, "validate_conditioning", lambda r: {"contract": {}})
    monkeypatch.setattr(conditioning_media, "inspect_media", lambda *a: {})
    monkeypatch.setattr(preflight, "preflight_components", lambda *a: pytest.fail("too late"))
    with pytest.raises(ValueError, match="normal memory"):
        preflight_recipe({"engine": "h3", "components": {"checkpoint": str(tmp_path)},
                          "config": {"memory_mode": "low_memory_bf16"},
                          "block_residency": "resident"})


@pytest.mark.parametrize("engine", ["ltx23", "ltx25"])
def test_block_residency_is_rejected_for_other_engines(engine):
    from wee_todd_mlx.task_conditioning import normalize_conditioning

    with pytest.raises(ValueError, match="block_residency.*H3"):
        normalize_conditioning({
            "engine": engine, "components": {}, "config": {},
            "conditioning": {"version": 1, "task": "t2v", "inputs": []},
            "block_residency": "resident",
        })
