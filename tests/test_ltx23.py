import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from ltx23_mlx.runtime import (
    LTX23GenerationConfig,
    LTX23ModelSpec,
    LTX23RuntimeCache,
    _comfy_sampler_progress,
)
from ltx23_mlx.upscale import LTX23UpscalerSpec, _host_audio, _host_video, _mux_command
from wee_todd_nodes.ltx_nodes import WeeToddLTX23GenerationConfig


def _touch_bundle(root, mode):
    common = [
        "connector.safetensors",
        "vae_encoder.safetensors",
        "vae_decoder.safetensors",
        "audio_vae.safetensors",
        "vocoder.safetensors",
    ]
    for name in common:
        (root / name).touch()
    gemma = root / "gemma"
    gemma.mkdir(exist_ok=True)
    if mode in {"two_stage", "two_stage_hq"}:
        names = [
            "transformer-dev.safetensors",
            "ltx-2.3-22b-distilled-lora-384.safetensors",
            "spatial_upscaler_x2_v1_1_config.json",
            "spatial_upscaler_x2_v1_1.safetensors",
        ]
    elif mode == "distilled":
        names = [
            "transformer-distilled.safetensors",
            "spatial_upscaler_x2_v1_1_config.json",
            "spatial_upscaler_x2_v1_1.safetensors",
        ]
    else:
        names = ["transformer-dev.safetensors"]
    for name in names:
        (root / name).touch()


@pytest.mark.parametrize(
    ("mode", "modulus"),
    [("two_stage", 64), ("two_stage_hq", 64), ("distilled", 64), ("one_stage", 32)],
)
def test_ltx23_config_normalizes_frames_and_enforces_grid(mode, modulus):
    config = LTX23GenerationConfig(pipeline_mode=mode, width=704, height=448)
    config.validate()
    assert config.num_frames == 121
    assert config.delivered_duration_seconds == 5.0
    with pytest.raises(ValueError, match="divisible"):
        LTX23GenerationConfig(
            pipeline_mode=mode,
            width=704 + modulus // 2,
            height=448,
        ).validate()


@pytest.mark.parametrize("mode", ["two_stage", "two_stage_hq", "distilled", "one_stage"])
def test_ltx23_model_preflight_is_mode_specific(tmp_path, mode):
    _touch_bundle(tmp_path, mode)
    spec = LTX23ModelSpec(str(tmp_path), str(tmp_path / "gemma"))
    spec.validate(mode)
    inventory = spec.inventory(mode)
    assert inventory["components"]
    assert all(item["files"] for item in inventory["components"])


def test_ltx23_generation_node_uses_mode_recommended_steps(monkeypatch):
    monkeypatch.setattr("wee_todd_nodes.ltx_nodes.secrets.randbelow", lambda _limit: 1234)
    node = WeeToddLTX23GenerationConfig()
    config, raw = node.configure(
        "distilled", 704, 448, 5.0, 24.0, -1, 0, 0, 3.0, 1.0, True, True
    )
    info = json.loads(raw)
    assert config.seed == 1234
    assert config.stage1_steps == 8
    assert config.stage2_steps == 3
    assert config.num_frames == 121
    assert info["low_ram_streaming"] is True


def test_ltx23_runtime_is_lazy_and_filters_pipeline_signature(tmp_path, monkeypatch):
    import mlx.core as mx

    _touch_bundle(tmp_path, "two_stage")
    calls = []

    class FakePipeline:
        def __init__(self, **kwargs):
            calls.append(("init", kwargs))

        def generate_and_save(self, prompt, output_path, height, width, num_frames, frame_rate):
            calls.append(
                (
                    "generate",
                    {
                        "prompt": prompt,
                        "output_path": output_path,
                        "height": height,
                        "width": width,
                        "num_frames": num_frames,
                        "frame_rate": frame_rate,
                    },
                )
            )
            return output_path

    monkeypatch.setitem(
        sys.modules,
        "ltx_pipelines_mlx",
        SimpleNamespace(TI2VidTwoStagesPipeline=FakePipeline),
    )
    monkeypatch.setattr(mx, "reset_peak_memory", lambda: None)
    monkeypatch.setattr(mx, "get_peak_memory", lambda: 42)
    monkeypatch.setattr(mx, "clear_cache", lambda: None)
    runtime = LTX23RuntimeCache()
    config = LTX23GenerationConfig()
    info = runtime.generate_to_file(
        LTX23ModelSpec(str(tmp_path), str(tmp_path / "gemma")),
        config,
        "test prompt",
        tmp_path / "out.mp4",
    )
    assert info["mlx_peak_bytes"] == 42
    assert calls[1][1]["num_frames"] == 121
    assert not runtime.loaded


def test_ltx23_upscaler_preflight_and_host_contracts(tmp_path):
    for name in ("vae_encoder.safetensors", "vae_decoder.safetensors"):
        (tmp_path / name).touch()
    (tmp_path / "spatial_upscaler_x2_v1_1.safetensors").touch()
    (tmp_path / "spatial_upscaler_x2_v1_1_config.json").write_text(
        json.dumps(
            {
                "config": {
                    "spatial_upsample": True,
                    "temporal_upsample": False,
                    "spatial_scale": 2.0,
                }
            }
        )
    )
    spec = LTX23UpscalerSpec(str(tmp_path))
    assert spec.validate()["spatial_scale"] == 2.0
    video = _host_video(np.zeros((9, 64, 96, 3), dtype=np.float32))
    waveform, sample_rate = _host_audio(
        {"waveform": np.zeros((1, 1, 100), dtype=np.float32), "sample_rate": 32000}
    )
    assert video.shape == (9, 64, 96, 3)
    assert waveform.shape == (2, 100)
    assert sample_rate == 32000


def test_ltx23_mux_preserves_requested_video_frames_without_shortest(tmp_path):
    command = _mux_command(
        tmp_path / "ffmpeg",
        tmp_path / "silent.mp4",
        tmp_path / "audio.wav",
        tmp_path / "output.mp4",
        124,
    )
    assert command[command.index("-frames:v") + 1] == "124"
    assert "-shortest" not in command
    assert command[command.index("-map") + 1] == "0:v:0"


def test_ltx23_low_ram_restores_process_cache_limit(tmp_path, monkeypatch):
    import mlx.core as mx

    _touch_bundle(tmp_path, "two_stage")
    calls = []

    class FakePipeline:
        def __init__(self, **_kwargs):
            pass

    monkeypatch.setitem(
        sys.modules,
        "ltx_pipelines_mlx",
        SimpleNamespace(TI2VidTwoStagesPipeline=FakePipeline),
    )

    current = 987654

    def set_limit(value):
        nonlocal current
        previous = current
        current = value
        calls.append(value)
        return previous

    monkeypatch.setattr(mx, "set_cache_limit", set_limit)
    monkeypatch.setattr(mx, "clear_cache", lambda: None)
    runtime = LTX23RuntimeCache()
    config = LTX23GenerationConfig(low_ram_streaming=True)
    runtime.get(LTX23ModelSpec(str(tmp_path), str(tmp_path / "gemma")), config)
    assert current == 0
    runtime.unload()
    assert current == 987654
    assert calls == [0, 987654]


def test_ltx23_preflight_rejects_uncached_gemma_without_downloading(tmp_path, monkeypatch):
    _touch_bundle(tmp_path, "one_stage")
    calls = []

    def local_only(model_id, *, local_files_only):
        calls.append((model_id, local_files_only))
        raise RuntimeError("not cached")

    monkeypatch.setattr("huggingface_hub.snapshot_download", local_only)
    spec = LTX23ModelSpec(str(tmp_path), "example/missing-gemma")
    with pytest.raises(FileNotFoundError, match="complete cached snapshot"):
        spec.validate("one_stage")
    assert calls == [("example/missing-gemma", True)]


def test_ltx23_sampler_bridge_restores_tqdm_and_checks_each_step(monkeypatch):
    fake_samplers = SimpleNamespace(tqdm=lambda values, **_kwargs: values)
    fake_utils = SimpleNamespace(samplers=fake_samplers)
    monkeypatch.setitem(sys.modules, "ltx_pipelines_mlx.utils", fake_utils)
    checks = []
    progress = []
    original = fake_samplers.tqdm

    with _comfy_sampler_progress(
        lambda: checks.append(True),
        lambda completed, total: progress.append((completed, total)),
        3,
    ):
        assert list(fake_samplers.tqdm(range(3))) == [0, 1, 2]

    assert fake_samplers.tqdm is original
    assert len(checks) == 3
    assert progress == [(1, 3), (2, 3), (3, 3)]
