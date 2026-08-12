import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from safetensors.numpy import save_file

from ltx25_mlx.runtime import (
    LTX25ComponentSpec,
    LTX25GenerationConfig,
    LTX25RuntimeCache,
    backend_capability,
)
from wee_todd_nodes.ltx25_nodes import WeeToddLTX25GenerationConfig


def _component(path, **metadata):
    encoded = {
        key: json.dumps(value) if not isinstance(value, str) else value
        for key, value in metadata.items()
    }
    save_file({"test": np.zeros((1,), dtype=np.float32)}, path, metadata=encoded)


def _bundle(root, *, version="2.5.0"):
    gemma = {"gemma_version": "gemma4-12b-ltx-v1"}
    _component(
        root / "transformer.safetensors",
        model_version=version,
        gemma_source_checkpoint=gemma,
        config={"transformer": {"ff_bias": False, "use_prompt_adaln_single": False}},
    )
    _component(
        root / "text_encoder.safetensors",
        model_version=version,
        gemma_source_checkpoint=gemma,
    )
    _component(
        root / "video_vae.safetensors",
        model_version=version,
        config={
            "vae": {
                "_class_name": "ConvVideoDecoder",
                "patch_size": 4,
                "encoder_blocks": [
                    ["compress_space", 1],
                    ["compress_all", 1],
                    ["compress_all", 1],
                    ["compress_time", 1],
                ],
            }
        },
    )
    for name in ("audio_vae", "spatial_upscaler"):
        _component(root / f"{name}.safetensors", model_version=version)
    return LTX25ComponentSpec(
        transformer_path=str(root / "transformer.safetensors"),
        text_encoder_path=str(root / "text_encoder.safetensors"),
        video_vae_path=str(root / "video_vae.safetensors"),
        audio_vae_path=str(root / "audio_vae.safetensors"),
        spatial_upscaler_path=str(root / "spatial_upscaler.safetensors"),
    )


def test_ltx25_split_preflight_reads_metadata_without_weights(tmp_path):
    spec = _bundle(tmp_path)
    report = spec.validate()
    assert report["model_version"] == "2.5.0"
    assert report["gemma_source_checkpoint"] == {"gemma_version": "gemma4-12b-ltx-v1"}
    assert report["video_scale_factors"] == [8, 32, 32]
    assert report["video_decoder"] == "convolutional"
    assert len(report["components"]) == 5


def test_ltx25_preflight_rejects_23_transformer(tmp_path):
    spec = _bundle(tmp_path, version="2.3.0")
    with pytest.raises(ValueError, match="not identified as LTX 2.5"):
        spec.validate()


def test_ltx25_config_pins_official_distilled_schedule_and_grid():
    config = LTX25GenerationConfig()
    config.validate()
    assert config.num_frames == 121
    assert config.delivered_duration_seconds == 5.0
    with pytest.raises(ValueError, match=r"8\+4"):
        LTX25GenerationConfig(stage2_steps=3).validate()
    with pytest.raises(ValueError, match="divisible"):
        LTX25GenerationConfig(width=736).validate()


def test_ltx25_generation_config_node_resolves_random_seed(monkeypatch):
    monkeypatch.setattr("wee_todd_nodes.ltx25_nodes.secrets.randbelow", lambda _limit: 2468)
    config, raw = WeeToddLTX25GenerationConfig().configure(768, 512, 5.0, 24.0, -1, True, False)
    assert config.seed == 2468
    assert json.loads(raw)["real_evaluations"] == 12


def test_ltx25_runtime_requires_versioned_backend_and_filters_signature(tmp_path, monkeypatch):
    import mlx.core as mx

    spec = _bundle(tmp_path)
    calls = []

    class FakePipeline:
        def __init__(self, transformer_path, video_vae_path, low_memory):
            calls.append(("init", transformer_path, video_vae_path, low_memory))

        def generate_and_save(self, prompt, output_path, height, width, num_frames):
            calls.append(("generate", prompt, height, width, num_frames))
            return output_path

    monkeypatch.setitem(
        sys.modules,
        "ltx_pipelines_mlx",
        SimpleNamespace(LTX25DistilledPipeline=FakePipeline),
    )
    monkeypatch.setattr(mx, "reset_peak_memory", lambda: None)
    monkeypatch.setattr(mx, "get_peak_memory", lambda: 123)
    monkeypatch.setattr(mx, "clear_cache", lambda: None)
    runtime = LTX25RuntimeCache()
    info = runtime.generate_to_file(
        spec,
        LTX25GenerationConfig(),
        "A literal chronological test prompt.",
        tmp_path / "output.mp4",
    )
    assert calls[0][0] == "init"
    assert calls[1] == ("generate", "A literal chronological test prompt.", 512, 768, 121)
    assert info["mlx_peak_bytes"] == 123
    assert not runtime.loaded


def test_ltx25_backend_capability_rejects_23_only_entrypoint(monkeypatch):
    monkeypatch.setitem(sys.modules, "ltx_pipelines_mlx", SimpleNamespace())
    status = backend_capability()
    assert status["ready"] is False
    assert "LTX25DistilledPipeline" in status["reason"]
