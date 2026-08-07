import json
import sys
from types import SimpleNamespace

import pytest

from wee_todd_nodes.nodes import (
    NODE_CLASS_MAPPINGS,
    WeeToddH3ComponentLoader,
    WeeToddH3GenerationConfig,
    _resolve_h3_resolution,
    _safe_output_target,
)


def test_sampling_metadata_preserves_exact_prompt(monkeypatch):
    from wee_todd_nodes.conditioning import H3Conditioning
    from wee_todd_nodes.nodes import WeeToddH3Sample
    from wee_todd_nodes.runtime import H3GenerationConfig

    prompt = "integrated_multimodal_description: exact prompt"
    conditioning = H3Conditioning(
        embeddings="embeddings",
        token_tags="tags",
        token_count=1,
        prompt=prompt,
        load_vision=False,
        encoder_spec="encoder-spec",
    )
    latents = SimpleNamespace(
        num_frames=124,
        width=640,
        height=384,
        fps=24,
        sample_rate=32000,
        transformer_evaluations=2,
        easycache_skipped_steps=0,
        easycache_resolved_threshold=None,
        seconds_per_evaluation=1.0,
        total_seconds=2.0,
    )
    monkeypatch.setattr(
        "wee_todd_nodes.nodes.H3TransformerSpec.from_components",
        lambda components: "transformer-spec",
    )
    monkeypatch.setattr(
        "wee_todd_nodes.nodes.TRANSFORMER_RUNTIME",
        SimpleNamespace(sample=lambda *args, **kwargs: latents, loaded=False),
    )

    _, metadata = WeeToddH3Sample().sample(
        "components", conditioning, H3GenerationConfig(steps=3), True
    )

    assert json.loads(metadata)["prompt"] == prompt


def test_expected_nodes_are_registered():
    assert len(NODE_CLASS_MAPPINGS) == 20
    assert "WeeToddH3ComponentLoader" in NODE_CLASS_MAPPINGS
    assert "WeeToddH3Preflight" in NODE_CLASS_MAPPINGS
    assert "WeeToddH3TextEncode" in NODE_CLASS_MAPPINGS
    assert "WeeToddH3TrajectoryForecast" in NODE_CLASS_MAPPINGS
    assert "WeeToddH3UnloadTextEncoder" in NODE_CLASS_MAPPINGS
    assert "WeeToddH3Sample" in NODE_CLASS_MAPPINGS
    assert "WeeToddH3EasyCache" in NODE_CLASS_MAPPINGS
    assert "WeeToddH3BlockCache" in NODE_CLASS_MAPPINGS
    assert "WeeToddH3LoRALoader" in NODE_CLASS_MAPPINGS
    assert "WeeToddH3UnloadTransformer" in NODE_CLASS_MAPPINGS
    assert "WeeToddH3VideoVAEDecode" in NODE_CLASS_MAPPINGS
    assert "WeeToddH3UnloadVideoVAE" in NODE_CLASS_MAPPINGS
    assert "WeeToddH3AudioVAEDecode" in NODE_CLASS_MAPPINGS
    assert "WeeToddH3UnloadAudioVAE" in NODE_CLASS_MAPPINGS
    assert "WeeToddH3PublishVideoAudio" in NODE_CLASS_MAPPINGS
    assert "WeeToddH3DirectPublishLatents" in NODE_CLASS_MAPPINGS


def test_component_loader_returns_lazy_immutable_spec():
    (spec,) = WeeToddH3ComponentLoader().specify("MiniMax-H3/FL2VA", "t2va")

    assert spec.task == "t2va"
    assert spec.transformer is None


def test_component_loader_resolves_relative_root_below_comfy_models(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "folder_paths", SimpleNamespace(models_dir=str(tmp_path)))

    (spec,) = WeeToddH3ComponentLoader().specify("MiniMax-H3/FL2VA", "t2va")

    assert spec.checkpoint == str(tmp_path / "MiniMax-H3" / "FL2VA")


def test_generation_config_node_returns_validated_value():
    config, resolved = WeeToddH3GenerationConfig().configure(
        5.0,
        8,
        42,
        "preset",
        "768P (native quality)",
        "16:9",
        1344,
        768,
        True,
        "low_memory_bf16",
        "1024",
    )
    assert config.seed == 42
    assert config.drop_adaln is True
    assert (config.width, config.height) == (1344, 768)
    assert config.aspect_ratio == "16:9"
    assert config.memory_mode == "low_memory_bf16"
    assert config.attention_query_chunk_size == 1024
    assert resolved == "1344 x 768 pixels"


@pytest.mark.parametrize(
    ("ratio", "expected"),
    [
        ("21:9", (1792, 768)),
        ("16:9", (1344, 768)),
        ("4:3", (1024, 768)),
        ("1:1", (768, 768)),
        ("3:4", (768, 1024)),
        ("9:16", (768, 1344)),
        ("9:21", (768, 1792)),
    ],
)
def test_resolution_presets_follow_ratio_and_h3_grid(ratio, expected):
    resolved = _resolve_h3_resolution("preset", "768P (native quality)", ratio, 640, 384)
    assert resolved == expected
    assert all(value % 32 == 0 for value in resolved)


def test_custom_resolution_uses_exact_dimensions():
    assert _resolve_h3_resolution("custom", "unused", "unused", 640, 384) == (640, 384)


def test_experimental_2k_preset_resolves_common_widescreen_canvas():
    assert _resolve_h3_resolution(
        "preset", "2K (experimental, very high memory)", "16:9", 640, 384
    ) == (2048, 1152)


def test_output_target_stays_below_comfy_output(tmp_path):
    target = _safe_output_target(tmp_path, "WeeTodd/H3", 42)
    assert target == tmp_path / "WeeTodd" / "H3_42.mp4"


@pytest.mark.parametrize("prefix", ["../escape", "/tmp/escape", "safe/../../escape"])
def test_output_target_rejects_escape(prefix, tmp_path):
    with pytest.raises(ValueError, match="output directory"):
        _safe_output_target(tmp_path, prefix, 42)
