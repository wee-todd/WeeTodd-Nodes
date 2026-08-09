import json
import sys
from types import SimpleNamespace

import pytest

from wee_todd_nodes.nodes import (
    NODE_CLASS_MAPPINGS,
    WeeToddH3ComponentLoader,
    WeeToddH3GenerationConfig,
    WeeToddH3QuantizedTransformerLoader,
    WeeToddH3TrajectoryForecast,
    _output_directory,
    _publication_environment,
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
        paging_report={"format": "weetodd-h3-qwen-paged-v1"},
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
        paging_report={"format": "weetodd-h3-paged-v1"},
        text_encoder_paging_report={"format": "weetodd-h3-qwen-paged-v1"},
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

    parsed = json.loads(metadata)
    assert parsed["prompt"] == prompt
    assert parsed["paged_weights"] == {
        "transformer": {"format": "weetodd-h3-paged-v1"},
        "text_encoder": {"format": "weetodd-h3-qwen-paged-v1"},
    }


def test_expected_nodes_are_registered():
    assert len(NODE_CLASS_MAPPINGS) == 22
    assert "WeeToddH3ComponentLoader" in NODE_CLASS_MAPPINGS
    assert "WeeToddH3QuantizedTransformerLoader" in NODE_CLASS_MAPPINGS
    assert "WeeToddH3Preflight" in NODE_CLASS_MAPPINGS
    assert "WeeToddH3TextEncode" in NODE_CLASS_MAPPINGS
    assert "WeeToddH3TrajectoryForecast" in NODE_CLASS_MAPPINGS
    assert "WeeToddH3UnloadTextEncoder" in NODE_CLASS_MAPPINGS
    assert "WeeToddH3Sample" in NODE_CLASS_MAPPINGS
    assert "WeeToddH3EasyCache" in NODE_CLASS_MAPPINGS
    assert "WeeToddH3BlockCache" in NODE_CLASS_MAPPINGS
    assert "WeeToddH3HierarchicalBlockCache" in NODE_CLASS_MAPPINGS
    assert "WeeToddH3LoRALoader" in NODE_CLASS_MAPPINGS
    assert "WeeToddH3UnloadTransformer" in NODE_CLASS_MAPPINGS
    assert "WeeToddH3VideoVAEDecode" in NODE_CLASS_MAPPINGS
    assert "WeeToddH3UnloadVideoVAE" in NODE_CLASS_MAPPINGS
    assert "WeeToddH3AudioVAEDecode" in NODE_CLASS_MAPPINGS
    assert "WeeToddH3UnloadAudioVAE" in NODE_CLASS_MAPPINGS
    assert "WeeToddH3PublishVideoAudio" in NODE_CLASS_MAPPINGS
    assert "WeeToddH3DirectPublishLatents" in NODE_CLASS_MAPPINGS


def test_trajectory_forecast_node_exposes_opt_in_bootstrap():
    (config,) = WeeToddH3TrajectoryForecast().configure(
        "automatic_speed",
        1.0,
        2,
        1,
        2,
        0.5,
        2.5,
        True,
    )

    assert config.bootstrap_first_forecast is True
    assert (
        WeeToddH3TrajectoryForecast.INPUT_TYPES()["optional"]["bootstrap_first_forecast"][1][
            "default"
        ]
        is False
    )


def test_trajectory_forecast_node_exposes_opt_in_offline_audio_isolation():
    (config,) = WeeToddH3TrajectoryForecast().configure(
        "automatic_balanced",
        0.75,
        2,
        1,
        2,
        0.35,
        1.75,
        False,
        True,
        0.5,
        0.0,
    )

    assert config.offline_smoothing_replay is True
    assert config.offline_video_blend == 0.5
    assert config.offline_audio_blend == 0.0
    optional = WeeToddH3TrajectoryForecast.INPUT_TYPES()["optional"]
    assert list(optional).index("bootstrap_first_forecast") < list(optional).index(
        "offline_smoothing_replay"
    )
    assert optional["offline_smoothing_replay"][1]["default"] is False


def test_component_loader_returns_lazy_immutable_spec():
    (spec,) = WeeToddH3ComponentLoader().specify("MiniMax-H3/FL2VA", "t2va")

    assert spec.task == "t2va"
    assert spec.transformer is None


def test_component_loader_resolves_relative_root_below_comfy_models(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "folder_paths", SimpleNamespace(models_dir=str(tmp_path)))

    (spec,) = WeeToddH3ComponentLoader().specify(
        "MiniMax-H3/FL2VA",
        "t2va",
        transformer="MiniMax-H3/transformers/q8_extended_paged",
        text_encoder="MiniMax-H3/text_encoders/q8-paged",
    )

    assert spec.checkpoint == str(tmp_path / "MiniMax-H3" / "FL2VA")
    assert spec.transformer == str(tmp_path / "MiniMax-H3" / "transformers" / "q8_extended_paged")
    assert spec.text_encoder == str(tmp_path / "MiniMax-H3" / "text_encoders" / "q8-paged")


def test_component_loader_resolves_shared_comfy_model_roots(monkeypatch, tmp_path):
    instance_models = tmp_path / "instance" / "models"
    shared_checkpoints = tmp_path / "shared" / "checkpoints"
    shared_text_encoders = tmp_path / "shared" / "text_encoders"
    checkpoint = shared_checkpoints / "MiniMax-H3" / "FL2VA"
    text_encoder = shared_text_encoders / "MiniMax-H3" / "text_encoders" / "q8-paged"
    checkpoint.mkdir(parents=True)
    text_encoder.mkdir(parents=True)

    roots = {
        "checkpoints": [str(instance_models / "checkpoints"), str(shared_checkpoints)],
        "text_encoders": [
            str(instance_models / "text_encoders"),
            str(shared_text_encoders),
        ],
    }
    monkeypatch.setitem(
        sys.modules,
        "folder_paths",
        SimpleNamespace(
            models_dir=str(instance_models),
            get_folder_paths=lambda category: roots.get(category, []),
            folder_names_and_paths={
                category: (paths, {".safetensors"}) for category, paths in roots.items()
            },
        ),
    )

    (spec,) = WeeToddH3ComponentLoader().specify(
        "MiniMax-H3/FL2VA",
        "t2va",
        text_encoder="MiniMax-H3/text_encoders/q8-paged",
    )

    assert spec.checkpoint == str(checkpoint)
    assert spec.text_encoder == str(text_encoder)


def test_component_loader_rejects_relative_parent_traversal():
    with pytest.raises(ValueError, match="cannot contain"):
        WeeToddH3ComponentLoader().specify("../outside", "t2va")


def test_publication_environment_uses_registered_comfy_output_root(monkeypatch, tmp_path):
    shared_output = tmp_path / "shared_output"
    monkeypatch.setitem(
        sys.modules,
        "folder_paths",
        SimpleNamespace(get_output_directory=lambda: str(shared_output)),
    )
    monkeypatch.setattr(
        "minimax_h3_mlx.media.ffmpeg_status",
        lambda path=None: {"available": True, "path": "/portable/ffmpeg", "source": "test"},
    )

    assert _output_directory() == shared_output
    assert _publication_environment() == {
        "output_directory": str(shared_output.resolve()),
        "ffmpeg": {"available": True, "path": "/portable/ffmpeg", "source": "test"},
    }


def test_quantized_loader_selects_validated_named_profile(tmp_path):
    from minimax_h3_mlx.mixed_checkpoint import (
        MIXED_CHECKPOINT_FORMAT,
        Q8_EXTENDED_PROFILE,
        extended_q8_mlp_recipe,
    )

    transformer = tmp_path / "q8_extended"
    transformer.mkdir()
    (transformer / "config.json").write_text("{}\n")
    (transformer / "model.safetensors.index.json").write_text("{}\n")
    (transformer / "quant_config.json").write_text(
        json.dumps(
            {
                "format": MIXED_CHECKPOINT_FORMAT,
                "format_version": 1,
                "profile": Q8_EXTENDED_PROFILE,
                "bits": 8,
                "group_size": 64,
                "quantize_core": False,
                "quantize_adaln": False,
                "overrides": extended_q8_mlp_recipe().overrides,
            }
        )
    )
    components = WeeToddH3ComponentLoader().specify(str(tmp_path), "t2va")[0]

    selected, info = WeeToddH3QuantizedTransformerLoader().select(
        components,
        "q8_extended",
        str(tmp_path),
        str(transformer),
    )

    assert selected.transformer == str(transformer)
    assert json.loads(info)["selected_modules"] == 82


def test_generation_config_node_returns_validated_value():
    config, resolved = WeeToddH3GenerationConfig().configure(
        5.0,
        8,
        42,
        "ratio + size",
        "768 px short edge — native",
        "16:9 — widescreen landscape",
        1344,
        768,
        True,
        "low_memory_bf16",
        "1024",
        short_edge=768,
    )
    assert config.seed == 42
    assert config.drop_adaln is True
    assert (config.width, config.height) == (1344, 768)
    assert config.aspect_ratio == "16:9"
    assert config.memory_mode == "low_memory_bf16"
    assert config.attention_query_chunk_size == 1024
    assert config.projection_backend == "mlx"
    assert resolved == "1344 × 768 pixels — 16:9 — 768 px short edge"


def test_generation_config_exposes_clear_ratio_size_controls():
    inputs = WeeToddH3GenerationConfig.INPUT_TYPES()

    assert inputs["required"]["resolution_mode"][0][:2] == [
        "ratio + size",
        "exact dimensions",
    ]
    assert "16:9 — widescreen landscape" in inputs["required"]["aspect_ratio"][0]
    assert "1:1 — square" in inputs["required"]["aspect_ratio"][0]
    assert "9:16 — vertical portrait" in inputs["required"]["aspect_ratio"][0]
    slider = inputs["optional"]["short_edge"][1]
    assert slider["default"] == 768
    assert slider["min"] == 32
    assert slider["max"] == 1088
    assert slider["step"] == 32
    assert slider["display"] == "slider"


def test_manual_nonstandard_resolution_records_custom_aspect_ratio():
    config, resolved = WeeToddH3GenerationConfig().configure(
        5.0,
        8,
        42,
        "exact dimensions",
        "Use size slider — 32 px steps",
        "custom — exact dimensions",
        992,
        608,
        True,
    )

    assert (config.width, config.height) == (992, 608)
    assert config.resolution_mode == "exact dimensions"
    assert config.aspect_ratio == "custom"
    assert resolved == "992 × 608 pixels — custom"


@pytest.mark.parametrize(
    ("ratio", "expected"),
    [
        ("21:9 — ultrawide landscape", (1792, 768)),
        ("16:9 — widescreen landscape", (1344, 768)),
        ("4:3 — standard landscape", (1024, 768)),
        ("5:4 — near-square landscape", (960, 768)),
        ("1:1 — square", (768, 768)),
        ("4:5 — near-square portrait", (768, 960)),
        ("3:4 — standard portrait", (768, 1024)),
        ("9:16 — vertical portrait", (768, 1344)),
        ("9:21 — ultratall portrait", (768, 1792)),
    ],
)
def test_resolution_presets_follow_ratio_and_h3_grid(ratio, expected):
    resolved = _resolve_h3_resolution(
        "ratio + size", "768 px short edge — native", ratio, 640, 384, 768
    )
    assert resolved == expected
    assert all(value % 32 == 0 for value in resolved)


def test_ratio_slider_reaches_1920_by_1088_for_widescreen():
    assert _resolve_h3_resolution(
        "ratio + size",
        "1088 px short edge — maximum slider size",
        "16:9 — widescreen landscape",
        640,
        384,
        1088,
    ) == (1920, 1088)


def test_ultrawide_slider_rejects_canvas_above_1920():
    assert _resolve_h3_resolution(
        "ratio + size",
        "Use size slider — 32 px steps",
        "21:9 — ultrawide landscape",
        640,
        384,
        800,
    ) == (1856, 800)
    with pytest.raises(ValueError, match="must not exceed 1920"):
        _resolve_h3_resolution(
            "ratio + size",
            "Use size slider — 32 px steps",
            "21:9 — ultrawide landscape",
            640,
            384,
            832,
        )


@pytest.mark.parametrize("short_edge", range(32, 1089, 32))
def test_widescreen_size_slider_always_stays_on_h3_grid(short_edge):
    width, height = _resolve_h3_resolution(
        "ratio + size",
        "Use size slider — 32 px steps",
        "16:9 — widescreen landscape",
        640,
        384,
        short_edge,
    )
    assert height == short_edge
    assert width <= 1920
    assert width % 32 == height % 32 == 0


def test_custom_resolution_uses_exact_dimensions():
    assert _resolve_h3_resolution("custom", "unused", "unused", 640, 384) == (640, 384)


def test_640p_widescreen_preset_uses_h3_grid():
    assert _resolve_h3_resolution("preset", "640P (quality preview)", "16:9", 640, 384) == (
        1120,
        640,
    )


def test_experimental_2k_preset_resolves_common_widescreen_canvas():
    assert _resolve_h3_resolution(
        "preset", "2K (experimental, very high memory)", "16:9", 640, 384
    ) == (1920, 1088)


def test_output_target_stays_below_comfy_output(tmp_path):
    target = _safe_output_target(tmp_path, "WeeTodd/H3", 42)
    assert target == tmp_path / "WeeTodd" / "H3_42.mp4"


@pytest.mark.parametrize("prefix", ["../escape", "/tmp/escape", "safe/../../escape"])
def test_output_target_rejects_escape(prefix, tmp_path):
    with pytest.raises(ValueError, match="output directory"):
        _safe_output_target(tmp_path, prefix, 42)
