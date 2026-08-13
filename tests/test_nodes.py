import json
import sys
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from wee_todd_nodes.nodes import (
    NODE_CLASS_MAPPINGS,
    WeeToddH3ComponentLoader,
    WeeToddH3EasyCache,
    WeeToddH3FirstFrame,
    WeeToddH3FirstLastFrame,
    WeeToddH3Frames,
    WeeToddH3GenerationConfig,
    WeeToddH3KeyframeEncode,
    WeeToddH3LastFrame,
    WeeToddH3LatentHiresFix,
    WeeToddH3LoRALoader,
    WeeToddH3QuantizedTransformerLoader,
    WeeToddH3ReferenceAudio,
    WeeToddH3ReferenceEncode,
    WeeToddH3ReferenceImage,
    WeeToddH3ReferenceStrength,
    WeeToddH3ReferenceVideo,
    WeeToddH3TrajectoryForecast,
    WeeToddH3ValidatedSamplingPreset,
    _frames_from_manifest,
    _output_directory,
    _parse_media_timing_info,
    _publication_environment,
    _resolve_h3_resolution,
    _safe_output_target,
    _save_h3_preview_contact_sheet,
)


def test_live_preview_contact_sheet_is_published_atomically(tmp_path, monkeypatch):
    from PIL import Image

    monkeypatch.setitem(
        sys.modules,
        "folder_paths",
        SimpleNamespace(get_output_directory=lambda: str(tmp_path)),
    )
    path = _save_h3_preview_contact_sheet(Image.new("RGB", (32, 16), "red"), 1, 4)

    assert path == tmp_path / "WeeTodd" / "previews" / "h3_live_preview_eval_01_of_04.jpg"
    assert path.is_file()
    assert not path.with_suffix(".tmp.jpg").exists()


def test_easycache_exposes_fresh_head_core_residual_strategy_without_changing_default():
    inputs = WeeToddH3EasyCache.INPUT_TYPES()
    assert inputs["optional"]["reuse_strategy"][0] == [
        "output_residual",
        "core_residual_fresh_heads",
    ]
    assert inputs["optional"]["allow_turbo_experimental"][1]["default"] is False

    (legacy,) = WeeToddH3EasyCache().configure("manual", 0.2, 0.15, 0.95, 1.15, 0.25)
    (fresh_heads,) = WeeToddH3EasyCache().configure(
        "manual",
        0.2,
        0.15,
        0.95,
        1.15,
        0.25,
        "core_residual_fresh_heads",
        True,
    )

    assert legacy.reuse_strategy == "output_residual"
    assert fresh_heads.reuse_strategy == "core_residual_fresh_heads"
    assert fresh_heads.allow_turbo_experimental is True


def test_trim_timing_metadata_explicitly_authorizes_changed_frame_count():
    timing = {
        "context_frames_removed": 22,
        "output_frames": 102,
        "fps": 24,
        "sample_rate": 32000,
    }

    assert (
        _parse_media_timing_info(json.dumps(timing), image_frames=102, sample_rate=32000) == timing
    )
    with pytest.raises(ValueError, match="frame count"):
        _parse_media_timing_info(json.dumps(timing), image_frames=101, sample_rate=32000)
    with pytest.raises(ValueError, match="sample rate"):
        _parse_media_timing_info(json.dumps(timing), image_frames=102, sample_rate=48000)


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
    assert parsed["prepared_state"] is None


def test_hires_fix_resolves_target_and_preserves_audio_contract(monkeypatch):
    from minimax_h3_mlx.trajectory_forecast import H3TrajectoryForecastConfig
    from wee_todd_nodes.runtime import H3GenerationConfig

    inputs = WeeToddH3LatentHiresFix.INPUT_TYPES()
    resize_input = inputs["optional"]["latent_resize_method"]
    assert "latent_resize_method" not in inputs["required"]
    assert resize_input[0] == ["bilinear", "nearest exact", "bicubic", "lanczos-3"]
    assert resize_input[1]["default"] == "bilinear"

    source = SimpleNamespace(
        width=640,
        height=384,
        generation_config=H3GenerationConfig(
            width=640,
            height=384,
            duration_seconds=5.0,
            steps=8,
        ),
    )
    refined = SimpleNamespace(
        transformer_evaluations=2,
        refinement_audio_preserved=True,
        trajectory_forecasts=2,
        trajectory_fallbacks=0,
        trajectory_offline_replay=True,
        trajectory_replay_steps=6,
        trajectory_replay_anchor_steps=4,
        trajectory_replay_smoothed_steps=2,
        trajectory_replay_seconds=0.125,
        trajectory_replay_fallback_reason=None,
    )
    forecast = H3TrajectoryForecastConfig(
        mode="automatic_speed",
        offline_smoothing_replay=True,
    )
    calls = []

    def sample(*args, **kwargs):
        calls.append((args, kwargs))
        return refined

    monkeypatch.setattr(
        "wee_todd_nodes.nodes.H3TransformerSpec.from_components",
        lambda components: "transformer-spec",
    )
    monkeypatch.setattr(
        "wee_todd_nodes.nodes.TRANSFORMER_RUNTIME",
        SimpleNamespace(sample=sample, loaded=False),
    )

    result, refined_config, metadata = WeeToddH3LatentHiresFix().refine(
        "components",
        "conditioning",
        source,
        "1.5x — balanced",
        5,
        0.35,
        True,
        trajectory_forecast=forecast,
        latent_resize_method="bicubic",
    )

    assert result is refined
    config = calls[0][0][2]
    assert refined_config is config
    assert (config.width, config.height, config.steps) == (960, 576, 5)
    assert calls[0][1]["refinement_source"] is source
    assert calls[0][1]["refinement_strength"] == 0.35
    assert calls[0][1]["refinement_resize_method"] == "bicubic"
    assert calls[0][1]["trajectory_forecast"] is forecast
    parsed = json.loads(metadata)
    assert parsed["audio_preserved"] is True
    assert parsed["latent_resize_method"] == "bicubic"
    assert parsed["trajectory_forecasts"] == 2
    assert parsed["transformer_evaluations"] == 2
    assert parsed["trajectory_replay_seconds"] == 0.125
    assert parsed["trajectory_replay_fallback_reason"] is None


def test_expected_nodes_are_registered():
    assert len(NODE_CLASS_MAPPINGS) == 55
    assert "WeeToddH3ComponentLoader" in NODE_CLASS_MAPPINGS
    assert "WeeToddH3QuantizedTransformerLoader" in NODE_CLASS_MAPPINGS
    assert "WeeToddH3Preflight" in NODE_CLASS_MAPPINGS
    assert "WeeToddH3FirstFrame" in NODE_CLASS_MAPPINGS
    assert "WeeToddH3LastFrame" in NODE_CLASS_MAPPINGS
    assert "WeeToddH3FirstLastFrame" in NODE_CLASS_MAPPINGS
    assert "WeeToddH3ChainedTimeline" in NODE_CLASS_MAPPINGS
    assert "WeeToddH3Frames" in NODE_CLASS_MAPPINGS
    assert "WeeToddH3TimedKeyframe" in NODE_CLASS_MAPPINGS
    assert "WeeToddH3ReferenceImage" in NODE_CLASS_MAPPINGS
    assert "WeeToddH3ReferenceVideo" in NODE_CLASS_MAPPINGS
    assert "WeeToddH3ReferenceAudio" in NODE_CLASS_MAPPINGS
    assert "WeeToddH3KeyframeEncode" in NODE_CLASS_MAPPINGS
    assert "WeeToddH3TimedKeyframeEncode" in NODE_CLASS_MAPPINGS
    assert "WeeToddH3ReferenceEncode" in NODE_CLASS_MAPPINGS
    assert "WeeToddH3ReferenceStrength" in NODE_CLASS_MAPPINGS
    assert "WeeToddH3TextEncode" in NODE_CLASS_MAPPINGS
    assert "WeeToddH3TrajectoryForecast" in NODE_CLASS_MAPPINGS
    assert "WeeToddH3UnloadTextEncoder" in NODE_CLASS_MAPPINGS
    assert "WeeToddH3ContinuationContext" in NODE_CLASS_MAPPINGS
    assert "WeeToddH3ChainAppend" in NODE_CLASS_MAPPINGS
    assert "WeeToddH3Sample" in NODE_CLASS_MAPPINGS
    assert "WeeToddH3LatentHiresFix" in NODE_CLASS_MAPPINGS
    assert "WeeToddH3EasyCache" in NODE_CLASS_MAPPINGS
    assert "WeeToddH3BlockCache" in NODE_CLASS_MAPPINGS
    assert "WeeToddH3HierarchicalBlockCache" in NODE_CLASS_MAPPINGS
    assert "WeeToddH3LoRALoader" in NODE_CLASS_MAPPINGS
    assert "WeeToddH3ValidatedSamplingPreset" in NODE_CLASS_MAPPINGS
    assert "WeeToddH3UnloadTransformer" in NODE_CLASS_MAPPINGS
    assert "WeeToddH3VideoVAEDecode" in NODE_CLASS_MAPPINGS
    assert "WeeToddH3UnloadVideoVAE" in NODE_CLASS_MAPPINGS
    assert "WeeToddH3AudioVAEDecode" in NODE_CLASS_MAPPINGS
    assert "WeeToddH3UnloadAudioVAE" in NODE_CLASS_MAPPINGS
    assert "WeeToddH3TrimContinuation" in NODE_CLASS_MAPPINGS
    assert "WeeToddH3PublishVideoAudio" in NODE_CLASS_MAPPINGS
    assert "WeeToddH3DirectPublishLatents" in NODE_CLASS_MAPPINGS
    assert "WeeToddH3DirectPublishChain" in NODE_CLASS_MAPPINGS
    assert "WeeToddLTX23ModelLoader" in NODE_CLASS_MAPPINGS
    assert "WeeToddLTX23GenerationConfig" in NODE_CLASS_MAPPINGS
    assert "WeeToddLTX23Preflight" in NODE_CLASS_MAPPINGS
    assert "WeeToddLTX23Generate" in NODE_CLASS_MAPPINGS
    assert "WeeToddLTX23UpscalerLoader" in NODE_CLASS_MAPPINGS
    assert "WeeToddLTX23UpscalePublish" in NODE_CLASS_MAPPINGS
    assert "WeeToddLTX23Unload" in NODE_CLASS_MAPPINGS
    assert "WeeToddLTX25ComponentLoader" in NODE_CLASS_MAPPINGS
    assert "WeeToddLTX25GenerationConfig" in NODE_CLASS_MAPPINGS
    assert "WeeToddLTX25Preflight" in NODE_CLASS_MAPPINGS
    assert "WeeToddLTX25Generate" in NODE_CLASS_MAPPINGS
    assert "WeeToddLTX25VideoUpscale" in NODE_CLASS_MAPPINGS
    assert "WeeToddLTX25Unload" in NODE_CLASS_MAPPINGS


def test_continuation_context_defaults_to_quality_first_22_frames():
    node_class = NODE_CLASS_MAPPINGS["WeeToddH3ContinuationContext"]
    specification = node_class.INPUT_TYPES()["required"]["context_frames"]

    assert specification[0] == ["5", "22", "39", "56"]
    assert specification[1]["default"] == "22"


def test_lora_loader_exposes_qkv_layout_and_staged_activation(tmp_path):
    path = tmp_path / "example_turbo.safetensors"
    mx.save_safetensors(
        str(path),
        {
            "blocks.0.attn.qkv_proj.lora_A.weight": mx.zeros((1, 2)),
            "blocks.0.attn.qkv_proj.lora_B.weight": mx.zeros((6, 1)),
        },
    )
    inputs = WeeToddH3LoRALoader.INPUT_TYPES()

    stack, raw_info = WeeToddH3LoRALoader().load(
        str(path),
        1.0,
        "turbo",
        qkv_layout="auto",
        start_after_evaluations=2,
    )
    info = json.loads(raw_info)

    assert inputs["required"]["qkv_layout"][0] == [
        "auto",
        "native_interleaved",
        "contiguous_qkv",
    ]
    assert stack.adapters[0].resolved_qkv_layout == "contiguous_qkv"
    assert stack.adapters[0].start_after_evaluations == 2
    assert info["qkv_layout"] == "contiguous_qkv"
    assert info["start_after_evaluations"] == 2
    assert inputs["optional"]["start_after_evaluations"][1]["default"] == 0


def test_validated_sampling_preset_applies_dense_and_trajectory_policies():
    from wee_todd_nodes.runtime import H3GenerationConfig

    source = H3GenerationConfig(
        duration_seconds=7.0,
        steps=8,
        seed=42,
        width=896,
        height=512,
        memory_mode="low_memory_bf16",
        sampling_method="res_multistep",
    )
    node = WeeToddH3ValidatedSamplingPreset()

    dense, dense_loras, dense_forecast, dense_raw = node.apply(
        source, "Dense baseline — 20 points / 19 evaluations"
    )
    dense_info = json.loads(dense_raw)
    assert dense.steps == 20
    assert dense.sampling_method == "euler"
    assert (dense.width, dense.height) == (896, 512)
    assert dense.duration_seconds == 7.0
    assert dense.seed == 42
    assert dense.memory_mode == "low_memory_bf16"
    assert dense_loras is None
    assert dense_forecast is None
    assert dense_info["policy"] == "dense"
    assert dense_info["transformer_evaluations_without_forecast"] == 19

    replay, replay_loras, replay_forecast, replay_raw = node.apply(
        source,
        "Trajectory speed + offline replay — 20 points / up to 11 evaluations",
    )
    replay_info = json.loads(replay_raw)
    assert replay.steps == 20
    assert replay_loras is None
    assert replay_forecast.mode == "automatic_speed"
    assert replay_forecast.bootstrap_first_forecast is False
    assert replay_forecast.offline_smoothing_replay is True
    assert replay_forecast.offline_video_blend == 0.5
    assert replay_forecast.offline_audio_blend == 0.0
    assert replay_info["trajectory_offline_replay"] is True

    ref2va, ref2va_loras, ref2va_forecast, ref2va_raw = node.apply(
        source,
        (
            "Ref2VA four-reference BF16 — Forward Attention replay — "
            "20 points / up to 11 evaluations"
        ),
    )
    ref2va_info = json.loads(ref2va_raw)
    assert ref2va.steps == 20
    assert ref2va_loras is None
    assert ref2va_forecast.mode == "automatic_speed"
    assert ref2va_forecast.offline_smoothing_replay is True
    assert ref2va_info["measurement"] == {
        "task": "ref2va",
        "reference_images": 4,
        "canvas": [896, 512],
        "duration_seconds": 5.0,
        "memory_mode": "normal",
        "checkpoint_policy": "experimental_fl2va_weights_for_ref2va",
        "transformer_evaluations": 11,
        "mlx_peak_bytes": 47323507330,
    }


def test_validated_chained_context_presets_match_measured_policies(tmp_path, monkeypatch):
    from wee_todd_nodes.runtime import H3GenerationConfig

    lora_name = (
        "minimax_h3_fl2v_lightx2v_turbo_4step_v0.1_comfy_resized_avg_rank_21_bf16.safetensors"
    )
    path = tmp_path / lora_name
    mx.save_safetensors(
        str(path),
        {
            "blocks.0.attn.qkv_proj.lora_A.weight": mx.zeros((1, 2)),
            "blocks.0.attn.qkv_proj.lora_B.weight": mx.zeros((6, 1)),
        },
    )
    monkeypatch.setattr("wee_todd_nodes.nodes._resolve_lora_path", lambda name: tmp_path / name)
    source = H3GenerationConfig(width=960, height=544, duration_seconds=5.17)
    node = WeeToddH3ValidatedSamplingPreset()

    turbo, loras, forecast, turbo_raw = node.apply(
        source,
        "Chained context — Dense Turbo LightX2V rank 21 — 5 points / 4 evaluations",
    )
    turbo_info = json.loads(turbo_raw)
    assert turbo.steps == 5
    assert forecast is None
    assert loras.has_turbo is True
    assert turbo_info["measurement"]["complete_workflow_seconds"] == 1570

    replay, loras, forecast, replay_raw = node.apply(
        source,
        ("Chained context — Trajectory target-only replay — 20 points / up to 11 evaluations"),
    )
    replay_info = json.loads(replay_raw)
    assert replay.steps == 20
    assert loras is None
    assert forecast.offline_smoothing_replay is True
    assert forecast.conditioned_row_policy == "target_only"
    assert replay_info["measurement"]["complete_workflow_seconds"] == 3765


@pytest.mark.parametrize(
    ("preset", "filename"),
    [
        (
            "Turbo — Larry EMA-850 — 5 points / 4 evaluations",
            "minimax_h3_turbo_4step_ema_ckpt850.safetensors",
        ),
        (
            "Turbo — Larry v4 step-600 — 5 points / 4 evaluations",
            "minimax_h3_turbo_v4_step600_ema.safetensors",
        ),
        (
            "Turbo — drbaph v4 step-600 — 5 points / 4 evaluations",
            "minimax_h3_turbo_v4_step600_ema_pruned_comfyui.safetensors",
        ),
        (
            ("Turbo — drbaph v4 step-600 — 384p low-memory — 5 points / 4 evaluations"),
            "minimax_h3_turbo_v4_step600_ema_pruned_comfyui.safetensors",
        ),
        (
            "Staged Turbo — drbaph v4 step-600 — 2 base + 4 Turbo evaluations",
            "minimax_h3_turbo_v4_step600_ema_pruned_comfyui.safetensors",
        ),
        (
            ("Staged Turbo — drbaph v4 step-600 — 384p low-memory — 2 base + 4 Turbo evaluations"),
            "minimax_h3_turbo_v4_step600_ema_pruned_comfyui.safetensors",
        ),
        (
            "One-shot staged Turbo — drbaph v4 step-600 — 15-second quality baseline",
            "minimax_h3_turbo_v4_step600_ema_pruned_comfyui.safetensors",
        ),
        (
            "Chained staged Turbo — drbaph v4 step-600 — 4 windows / 22-frame context",
            "minimax_h3_turbo_v4_step600_ema_pruned_comfyui.safetensors",
        ),
        (
            "Turbo — LightX2V full rank — 5 points / 4 evaluations",
            "minimax_h3_fl2v_lightx2v_turbo_4step_v0.1_comfy.safetensors",
        ),
        (
            "Turbo — LightX2V dynamic rank 21 — 5 points / 4 evaluations",
            (
                "minimax_h3_fl2v_lightx2v_turbo_4step_v0.1_comfy_"
                "resized_avg_rank_21_bf16.safetensors"
            ),
        ),
    ],
)
def test_validated_sampling_preset_builds_each_lazy_turbo_stack(
    tmp_path, monkeypatch, preset, filename
):
    from wee_todd_nodes.runtime import H3GenerationConfig

    path = tmp_path / filename
    mx.save_safetensors(
        str(path),
        {
            "blocks.0.attn.qkv_proj.lora_A.weight": mx.zeros((1, 2)),
            "blocks.0.attn.qkv_proj.lora_B.weight": mx.zeros((6, 1)),
        },
    )
    monkeypatch.setattr("wee_todd_nodes.nodes._resolve_lora_path", lambda name: tmp_path / name)

    config, loras, forecast, raw_info = WeeToddH3ValidatedSamplingPreset().apply(
        H3GenerationConfig(width=896, height=512), preset
    )
    info = json.loads(raw_info)

    assert config.sampling_method == "euler"
    assert forecast is None
    assert len(loras.adapters) == 1
    assert loras.adapters[0].resolved_profile == "turbo"
    assert loras.adapters[0].resolved_qkv_layout == "contiguous_qkv"
    assert loras.adapters[0].strength == 1.0
    assert info["lora_file"] == filename
    if loras.adapters[0].start_after_evaluations == 2:
        assert config.steps == 7
        assert loras.adapters[0].start_after_evaluations == 2
        assert info["lora_start_after_evaluations"] == 2
        assert info["transformer_evaluations_without_forecast"] == 6
        assert info["measurement"]["base_evaluations"] == 2
        assert info["measurement"]["lora_evaluations"] == 4
    else:
        assert config.steps == 5
        assert info["lora_start_after_evaluations"] == 0
        assert info["transformer_evaluations_without_forecast"] == 4


def test_validated_15_second_comparison_presets_record_measured_boundaries(tmp_path, monkeypatch):
    from wee_todd_nodes.runtime import H3GenerationConfig

    filename = "minimax_h3_turbo_v4_step600_ema_pruned_comfyui.safetensors"
    mx.save_safetensors(
        str(tmp_path / filename),
        {
            "blocks.0.attn.qkv_proj.lora_A.weight": mx.zeros((1, 2)),
            "blocks.0.attn.qkv_proj.lora_B.weight": mx.zeros((6, 1)),
        },
    )
    monkeypatch.setattr("wee_todd_nodes.nodes._resolve_lora_path", lambda name: tmp_path / name)
    node = WeeToddH3ValidatedSamplingPreset()
    source = H3GenerationConfig(width=1344, height=768, duration_seconds=15.0)

    _, _, _, one_shot_raw = node.apply(
        source,
        "One-shot staged Turbo — drbaph v4 step-600 — 15-second quality baseline",
    )
    _, _, _, chain_raw = node.apply(
        source,
        "Chained staged Turbo — drbaph v4 step-600 — 4 windows / 22-frame context",
    )
    one_shot = json.loads(one_shot_raw)["measurement"]
    chain = json.loads(chain_raw)["measurement"]

    assert one_shot["complete_workflow_seconds"] == pytest.approx(8207.699172)
    assert one_shot["mlx_peak_bytes"] == 30783349650
    assert one_shot["seed"] == 20260811
    assert chain["complete_workflow_seconds"] == 5089.0
    assert chain["mlx_peak_bytes"] == 14453992534
    assert chain["context_frames"] == 22
    assert chain["join_policy"].startswith("motion-matched overlap")


def test_reference_strength_node_preserves_defaults_and_warns_for_weak_fl2va():
    from wee_todd_nodes.conditioning import H3Conditioning, H3TextEncoderSpec

    conditioning = H3Conditioning(
        embeddings="embeddings",
        token_tags="tags",
        token_count=1,
        prompt="prompt",
        load_vision=True,
        encoder_spec=H3TextEncoderSpec("text", "processor", "tokenizer", True),
        task="fl2va",
        condition_video_rows="rows",
        keyframe_anchors=("first", "last"),
    )

    default, default_info = WeeToddH3ReferenceStrength().configure(conditioning, 0.999, 1.0)
    weak, weak_info = WeeToddH3ReferenceStrength().configure(conditioning, 0.5, 0.9)

    assert default.visual_condition_strength == 0.999
    assert default.audio_condition_strength == 1.0
    assert json.loads(default_info)["warning"] is None
    assert weak.visual_condition_strength == 0.5
    assert weak.audio_condition_strength == 0.9
    assert "last-frame anchor" in json.loads(weak_info)["warning"]


def test_keyframe_nodes_emit_explicit_anchor_contracts():
    image = SimpleNamespace(shape=(1, 384, 640, 3))
    first, _ = WeeToddH3FirstFrame().configure(image)
    last, _ = WeeToddH3LastFrame().configure(image)
    both, _ = WeeToddH3FirstLastFrame().configure(image, image)

    assert first.anchors == ("first",)
    assert last.anchors == ("last",)
    assert both.anchors == ("first", "last")


def test_frames_node_resolves_one_based_endpoints_and_middle_frames():
    images = {
        "first.png": SimpleNamespace(shape=(1, 384, 640, 3)),
        "middle.png": SimpleNamespace(shape=(1, 384, 640, 3)),
        "last.png": SimpleNamespace(shape=(1, 384, 640, 3)),
    }
    manifest = json.dumps(
        [
            {"role": "last", "image": "last.png"},
            {"role": "middle", "frame": 60, "image": "middle.png"},
            {"role": "first", "image": "first.png"},
        ]
    )

    stack = _frames_from_manifest(manifest, 155, images.__getitem__)

    assert [item.timestamp_seconds for item in stack.keyframes] == [0.0, 59 / 24, 154 / 24]
    assert [item.image for item in stack.keyframes] == [
        images["first.png"],
        images["middle.png"],
        images["last.png"],
    ]
    assert stack.metadata(155)["resolved_frames"] == [0, 59, 154]


@pytest.mark.parametrize(
    ("manifest", "message"),
    [
        ("[]", "at least one"),
        (
            json.dumps(
                [
                    {"role": "first", "image": "a.png"},
                    {"role": "middle", "frame": 1, "image": "b.png"},
                ]
            ),
            "between frame 2",
        ),
        (
            json.dumps(
                [
                    {"role": "middle", "frame": 12, "image": "a.png"},
                    {"role": "middle", "frame": 12, "image": "b.png"},
                ]
            ),
            "same frame",
        ),
    ],
)
def test_frames_node_rejects_invalid_editor_manifests(manifest, message):
    with pytest.raises(ValueError, match=message):
        _frames_from_manifest(
            manifest,
            121,
            lambda _name: SimpleNamespace(shape=(1, 384, 640, 3)),
        )


def test_frames_node_uses_connected_config_duration(monkeypatch):
    image = np.zeros((1, 384, 640, 3), dtype=np.float32)
    monkeypatch.setattr("wee_todd_nodes.nodes._load_h3_frame_image", lambda _name: image)
    config = SimpleNamespace(duration_seconds=5.0, validate=lambda: None)
    manifest = json.dumps(
        [
            {"role": "first", "image": "first.png"},
            {"role": "last", "image": "last.png"},
        ]
    )

    stack, raw_info = WeeToddH3Frames().configure(config, manifest)

    assert [item.timestamp_seconds for item in stack.keyframes] == [0.0, 123 / 24]
    assert json.loads(raw_info)["last_frame"] == 124


def test_reference_nodes_build_one_ordered_stack():
    image = SimpleNamespace(shape=(1, 384, 640, 3))
    video = SimpleNamespace(shape=(48, 384, 640, 3))
    audio = {
        "waveform": SimpleNamespace(shape=(1, 2, 32000)),
        "sample_rate": 32000,
    }
    references, _ = WeeToddH3ReferenceImage().append(image, 100)
    references, _ = WeeToddH3ReferenceVideo().append(
        video,
        24.0,
        soundtrack=audio,
        previous_references=references,
    )
    references, _ = WeeToddH3ReferenceAudio().append(audio, references)

    references.validate_request()
    assert [reference.kind for reference in references.references] == ["image", "video", "audio"]


def test_keyframe_encode_stages_qwen_and_video_vae(monkeypatch):
    from wee_todd_nodes.conditioning import H3Conditioning

    calls = []

    class FakeTextRuntime:
        loaded = False

        def encode(self, spec, prompt, **kwargs):
            kwargs["prepare_stage"]()
            calls.append(("text", spec, len(kwargs["images"])))
            return H3Conditioning(
                embeddings="vision-conditioning",
                token_tags="vision-tags",
                token_count=7,
                prompt=prompt,
                load_vision=True,
                encoder_spec=spec,
                task="fl2va",
            )

    class FakeVideoRuntime:
        loaded = False

        def encode_keyframes(self, spec, images, **kwargs):
            kwargs["prepare_stage"]()
            calls.append(("video_vae", spec, len(images)))
            return np.zeros((480, 96), dtype=np.float32)

    monkeypatch.setattr("wee_todd_nodes.nodes.TEXT_ENCODER_RUNTIME", FakeTextRuntime())
    monkeypatch.setattr("wee_todd_nodes.nodes.VIDEO_VAE_RUNTIME", FakeVideoRuntime())
    monkeypatch.setattr(
        "wee_todd_nodes.nodes.H3TextEncoderSpec.from_components",
        lambda components, load_vision: "vision-spec",
    )
    monkeypatch.setattr(
        "wee_todd_nodes.nodes.H3VideoVAESpec.from_components",
        lambda components: "video-vae-spec",
    )
    monkeypatch.setattr(
        "wee_todd_nodes.nodes.prepare_low_memory_stage",
        lambda stage, mode: (f"released-for-{stage}",),
    )

    frame = np.zeros((1, 384, 640, 3), dtype=np.float32)
    keyframes, _ = WeeToddH3FirstLastFrame().configure(frame, frame)
    config = SimpleNamespace(
        height=384,
        width=640,
        memory_mode="low_memory_bf16",
        validate=lambda: None,
    )
    conditioning, encoded_info = WeeToddH3KeyframeEncode().encode(
        SimpleNamespace(task="fl2va"),
        config,
        keyframes,
        "A precise test prompt.",
    )

    info = json.loads(encoded_info)
    assert calls == [
        ("text", "vision-spec", 2),
        ("video_vae", "video-vae-spec", 2),
    ]
    assert conditioning.keyframe_anchors == ("first", "last")
    assert conditioning.condition_video_rows.shape == (480, 96)
    assert info["staged_releases"] == {
        "text_encoder": ["released-for-text_encoder"],
        "video_vae": ["released-for-video_vae"],
    }


def test_reference_encode_stages_qwen_and_both_vaes(monkeypatch):
    from wee_todd_nodes.conditioning import H3Conditioning

    calls = []
    prepared = [
        SimpleNamespace(
            has_audio=True,
            kind="video",
            num_latent_frames=3,
            latent_height=24,
            latent_width=40,
            num_audio_latents=10,
        ),
        SimpleNamespace(
            has_audio=True,
            kind="audio",
            num_latent_frames=0,
            latent_height=0,
            latent_width=0,
            num_audio_latents=10,
        ),
    ]
    stack = SimpleNamespace(
        validate_request=lambda: None,
        prepare=lambda **kwargs: prepared,
        metadata=lambda: {"references": [{"kind": "video"}, {"kind": "audio"}]},
    )

    class FakeTextRuntime:
        loaded = False

        def encode(self, spec, prompt, **kwargs):
            kwargs["prepare_stage"]()
            calls.append(("text", len(kwargs["references"])))
            return H3Conditioning(
                embeddings="reference-conditioning",
                token_tags="reference-tags",
                token_count=11,
                prompt=prompt,
                load_vision=True,
                encoder_spec=spec,
                task="ref2va",
            )

    class FakeVideoRuntime:
        loaded = False

        def encode_references(self, spec, references, **kwargs):
            kwargs["prepare_stage"]()
            calls.append(("video_vae", len(references)))
            return np.zeros((32, 96), dtype=np.float32)

    class FakeAudioRuntime:
        loaded = False

        def encode_references(self, spec, references, **kwargs):
            kwargs["prepare_stage"]()
            calls.append(("audio_vae", len(references)))
            return np.zeros((20, 32), dtype=np.float32)

    monkeypatch.setattr("wee_todd_nodes.nodes.TEXT_ENCODER_RUNTIME", FakeTextRuntime())
    monkeypatch.setattr("wee_todd_nodes.nodes.VIDEO_VAE_RUNTIME", FakeVideoRuntime())
    monkeypatch.setattr("wee_todd_nodes.nodes.AUDIO_VAE_RUNTIME", FakeAudioRuntime())
    monkeypatch.setattr(
        "wee_todd_nodes.nodes.H3TextEncoderSpec.from_components",
        lambda components, load_vision: "vision-spec",
    )
    monkeypatch.setattr(
        "wee_todd_nodes.nodes.H3VideoVAESpec.from_components",
        lambda components: "video-vae-spec",
    )
    monkeypatch.setattr(
        "wee_todd_nodes.nodes.H3AudioVAESpec.from_components",
        lambda components: "audio-vae-spec",
    )
    monkeypatch.setattr(
        "wee_todd_nodes.nodes.prepare_low_memory_stage",
        lambda stage, mode: (f"released-for-{stage}",),
    )
    config = SimpleNamespace(
        duration_seconds=5.0,
        width=640,
        height=384,
        memory_mode="low_memory_bf16",
        validate=lambda: None,
    )

    conditioning, encoded_info = WeeToddH3ReferenceEncode().encode(
        SimpleNamespace(task="ref2va"), config, stack, "A reference test."
    )

    assert calls == [("text", 2), ("video_vae", 2), ("audio_vae", 2)]
    assert conditioning.condition_video_rows.shape == (32, 96)
    assert conditioning.condition_audio_rows.shape == (20, 32)
    assert conditioning.references == tuple(prepared)
    assert json.loads(encoded_info)["staged_releases"] == {
        "text_encoder": ["released-for-text_encoder"],
        "video_vae": ["released-for-video_vae"],
        "audio_vae": ["released-for-audio_vae"],
    }


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


def test_trajectory_forecast_node_defaults_to_context_safe_target_rows():
    (config,) = WeeToddH3TrajectoryForecast().configure(
        "automatic_speed",
        1.0,
        2,
        1,
        2,
        0.5,
        2.5,
    )

    assert config.conditioned_row_policy == "target_only"
    choices, options = WeeToddH3TrajectoryForecast.INPUT_TYPES()["optional"][
        "conditioned_row_policy"
    ]
    assert choices == ["target_only", "all_rows_legacy"]
    assert options["default"] == "target_only"


def test_component_loader_returns_lazy_immutable_spec():
    (spec,) = WeeToddH3ComponentLoader().specify("MiniMax-H3/FL2VA", "t2va")

    assert spec.task == "t2va"
    assert spec.transformer is None
    assert spec.allow_fl2va_weights_for_ref2va is False


def test_component_loader_exposes_cross_partition_ref2va_opt_in_last():
    optional = WeeToddH3ComponentLoader.INPUT_TYPES()["optional"]
    assert list(optional)[-1] == "allow_fl2va_weights_for_ref2va"

    (spec,) = WeeToddH3ComponentLoader().specify(
        "MiniMax-H3/FL2VA",
        "ref2va",
        allow_fl2va_weights_for_ref2va=True,
    )

    assert spec.allow_fl2va_weights_for_ref2va is True


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
    assert config.sampling_method == "euler"
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
    assert inputs["optional"]["sampling_method"][0] == ["euler", "res_multistep"]


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
