import json

import numpy as np
import pytest
from safetensors.numpy import save_file

from ltx25_mlx.gemma_pack import gemma4_mlx_model_config, remap_gemma4_weight_key
from ltx25_mlx.runtime import (
    LTX25_GENERATION_PRESETS,
    LTX25ComponentSpec,
    LTX25GenerationConfig,
    LTX25RuntimeCache,
    apply_ltx25_generation_preset,
    backend_capability,
)
from ltx25_mlx.transformer import inspect_ltx25_ic_lora, remap_comfy_transformer_key
from ltx25_mlx.upscale import (
    LTX25_INPUT_SIZE_POLICIES,
    LTX25_PIXEL_SPATIAL_MODE,
    LTX25_SOURCE_FRAME_ANCHORS,
    LTX25_UPSCALE_MODES,
    _host_audio,
    _host_audio_or_silence,
    _host_video,
)
from wee_todd_nodes.ltx25_nodes import (
    WeeToddLTX25GenerationConfig,
    WeeToddLTX25VideoUpscale,
)


def _component(path, **metadata):
    encoded = {
        key: json.dumps(value) if not isinstance(value, str) else value
        for key, value in metadata.items()
    }
    save_file({"test": np.zeros((1,), dtype=np.float32)}, path, metadata=encoded)


def _gemma_pack(path, *, gemma_version="gemma4-12b-ltx-v1", model_type="gemma4_unified"):
    metadata = {
        "gemma_config": json.dumps(
            {
                "model_type": model_type,
                "gemma_version": gemma_version,
                "text_config": {"hidden_size": 3840, "num_hidden_layers": 48},
            }
        )
    }
    save_file(
        {
            "tokenizer_json": np.frombuffer(b"{}", dtype=np.uint8),
            "hf_asset__tokenizer_config.json": np.frombuffer(b"{}", dtype=np.uint8),
            "hf_asset__processor_config.json": np.frombuffer(b"{}", dtype=np.uint8),
            "model.language_model.layers.0.weight": np.zeros((1,), dtype=np.float32),
            "text_embedding_projection.video_aggregate_embed.weight": np.zeros(
                (1,), dtype=np.float32
            ),
            "text_embedding_projection.audio_aggregate_embed.weight": np.zeros(
                (1,), dtype=np.float32
            ),
            "model.diffusion_model.video_embeddings_connector.learnable_registers": np.zeros(
                (1,), dtype=np.float32
            ),
            "model.diffusion_model.audio_embeddings_connector.learnable_registers": np.zeros(
                (1,), dtype=np.float32
            ),
        },
        path,
        metadata=metadata,
    )


def _bundle(root, *, version="2.5.0"):
    gemma = {"gemma_version": "gemma4-12b-ltx-v1"}
    _component(
        root / "transformer.safetensors",
        model_version=version,
        gemma_source_checkpoint=gemma,
        config={
            "transformer": {
                "caption_proj_before_connector": True,
                "cross_attention_adaln": True,
                "ff_bias": False,
                "audio_ff_bias": True,
                "use_prompt_adaln_single": True,
                "use_keyframes_abs_pos_embedding": True,
            }
        },
    )
    _gemma_pack(root / "text_encoder.safetensors")
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
    assert report["gemma_pack"]["model_type"] == "gemma4_unified"
    assert report["gemma_pack"]["weight_layout"] == "huggingface_unified"
    assert report["video_scale_factors"] == [8, 32, 32]
    assert report["video_decoder"] == "convolutional"
    assert report["transformer_architecture"]["ff_bias"] is False
    assert report["transformer_architecture"]["use_prompt_adaln_single"] is True
    assert report["transformer_architecture"]["caption_proj_before_connector"] is True
    assert len(report["components"]) == 5


def test_ltx25_preflight_rejects_23_transformer(tmp_path):
    spec = _bundle(tmp_path, version="2.3.0")
    with pytest.raises(ValueError, match="not identified as LTX 2.5"):
        spec.validate()


def test_ltx25_preflight_rejects_legacy_transformer_construction(tmp_path):
    spec = _bundle(tmp_path)
    _component(
        tmp_path / "transformer.safetensors",
        model_version="2.5.0",
        gemma_source_checkpoint={"gemma_version": "gemma4-12b-ltx-v1"},
        config={"transformer": {}},
    )
    with pytest.raises(ValueError, match="required architecture"):
        spec.validate()


def test_ltx25_preflight_rejects_incomplete_gemma_pack(tmp_path):
    spec = _bundle(tmp_path)
    _component(tmp_path / "text_encoder.safetensors", gemma_config={"model_type": "gemma4_unified"})
    with pytest.raises(ValueError, match="tokenizer_json"):
        spec.validate()


def test_ltx25_preflight_rejects_dense_enhancer_in_generation_slot(tmp_path):
    spec = _bundle(tmp_path)
    _gemma_pack(tmp_path / "text_encoder.safetensors", model_type="gemma4")
    with pytest.raises(ValueError, match="encode-capable Gemma 4 unified"):
        spec.validate()


def test_ltx25_preflight_rejects_wrong_gemma_version(tmp_path):
    spec = _bundle(tmp_path)
    _gemma_pack(tmp_path / "text_encoder.safetensors", gemma_version="wrong-version")
    with pytest.raises(ValueError, match="different Gemma versions"):
        spec.validate()


def test_ltx25_gemma4_config_translation_disables_dense_only_features():
    translated = gemma4_mlx_model_config(
        {
            "model_type": "gemma4_unified",
            "text_config": {
                "vocab_size": 256,
                "hidden_size": 16,
                "num_hidden_layers": 2,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
            },
        }
    )
    assert translated["model_type"] == "gemma4"
    assert translated["text_config"]["model_type"] == "gemma4_text"
    assert translated["text_config"]["hidden_size_per_layer_input"] == 0
    assert translated["text_config"]["enable_moe_block"] is False


def test_ltx25_gemma4_weight_key_mapping_excludes_multimodal_towers():
    assert (
        remap_gemma4_weight_key(
            "model.language_model.layers.0.input_layernorm.weight",
            layout="huggingface_unified",
        )
        == "language_model.model.layers.0.input_layernorm.weight"
    )
    assert (
        remap_gemma4_weight_key(
            "model.layers.0.input_layernorm.weight",
            layout="comfy_flat",
        )
        == "language_model.model.layers.0.input_layernorm.weight"
    )
    assert remap_gemma4_weight_key("model.embed_audio.weight", layout="comfy_flat") is None


def test_ltx25_config_pins_official_distilled_schedule_and_grid():
    config = LTX25GenerationConfig()
    config.validate()
    assert config.num_frames == 121
    assert config.delivered_duration_seconds == 5.0
    assert config.stage1_steps + config.stage2_steps == 11
    assert config.stage1_sampler == "euler_ancestral"
    assert config.stage2_sampler == "euler_ancestral"
    from ltx25_mlx.runtime import LTX25_STAGE2_SIGMAS

    assert LTX25_STAGE2_SIGMAS == (0.85, 0.725, 0.421875, 0.0)
    assert config.seed + config.ancestral_seed_offset == 10000
    with pytest.raises(ValueError, match="eight stage-one and three stage-two"):
        LTX25GenerationConfig(stage2_steps=4).validate()
    with pytest.raises(ValueError, match="divisible"):
        LTX25GenerationConfig(width=736).validate()
    with pytest.raises(ValueError, match="feed-forward backend"):
        LTX25GenerationConfig(feed_forward_backend="unknown").validate()
    with pytest.raises(ValueError, match="not compatible with low-RAM"):
        LTX25GenerationConfig(
            low_ram_streaming=True,
            feed_forward_backend="bf16_mpp_experimental",
        ).validate()


def test_ltx25_video_upscaler_exposes_generic_movie_contract():
    inputs = WeeToddLTX25VideoUpscale.INPUT_TYPES()["required"]
    assert inputs["mode"][0] == list(LTX25_UPSCALE_MODES)
    assert inputs["max_av_drift_seconds"][1]["default"] == 0.05
    assert inputs["refinement_strength"][1]["default"] == 0.35
    assert inputs["input_size_policy"][0] == list(LTX25_INPUT_SIZE_POLICIES)
    assert inputs["source_frame_anchors"][0] == list(LTX25_SOURCE_FRAME_ANCHORS)
    assert inputs["source_frame_anchors"][1]["default"] == "first frame"
    assert inputs["reference_strength"][1]["default"] == 0.7
    assert LTX25_PIXEL_SPATIAL_MODE in inputs["mode"][0]
    assert inputs["pixel_spatial_lora_strength"][1]["default"] == 1.0
    assert "pixel-spatial-upscaler-x2" in inputs["pixel_spatial_lora"][1]["default"]
    assert set(WeeToddLTX25VideoUpscale.INPUT_TYPES()["optional"]) == {
        "first_reference",
        "last_reference",
        "audio",
    }
    assert WeeToddLTX25VideoUpscale.OUTPUT_NODE is True


def test_ltx25_pixel_spatial_lora_header_and_key_mapping(tmp_path):
    path = tmp_path / "pixel-spatial.safetensors"
    save_file(
        {
            "diffusion_model.transformer_blocks.0.attn1.to_out.0.lora_A.weight": np.zeros(
                (2, 4), dtype=np.float16
            ),
            "diffusion_model.transformer_blocks.0.attn1.to_out.0.lora_B.weight": np.zeros(
                (4, 2), dtype=np.float16
            ),
        },
        path,
        metadata={"model_version": "2.5", "reference_downscale_factor": "2"},
    )
    report = inspect_ltx25_ic_lora(path)
    assert report["model_version"] == "2.5"
    assert report["reference_downscale_factor"] == 2
    assert report["adapter_pairs"] == 1
    assert (
        remap_comfy_transformer_key(
            "diffusion_model.transformer_blocks.0.attn1.to_out.0.lora_A.weight"
        )
        == "transformer_blocks.0.attn1.to_out.lora_A.weight"
    )


def test_ltx25_pixel_spatial_lora_rejects_wrong_model_version(tmp_path):
    path = tmp_path / "wrong.safetensors"
    save_file(
        {
            "diffusion_model.transformer_blocks.0.attn1.to_q.lora_A.weight": np.zeros(
                (2, 4), dtype=np.float16
            ),
            "diffusion_model.transformer_blocks.0.attn1.to_q.lora_B.weight": np.zeros(
                (4, 2), dtype=np.float16
            ),
        },
        path,
        metadata={"model_version": "2.3", "reference_downscale_factor": "2"},
    )
    with pytest.raises(ValueError, match="not identified as LTX 2.5"):
        inspect_ltx25_ic_lora(path)


def test_ltx25_video_upscaler_validates_and_crops_generic_comfy_media():
    frames = _host_video(np.zeros((9, 65, 99, 3), dtype=np.float32))
    from ltx25_mlx.upscale import _prepare_video_size

    frames, report = _prepare_video_size(frames, LTX25_INPUT_SIZE_POLICIES[0])
    waveform, sample_rate = _host_audio(
        {
            "waveform": np.zeros((1, 1, 32000), dtype=np.float32),
            "sample_rate": 32000,
        }
    )
    assert frames.shape == (9, 64, 96, 3)
    assert report["crop"] == {"left": 1, "top": 0, "right": 2, "bottom": 1}
    assert waveform.shape == (2, 32000)
    assert sample_rate == 32000
    with pytest.raises(ValueError, match="divisible by 32"):
        _prepare_video_size(
            _host_video(np.zeros((9, 65, 96, 3), dtype=np.float32)),
            LTX25_INPUT_SIZE_POLICIES[1],
        )
    with pytest.raises(ValueError, match="waveform and sample_rate"):
        _host_audio({})


def test_ltx25_video_upscaler_supplies_matched_silence_for_silent_movies():
    waveform, sample_rate, supplied = _host_audio_or_silence(None, 1.25)

    assert waveform.shape == (2, 60000)
    assert sample_rate == 48000
    assert supplied is False
    assert np.count_nonzero(waveform) == 0


def test_ltx25_generation_config_node_resolves_random_seed(monkeypatch):
    monkeypatch.setattr("wee_todd_nodes.ltx25_nodes.secrets.randbelow", lambda _limit: 2468)
    config, raw = WeeToddLTX25GenerationConfig().configure(
        "Custom",
        768,
        512,
        5.0,
        24.0,
        -1,
        True,
        False,
        "official_1024",
        "reference_fp32",
    )
    assert config.seed == 2468
    assert json.loads(raw)["real_evaluations"] == 11
    assert config.prompt_context == "official_1024"


def test_ltx25_official_parity_preset_pins_recipe_and_preserves_extra_values():
    values = apply_ltx25_generation_preset(
        LTX25_GENERATION_PRESETS[1],
        {
            "width": 1024,
            "height": 1024,
            "duration_seconds": 9.0,
            "frame_rate": 30.0,
            "seed": 123,
            "low_memory": False,
            "low_ram_streaming": True,
            "prompt_context": "128",
            "feed_forward_backend": "bf16_mpp_experimental",
        },
    )
    assert values == {
        "width": 768,
        "height": 512,
        "duration_seconds": 5.0,
        "frame_rate": 24.0,
        "seed": 123,
        "low_memory": True,
        "low_ram_streaming": False,
        "prompt_context": "official_1024",
        "feed_forward_backend": "reference_fp32",
    }


def test_ltx25_high_quality_preset_pins_verified_1088p_recipe():
    values = apply_ltx25_generation_preset(
        LTX25_GENERATION_PRESETS[2],
        {
            "width": 768,
            "height": 512,
            "duration_seconds": 9.0,
            "frame_rate": 30.0,
            "seed": 584293325,
            "low_memory": False,
            "low_ram_streaming": True,
            "prompt_context": "128",
            "feed_forward_backend": "bf16_mpp_experimental",
        },
    )
    assert values == {
        "width": 1920,
        "height": 1088,
        "duration_seconds": 5.0,
        "frame_rate": 24.0,
        "seed": 584293325,
        "low_memory": True,
        "low_ram_streaming": False,
        "prompt_context": "official_1024",
        "feed_forward_backend": "reference_fp32",
    }


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

    monkeypatch.setattr("ltx25_mlx.runtime._pipeline_class", lambda: FakePipeline)
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


def test_ltx25_backend_capability_reports_project_native_pipeline():
    status = backend_capability()
    assert status == {"ready": True, "pipeline_class": "LTX25DistilledPipeline"}
