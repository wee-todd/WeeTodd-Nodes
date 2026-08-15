import json
from dataclasses import replace

import numpy as np
import pytest
from safetensors.numpy import save_file

from ltx25_mlx.components import LTX25VideoDecoder
from ltx25_mlx.dfr import (
    DFRTemporalImageAnchor,
    choose_dfr_segment_length,
    extract_dfr_temporal_image_anchors,
    plan_dfr_temporal_tiles,
    resolve_dfr_canvas,
    scale_dfr_temporal_image_anchors,
    select_dfr_generated_slot_tokens,
    stitch_dfr_temporal_tiles,
)
from ltx25_mlx.duration_head import LTX25DurationHead, seconds_to_ltx25_frames
from ltx25_mlx.gemma_pack import gemma4_mlx_model_config, remap_gemma4_weight_key
from ltx25_mlx.generated_keyframes import (
    GeneratedKeyframeSlots,
    evenly_spaced_keyframe_positions,
    set_generated_keyframe_marker,
)
from ltx25_mlx.runtime import (
    LTX25_GENERATION_PRESETS,
    LTX25ComponentSpec,
    LTX25GenerationConfig,
    LTX25RuntimeCache,
    apply_ltx25_generation_preset,
    backend_capability,
    resolve_ltx25_dfr_recipe,
)
from ltx25_mlx.transformer import (
    _fuse_non_block_loras,
    inspect_ltx25_ic_lora,
    inspect_ltx25_lora,
    remap_comfy_transformer_key,
)
from ltx25_mlx.upscale import (
    LTX25_INPUT_SIZE_POLICIES,
    LTX25_PIXEL_SPATIAL_MODE,
    LTX25_SOURCE_FRAME_ANCHORS,
    LTX25_UPSCALE_MODES,
    _host_audio,
    _host_audio_or_silence,
    _host_video,
)
from ltx25_mlx.video_only import LTX25VideoOnlyX0Model
from wee_todd_nodes.ltx25_nodes import (
    LTX25_QUALITY_MODES,
    LTX25KeyframeStack,
    LTX25MediaConditioningStack,
    WeeToddLTX25AutoDuration,
    WeeToddLTX25DFRDetailing,
    WeeToddLTX25DFRTemporalRefinement,
    WeeToddLTX25DiffVAEOptimization,
    WeeToddLTX25GenerateChained,
    WeeToddLTX25GeneratedKeyframes,
    WeeToddLTX25GenerationConfig,
    WeeToddLTX25GuidedModelLoader,
    WeeToddLTX25Keyframe,
    WeeToddLTX25LoRALoader,
    WeeToddLTX25MediaConditioning,
    WeeToddLTX25QualityMode,
    WeeToddLTX25VideoUpscale,
)


def _rank450_lora(path):
    save_file(
        {
            "diffusion_model.adaln_single.emb.timestep_embedder.linear_1.lora_A.weight": (
                np.zeros((2, 4), dtype=np.float32)
            ),
            "diffusion_model.adaln_single.emb.timestep_embedder.linear_1.lora_B.weight": (
                np.zeros((4, 2), dtype=np.float32)
            ),
        },
        path,
        metadata={
            "model_version": "2.5.0",
            "lora_rank": "450",
            "lora_alpha": "450",
        },
    )
    return path


def test_ltx25_duration_head_matches_constant_log_seconds():
    import mlx.core as mx
    from mlx.utils import tree_flatten

    head = LTX25DurationHead(
        video_dim=4,
        audio_dim=2,
        hidden_dim=4,
        num_queries=1,
        num_heads=2,
        mlp_hidden=4,
    )
    weights = [(name, mx.zeros_like(value)) for name, value in tree_flatten(head.parameters())]
    # A zero network with this final bias predicts exp(log(2.5)) seconds.
    weights = [
        (name, mx.array([np.log(2.5)], dtype=value.dtype) if name == "mlp_out.bias" else value)
        for name, value in weights
    ]
    head.load_weights(weights, strict=True)
    seconds = head(video_tokens=mx.zeros((1, 3, 4), dtype=mx.bfloat16))
    mx.eval(seconds)
    assert float(seconds.item()) == pytest.approx(2.5, abs=0.02)


@pytest.mark.parametrize(
    ("seconds", "expected"),
    ((2.375, 57), (5.0, 113), (0.1, 25), (99.0, 473)),
)
def test_ltx25_duration_snaps_to_official_temporal_grid(seconds, expected):
    assert seconds_to_ltx25_frames(seconds, frame_rate=24.0) == expected


def test_ltx25_auto_duration_modifier_is_explicit_and_bounded():
    source = LTX25GenerationConfig(duration_seconds=5.0)
    updated, raw = WeeToddLTX25AutoDuration().apply(source, 2.0, 12.0)
    assert source.duration_mode == "manual"
    assert updated.duration_mode == "automatic"
    assert updated.duration_seconds == 5.0
    assert updated.auto_duration_min_seconds == 2.0
    assert updated.auto_duration_max_seconds == 12.0
    assert json.loads(raw)["scope"] == "one-shot generation"
    with pytest.raises(ValueError, match="bounds"):
        WeeToddLTX25AutoDuration().apply(source, 10.0, 2.0)


def test_ltx25_generated_keyframes_modifier_preserves_base_workflow_contract():
    source = LTX25GenerationConfig()
    updated = WeeToddLTX25GeneratedKeyframes().apply(source, 3)[0]
    assert source.generated_keyframes == 0
    assert updated.generated_keyframes == 3


def test_ltx25_diffvae_modifier_preserves_base_node_schema():
    source = LTX25GenerationConfig()
    updated, raw = WeeToddLTX25DiffVAEOptimization().apply(source, "deferred_stage4", 512, 4, 32)
    assert source.diffvae_optimization == "combined"
    assert updated.diffvae_optimization == "deferred_stage4"
    assert json.loads(raw)["applies_only_to"] == "Diffusion VAE checkpoints"


def test_ltx25_diffvae_modifier_accepts_metal_na3d_experiment():
    updated, raw = WeeToddLTX25DiffVAEOptimization().apply(
        LTX25GenerationConfig(), "metal_na3d_experimental", 512, 4, 32
    )
    assert updated.diffvae_optimization == "metal_na3d_experimental"
    assert json.loads(raw)["optimization"] == "metal_na3d_experimental"


def test_ltx25_diffvae_modifier_accepts_query_tiled_metal_na3d_experiment():
    updated, raw = WeeToddLTX25DiffVAEOptimization().apply(
        LTX25GenerationConfig(), "metal_na3d_query_tiled_experimental", 65536, 4, 32
    )
    assert updated.diffvae_query_chunk_size == 65536
    assert json.loads(raw)["optimization"] == "metal_na3d_query_tiled_experimental"


@pytest.mark.parametrize(
    ("optimization", "expected_backend"),
    (
        ("combined", "einsum"),
        ("metal_na3d_experimental", "metal"),
        ("metal_na3d_query_tiled_experimental", "metal_tiled"),
    ),
)
def test_ltx25_video_decoder_maps_diffvae_attention_backend(
    monkeypatch, tmp_path, optimization, expected_backend
):
    import ltx25_mlx.components as components
    import ltx25_mlx.diffusion_vae as diffusion_vae

    checkpoint = tmp_path / "diffusion-vae.safetensors"
    checkpoint.touch()
    metadata = {"vae": {"decoder": {"_class_name": "DiffusionDecoder"}}}
    monkeypatch.setattr(components, "_metadata_config", lambda _path: metadata)
    monkeypatch.setattr(components, "_cleanup", lambda: None)
    calls = {}
    sentinel = object()

    def fake_loader(path, config, **kwargs):
        calls.update(path=path, config=config, **kwargs)
        return sentinel

    monkeypatch.setattr(diffusion_vae, "load_diffusion_video_decoder", fake_loader)
    decoder = LTX25VideoDecoder(checkpoint, diffvae_optimization=optimization)

    assert decoder.load() is sentinel
    assert calls["attention_backend"] == expected_backend


def test_ltx25_lora_loader_builds_lazy_ordered_stack(tmp_path):
    spec = _bundle(tmp_path)
    adapter = tmp_path / "motion.safetensors"
    save_file(
        {
            "transformer_blocks.0.attn1.to_q.lora_A.weight": np.zeros((2, 4), dtype=np.float32),
            "transformer_blocks.0.attn1.to_q.lora_B.weight": np.zeros((4, 2), dtype=np.float32),
        },
        adapter,
        metadata={"model_version": "2.5.0", "reference_downscale_factor": "1"},
    )

    attached, raw = WeeToddLTX25LoRALoader().attach(spec, str(adapter), 0.75)
    assert spec.loras == ()
    assert attached.loras == ((str(adapter), 0.75),)
    assert json.loads(raw)["adapter_pairs"] == 1
    report = attached.validate()
    lora_component = report["components"][-1]
    assert lora_component["component"] == "transformer_lora_1"
    assert lora_component["strength"] == 0.75
    assert lora_component["adapter_role"] == "ic_lora"
    assert report["checkpoint_bytes"] >= adapter.stat().st_size


def test_ltx25_distilled_lora_header_is_generic_transformer_adapter(tmp_path):
    adapter = tmp_path / "distilled.safetensors"
    save_file(
        {
            "diffusion_model.adaln_single.emb.timestep_embedder.linear_1.lora_A.weight": np.zeros(
                (2, 4), dtype=np.float32
            ),
            "diffusion_model.adaln_single.emb.timestep_embedder.linear_1.lora_B.weight": np.zeros(
                (4, 2), dtype=np.float32
            ),
        },
        adapter,
        metadata={"model_version": "2.5.0", "lora_rank": "2", "lora_alpha": "2"},
    )

    report = inspect_ltx25_lora(adapter)
    assert report["adapter_role"] == "transformer_lora"
    assert report["reference_downscale_factor"] == 1
    assert report["lora_rank"] == 2
    assert report["lora_alpha"] == 2


def test_ltx25_non_block_lora_targets_are_fused_independently():
    import mlx.core as mx

    weights = {
        "adaln.weight": mx.zeros((2, 2), dtype=mx.float32),
        "transformer_blocks.0.proj.weight": mx.zeros((2, 2), dtype=mx.float32),
    }
    adapter = {
        "adaln.lora_A.weight": mx.eye(2, dtype=mx.float32),
        "adaln.lora_B.weight": mx.eye(2, dtype=mx.float32),
        "transformer_blocks.0.proj.lora_A.weight": mx.eye(2, dtype=mx.float32),
        "transformer_blocks.0.proj.lora_B.weight": mx.eye(2, dtype=mx.float32),
    }

    fused = dict(_fuse_non_block_loras(weights, [(adapter, 0.5)]))
    assert set(fused) == {"adaln.weight"}
    assert mx.array_equal(fused["adaln.weight"], mx.eye(2, dtype=mx.float32) * 0.5)


def test_ltx25_dfr_modifier_selects_stage2_pixel_spatial_lora(tmp_path):
    adapter = tmp_path / "detail.safetensors"
    save_file(
        {
            "transformer_blocks.0.attn1.to_q.lora_A.weight": np.zeros((2, 4), dtype=np.float32),
            "transformer_blocks.0.attn1.to_q.lora_B.weight": np.zeros((4, 2), dtype=np.float32),
        },
        adapter,
        metadata={
            "model_version": "2.5.0",
            "reference_downscale_factor": "2",
            "reference_spatial_scale_factor": "2",
        },
    )
    config, raw = WeeToddLTX25DFRDetailing().apply(
        LTX25GenerationConfig(duration_seconds=2.0), str(adapter), 1.0
    )
    report = json.loads(raw)
    assert config.dfr_enabled is True
    assert config.generated_keyframes == 0
    assert report["generated_keyframe_positions"] == [24, 48]
    assert report["audio_source"] == "stage_1"


def test_ltx25_dfr_recipe_detects_official_rank_450_adapter():
    config = LTX25GenerationConfig(dfr_enabled=True)
    report = {
        "components": [
            {
                "adapter_role": "transformer_lora",
                "lora_rank": 450,
                "lora_alpha": 450,
            }
        ]
    }
    assert resolve_ltx25_dfr_recipe(config, report) == "official_dev_distilled_lora"
    assert (
        resolve_ltx25_dfr_recipe(config, {"components": []})
        == "fused_distilled_experimental"
    )
    assert resolve_ltx25_dfr_recipe(LTX25GenerationConfig(), report) == "disabled"


def test_ltx25_timed_keyframes_are_composable_and_validate_window():
    image = np.zeros((1, 64, 64, 3), dtype=np.float32)
    node = WeeToddLTX25Keyframe()
    stack, _ = node.append(image, 0, 1.0)
    stack, info = node.append(image, 120, 0.7, stack)
    assert isinstance(stack, LTX25KeyframeStack)
    stack.validate(121)
    assert json.loads(info)["keyframes"] == [
        {"frame_index": 0, "strength": 1.0},
        {"frame_index": 120, "strength": 0.7},
    ]
    with pytest.raises(ValueError, match="outside"):
        stack.validate(120)


def test_ltx25_media_conditioning_composes_image_keyframes():
    image = np.zeros((1, 32, 32, 3), dtype=np.float32)
    node = WeeToddLTX25MediaConditioning()

    stack, raw = node.append("image_keyframe", 8, 8, 0.75, images=image)

    assert isinstance(stack, LTX25MediaConditioningStack)
    stack.validate_for_generation(17)
    assert json.loads(raw)["items"] == [
        {
            "end_frame": 8,
            "role": "image_keyframe",
            "start_frame": 8,
            "strength": 0.75,
        }
    ]


def test_ltx25_media_conditioning_rejects_unimplemented_roles_before_generation():
    images = np.zeros((9, 32, 32, 3), dtype=np.float32)
    stack = WeeToddLTX25MediaConditioning().append(
        "video_reference", 0, 8, 1.0, images=images
    )[0]

    with pytest.raises(ValueError, match="requires its IC-LoRA or audio pipeline"):
        stack.validate_for_generation(17)


@pytest.mark.parametrize(
    ("mode_index", "pipeline_mode", "stage1_steps", "stage1_sampler", "stg_scale"),
    (
        (0, "distilled", 8, "euler_ancestral", 0.0),
        (1, "guided", 30, "euler_guided", 1.0),
        (2, "guided_hq", 15, "res_2s_guided", 0.0),
    ),
)
def test_ltx25_quality_mode_pins_complete_recipe(
    mode_index, pipeline_mode, stage1_steps, stage1_sampler, stg_scale
):
    updated, raw = WeeToddLTX25QualityMode().apply(
        LTX25GenerationConfig(),
        LTX25_QUALITY_MODES[mode_index],
        "bad output",
    )

    assert updated.pipeline_mode == pipeline_mode
    assert updated.stage1_steps == stage1_steps
    assert updated.stage1_sampler == stage1_sampler
    assert updated.stage2_steps == 3
    assert updated.stg_scale == stg_scale
    assert json.loads(raw)["requires_guided_model_loader"] == (mode_index != 0)


def test_ltx25_generated_keyframes_use_even_interior_pixel_frames():
    assert evenly_spaced_keyframe_positions(3, 121) == (30, 60, 90)
    slots = GeneratedKeyframeSlots((30, 60, 90), spatial_dims=(16, 4, 6), frame_rate=24.0)
    assert slots.token_count == 72


def test_ltx25_dfr_canvas_matches_official_segment_policy():
    assert choose_dfr_segment_length(48) == 24
    assert choose_dfr_segment_length(96) == 32
    assert resolve_dfr_canvas(49) == (49, 24, (24, 48))
    assert resolve_dfr_canvas(41) == (49, 24, (24, 48))


def test_ltx25_dfr_temporal_tiles_are_gapless_after_discarded_lead_in():
    import mlx.core as mx

    tiles = plan_dfr_temporal_tiles((48, 96), 97, 2)
    assert [(tile.pixel_start, tile.pixel_end) for tile in tiles] == [(0, 48), (0, 96)]
    assert [tile.drop_latent_prefix for tile in tiles] == [0, 7]
    latents = [
        mx.full((1, 1, tile.latent_end_exclusive - tile.latent_start, 1, 1), index)
        for index, tile in enumerate(tiles)
    ]
    stitched = stitch_dfr_temporal_tiles(latents, tiles)
    assert stitched.shape == (1, 1, 13, 1, 1)
    assert stitched[:, :, :7].tolist() == latents[0].tolist()


def test_ltx25_dfr_temporal_image_anchors_reuse_encoded_rows_and_scale_frames():
    import mlx.core as mx

    class First:
        frame_indices = [0]
        clean_latent = mx.arange(24).reshape(1, 6, 4)
        strength = 1.0

    class Middle:
        frame_idx = 17
        keyframe_latent = mx.arange(24, 48).reshape(1, 6, 4)
        strength = 0.7

    anchors = extract_dfr_temporal_image_anchors([First(), Middle()], latent_h=2, latent_w=3)
    assert [anchor.pixel_frame for anchor in anchors] == [0, 17]
    assert [anchor.replace for anchor in anchors] == [True, False]
    assert anchors[0].latent_tokens.tolist() == First.clean_latent.tolist()
    scaled = scale_dfr_temporal_image_anchors(anchors)
    assert [anchor.pixel_frame for anchor in scaled] == [0, 34]
    assert scaled[1].strength == pytest.approx(0.7)
    assert scaled[1].latent_tokens.tolist() == Middle.keyframe_latent.tolist()


def test_ltx25_dfr_temporal_image_anchor_rejects_duplicate_frames():
    from types import SimpleNamespace

    import mlx.core as mx

    duplicate = DFRTemporalImageAnchor(8, mx.zeros((1, 4, 2)), 1.0, False)
    with pytest.raises(ValueError, match="distinct"):
        # Exercise extraction's collision contract through two keyframe-like values.
        extract_dfr_temporal_image_anchors(
            [
                SimpleNamespace(
                    frame_idx=duplicate.pixel_frame,
                    keyframe_latent=duplicate.latent_tokens,
                    strength=1.0,
                ),
                SimpleNamespace(
                    frame_idx=duplicate.pixel_frame,
                    keyframe_latent=duplicate.latent_tokens,
                    strength=0.8,
                ),
            ],
            latent_h=2,
            latent_w=2,
        )


def test_ltx25_dfr_temporal_slots_are_selected_after_appended_anchors():
    import mlx.core as mx

    generated = mx.full((1, 4, 2), 1)
    explicit_anchor = mx.full((1, 2, 2), 2)
    generated_slot = mx.full((1, 2, 2), 3)
    result = mx.concatenate([generated, explicit_anchor, generated_slot], axis=1)
    selected = select_dfr_generated_slot_tokens(result, 2)
    assert selected.tolist() == generated_slot.tolist()


def test_ltx25_dfr_temporal_modifier_preserves_audio_policy(tmp_path):
    adapter = tmp_path / "detail.safetensors"
    save_file(
        {
            "transformer_blocks.0.attn1.to_q.lora_A.weight": np.zeros(
                (2, 4), dtype=np.float32
            ),
            "transformer_blocks.0.attn1.to_q.lora_B.weight": np.zeros(
                (4, 2), dtype=np.float32
            ),
        },
        adapter,
        metadata={"model_version": "2.5.0", "reference_downscale_factor": "2"},
    )
    temporal = tmp_path / "temporal.safetensors"
    save_file(
        {"conv_in.weight": np.zeros((1,), dtype=np.float32)},
        temporal,
        metadata={
            "config": json.dumps(
                {
                    "_class_name": "LatentUpsampler",
                    "in_channels": 128,
                    "dims": 3,
                    "spatial_upsample": False,
                    "temporal_upsample": True,
                    "rational_resampler": True,
                }
            )
        },
    )
    config, _ = WeeToddLTX25DFRDetailing().apply(
        LTX25GenerationConfig(duration_seconds=2.0), str(adapter), 1.0
    )
    updated, raw = WeeToddLTX25DFRTemporalRefinement().apply(config, str(temporal), 1)
    assert updated.dfr_temporal_rounds == 1
    assert updated.dfr_temporal_upsampler_path == str(temporal)
    assert json.loads(raw)["audio_policy"] == "preserve stage-one audio"


def test_ltx25_video_only_transformer_skips_audio_contract():
    import mlx.core as mx
    from ltx_core_mlx.model.transformer.model import LTXModel, LTXModelConfig

    transformer = LTXModel(
        LTXModelConfig(
            num_layers=1,
            video_dim=8,
            audio_dim=4,
            video_num_heads=2,
            audio_num_heads=1,
            video_head_dim=4,
            audio_head_dim=4,
            av_cross_num_heads=1,
            av_cross_head_dim=4,
            video_patch_channels=4,
            audio_patch_channels=4,
            ff_mult=2.0,
            timestep_embedding_dim=8,
            positional_embedding_max_pos=(20, 32, 32),
            audio_positional_embedding_max_pos=(20,),
        )
    )
    model = LTX25VideoOnlyX0Model(transformer)
    latent = mx.zeros((1, 3, 4), dtype=mx.bfloat16)
    output, audio = model(
        video_latent=latent,
        audio_latent=None,
        sigma=mx.array([0.5], dtype=mx.bfloat16),
        video_text_embeds=None,
        audio_text_embeds=None,
        video_positions=None,
        audio_positions=None,
        video_attention_mask=None,
        audio_attention_mask=None,
    )
    mx.eval(output)
    assert output.shape == latent.shape
    assert bool(mx.all(mx.isfinite(output)).item())
    assert audio is None


def test_ltx25_generated_keyframes_accept_spatially_upscaled_stage1_seeds():
    import mlx.core as mx
    from ltx_core_mlx.conditioning.types.latent_cond import LatentState
    from ltx_core_mlx.utils.positions import compute_video_positions

    initial = mx.arange(4 * 128, dtype=mx.float32).reshape(1, 128, 1, 2, 2)
    state = LatentState(
        latent=mx.zeros((1, 4, 128), dtype=mx.float32),
        clean_latent=mx.zeros((1, 4, 128), dtype=mx.float32),
        denoise_mask=mx.ones((1, 4, 1), dtype=mx.float32),
        positions=compute_video_positions(1, 2, 2, frame_rate=24.0),
    )
    output = GeneratedKeyframeSlots(
        (24,),
        spatial_dims=(1, 2, 2),
        frame_rate=24.0,
        initial_keyframes=initial,
    ).apply(state, (1, 2, 2))
    mx.eval(output.latent)
    assert output.latent.shape == (1, 8, 128)
    expected = initial.transpose(0, 2, 3, 4, 1).reshape(1, 4, 128)
    assert mx.array_equal(output.latent[:, 4:], expected)


def test_ltx25_generated_keyframe_marker_resolves_streaming_wrapper():
    import mlx.core as mx
    import mlx.nn as nn

    class Inner(nn.Module):
        def __init__(self):
            super().__init__()
            self.patchify_proj = nn.Linear(2, 2, bias=False)
            self.keyframes_abs_pos_embedding = mx.array([[3.0, 5.0]])

    class Streaming(nn.Module):
        def __init__(self):
            super().__init__()
            self.inner = Inner()

        def __getattr__(self, name):
            try:
                return super().__getattr__(name)
            except AttributeError:
                return getattr(super().__getattr__("inner"), name)

    model = Streaming()
    model.inner.patchify_proj.weight = mx.eye(2)
    set_generated_keyframe_marker(model, 1)
    output = model.inner.patchify_proj(mx.zeros((1, 2, 2)))
    assert mx.array_equal(output[:, 0], mx.zeros((1, 2)))
    assert mx.array_equal(output[:, 1], mx.array([[3.0, 5.0]]))
    set_generated_keyframe_marker(model, 0)
    assert mx.array_equal(model.inner.patchify_proj(mx.zeros((1, 2, 2))), mx.zeros((1, 2, 2)))


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


def test_ltx25_guided_loader_selects_dev_transformer_and_rank450_stage2_lora(
    monkeypatch, tmp_path
):
    spec = _bundle(tmp_path)
    development = tmp_path / "development.safetensors"
    development.write_bytes((tmp_path / "transformer.safetensors").read_bytes())
    distilled_lora = _rank450_lora(tmp_path / "distilled-lora.safetensors")
    monkeypatch.setattr(
        "wee_todd_nodes.ltx25_nodes._resolve_component",
        lambda value, _categories: value,
    )

    updated, raw = WeeToddLTX25GuidedModelLoader().attach(
        spec, str(development), str(distilled_lora)
    )

    assert updated.transformer_path == str(development)
    assert updated.distilled_lora_path == str(distilled_lora)
    assert json.loads(raw)["lora_rank"] == 450
    report = updated.validate("guided")
    assert report["model_version"] == "2.5.0"
    with pytest.raises(ValueError, match="Guided Model Loader"):
        updated.validate("distilled")


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
    assert config.stage2_sampler == "euler"
    from ltx25_mlx.runtime import LTX25_STAGE2_SIGMAS

    assert LTX25_STAGE2_SIGMAS == (0.909375, 0.725, 0.421875, 0.0)
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


def test_ltx25_video_decoder_reports_direct_temporal_streaming(monkeypatch):
    import mlx.core as mx

    class FakeDecoder:
        def decode_and_stream(self, *args, **kwargs):
            self.call = (args, kwargs)

    monkeypatch.setenv("LTX2_VAE_DECODE_BUDGET_GB", "0.00001")
    wrapper = LTX25VideoDecoder("unused.safetensors")
    wrapper._decoder = FakeDecoder()
    latent = mx.zeros((1, 24, 16, 16, 24), dtype=mx.bfloat16)
    assert wrapper.decode_and_stream(latent, "unused.mp4", frame_rate=24.0) == "unused.mp4"
    assert wrapper.last_decode_report["publication"] == "direct_ffmpeg_stream"
    assert wrapper.last_decode_report["temporal_tiling"] is True
    assert wrapper.last_decode_report["tile_frames"] >= 16
    assert wrapper.last_decode_report["overlap_frames"] < wrapper.last_decode_report["tile_frames"]


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


def test_ltx25_chained_node_exposes_exact_timeline_controls():
    inputs = WeeToddLTX25GenerateChained.INPUT_TYPES()["required"]

    assert inputs["window_count"][1]["default"] == 3
    assert inputs["overlap_frames"][1]["default"] == 25
    assert inputs["overlap_frames"][1]["step"] == 8
    assert all(f"prompt_{index}" in inputs for index in range(1, 5))
    assert WeeToddLTX25GenerateChained.OUTPUT_NODE is True


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


def test_ltx25_runtime_forwards_guided_recipe_without_loading_during_preflight(
    tmp_path, monkeypatch
):
    import mlx.core as mx

    base = _bundle(tmp_path)
    development = tmp_path / "development.safetensors"
    development.write_bytes((tmp_path / "transformer.safetensors").read_bytes())
    spec = replace(
        base,
        transformer_path=str(development),
        distilled_lora_path=str(_rank450_lora(tmp_path / "distilled-lora.safetensors")),
    )
    config = WeeToddLTX25QualityMode().apply(
        LTX25GenerationConfig(), LTX25_QUALITY_MODES[2], "bad output"
    )[0]
    calls = {}

    class FakePipeline:
        def __init__(self, transformer_path, distilled_lora_path, low_memory):
            calls["init"] = (transformer_path, distilled_lora_path, low_memory)

        def generate_and_save(self, **kwargs):
            calls["generate"] = kwargs
            return kwargs["output_path"]

    monkeypatch.setattr("ltx25_mlx.runtime._pipeline_class", lambda: FakePipeline)
    monkeypatch.setattr(mx, "reset_peak_memory", lambda: None)
    monkeypatch.setattr(mx, "get_peak_memory", lambda: 456)
    monkeypatch.setattr(mx, "clear_cache", lambda: None)

    info = LTX25RuntimeCache().generate_to_file(
        spec, config, "A guided test prompt.", tmp_path / "guided.mp4"
    )

    assert calls["init"] == (str(development), spec.distilled_lora_path, True)
    assert calls["generate"]["pipeline_mode"] == "guided_hq"
    assert calls["generate"]["stage1_steps"] == 15
    assert calls["generate"]["stage1_sampler"] == "res_2s_guided"
    assert calls["generate"]["video_cfg_scale"] == 3.0
    assert calls["generate"]["audio_cfg_scale"] == 7.0
    assert info["mlx_peak_bytes"] == 456


def test_ltx25_backend_capability_reports_project_native_pipeline():
    status = backend_capability()
    assert status == {"ready": True, "pipeline_class": "LTX25DistilledPipeline"}
