import mlx.core as mx
from mlx.utils import tree_flatten

from ltx25_mlx.diffusion_vae import (
    DiffusionVAEConfig,
    MLXDiffusionVideoDecoder,
    _Attention,
    _fold_legacy_diffusion_gates,
    _Upsample,
)


def _tiny_config(**changes):
    values = {
        "in_channels": 4,
        "out_channels": 3,
        "patch_size": 2,
        "head_dim": 4,
        "stage_channels": (8, 8, 8, 8, 8),
        "stage_depths": (1, 1, 1, 1, 1),
        "stage_kernels": ((3, 3, 3),) * 5,
        "upsamples": (
            ((1, 2, 2), 1),
            ((2, 1, 1), 1),
            ((2, 2, 2), 1),
            ((2, 2, 2), 1),
        ),
        "timestep_scale_multiplier": 1000.0,
        "default_num_inference_steps": 1,
        "model_output_type": "x0",
        "stage5_kernel": (3, 3, 3),
    }
    values.update(changes)
    return DiffusionVAEConfig(**values)


def test_split_qkv_projection_matches_combined_projection():
    attention = _Attention(64, (3, 3, 3), 64, "metal")
    attention.set_dtype(mx.bfloat16)
    inputs = mx.random.normal(
        (1, 3, 3, 3, 64), key=mx.random.key(51), dtype=mx.bfloat16
    )
    combined = attention.qkv(inputs)
    split = mx.concatenate(
        [attention._project_qkv_slice(inputs, index) for index in range(3)], axis=-1
    )
    mx.eval(combined, split)
    assert mx.allclose(combined, split, rtol=2e-2, atol=2e-2)


def test_query_tiled_metal_attention_matches_complete_metal_attention():
    complete_attention = _Attention(64, (3, 3, 3), 64, "metal")
    tiled_attention = _Attention(64, (3, 3, 3), 64, "metal_tiled")
    complete_attention.set_dtype(mx.bfloat16)
    tiled_attention.set_dtype(mx.bfloat16)
    tiled_attention.load_weights(list(tree_flatten(complete_attention.parameters())), strict=True)
    inputs = mx.random.normal(
        (1, 3, 3, 3, 64), key=mx.random.key(52), dtype=mx.bfloat16
    )
    complete = complete_attention(inputs, query_chunk_size=7)
    tiled = tiled_attention(inputs, query_chunk_size=7)
    mx.eval(complete, tiled)
    assert mx.allclose(complete, tiled, rtol=2e-2, atol=2e-2)


def test_diffusion_vae_config_reads_official_nested_metadata():
    config = DiffusionVAEConfig.from_metadata(
        {
            "vae": {
                "model_output_type": "x0",
                "decoder": {
                    "_class_name": "NADiffusionDecoder",
                    "in_channels": 4,
                    "out_channels": 3,
                    "patch_size": 2,
                    "head_dim": 4,
                    "stage_channels": [8, 8, 8, 8, 8],
                    "stage_depths": [1, 1, 1, 1, 1],
                    "stage_kernels": [[3, 3, 3]] * 5,
                    "upsamples": [[(1, 2, 2), 1], [(2, 1, 1), 1], [(2, 2, 2), 1], [(2, 2, 2), 1]],
                    "timestep_scale_multiplier": 1000.0,
                    "default_num_inference_steps": 1,
                    "stage5_kernel": [3, 3, 3],
                },
            }
        }
    )
    assert config == _tiny_config()


def test_diffusion_vae_parameter_tree_matches_checkpoint_contract():
    model = MLXDiffusionVideoDecoder(_tiny_config())
    assert model.query_chunk_size == 512
    assert model.stage4_tile_width == 0
    parameters = dict(tree_flatten(model.parameters()))
    assert parameters["conv_in.weight"].shape == (8, 4)
    assert parameters["det_stages.0.0.attn.qkv.weight"].shape == (24, 8)
    assert parameters["upsamples.3.proj.weight"].shape == (64, 8)
    assert parameters["diff_blocks.0.scale_shift_table"].shape == (7, 8)
    assert parameters["t_embedder.mlp.0.weight"].shape == (384, 256)
    assert parameters["t_embedder.mlp.2.weight"].shape == (384, 384)
    assert parameters["type_emb"].shape == (4,)


def test_diffusion_vae_tiny_decode_is_seeded_and_has_expected_shape():
    model = MLXDiffusionVideoDecoder(
        _tiny_config(), query_chunk_size=8, token_chunk_size=64, seed=12
    )
    latent = mx.zeros((1, 4, 3, 3, 3), dtype=mx.bfloat16)
    first = model.decode(latent)
    second = model.decode(latent)
    mx.eval(first, second)
    assert first.shape == (1, 3, 17, 48, 48)
    assert mx.array_equal(first, second)


def test_diffusion_vae_two_step_x0_schedule_reaches_last_prediction():
    model = MLXDiffusionVideoDecoder(_tiny_config(default_num_inference_steps=2))
    calls = []

    def predict(_context, x_t, timestep, **_kwargs):
        calls.append(timestep)
        return mx.zeros_like(x_t)

    model._predict_diffusion_stage = predict
    noise = mx.ones((1, 1, 1, 1, 12), dtype=mx.float32)
    result = model._run_diffusion_stage(mx.zeros((1, 1, 1, 1, 8)), noise)
    mx.eval(result)
    assert calls == [1.0, 0.5]
    assert mx.array_equal(result, mx.zeros_like(result))


def test_diffusion_vae_velocity_schedule_uses_checkpoint_prediction_directly():
    model = MLXDiffusionVideoDecoder(
        _tiny_config(default_num_inference_steps=2, model_output_type="v")
    )
    model._predict_diffusion_stage = lambda _context, x_t, _t, **_kwargs: mx.ones_like(x_t)
    noise = mx.ones((1, 1, 1, 1, 12), dtype=mx.float32)
    result = model._run_diffusion_stage(mx.zeros((1, 1, 1, 1, 8)), noise)
    mx.eval(result)
    assert mx.array_equal(result, mx.zeros_like(result))


def test_legacy_diffusion_gates_fold_into_projection_and_are_removed():
    weights = {
        "diff_blocks.0.gate_ctx": mx.array([2.0, 3.0]),
        "diff_blocks.0.context_proj.weight": mx.ones((2, 3), dtype=mx.bfloat16),
        "diff_blocks.0.context_proj.bias": mx.ones((2,), dtype=mx.bfloat16),
    }
    folded = _fold_legacy_diffusion_gates(weights)
    mx.eval(*folded.values())
    assert "diff_blocks.0.gate_ctx" not in folded
    assert folded["diff_blocks.0.context_proj.weight"].astype(mx.float32).tolist() == [
        [2.0, 2.0, 2.0],
        [3.0, 3.0, 3.0],
    ]
    assert folded["diff_blocks.0.context_proj.bias"].astype(mx.float32).tolist() == [2.0, 3.0]


def test_deferred_stage_four_matches_combined_context_on_tiny_model():
    combined = MLXDiffusionVideoDecoder(
        _tiny_config(), query_chunk_size=8, token_chunk_size=64, seed=8
    )
    deferred = MLXDiffusionVideoDecoder(
        _tiny_config(),
        query_chunk_size=8,
        token_chunk_size=64,
        deferred_stage4=True,
        context_width_chunks=2,
        seed=8,
    )
    deferred.load_weights(list(tree_flatten(combined.parameters())), strict=True)
    latent = mx.zeros((1, 4, 3, 3, 3), dtype=mx.float32)
    expected = combined.decode(latent)
    actual = deferred.decode(latent)
    mx.eval(expected, actual)
    assert mx.allclose(expected, actual, rtol=2e-4, atol=2e-4)


def test_diffusion_vae_halo_includes_stage4_and_stage5_receptive_fields():
    model = MLXDiffusionVideoDecoder(_tiny_config())
    # One radius-1 block in stage 4 plus ceil(radius-1 / stride-2) for stage 5.
    assert model._stage5_halo_in_stage4_cells() == 2


def test_ltx_patch_channel_order_is_width_then_height():
    channels = mx.arange(4).reshape(1, 1, 1, 1, 1, 2, 2)
    image = channels.transpose(0, 4, 1, 2, 6, 3, 5).reshape(1, 1, 1, 2, 2)
    mx.eval(image)
    assert image.tolist() == [[[[[0, 2], [1, 3]]]]]


def test_diffusion_vae_pixel_shuffle_matches_official_p1_p2_p3_order():
    upsample = _Upsample(1, (2, 2, 2), 1)
    upsample.proj.weight = mx.arange(8, dtype=mx.float32).reshape(8, 1)
    upsample.proj.bias = mx.zeros((8,), dtype=mx.float32)
    value = mx.ones((1, 1, 1, 1, 1), dtype=mx.float32)
    output = upsample(value)
    mx.eval(output)
    assert output.shape == (1, 1, 2, 2, 1)
    assert output[0, 0, :, :, 0].tolist() == [[4.0, 5.0], [6.0, 7.0]]
