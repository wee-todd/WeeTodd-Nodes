from dataclasses import replace

import mlx.core as mx
import mlx.nn as nn
import pytest

from minimax_h3_mlx.dit import MiniMaxH3DiT, projection_weight_shape
from minimax_h3_mlx.lora import (
    LoRALinear,
    LoRARequest,
    _LoRAProjection,
    apply_lora,
    lora_evaluation,
    prepare_lora_timesteps,
)
from tests.test_dit_smoke import tiny_config
from wee_todd_nodes.lora import H3LoRASpec, H3LoRAStack


@pytest.mark.parametrize("split", [False, True])
@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
def test_batched_lora_inputs_preserve_scaled_ordered_updates(split, dtype):
    mx.random.seed(33)
    base = nn.Linear(64, 192 if split else 64, bias=False)
    base.set_dtype(dtype)
    adapters = [
        _LoRAProjection(
            mx.random.normal((rank, 64)).astype(dtype) * 0.03,
            mx.random.normal((64, rank)).astype(dtype) * 0.02,
            scale, start_after_evaluations=start,
            output_slot=slot if split else None,
            qkv_head_dim=16 if split else None,
        )
        for rank, scale, start, slot in (
            (4, 0.7, 0, 0), (4, -0.3, 0, 1), (4, 1.2, 0, 2),
            (2, 0.4, 0, 0), (3, 0.9, 2, 2), (3, -0.2, 2, 1),
        )
    ]
    layer = LoRALinear(base, adapters)
    value = mx.random.normal((2, 13, 64)).astype(dtype)
    for step in (0, 2):
        with lora_evaluation(step, 4):
            layer.configure_batched_inputs(False)
            expected = layer(value)
            layer.configure_batched_inputs(True)
            actual = layer(value)
            mx.eval(actual, expected)
        assert len(layer._input_groups) == 2
        assert mx.allclose(actual, expected, atol=0.008 if dtype == mx.bfloat16 else 1e-6).item()


def test_batched_lora_does_not_batch_prepared_adaln_grids():
    base = nn.Linear(4, 6)
    adapters = [
        _LoRAProjection(mx.ones((2, 4)), mx.ones((6, 2)), 0.5,
                        source_grid=mx.ones((3, 4))) for _ in range(2)
    ]
    layer = LoRALinear(base, adapters)
    layer.prepare(mx.array([0.2, 0.8]))
    value = mx.ones((2, 4))
    expected = layer(value)
    layer.configure_batched_inputs()
    assert not layer._input_groups
    assert mx.array_equal(expected, layer(value)).item()


def _save(path, tensors):
    mx.save_safetensors(path, tensors, metadata={"base_model": "MiniMax-H3"})


def test_generic_lora_runs_as_activation_space_update(tmp_path):
    model = MiniMaxH3DiT(tiny_config())
    layer = model.blocks[0].attn.out_proj
    value = mx.arange(2 * 64, dtype=mx.float32).reshape(2, 64) / 128
    base = layer(value)
    a = mx.full((2, 64), 0.01, dtype=mx.bfloat16)
    b = mx.full((64, 2), 0.02, dtype=mx.bfloat16)
    path = tmp_path / "generic.safetensors"
    _save(
        path,
        {
            "blocks.0.attn.out_proj.lora_A.weight": a,
            "blocks.0.attn.out_proj.lora_B.weight": b,
        },
    )

    report = apply_lora(model, LoRARequest(str(path), strength=0.5))
    output = model.blocks[0].attn.out_proj(value)
    expected = base + ((value.astype(a.dtype) @ a.T) @ b.T * 0.5).astype(base.dtype)
    mx.eval(output, expected)

    assert mx.array_equal(output, expected)
    assert report.targets == 1
    assert projection_weight_shape(model.blocks[0].attn.out_proj) == list(layer.weight.shape)
    assert report.adaln_targets == 0
    assert report.qkv_permuted_targets == 0


def test_h3_lowercase_peft_lora_is_normalized_and_applied(tmp_path):
    model = MiniMaxH3DiT(tiny_config())
    layer = model.blocks[0].attn.out_proj
    value = mx.arange(2 * 64, dtype=mx.float32).reshape(2, 64) / 128
    base = layer(value)
    a = mx.full((2, 64), 0.01, dtype=mx.float32)
    b = mx.full((64, 2), 0.02, dtype=mx.float32)
    path = tmp_path / "renamed-community-adapter.safetensors"
    mx.save_safetensors(
        path,
        {
            "base_model.model.transformer.blocks.0.attn.out_proj.lora_a.weight": a,
            "base_model.model.transformer.blocks.0.attn.out_proj.lora_b.weight": b,
        },
        metadata={"base_model": "MiniMax-H3"},
    )

    report = apply_lora(model, LoRARequest(str(path), strength=0.5))
    output = model.blocks[0].attn.out_proj(value)
    expected = base + ((value @ a.T) @ b.T * 0.5).astype(base.dtype)
    mx.eval(output, expected)

    spec = H3LoRASpec(str(path))
    spec.validate()
    assert mx.array_equal(output, expected)
    assert report.targets == 1
    assert spec.structural_descriptor["pair_schemas"] == ["lowercase_ab"]


def test_h3_global_alpha_and_rank_metadata_are_applied_once(tmp_path):
    model = MiniMaxH3DiT(tiny_config())
    layer = model.blocks[0].attn.out_proj
    value = mx.ones((1, 64), dtype=mx.float32)
    base = layer(value)
    a = mx.ones((2, 64), dtype=mx.float32)
    b = mx.ones((64, 2), dtype=mx.float32)
    path = tmp_path / "global-scale.safetensors"
    mx.save_safetensors(
        path,
        {
            "blocks.0.attn.out_proj.lora_down.weight": a,
            "blocks.0.attn.out_proj.lora_up.weight": b,
        },
        metadata={"ss_network_dim": "4", "ss_network_alpha": "1"},
    )

    apply_lora(model, LoRARequest(str(path), strength=0.5))
    output = model.blocks[0].attn.out_proj(value)
    expected = base + ((value @ a.T) @ b.T * 0.125).astype(base.dtype)
    mx.eval(output, expected)

    assert mx.array_equal(output, expected)


def test_h3_dynamic_exporter_alpha_is_treated_as_baked_scale(tmp_path):
    model = MiniMaxH3DiT(tiny_config())
    value = mx.ones((1, 64), dtype=mx.float32)
    base = model.blocks[0].attn.out_proj(value)
    a = mx.ones((2, 64), dtype=mx.float32)
    b = mx.ones((64, 2), dtype=mx.float32)
    path = tmp_path / "dynamic-alpha.safetensors"
    mx.save_safetensors(
        path,
        {
            "blocks.0.attn.out_proj.lora_A.weight": a,
            "blocks.0.attn.out_proj.lora_B.weight": b,
        },
        metadata={"ss_network_alpha": "Dynamic"},
    )

    apply_lora(model, LoRARequest(str(path)))
    output = model.blocks[0].attn.out_proj(value)
    expected = base + ((value @ a.T) @ b.T).astype(base.dtype)
    mx.eval(output, expected)

    assert mx.array_equal(output, expected)


def test_generic_lora_can_activate_after_base_evaluations(tmp_path):
    model = MiniMaxH3DiT(tiny_config())
    layer = model.blocks[0].attn.out_proj
    value = mx.arange(2 * 64, dtype=mx.float32).reshape(2, 64) / 128
    base = layer(value)
    a = mx.full((2, 64), 0.01, dtype=mx.bfloat16)
    b = mx.full((64, 2), 0.02, dtype=mx.bfloat16)
    path = tmp_path / "staged.safetensors"
    _save(
        path,
        {
            "blocks.0.attn.out_proj.lora_A.weight": a,
            "blocks.0.attn.out_proj.lora_B.weight": b,
        },
    )

    report = apply_lora(model, LoRARequest(str(path), start_after_evaluations=2))
    expected_active = base + ((value.astype(a.dtype) @ a.T) @ b.T).astype(base.dtype)
    with lora_evaluation(0, 6):
        first = model.blocks[0].attn.out_proj(value)
    with lora_evaluation(1, 6):
        second = model.blocks[0].attn.out_proj(value)
    with lora_evaluation(2, 6):
        third = model.blocks[0].attn.out_proj(value)
    mx.eval(first, second, third, expected_active)

    assert mx.array_equal(first, base)
    assert mx.array_equal(second, base)
    assert mx.array_equal(third, expected_active)
    assert report.start_after_evaluations == 2


def test_turbo_qkv_lora_converts_contiguous_rows_to_native_head_layout(tmp_path):
    model = MiniMaxH3DiT(tiny_config())
    layer = model.blocks[0].attn.qkv_proj
    heads = model.config.num_attention_heads
    head_dim = model.config.attention_head_dim
    output_width = 3 * heads * head_dim
    rank = 1
    value = mx.ones((1, model.config.hidden_size), dtype=mx.float32)
    base = layer(value)
    a = mx.ones((rank, model.config.hidden_size), dtype=mx.float32)
    contiguous_b = mx.arange(output_width, dtype=mx.float32).reshape(output_width, rank)
    expected_b = (
        contiguous_b.reshape(3, heads, head_dim, rank)
        .transpose(1, 0, 2, 3)
        .reshape(output_width, rank)
    )
    path = tmp_path / "turbo_qkv.safetensors"
    _save(
        path,
        {
            "blocks.0.attn.qkv_proj.lora_A.weight": a,
            "blocks.0.attn.qkv_proj.lora_B.weight": contiguous_b,
        },
    )

    report = apply_lora(model, LoRARequest(str(path), qkv_layout="contiguous_qkv"))
    output = model.blocks[0].attn.qkv_proj(value)
    expected = base + ((value @ a.T) @ expected_b.T).astype(base.dtype)
    mx.eval(output, expected)

    assert mx.array_equal(output, expected)
    assert report.qkv_permuted_targets == 1


def test_fastvideo_v2_adapter_maps_split_qkv_swiglu_and_exact_deltas(tmp_path):
    model = MiniMaxH3DiT(tiny_config())
    hidden = model.config.hidden_size
    inner = model.config.inner_dim
    ffn = model.config.ffn_hidden_size
    rank = 1

    qkv_value = mx.ones((1, hidden), dtype=mx.float32)
    qkv_base = model.blocks[0].attn.qkv_proj(qkv_value)
    q_a = mx.ones((rank, hidden), dtype=mx.bfloat16)
    q_b = mx.arange(inner, dtype=mx.float32).reshape(inner, rank).astype(mx.bfloat16)

    ff_value = mx.ones((1, hidden), dtype=mx.float32)
    ff_base = model.blocks[0].mlp.fc1(ff_value)
    ff_a = mx.ones((rank, hidden), dtype=mx.bfloat16)
    ff_b = mx.arange(2 * ffn, dtype=mx.float32).reshape(2 * ffn, rank).astype(mx.bfloat16)

    patch_layer = model.video_patch_proj
    patch_value = mx.ones((1, patch_layer.weight.shape[1]), dtype=mx.float32)
    patch_base = patch_layer(patch_value)
    patch_weight_delta = mx.full(patch_layer.weight.shape, 0.01, dtype=mx.bfloat16)
    patch_bias_delta = mx.full(patch_layer.bias.shape, 0.02, dtype=mx.bfloat16)
    norm_delta = mx.full(model.blocks[0].norm1.weight.shape, 0.03, dtype=mx.bfloat16)
    norm_before = model.blocks[0].norm1.weight
    q_norm_delta = mx.full(model.blocks[0].attn.q_norm.weight.shape, 0.04, dtype=mx.bfloat16)
    q_norm_before = model.blocks[0].attn.q_norm.weight

    path = tmp_path / "fasth3.safetensors"
    mx.save_safetensors(
        path,
        {
            "transformer_blocks.0.attn.to_q.lora_A.weight": q_a,
            "transformer_blocks.0.attn.to_q.lora_B.weight": q_b,
            "transformer_blocks.0.ff.net.0.proj.lora_A.weight": ff_a,
            "transformer_blocks.0.ff.net.0.proj.lora_B.weight": ff_b,
            "transformer_blocks.0.norm1.diff": norm_delta,
            "transformer_blocks.0.attn.norm_q.diff": q_norm_delta,
            "proj_in.diff": patch_weight_delta,
            "proj_in.diff_b": patch_bias_delta,
        },
        metadata={"format": "fastvideo-lora-v2"},
    )

    report = apply_lora(model, LoRARequest(str(path)))
    qkv_output = model.blocks[0].attn.qkv_proj(qkv_value)
    ff_output = model.blocks[0].mlp.fc1(ff_value)
    patch_output = model.video_patch_proj(patch_value)

    q_delta = (qkv_value.astype(q_a.dtype) @ q_a.T) @ q_b.T
    expected_qkv = qkv_base.reshape(
        1, model.config.num_attention_heads, 3, model.config.attention_head_dim
    )
    expected_qkv = expected_qkv.at[..., 0, :].add(
        q_delta.reshape(1, model.config.num_attention_heads, model.config.attention_head_dim)
    )
    expected_qkv = expected_qkv.reshape(qkv_base.shape)
    swapped_ff_b = mx.concatenate([ff_b[ffn:], ff_b[:ffn]], axis=0)
    expected_ff = ff_base + ((ff_value.astype(ff_a.dtype) @ ff_a.T) @ swapped_ff_b.T).astype(
        ff_base.dtype
    )
    expected_patch = patch_base + patch_bias_delta.astype(patch_base.dtype)
    expected_patch += (patch_value.astype(patch_weight_delta.dtype) @ patch_weight_delta.T).astype(
        patch_base.dtype
    )
    mx.eval(qkv_output, ff_output, patch_output)

    assert mx.array_equal(qkv_output, expected_qkv)
    assert mx.array_equal(ff_output, expected_ff)
    assert mx.allclose(patch_output, expected_patch, rtol=1e-6, atol=1e-6)
    assert mx.array_equal(model.blocks[0].norm1.weight, norm_before + norm_delta)
    assert mx.array_equal(model.blocks[0].attn.q_norm.weight, q_norm_before + q_norm_delta)
    assert report.targets == 6


def test_generic_lora_targets_quantized_projection_by_logical_width(tmp_path):
    model = MiniMaxH3DiT(tiny_config())
    nn.quantize(
        model,
        group_size=32,
        bits=8,
        class_predicate=lambda path, module: path == "blocks.0.attn.out_proj",
    )
    layer = model.blocks[0].attn.out_proj
    value = mx.arange(2 * 64, dtype=mx.float32).reshape(2, 64) / 128
    base = layer(value)
    a = mx.full((2, 64), 0.01, dtype=mx.bfloat16)
    b = mx.full((64, 2), 0.02, dtype=mx.bfloat16)
    path = tmp_path / "quantized.safetensors"
    _save(
        path,
        {
            "blocks.0.attn.out_proj.lora_A.weight": a,
            "blocks.0.attn.out_proj.lora_B.weight": b,
        },
    )

    report = apply_lora(model, LoRARequest(str(path)))
    output = model.blocks[0].attn.out_proj(value)
    expected = base + ((value.astype(a.dtype) @ a.T) @ b.T).astype(base.dtype)
    mx.eval(output, expected)

    assert mx.array_equal(output, expected)
    assert report.targets == 1
    assert projection_weight_shape(model.blocks[0].attn.out_proj) == list(layer.weight.shape)


def test_pruned_adaln_lora_uses_supplied_original_input_grid(tmp_path):
    config = replace(tiny_config(), time_embed_dim=8, adaln_curve_grid=5)
    model = MiniMaxH3DiT(config)
    layer = model.blocks[0].adaln_proj.linear
    rank = 2
    original_width = 6
    a = mx.full((rank, original_width), 0.01, dtype=mx.bfloat16)
    b = mx.full((config.adaln_out_features, rank), 0.02, dtype=mx.bfloat16)
    lora_path = tmp_path / "pruned.safetensors"
    grid_path = tmp_path / "grid.safetensors"
    grid = mx.arange(5 * original_width, dtype=mx.float32).reshape(5, original_width)
    _save(
        lora_path,
        {
            "blocks.0.adaln_proj.linear.lora_A.weight": a,
            "blocks.0.adaln_proj.linear.lora_B.weight": b,
        },
    )
    _save(grid_path, {"silu_t_emb_grid": grid})

    apply_lora(model, LoRARequest(str(lora_path), adaln_input_grid=str(grid_path)))
    timesteps = mx.array([0.0, 0.5, 1.0])
    prepare_lora_timesteps(model, timesteps)
    value = mx.ones((3, config.time_embed_dim), dtype=mx.float32)
    base = layer(value)
    output = model.blocks[0].adaln_proj.linear(value)
    selected = mx.stack([grid[0], grid[2], grid[4]])
    expected = base + ((selected.astype(a.dtype) @ a.T) @ b.T).astype(base.dtype)
    mx.eval(output, expected)

    assert mx.array_equal(output, expected)


def test_fastvideo_exact_delta_uses_pruned_adaln_input_grid(tmp_path):
    config = replace(tiny_config(), time_embed_dim=8, adaln_curve_grid=5)
    model = MiniMaxH3DiT(config)
    layer = model.final_layer.adaln_proj.linear
    original_width = 6
    weight_delta = mx.full((config.final_adaln_out_features, original_width), 0.01)
    lora_a = mx.full((2, original_width), 0.02)
    lora_b = mx.full((config.final_adaln_out_features, 2), 0.03)
    grid = mx.arange(5 * original_width, dtype=mx.float32).reshape(5, original_width)
    adapter_path = tmp_path / "fasth3_pruned.safetensors"
    grid_path = tmp_path / "grid.safetensors"
    mx.save_safetensors(
        adapter_path,
        {
            "norm_out.linear.diff": weight_delta,
            "norm_out.linear.lora_A.weight": lora_a,
            "norm_out.linear.lora_B.weight": lora_b,
        },
        metadata={"format": "fastvideo-lora-v2"},
    )
    _save(grid_path, {"silu_t_emb_grid": grid})

    apply_lora(
        model,
        LoRARequest(str(adapter_path), adaln_input_grid=str(grid_path)),
    )
    timesteps = mx.array([0.0, 0.5, 1.0])
    prepare_lora_timesteps(model, timesteps)
    value = mx.ones((3, config.time_embed_dim), dtype=mx.float32)
    base = layer(value)
    selected = mx.stack([grid[0], grid[2], grid[4]])
    expected = base + (selected @ weight_delta.T).astype(base.dtype)
    expected += ((selected @ lora_a.T) @ lora_b.T).astype(base.dtype)
    output = model.final_layer.adaln_proj.linear(value)
    mx.eval(output, expected)

    assert mx.allclose(output, expected, rtol=1e-6, atol=1e-6)


def test_fastvideo_rejects_pruned_model_when_timestep_embedder_changed(tmp_path):
    config = replace(tiny_config(), time_embed_dim=8, adaln_curve_grid=5)
    model = MiniMaxH3DiT(config)
    path = tmp_path / "fasth3_time_embed.safetensors"
    mx.save_safetensors(
        path,
        {
            "time_embedder.linear_1.diff": mx.zeros(
                (config.time_embed_hidden_size, config.timestep_input_dim)
            ),
        },
        metadata={"format": "fastvideo-lora-v2"},
    )

    with pytest.raises(ValueError, match="cannot be applied to a pruned-AdaLN"):
        apply_lora(model, LoRARequest(str(path)))


def test_pruned_adaln_lora_requires_input_grid(tmp_path):
    config = replace(tiny_config(), time_embed_dim=8, adaln_curve_grid=5)
    model = MiniMaxH3DiT(config)
    path = tmp_path / "turbo.safetensors"
    _save(
        path,
        {
            "blocks.0.adaln_proj.linear.lora_A.weight": mx.zeros((2, 6)),
            "blocks.0.adaln_proj.linear.lora_B.weight": mx.zeros((config.adaln_out_features, 2)),
        },
    )

    with pytest.raises(ValueError, match="Supply an AdaLN input-grid"):
        apply_lora(model, LoRARequest(str(path)))


def test_lazy_lora_stack_auto_is_filename_invariant_and_defaults_standard(tmp_path):
    path = tmp_path / "minimax_h3_turbo_example.safetensors"
    _save(
        path,
        {
            "blocks.0.attn.out_proj.lora_A.weight": mx.zeros((2, 64)),
            "blocks.0.attn.out_proj.lora_B.weight": mx.zeros((64, 2)),
        },
    )
    spec = H3LoRASpec(str(path), strength=1.0)
    stack = H3LoRAStack().append(spec)

    assert spec.resolved_profile == "standard"
    assert spec.resolved_qkv_layout == "native_interleaved"
    assert spec.engine_request()["qkv_layout"] == "native_interleaved"
    assert spec.profile_classification_basis == (
        "metadata ambiguous; source-independent standard default"
    )
    assert stack.metadata()[0]["file"] == path.name
    assert stack.metadata()[0]["qkv_layout"] == "native_interleaved"
    assert stack.metadata()[0]["structural_descriptor"]["pair_schemas"] == ["ab"]
    assert stack.metadata()[0]["structural_descriptor"]["ranks"] == [2]
    original_profile = spec.resolved_profile
    original_layout = spec.resolved_qkv_layout
    renamed = tmp_path / "ordinary_name.safetensors"
    path.rename(renamed)
    renamed_spec = H3LoRASpec(str(renamed), strength=1.0)
    assert renamed_spec.resolved_profile == original_profile
    assert renamed_spec.resolved_qkv_layout == original_layout


def test_h3_auto_profile_and_qkv_layout_use_metadata(tmp_path):
    path = tmp_path / "ordinary-name.safetensors"
    mx.save_safetensors(
        path,
        {
            "blocks.0.attn.out_proj.lora_A.weight": mx.zeros((2, 64)),
            "blocks.0.attn.out_proj.lora_B.weight": mx.zeros((64, 2)),
        },
        metadata={
            "base_model": "MiniMax-H3",
            "adapter_profile": "turbo",
            "qkv_layout": "contiguous",
        },
    )
    spec = H3LoRASpec(str(path))

    assert spec.resolved_profile == "turbo"
    assert spec.resolved_qkv_layout == "contiguous_qkv"
    assert spec.profile_classification_basis == "checkpoint metadata"
    with pytest.raises(ValueError, match="at least four active"):
        H3LoRAStack((spec,)).validate_for_steps(4)
    H3LoRAStack((spec,)).validate_for_steps(5)


def test_staged_turbo_stack_requires_four_active_evaluations(tmp_path):
    path = tmp_path / "minimax_h3_turbo_staged.safetensors"
    _save(
        path,
        {
            "blocks.0.attn.out_proj.lora_A.weight": mx.zeros((2, 64)),
            "blocks.0.attn.out_proj.lora_B.weight": mx.zeros((64, 2)),
        },
    )
    spec = H3LoRASpec(
        str(path),
        strength=1.0,
        profile="turbo",
        start_after_evaluations=2,
    )
    stack = H3LoRAStack().append(spec)

    with pytest.raises(ValueError, match="at least four active"):
        stack.validate_for_steps(6)
    stack.validate_for_steps(7)
    assert spec.engine_request()["start_after_evaluations"] == 2
    assert stack.metadata()[0]["start_after_evaluations"] == 2


def test_staged_lora_rejects_adaln_targets(tmp_path):
    path = tmp_path / "minimax_h3_turbo_adaln.safetensors"
    _save(
        path,
        {
            "blocks.0.adaln_proj.linear.lora_A.weight": mx.zeros((2, 64)),
            "blocks.0.adaln_proj.linear.lora_B.weight": mx.zeros((64, 2)),
        },
    )

    with pytest.raises(ValueError, match="does not support AdaLN"):
        H3LoRASpec(str(path), start_after_evaluations=2).validate()
