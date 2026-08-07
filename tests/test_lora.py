from dataclasses import replace

import mlx.core as mx
import mlx.nn as nn
import pytest

from minimax_h3_mlx.dit import MiniMaxH3DiT, projection_weight_shape
from minimax_h3_mlx.lora import LoRARequest, apply_lora, prepare_lora_timesteps
from tests.test_dit_smoke import tiny_config
from wee_todd_nodes.lora import H3LoRASpec, H3LoRAStack


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


def test_lazy_lora_stack_validates_headers_and_turbo_steps(tmp_path):
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

    assert spec.resolved_profile == "turbo"
    assert stack.metadata()[0]["file"] == path.name
    with pytest.raises(ValueError, match="at least five"):
        stack.validate_for_steps(4)
    stack.validate_for_steps(5)
