import json
import math
import struct
import sys
from dataclasses import replace
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from minimax_h3_mlx.lora import _split_lora_key
from minimax_h3_mlx.vdn import (
    VDNLayout,
    _depthwise_short_conv,
    _factor,
    _frame_alpha,
    _linear_branch,
    _softmax_branch,
    _window_bounds,
)
from wee_todd_nodes.nodes import WeeToddH3VDNCheckpoint
from wee_todd_nodes.preflight import H3ComponentSetSpec
from wee_todd_nodes.runtime import H3GenerationConfig
from wee_todd_nodes.vdn import VDN_STAGES, resolve_vdn_spec, vdn_branch_shapes


def _write_branch_header(path, shapes):
    # A sparse file validates real production shapes without allocating/loading 4 GB.
    offset = 0
    header = {}
    for name, shape in shapes.items():
        size = math.prod(shape) * 2
        header[name] = {"dtype": "BF16", "shape": shape, "data_offsets": [offset, offset + size]}
        offset += size
    raw = json.dumps(header).encode()
    raw += b" " * ((-len(raw)) % 8)
    with path.open("wb") as handle:
        handle.write(struct.pack("<Q", len(raw)) + raw)
        handle.truncate(8 + len(raw) + offset)


def _write_vdn_repository(root, *, turbo=True):
    stage = root / ("stage-dmd-step-250" if turbo else "stage-b-step-2000")
    (stage / "linear_branch").mkdir(parents=True)
    (stage / "adapters" / "default").mkdir(parents=True)
    if turbo:
        (stage / "adapters" / "turbo").mkdir(parents=True)
    model_spec = {
        "format_version": 2,
        "base": {"class_name": "MiniMaxH3Transformer3DModel"},
        "transforms": [
            {
                "type": "hybrid_attention",
                "version": 2,
                "config": {
                    "anchor_frames": "both",
                    "enable_softmax_gate": True,
                    "softmax_attention": {"chunk": 5, "radius": 1},
                    "linear_attention": {
                        "delta_rule": "vdn_solve",
                        "linear_head_dim": 128,
                        "enable_text_state": True,
                        "bridge": "alpha",
                        "a_fp32": True,
                        "short_conv": {"targets": ["k", "v"]},
                    },
                },
            }
        ],
    }
    (stage / "model_spec.json").write_text(json.dumps(model_spec))
    _write_branch_header(stage / "linear_branch" / "model.safetensors", vdn_branch_shapes())
    adapter = {
        "transformer_blocks.0.attn.orig.to_q.lora_A.default.weight": mx.zeros((1, 2)),
        "transformer_blocks.0.attn.orig.to_q.lora_B.default.weight": mx.zeros((6, 1)),
    }
    mx.save_safetensors(str(stage / "adapters" / "default" / "adapter_model.safetensors"), adapter)
    if turbo:
        mx.save_safetensors(
            str(stage / "adapters" / "turbo" / "adapter_model.safetensors"),
            {name.replace(".default.", ".turbo."): value for name, value in adapter.items()},
        )
    return stage


def test_vdn_checkpoint_node_builds_required_8_step_adapter_stack(tmp_path):
    _write_vdn_repository(tmp_path)
    base = tmp_path / "base"
    (base / "transformer").mkdir(parents=True)
    components = H3ComponentSetSpec(checkpoint=str(base), task="t2va")

    returned, config, vdn, loras, raw = WeeToddH3VDNCheckpoint().select(
        components,
        H3GenerationConfig(),
        str(tmp_path),
        "VDN-H3 8-step (recommended)",
    )

    assert returned is components
    assert config.steps == 9
    assert vdn.stage == "stage-dmd-step-250"
    assert [adapter.resolved_profile for adapter in loras.adapters] == ["standard", "turbo"]
    assert all(adapter.resolved_qkv_layout == "contiguous_qkv" for adapter in loras.adapters)
    assert json.loads(raw)["transformer_evaluations"] == 8


def test_vdn_checkpoint_rejects_non_t2va_and_accepts_paged_base(tmp_path):
    _write_vdn_repository(tmp_path)
    transformer = tmp_path / "base" / "transformer"
    transformer.mkdir(parents=True)
    node = WeeToddH3VDNCheckpoint()
    with pytest.raises(ValueError, match="T2VA"):
        node.select(
            H3ComponentSetSpec(checkpoint=str(tmp_path / "base"), task="fl2va"),
            H3GenerationConfig(),
            str(tmp_path),
            "VDN-H3 8-step (recommended)",
        )
    (transformer / "paged_manifest.json").write_text("{}")
    result = node.select(
        H3ComponentSetSpec(checkpoint=str(tmp_path / "base"), task="t2va"),
        H3GenerationConfig(),
        str(tmp_path),
        "VDN-H3 8-step (recommended)",
    )
    assert result[2].stage == "stage-dmd-step-250"


def test_vdn_50_step_checkpoint_uses_default_adapter_only(tmp_path):
    _write_vdn_repository(tmp_path, turbo=False)
    components = H3ComponentSetSpec(checkpoint=str(tmp_path / "base"), task="t2va")
    _, config, vdn, loras, raw = WeeToddH3VDNCheckpoint().select(
        components, H3GenerationConfig(), str(tmp_path), "VDN-H3 50-step",
    )
    assert config.steps == 51
    assert config.sampling_method == "euler"
    assert vdn.turbo_adapter is None
    assert len(loras.adapters) == 1
    assert loras.adapters[0].resolved_profile == "standard"
    assert json.loads(raw)["transformer_evaluations"] == 50
    vdn.validate_sampling(config, loras)


def test_vdn_spec_rejects_drifted_hybrid_contract(tmp_path):
    stage = _write_vdn_repository(tmp_path, turbo=False)
    payload = json.loads((stage / "model_spec.json").read_text())
    payload["transforms"][0]["config"]["softmax_attention"]["radius"] = 2
    (stage / "model_spec.json").write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="differ"):
        resolve_vdn_spec(tmp_path, "VDN-H3 50-step")


def test_vdn_reference_factor_and_window_geometry():
    a = mx.zeros((2, 1, 3, 3), dtype=mx.float32)
    b = mx.ones((2, 1, 3, 3), dtype=mx.float32)
    alpha = mx.full((2, 1, 3), 0.5, dtype=mx.float32)
    transition, injection = _factor(a, b, alpha)

    assert mx.allclose(transition, 0.5 * mx.eye(3)[None, None]).item()
    assert mx.allclose(injection, b).item()
    bounds = _window_bounds(12)
    assert bounds[0] == (-5, 9)
    assert bounds[5] == (0, 14)
    assert bounds[10] == (5, 19)


def test_vdn_stage_labels_are_stable():
    assert list(VDN_STAGES) == ["VDN-H3 8-step (recommended)", "VDN-H3 50-step"]


def test_vdn_peft_attention_wrapper_maps_to_native_fused_qkv():
    assert _split_lora_key("transformer_blocks.0.attn.orig.to_q.lora_A.default.weight") == (
        "blocks.0.attn.qkv_proj",
        "a",
        0,
        False,
    )


def test_vdn_turbo_named_adapter_maps_to_native_fused_qkv():
    assert _split_lora_key("transformer_blocks.0.attn.orig.to_v.lora_B.turbo.weight") == (
        "blocks.0.attn.qkv_proj",
        "b",
        2,
        False,
    )


def test_vdn_node_resolves_registered_comfy_model_roots(tmp_path, monkeypatch):
    repository = tmp_path / "external-models" / "OpenVDN" / "vdn-minimax-h3"
    _write_vdn_repository(repository)
    monkeypatch.setitem(
        sys.modules,
        "folder_paths",
        SimpleNamespace(
            models_dir=str(tmp_path / "comfy-models"),
            get_folder_paths=lambda category: [str(tmp_path / "external-models")],
            folder_names_and_paths={},
        ),
    )
    result = WeeToddH3VDNCheckpoint().select(
        H3ComponentSetSpec(checkpoint=str(tmp_path / "base")),
        H3GenerationConfig(),
        "OpenVDN/vdn-minimax-h3",
        "VDN-H3 8-step (recommended)",
    )
    assert result[2].repository == str(repository.resolve())


@pytest.mark.parametrize("damage", ["missing", "shape", "extra"])
def test_vdn_checks_every_branch_tensor(tmp_path, damage):
    stage = _write_vdn_repository(tmp_path)
    shapes = vdn_branch_shapes()
    key = "transformer_blocks.23.attn.linear_attention.alpha.up.weight"
    if damage == "missing":
        del shapes[key]
    elif damage == "shape":
        shapes[key] = (1, 1)
    else:
        shapes["unexpected.weight"] = (1,)
    _write_branch_header(stage / "linear_branch" / "model.safetensors", shapes)
    with pytest.raises(ValueError, match="incompatible|shape"):
        resolve_vdn_spec(tmp_path, "VDN-H3 8-step (recommended)")


def test_vdn_sampling_requires_stage_schedule_and_full_stack(tmp_path):
    _write_vdn_repository(tmp_path)
    _, config, vdn, loras, _ = WeeToddH3VDNCheckpoint().select(
        H3ComponentSetSpec(checkpoint=str(tmp_path / "base")),
        H3GenerationConfig(),
        str(tmp_path),
        "VDN-H3 8-step (recommended)",
    )
    vdn.validate_sampling(config, loras)
    for invalid in (None, replace(loras, adapters=loras.adapters[:1])):
        with pytest.raises(ValueError, match="complete, unmodified"):
            vdn.validate_sampling(config, invalid)
    for invalid in (replace(config, steps=8), replace(config, sampling_method="res_multistep")):
        with pytest.raises(ValueError, match="Euler"):
            vdn.validate_sampling(invalid, loras)


def _small_weights(heads=2, head_dim=4, hidden=8):
    rng = np.random.default_rng(74)
    inner = heads * head_dim
    shapes = {
        "linear_attention.alpha.A_log": (heads,),
        "linear_attention.alpha.down.weight": (3, hidden),
        "linear_attention.alpha.dt_bias": (inner,),
        "linear_attention.alpha.up.weight": (inner, 3),
        "linear_attention.beta_proj.weight": (heads, hidden),
        "linear_attention.norm.weight": (head_dim,),
        "linear_attention.output_gate.down.weight": (3, hidden),
        "linear_attention.output_gate.up.bias": (inner,),
        "linear_attention.output_gate.up.weight": (inner, 3),
        "softmax_gate.up.bias": (heads,),
        "softmax_gate.up.weight": (heads, hidden),
        "to_out_linear.weight": (hidden, inner),
    }
    weights = {
        name: mx.array(rng.normal(0, 0.2, shape).astype(np.float32))
        for name, shape in shapes.items()
    }
    weights["linear_attention.norm.weight"] = mx.ones((head_dim,))
    for projection in ("k", "v"):
        spatial = np.zeros((inner, 1, 5, 5), np.float32)
        spatial[:, 0, 2, 2] = 1
        temporal = np.zeros((inner, 1, 5), np.float32)
        temporal[:, 0, 2] = 1
        weights[f"linear_attention.short_conv.{projection}_sp.weight"] = mx.array(spatial)
        weights[f"linear_attention.short_conv.{projection}_tm.weight"] = mx.array(temporal)
    return weights


def test_vdn_alpha_keeps_fp32_projections_with_bf16_weights():
    rng = np.random.default_rng(19)
    mean = mx.array(rng.normal(size=(5, 8)).astype(np.float32))
    weights = {name: value.astype(mx.bfloat16) for name, value in _small_weights().items()}
    w = {name: np.array(value.astype(mx.float32)) for name, value in weights.items()}
    delta = np.array(mean) @ w["linear_attention.alpha.down.weight"].T
    delta = delta @ w["linear_attention.alpha.up.weight"].T + w["linear_attention.alpha.dt_bias"]
    expected = np.exp(
        -np.exp(w["linear_attention.alpha.A_log"])[None, :, None]
        * np.logaddexp(delta.reshape(5, 2, 4), 0)
    )
    actual = _frame_alpha(mean, weights, 2, 4)
    np.testing.assert_allclose(np.array(actual), expected, rtol=2e-6, atol=2e-7)


def test_vdn_dispatch_changes_joint_dit_outputs_and_uninstalls_cleanly():
    from minimax_h3_mlx.vdn import H3VDNRuntime
    from tests.test_dit_smoke import _forward_fixture

    dit, config, original_args = _forward_fixture()
    args = list(original_args)
    positions = mx.zeros_like(args[6])
    positions = positions.at[args[7], 0].add(mx.arange(len(args[7])))
    args[6] = positions
    dense_video, dense_audio = dit(*args)
    runtime = H3VDNRuntime.__new__(H3VDNRuntime)
    runtime._configure_inference("verified")
    from minimax_h3_mlx.vdn_metal import VDNFeatureKernels, VDNMatrixSolver

    runtime.solver = VDNMatrixSolver()
    runtime.feature_kernels = VDNFeatureKernels()
    runtime.layout = None
    runtime.calls = 0
    block = _small_weights(
        config.num_attention_heads, config.attention_head_dim, config.hidden_size
    )
    runtime.weights = {
        f"transformer_blocks.{i}.attn.{name}": value
        for i in range(config.num_layers)
        for name, value in block.items()
    }
    dit.set_vdn_runtime(runtime)
    actual_video, actual_audio = dit(*args)
    mx.eval(dense_video, dense_audio, actual_video, actual_audio)
    assert runtime.calls == config.num_layers
    assert all(block.attn.vdn_runtime is None for block in dit.token_refiner.blocks)
    assert mx.all(mx.isfinite(actual_video)).item()
    assert mx.all(mx.isfinite(actual_audio)).item()
    assert not mx.allclose(dense_video, actual_video).item()
    assert not mx.allclose(dense_audio, actual_audio).item()
    with pytest.raises(ValueError, match="diagnostic"):
        dit.blocks[0].attn(mx.zeros((1, 17, config.hidden_size)), diagnostics=object())
    dit.set_vdn_runtime(None)
    restored_video, restored_audio = dit(*args)
    assert mx.allclose(restored_video, dense_video).item()
    assert mx.allclose(restored_audio, dense_audio).item()


def test_vdn_node_attaches_explicit_adaln_input_grid(tmp_path):
    _write_vdn_repository(tmp_path)
    grid = tmp_path / "grid.safetensors"
    mx.save_safetensors(str(grid), {"silu_t_emb_grid": mx.zeros((2, 2688))})
    result = WeeToddH3VDNCheckpoint().select(
        H3ComponentSetSpec(checkpoint=str(tmp_path / "base")),
        H3GenerationConfig(),
        str(tmp_path),
        "VDN-H3 8-step (recommended)",
        str(grid),
    )
    assert result[3].adapters[-1].adaln_input_grid == str(grid)


def test_vdn_paged_forward_matches_resident_and_releases_pages(tmp_path):
    from dataclasses import asdict

    from mlx.utils import tree_flatten

    from minimax_h3_mlx.dit import MiniMaxH3DiT
    from minimax_h3_mlx.paged_checkpoint import convert_to_paged_checkpoint, load_paged_dit
    from minimax_h3_mlx.vdn import H3VDNRuntime
    from tests.test_paged_checkpoint import _tiny_dit_config, _tiny_inputs

    config = _tiny_dit_config()
    resident = MiniMaxH3DiT(config)
    source = tmp_path / "base"
    source.mkdir()
    (source / "config.json").write_text(json.dumps(asdict(config)))
    mx.save_safetensors(
        str(source / "model.safetensors"), dict(tree_flatten(resident.parameters()))
    )
    convert_to_paged_checkpoint(source, tmp_path / "paged")
    paged = load_paged_dit(tmp_path / "paged", window_size=2)
    from minimax_h3_mlx.lora import LoRARequest, apply_lora

    # Preserve all three split QKV updates (they must not collapse under one target),
    # the reversed SwiGLU halves, and fixed token-refiner targets for BOTH adapter names.
    for adapter_name in ("default", "turbo"):
        tensors = {}
        targets = {
            **{
                f"transformer_blocks.0.attn.orig.to_{part}": (config.inner_dim, config.hidden_size)
                for part in ("q", "k", "v")
            },
            "transformer_blocks.0.ff.net.0.proj": (2 * config.ffn_hidden_size, config.hidden_size),
            "token_refiner.refiner_blocks.0.attn.to_q": (config.inner_dim, config.hidden_size),
        }
        for index, (target, (output_width, input_width)) in enumerate(targets.items()):
            tensors[f"{target}.lora_A.{adapter_name}.weight"] = mx.full((2, input_width), 0.01)
            tensors[f"{target}.lora_B.{adapter_name}.weight"] = (
                mx.arange(output_width * 2).reshape(output_width, 2) * (index + 1) / 10000
            )
        adapter_path = tmp_path / f"{adapter_name}.safetensors"
        mx.save_safetensors(str(adapter_path), tensors)
        request = LoRARequest(str(adapter_path), qkv_layout="contiguous_qkv")
        assert apply_lora(resident, request).targets == len(targets)
        assert apply_lora(paged, request).targets == len(targets)
    args = list(_tiny_inputs(config))
    args[6] = mx.zeros_like(args[6]).at[args[7], 0].add(mx.arange(len(args[7])))
    runtime = H3VDNRuntime.__new__(H3VDNRuntime)
    runtime._configure_inference("verified")
    from minimax_h3_mlx.vdn_metal import VDNFeatureKernels, VDNMatrixSolver

    runtime.solver = VDNMatrixSolver()
    runtime.feature_kernels = VDNFeatureKernels()
    runtime.layout = None
    runtime.calls = 0
    block = _small_weights(
        config.num_attention_heads, config.attention_head_dim, config.hidden_size
    )
    runtime.weights = {
        f"transformer_blocks.{i}.attn.{name}": value
        for i in range(config.num_layers)
        for name, value in block.items()
    }
    resident.set_vdn_runtime(runtime)
    paged.set_vdn_runtime(runtime)
    try:
        expected = resident(*args)
        actual = paged(*args)
        mx.eval(expected, actual)
        for left, right in zip(expected, actual, strict=True):
            np.testing.assert_array_equal(np.array(left), np.array(right))
        assert runtime.calls == 2 * config.num_layers
        assert paged.paged_blocks.store.active_page is None
        paged.set_vdn_runtime(None)
        assert paged.paged_blocks.vdn_runtime is None
    finally:
        paged.paged_blocks.close()


def test_vdn_short_conv_identity_preserves_dense_spatial_grid():
    layout = VDNLayout(24, 0, 4, 6, 2, 3, 0, 1)
    tokens = mx.arange(24 * 8).reshape(24, 2, 4).astype(mx.float32)
    weights = _small_weights()
    actual = _depthwise_short_conv(
        tokens,
        weights["linear_attention.short_conv.k_sp.weight"],
        weights["linear_attention.short_conv.k_tm.weight"],
        layout,
    )
    assert mx.array_equal(actual, tokens).item()


def test_vdn_spatial_conv_accumulates_before_bf16_rounding():
    rng = np.random.default_rng(6)
    layout = VDNLayout(72, 0, 3, 24, 4, 6, 0, 1)
    tokens = mx.array(rng.normal(size=(72, 2, 4))).astype(mx.bfloat16)
    spatial = mx.array(rng.normal(size=(8, 1, 5, 5))).astype(mx.bfloat16)
    temporal = mx.zeros((8, 1, 5))
    temporal[:, 0, 2] = 1
    actual = _depthwise_short_conv(tokens, spatial, temporal, layout)
    volume = np.array(tokens.astype(mx.float32)).reshape(3, 4, 6, 8)
    kernel = np.array(spatial.astype(mx.float32))
    padded = np.pad(volume, ((0, 0), (2, 2), (2, 2), (0, 0)))
    expected = np.zeros_like(volume)
    for row in range(5):
        for column in range(5):
            expected += padded[:, row:row + 4, column:column + 6] * kernel[:, 0, row, column]
    expected = mx.array(expected.reshape(72, 2, 4)).astype(mx.bfloat16)
    np.testing.assert_allclose(
        np.array(actual.astype(mx.float32)), np.array(expected.astype(mx.float32)),
        rtol=0, atol=0.0001,
    )


def test_vdn_softmax_matches_independent_dense_mask():
    from minimax_h3_mlx.config import DiTConfig
    from minimax_h3_mlx.dit import Attention

    mx.random.seed(42)
    attn = Attention(DiTConfig(hidden_size=8, num_attention_heads=2, attention_head_dim=4))
    layout = VDNLayout(23, 3, 18, 1, 1, 1, 0, 2)
    x = mx.random.normal((1, 23, 8))
    q, k, v = [mx.random.normal((1, 2, 23, 4)) for _ in range(3)]
    weights = _small_weights()
    mask = np.ones((23, 23), dtype=bool)
    for frame in range(1, 17):
        for other in range(1, 17):
            mask[3 + frame, 3 + other] = abs(frame // 5 - other // 5) <= 1
    expected = mx.fast.scaled_dot_product_attention(q, k, v, scale=attn.scale, mask=mx.array(mask))
    gate = mx.sigmoid(
        x[0] @ weights["softmax_gate.up.weight"].T + weights["softmax_gate.up.bias"]
    ).reshape(23, 2, 1)
    expected = attn.out_proj((expected.transpose(0, 2, 1, 3)[0] * gate).reshape(1, 23, 8))
    actual = _softmax_branch(attn, x, q, k, v, weights, layout)
    np.testing.assert_allclose(np.array(actual), np.array(expected), rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize("frames", [1, 2, 5, 12, 18, 37])
def test_grouped_windows_preserve_anchors_suffix_and_memory_chunks(frames):
    from minimax_h3_mlx.config import DiTConfig
    from minimax_h3_mlx.dit import Attention

    mx.random.seed(13)
    attn = Attention(DiTConfig(hidden_size=8, num_attention_heads=2, attention_head_dim=4))
    attn.query_chunk_size = 3
    attn.head_chunk_size = 1
    layout = VDNLayout(5 + frames * 2, 3, frames, 2, 1, 2, 0, 2)
    x = mx.random.normal((1, layout.sequence, 8))
    q, k, v = [mx.random.normal((1, 2, layout.sequence, 4)) for _ in range(3)]
    weights = _small_weights()
    mask = np.ones((layout.sequence, layout.sequence), dtype=bool)
    for frame in range(1, frames - 1):
        for other in range(1, frames - 1):
            mask[3 + frame * 2:5 + frame * 2, 3 + other * 2:5 + other * 2] = (
                abs(frame // 5 - other // 5) <= 1
            )
    expected = mx.fast.scaled_dot_product_attention(q, k, v, scale=attn.scale, mask=mx.array(mask))
    gate = mx.sigmoid(
        x[0] @ weights["softmax_gate.up.weight"].T + weights["softmax_gate.up.bias"]
    ).reshape(layout.sequence, 2, 1)
    expected = attn.out_proj(
        (expected.transpose(0, 2, 1, 3)[0] * gate).reshape(1, layout.sequence, 8)
    )
    actual = _softmax_branch(attn, x, q, k, v, weights, layout)
    np.testing.assert_allclose(np.array(actual), np.array(expected), rtol=2e-5, atol=2e-6)


def test_grouped_windows_reduce_attention_dispatches(monkeypatch):
    calls = []
    original = mx.fast.scaled_dot_product_attention
    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(mx.fast, "scaled_dot_product_attention", counted)
    layout = VDNLayout(40, 3, 37, 1, 1, 1, 0, 2)
    attn = SimpleNamespace(heads=2, head_dim=4, scale=0.5, out_proj=lambda x: x)
    x = mx.ones((1, 40, 8))
    q = mx.ones((1, 2, 40, 4))
    mx.eval(_softmax_branch(attn, x, q, q, q, _small_weights(), layout))
    assert len(calls) == 11  # Prefix, two anchors, eight distinct interior windows.


def test_metal_solver_preserves_full_bidirectional_linear_branch():
    from minimax_h3_mlx.vdn_metal import VDNMatrixSolver

    mx.random.seed(31)
    layout = VDNLayout(77, 3, 37, 2, 1, 2, 0, 2)
    x = mx.random.normal((1, 77, 8)) * 0.2
    raw = tuple(mx.random.normal((1, 77, 2, 128)) for _ in range(3))
    weights = _small_weights(2, 128, 8)
    expected = _linear_branch(x, raw, weights, layout)
    solver = VDNMatrixSolver()
    actual = _linear_branch(x, raw, weights, layout, solver=solver)
    np.testing.assert_allclose(np.array(actual), np.array(expected), rtol=2e-4, atol=2e-5)
    assert solver.report()["metal_calls"] == 2
    assert solver.report()["cpu_calls"] == 0


def test_vdn_linear_branch_matches_independent_numpy_recurrence():
    rng = np.random.default_rng(27)
    layout = VDNLayout(21, 3, 18, 1, 1, 1, 0, 2)
    x = rng.normal(0, 0.5, (1, 21, 8)).astype(np.float32)
    raw = [rng.normal(0, 0.5, (1, 21, 2, 4)).astype(np.float32) for _ in range(3)]
    weights = _small_weights()
    w = {name: np.array(value) for name, value in weights.items()}

    def sigmoid(value):
        return 1 / (1 + np.exp(-value))

    def features(value, normalize):
        value = value * sigmoid(value)
        return (
            value / np.maximum(np.linalg.norm(value, axis=-1, keepdims=True), 1e-6)
            if normalize
            else value
        )

    def statistics(key, value, beta):
        return (
            np.einsum("shk,sh,shj->hkj", key, beta, key),
            np.einsum("shv,sh,shk->hvk", value, beta, key),
        )

    q, k, v = [features(value[0], index != 2) for index, value in enumerate(raw)]
    a, b = statistics(k[:2], v[:2], sigmoid(x[0, :2] @ w["linear_attention.beta_proj.weight"].T))
    text = 0.5 * (b @ np.linalg.inv(np.eye(4) + a))
    xv = x[0, 4:20]
    delta = xv @ w["linear_attention.alpha.down.weight"].T
    delta = delta @ w["linear_attention.alpha.up.weight"].T + w["linear_attention.alpha.dt_bias"]
    alpha = np.exp(
        -np.exp(w["linear_attention.alpha.A_log"])[None, :, None]
        * np.logaddexp(delta.reshape(16, 2, 4), 0)
    )
    transitions, injections = [], []
    for frame in range(16):
        row = slice(4 + frame, 5 + frame)
        beta = sigmoid(x[0, row] @ w["linear_attention.beta_proj.weight"].T)
        a, b = statistics(k[row], v[row], beta)
        inverse = np.linalg.inv(np.eye(4) + a)
        transitions.append(alpha[frame, :, :, None] * inverse)
        injections.append(b @ inverse)
    prefix, suffix = [], [None] * 16
    state = text.copy()
    for frame in range(16):
        state = state @ transitions[frame] + injections[frame]
        prefix.append(state)
    state = text.copy()
    for frame in reversed(range(16)):
        state = state @ transitions[frame] + injections[frame]
        suffix[frame] = state
    readout = []
    for frame in range(16):
        lo = (((frame + 1) // 5) - 1) * 5 - 1
        hi = (((frame + 1) // 5) + 2) * 5 - 2
        left = text if lo <= 0 else prefix[lo - 1]
        right = text if hi >= 15 else suffix[hi + 1]
        left = left * np.prod(alpha[max(lo, 0) : frame + 1], axis=0)[:, None, :]
        right = right * np.prod(alpha[frame : min(hi + 1, 16)], axis=0)[:, None, :]
        readout.append(np.einsum("hvk,hk->hv", left + right, q[4 + frame]))
    readout = np.asarray(readout)
    readout = readout / np.sqrt(np.mean(readout**2, axis=-1, keepdims=True) + 1e-6)
    gate = sigmoid(
        (xv @ w["linear_attention.output_gate.down.weight"].T)
        @ w["linear_attention.output_gate.up.weight"].T
        + w["linear_attention.output_gate.up.bias"]
    ).reshape(16, 2, 4)
    expected = (readout * gate).reshape(16, 8) @ w["to_out_linear.weight"].T
    actual = np.array(
        _linear_branch(mx.array(x), tuple(mx.array(value) for value in raw), weights, layout)
    )
    np.testing.assert_allclose(actual[1:-1], expected, rtol=3e-5, atol=2e-6)
    np.testing.assert_array_equal(actual[[0, -1]], 0)
