from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten

from minimax_h3_mlx.config import DiTConfig
from minimax_h3_mlx.controlnet import MiniMaxH3FunControl, load_fun_controlnet
from wee_todd_nodes.h3_controlnet import (
    inspect_h3_fun_control_header,
    prepare_h3_control_frames,
)


def _config(num_layers=1):
    hidden = 8
    return DiTConfig(
        hidden_size=hidden,
        num_layers=num_layers,
        token_refiner_num_layers=0,
        num_attention_heads=1,
        attention_head_dim=8,
        ffn_hidden_size=12,
        latents_dim=2,
        audio_latents_dim=4,
        text_dim=6,
        time_embed_dim=6,
        time_embed_hidden_size=8,
        timestep_input_dim=4,
        adaln_out_features=6 * 3 * hidden,
        final_adaln_out_features=2 * hidden,
        rope_inv_freq_len=1,
    )


def test_control_branch_pads_pure_control_and_suppresses_audio_skip():
    model = MiniMaxH3FunControl(_config(), control_in_dim=3, injection_layers=(0,))
    temb = mx.zeros((1, 6), dtype=mx.float32)
    model.prepare_modulation(temb, (1.0,))
    hidden = mx.zeros((1, 5, 8), dtype=mx.bfloat16)
    control_latent = mx.zeros((1, 2, 1, 2, 4), dtype=mx.float32)
    video_indices = mx.array([2, 3], dtype=mx.int32)
    audio_indices = mx.array([4], dtype=mx.int32)
    control = model.init_stream(hidden, control_latent, video_indices, 0)
    rotary = (
        mx.ones((5, 6), dtype=mx.float32),
        mx.zeros((5, 6), dtype=mx.float32),
    )
    control, skip = model.step(
        0,
        control,
        mx.array([1, 1, 0, 0, 2], dtype=mx.int32),
        rotary,
        audio_indices,
        None,
    )
    mx.eval(control, skip)
    assert control.shape == hidden.shape
    assert skip.shape == hidden.shape
    assert mx.allclose(skip[:, audio_indices], mx.zeros((1, 1, 8)))


def _raw_checkpoint(model):
    raw = {}
    for key, value in tree_flatten(model.parameters()):
        if key.endswith(".attn.qkv_proj.weight"):
            base = key[: -len("qkv_proj.weight")]
            separated = value.reshape(1, 3, 8, value.shape[1])
            raw[base + "to_q.weight"] = separated[:, 0].reshape(8, value.shape[1])
            raw[base + "to_k.weight"] = separated[:, 1].reshape(8, value.shape[1])
            raw[base + "to_v.weight"] = separated[:, 2].reshape(8, value.shape[1])
            continue
        target = (
            key.replace(".attn.q_norm.", ".attn.norm_q.")
            .replace(".attn.k_norm.", ".attn.norm_k.")
            .replace(".attn.out_proj.", ".attn.to_out.0.")
            .replace(".mlp.fc1.", ".ff.net.0.proj.")
            .replace(".mlp.fc2.", ".ff.net.2.")
        )
        if target.endswith(".ff.net.0.proj.weight"):
            half = int(value.shape[0]) // 2
            value = mx.concatenate((value[half:], value[:half]), axis=0)
        raw[target] = value
    return raw


def test_loader_converts_raw_videox_fun_keys(tmp_path: Path):
    original = MiniMaxH3FunControl(_config(), control_in_dim=2, injection_layers=(0,))
    checkpoint = tmp_path / "fun.safetensors"
    mx.save_safetensors(str(checkpoint), _raw_checkpoint(original))

    loaded = load_fun_controlnet(checkpoint)

    expected = dict(tree_flatten(original.parameters()))
    actual = dict(tree_flatten(loaded.parameters()))
    assert expected.keys() == actual.keys()
    for key in expected:
        assert mx.allclose(expected[key], actual[key]), key


def test_header_preflight_accepts_comfy_prefixed_raw_checkpoint(tmp_path: Path):
    model = MiniMaxH3FunControl(
        _config(num_layers=5),
        control_in_dim=49,
        injection_layers=(0, 10, 20, 30, 40),
    )
    checkpoint = tmp_path / "fun-prefixed.safetensors"
    mx.save_safetensors(
        str(checkpoint),
        {f"model.diffusion_model.{key}": value for key, value in _raw_checkpoint(model).items()},
    )

    report = inspect_h3_fun_control_header(checkpoint)

    assert report["control_in_dim"] == 49
    assert report["injection_layers"] == [0, 10, 20, 30, 40]
    assert report["checkpoint_layout"] == "videox_fun_split_qkv"


def test_control_frame_preparation_scales_unit_floats_and_holds_last_frame():
    source = np.ones((1, 8, 16, 3), dtype=np.float32)
    prepared = prepare_h3_control_frames(source, target_frames=5, height=16, width=16)
    assert prepared.shape == (5, 16, 16, 3)
    assert prepared.dtype == np.uint8
    assert int(prepared.min()) == 255
