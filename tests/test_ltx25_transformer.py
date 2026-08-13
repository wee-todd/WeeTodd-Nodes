import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from mlx.utils import tree_flatten

from ltx25_mlx.transformer import (
    LTX25Model,
    LTX25TransformerConfig,
    precompute_rope_freqs_float64,
    remap_comfy_transformer_key,
)


def _tiny_config(**overrides):
    values = {
        "num_layers": 1,
        "video_dim": 16,
        "audio_dim": 8,
        "video_num_heads": 2,
        "audio_num_heads": 2,
        "video_head_dim": 8,
        "audio_head_dim": 4,
        "video_patch_channels": 4,
        "audio_patch_channels": 4,
        "positional_embedding_max_pos": (20, 16, 16),
        "audio_positional_embedding_max_pos": (20,),
    }
    values.update(overrides)
    return LTX25TransformerConfig(**values)


def test_ltx25_transformer_keeps_prompt_and_audio_bias_but_removes_video_ff_bias():
    model = LTX25Model.build(_tiny_config())
    parameters = dict(tree_flatten(model.parameters()))

    assert any(key.startswith("prompt_adaln_single.") for key in parameters)
    assert any(key.startswith("audio_prompt_adaln_single.") for key in parameters)
    assert "transformer_blocks.0.ff.proj_in.bias" not in parameters
    assert "transformer_blocks.0.ff.proj_out.bias" not in parameters
    assert "transformer_blocks.0.audio_ff.proj_in.bias" in parameters
    assert "transformer_blocks.0.audio_ff.proj_out.bias" in parameters
    assert "keyframes_abs_pos_embedding" in parameters
    assert isinstance(model.transformer_blocks[0].ff.proj_in, nn.Linear)


def test_ltx25_transformer_config_reads_real_checkpoint_defaults():
    config = LTX25TransformerConfig.from_metadata(
        {
            "config": {
                "transformer": {
                    "num_layers": 2,
                    "cross_attention_dim": 32,
                    "audio_cross_attention_dim": 16,
                    "cross_attention_adaln": True,
                    "ff_bias": False,
                    "caption_proj_before_connector": True,
                    "use_keyframes_abs_pos_embedding": True,
                    "frequencies_precision": "float64",
                }
            }
        }
    )
    assert config.num_layers == 2
    assert config.video_dim == 32
    assert config.audio_dim == 16
    assert config.use_prompt_adaln_single is True
    assert config.audio_ff_bias is True
    assert config.frequencies_precision == "float64"

    with pytest.raises(ValueError, match="audio_ff_bias=true"):
        LTX25TransformerConfig.from_metadata(
            {
                "config": {
                    "transformer": {
                        "cross_attention_adaln": True,
                        "use_prompt_adaln_single": True,
                        "ff_bias": False,
                        "audio_ff_bias": False,
                        "caption_proj_before_connector": True,
                        "use_keyframes_abs_pos_embedding": True,
                    }
                }
            }
        )


def test_remaps_official_comfy_transformer_keys_and_splits_connectors():
    assert (
        remap_comfy_transformer_key(
            "model.diffusion_model.transformer_blocks.0.attn1.to_out.0.weight"
        )
        == "transformer_blocks.0.attn1.to_out.weight"
    )
    assert (
        remap_comfy_transformer_key(
            "model.diffusion_model.transformer_blocks.0.ff.net.0.proj.weight"
        )
        == "transformer_blocks.0.ff.proj_in.weight"
    )
    assert (
        remap_comfy_transformer_key(
            "model.diffusion_model.transformer_blocks.0.audio_ff.net.2.bias"
        )
        == "transformer_blocks.0.audio_ff.proj_out.bias"
    )
    assert (
        remap_comfy_transformer_key(
            "model.diffusion_model.adaln_single.emb.timestep_embedder.linear_1.weight"
        )
        == "adaln_single.emb.timestep_embedder.linear1.weight"
    )
    assert (
        remap_comfy_transformer_key(
            "model.diffusion_model.video_embeddings_connector.learnable_registers"
        )
        is None
    )


def test_official_streaming_key_map_matches_shared_block(tmp_path):
    import numpy as np
    from safetensors.numpy import save_file

    from ltx25_mlx.transformer import _OfficialComfyBlockStreamer

    path = tmp_path / "transformer.safetensors"
    save_file(
        {
            "model.diffusion_model.transformer_blocks.0.attn1.to_out.0.weight": np.zeros(
                (2, 2), dtype=np.float32
            ),
            "model.diffusion_model.transformer_blocks.0.ff.net.0.proj.weight": np.zeros(
                (2, 2), dtype=np.float32
            ),
        },
        path,
    )
    streamer = _OfficialComfyBlockStreamer(path)
    assert set(streamer.block_keys(0)) == {"attn1.to_out.weight", "ff.proj_in.weight"}
    streamer.close()


def test_ltx25_rope_uses_numpy_float64_grid_before_mlx():
    positions = mx.array([[[0.0], [1.0], [2.0]]])
    cos_f, sin_f, kind = precompute_rope_freqs_float64(
        positions,
        inner_dim=8,
        num_heads=1,
        theta=10000.0,
        max_pos=[20],
    )
    grid = np.power(10000.0, np.linspace(0.0, 1.0, 4, dtype=np.float64))
    grid = (grid * np.pi / 2.0).astype(np.float32)
    fractional = np.array([0.0, 1.0, 2.0], dtype=np.float32) / 20.0
    angles = grid[None, :] * (fractional[:, None] * 2.0 - 1.0)
    assert kind == "split"
    assert np.allclose(np.asarray(cos_f[0, 0]), np.cos(angles), rtol=1e-6, atol=1e-7)
    assert np.allclose(np.asarray(sin_f[0, 0]), np.sin(angles), rtol=1e-6, atol=1e-7)
