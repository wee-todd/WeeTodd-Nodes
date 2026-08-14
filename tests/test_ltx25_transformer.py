import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from mlx.utils import tree_flatten

from ltx25_mlx.transformer import (
    LTX25Model,
    LTX25TransformerConfig,
    _PrefetchedBlockStreamer,
    _streaming_window_from_environment,
    _StreamingEvalWindow,
    _WindowedStreamingLTXModel,
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


def test_prefetched_streamer_schedules_next_page_and_wraps(monkeypatch, tmp_path):
    from types import SimpleNamespace

    calls = []

    class FakePrefetch:
        @staticmethod
        def default_enabled():
            return True

        def __init__(self, *_args, **_kwargs):
            pass

        def start(self, index):
            calls.append(("start", index))

        def wait(self, index):
            calls.append(("wait", index))

        def report(self):
            return {"prefetch_hits": 2}

        def close(self):
            calls.append(("close", None))

    class FakeStreamer:
        block_count = 2
        block_prefix = "blocks."

        def block_keys(self, index):
            return [str(index)]

        def bind(self, _block, index, **_kwargs):
            calls.append(("bind", index))

        def close(self):
            calls.append(("streamer_close", None))

    monkeypatch.setattr("ltx25_mlx.page_prefetch.LTX25PagePrefetch", FakePrefetch)
    manifest = SimpleNamespace(root=tmp_path, layers=(object(), object()), num_layers=2)
    wrapped = _PrefetchedBlockStreamer(FakeStreamer(), manifest)
    wrapped.bind(object(), 0)
    wrapped.bind(object(), 1)
    report = wrapped.report()
    wrapped.close()

    assert calls[:7] == [
        ("start", 0),
        ("wait", 0),
        ("bind", 0),
        ("start", 1),
        ("wait", 1),
        ("bind", 1),
        ("start", 0),
    ]
    assert report["streamed_bind_calls"] == 2
    assert report["prefetch_hits"] == 2
    assert calls[-2:] == [("close", None), ("streamer_close", None)]


def test_streaming_eval_window_flushes_before_slot_reuse():
    evaluations = []

    def evaluate(*arrays):
        evaluations.append(tuple(int(value.item()) for value in arrays))

    gate = _StreamingEvalWindow(2, evaluate)
    for index in range(5):
        gate(mx.array(index))

    assert evaluations == [(1,), (3,)]
    gate.flush()
    assert evaluations == [(1,), (3,), (4,)]
    assert gate.calls == 5
    assert gate.flushes == 3


def test_streaming_window_environment_is_bounded_and_paged(monkeypatch):
    monkeypatch.delenv("WEETODD_LTX25_STREAMING_WINDOW", raising=False)
    assert _streaming_window_from_environment(paged=True) == 1

    monkeypatch.setenv("WEETODD_LTX25_STREAMING_WINDOW", "2")
    assert _streaming_window_from_environment(paged=True) == 2
    with pytest.raises(ValueError, match="paged transformer"):
        _streaming_window_from_environment(paged=False)

    monkeypatch.setenv("WEETODD_LTX25_STREAMING_WINDOW", "3")
    with pytest.raises(ValueError, match="integer 1 or 2"):
        _streaming_window_from_environment(paged=True)


def test_windowed_streaming_restores_global_eval_after_failure():
    from ltx_core_mlx.model.transformer import model as model_module

    class IdentityBlock(nn.Module):
        def __call__(self, value):
            return value

    class FailingModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.transformer_blocks = [IdentityBlock(), IdentityBlock()]

        def __call__(self, *_args, **_kwargs):
            model_module._mx_eval(mx.array(1))
            raise RuntimeError("probe failure")

    original_eval = model_module._mx_eval
    wrapped = _WindowedStreamingLTXModel(
        FailingModel(),
        object(),
        window=2,
    )
    with pytest.raises(RuntimeError, match="probe failure"):
        wrapped(mx.array(0))

    assert model_module._mx_eval is original_eval
    assert wrapped.streaming_window_report() == {
        "streaming_window": 2,
        "streaming_eval_calls": 1,
        "streaming_eval_flushes": 1,
    }


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
