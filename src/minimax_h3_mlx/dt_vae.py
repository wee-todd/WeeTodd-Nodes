"""DT VAE tensor layouts adapted to the existing H3 audio/video decoders."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .dt_h3_checkpoint import fuse_qkv, unpair_rotary
from .dt_source import dt_source
from .dt_tensor_store import DTTensorStore


def _assign(model, values):
    import mlx.core as mx
    from mlx.utils import tree_flatten, tree_unflatten

    expected = {k: v.shape for k, v in tree_flatten(model.parameters())}
    seen = set()
    for key, value in values:
        if key in seen or key not in expected or tuple(value.shape) != expected[key]:
            raise ValueError(
                f"DT VAE tensor mapping mismatch at {key}: {value.shape}, "
                f"expected {expected.get(key)}"
            )
        array = mx.array(value).astype(mx.float32)
        mx.eval(array)
        model.update(tree_unflatten([(key, array)]))
        seen.add(key)
    if seen != expected.keys():
        raise ValueError(f"DT VAE missing tensors: {sorted(expected.keys() - seen)[:8]}")
    mx.eval(model.parameters())
    return model


def video_values(store):
    def read(role, index=0, parameter=0, decoder=True):
        section = "video_decoder" if decoder else "video_encoder"
        return store.read(f"__{section}__[t-{role}-{index}-{parameter}]")

    for source, target in [
        ("decoder_proj_in", "decoder.x_embedder"),
        ("decoder_proj_out", "decoder.proj_out"),
        ("decoder_norm_out", "decoder.norm_out"),
    ]:
        for param, suffix in ((0, "weight"), (1, "bias")):
            a = read(source, parameter=param)
            yield f"{target}.{suffix}", a.reshape(-1) if "norm_out" in target else a
    yield "decoder.register_tokens", read("register_tokens")
    for param, suffix in ((0, "weight"), (1, "bias")):
        a = read("post_quant_conv", parameter=param)
        yield f"post_quant_conv.{suffix}", a.reshape(24, 1, 1, 1, 24) if param == 0 else a
    for i in range(36):
        prefix = f"decoder.transformer_blocks.{i}."
        for role in ("norm1", "norm2"):
            yield prefix + role + ".weight", read(role, i).reshape(-1)
        for role in ("scale1", "scale2"):
            yield prefix + role, read(role, i).reshape(-1)
        for param, suffix in ((0, "weight"), (1, "bias")):
            q, k, v = (read("to_" + role, i, param) for role in ("q", "k", "v"))
            q = unpair_rotary(q, heads=32, head_dim=64, rotary_dim=48)
            k = unpair_rotary(k, heads=32, head_dim=64, rotary_dim=48)
            yield prefix + "attn.to_qkv." + suffix, fuse_qkv(q, k, v, heads=32, head_dim=64)
            yield prefix + "attn.to_out." + suffix, read("to_out", i, param)
            yield (
                prefix + "ff.w1." + suffix,
                np.concatenate([read("ff_gate", i, param), read("ff_up", i, param)]),
            )
            yield prefix + "ff.w2." + suffix, read("ff_down", i, param)

    def encoder_pair(source, target, index=0):
        for param, suffix in ((0, "weight"), (1, "bias")):
            a = read(source, index, param, decoder=False)
            if a.ndim == 5:
                a = a.transpose(0, 2, 3, 4, 1)
            elif "norm" in target:
                a = a.reshape(-1)
            yield target + "." + suffix, a

    for role in ("conv_in", "norm_out", "conv_out"):
        yield from encoder_pair(role, "encoder." + role)
    yield from encoder_pair("quant_conv", "quant_conv")
    shortcut = 0
    for level in range(6):
        for block in range(2):
            prefix = f"encoder.down.{level}.block.{block}."
            for role in ("norm1", "norm2", "conv1", "conv2"):
                yield from encoder_pair(role, prefix + role, level * 2 + block)
            if block == 0 and level in (1, 3, 5):
                yield from encoder_pair("nin_shortcut", prefix + "nin_shortcut", shortcut)
                shortcut += 1
        if level < 4:
            yield from encoder_pair(f"downsample_{level}", f"encoder.down.{level}.downsample.conv")


def load_dt_video_vae(directory):
    from .video_vae import VideoVAE, VideoVAEConfig

    wrapper = json.loads((Path(directory) / "config.json").read_text())
    config = VideoVAEConfig(
        latents_mean=tuple(wrapper["latents_mean"]), latents_std=tuple(wrapper["latents_std"])
    )
    with DTTensorStore(dt_source(directory, "video_vae")) as store:
        return _assign(VideoVAE(config), video_values(store))


def load_dt_audio_vae(directory):
    import mlx.core as mx
    import mlx.nn as nn

    from .audio_vae import AudioVAE, AudioVAEConfig

    wrapper = json.loads((Path(directory) / "config.json").read_text())
    config = AudioVAEConfig(
        latents_mean=tuple(wrapper["latents_mean"]), latents_std=tuple(wrapper["latents_std"])
    )

    class DirectSnake(nn.Module):
        def __init__(self, channels):
            super().__init__()
            self.alpha = mx.ones((channels,))
            self.beta = mx.ones((channels,))

        def __call__(self, x):
            return x + self.beta * mx.square(mx.sin(self.alpha * x))

    class DecodeOnlyAudioVAE(AudioVAE):
        def encode(self, *_):
            raise ValueError("DT direct audio encoder is not yet qualified; use generated audio.")

    model = DecodeOnlyAudioVAE(config)
    for name in ("encoder", "pre_block", "mean_proj", "logs_proj"):
        delattr(model, name)
    for block in model.decoder.resblocks:
        for activation in block.activations:
            activation.act = DirectSnake(activation.act.alpha.shape[0])
    model.decoder.activation_post.act = DirectSnake(8)
    with DTTensorStore(dt_source(directory, "audio_vae")) as store:

        def read(role, param=0):
            return store.read(f"__audio_decoder__[t-{role}-0-{param}]")

        def conv(source, target, transpose=False, bias=True):
            a = read(source)[:, :, 0, :]
            yield target + ".weight", a.transpose(1, 2, 0) if transpose else a.transpose(0, 2, 1)
            if bias:
                yield target + ".bias", read(source, 1)

        def activation(source, target):
            # DT stores exp(alpha) and reciprocal exp(beta) as execution-ready values.
            for name in ("alpha", "beta"):
                yield target + ".act." + name, read(source + "_snake_" + name).reshape(-1)

        def values():
            yield from conv("dec_in_proj", "dec_in_proj")
            yield from conv("audio_conv_pre", "decoder.conv_pre")
            yield from conv("audio_conv_post", "decoder.conv_post", bias=False)
            for i in range(7):
                yield from conv(f"audio_up_{i}", f"decoder.ups.{i}.0", transpose=True)
            for i in range(21):
                for j in range(3):
                    for side in (0, 1):
                        yield from conv(
                            f"audio_amp_{i}_c{j}_{side}",
                            f"decoder.resblocks.{i}.convs{side + 1}.{j}",
                        )
                        yield from activation(
                            f"audio_amp_{i}_a{j}_{side}",
                            f"decoder.resblocks.{i}.activations.{j * 2 + side}",
                        )
            yield from activation("audio_post", "decoder.activation_post")

        return _assign(model, values())
