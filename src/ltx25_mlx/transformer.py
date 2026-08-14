"""LTX 2.5 transformer construction and direct MLX checkpoint loading."""

from __future__ import annotations

import copy
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from types import MethodType
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from safetensors import safe_open


class _BiasFreeFeedForward(nn.Module):
    """LTX 2.5 video GELU feed-forward network without projection biases."""

    def __init__(self, dim: int, *, mult: float = 4.0) -> None:
        super().__init__()
        inner_dim = int(dim * mult)
        self.proj_in = nn.Linear(dim, inner_dim, bias=False)
        self.proj_out = nn.Linear(inner_dim, dim, bias=False)

    def __call__(self, value: mx.array) -> mx.array:
        return self.proj_out(nn.gelu_approx(self.proj_in(value)))


@dataclass(frozen=True)
class LTX25TransformerConfig:
    """Checkpoint-driven LTX 2.5 architecture fields used by the MLX port."""

    num_layers: int = 48
    video_dim: int = 4096
    audio_dim: int = 2048
    video_num_heads: int = 32
    audio_num_heads: int = 32
    video_head_dim: int = 128
    audio_head_dim: int = 64
    video_patch_channels: int = 128
    audio_patch_channels: int = 128
    timestep_scale_multiplier: float = 1000.0
    av_ca_timestep_scale_multiplier: float = 1.0
    rope_theta: float = 10000.0
    rope_type: str = "split"
    positional_embedding_max_pos: tuple[int, ...] = (20, 2048, 2048)
    audio_positional_embedding_max_pos: tuple[int, ...] = (20,)
    norm_eps: float = 1e-6
    cross_attention_adaln: bool = True
    use_prompt_adaln_single: bool = True
    ff_bias: bool = False
    audio_ff_bias: bool = True
    caption_proj_before_connector: bool = True
    use_keyframes_abs_pos_embedding: bool = True
    frequencies_precision: str = "float64"

    @classmethod
    def from_metadata(cls, metadata: dict[str, Any]) -> LTX25TransformerConfig:
        config = metadata.get("config", metadata)
        transformer = config.get("transformer", config) if isinstance(config, dict) else {}
        if not isinstance(transformer, dict):
            raise ValueError("LTX 2.5 transformer metadata has no transformer configuration.")
        defaults = cls()
        result = cls(
            num_layers=int(transformer.get("num_layers", defaults.num_layers)),
            video_dim=int(transformer.get("cross_attention_dim", defaults.video_dim)),
            audio_dim=int(transformer.get("audio_cross_attention_dim", defaults.audio_dim)),
            video_num_heads=int(transformer.get("num_attention_heads", defaults.video_num_heads)),
            audio_num_heads=int(
                transformer.get("audio_num_attention_heads", defaults.audio_num_heads)
            ),
            video_head_dim=int(transformer.get("attention_head_dim", defaults.video_head_dim)),
            audio_head_dim=int(
                transformer.get("audio_attention_head_dim", defaults.audio_head_dim)
            ),
            video_patch_channels=int(transformer.get("in_channels", defaults.video_patch_channels)),
            audio_patch_channels=int(
                transformer.get("audio_in_channels", defaults.audio_patch_channels)
            ),
            timestep_scale_multiplier=float(
                transformer.get("timestep_scale_multiplier", defaults.timestep_scale_multiplier)
            ),
            av_ca_timestep_scale_multiplier=float(
                transformer.get(
                    "av_ca_timestep_scale_multiplier",
                    defaults.av_ca_timestep_scale_multiplier,
                )
            ),
            rope_theta=float(transformer.get("positional_embedding_theta", defaults.rope_theta)),
            rope_type=str(transformer.get("rope_type", defaults.rope_type)),
            positional_embedding_max_pos=tuple(
                transformer.get(
                    "positional_embedding_max_pos", defaults.positional_embedding_max_pos
                )
            ),
            audio_positional_embedding_max_pos=tuple(
                transformer.get(
                    "audio_positional_embedding_max_pos",
                    defaults.audio_positional_embedding_max_pos,
                )
            ),
            norm_eps=float(transformer.get("norm_eps", defaults.norm_eps)),
            cross_attention_adaln=bool(
                transformer.get("cross_attention_adaln", defaults.cross_attention_adaln)
            ),
            use_prompt_adaln_single=bool(
                transformer.get("use_prompt_adaln_single", defaults.use_prompt_adaln_single)
            ),
            ff_bias=bool(transformer.get("ff_bias", defaults.ff_bias)),
            audio_ff_bias=bool(transformer.get("audio_ff_bias", defaults.audio_ff_bias)),
            caption_proj_before_connector=bool(
                transformer.get(
                    "caption_proj_before_connector",
                    defaults.caption_proj_before_connector,
                )
            ),
            use_keyframes_abs_pos_embedding=bool(
                transformer.get(
                    "use_keyframes_abs_pos_embedding",
                    defaults.use_keyframes_abs_pos_embedding,
                )
            ),
            frequencies_precision=str(
                transformer.get("frequencies_precision", defaults.frequencies_precision)
            ),
        )
        result.validate()
        return result

    def validate(self) -> None:
        incompatible = []
        if not self.cross_attention_adaln:
            incompatible.append("cross_attention_adaln=true")
        if not self.use_prompt_adaln_single:
            incompatible.append("use_prompt_adaln_single=true")
        if self.ff_bias:
            incompatible.append("ff_bias=false")
        if not self.audio_ff_bias:
            incompatible.append("audio_ff_bias=true")
        if not self.caption_proj_before_connector:
            incompatible.append("caption_proj_before_connector=true")
        if not self.use_keyframes_abs_pos_embedding:
            incompatible.append("use_keyframes_abs_pos_embedding=true")
        if incompatible:
            raise ValueError(
                "The selected checkpoint does not declare the supported LTX 2.5 "
                "transformer architecture: " + ", ".join(incompatible)
            )

    def base_config(self):
        from ltx_core_mlx.model.transformer.model import LTXModelConfig

        return LTXModelConfig(
            num_layers=self.num_layers,
            video_dim=self.video_dim,
            audio_dim=self.audio_dim,
            video_num_heads=self.video_num_heads,
            audio_num_heads=self.audio_num_heads,
            video_head_dim=self.video_head_dim,
            audio_head_dim=self.audio_head_dim,
            av_cross_num_heads=self.audio_num_heads,
            av_cross_head_dim=self.audio_head_dim,
            video_patch_channels=self.video_patch_channels,
            audio_patch_channels=self.audio_patch_channels,
            timestep_scale_multiplier=self.timestep_scale_multiplier,
            av_ca_timestep_scale_multiplier=self.av_ca_timestep_scale_multiplier,
            rope_theta=self.rope_theta,
            rope_type=self.rope_type,
            positional_embedding_max_pos=self.positional_embedding_max_pos,
            audio_positional_embedding_max_pos=self.audio_positional_embedding_max_pos,
            norm_eps=self.norm_eps,
        )


class LTX25Model:
    """Construct the LTX 2.5 DiT using the established MLX LTX kernels."""

    @staticmethod
    def build(config: LTX25TransformerConfig):
        from ltx_core_mlx.model.transformer.model import LTXModel

        config.validate()
        model = LTXModel(config.base_config())
        # The released 2.5 checkpoint removes video FF biases only. Audio FF and
        # both prompt AdaLN modules retain their trained bias tensors.
        for block in model.transformer_blocks:
            block.ff = _BiasFreeFeedForward(config.video_dim)
        model.keyframes_abs_pos_embedding = mx.zeros((1, config.video_dim))
        model.ltx25_config = config
        model._compute_rope_freqs = MethodType(_compute_rope_freqs_float64, model)
        return model


def precompute_rope_freqs_float64(
    positions: mx.array,
    *,
    inner_dim: int,
    num_heads: int,
    theta: float,
    max_pos: list[int],
    rope_type: str = "split",
):
    """Build the released 2.5 frequency grid in NumPy float64, then use MLX."""
    from ltx_core_mlx.model.transformer.rope import compute_freqs

    num_pos_dims = positions.shape[-1]
    count = inner_dim // (2 * num_pos_dims)
    powers = np.linspace(0.0, 1.0, count, dtype=np.float64)
    indices = np.power(theta, powers) * (math.pi / 2.0)
    freqs = compute_freqs(mx.array(indices.astype(np.float32)), positions, max_pos)
    batch, tokens, frequency_count = freqs.shape
    if rope_type == "interleaved":
        cos_f = mx.repeat(mx.cos(freqs), 2, axis=-1)
        sin_f = mx.repeat(mx.sin(freqs), 2, axis=-1)
        padding = inner_dim - cos_f.shape[-1]
        if padding > 0:
            cos_f = mx.concatenate([mx.ones((*cos_f.shape[:-1], padding)), cos_f], axis=-1)
            sin_f = mx.concatenate([mx.zeros((*sin_f.shape[:-1], padding)), sin_f], axis=-1)
        head_dim = inner_dim // num_heads
        return (
            cos_f.reshape(batch, tokens, num_heads, head_dim).transpose(0, 2, 1, 3),
            sin_f.reshape(batch, tokens, num_heads, head_dim).transpose(0, 2, 1, 3),
            rope_type,
        )
    expected = inner_dim // 2
    padding = expected - frequency_count
    if padding > 0:
        freqs = mx.concatenate([mx.zeros((*freqs.shape[:-1], padding)), freqs], axis=-1)
    head_dim_half = inner_dim // (2 * num_heads)
    return (
        mx.cos(freqs).reshape(batch, tokens, num_heads, head_dim_half).transpose(0, 2, 1, 3),
        mx.sin(freqs).reshape(batch, tokens, num_heads, head_dim_half).transpose(0, 2, 1, 3),
        rope_type,
    )


def _compute_rope_freqs_float64(
    self,
    positions: mx.array,
    num_heads: int,
    head_dim: int,
    max_pos_override: list[int] | None = None,
):
    max_pos = (
        max_pos_override
        if max_pos_override is not None
        else list(self.config.positional_embedding_max_pos[: positions.shape[-1]])
    )
    return precompute_rope_freqs_float64(
        positions,
        inner_dim=num_heads * head_dim,
        num_heads=num_heads,
        theta=self.config.rope_theta,
        max_pos=max_pos,
        rope_type=self.config.rope_type,
    )


def transformer_metadata(path: str | Path) -> dict[str, Any]:
    """Return decoded safetensors metadata without materializing tensors."""
    source = Path(path).expanduser()
    if source.is_dir():
        from .paged_checkpoint import LTX25PagedManifest

        manifest = LTX25PagedManifest.load(source)
        if manifest.kind != "transformer":
            raise ValueError(f"Expected a paged transformer, got {manifest.kind!r}: {source}")
        return manifest.metadata
    with safe_open(source, framework="numpy") as handle:
        raw = handle.metadata() or {}
    decoded: dict[str, Any] = {}
    for key, value in raw.items():
        try:
            decoded[key] = json.loads(value)
        except (json.JSONDecodeError, TypeError):
            decoded[key] = value
    return decoded


def inspect_ltx25_ic_lora(path: str | Path) -> dict[str, Any]:
    """Validate an LTX 2.5 IC-LoRA header without materializing its tensors."""
    source = Path(path).expanduser()
    if not source.is_file() or source.suffix != ".safetensors":
        raise FileNotFoundError(f"LTX 2.5 IC-LoRA is not a safetensors file: {source}")
    with safe_open(source, framework="numpy") as handle:
        metadata = handle.metadata() or {}
        keys = list(handle.keys())
    model_version = str(metadata.get("model_version", ""))
    if not model_version.startswith("2.5"):
        raise ValueError(
            f"The selected IC-LoRA is not identified as LTX 2.5; model_version={model_version!r}."
        )
    try:
        downscale = int(metadata.get("reference_downscale_factor", "1"))
    except (TypeError, ValueError) as exc:
        raise ValueError("LTX 2.5 IC-LoRA has an invalid reference downscale factor.") from exc
    pairs = sum(key.endswith(".lora_A.weight") for key in keys)
    if pairs == 0 or pairs != sum(key.endswith(".lora_B.weight") for key in keys):
        raise ValueError("LTX 2.5 IC-LoRA does not contain balanced A/B adapter pairs.")
    return {
        "path": source,
        "model_version": model_version,
        "reference_downscale_factor": downscale,
        "adapter_pairs": pairs,
        "bytes": source.stat().st_size,
    }


def remap_comfy_transformer_key(key: str) -> str | None:
    """Map an official ComfyUI LTX transformer key to the MLX module tree.

    The official checkpoint stores the text connectors beside the DiT. They
    are excluded here and loaded with the Gemma feature extractor so staged
    unloading can release the language model before sampling.
    """
    for prefix in ("model.diffusion_model.", "diffusion_model.", "transformer."):
        if key.startswith(prefix):
            key = key.removeprefix(prefix)
            break
    if key.startswith(("video_embeddings_connector.", "audio_embeddings_connector.")):
        return None
    replacements = (
        (".to_out.0.", ".to_out."),
        (".ff.net.0.proj.", ".ff.proj_in."),
        (".ff.net.2.", ".ff.proj_out."),
        (".audio_ff.net.0.proj.", ".audio_ff.proj_in."),
        (".audio_ff.net.2.", ".audio_ff.proj_out."),
        (".linear_1.", ".linear1."),
        (".linear_2.", ".linear2."),
    )
    for source, target in replacements:
        key = key.replace(source, target)
    return key


def remap_comfy_transformer_weights(weights: dict[str, mx.array]) -> dict[str, mx.array]:
    """Remap and split official ComfyUI transformer tensors without copying."""
    mapped: dict[str, mx.array] = {}
    for key, value in weights.items():
        mapped_key = remap_comfy_transformer_key(key)
        if mapped_key is not None:
            mapped[mapped_key] = value
    return mapped


def _remap_comfy_lora_weights(weights: dict[str, mx.array]) -> dict[str, mx.array]:
    mapped: dict[str, mx.array] = {}
    for key, value in weights.items():
        mapped_key = remap_comfy_transformer_key(key)
        if mapped_key is not None:
            mapped[mapped_key] = value
    return mapped


def _load_resident_transformer_with_loras(
    model,
    weights: dict[str, mx.array],
    loras: tuple[tuple[Path, float], ...],
) -> None:
    """Fuse adapters one block at a time to bound temporary unified memory."""
    from ltx_core_mlx.loader.fuse_loras import apply_loras
    from ltx_core_mlx.loader.primitives import LoraStateDictWithStrength, StateDict

    loaded_loras = []
    for path, strength in loras:
        remapped = _remap_comfy_lora_weights(dict(mx.load(str(path))))
        loaded_loras.append((remapped, float(strength)))

    non_block = [
        (key, value) for key, value in weights.items() if not key.startswith("transformer_blocks.")
    ]
    model.load_weights(non_block, strict=False)

    for index, block in enumerate(model.transformer_blocks):
        prefix = f"transformer_blocks.{index}."
        block_weights = {
            key.removeprefix(prefix): value
            for key, value in weights.items()
            if key.startswith(prefix)
        }
        if not block_weights:
            raise ValueError(f"LTX 2.5 transformer block {index} has no checkpoint weights.")
        block_loras = []
        for remapped, strength in loaded_loras:
            values = {
                key.removeprefix(prefix): value
                for key, value in remapped.items()
                if key.startswith(prefix)
            }
            if values:
                block_loras.append(
                    LoraStateDictWithStrength(
                        StateDict(sd=values, size=0, dtype=set()),
                        strength,
                    )
                )
        fused = apply_loras(
            StateDict(sd=block_weights, size=0, dtype=set()),
            block_loras,
        )
        block.load_weights(list(fused.sd.items()), strict=True)
        mx.eval(block.parameters())

    loaded_loras.clear()
    mx.eval(model.parameters())


class _OfficialComfyBlockStreamer:
    """Adapt official Comfy block names to the existing MLX streaming API."""

    def __new__(cls, path, *, paged_manifest=None):
        from ltx_core_mlx.loader.block_streaming import BlockStreamer

        prefix = "model.diffusion_model.transformer_blocks."
        streamer = BlockStreamer(path, block_prefix=prefix)
        remapped: dict[int, list[tuple[str, str]]] = {}
        for index, entries in streamer._block_key_map.items():
            converted = []
            for full_key, _relative_key in entries:
                mapped = remap_comfy_transformer_key(full_key)
                block_prefix = f"transformer_blocks.{index}."
                if mapped is None or not mapped.startswith(block_prefix):
                    raise ValueError(f"Could not map streamed LTX 2.5 key: {full_key}")
                converted.append((full_key, mapped.removeprefix(block_prefix)))
            remapped[index] = converted
        streamer._block_key_map = remapped
        if paged_manifest is not None:
            return _PrefetchedBlockStreamer(streamer, paged_manifest)
        return streamer


class _PrefetchedBlockStreamer:
    """Add bounded read-ahead and measurements to the upstream block streamer."""

    def __init__(self, streamer, manifest, *, enabled: bool | None = None):
        from .page_prefetch import LTX25PagePrefetch

        self._streamer = streamer
        self._manifest = manifest
        self._prefetch = LTX25PagePrefetch(
            manifest.root,
            manifest.layers,
            enabled=LTX25PagePrefetch.default_enabled() if enabled is None else enabled,
            thread_name="ltx25-transformer-prefetch",
        )
        self.bind_calls = 0
        self.bind_seconds = 0.0
        self._prefetch.start(0)

    @property
    def block_count(self):
        return self._streamer.block_count

    @property
    def block_prefix(self):
        return self._streamer.block_prefix

    def block_keys(self, index):
        return self._streamer.block_keys(index)

    def bind(self, block, index, evict_previous=None, lora_sources=None):
        import time

        self._prefetch.wait(index)
        started = time.perf_counter()
        self._streamer.bind(
            block,
            index,
            evict_previous=evict_previous,
            lora_sources=lora_sources,
        )
        self.bind_calls += 1
        self.bind_seconds += time.perf_counter() - started
        next_index = (index + 1) % self._manifest.num_layers
        self._prefetch.start(next_index)

    def report(self):
        return {
            "streamed_bind_calls": self.bind_calls,
            "streamed_bind_seconds": self.bind_seconds,
            **self._prefetch.report(),
        }

    def close(self):
        self._prefetch.close()
        self._streamer.close()


_STREAMING_EVAL_LOCK = RLock()


class _StreamingEvalWindow:
    """Delay the upstream streaming barrier until a safe block window ends."""

    def __init__(self, window: int, eval_fn=mx.eval) -> None:
        if window < 1:
            raise ValueError("LTX 2.5 streaming window must be at least one block.")
        self.window = int(window)
        self.eval_fn = eval_fn
        self.calls = 0
        self.flushes = 0
        self._pending = None

    def __call__(self, *arrays):
        self.calls += 1
        self._pending = arrays
        if self.calls % self.window == 0:
            return self.flush()
        return None

    def flush(self):
        if self._pending is None:
            return None
        arrays = self._pending
        self._pending = None
        self.flushes += 1
        return self.eval_fn(*arrays)


def _streaming_window_from_environment(*, paged: bool) -> int:
    raw = os.environ.get("WEETODD_LTX25_STREAMING_WINDOW", "1").strip()
    try:
        window = int(raw)
    except ValueError as exc:
        raise ValueError(
            "WEETODD_LTX25_STREAMING_WINDOW must be the integer 1 or 2."
        ) from exc
    if window not in {1, 2}:
        raise ValueError("WEETODD_LTX25_STREAMING_WINDOW must be the integer 1 or 2.")
    if window > 1 and not paged:
        raise ValueError(
            "LTX 2.5 two-block streaming requires a paged transformer checkpoint."
        )
    return window


class _WindowedStreamingLTXModel(nn.Module):
    """Stream through multiple compiled block slots before one Metal barrier."""

    def __init__(self, model, streamer, *, window: int, lora_sources=None) -> None:
        super().__init__()
        if len(model.transformer_blocks) != window:
            raise ValueError("LTX 2.5 streaming block slots do not match the window size.")
        self.inner = model
        shared_blocks = tuple(model.transformer_blocks)
        compiled_blocks = tuple(mx.compile(block, inputs=block) for block in shared_blocks)
        object.__setattr__(self, "_streamer", streamer)
        object.__setattr__(self, "_shared_blocks", shared_blocks)
        object.__setattr__(self, "_compiled_blocks", compiled_blocks)
        object.__setattr__(self, "_lora_sources", lora_sources or [])
        object.__setattr__(self, "_window", int(window))
        object.__setattr__(self, "_eval_calls", 0)
        object.__setattr__(self, "_eval_flushes", 0)

    def __call__(self, *args, **kwargs):
        if kwargs.get("block_provider") is not None:
            return self.inner(*args, **kwargs)

        from ltx_core_mlx.model.transformer import model as model_module

        streamer = object.__getattribute__(self, "_streamer")
        shared_blocks = object.__getattribute__(self, "_shared_blocks")
        compiled_blocks = object.__getattribute__(self, "_compiled_blocks")
        lora_sources = object.__getattribute__(self, "_lora_sources")
        window = object.__getattribute__(self, "_window")
        previous = [None] * window
        use_compiled = kwargs.get("perturbations") is None

        def provider(index: int):
            slot = index % window
            streamer.bind(
                shared_blocks[slot],
                index,
                evict_previous=previous[slot],
                lora_sources=lora_sources or None,
            )
            previous[slot] = index
            return compiled_blocks[slot] if use_compiled else shared_blocks[slot]

        kwargs["block_provider"] = provider
        with _STREAMING_EVAL_LOCK:
            original_eval = model_module._mx_eval
            gate = _StreamingEvalWindow(window, original_eval)
            model_module._mx_eval = gate
            try:
                result = self.inner(*args, **kwargs)
            finally:
                try:
                    gate.flush()
                finally:
                    model_module._mx_eval = original_eval
                    object.__setattr__(self, "_eval_calls", gate.calls)
                    object.__setattr__(self, "_eval_flushes", gate.flushes)
            return result

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(super().__getattr__("inner"), name)

    def streaming_window_report(self) -> dict[str, int]:
        return {
            "streaming_window": object.__getattribute__(self, "_window"),
            "streaming_eval_calls": object.__getattribute__(self, "_eval_calls"),
            "streaming_eval_flushes": object.__getattribute__(self, "_eval_flushes"),
        }


def load_ltx25_transformer(
    path: str | Path,
    *,
    low_ram_streaming: bool = False,
    feed_forward_backend: str = "reference_fp32",
    loras: tuple[tuple[str | Path, float], ...] = (),
):
    """Strictly load an LTX 2.5 transformer, including MLX q8/q4 tensors."""
    from ltx_core_mlx.utils.memory import aggressive_cleanup
    from ltx_core_mlx.utils.weights import apply_quantization, load_split_safetensors

    source = Path(path).expanduser()
    paged_manifest = None
    if source.is_dir():
        from .paged_checkpoint import LTX25PagedManifest

        paged_manifest = LTX25PagedManifest.load(source)
        if paged_manifest.kind != "transformer":
            raise ValueError(f"Expected a paged transformer, got {paged_manifest.kind!r}.")
        if not low_ram_streaming:
            raise ValueError(
                "Paged LTX 2.5 transformer checkpoints require low_ram_streaming=true."
            )
    resolved_loras = tuple((Path(item).expanduser(), float(strength)) for item, strength in loras)
    for lora_path, _strength in resolved_loras:
        inspect_ltx25_ic_lora(lora_path)
    config = LTX25TransformerConfig.from_metadata(transformer_metadata(source))
    model = LTX25Model.build(config)
    if paged_manifest is None:
        raw_weights = load_split_safetensors(source)
        block_sources = source
    else:
        raw_weights = dict(mx.load(str(paged_manifest.fixed_path)))
        block_sources = list(paged_manifest.layer_paths)
    weights = remap_comfy_transformer_weights(raw_weights)
    if not weights:
        raise ValueError(f"LTX 2.5 transformer {source} has no recognized weights.")
    if low_ram_streaming:
        from ltx_core_mlx.loader.block_streaming import BlockLoraSource, StreamingLTXModel
        from ltx_core_mlx.loader.sd_ops import (
            LTXV_LORA_BLOCK_PREFIX,
            LTXV_LORA_COMFY_RENAMING_MAP,
        )

        streaming_window = _streaming_window_from_environment(paged=paged_manifest is not None)
        model.transformer_blocks = [model.transformer_blocks[0]]
        quantization_weights = weights
        if paged_manifest is not None:
            first_page = remap_comfy_transformer_weights(
                dict(mx.load(str(paged_manifest.layer_paths[0])))
            )
            quantization_weights = {**weights, **first_page}
        apply_quantization(model, quantization_weights)
        non_block = [
            (key, value)
            for key, value in weights.items()
            if not key.startswith("transformer_blocks.")
        ]
        model.load_weights(non_block, strict=False)
        if streaming_window > 1:
            first_block = model.transformer_blocks[0]
            model.transformer_blocks = [
                first_block,
                *(copy.deepcopy(first_block) for _ in range(streaming_window - 1)),
            ]
        lora_sources = [
            BlockLoraSource(
                lora_path,
                block_prefix=LTXV_LORA_BLOCK_PREFIX,
                strength=strength,
                sd_ops=LTXV_LORA_COMFY_RENAMING_MAP,
            )
            for lora_path, strength in resolved_loras
        ]
        streamer = _OfficialComfyBlockStreamer(block_sources, paged_manifest=paged_manifest)
        if streaming_window == 1:
            model = StreamingLTXModel(model, streamer, lora_sources=lora_sources)
        else:
            model = _WindowedStreamingLTXModel(
                model,
                streamer,
                window=streaming_window,
                lora_sources=lora_sources,
            )
        if paged_manifest is not None:
            object.__setattr__(
                model,
                "paged_checkpoint_report",
                {
                    "format": paged_manifest.format,
                    "bits": paged_manifest.bits,
                    "group_size": paged_manifest.group_size,
                    "fixed_bytes": paged_manifest.fixed.tensor_bytes,
                    "peak_layer_bytes": max(
                        record.tensor_bytes for record in paged_manifest.layers
                    ),
                    "streaming_window": streaming_window,
                },
            )
            object.__setattr__(
                model,
                "_weetodd_paged_streamer",
                object.__getattribute__(model, "_streamer"),
            )
        mx.eval(model.parameters())
        aggressive_cleanup()
        return model
    apply_quantization(model, weights)
    if resolved_loras:
        _load_resident_transformer_with_loras(model, weights, resolved_loras)
    else:
        model.load_weights(list(weights.items()), strict=True)
        mx.eval(model.parameters())
    from .feed_forward import configure_feed_forward_backend

    report = configure_feed_forward_backend(model, feed_forward_backend).to_dict()
    object.__setattr__(model, "feed_forward_backend_report", report)
    aggressive_cleanup()
    return model


__all__ = [
    "LTX25Model",
    "LTX25TransformerConfig",
    "inspect_ltx25_ic_lora",
    "load_ltx25_transformer",
    "remap_comfy_transformer_key",
    "remap_comfy_transformer_weights",
    "precompute_rope_freqs_float64",
    "transformer_metadata",
]
