"""MLX runtime for the MiniMax-H3 Fun ControlNet-Union branch.

The released checkpoint contains one input projection and five ordinary H3 blocks.  The branch
runs alongside base layers 0, 10, 20, 30, and 40; its projected residual is suppressed on audio
rows and added to the main packed stream at the requested strength.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten

from .config import DiTConfig
from .dit import TransformerBlock, param_dtype
from .packing import patchify_video_latents


@dataclass(frozen=True)
class H3FunControlSpec:
    checkpoint: str
    strength: float = 1.0

    def validate(self) -> Path:
        path = Path(self.checkpoint).expanduser()
        if not path.is_file() or path.suffix.lower() != ".safetensors":
            raise FileNotFoundError(f"H3 Fun ControlNet checkpoint not found: {path}")
        if not 0.0 <= self.strength <= 1.0:
            raise ValueError("H3 Fun ControlNet strength must be between 0 and 1")
        return path


@dataclass(frozen=True)
class H3FunControlCondition:
    model: MiniMaxH3FunControl
    latent: mx.array
    strength: float


class ControlTransformerBlock(TransformerBlock):
    def __init__(self, config: DiTConfig, *, first: bool):
        super().__init__(config)
        if first:
            self.before_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.after_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=True)


class MiniMaxH3FunControl(nn.Module):
    """The five-block Union control branch, kept independent of the base transformer weights."""

    def __init__(
        self,
        config: DiTConfig,
        *,
        control_in_dim: int = 49,
        injection_layers: tuple[int, ...] = (0, 10, 20, 30, 40),
    ):
        super().__init__()
        self.config = config
        self.control_in_dim = int(control_in_dim)
        self.injection_layers = tuple(int(value) for value in injection_layers)
        patch_volume = config.patch_size[0] * config.patch_size[1] * config.patch_size[2]
        self.control_proj_in = nn.Linear(
            self.control_in_dim * patch_volume, config.hidden_size, bias=True
        )
        self.control_blocks = [
            ControlTransformerBlock(config, first=index == 0)
            for index in range(len(self.injection_layers))
        ]
        self._modulation_cache: tuple[tuple[mx.array, ...], ...] | None = None
        self._modulation_key: tuple[float, ...] | None = None

    def validate_base(self, base) -> None:
        if self.config.hidden_size != base.config.hidden_size:
            raise ValueError("H3 Fun ControlNet hidden width does not match the base transformer")
        if self.config.time_embed_dim != base.config.time_embed_dim:
            raise ValueError(
                "H3 Fun ControlNet AdaLN width does not match the base transformer; "
                "convert both checkpoints to the same AdaLN form"
            )
        if self.config.patch_size != base.config.patch_size:
            raise ValueError("H3 Fun ControlNet patch size does not match the base transformer")
        if not self.injection_layers or self.injection_layers[-1] >= base.config.num_layers:
            raise ValueError("H3 Fun ControlNet injection layers exceed the base transformer")

    def set_chunk_sizes(
        self,
        *,
        query_rows: int | None,
        attention_heads: int | None,
        ffn_rows: int | None,
    ) -> None:
        for block in self.control_blocks:
            block.attn.query_chunk_size = query_rows
            block.attn.head_chunk_size = attention_heads
            block.mlp.row_chunk_size = ffn_rows

    def prepare_modulation(self, temb: mx.array, key: tuple[float, ...]) -> None:
        if self._modulation_cache is not None and self._modulation_key == key:
            return
        tables = tuple(block.adaln_proj(temb) for block in self.control_blocks)
        mx.eval(tables)
        self._modulation_cache = tables
        self._modulation_key = key

    def init_stream(
        self,
        hidden: mx.array,
        control_latent: mx.array,
        video_indices: mx.array,
        target_video_row_start: int,
    ) -> mx.array:
        if control_latent.ndim != 5 or int(control_latent.shape[0]) != int(hidden.shape[0]):
            raise ValueError("H3 Fun control latent must have shape (batch, channels, T, H, W)")
        rows = patchify_video_latents(control_latent.astype(mx.float32), self.config.patch_size)
        target_indices = video_indices[target_video_row_start:]
        if int(rows.shape[0]) != int(target_indices.shape[0]):
            raise ValueError(
                "H3 Fun control latent geometry does not match the target video rows: "
                f"{rows.shape[0]} control rows versus {target_indices.shape[0]} target rows"
            )
        patch_dim = int(self.control_proj_in.weight.shape[1])
        if int(rows.shape[1]) > patch_dim:
            raise ValueError("H3 Fun control latent has more channels than the checkpoint accepts")
        if int(rows.shape[1]) < patch_dim:
            rows = mx.pad(rows, ((0, 0), (0, patch_dim - int(rows.shape[1]))))
        projected = self.control_proj_in(rows.astype(param_dtype(self.control_proj_in)))
        projected = projected.reshape(hidden.shape[0], -1, hidden.shape[-1]).astype(hidden.dtype)
        control = hidden.at[:, target_indices].add(projected - hidden[:, target_indices])
        return self.control_blocks[0].before_proj(
            control.astype(param_dtype(self.control_blocks[0].before_proj))
        ).astype(hidden.dtype) + hidden

    def step(
        self,
        branch_index: int,
        control: mx.array,
        adaln_indices: mx.array,
        rotary: tuple[mx.array, mx.array],
        audio_indices: mx.array,
        mask: mx.array | None,
    ) -> tuple[mx.array, mx.array]:
        if self._modulation_cache is None:
            raise RuntimeError("H3 Fun ControlNet modulation was not prepared")
        block = self.control_blocks[branch_index]
        control = block(
            control,
            self._modulation_cache[branch_index],
            adaln_indices,
            rotary,
            mask,
            block_index=branch_index,
        )
        skip = block.after_proj(control.astype(param_dtype(block.after_proj))).astype(control.dtype)
        if int(audio_indices.shape[0]):
            skip = skip.at[:, audio_indices].multiply(0.0)
        return control, skip

    def release(self) -> None:
        self._modulation_cache = None
        self._modulation_key = None


def _strip_prefix(key: str) -> str:
    for prefix in ("model.diffusion_model.", "diffusion_model.", "controlnet."):
        if key.startswith(prefix):
            return key[len(prefix) :]
    return key


def _direct_name(key: str) -> str:
    return (
        _strip_prefix(key)
        .replace(".attn.norm_q.", ".attn.q_norm.")
        .replace(".attn.norm_k.", ".attn.k_norm.")
        .replace(".attn.to_out.0.", ".attn.out_proj.")
        .replace(".ff.net.0.proj.", ".mlp.fc1.")
        .replace(".ff.net.2.", ".mlp.fc2.")
    )


def _fuse_qkv(q: mx.array, k: mx.array, v: mx.array, heads: int, head_dim: int):
    if q.shape != k.shape or q.shape != v.shape:
        raise ValueError(f"H3 Fun ControlNet Q/K/V shapes differ: {q.shape}, {k.shape}, {v.shape}")
    if int(q.shape[0]) != heads * head_dim:
        raise ValueError("H3 Fun ControlNet Q projection has an invalid head layout")
    stacked = mx.stack((q, k, v), axis=1).reshape(heads, head_dim, 3, q.shape[1])
    return stacked.transpose(0, 2, 1, 3).reshape(3 * heads * head_dim, q.shape[1])


def _convert_weights(values: dict[str, mx.array], heads: int, head_dim: int):
    source = {_strip_prefix(key): value for key, value in values.items()}
    converted: dict[str, mx.array] = {}
    for key, value in source.items():
        if key.endswith((".attn.to_q.weight", ".attn.to_k.weight", ".attn.to_v.weight")):
            continue
        target = _direct_name(key)
        if target.endswith(".mlp.fc1.weight") and ".ff.net.0.proj." in key:
            half = int(value.shape[0]) // 2
            value = mx.concatenate((value[half:], value[:half]), axis=0)
        converted[target] = value
    for q_key in sorted(key for key in source if key.endswith(".attn.to_q.weight")):
        stem = q_key[: -len("to_q.weight")]
        k_key, v_key = stem + "to_k.weight", stem + "to_v.weight"
        if k_key not in source or v_key not in source:
            raise KeyError(f"Incomplete H3 Fun ControlNet QKV tensors at {stem}")
        converted[_direct_name(stem + "qkv_proj.weight")] = _fuse_qkv(
            source[q_key], source[k_key], source[v_key], heads, head_dim
        )
    return converted


def load_fun_controlnet(path: str | Path, *, strict: bool = True) -> MiniMaxH3FunControl:
    """Load either raw VideoX-Fun split projections or already-converted native keys."""
    path = Path(path).expanduser()
    values = {_strip_prefix(key): value for key, value in mx.load(str(path)).items()}
    proj = values.get("control_proj_in.weight")
    if proj is None:
        raise KeyError("H3 Fun ControlNet checkpoint has no control_proj_in.weight")
    block_indexes = sorted(
        {
            int(key.split(".", 2)[1])
            for key in values
            if key.startswith("control_blocks.") and key.split(".", 2)[1].isdigit()
        }
    )
    if block_indexes != list(range(len(block_indexes))) or not block_indexes:
        raise ValueError("H3 Fun ControlNet blocks must be contiguous from zero")
    head_dim = int(values["control_blocks.0.attn.norm_q.weight"].shape[0]) if (
        "control_blocks.0.attn.norm_q.weight" in values
    ) else int(values["control_blocks.0.attn.q_norm.weight"].shape[0])
    q_key = "control_blocks.0.attn.to_q.weight"
    qkv_key = "control_blocks.0.attn.qkv_proj.weight"
    heads = int(values[q_key].shape[0]) // head_dim if q_key in values else (
        int(values[qkv_key].shape[0]) // (3 * head_dim)
    )
    raw_fc1 = values.get("control_blocks.0.ff.net.0.proj.weight")
    native_fc1 = values.get("control_blocks.0.mlp.fc1.weight")
    fc1 = raw_fc1 if raw_fc1 is not None else native_fc1
    adaln = values["control_blocks.0.adaln_proj.linear.weight"]
    hidden = int(proj.shape[0])
    patch_volume = 4
    config = DiTConfig(
        hidden_size=hidden,
        num_layers=len(block_indexes),
        token_refiner_num_layers=0,
        num_attention_heads=heads,
        attention_head_dim=head_dim,
        ffn_hidden_size=int(fc1.shape[0]) // 2,
        time_embed_dim=int(adaln.shape[1]),
        adaln_out_features=int(adaln.shape[0]),
    )
    model = MiniMaxH3FunControl(
        config,
        control_in_dim=int(proj.shape[1]) // patch_volume,
        injection_layers=tuple(range(0, len(block_indexes) * 10, 10)),
    )
    converted = _convert_weights(values, heads, head_dim)
    expected = {key for key, _ in tree_flatten(model.parameters())}
    missing = sorted(expected - converted.keys())
    unexpected = sorted(converted.keys() - expected)
    if strict and (missing or unexpected):
        raise KeyError(
            f"H3 Fun ControlNet checkpoint mismatch: {len(missing)} missing {missing[:4]}, "
            f"{len(unexpected)} unexpected {unexpected[:4]}"
        )
    usable = [(key, value) for key, value in converted.items() if key in expected]
    model.update(tree_unflatten(usable))
    mx.eval(model.parameters())
    return model
