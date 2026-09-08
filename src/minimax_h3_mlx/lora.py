"""Activation-space Low-Rank Adaptation (LoRA) for MiniMax H3 MLX modules."""

from __future__ import annotations

import math
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from safetensors import safe_open

from wee_todd_mlx.adapter_contract import inspect_adapter_scaling


@dataclass(frozen=True)
class LoRARequest:
    """One lazy LoRA application request supplied by the host adapter."""

    path: str
    strength: float = 1.0
    adaln_input_grid: str | None = None
    qkv_layout: str = "native_interleaved"
    start_after_evaluations: int = 0


@dataclass(frozen=True)
class LoRAApplyReport:
    """Auditable result of applying one LoRA file to a loaded transformer."""

    path: str
    strength: float
    targets: int
    adaln_targets: int
    qkv_permuted_targets: int
    tensor_bytes: int
    start_after_evaluations: int


_LORA_EVALUATION: ContextVar[tuple[int, int] | None] = ContextVar(
    "minimax_h3_lora_evaluation",
    default=None,
)


@contextmanager
def lora_evaluation(index: int, total: int):
    """Select the LoRA activation state for one transformer evaluation."""
    if not 0 <= index < total:
        raise ValueError(f"LoRA evaluation index {index} is outside a {total}-evaluation run.")
    token = _LORA_EVALUATION.set((int(index), int(total)))
    try:
        yield
    finally:
        _LORA_EVALUATION.reset(token)


class _LoRAProjection(nn.Module):
    def __init__(
        self,
        a: mx.array,
        b: mx.array,
        scale: float,
        source_grid=None,
        start_after_evaluations: int = 0,
        output_slot: int | None = None,
        qkv_head_dim: int | None = None,
    ):
        super().__init__()
        self.a = a
        self.b = b
        self.scale = float(scale)
        self.source_grid = source_grid
        self.prepared_input = None
        self.start_after_evaluations = int(start_after_evaluations)
        self.output_slot = output_slot
        self.qkv_head_dim = qkv_head_dim

    def active(self) -> bool:
        evaluation = _LORA_EVALUATION.get()
        if evaluation is None:
            return True
        index, _ = evaluation
        return index >= self.start_after_evaluations

    def prepare(self, timesteps: mx.array) -> None:
        if self.source_grid is None:
            return
        grid = self.source_grid.astype(mx.float32)
        position = mx.clip(timesteps.astype(mx.float32), 0.0, 1.0) * (grid.shape[0] - 1)
        lower = mx.minimum(mx.floor(position).astype(mx.int32), grid.shape[0] - 2)
        fraction = (position - lower.astype(mx.float32))[:, None]
        self.prepared_input = grid[lower] * (1.0 - fraction) + grid[lower + 1] * fraction
        mx.eval(self.prepared_input)

    def delta(self, value: mx.array) -> mx.array:
        source = self.prepared_input if self.source_grid is not None else value
        if source is None:
            raise RuntimeError(
                "The pruned AdaLN LoRA input grid was not prepared for the sampling schedule."
            )
        hidden = source.astype(self.a.dtype) @ self.a.T
        return (hidden @ self.b.T) * self.scale

    def apply(self, output: mx.array, value: mx.array) -> mx.array:
        return self.apply_delta(output, self.delta(value))

    def apply_delta(self, output: mx.array, delta: mx.array) -> mx.array:
        delta = delta.astype(output.dtype)
        if self.output_slot is None:
            return output + delta
        if self.qkv_head_dim is None:
            raise RuntimeError("Split QKV LoRA adapter has no attention head dimension.")
        slot_width = int(delta.shape[-1])
        if slot_width % self.qkv_head_dim or int(output.shape[-1]) != 3 * slot_width:
            raise ValueError(
                "Split QKV LoRA output does not match the fused H3 projection layout."
            )
        heads = slot_width // self.qkv_head_dim
        fused = output.reshape(*output.shape[:-1], heads, 3, self.qkv_head_dim)
        update = delta.reshape(*delta.shape[:-1], heads, self.qkv_head_dim)
        fused = fused.at[..., self.output_slot, :].add(update)
        return fused.reshape(output.shape)


class LoRALinear(nn.Module):
    """Run a base linear layer and add one or more LoRA updates in activation space."""

    def __init__(self, base, adapters: list[_LoRAProjection]):
        super().__init__()
        self.base = base
        self.adapters = adapters
        self.batched_inputs = False
        self._input_groups = []

    def configure_batched_inputs(self, enabled: bool = True) -> None:
        """Batch compatible A/B projections without merging weights or rounded updates.

        Group only ordinary activation inputs with identical dtype/activation schedules.
        AdaLN grids retain their original preparation and execution path.
        """
        self.batched_inputs = bool(enabled)
        self._input_groups = []
        if not enabled:
            return
        groups = {}
        for index, adapter in enumerate(self.adapters):
            if adapter.source_grid is None:
                key = (adapter.a.dtype, adapter.b.dtype, adapter.a.shape, adapter.b.shape,
                       adapter.start_after_evaluations)
                groups.setdefault(key, []).append(index)
        for indices in groups.values():
            if len(indices) > 1:
                combined_a = mx.stack([self.adapters[i].a for i in indices])
                combined_b = mx.stack([self.adapters[i].b for i in indices])
                self._input_groups.append((tuple(indices), combined_a, combined_b))

    def __call__(self, value: mx.array) -> mx.array:
        output = self.base(value)
        if self.batched_inputs and self._input_groups:
            return self._batched(output, value)
        for adapter in self.adapters:
            if adapter.active():
                output = adapter.apply(output, value)
        return output

    def _batched(self, output: mx.array, value: mx.array) -> mx.array:
        deltas = {}
        for indices, combined_a, combined_b in self._input_groups:
            if not self.adapters[indices[0]].active():
                continue
            source = value.astype(combined_a.dtype).reshape(-1, value.shape[-1])
            hidden = source[None] @ combined_a.swapaxes(-1, -2)
            updates = hidden @ combined_b.swapaxes(-1, -2)
            for offset, index in enumerate(indices):
                adapter = self.adapters[index]
                deltas[index] = updates[offset].reshape(
                    *value.shape[:-1], adapter.b.shape[0]
                ) * adapter.scale
        active = [(i, a) for i, a in enumerate(self.adapters) if a.active()]
        # Split-QKV updates touch independent slots. Preserve per-slot addition order
        # and dtype rounding, but assemble the full tensor only once (no repeated scatter).
        if active and all(a.output_slot is not None for _, a in active):
            head_dim = active[0][1].qkv_head_dim
            if head_dim and all(a.qkv_head_dim == head_dim for _, a in active):
                heads = output.shape[-1] // (3 * head_dim)
                fused = output.reshape(*output.shape[:-1], heads, 3, head_dim)
                slots = [fused[..., slot, :] for slot in range(3)]
                for index, adapter in active:
                    delta = deltas[index] if index in deltas else adapter.delta(value)
                    delta = delta.astype(output.dtype).reshape(slots[0].shape)
                    slots[adapter.output_slot] = slots[adapter.output_slot] + delta
                return mx.stack(slots, axis=-2).reshape(output.shape)
        for index, adapter in active:
            delta = deltas[index] if index in deltas else adapter.delta(value)
            output = adapter.apply_delta(output, delta)
        return output

    def prepare(self, timesteps: mx.array) -> None:
        prepare_base = getattr(self.base, "prepare", None)
        if callable(prepare_base):
            prepare_base(timesteps)
        for adapter in self.adapters:
            adapter.prepare(timesteps)


def _canonical_target(name: str) -> str:
    for prefix in (
        "base_model.model.model.diffusion_model.",
        "base_model.model.diffusion_model.",
        "base_model.model.transformer.",
        "base_model.model.",
        "model.diffusion_model.",
        "diffusion_model.",
        "transformer.",
        "model.",
    ):
        if name.startswith(prefix):
            return name[len(prefix) :]
    return name


def _fastvideo_target(name: str) -> tuple[str, int | None, bool]:
    """Map FastVideo's Diffusers-layout H3 modules to the native fused MLX layout."""
    name = _canonical_target(name)
    prefix_replacements = (
        ("transformer_blocks.", "blocks."),
        ("token_refiner.refiner_blocks.", "token_refiner.blocks."),
        ("audio_proj_in", "audio_patch_proj"),
        ("audio_proj_out", "final_layer.audio_out"),
        ("context_embedder", "condition_proj"),
        ("norm_out.linear", "final_layer.adaln_proj.linear"),
        ("norm_out.norm", "final_layer.norm"),
        ("proj_in", "video_patch_proj"),
        ("proj_out", "final_layer.video_out"),
        ("time_embedder.linear_1", "time_embedder.proj_in"),
        ("time_embedder.linear_2", "time_embedder.proj_out"),
    )
    for source, target in prefix_replacements:
        if name == source or name.startswith(source if source.endswith(".") else source + "."):
            name = target + name[len(source) :]
            break

    # VDN wraps the released Diffusers attention as attn.orig before training
    # its hybrid branch. MLX keeps that inherited attention directly on the block.
    name = name.replace(".attn.orig.", ".attn.")

    for suffix, slot in ((".attn.to_q", 0), (".attn.to_k", 1), (".attn.to_v", 2)):
        if name.endswith(suffix):
            return name[: -len(suffix)] + ".attn.qkv_proj", slot, False
    if name.endswith(".attn.to_out.0"):
        return name[: -len(".attn.to_out.0")] + ".attn.out_proj", None, False
    if name.endswith(".attn.norm_q"):
        return name[: -len(".attn.norm_q")] + ".attn.q_norm", None, False
    if name.endswith(".attn.norm_k"):
        return name[: -len(".attn.norm_k")] + ".attn.k_norm", None, False
    if name.endswith(".ff.net.0.proj"):
        return name[: -len(".ff.net.0.proj")] + ".mlp.fc1", None, True
    if name.endswith(".ff.net.2"):
        return name[: -len(".ff.net.2")] + ".mlp.fc2", None, False
    return name, None, False


def _split_lora_key(name: str) -> tuple[str, str, int | None, bool] | None:
    endings = {
        ".lora_A.turbo.weight": "a",
        ".lora_B.turbo.weight": "b",
        ".lora_A.default.weight": "a",
        ".lora_B.default.weight": "b",
        ".lora_A.weight": "a",
        ".lora_B.weight": "b",
        ".lora_down.weight": "a",
        ".lora_up.weight": "b",
        ".lora_a.weight": "a",
        ".lora_b.weight": "b",
        ".alpha.weight": "alpha",
        ".lora_alpha": "alpha",
        ".alpha": "alpha",
    }
    for ending, kind in endings.items():
        if name.endswith(ending):
            target, output_slot, swap_halves = _fastvideo_target(name[: -len(ending)])
            return target, kind, output_slot, swap_halves
    return None


def _declared_network_scaling(path: Path) -> tuple[float | None, float | None]:
    """Read source-level alpha/rank metadata without relying on exporter names."""
    resolved = path.resolve()
    stat = resolved.stat()
    return _cached_declared_network_scaling(str(resolved), stat.st_size, stat.st_mtime_ns)


@lru_cache(maxsize=64)
def _cached_declared_network_scaling(
    path: str, _size: int, _mtime_ns: int
) -> tuple[float | None, float | None]:
    with safe_open(path, framework="numpy") as handle:
        metadata = handle.metadata() or {}
    scaling = inspect_adapter_scaling(metadata, adapter_label="MiniMax H3 LoRA")
    return scaling["rank"], scaling["alpha"]


def _effective_lora_scale(
    values: dict[str, mx.array],
    pair_rank: int,
    strength: float,
    declared_scaling: tuple[float | None, float | None],
) -> float:
    """Normalize per-target and exporter-level alpha conventions to one scale."""
    alpha_tensor = values.get("alpha")
    if alpha_tensor is not None:
        if alpha_tensor.size != 1:
            raise ValueError("MiniMax H3 LoRA per-target alpha must be scalar.")
        alpha = float(alpha_tensor.item())
        denominator = float(pair_rank)
    else:
        global_rank, global_alpha = declared_scaling
        if global_alpha is None:
            return float(strength)
        alpha = global_alpha
        denominator = global_rank if global_rank is not None else float(pair_rank)
    if not math.isfinite(alpha) or alpha < 0 or denominator <= 0:
        raise ValueError("MiniMax H3 LoRA has invalid alpha/rank scaling values.")
    return float(strength) * alpha / denominator


def _split_exact_delta_key(name: str) -> tuple[str, str] | None:
    for ending, kind in ((".diff_b", "bias"), (".diff", "weight")):
        if name.endswith(ending):
            target, output_slot, swap_halves = _fastvideo_target(name[: -len(ending)])
            if output_slot is not None or swap_halves:
                raise ValueError(
                    f"FastH3 exact delta target {name!r} requires unsupported fused surgery."
                )
            return target, kind
    return None


def _resolve_child(parent, part: str):
    if part.isdigit():
        return parent[int(part)]
    return getattr(parent, part)


def _get_target(model, path: str):
    target = model
    for part in path.split("."):
        target = _resolve_child(target, part)
    return target


def _set_target(model, path: str, value) -> None:
    parts = path.split(".")
    parent = model
    for part in parts[:-1]:
        parent = _resolve_child(parent, part)
    leaf = parts[-1]
    if leaf.isdigit():
        parent[int(leaf)] = value
    else:
        setattr(parent, leaf, value)


def _base_layer(layer):
    while hasattr(layer, "base"):
        layer = layer.base
    return layer


class DenseDeltaLinear(nn.Module):
    """Add a FastVideo full-rank delta without dequantizing the base projection."""

    def __init__(self, base, weight: mx.array, scale: float, source_grid=None):
        super().__init__()
        self.base = base
        self.delta_weight = weight
        self.scale = float(scale)
        self.source_grid = source_grid
        self.prepared_input = None
        self.output_dims, self.input_dims = _logical_shape(base)

    def __call__(self, value: mx.array) -> mx.array:
        output = self.base(value)
        source = self.prepared_input if self.source_grid is not None else value
        if source is None:
            raise RuntimeError(
                "The pruned AdaLN FastH3 exact-delta input grid was not prepared."
            )
        delta = source.astype(self.delta_weight.dtype) @ self.delta_weight.T
        return output + (delta * self.scale).astype(output.dtype)

    def prepare(self, timesteps: mx.array) -> None:
        if self.source_grid is None:
            return
        grid = self.source_grid.astype(mx.float32)
        position = mx.clip(timesteps.astype(mx.float32), 0.0, 1.0) * (grid.shape[0] - 1)
        lower = mx.minimum(mx.floor(position).astype(mx.int32), grid.shape[0] - 2)
        fraction = (position - lower.astype(mx.float32))[:, None]
        self.prepared_input = grid[lower] * (1.0 - fraction) + grid[lower + 1] * fraction
        mx.eval(self.prepared_input)


def _apply_exact_deltas(
    model,
    deltas: dict[str, dict[str, mx.array]],
    strength: float,
    source_grid=None,
) -> int:
    applied = 0
    for target, values in sorted(deltas.items()):
        try:
            layer = _get_target(model, target)
        except AttributeError as exc:
            if target.startswith("time_embedder.") and not hasattr(model, "time_embedder"):
                raise ValueError(
                    "FastH3 v2 contains trained timestep-embedder deltas and cannot be applied "
                    "to a pruned-AdaLN H3 transformer. Select an unpruned BF16 or Q8 base "
                    "transformer; a base-model AdaLN input grid is not equivalent."
                ) from exc
            raise
        if "weight" in values:
            delta = values["weight"]
            weight = getattr(_base_layer(layer), "weight", None)
            if delta.ndim == 1 and isinstance(weight, mx.array) and weight.shape == delta.shape:
                _base_layer(layer).weight = weight + delta.astype(weight.dtype) * strength
            elif delta.ndim == 2:
                output_width, input_width = _logical_shape(layer)
                uses_source_grid = (
                    source_grid is not None
                    and ".adaln_proj.linear" in f".{target}"
                    and int(delta.shape[0]) == output_width
                    and int(delta.shape[1]) == int(source_grid.shape[1])
                )
                if tuple(delta.shape) != (output_width, input_width) and not uses_source_grid:
                    raise ValueError(
                        f"FastH3 exact delta {target!r} has shape {delta.shape}; expected "
                        f"({output_width}, {input_width})."
                    )
                replacement = DenseDeltaLinear(
                    layer, delta, strength, source_grid if uses_source_grid else None
                )
                _set_target(model, target, replacement)
                layer = replacement
            else:
                raise ValueError(f"FastH3 exact delta {target!r} is not a supported parameter.")
            applied += 1
        if "bias" in values:
            base = _base_layer(layer)
            bias = getattr(base, "bias", None)
            delta = values["bias"]
            if not isinstance(bias, mx.array) or bias.shape != delta.shape:
                raise ValueError(f"FastH3 bias delta {target!r} does not match the base layer.")
            base.bias = bias + delta.astype(bias.dtype) * strength
            applied += 1
    return applied


def _logical_shape(layer) -> tuple[int, int]:
    base = _base_layer(layer)
    input_dims = getattr(base, "input_dims", None)
    output_dims = getattr(base, "output_dims", None)
    if input_dims is not None and output_dims is not None:
        return int(output_dims), int(input_dims)
    weight = getattr(base, "weight", None)
    if not isinstance(weight, mx.array) or weight.ndim != 2:
        raise TypeError(f"LoRA target {type(base).__name__} is not a supported linear layer.")
    bits = getattr(base, "bits", None)
    if weight.dtype == mx.uint32 and bits is not None:
        logical_input_width = int(weight.shape[1]) * 32 // int(bits)
        return int(weight.shape[0]), logical_input_width
    return int(weight.shape[0]), int(weight.shape[1])


def _is_qkv_target(target: str) -> bool:
    return target == "attn.qkv_proj" or target.endswith(".attn.qkv_proj")


def _prepare_lora_b(dit, target: str, b: mx.array, qkv_layout: str) -> tuple[mx.array, bool]:
    """Convert a fused QKV LoRA output from contiguous Q/K/V rows when requested."""
    if qkv_layout not in {"native_interleaved", "contiguous_qkv"}:
        raise ValueError(
            "MiniMax H3 LoRA QKV layout must be native_interleaved or contiguous_qkv."
        )
    if qkv_layout == "native_interleaved" or not _is_qkv_target(target):
        return b, False
    heads = int(dit.config.num_attention_heads)
    head_dim = int(dit.config.attention_head_dim)
    expected_rows = 3 * heads * head_dim
    if b.ndim != 2 or int(b.shape[0]) != expected_rows:
        raise ValueError(
            f"LoRA target {target!r} cannot convert contiguous QKV rows: expected "
            f"B shape ({expected_rows}, rank), got {b.shape}."
        )
    rank = int(b.shape[1])
    converted = (
        b.reshape(3, heads, head_dim, rank)
        .transpose(1, 0, 2, 3)
        .reshape(expected_rows, rank)
    )
    return converted, True


def _prepare_block_lora_b(block, target: str, b: mx.array, qkv_layout: str) -> mx.array:
    """Apply the same QKV conversion to a single materialized paged block."""
    if qkv_layout not in {"native_interleaved", "contiguous_qkv"}:
        raise ValueError(
            "MiniMax H3 LoRA QKV layout must be native_interleaved or contiguous_qkv."
        )
    if qkv_layout == "native_interleaved" or not _is_qkv_target(target):
        return b
    heads = int(block.attn.heads)
    head_dim = int(block.attn.head_dim)
    expected_rows = 3 * heads * head_dim
    if b.ndim != 2 or int(b.shape[0]) != expected_rows:
        raise ValueError(
            f"LoRA target {target!r} cannot convert contiguous QKV rows: expected "
            f"B shape ({expected_rows}, rank), got {b.shape}."
        )
    rank = int(b.shape[1])
    return (
        b.reshape(3, heads, head_dim, rank)
        .transpose(1, 0, 2, 3)
        .reshape(expected_rows, rank)
    )


def _load_input_grid(path: str | Path, expected_width: int) -> mx.array:
    values = mx.load(str(path))
    if "silu_t_emb_grid" in values:
        grid = values["silu_t_emb_grid"]
    elif len(values) == 1:
        grid = next(iter(values.values()))
    else:
        raise KeyError(
            "The AdaLN input-grid safetensors must contain `silu_t_emb_grid` or one tensor."
        )
    if grid.ndim != 2 or grid.shape[0] < 2 or grid.shape[1] != expected_width:
        raise ValueError(
            "The AdaLN input grid must have shape (at least 2, "
            f"{expected_width}), got {grid.shape}."
        )
    mx.eval(grid)
    return grid


def apply_lora(dit, request: LoRARequest) -> LoRAApplyReport:
    """Apply a generic LoRA safetensors file to a loaded H3 transformer."""
    if request.start_after_evaluations < 0:
        raise ValueError("LoRA start_after_evaluations must be zero or greater.")
    if getattr(dit, "paged_blocks", None) is not None:
        return _apply_paged_lora(dit, request)
    path = Path(request.path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"MiniMax H3 LoRA file not found: {path}")
    declared_scaling = _declared_network_scaling(path)
    tensors = mx.load(str(path))
    grouped: dict[tuple[str, int | None, bool], dict[str, mx.array]] = {}
    exact_deltas: dict[str, dict[str, mx.array]] = {}
    for key, value in tensors.items():
        parsed = _split_lora_key(key)
        if parsed is not None:
            target, kind, output_slot, swap_halves = parsed
            grouped.setdefault((target, output_slot, swap_halves), {})[kind] = value
            continue
        exact = _split_exact_delta_key(key)
        if exact is not None:
            target, kind = exact
            exact_deltas.setdefault(target, {})[kind] = value

    if not grouped and not exact_deltas:
        raise ValueError(f"The LoRA file contains no supported adapter pairs: {path}")

    source_grid = None
    for target, values in exact_deltas.items():
        delta = values.get("weight")
        if delta is None or delta.ndim != 2 or ".adaln_proj.linear" not in f".{target}":
            continue
        _output_width, input_width = _logical_shape(_get_target(dit, target))
        if int(delta.shape[1]) != input_width:
            if request.adaln_input_grid is None:
                raise ValueError(
                    "This FastH3 exact delta targets the original H3 AdaLN timestep embedding, "
                    "but the selected transformer uses a pruned curve. Supply an AdaLN "
                    "input-grid safetensors file."
                )
            source_grid = _load_input_grid(request.adaln_input_grid, int(delta.shape[1]))
            break

    exact_delta_targets = _apply_exact_deltas(
        dit, exact_deltas, request.strength, source_grid
    )
    prepared: list[tuple[str, _LoRAProjection]] = []
    adaln_targets = 0
    qkv_permuted_targets = 0
    for (target, output_slot, swap_halves), values in sorted(
        grouped.items(), key=lambda item: (item[0][0], item[0][1] or -1)
    ):
        if "a" not in values or "b" not in values:
            raise ValueError(f"LoRA target {target!r} does not contain both A and B tensors.")
        a, b = values["a"], values["b"]
        qkv_permuted = False
        if output_slot is None:
            b, qkv_permuted = _prepare_lora_b(dit, target, b, request.qkv_layout)
        if swap_halves:
            if int(b.shape[0]) % 2:
                raise ValueError(f"FastH3 SwiGLU target {target!r} has an odd output width.")
            half = int(b.shape[0]) // 2
            b = mx.concatenate([b[half:], b[:half]], axis=0)
        if a.ndim != 2 or b.ndim != 2 or a.shape[0] != b.shape[1]:
            raise ValueError(
                f"LoRA target {target!r} has incompatible A/B shapes {a.shape}/{b.shape}."
            )
        layer = _get_target(dit, target)
        output_width, input_width = _logical_shape(layer)
        is_adaln = ".adaln_proj.linear" in f".{target}"
        grid = None
        if a.shape[1] != input_width:
            if not is_adaln or dit.config.adaln_curve_grid is None:
                raise ValueError(
                    f"LoRA target {target!r} expects input width {a.shape[1]}, "
                    f"but the base layer uses {input_width}."
                )
            if request.adaln_input_grid is None:
                raise ValueError(
                    "This LoRA targets the original H3 AdaLN timestep embedding, but the selected "
                    "transformer is a pruned curve checkpoint. Supply an AdaLN input-grid "
                    "safetensors file."
                )
            if source_grid is None:
                source_grid = _load_input_grid(request.adaln_input_grid, int(a.shape[1]))
            grid = source_grid
        expected_output = output_width // 3 if output_slot is not None else output_width
        if b.shape[0] != expected_output:
            raise ValueError(
                f"LoRA target {target!r} expects output width {b.shape[0]}, "
                f"but the base layer uses {expected_output}."
            )
        scale = _effective_lora_scale(
            values, int(a.shape[0]), request.strength, declared_scaling
        )
        if request.start_after_evaluations and is_adaln:
            raise ValueError(
                "Staged LoRA activation does not support AdaLN adapter targets because the "
                "current H3 modulation cache is shared by the complete sampling schedule."
            )
        prepared.append(
            (
                target,
                _LoRAProjection(
                    a,
                    b,
                    scale,
                    grid,
                    request.start_after_evaluations,
                    output_slot,
                    int(dit.config.attention_head_dim) if output_slot is not None else None,
                ),
            )
        )
        adaln_targets += int(is_adaln)
        qkv_permuted_targets += int(qkv_permuted)

    for target, adapter in prepared:
        current = _get_target(dit, target)
        if isinstance(current, LoRALinear):
            replacement = LoRALinear(current.base, [*current.adapters, adapter])
        else:
            replacement = LoRALinear(current, [adapter])
        _set_target(dit, target, replacement)

    mx.eval(dit.parameters())
    return LoRAApplyReport(
        path=str(path),
        strength=request.strength,
        targets=len(prepared) + exact_delta_targets,
        adaln_targets=adaln_targets,
        qkv_permuted_targets=qkv_permuted_targets,
        tensor_bytes=sum(value.nbytes for value in tensors.values()),
        start_after_evaluations=request.start_after_evaluations,
    )


def apply_lora_stack(dit, requests) -> tuple[LoRAApplyReport, ...]:
    """Apply a host-provided LoRA stack in graph order."""
    return tuple(apply_lora(dit, LoRARequest(**request)) for request in requests)


def configure_block_lora_batching(block, enabled: bool = True) -> int:
    """Benchmark-only opt-in: batching did not beat sequential updates on M3 Ultra."""
    count = 0
    for parent, name in (
        (block.attn, "qkv_proj"), (block.attn, "out_proj"),
        (block.mlp, "fc1"), (block.mlp, "fc2"),
    ):
        layer = getattr(parent, name)
        if isinstance(layer, LoRALinear):
            layer.configure_batched_inputs(enabled)
            count += len(layer._input_groups)
    return count


def prepare_lora_timesteps(dit, timesteps: mx.array) -> None:
    """Prepare pruned-AdaLN adapter inputs for the current global timestep table."""
    paged = getattr(dit, "paged_blocks", None)
    if paged is not None:
        paged.lora_timesteps = timesteps
    for block in dit.blocks:
        linear = block.adaln_proj.linear
        if callable(getattr(linear, "prepare", None)):
            linear.prepare(timesteps)
    linear = dit.final_layer.adaln_proj.linear
    if callable(getattr(linear, "prepare", None)):
        linear.prepare(timesteps)


def _apply_paged_lora(dit, request: LoRARequest) -> LoRAApplyReport:
    """Validate one adapter without retaining block weights, then register it with the pager."""
    from .dit import TransformerBlock

    path = Path(request.path).expanduser()
    if request.start_after_evaluations < 0:
        raise ValueError("LoRA start_after_evaluations must be zero or greater.")
    if not path.is_file():
        raise FileNotFoundError(f"MiniMax H3 LoRA file not found: {path}")
    declared_scaling = _declared_network_scaling(path)
    tensors = dict(mx.load(str(path)))
    grouped: dict[tuple[str, int | None, bool], dict[str, mx.array]] = {}
    exact_deltas_present = False
    for key, value in tensors.items():
        parsed = _split_lora_key(key)
        if parsed is not None:
            target, kind, output_slot, swap_halves = parsed
            grouped.setdefault((target, output_slot, swap_halves), {})[kind] = value
        elif _split_exact_delta_key(key) is not None:
            exact_deltas_present = True
    if exact_deltas_present:
        raise ValueError(
            "FastH3 v2 adapters currently require a resident H3 transformer. "
            "Select a non-paged transformer until per-page exact-delta loading is enabled."
        )
    if not grouped:
        raise ValueError(f"The LoRA file contains no supported adapter pairs: {path}")

    representative = TransformerBlock(dit.config)
    fixed: list[tuple[str, _LoRAProjection]] = []
    source_grid = None
    adaln_targets = 0
    qkv_permuted_targets = 0
    for (target, output_slot, swap_halves), values in sorted(
        grouped.items(), key=lambda item: (item[0][0], item[0][1] or -1)
    ):
        if "a" not in values or "b" not in values:
            raise ValueError(f"LoRA target {target!r} does not contain both A and B tensors.")
        parts = target.split(".")
        if len(parts) > 2 and parts[0] == "blocks" and parts[1].isdigit():
            index = int(parts[1])
            if not 0 <= index < dit.paged_blocks.num_blocks:
                raise ValueError(f"LoRA target {target!r} addresses a missing H3 block.")
            layer = _get_target(representative, ".".join(parts[2:]))
            is_block = True
        else:
            layer = _get_target(dit, target)
            is_block = False
        a, b = values["a"], values["b"]
        qkv_permuted = False
        if output_slot is None:
            b, qkv_permuted = _prepare_lora_b(dit, target, b, request.qkv_layout)
        if swap_halves:
            if int(b.shape[0]) % 2:
                raise ValueError(f"SwiGLU LoRA target {target!r} has an odd output width.")
            half = int(b.shape[0]) // 2
            b = mx.concatenate([b[half:], b[:half]], axis=0)
        if a.ndim != 2 or b.ndim != 2 or a.shape[0] != b.shape[1]:
            raise ValueError(
                f"LoRA target {target!r} has incompatible A/B shapes {a.shape}/{b.shape}."
            )
        output_width, input_width = _logical_shape(layer)
        is_adaln = ".adaln_proj.linear" in f".{target}"
        if request.start_after_evaluations and is_adaln:
            raise ValueError(
                "Staged LoRA activation does not support AdaLN adapter targets because the "
                "current H3 modulation cache is shared by the complete sampling schedule."
            )
        grid = None
        if a.shape[1] != input_width:
            if not is_adaln or dit.config.adaln_curve_grid is None:
                raise ValueError(
                    f"LoRA target {target!r} expects input width {a.shape[1]}, "
                    f"but the base layer uses {input_width}."
                )
            if request.adaln_input_grid is None:
                raise ValueError(
                    "This LoRA targets the original H3 AdaLN timestep embedding, but the selected "
                    "transformer is a pruned curve checkpoint. Supply an AdaLN input-grid "
                    "safetensors file."
                )
            if source_grid is None:
                source_grid = _load_input_grid(request.adaln_input_grid, int(a.shape[1]))
            grid = source_grid
        expected_output = output_width // 3 if output_slot is not None else output_width
        if b.shape[0] != expected_output:
            raise ValueError(
                f"LoRA target {target!r} expects output width {b.shape[0]}, "
                f"but the base layer uses {expected_output}."
            )
        adapter = _LoRAProjection(
            a,
            b,
            _effective_lora_scale(
                values, int(a.shape[0]), request.strength, declared_scaling
            ),
            grid,
            request.start_after_evaluations,
            output_slot=output_slot,
            qkv_head_dim=int(dit.config.attention_head_dim) if output_slot is not None else None,
        )
        if not is_block:
            fixed.append((target, adapter))
        adaln_targets += int(is_adaln)
        qkv_permuted_targets += int(qkv_permuted)

    for target, adapter in fixed:
        current = _get_target(dit, target)
        replacement = (
            LoRALinear(current.base, [*current.adapters, adapter])
            if isinstance(current, LoRALinear)
            else LoRALinear(current, [adapter])
        )
        _set_target(dit, target, replacement)
    dit.paged_blocks.lora_requests.append((request, source_grid))
    mx.eval(dit.parameters())
    tensor_bytes = sum(value.nbytes for value in tensors.values())
    tensors.clear()
    mx.clear_cache()
    return LoRAApplyReport(
        path=str(path),
        strength=request.strength,
        targets=len(grouped),
        adaln_targets=adaln_targets,
        qkv_permuted_targets=qkv_permuted_targets,
        tensor_bytes=tensor_bytes,
        start_after_evaluations=request.start_after_evaluations,
    )


def apply_paged_loras_to_block(
    block,
    index: int,
    requests: list[tuple[LoRARequest, mx.array | None]],
    timesteps: mx.array | None,
    *,
    tensor_cache: dict | None = None,
    skip_adaln: bool = False,
) -> None:
    """Attach only one materialized block's adapter tensors for its bounded lifetime."""
    prefix = f"blocks.{index}."
    for request, source_grid in requests:
        cache_key = str(Path(request.path).expanduser())
        declared_scaling = _declared_network_scaling(Path(cache_key))
        if tensor_cache is None:
            tensors = mx.load(cache_key)
        else:
            if cache_key not in tensor_cache:
                tensor_cache[cache_key] = mx.load(cache_key)
            tensors = tensor_cache[cache_key]
        grouped: dict[tuple[str, int | None, bool], dict[str, mx.array]] = {}
        for key, value in tensors.items():
            parsed = _split_lora_key(key)
            if parsed is None or not parsed[0].startswith(prefix):
                continue
            target, kind, output_slot, swap_halves = parsed
            if skip_adaln and ".adaln_proj.linear" in f".{target}":
                continue
            grouped.setdefault((target[len(prefix):], output_slot, swap_halves), {})[kind] = value
        for (target, output_slot, swap_halves), values in grouped.items():
            a, b = values["a"], values["b"]
            if output_slot is None:
                b = _prepare_block_lora_b(block, target, b, request.qkv_layout)
            if swap_halves:
                half = int(b.shape[0]) // 2
                b = mx.concatenate([b[half:], b[:half]], axis=0)
            grid = source_grid if ".adaln_proj.linear" in f".{target}" else None
            adapter = _LoRAProjection(
                a,
                b,
                _effective_lora_scale(
                    values, int(a.shape[0]), request.strength, declared_scaling
                ),
                grid,
                request.start_after_evaluations,
                output_slot=output_slot,
                qkv_head_dim=int(block.attn.head_dim) if output_slot is not None else None,
            )
            if grid is not None:
                if timesteps is None:
                    raise RuntimeError(
                        "Paged H3 AdaLN LoRA inputs were not prepared for the sampling schedule."
                    )
                adapter.prepare(timesteps)
            current = _get_target(block, target)
            replacement = (
                LoRALinear(current.base, [*current.adapters, adapter])
                if isinstance(current, LoRALinear)
                else LoRALinear(current, [adapter])
            )
            _set_target(block, target, replacement)
