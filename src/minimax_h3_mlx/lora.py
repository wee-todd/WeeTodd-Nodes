"""Activation-space Low-Rank Adaptation (LoRA) for MiniMax H3 MLX modules."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn


@dataclass(frozen=True)
class LoRARequest:
    """One lazy LoRA application request supplied by the host adapter."""

    path: str
    strength: float = 1.0
    adaln_input_grid: str | None = None


@dataclass(frozen=True)
class LoRAApplyReport:
    """Auditable result of applying one LoRA file to a loaded transformer."""

    path: str
    strength: float
    targets: int
    adaln_targets: int
    tensor_bytes: int


class _LoRAProjection(nn.Module):
    def __init__(self, a: mx.array, b: mx.array, scale: float, source_grid=None):
        super().__init__()
        self.a = a
        self.b = b
        self.scale = float(scale)
        self.source_grid = source_grid
        self.prepared_input = None

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


class LoRALinear(nn.Module):
    """Run a base linear layer and add one or more LoRA updates in activation space."""

    def __init__(self, base, adapters: list[_LoRAProjection]):
        super().__init__()
        self.base = base
        self.adapters = adapters

    def __call__(self, value: mx.array) -> mx.array:
        output = self.base(value)
        for adapter in self.adapters:
            output = output + adapter.delta(value).astype(output.dtype)
        return output

    def prepare(self, timesteps: mx.array) -> None:
        for adapter in self.adapters:
            adapter.prepare(timesteps)


def _canonical_target(name: str) -> str:
    for prefix in (
        "model.diffusion_model.",
        "diffusion_model.",
        "transformer.",
        "model.",
    ):
        if name.startswith(prefix):
            return name[len(prefix) :]
    return name


def _split_lora_key(name: str) -> tuple[str, str] | None:
    endings = {
        ".lora_A.weight": "a",
        ".lora_B.weight": "b",
        ".lora_down.weight": "a",
        ".lora_up.weight": "b",
        ".alpha": "alpha",
    }
    for ending, kind in endings.items():
        if name.endswith(ending):
            return _canonical_target(name[: -len(ending)]), kind
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
    return layer.base if isinstance(layer, LoRALinear) else layer


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
    if getattr(dit, "paged_blocks", None) is not None:
        return _apply_paged_lora(dit, request)
    path = Path(request.path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"MiniMax H3 LoRA file not found: {path}")
    tensors = mx.load(str(path))
    grouped: dict[str, dict[str, mx.array]] = {}
    for key, value in tensors.items():
        parsed = _split_lora_key(key)
        if parsed is None:
            continue
        target, kind = parsed
        grouped.setdefault(target, {})[kind] = value

    if not grouped:
        raise ValueError(f"The LoRA file contains no supported adapter pairs: {path}")

    prepared: list[tuple[str, _LoRAProjection]] = []
    adaln_targets = 0
    source_grid = None
    for target, values in sorted(grouped.items()):
        if "a" not in values or "b" not in values:
            raise ValueError(f"LoRA target {target!r} does not contain both A and B tensors.")
        a, b = values["a"], values["b"]
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
        if b.shape[0] != output_width:
            raise ValueError(
                f"LoRA target {target!r} expects output width {b.shape[0]}, "
                f"but the base layer uses {output_width}."
            )
        alpha = float(values.get("alpha", mx.array(a.shape[0])).item())
        scale = request.strength * alpha / max(int(a.shape[0]), 1)
        prepared.append((target, _LoRAProjection(a, b, scale, grid)))
        adaln_targets += int(is_adaln)

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
        targets=len(prepared),
        adaln_targets=adaln_targets,
        tensor_bytes=sum(value.nbytes for value in tensors.values()),
    )


def apply_lora_stack(dit, requests) -> tuple[LoRAApplyReport, ...]:
    """Apply a host-provided LoRA stack in graph order."""
    return tuple(apply_lora(dit, LoRARequest(**request)) for request in requests)


def prepare_lora_timesteps(dit, timesteps: mx.array) -> None:
    """Prepare pruned-AdaLN adapter inputs for the current global timestep table."""
    paged = getattr(dit, "paged_blocks", None)
    if paged is not None:
        paged.lora_timesteps = timesteps
    for block in dit.blocks:
        linear = block.adaln_proj.linear
        if isinstance(linear, LoRALinear):
            linear.prepare(timesteps)
    linear = dit.final_layer.adaln_proj.linear
    if isinstance(linear, LoRALinear):
        linear.prepare(timesteps)


def _apply_paged_lora(dit, request: LoRARequest) -> LoRAApplyReport:
    """Validate one adapter without retaining block weights, then register it with the pager."""
    from .dit import TransformerBlock

    path = Path(request.path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"MiniMax H3 LoRA file not found: {path}")
    tensors = dict(mx.load(str(path)))
    grouped: dict[str, dict[str, mx.array]] = {}
    for key, value in tensors.items():
        parsed = _split_lora_key(key)
        if parsed is not None:
            target, kind = parsed
            grouped.setdefault(target, {})[kind] = value
    if not grouped:
        raise ValueError(f"The LoRA file contains no supported adapter pairs: {path}")

    representative = TransformerBlock(dit.config)
    fixed: list[tuple[str, _LoRAProjection]] = []
    source_grid = None
    adaln_targets = 0
    for target, values in sorted(grouped.items()):
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
        if a.ndim != 2 or b.ndim != 2 or a.shape[0] != b.shape[1]:
            raise ValueError(
                f"LoRA target {target!r} has incompatible A/B shapes {a.shape}/{b.shape}."
            )
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
        if b.shape[0] != output_width:
            raise ValueError(
                f"LoRA target {target!r} expects output width {b.shape[0]}, "
                f"but the base layer uses {output_width}."
            )
        alpha = float(values.get("alpha", mx.array(a.shape[0])).item())
        adapter = _LoRAProjection(
            a,
            b,
            request.strength * alpha / max(int(a.shape[0]), 1),
            grid,
        )
        if not is_block:
            fixed.append((target, adapter))
        adaln_targets += int(is_adaln)

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
        tensor_bytes=tensor_bytes,
    )


def apply_paged_loras_to_block(
    block,
    index: int,
    requests: list[tuple[LoRARequest, mx.array | None]],
    timesteps: mx.array | None,
) -> None:
    """Attach only one materialized block's adapter tensors for its bounded lifetime."""
    prefix = f"blocks.{index}."
    for request, source_grid in requests:
        tensors = mx.load(str(Path(request.path).expanduser()))
        grouped: dict[str, dict[str, mx.array]] = {}
        for key, value in tensors.items():
            parsed = _split_lora_key(key)
            if parsed is None or not parsed[0].startswith(prefix):
                continue
            target, kind = parsed
            grouped.setdefault(target[len(prefix) :], {})[kind] = value
        for target, values in grouped.items():
            a, b = values["a"], values["b"]
            alpha = float(values.get("alpha", mx.array(a.shape[0])).item())
            grid = source_grid if ".adaln_proj.linear" in f".{target}" else None
            adapter = _LoRAProjection(
                a,
                b,
                request.strength * alpha / max(int(a.shape[0]), 1),
                grid,
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
