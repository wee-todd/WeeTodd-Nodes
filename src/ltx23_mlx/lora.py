"""Source-independent standard LoRA loading for resident or streamed LTX 2.3 transformers.

Only selected projections are fused in memory. Base files are never rewritten.
Ordinary adapters remain active through the upstream refinement stage because
the model's original parameter names and Linear/QuantizedLinear types are retained.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from types import MethodType

from wee_todd_mlx.adapter_contract import inspect_adapter
from wee_todd_mlx.model_library import inspect_safetensors_header


def recipe_specs(recipe):
    embedded = recipe.get("components", {}).get("loras") or []
    if recipe.get("loras") and embedded:
        raise ValueError("Declare LTX 2.3 adapters either at top level or in components, not both")
    value = recipe.get("loras") or {"adapters": embedded}
    if (
        not isinstance(value, dict)
        or set(value) != {"adapters"}
        or not isinstance(value["adapters"], list)
    ):
        raise ValueError(
            "Expected a loras.adapters list; refusing to render without requested adapters"
        )
    return tuple(LTX23LoRASpec(**item) for item in value["adapters"])


def target_name(name):
    for prefix in (
        "base_model.model.",
        "model.diffusion_model.",
        "diffusion_model.",
        "transformer.",
    ):
        if name.startswith(prefix):
            name = name[len(prefix) :]
    for before, after in (
        (".to_out.0", ".to_out"),
        (".ff.net.0.proj", ".ff.proj_in"),
        (".ff.net.2", ".ff.proj_out"),
        (".audio_ff.net.0.proj", ".audio_ff.proj_in"),
        (".audio_ff.net.2", ".audio_ff.proj_out"),
        (".linear_1", ".linear1"),
        (".linear_2", ".linear2"),
    ):
        name = name.replace(before, after)
    return name


@dataclass(frozen=True)
class LTX23LoRASpec:
    path: str
    strength: float = 1.0
    alpha: float | None = None

    def inspect(self):
        if Path(self.path).suffix.lower() != ".safetensors":
            raise ValueError("LTX 2.3 LoRA files must use safetensors")
        if not math.isfinite(self.strength) or not -10 <= self.strength <= 10:
            raise ValueError("LTX 2.3 LoRA strength must be finite and between -10 and 10")
        if self.alpha is not None and (not math.isfinite(self.alpha) or self.alpha < 0):
            raise ValueError("LoRA alpha must be finite and nonnegative")
        report = inspect_adapter(Path(self.path).expanduser())
        metadata = report["metadata"]
        network_args = json.loads(metadata.get("ss_network_args", "{}"))
        if not isinstance(network_args, dict):
            raise ValueError("Invalid ss_network_args metadata")
        if "lycoris" in metadata.get("ss_network_module", "").lower():
            raise ValueError("LyCORIS requires a separate adapter implementation")
        version = metadata.get("model_version")
        if version and not (version == "2.3" or version.startswith("2.3.")):
            raise ValueError(f"Adapter declares a different model version: {version}")
        if any(key.startswith("reference_") for key in metadata):
            raise ValueError("Task/control adapters require their own conditioning pipeline")
        if metadata.get("adapter_role", "transformer_lora") not in {
            "transformer_lora",
            "standard",
            "style",
            "character",
        }:
            raise ValueError("Declared adapter role requires a specialized pipeline")
        for key in ("use_dora", "use_rslora"):
            if any(
                str(source.get(key, "false")).lower() not in {"false", "0", "none"}
                for source in (metadata, network_args)
            ):
                raise ValueError(f"Unsupported LoRA scaling/format: {key}")
        scaling = report["declared_scaling"]
        report["global_alpha"] = (
            self.alpha if self.alpha is not None else scaling["alpha"]
        )
        report["global_rank"] = scaling["rank"]
        seen = set()
        for pair in report["pairs"]:
            pair["mapped_target"] = target_name(pair["target"])
            if pair["mapped_target"] in seen:
                raise ValueError("Multiple adapter targets map to the same LTX projection")
            seen.add(pair["mapped_target"])
        return report


def validate_stack(specs, model_dir, mode, *, low_ram_streaming=False):
    if len(specs) > 8:
        raise ValueError("LTX 2.3 supports at most eight generic LoRAs")
    if not specs:
        return []
    root = Path(model_dir).expanduser()
    stem = "transformer-distilled" if mode == "distilled" else "transformer-dev"
    # Match the active upstream selection: distilled prefers a versioned file;
    # dev enters through its exact path (the split reader expands numbered shards).
    versioned = sorted(root.glob(stem + "-*.safetensors")) if mode == "distilled" else []
    selected = versioned[-1] if versioned else root / (stem + ".safetensors")
    files = (
        sorted(root.glob(stem + "-*-of-*.safetensors"))
        if "-of-" in selected.name or not selected.is_file()
        else [selected]
    )
    if not files:
        raise FileNotFoundError(f"No transformer weights for LoRA target validation: {selected}")
    tensors = {}
    for file in files:
        for key, value in inspect_safetensors_header(file, include_tensors=True)["tensors"].items():
            key = key.removeprefix("transformer.")
            if key in tensors:
                raise ValueError(f"Duplicate transformer tensor across shards: {key}")
            tensors[key] = value
    reports = []
    for spec in specs:
        report = spec.inspect()
        for pair in report["pairs"]:
            target = pair["mapped_target"]
            weight = tensors.get(target + ".weight")
            if weight is None:
                raise ValueError(f"Unmatched LTX 2.3 LoRA target: {target}")
            actual = pair["logical_shape"]
            if target + ".scales" in tensors:
                if len(weight["shape"]) != 2 or weight["shape"][0] != actual[0]:
                    raise ValueError(f"Quantized LoRA target output shape mismatch: {target}")
                pair["input_shape_check"] = "deferred_to_loaded_quantized_projection"
            elif weight["shape"] != actual:
                raise ValueError(f"LoRA target shape mismatch: {target}")
        reports.append(report)
    return reports


def _module(model, name):
    value = model
    for part in name.split("."):
        value = value[int(part)] if part.isdecimal() else getattr(value, part)
    return value


def _pair_alpha_ratio(spec, report, pair, weights):
    alpha = report["global_alpha"]
    if spec.alpha is None and pair["alpha_tensor"]:
        alpha = float(weights[pair["alpha_tensor"]].item())
    alpha = pair["rank"] if alpha is None else alpha
    if not math.isfinite(alpha) or alpha < 0:
        raise ValueError(f"Invalid LoRA alpha: {pair['mapped_target']}")
    declared_rank = (
        report["global_rank"]
        if spec.alpha is None
        and pair["alpha_tensor"] is None
        and report["global_alpha"] is not None
        else None
    )
    ratio = alpha / (declared_rank or pair["rank"])
    if not math.isfinite(ratio):
        raise ValueError(f"Non-finite effective LoRA scale: {pair['mapped_target']}")
    return ratio, alpha, declared_rank


class LTX23StreamingLoRASource:
    """Normalize arbitrary supported LoRA schemas at streamed block bind time."""

    def __init__(self, spec: LTX23LoRASpec) -> None:
        import mlx.core as mx

        self.spec = spec
        self.strength = spec.strength
        self.report = spec.inspect()
        self._weights = dict(mx.load(str(Path(spec.path).expanduser())))
        self._blocks = {}
        self._fixed = []
        for pair in self.report["pairs"]:
            name = pair["mapped_target"]
            parts = name.split(".", 2)
            if len(parts) == 3 and parts[0] == "transformer_blocks" and parts[1].isdigit():
                self._blocks.setdefault(int(parts[1]), []).append((parts[2], pair))
            else:
                self._fixed.append((name, pair))

    def has_block(self, block_idx):
        return bool(self._blocks.get(block_idx))

    def _pair_values(self, name, pair):
        import mlx.core as mx

        a, b = self._weights[pair["a"]], self._weights[pair["b"]]
        if not bool(mx.all(mx.isfinite(a)).item()) or not bool(mx.all(mx.isfinite(b)).item()):
            raise ValueError(f"Non-finite LoRA weights: {pair['mapped_target']}")
        ratio, _alpha, _declared_rank = _pair_alpha_ratio(
            self.spec, self.report, pair, self._weights
        )
        return {
            name + ".lora_A.weight": a,
            name + ".lora_B.weight": b if ratio == 1.0 else b * ratio,
        }

    def get_block_lora_dict(self, block_idx):
        values = {}
        for name, pair in self._blocks.get(block_idx, ()):
            values.update(self._pair_values(name, pair))
        return values

    def apply_non_block(self, model):
        """Fuse the small fixed-target subset while leaving block pairs lazy."""
        import mlx.core as mx
        import mlx.nn as nn

        prepared = []
        for name, pair in self._fixed:
            try:
                layer = _module(model, name)
            except (AttributeError, IndexError, KeyError, TypeError) as exc:
                raise ValueError(f"Unmatched LTX 2.3 LoRA target: {name}") from exc
            if not isinstance(layer, (nn.Linear, nn.QuantizedLinear)):
                raise ValueError(f"LoRA target is not a supported linear projection: {name}")
            shape = list(layer.weight.shape)
            if isinstance(layer, nn.QuantizedLinear):
                if layer.bits not in {4, 8} or getattr(layer, "mode", "affine") != "affine":
                    raise ValueError("Only affine Q4/Q8 LoRA fusion is supported")
                shape[1] = shape[1] * 32 // layer.bits
            if shape != pair["logical_shape"]:
                raise ValueError(f"Loaded LoRA projection shape mismatch: {name}")
            values = self._pair_values(name, pair)
            ratio, alpha, declared_rank = _pair_alpha_ratio(
                self.spec, self.report, pair, self._weights
            )
            prepared.append((layer, values, ratio, alpha, declared_rank, pair))

        details = []
        for layer, values, ratio, alpha, declared_rank, pair in prepared:
            name = pair["mapped_target"]
            effective_scale = self.strength * ratio
            quantized = isinstance(layer, nn.QuantizedLinear)
            if effective_scale != 0:
                weight = (
                    mx.dequantize(
                        layer.weight,
                        layer.scales,
                        layer.biases,
                        group_size=layer.group_size,
                        bits=layer.bits,
                    )
                    if quantized
                    else layer.weight
                )
                a = values[name + ".lora_A.weight"]
                # _pair_values already applies alpha/rank to B, so strength is
                # the only remaining multiplier here.
                b = values[name + ".lora_B.weight"]
                updated = (
                    weight.astype(mx.float32)
                    + (b.astype(mx.float32) @ a.astype(mx.float32)) * self.strength
                )
                if not bool(mx.all(mx.isfinite(updated)).item()):
                    raise ValueError("Non-finite fused LoRA projection")
                updated = updated.astype(weight.dtype)
                if quantized:
                    layer.weight, layer.scales, layer.biases = mx.quantize(
                        updated, group_size=layer.group_size, bits=layer.bits
                    )
                    mx.eval(layer.weight, layer.scales, layer.biases)
                else:
                    layer.weight = updated
                    mx.eval(layer.weight)
            details.append(
                {
                    "target": name,
                    "rank": pair["rank"],
                    "alpha": alpha,
                    "declared_rank": declared_rank,
                    "effective_scale": effective_scale,
                    "bits": layer.bits if quantized else None,
                    "group_size": layer.group_size if quantized else None,
                }
            )
        return details

    def application_report(self, fixed_details):
        block_targets = sum(len(values) for values in self._blocks.values())
        return {
            "path": self.spec.path,
            "strength": self.spec.strength,
            "targets": block_targets + len(fixed_details),
            "nonzero_targets": (block_targets + len(fixed_details)) if self.strength != 0 else 0,
            "application": "normalized_per_block_streaming",
            "stages": "all",
            "target_details": fixed_details,
            "streamed_block_targets": block_targets,
            "resident_fixed_targets": len(fixed_details),
            "pair_schemas": self.report["pair_schemas"],
        }

    def close(self):
        self._weights = {}
        self._blocks = {}
        self._fixed = []


def apply_stack(model, specs, *, check_interrupted=None):
    """Validate all targets, then fuse ordered deltas without duplicating a checkpoint.

    FP32 B@A is scaled by strength * alpha/rank (alpha defaults to rank).
    Q4/Q8 projections retain their original affine group size and precision.
    """
    import mlx.core as mx
    import mlx.nn as nn

    prepared = []
    for spec in specs:
        if check_interrupted:
            check_interrupted()
        report = spec.inspect()
        weights = mx.load(str(Path(spec.path).expanduser()))
        pairs, details = [], []
        for pair in report["pairs"]:
            if check_interrupted:
                check_interrupted()
            name = pair["mapped_target"]
            try:
                layer = _module(model, name)
            except (AttributeError, IndexError, KeyError, TypeError) as exc:
                raise ValueError(f"Unmatched LTX 2.3 LoRA target: {name}") from exc
            if not isinstance(layer, (nn.Linear, nn.QuantizedLinear)):
                raise ValueError(f"LoRA target is not a supported linear projection: {name}")
            shape = list(layer.weight.shape)
            if isinstance(layer, nn.QuantizedLinear):
                if layer.bits not in {4, 8} or getattr(layer, "mode", "affine") != "affine":
                    raise ValueError("Only affine Q4/Q8 LoRA fusion is supported")
                shape[1] = shape[1] * 32 // layer.bits
            if shape != pair["logical_shape"]:
                raise ValueError(f"Loaded LoRA projection shape mismatch: {name}")
            a, b = weights[pair["a"]], weights[pair["b"]]
            alpha = report["global_alpha"]
            if spec.alpha is None and pair["alpha_tensor"]:
                alpha = float(weights[pair["alpha_tensor"]].item())
            alpha = pair["rank"] if alpha is None else alpha
            if not math.isfinite(alpha) or alpha < 0:
                raise ValueError(f"Invalid LoRA alpha: {name}")
            if not bool(mx.all(mx.isfinite(a)).item()) or not bool(mx.all(mx.isfinite(b)).item()):
                raise ValueError(f"Non-finite LoRA weights: {name}")
            declared_rank = (
                report["global_rank"]
                if spec.alpha is None
                and pair["alpha_tensor"] is None
                and report["global_alpha"] is not None
                else None
            )
            effective_scale = spec.strength * alpha / (declared_rank or pair["rank"])
            if not math.isfinite(effective_scale):
                raise ValueError(f"Non-finite effective LoRA scale: {name}")
            pairs.append((layer, a, b, effective_scale))
            details.append(
                {
                    "target": name,
                    "rank": pair["rank"],
                    "alpha": alpha,
                    "declared_rank": declared_rank,
                    "effective_scale": effective_scale,
                    "bits": layer.bits if isinstance(layer, nn.QuantizedLinear) else None,
                    "group_size": layer.group_size
                    if isinstance(layer, nn.QuantizedLinear)
                    else None,
                }
            )
        prepared.append((spec, pairs, details))
    result = []
    for spec, pairs, details in prepared:
        for layer, a, b, scale in pairs:
            if check_interrupted:
                check_interrupted()
            if scale == 0:
                continue
            quantized = isinstance(layer, nn.QuantizedLinear)
            weight = (
                mx.dequantize(
                    layer.weight,
                    layer.scales,
                    layer.biases,
                    group_size=layer.group_size,
                    bits=layer.bits,
                )
                if quantized
                else layer.weight
            )
            updated = (
                weight.astype(mx.float32) + (b.astype(mx.float32) @ a.astype(mx.float32)) * scale
            )
            if not bool(mx.all(mx.isfinite(updated)).item()):
                raise ValueError("Non-finite fused LoRA projection")
            updated = updated.astype(weight.dtype)
            if not bool(mx.all(mx.isfinite(updated)).item()):
                raise ValueError("LoRA projection overflows the base precision")
            if quantized:
                layer.weight, layer.scales, layer.biases = mx.quantize(
                    updated, group_size=layer.group_size, bits=layer.bits
                )
                mx.eval(layer.weight, layer.scales, layer.biases)
            else:
                layer.weight = updated
                mx.eval(layer.weight)
        result.append(
            {
                "path": spec.path,
                "strength": spec.strength,
                "targets": len(pairs),
                "nonzero_targets": sum(scale != 0 for _, _, _, scale in pairs),
                "application": "ordered_in_memory_fusion",
                "stages": "all",
                "target_details": details,
            }
        )
    return result


def install_loader(pipeline, specs):
    """Instance-local hook at the shared transformer construction entry point."""
    name = "_load_transformer_with_optional_streaming"
    original = getattr(pipeline, name, None)
    if not callable(original):
        raise RuntimeError("Installed LTX pipeline lacks the generic transformer loader hook")
    reports = []

    def load(_self, *args, **kwargs):
        check = getattr(_self, "_weetodd_check_interrupted", None)
        if check:
            check()
        model = original(*args, **kwargs)
        if getattr(_self, "low_ram_streaming", False):
            sources = list(object.__getattribute__(model, "_lora_sources"))
            for spec in specs:
                if check:
                    check()
                source = LTX23StreamingLoRASource(spec)
                fixed_details = source.apply_non_block(model)
                sources.append(source)
                owned_sources = list(getattr(_self, "_weetodd_lora_sources", ()))
                owned_sources.append(source)
                _self._weetodd_lora_sources = owned_sources
                reports.append(source.application_report(fixed_details))
            object.__setattr__(model, "_lora_sources", sources)
        else:
            reports.extend(apply_stack(model, specs, check_interrupted=check))
        return model

    setattr(pipeline, name, MethodType(load, pipeline))
    return reports
