"""Explicit IC-LoRA policy; shared projection math, separate stage/task semantics."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from types import MethodType

from wee_todd_mlx.adapter_contract import inspect_adapter

from .lora import LTX23LoRASpec, apply_stack, target_name, validate_stack

INGREDIENTS_DEV_TRANSFORMER = "transformer-dev.safetensors"
INGREDIENTS_DISTILLED_LORA = "ltx-2.3-22b-distilled-lora-384.safetensors"
INGREDIENTS_DISTILLED_LORA_STRENGTH = 0.5

CONTROL_TYPES = {
    "canny_edges": "union_control",
    "depth_map": "union_control",
    "pose_skeleton": "union_control",
    "motion_track": "motion_track",
    "ingredients_reference_sheet": "ingredients_reference_sheet",
}


@dataclass(frozen=True)
class LTX23ICLoRASpec:
    path: str
    family: str
    strength: float = 1.0
    alpha: float | None = None

    def inspect(self):
        if self.family not in set(CONTROL_TYPES.values()):
            raise ValueError("Unsupported LTX 2.3 IC-LoRA family")
        if Path(self.path).suffix.lower() != ".safetensors":
            raise ValueError("IC-LoRA requires a local safetensors file")
        if not math.isfinite(self.strength) or not -10 <= self.strength <= 10:
            raise ValueError("IC-LoRA strength must be finite and in [-10, 10]")
        if self.alpha is not None and (not math.isfinite(self.alpha) or self.alpha < 0):
            raise ValueError("IC-LoRA alpha must be finite and nonnegative")
        report = inspect_adapter(Path(self.path).expanduser())
        metadata = dict(report["metadata"])
        report["metadata"] = metadata
        if metadata.pop("license", None) is not None:
            report["license_metadata_present"] = True
        version = metadata.get("model_version", "")
        if version and not (version == "2.3" or version.startswith("2.3.")):
            raise ValueError("IC-LoRA declares a different model version")
        expected_scale = "1" if self.family == "ingredients_reference_sheet" else "2"
        if metadata.get("reference_downscale_factor") != expected_scale:
            raise ValueError(f"IC-LoRA must declare reference_downscale_factor={expected_scale}")
        if metadata.get("adapter_family", self.family) != self.family:
            raise ValueError("IC-LoRA family conflicts with adapter metadata")
        args = json.loads(metadata.get("ss_network_args", "{}"))
        if not isinstance(args, dict):
            raise ValueError("Invalid IC-LoRA ss_network_args")
        if "lycoris" in metadata.get("ss_network_module", "").lower():
            raise ValueError("LyCORIS IC-LoRA is unsupported")
        for key in ("use_dora", "use_rslora"):
            if any(
                str(s.get(key, "false")).lower() not in {"false", "0", "none"}
                for s in (metadata, args)
            ):
                raise ValueError(f"Unsupported IC-LoRA scaling: {key}")
        scaling = report["declared_scaling"]
        report["global_alpha"] = (
            self.alpha if self.alpha is not None else scaling["alpha"]
        )
        report["global_rank"] = scaling["rank"]
        seen = set()
        for pair in report["pairs"]:
            name = target_name(pair["target"])
            if name in seen:
                raise ValueError("Duplicate mapped IC-LoRA target")
            seen.add(name)
            pair["mapped_target"] = name
        # Official files do not identify their trained control family in metadata.
        # The explicit declaration is not a filename-based compatibility inference.
        report.update(
            adapter_family=self.family,
            family_provenance="explicit_declaration",
            reference_downscale_factor=int(expected_scale),
            stages="stage1_only",
        )
        return report


def recipe_ic_specs(recipe):
    value = recipe.get("components", {}).get("ic_loras", [])
    if not isinstance(value, list) or any(not isinstance(i, dict) for i in value):
        raise ValueError("LTX 2.3 components.ic_loras must be a list of adapter objects")
    return tuple(LTX23ICLoRASpec(**item) for item in value)


def ingredients_distillation_spec(spec):
    """The installed IC backend's explicit Dev-mode helper adapter."""
    return LTX23LoRASpec(
        str(spec.root() / INGREDIENTS_DISTILLED_LORA),
        strength=INGREDIENTS_DISTILLED_LORA_STRENGTH,
    )


def validate_ic_stack(spec, config):
    if not spec.ic_loras:
        return []
    if len(spec.ic_loras) != 1:
        raise ValueError("LTX 2.3 currently supports one IC-LoRA per render")
    ingredients = spec.ic_loras[0].family == "ingredients_reference_sheet"
    required_mode = "two_stage" if ingredients else "distilled"
    if config.pipeline_mode != required_mode or config.low_ram_streaming or spec.loras:
        label = "Dev two_stage" if ingredients else "distilled"
        raise ValueError(
            f"LTX 2.3 {spec.ic_loras[0].family} IC-LoRA requires resident {label} mode "
            "without generic LoRAs"
        )
    factor = 1 if ingredients else 2
    if config.width % (64 * factor) or config.height % (64 * factor):
        raise ValueError(
            "LTX 2.3 half-resolution IC references require dimensions divisible by 128"
        )
    # Upstream IC prefers this unversioned alias over the selected distilled bundle.
    # Refuse ambiguity rather than validating one transformer and loading another.
    if not ingredients and (spec.root() / "transformer.safetensors").exists():
        raise ValueError(
            "Ambiguous IC bundle: remove transformer.safetensors alias from the bundle selection"
        )
    reports = validate_stack(spec.ic_loras, spec.root(), required_mode)
    if ingredients:
        # Validate the backend-specific distilled helper against the exact Dev
        # checkpoint before allocating or mutating the runtime transformer.
        validate_stack((ingredients_distillation_spec(spec),), spec.root(), required_mode)
    return reports


def validate_control_inputs(spec, config, inputs, *, inspect_files=True):
    if not inputs or len(inputs) != 1:
        raise ValueError("LTX 2.3 IC-LoRA currently requires exactly one guide")
    if len(spec.ic_loras) != 1:
        raise ValueError("LTX 2.3 control requires one explicit IC-LoRA adapter")
    for item in inputs:
        if set(item) != {"path", "kind", "strength", "control_type"}:
            raise ValueError("LTX 2.3 guides require path, kind, strength, control_type")
        if CONTROL_TYPES.get(item["control_type"]) != spec.ic_loras[0].family:
            raise ValueError("Control guide does not match the selected IC-LoRA family")
        if not 0 <= float(item["strength"]) <= 1:
            raise ValueError("Control guide strength must be in [0, 1]")
        if not Path(item["path"]).is_file():
            raise FileNotFoundError(item["path"])
        ingredients = item["control_type"] == "ingredients_reference_sheet"
        if item["kind"] != ("image" if ingredients else "video"):
            raise ValueError("Ingredients requires an image; Union/Motion require video")
        if ingredients and (
            (config.width, config.height) != (768, 448)
            or config.num_frames < 121
            or config.frame_rate != 24
        ):
            raise ValueError("Ingredients requires 768x448, at least 121 frames, and 24 fps")
    if inspect_files:
        from dataclasses import asdict

        from wee_todd_mlx.conditioning_media import inspect_media

        contract = {
            "inputs": [dict(i, id=f"control-{n}", role="control") for n, i in enumerate(inputs)]
        }
        reports = inspect_media({"engine": "ltx23", "config": asdict(config)}, contract)
        for report in reports:
            if report["kind"] == "video" and (
                report["duration_seconds"] * report["fps"] + 0.01 < config.num_frames
            ):
                raise ValueError(
                    "Control video must cover all output frames, including the 8n+1 anchor"
                )


def install_ic_fusion(pipeline, specs, auxiliary_specs=()):
    """Fuse at IC generation entry; the selected topology controls any clean reload."""
    if not callable(getattr(pipeline, "_fuse_loras", None)) or not callable(
        getattr(pipeline, "_reload_clean_transformer", None)
    ):
        raise ValueError("Installed IC pipeline lacks the qualified stage lifecycle")
    reports = []
    auxiliary_reports = []
    pipeline._weetodd_auxiliary_lora_reports = auxiliary_reports

    def fuse(instance):
        combined = (*specs, *auxiliary_specs)
        combined_reports = apply_stack(
            instance.dit,
            combined,
            check_interrupted=getattr(instance, "_weetodd_check_interrupted", None),
        )
        reports[:] = combined_reports[: len(specs)]
        auxiliary_reports[:] = combined_reports[len(specs) :]
        for report, spec in zip(reports, specs, strict=True):
            report.update(
                stages=getattr(instance, "_weetodd_ic_stages", "stage1_only"),
                adapter_family=spec.family,
            )
        for report in auxiliary_reports:
            report.update(
                stages=getattr(instance, "_weetodd_ic_stages", "stage1_only"),
                adapter_role="distillation_helper",
            )

    pipeline._fuse_loras = MethodType(fuse, pipeline)
    return reports
