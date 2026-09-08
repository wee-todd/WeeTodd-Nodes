"""Lazy generic LoRA specifications for the ComfyUI-to-MLX boundary."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from wee_todd_mlx.adapter_contract import inspect_adapter

from .preflight import read_safetensors_header


def _adapter_targets(names: tuple[str, ...]) -> set[str]:
    targets = set()
    for name in names:
        for ending in (
            ".lora_A.turbo.weight",
            ".lora_B.turbo.weight",
            ".lora_A.default.weight",
            ".lora_B.default.weight",
            ".lora_A.weight",
            ".lora_B.weight",
            ".lora_down.weight",
            ".lora_up.weight",
            ".lora_a.weight",
            ".lora_b.weight",
        ):
            if name.endswith(ending):
                targets.add(name[: -len(ending)])
    return targets


@dataclass(frozen=True)
class H3LoRASpec:
    """Immutable, header-validated LoRA request that does not load tensor payloads."""

    path: str
    strength: float = 1.0
    profile: str = "auto"
    adaln_input_grid: str | None = None
    qkv_layout: str = "auto"
    start_after_evaluations: int = 0

    def validate(self) -> None:
        path = Path(self.path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"MiniMax H3 LoRA file not found: {path}")
        if path.suffix.lower() != ".safetensors":
            raise ValueError("MiniMax H3 LoRA files must use the `.safetensors` format.")
        if not math.isfinite(self.strength) or not -10.0 <= self.strength <= 10.0:
            raise ValueError("MiniMax H3 LoRA strength must be finite and between -10 and 10.")
        if self.profile not in {"auto", "standard", "turbo"}:
            raise ValueError("MiniMax H3 LoRA profile must be auto, standard, or turbo.")
        if self.qkv_layout not in {"auto", "native_interleaved", "contiguous_qkv"}:
            raise ValueError(
                "MiniMax H3 LoRA QKV layout must be auto, native_interleaved, or contiguous_qkv."
            )
        if self.start_after_evaluations < 0:
            raise ValueError("MiniMax H3 LoRA start_after_evaluations must be zero or greater.")

        header = read_safetensors_header(path)
        # The common contract rejects mixed schemas, malformed ranks, and unknown
        # tensor fields before either the paged or resident engine can allocate.
        contract = inspect_adapter(path, allow_exact_deltas=True)
        names = header.tensor_names
        targets = {str(pair["target"]) for pair in contract["pairs"]}
        if not targets:
            raise ValueError("The selected safetensors file contains no supported LoRA targets.")
        for target in targets:
            a_names = {
                target + ".lora_A.turbo.weight",
                target + ".lora_A.default.weight",
                target + ".lora_A.weight",
                target + ".lora_down.weight",
                target + ".lora_a.weight",
            }
            b_names = {
                target + ".lora_B.turbo.weight",
                target + ".lora_B.default.weight",
                target + ".lora_B.weight",
                target + ".lora_up.weight",
                target + ".lora_b.weight",
            }
            if not a_names.intersection(names) or not b_names.intersection(names):
                raise ValueError(f"LoRA target {target!r} does not contain a complete A/B pair.")
        if self.start_after_evaluations and any("adaln_proj" in target for target in targets):
            raise ValueError(
                "Staged MiniMax H3 LoRA activation does not support AdaLN adapter targets. "
                "Use an adapter without AdaLN targets or apply the LoRA for the full schedule."
            )

        if self.adaln_input_grid is not None:
            grid = Path(self.adaln_input_grid).expanduser()
            if not grid.is_file():
                raise FileNotFoundError(f"MiniMax H3 AdaLN input-grid file not found: {grid}")
            if grid.suffix.lower() != ".safetensors":
                raise ValueError("The MiniMax H3 AdaLN input grid must use safetensors.")
            grid_header = read_safetensors_header(grid)
            if grid_header.tensor_count != 1:
                raise ValueError("The MiniMax H3 AdaLN input grid must contain exactly one tensor.")

    @property
    def resolved_profile(self) -> str:
        if self.profile != "auto":
            return self.profile
        metadata = inspect_adapter(Path(self.path).expanduser(), allow_exact_deltas=True)[
            "metadata"
        ]
        declared = str(
            metadata.get("adapter_profile")
            or metadata.get("profile")
            or metadata.get("distillation_profile")
            or ""
        ).strip().lower()
        if declared in {"turbo", "distilled", "dmd"}:
            return "turbo"
        if declared in {"standard", "base", "quality"}:
            return "standard"
        for key in ("inference_steps", "num_inference_steps", "steps"):
            try:
                if 1 <= int(metadata.get(key, 0)) <= 8:
                    return "turbo"
            except (TypeError, ValueError):
                pass
        # Ambiguous metadata must not let a downloaded filename alter schedule math.
        return "standard"

    @property
    def profile_classification_basis(self) -> str:
        if self.profile != "auto":
            return "explicit user selection"
        metadata = inspect_adapter(Path(self.path).expanduser(), allow_exact_deltas=True)[
            "metadata"
        ]
        if any(
            metadata.get(key) not in (None, "")
            for key in (
                "adapter_profile",
                "profile",
                "distillation_profile",
                "inference_steps",
                "num_inference_steps",
                "steps",
            )
        ):
            return "checkpoint metadata"
        return "metadata ambiguous; source-independent standard default"

    @property
    def tensor_bytes(self) -> int:
        return read_safetensors_header(self.path).tensor_bytes

    @property
    def structural_descriptor(self) -> dict[str, object]:
        report = inspect_adapter(Path(self.path).expanduser(), allow_exact_deltas=True)
        return {
            "contract": report["format"],
            "pair_schemas": report["pair_schemas"],
            "ranks": report["ranks"],
            "target_fingerprint": report["target_fingerprint"],
            "target_count": len(report["pairs"]),
            "exact_delta_count": len(report["exact_deltas"]),
            "declared_scaling": report["declared_scaling"],
        }

    @property
    def resolved_qkv_layout(self) -> str:
        if self.qkv_layout != "auto":
            return self.qkv_layout
        metadata = inspect_adapter(Path(self.path).expanduser(), allow_exact_deltas=True)[
            "metadata"
        ]
        declared = str(metadata.get("qkv_layout") or metadata.get("qkv_fusion") or "").lower()
        if any(marker in declared for marker in ("contiguous", "concat", "block diagonal")):
            return "contiguous_qkv"
        if any(marker in declared for marker in ("interleaved", "per-head")):
            return "native_interleaved"
        return "contiguous_qkv" if self.resolved_profile == "turbo" else "native_interleaved"

    def engine_request(self) -> dict[str, object]:
        return {
            "path": str(Path(self.path).expanduser()),
            "strength": self.strength,
            "qkv_layout": self.resolved_qkv_layout,
            "adaln_input_grid": (
                str(Path(self.adaln_input_grid).expanduser())
                if self.adaln_input_grid is not None
                else None
            ),
            "start_after_evaluations": self.start_after_evaluations,
        }


@dataclass(frozen=True)
class H3LoRAStack:
    """Ordered LoRA stack passed as one stable ComfyUI connection."""

    adapters: tuple[H3LoRASpec, ...] = ()

    def append(self, spec: H3LoRASpec) -> H3LoRAStack:
        spec.validate()
        if len(self.adapters) >= 8:
            raise ValueError("MiniMax H3 supports at most eight LoRAs in one stack.")
        return H3LoRAStack((*self.adapters, spec))

    def validate_for_steps(self, steps: int) -> None:
        evaluations = max(int(steps) - 1, 0)
        for spec in self.adapters:
            spec.validate()
            if spec.start_after_evaluations >= evaluations:
                raise ValueError(
                    "MiniMax H3 LoRA activation must begin before the final transformer "
                    "evaluation."
                )
            active_evaluations = evaluations - spec.start_after_evaluations
            if spec.resolved_profile == "turbo" and active_evaluations < 4:
                raise ValueError(
                    "MiniMax H3 Turbo LoRAs require at least four active transformer "
                    "evaluations after the configured activation point."
                )

    @property
    def has_turbo(self) -> bool:
        return any(spec.resolved_profile == "turbo" for spec in self.adapters)

    @property
    def cache_key(self) -> tuple[H3LoRASpec, ...]:
        return self.adapters

    def engine_requests(self) -> tuple[dict[str, object], ...]:
        return tuple(spec.engine_request() for spec in self.adapters)

    def metadata(self) -> list[dict[str, object]]:
        return [
            {
                "file": Path(spec.path).name,
                "strength": spec.strength,
                "profile": spec.resolved_profile,
                "profile_classification_basis": spec.profile_classification_basis,
                "qkv_layout": spec.resolved_qkv_layout,
                "structural_descriptor": spec.structural_descriptor,
                "tensor_bytes": spec.tensor_bytes,
                "adaln_input_grid": (
                    Path(spec.adaln_input_grid).name if spec.adaln_input_grid else None
                ),
                "start_after_evaluations": spec.start_after_evaluations,
            }
            for spec in self.adapters
        ]
