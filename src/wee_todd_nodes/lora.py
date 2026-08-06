"""Lazy generic LoRA specifications for the ComfyUI-to-MLX boundary."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from .preflight import read_safetensors_header


def _adapter_targets(names: tuple[str, ...]) -> set[str]:
    targets = set()
    for name in names:
        for ending in (
            ".lora_A.weight",
            ".lora_B.weight",
            ".lora_down.weight",
            ".lora_up.weight",
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

        header = read_safetensors_header(path)
        names = header.tensor_names
        targets = _adapter_targets(names)
        if not targets:
            raise ValueError("The selected safetensors file contains no supported LoRA targets.")
        for target in targets:
            a_names = {
                target + ".lora_A.weight",
                target + ".lora_down.weight",
            }
            b_names = {
                target + ".lora_B.weight",
                target + ".lora_up.weight",
            }
            if not a_names.intersection(names) or not b_names.intersection(names):
                raise ValueError(f"LoRA target {target!r} does not contain a complete A/B pair.")

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
        return "turbo" if "turbo" in Path(self.path).name.lower() else "standard"

    @property
    def tensor_bytes(self) -> int:
        return read_safetensors_header(self.path).tensor_bytes

    def engine_request(self) -> dict[str, object]:
        return {
            "path": str(Path(self.path).expanduser()),
            "strength": self.strength,
            "adaln_input_grid": (
                str(Path(self.adaln_input_grid).expanduser())
                if self.adaln_input_grid is not None
                else None
            ),
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
        for spec in self.adapters:
            spec.validate()
            if spec.resolved_profile == "turbo" and steps < 5:
                raise ValueError(
                    "MiniMax H3 Turbo LoRAs require at least five requested schedule points, "
                    "which produce four transformer evaluations."
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
                "tensor_bytes": spec.tensor_bytes,
                "adaln_input_grid": (
                    Path(spec.adaln_input_grid).name if spec.adaln_input_grid else None
                ),
            }
            for spec in self.adapters
        ]
