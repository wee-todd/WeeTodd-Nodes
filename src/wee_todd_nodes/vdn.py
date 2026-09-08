"""Lazy, header-only contracts for OpenVDN's MiniMax-H3 checkpoints."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .preflight import read_safetensors_header

VDN_REPOSITORY_ID = "OpenVDN/vdn-minimax-h3"
VDN_MODEL_SPEC_FORMAT = 2
VDN_STAGES = {
    "VDN-H3 8-step (recommended)": ("stage-dmd-step-250", 9, True),
    "VDN-H3 50-step": ("stage-b-step-2000", 51, False),
}


def vdn_branch_shapes() -> dict[str, tuple[int, ...]]:
    """Exact 50-block branch tensor contract of the released H3 checkpoints."""
    block = {
        "linear_attention.alpha.A_log": (56,),
        "linear_attention.alpha.down.weight": (128, 5376),
        "linear_attention.alpha.dt_bias": (7168,),
        "linear_attention.alpha.up.weight": (7168, 128),
        "linear_attention.beta_proj.weight": (56, 5376),
        "linear_attention.norm.weight": (128,),
        "linear_attention.output_gate.down.weight": (128, 5376),
        "linear_attention.output_gate.up.bias": (7168,),
        "linear_attention.output_gate.up.weight": (7168, 128),
        "linear_attention.short_conv.k_sp.weight": (7168, 1, 5, 5),
        "linear_attention.short_conv.k_tm.weight": (7168, 1, 5),
        "linear_attention.short_conv.v_sp.weight": (7168, 1, 5, 5),
        "linear_attention.short_conv.v_tm.weight": (7168, 1, 5),
        "softmax_gate.up.bias": (56,),
        "softmax_gate.up.weight": (56, 5376),
        "to_out_linear.weight": (5376, 7168),
    }
    return {
        f"transformer_blocks.{index}.attn.{name}": shape
        for index in range(50)
        for name, shape in block.items()
    }


@dataclass(frozen=True)
class H3VDNSpec:
    """One deferred VDN hybrid-attention checkpoint selected by a ComfyUI graph."""

    repository: str
    checkpoint: str
    model_spec: str
    linear_branch: str
    default_adapter: str
    turbo_adapter: str | None
    schedule_points: int
    stage: str
    inference_backend: str = "verified"

    def validate(self) -> None:
        if self.inference_backend not in {"reference", "verified", "indexed_experimental"}:
            raise ValueError("Unknown VDN inference backend.")
        root = Path(self.repository).expanduser().resolve()
        checkpoint = Path(self.checkpoint).expanduser().resolve()
        try:
            checkpoint.relative_to(root)
        except ValueError as exc:
            raise ValueError("VDN checkpoint must remain inside the selected repository.") from exc
        for label, value in (
            ("model specification", self.model_spec),
            ("linear branch", self.linear_branch),
            ("default adapter", self.default_adapter),
        ):
            if not Path(value).is_file():
                raise FileNotFoundError(f"VDN {label} not found: {value}")
        if self.turbo_adapter is not None and not Path(self.turbo_adapter).is_file():
            raise FileNotFoundError(f"VDN Turbo adapter not found: {self.turbo_adapter}")

        payload = json.loads(Path(self.model_spec).read_text(encoding="utf-8"))
        if payload.get("format_version") != VDN_MODEL_SPEC_FORMAT:
            raise ValueError(
                f"Unsupported VDN model_spec format {payload.get('format_version')!r}; "
                f"expected {VDN_MODEL_SPEC_FORMAT}."
            )
        base = payload.get("base", {})
        if base.get("class_name") != "MiniMaxH3Transformer3DModel":
            raise ValueError("VDN model_spec does not target MiniMax H3.")
        transforms = payload.get("transforms", [])
        hybrid = [item for item in transforms if item.get("type") == "hybrid_attention"]
        if len(hybrid) != 1 or hybrid[0].get("version") != 2:
            raise ValueError("VDN model_spec must contain one hybrid_attention v2 transform.")
        config = hybrid[0].get("config", {})
        linear = config.get("linear_attention", {})
        softmax = config.get("softmax_attention", {})
        if (
            config.get("anchor_frames") != "both"
            or config.get("enable_softmax_gate") is not True
            or linear.get("delta_rule") != "vdn_solve"
            or linear.get("linear_head_dim") != 128
            or linear.get("enable_text_state") is not True
            or linear.get("bridge") != "alpha"
            or linear.get("a_fp32") is not True
            or linear.get("short_conv", {}).get("targets") != ["k", "v"]
            or softmax.get("chunk") != 5
            or softmax.get("radius") != 1
        ):
            raise ValueError("VDN hybrid-attention settings differ from the supported release.")

        header = read_safetensors_header(self.linear_branch)
        expected = vdn_branch_shapes()
        missing = sorted(expected.keys() - header.tensor_shapes.keys())
        extra = sorted(header.tensor_shapes.keys() - expected.keys())
        if missing or extra:
            raise ValueError(
                f"VDN linear-branch checkpoint is incomplete or incompatible: "
                f"missing={missing[:4]}, unexpected={extra[:4]}."
            )
        for name, shape in expected.items():
            if header.tensor_shapes[name] != shape:
                raise ValueError(
                    f"VDN tensor {name} has shape {header.tensor_shapes[name]}; expected {shape}."
                )
        if not set(header.dtypes) <= {"BF16", "F16", "F32"}:
            raise ValueError("VDN branch tensors must use BF16, F16, or F32 weights.")

    def validate_sampling(self, config, loras) -> None:
        """Prevent a partially wired graph from silently running a different model."""
        if config.steps != self.schedule_points or config.sampling_method != "euler":
            raise ValueError("VDN-H3 requires the selected checkpoint's Euler sampling config.")
        expected = [self.default_adapter]
        if self.turbo_adapter is not None:
            expected.append(self.turbo_adapter)
        actual = list(loras.adapters) if loras is not None else []
        if [str(Path(item.path).resolve()) for item in actual] != expected or any(
            item.strength != 1.0 or item.start_after_evaluations != 0 for item in actual
        ):
            raise ValueError(
                "VDN-H3 requires the complete, unmodified LoRA stack from VDN Checkpoint. "
                "Connect its loras output to H3 Sample."
            )

    @property
    def cache_key(self) -> tuple:
        return (self.checkpoint, self.linear_branch, self.schedule_points, self.inference_backend)

    def engine_request(self) -> dict[str, object]:
        return {
            "checkpoint": self.checkpoint,
            "model_spec": self.model_spec,
            "linear_branch": self.linear_branch,
            "stage": self.stage,
            "inference_backend": self.inference_backend,
        }


def resolve_vdn_spec(repository: str | Path, stage_label: str) -> H3VDNSpec:
    """Resolve one downloaded Hugging Face repository without loading tensor payloads."""
    try:
        stage, schedule_points, has_turbo = VDN_STAGES[stage_label]
    except KeyError as exc:
        raise ValueError(f"Unknown VDN-H3 stage: {stage_label!r}.") from exc
    root = Path(repository).expanduser().resolve()
    checkpoint = root / stage
    spec = H3VDNSpec(
        repository=str(root),
        checkpoint=str(checkpoint),
        model_spec=str(checkpoint / "model_spec.json"),
        linear_branch=str(checkpoint / "linear_branch" / "model.safetensors"),
        default_adapter=str(checkpoint / "adapters" / "default" / "adapter_model.safetensors"),
        turbo_adapter=(
            str(checkpoint / "adapters" / "turbo" / "adapter_model.safetensors")
            if has_turbo
            else None
        ),
        schedule_points=schedule_points,
        stage=stage,
    )
    spec.validate()
    return spec
