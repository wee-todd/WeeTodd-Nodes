"""Independent H3 temporal expansion planning. No MLX or ComfyUI imports."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class MotionSettings:
    enabled: bool = False
    mode: str = "adaptive"
    strength: float = 0.5
    maxHold: int = 2
    sensitivity: float = 0.5
    seed: int = 42
    maxFrames: int = 345
    evaluations: int | None = None

    def validate(self):
        if type(self.enabled) is not bool or self.mode not in {"adaptive", "uniform"}:
            raise ValueError("Choose adaptive or uniform Motion Fidelity.")
        if not math.isfinite(self.strength) or not 0 < self.strength <= 1:
            raise ValueError("Motion Fidelity strength must be greater than zero and at most one.")
        if not math.isfinite(self.sensitivity) or not 0 <= self.sensitivity <= 1:
            raise ValueError("Motion sensitivity must be between zero and one.")
        if type(self.maxHold) is not int or not 2 <= self.maxHold <= 4:
            raise ValueError("Motion expansion must be 2–4 frames.")
        if type(self.maxFrames) is not int or not 73 <= self.maxFrames <= 345:
            raise ValueError("Expanded H3 clips must fit a 73–345 frame budget.")
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("Motion seed must be an unsigned 32-bit integer.")
        if self.evaluations is not None and (
            type(self.evaluations) is not int or not 1 <= self.evaluations <= 64
        ):
            raise ValueError("Motion refinement evaluations must be an integer from 1 to 64.")


def aligned_frames(count):
    return count + (5 - count) % 17


def latent_index(frame):
    """H3's 17 pixel frames occupy five latents with spans 1,4,4,4,4."""
    group, offset = divmod(frame, 17)
    return group * 5 + (0 if offset == 0 else 1 + (offset - 1) // 4)


def plan_motion(frame_count, settings, latents=None):
    settings.validate()
    if type(frame_count) is not int or not 60 <= frame_count <= 345:
        raise ValueError("Motion Fidelity requires 60–345 source frames at native 24 fps.")
    scores = np.zeros(frame_count, dtype=np.float64)
    if settings.mode == "uniform":
        holds = np.full(frame_count, settings.maxHold, dtype=np.int64)
    else:
        values = np.asarray(latents, dtype=np.float32)
        if values.ndim != 5 or values.shape[2] <= latent_index(frame_count - 1):
            raise ValueError("Motion analysis requires the complete H3 video latent sequence.")
        if not np.isfinite(values).all():
            raise ValueError("Non-finite source latents cannot be analyzed.")
        jerk = np.mean(np.abs(np.diff(values, n=3, axis=2)), axis=(0, 1, 3, 4))
        jerk = np.pad(jerk, (2, 1), mode="edge")
        # Normalize each temporal VAE phase independently, so its fixed cadence is not
        # mistaken for an action burst. This is a heuristic, not an artifact classifier.
        baseline = np.array([np.median(jerk[i::5]) for i in range(5)])
        residual = np.maximum(0, jerk - baseline[np.arange(len(jerk)) % 5])
        scale = float(np.max(residual))
        if scale > 1e-6:
            scores = residual[[latent_index(i) for i in range(frame_count)]] / scale
        threshold = 0.85 - 0.7 * settings.sensitivity
        holds = 1 + np.ceil(
            np.clip((scores - threshold) / (1 - threshold), 0, 1) * (settings.maxHold - 1)
        ).astype(np.int64)
        # Bound abrupt rate changes to one hold per source frame in either direction.
        for i in range(1, frame_count):
            holds[i] = max(holds[i], holds[i - 1] - 1)
        for i in range(frame_count - 2, -1, -1):
            holds[i] = max(holds[i], holds[i + 1] - 1)
    expanded = int(sum(holds))
    padded = aligned_frames(expanded)
    if padded > settings.maxFrames:
        raise ValueError(
            f"Expansion needs {padded} frames; budget is {settings.maxFrames}. "
            "Reduce expansion/sensitivity or split the clip."
        )
    recovery = np.cumsum(np.r_[0, holds[:-1]]).tolist()
    return {
        "format": "weetodd-motion-plan-v1",
        "refinementSchedule": "explicit-video-noise-v1",
        "settings": asdict(settings),
        "sourceFrames": frame_count,
        "expandedFrames": expanded,
        "paddedFrames": padded,
        "holds": holds.tolist(),
        "recovery": recovery,
        "scores": scores.round(5).tolist(),
        "noop": bool(np.all(holds == 1)),
        "fps": 24,
        "expandedSeconds": padded / 24,
    }


def expansion_indices(plan):
    indices = np.repeat(np.arange(plan["sourceFrames"]), plan["holds"])
    return np.pad(indices, (0, plan["paddedFrames"] - len(indices)), mode="edge")


def audio_filter(plan):
    """Pitch-preserving piecewise holds, aligned using cumulative sample boundaries."""
    holds = plan["holds"]
    runs = []
    start = 0
    for end in range(1, len(holds) + 1):
        if end == len(holds) or holds[end] != holds[start]:
            runs.append((start, end, holds[start]))
            start = end
    labels = "".join(f"[s{i}]" for i in range(len(runs)))
    filters = [f"[0:a]asplit={len(runs)}{labels}"]
    expanded = 0
    for i, (start, end, hold) in enumerate(runs):
        count = (end - start) * hold
        samples = round((expanded + count) * 32000 / 24) - round(expanded * 32000 / 24)
        tempo = (
            "atempo=0.5,atempo=0.5"
            if hold == 4
            else "atempo=0.5,atempo=0.666666666667"
            if hold == 3
            else f"atempo={1 / hold:.12f}"
        )
        filters.append(
            f"[s{i}]atrim=start_sample={round(start * 32000 / 24)}:"
            f"end_sample={round(end * 32000 / 24)},asetpts=PTS-STARTPTS,{tempo},"
            f"apad,atrim=end_sample={samples}[a{i}]"
        )
        expanded += count
    filters.append(
        "".join(f"[a{i}]" for i in range(len(runs)))
        + f"concat=n={len(runs)}:v=0:a=1,apad,"
        + f"atrim=end_sample={round(plan['paddedFrames'] * 32000 / 24)}[out]"
    )
    return ";".join(filters)


def motion_lora_stack(recipe):
    """Construct and qualify the optional standard full-schedule repair stack."""
    from wee_todd_nodes.lora import H3LoRAStack

    stack = H3LoRAStack.from_recipe(recipe)
    stack.validate_for_motion(recipe.get("config", {}).get("steps", 0))
    return stack


def validate_recipe(recipe):
    if recipe.get("engine") != "h3" or recipe.get("components", {}).get("task") != "t2va":
        raise ValueError("Motion Fidelity currently requires a plain H3 T2VA repair recipe.")
    unsupported = (
        "attention",
        "fastvideo",
        "vdn",
        "conditioning",
        "reference_images",
        "easycache",
        "blockcache",
        "trajectory_forecast",
        "sol_attention",
        "hires_fix",
    )
    if any(recipe.get(key) for key in unsupported) or recipe["components"].get("fun_controlnet"):
        raise ValueError("Motion Fidelity does not yet support conditioned or accelerated recipes.")
    if recipe.get("config", {}).get("steps", 0) < 16:
        raise ValueError("Motion Fidelity requires a full H3 recipe with at least 16 steps.")
    motion_lora_stack(recipe)
    import json
    from pathlib import Path

    transformer = Path(
        recipe["components"].get("transformer") or recipe["components"]["checkpoint"]
    )
    folder = transformer if transformer.is_dir() else transformer.parent
    for name in ("config.json", "paged_manifest.json", "quant_config.json"):
        metadata_file = folder / name
        if metadata_file.is_file():
            metadata = json.loads(metadata_file.read_text())
            provenance = " ".join(
                str(metadata.get(k, "")) for k in ("source", "model", "source_model")
            )
            sampling = metadata.get("sampling") or {}
            if (
                metadata.get("vsa_gate")
                or any(x in provenance.lower() for x in ("fasth3", "fastvideo", "vdn", "turbo"))
                or (sampling.get("transformer_evaluations", 100) < 15)
            ):
                raise ValueError(
                    "Distilled/FastH3 checkpoints are not qualified for Motion Fidelity."
                )
    from wee_todd_mlx.headless_preflight import preflight_recipe

    return preflight_recipe(recipe)
