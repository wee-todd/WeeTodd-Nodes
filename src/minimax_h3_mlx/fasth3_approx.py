"""Opt-in FastH3 layer thinning and target-video token pairing.

Both policies are generative approximations.  They are isolated from the exact VSA routing path,
preserve every multimodal-prefix row, and retain enough metadata for a render to disclose exactly
which transformer work was skipped or shortened.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import mlx.core as mx
import numpy as np

from wee_todd_mlx.numpy_import import adopt_numpy_array


@dataclass(frozen=True)
class FastH3ApproximationConfig:
    """Explicit approximation controls; the default is behavior-preserving."""

    active_layers: int | None = None
    protect_first_layers: int = 2
    protect_last_layers: int = 2
    pair_target_video: bool = False
    pair_start_layer: int = 4
    pair_end_layer: int = 30

    def validate(self, num_layers: int = 50) -> None:
        integer_fields = {
            "num_layers": num_layers,
            "protect_first_layers": self.protect_first_layers,
            "protect_last_layers": self.protect_last_layers,
            "pair_start_layer": self.pair_start_layer,
            "pair_end_layer": self.pair_end_layer,
        }
        if self.active_layers is not None:
            integer_fields["active_layers"] = self.active_layers
        for name, value in integer_fields.items():
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"FastH3 {name} must be an integer.")
        if num_layers < 1:
            raise ValueError("FastH3 approximation requires a positive layer count.")
        if self.active_layers is not None and not 1 <= self.active_layers <= num_layers:
            raise ValueError(f"active_layers must be between 1 and {num_layers}.")
        if min(self.protect_first_layers, self.protect_last_layers) < 0:
            raise ValueError("Protected FastH3 layer counts must be non-negative.")
        if self.protect_first_layers + self.protect_last_layers > num_layers:
            raise ValueError("Protected FastH3 layer ranges overlap.")
        if self.pair_target_video and not (
            0 <= self.pair_start_layer < self.pair_end_layer <= num_layers
        ):
            raise ValueError(
                "Target-video pairing requires 0 <= pair_start_layer < pair_end_layer "
                f"<= {num_layers}."
            )

    @property
    def enabled(self) -> bool:
        return self.active_layers is not None or self.pair_target_video


def modulation_layer_scores(modulation_cache) -> tuple[float, ...]:
    """Rank blocks by schedule-wide AdaLN residual-gate magnitude.

    Gate tensors 2 and 5 control the attention and MLP residual contributions respectively.
    The cache spans every timestep and modality used by the sampler, so this score is stable for
    the entire generation rather than being chosen independently at each denoising evaluation.
    """

    scores = []
    for table in modulation_cache.tables:
        if len(table) != 6:
            raise ValueError("FastH3 layer ranking requires six AdaLN modulation tensors.")
        score = mx.mean(mx.abs(table[2]).astype(mx.float32)) + mx.mean(
            mx.abs(table[5]).astype(mx.float32)
        )
        value = float(score.item())
        if not math.isfinite(value):
            raise ValueError("FastH3 layer ranking requires finite AdaLN residual gates.")
        scores.append(value)
    return tuple(scores)


def select_active_layers(
    config: FastH3ApproximationConfig,
    num_layers: int,
    modulation_cache=None,
) -> tuple[int, ...]:
    """Select the protected boundaries plus the highest schedule-wide gate scores."""

    config.validate(num_layers)
    requested = num_layers if config.active_layers is None else config.active_layers
    protected = set(range(config.protect_first_layers))
    protected.update(range(num_layers - config.protect_last_layers, num_layers))
    protected.add(0)  # Paged execution and cache telemetry require an evaluated block zero.
    if requested < len(protected):
        raise ValueError(
            f"active_layers={requested} is smaller than the {len(protected)} protected layers."
        )
    if requested == num_layers:
        return tuple(range(num_layers))
    if modulation_cache is None:
        raise ValueError(
            "FastH3 layer thinning requires the schedule-wide AdaLN modulation cache so layer "
            "selection is deterministic and evidence-based."
        )
    scores = modulation_layer_scores(modulation_cache)
    if len(scores) != num_layers:
        raise ValueError(f"AdaLN cache contains {len(scores)} block tables; expected {num_layers}.")
    candidates = sorted(
        (index for index in range(num_layers) if index not in protected),
        key=lambda index: (-scores[index], index),
    )
    protected.update(candidates[: requested - len(protected)])
    return tuple(sorted(protected))


@dataclass(frozen=True)
class TargetVideoPairingGeometry:
    """One horizontal 2:1 pairing map for a contiguous target-video suffix."""

    prefix_rows: int
    video_grid: tuple[int, int, int]
    reduced_rows: mx.array
    expand_rows: mx.array
    group_counts: mx.array

    @property
    def full_rows(self) -> int:
        return self.prefix_rows + math.prod(self.video_grid)

    @property
    def paired_video_rows(self) -> int:
        t, h, w = self.video_grid
        return t * h * math.ceil(w / 2)


def build_target_video_pairing(
    prefix_rows: int,
    video_grid: tuple[int, int, int],
) -> TargetVideoPairingGeometry:
    """Build deterministic horizontal pairs while leaving the prefix byte-for-byte addressable."""

    if prefix_rows < 0:
        raise ValueError("FastH3 target-video prefix size must be non-negative.")
    if len(video_grid) != 3 or any(int(value) < 1 for value in video_grid):
        raise ValueError("FastH3 target-video pairing requires a positive (t, h, w) grid.")
    t_size, h_size, w_size = (int(value) for value in video_grid)
    reduced_rows = list(range(prefix_rows))
    expand_rows = list(range(prefix_rows))
    counts = [1] * prefix_rows
    next_reduced = prefix_rows
    for t_index in range(t_size):
        for h_index in range(h_size):
            row_base = prefix_rows + (t_index * h_size + h_index) * w_size
            for w_index in range(0, w_size, 2):
                reduced_rows.append(row_base + w_index)
                count = min(2, w_size - w_index)
                counts.append(count)
                expand_rows.extend([next_reduced] * count)
                next_reduced += 1
    return TargetVideoPairingGeometry(
        prefix_rows=prefix_rows,
        video_grid=(t_size, h_size, w_size),
        reduced_rows=adopt_numpy_array(np.asarray(reduced_rows, dtype=np.int32)),
        expand_rows=adopt_numpy_array(np.asarray(expand_rows, dtype=np.int32)),
        group_counts=adopt_numpy_array(np.asarray(counts, dtype=np.float32)),
    )


@dataclass
class TargetVideoPairingState:
    full_baseline: mx.array
    reduced_baseline: mx.array
    geometry: TargetVideoPairingGeometry


def pair_target_video_rows(
    x: mx.array,
    adaln_indices: mx.array,
    rotary: tuple[mx.array, mx.array],
    geometry: TargetVideoPairingGeometry,
) -> tuple[mx.array, mx.array, tuple[mx.array, mx.array], TargetVideoPairingState]:
    """Average target-video pairs and retain the full residual bypass for later restoration."""

    if int(x.shape[1]) != geometry.full_rows:
        raise ValueError("FastH3 target-video pairing geometry does not match the packed rows.")
    rows = geometry.reduced_rows
    summed = (
        mx.zeros((x.shape[0], int(rows.shape[0]), x.shape[2]), dtype=x.dtype)
        .at[:, geometry.expand_rows, :]
        .add(x)
    )
    reduced = summed / geometry.group_counts.astype(x.dtype)[None, :, None]
    reduced_adaln = adaln_indices[rows]
    reduced_rotary = tuple(table[rows] for table in rotary)
    state = TargetVideoPairingState(x, reduced, geometry)
    return reduced, reduced_adaln, reduced_rotary, state


def restore_target_video_rows(
    reduced: mx.array,
    state: TargetVideoPairingState,
) -> mx.array:
    """Add the repeated reduced-stack update to each retained full-resolution row."""

    if int(reduced.shape[1]) != int(state.reduced_baseline.shape[1]):
        raise ValueError("FastH3 reduced target-video state changed row count before restoration.")
    update = reduced - state.reduced_baseline
    return state.full_baseline + update[:, state.geometry.expand_rows, :]


def approximation_report(
    config: FastH3ApproximationConfig,
    active_layers: tuple[int, ...],
    num_layers: int,
    pairing: TargetVideoPairingGeometry | None,
) -> dict[str, object]:
    report = asdict(config)
    report.update(
        {
            "status": "generatively_approximate" if config.enabled else "disabled",
            "active_layer_indices": list(active_layers),
            "skipped_layer_indices": [
                index for index in range(num_layers) if index not in active_layers
            ],
            "selection_policy": "schedule_wide_adaln_gate_magnitude_v1",
            "layer_scope": "joint_video_audio_and_text_rows",
            "executed_layers": len(active_layers),
            "skipped_layers": num_layers - len(active_layers),
            "full_sequence_rows": pairing.full_rows if pairing is not None else None,
            "paired_sequence_rows": (
                pairing.prefix_rows + pairing.paired_video_rows if pairing is not None else None
            ),
            "prefix_rows_preserved": pairing.prefix_rows if pairing is not None else None,
        }
    )
    return report


__all__ = [
    "FastH3ApproximationConfig",
    "TargetVideoPairingGeometry",
    "TargetVideoPairingState",
    "approximation_report",
    "build_target_video_pairing",
    "modulation_layer_scores",
    "pair_target_video_rows",
    "restore_target_video_rows",
    "select_active_layers",
]
