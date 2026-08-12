"""Bounded, opt-in profiling and representative-activation capture."""

from __future__ import annotations

import json
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import mlx.core as mx


@dataclass(frozen=True)
class CaptureConfig:
    enabled: bool = False
    output_directory: str = "algorithm_search_captures"
    targets: tuple[str, ...] = ()
    blocks: tuple[int, ...] = ()
    profile_blocks: tuple[int, ...] = ()
    evaluation_indices: tuple[int, ...] = ()
    attention_heads: tuple[int, ...] = ()
    max_total_bytes: int = 512 * 1024 * 1024
    profile_regions: bool = False

    def validate(self) -> None:
        if self.max_total_bytes < 1:
            raise ValueError("capture max_total_bytes must be positive")
        if self.enabled and not self.targets and not self.profile_regions:
            raise ValueError("enabled diagnostics require capture targets or region profiling")
        if any(block < 0 for block in self.blocks + self.profile_blocks):
            raise ValueError("capture block indices must be non-negative")
        if any(index < 0 for index in self.evaluation_indices):
            raise ValueError("capture evaluation indices must be non-negative")
        if any(head < 0 for head in self.attention_heads):
            raise ValueError("capture attention head indices must be non-negative")
        if self.enabled and "attention_qkv" in self.targets and not self.attention_heads:
            raise ValueError("attention QKV capture requires at least one selected head")


@dataclass
class RegionMeasurement:
    name: str
    block: int | None
    duration_seconds: float
    output_shapes: list[list[int]]
    output_dtypes: list[str]
    peak_memory_bytes: int | None
    evaluation_index: int | None = None
    timestep: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def _leaves(value: Any) -> list[Any]:
    if isinstance(value, (tuple, list)):
        return [item for child in value for item in _leaves(child)]
    if isinstance(value, dict):
        return [item for child in value.values() for item in _leaves(child)]
    return [value]


class DiagnosticSession:
    """Measure named regions and save only explicitly selected tensors."""

    def __init__(self, config: CaptureConfig, *, hybrid_controller=None) -> None:
        config.validate()
        self.config = config
        self.output_directory = Path(config.output_directory)
        self.measurements: list[RegionMeasurement] = []
        self.captures: list[dict[str, Any]] = []
        self.total_bytes = 0
        self._capture_index = 0
        self.evaluation_index: int | None = None
        self.timestep: float | None = None
        self.audio_timestep: float | None = None
        self.hybrid_controller = hybrid_controller
        self.packed_layout: dict[str, int] | None = None

    @property
    def requires_packed_layout(self) -> bool:
        return self.config.enabled and "attention_qkv" in self.config.targets

    def set_packed_layout(
        self,
        *,
        sequence_rows: int,
        video_indices: mx.array,
        audio_indices: mx.array,
        text_indices: mx.array,
    ) -> None:
        """Record the exact-prefix boundary for selected attention captures."""
        if sequence_rows < 1:
            raise ValueError("packed sequence rows must be positive")
        video = [int(value) for value in video_indices.tolist()]
        audio = [int(value) for value in audio_indices.tolist()]
        text = [int(value) for value in text_indices.tolist()]
        if not audio:
            raise ValueError("H3 sparse-attention diagnostics require generated-audio rows")
        prefix_rows = max(audio) + 1
        if not 0 < prefix_rows < sequence_rows:
            raise ValueError("H3 exact-prefix boundary must be inside the packed sequence")
        condition_video_rows = sum(index < prefix_rows for index in video)
        target_video_rows = sequence_rows - prefix_rows
        if target_video_rows != sum(index >= prefix_rows for index in video):
            raise ValueError("H3 target-video rows must be the contiguous packed tail")
        self.packed_layout = {
            "sequence_rows": sequence_rows,
            "prefix_rows": prefix_rows,
            "text_rows": len(text),
            "condition_video_rows": condition_video_rows,
            "audio_rows": len(audio),
            "target_video_rows": target_video_rows,
        }

    def begin_evaluation(
        self, index: int, *, timestep: float, audio_timestep: float | None = None
    ) -> None:
        """Label subsequent captures and measurements with their denoising evaluation."""
        if index < 0:
            raise ValueError("evaluation index must be non-negative")
        self.evaluation_index = int(index)
        self.timestep = float(timestep)
        self.audio_timestep = None if audio_timestep is None else float(audio_timestep)

    def record_external(
        self,
        name: str,
        duration_seconds: float,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Record already-materialized host or scheduler work outside a diagnostic callable."""
        if not self.config.profile_regions:
            return
        self.measurements.append(
            RegionMeasurement(
                name=name,
                block=None,
                duration_seconds=float(duration_seconds),
                output_shapes=[],
                output_dtypes=[],
                peak_memory_bytes=None,
                evaluation_index=self.evaluation_index,
                timestep=self.timestep,
                metadata=metadata or {},
            )
        )

    def _selected(self, name: str, block: int | None) -> bool:
        if not self.config.enabled or name not in self.config.targets:
            return False
        block_selected = block is None or not self.config.blocks or block in self.config.blocks
        evaluation_selected = (
            not self.config.evaluation_indices
            or self.evaluation_index in self.config.evaluation_indices
        )
        return block_selected and evaluation_selected

    def prepare_block(self, value: Any, block: int) -> None:
        """Materialize inherited lazy work before timing a selected block's first region."""
        if self.config.profile_regions and (
            not self.config.profile_blocks or block in self.config.profile_blocks
        ):
            leaves = [leaf for leaf in _leaves(value) if hasattr(leaf, "shape")]
            if leaves:
                mx.eval(*leaves)

    def try_hybrid_block(self, block, block_index: int, x, **kwargs):
        """Delegate an explicitly configured research-only block substitution."""
        if self.hybrid_controller is None:
            return None
        return self.hybrid_controller.try_apply(
            block,
            block_index,
            x,
            evaluation_index=self.evaluation_index,
            **kwargs,
        )

    def observe_hybrid_block(self, block_index: int, block_input, block_output) -> None:
        if self.hybrid_controller is not None:
            self.hybrid_controller.observe(
                block_index,
                block_input,
                block_output,
                evaluation_index=self.evaluation_index,
            )

    def capture(
        self,
        name: str,
        value: Any,
        *,
        block: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if not self._selected(name, block):
            return
        leaves = [leaf for leaf in _leaves(value) if hasattr(leaf, "shape")]
        mx.eval(*leaves)
        arrays = [
            leaf if type(leaf).__module__.startswith("mlx.") else mx.array(leaf) for leaf in leaves
        ]
        byte_count = sum(int(array.nbytes) for array in arrays)
        if self.total_bytes + byte_count > self.config.max_total_bytes:
            raise MemoryError(
                f"diagnostic capture limit exceeded: {self.total_bytes + byte_count} > "
                f"{self.config.max_total_bytes} bytes"
            )
        self.output_directory.mkdir(parents=True, exist_ok=True)
        evaluation = "none" if self.evaluation_index is None else str(self.evaluation_index)
        stem = f"{self._capture_index:04d}_eval_{evaluation}_{name.replace('.', '_')}_block_{block}"
        path = self.output_directory / f"{stem}.safetensors"
        mx.save_safetensors(
            str(path), {f"tensor_{index}": array for index, array in enumerate(arrays)}
        )
        self._capture_index += 1
        self.total_bytes += byte_count
        self.captures.append(
            {
                "name": name,
                "block": block,
                "evaluation_index": self.evaluation_index,
                "timestep": self.timestep,
                "audio_timestep": self.audio_timestep,
                "path": path.name,
                "bytes": byte_count,
                "shapes": [list(array.shape) for array in arrays],
                "dtypes": [str(array.dtype) for array in arrays],
                "metadata": metadata or {},
            }
        )

    def capture_attention_qkv(
        self,
        query: mx.array,
        key: mx.array,
        value: mx.array,
        *,
        block: int,
    ) -> None:
        """Save post-normalization, post-RoPE Q/K/V for selected heads only."""
        if not self._selected("attention_qkv", block):
            return
        if self.packed_layout is None:
            raise RuntimeError("attention QKV capture requires packed-layout metadata")
        heads = self.config.attention_heads
        if not heads or max(heads) >= int(query.shape[1]):
            raise ValueError("capture attention head index exceeds the model head count")
        selected = mx.array(heads, dtype=mx.int32)
        self.capture(
            "attention_qkv",
            (
                mx.take(query, selected, axis=1),
                mx.take(key, selected, axis=1),
                mx.take(value, selected, axis=1),
            ),
            block=block,
            metadata={
                **self.packed_layout,
                "attention_heads": list(heads),
                "layout": "batch_heads_rows_dim",
                "stage": "post_qk_norm_and_rope",
            },
        )

    def measure(
        self,
        name: str,
        fn,
        *,
        block: int | None = None,
        metadata: dict[str, Any] | None = None,
        capture_as: str | None = None,
    ):
        profile_selected = self.config.profile_regions and (
            block is None or not self.config.profile_blocks or block in self.config.profile_blocks
        )
        if not profile_selected:
            result = fn()
            if capture_as is not None:
                self.capture(capture_as, result, block=block)
            return result
        reset = getattr(mx, "reset_peak_memory", None)
        get_peak = getattr(mx, "get_peak_memory", None)
        if reset is not None:
            reset()
        started = time.perf_counter()
        result = fn()
        leaves = [leaf for leaf in _leaves(result) if hasattr(leaf, "shape")]
        if leaves:
            mx.eval(*leaves)
        elapsed = time.perf_counter() - started
        self.measurements.append(
            RegionMeasurement(
                name=name,
                block=block,
                duration_seconds=elapsed,
                output_shapes=[list(leaf.shape) for leaf in leaves],
                output_dtypes=[str(leaf.dtype) for leaf in leaves],
                peak_memory_bytes=int(get_peak()) if get_peak is not None else None,
                evaluation_index=self.evaluation_index,
                timestep=self.timestep,
                metadata=metadata or {},
            )
        )
        if capture_as is not None:
            self.capture(capture_as, result, block=block)
        return result

    def inactive_context(self):
        """Small helper for callers that need a context-manager shaped no-op."""
        return nullcontext()

    def write_metadata(self) -> Path:
        self.output_directory.mkdir(parents=True, exist_ok=True)
        target = self.output_directory / "metadata.json"
        target.write_text(
            json.dumps(
                {
                    "config": asdict(self.config),
                    "total_capture_bytes": self.total_bytes,
                    "captures": self.captures,
                    "measurements": [asdict(item) for item in self.measurements],
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        return target
