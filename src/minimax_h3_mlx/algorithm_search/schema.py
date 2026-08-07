"""Machine-readable contracts for H3 algorithm-search experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any


class AlgorithmClass(StrEnum):
    """Required disclosure class for every candidate transformation."""

    EXACT = "exact"
    NUMERICALLY_APPROXIMATE = "numerically_approximate"
    GENERATIVELY_APPROXIMATE = "generatively_approximate"


@dataclass(frozen=True)
class OperationRecord:
    """One expensive or structurally important operation in an H3 evaluation."""

    case: str
    module: str
    block: int | None
    operation_type: str
    input_shapes: tuple[tuple[int, ...], ...]
    output_shapes: tuple[tuple[int, ...], ...]
    weight_shapes: tuple[tuple[int, ...], ...] = ()
    dtype: str = "bfloat16"
    approximate_flops: int | None = None
    weight_parameters: int = 0
    weight_constant: bool = True
    shared_input_group: str | None = None
    result_reused: bool = False
    frequency: str = "every_diffusion_step"
    modalities: tuple[str, ...] = ("video", "text", "audio")
    materialization: str = "lazy_unless_profiled"
    temporary_bytes_estimate: int | None = None
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TimingStats:
    repetitions: int
    warmups: int
    samples_seconds: tuple[float, ...]
    median_seconds: float
    mean_seconds: float
    min_seconds: float
    max_seconds: float
    stddev_seconds: float


@dataclass(frozen=True)
class ErrorMetrics:
    max_absolute_error: float
    mean_absolute_error: float
    rmse: float
    relative_l2_error: float
    cosine_similarity: float


@dataclass(frozen=True)
class CandidateResult:
    """Serializable outcome of one baseline/candidate comparison."""

    candidate_id: str
    operator: str
    algorithm_class: AlgorithmClass
    parameters: dict[str, Any]
    baseline: TimingStats
    candidate: TimingStats
    errors: ErrorMetrics
    passed_error_budget: bool
    speedup: float
    dtype: str
    input_shapes: tuple[tuple[int, ...], ...]
    device: str
    machine: dict[str, Any]
    peak_memory_bytes: dict[str, int | None] = field(default_factory=dict)
    parent_candidate_id: str | None = None
    transformation: str = "unspecified"
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["algorithm_class"] = self.algorithm_class.value
        return value
