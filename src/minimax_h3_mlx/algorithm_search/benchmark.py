"""Reproducible MLX microbenchmarks for baseline and candidate algorithms."""

from __future__ import annotations

import math
import platform
import statistics
import time
from collections.abc import Callable, Sequence
from typing import Any

import mlx.core as mx
import numpy as np

from .schema import AlgorithmClass, CandidateResult, ErrorMetrics, TimingStats


def _arrays(value: Any) -> list[Any]:
    if isinstance(value, (tuple, list)):
        return [item for child in value for item in _arrays(child)]
    if isinstance(value, dict):
        return [item for child in value.values() for item in _arrays(child)]
    return [value]


def _synchronize(value: Any) -> None:
    values = _arrays(value)
    if values:
        mx.eval(*values)


def _timing_stats(fn: Callable[[], Any], warmups: int, repetitions: int) -> tuple[TimingStats, Any]:
    if warmups < 0 or repetitions < 1:
        raise ValueError("warmups must be non-negative and repetitions must be positive")
    output = None
    for _ in range(warmups):
        output = fn()
        _synchronize(output)
    samples = []
    for _ in range(repetitions):
        started = time.perf_counter()
        output = fn()
        _synchronize(output)
        samples.append(time.perf_counter() - started)
    return (
        TimingStats(
            repetitions=repetitions,
            warmups=warmups,
            samples_seconds=tuple(samples),
            median_seconds=statistics.median(samples),
            mean_seconds=statistics.mean(samples),
            min_seconds=min(samples),
            max_seconds=max(samples),
            stddev_seconds=statistics.pstdev(samples),
        ),
        output,
    )


def numerical_metrics(baseline: Any, candidate: Any) -> ErrorMetrics:
    """Calculate float32 error metrics over one array or matching nested outputs."""
    baseline_values = _arrays(baseline)
    candidate_values = _arrays(candidate)
    if len(baseline_values) != len(candidate_values):
        raise ValueError("baseline and candidate outputs must have matching structure")
    wanted = []
    got = []
    for left, right in zip(baseline_values, candidate_values, strict=True):
        # NumPy does not understand MLX's bfloat16 buffer descriptor. Cast on
        # the MLX side before crossing the array boundary.
        left_np = np.asarray(left.astype(mx.float32)).reshape(-1)
        right_np = np.asarray(right.astype(mx.float32)).reshape(-1)
        if left_np.shape != right_np.shape:
            raise ValueError("baseline and candidate output shapes must match")
        wanted.append(left_np)
        got.append(right_np)
    a = np.concatenate(wanted) if wanted else np.zeros(1, dtype=np.float32)
    b = np.concatenate(got) if got else np.zeros(1, dtype=np.float32)
    delta = b - a
    a_norm = float(np.linalg.norm(a))
    b_norm = float(np.linalg.norm(b))
    denominator = max(a_norm, 1e-12)
    cosine_denominator = max(a_norm * b_norm, 1e-12)
    return ErrorMetrics(
        max_absolute_error=float(np.max(np.abs(delta))),
        mean_absolute_error=float(np.mean(np.abs(delta))),
        rmse=float(math.sqrt(float(np.mean(delta * delta)))),
        relative_l2_error=float(np.linalg.norm(delta) / denominator),
        cosine_similarity=float(np.clip(np.dot(a, b) / cosine_denominator, -1.0, 1.0)),
    )


def _peak_memory(fn: Callable[[], Any]) -> int | None:
    reset = getattr(mx, "reset_peak_memory", None)
    get_peak = getattr(mx, "get_peak_memory", None)
    if reset is None or get_peak is None:
        return None
    reset()
    output = fn()
    _synchronize(output)
    return int(get_peak())


def benchmark_candidate(
    baseline: Callable[..., Any],
    candidate: Callable[..., Any],
    representative_inputs: Sequence[Any],
    *,
    candidate_id: str,
    operator: str,
    algorithm_class: AlgorithmClass,
    parameters: dict[str, Any] | None = None,
    parent_candidate_id: str | None = None,
    transformation: str = "unspecified",
    error_inputs: Sequence[Any] | None = None,
    error_budget: float = 0.0,
    warmups: int = 2,
    repetitions: int = 5,
    notes: str = "",
) -> CandidateResult:
    """Benchmark a candidate against a synchronized baseline on the same inputs."""
    inputs = tuple(representative_inputs)

    def baseline_fn():
        return baseline(*inputs)

    def candidate_fn():
        return candidate(*inputs)

    baseline_stats, baseline_output = _timing_stats(baseline_fn, warmups, repetitions)
    candidate_stats, candidate_output = _timing_stats(candidate_fn, warmups, repetitions)
    if error_inputs is None:
        error_baseline_output = baseline_output
        error_candidate_output = candidate_output
    else:
        comparison_inputs = tuple(error_inputs)
        error_baseline_output = baseline(*comparison_inputs)
        error_candidate_output = candidate(*comparison_inputs)
        _synchronize(error_baseline_output)
        _synchronize(error_candidate_output)
    errors = numerical_metrics(error_baseline_output, error_candidate_output)
    # Do not charge retained timing/comparison outputs to the isolated peak of
    # either implementation. Representative inputs intentionally stay live.
    baseline_output = None
    candidate_output = None
    error_baseline_output = None
    error_candidate_output = None
    mx.clear_cache()
    baseline_peak = _peak_memory(baseline_fn)
    mx.clear_cache()
    candidate_peak = _peak_memory(candidate_fn)
    dtype = str(getattr(inputs[0], "dtype", "unknown")) if inputs else "unknown"
    shapes = tuple(
        tuple(int(v) for v in value.shape) for value in inputs if hasattr(value, "shape")
    )
    machine = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "mlx": getattr(mx, "__version__", "unknown"),
    }
    return CandidateResult(
        candidate_id=candidate_id,
        operator=operator,
        algorithm_class=algorithm_class,
        parameters=parameters or {},
        baseline=baseline_stats,
        candidate=candidate_stats,
        errors=errors,
        passed_error_budget=errors.relative_l2_error <= error_budget,
        speedup=baseline_stats.median_seconds / max(candidate_stats.median_seconds, 1e-12),
        dtype=dtype,
        input_shapes=shapes,
        device="Metal" if platform.system() == "Darwin" else "unknown",
        machine=machine,
        peak_memory_bytes={
            "baseline": baseline_peak,
            "candidate": candidate_peak,
        },
        parent_candidate_id=parent_candidate_id,
        transformation=transformation,
        notes=notes,
    )
