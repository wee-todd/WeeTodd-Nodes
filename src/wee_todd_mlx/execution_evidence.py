"""Generation-scoped evidence that an optional optimization actually executed."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ExecutionEvidence:
    """Record runtime dispatch facts separately from the requested configuration."""

    requested_backend: str
    resolved_backend: str
    scope: str
    total_calls: int = 0
    eligible_calls: int = 0
    executed_calls: int = 0
    dense_policy_calls: int = 0
    fallback_counts: dict[str, int] = field(default_factory=dict)
    observed_dtypes: set[str] = field(default_factory=set)
    observed_shapes: set[tuple[int, ...]] = field(default_factory=set)
    work_units_processed: int = 0
    work_units_avoided: int = 0

    def record_call(self) -> None:
        self.total_calls += 1

    def record_observed(self, *, dtype, shape) -> None:
        self.observed_dtypes.add(str(dtype))
        self.observed_shapes.add(tuple(int(value) for value in shape))

    def record_dense_policy(self) -> None:
        self.dense_policy_calls += 1

    def record_eligible(self) -> None:
        self.eligible_calls += 1

    def record_executed(self, *, work_units: int = 1, avoided_units: int = 0) -> None:
        self.executed_calls += 1
        self.work_units_processed += int(work_units)
        self.work_units_avoided += int(avoided_units)

    def record_fallback(self, reason: str) -> None:
        key = str(reason).strip() or "unspecified"
        self.fallback_counts[key] = self.fallback_counts.get(key, 0) + 1

    def snapshot(self) -> dict[str, object]:
        return {
            "requested_backend": self.requested_backend,
            "resolved_backend": self.resolved_backend,
            "scope": self.scope,
            "total_calls": self.total_calls,
            "eligible_calls": self.eligible_calls,
            "executed_calls": self.executed_calls,
            "dense_policy_calls": self.dense_policy_calls,
            "fallback_calls": sum(self.fallback_counts.values()),
            "fallback_counts": dict(sorted(self.fallback_counts.items())),
            "observed_dtypes": sorted(self.observed_dtypes),
            "observed_shapes": [list(shape) for shape in sorted(self.observed_shapes)],
            "work_units_processed": self.work_units_processed,
            "work_units_avoided": self.work_units_avoided,
            "requested_but_not_executed": bool(
                self.requested_backend != "disabled" and self.executed_calls == 0
            ),
        }


def require_executed(report: dict[str, object]) -> None:
    """Reject benchmark evidence that requested an optimization but executed no work."""

    requested = str(
        report.get("requested_backend", "sol_attention" if report.get("enabled") else "disabled")
    )
    executed = int(report.get("executed_calls", report.get("sparse_kernel_calls", 0)))
    if requested != "disabled" and executed == 0:
        fallbacks = report.get("fallback_counts") or {}
        raise RuntimeError(
            f"Optimization {requested!r} executed zero calls; fallback counts: {fallbacks}."
        )


__all__ = ["ExecutionEvidence", "require_executed"]
