"""Opt-in research tools for H3 inference algorithm experiments.

This package is intentionally not imported by the normal ComfyUI adapter or inference path.
"""

from .benchmark import benchmark_candidate, numerical_metrics
from .capture import CaptureConfig, DiagnosticSession
from .inventory import InventoryCase, build_operation_inventory
from .results import ExperimentStore
from .schema import AlgorithmClass, CandidateResult, OperationRecord

__all__ = [
    "AlgorithmClass",
    "CandidateResult",
    "CaptureConfig",
    "DiagnosticSession",
    "ExperimentStore",
    "InventoryCase",
    "OperationRecord",
    "benchmark_candidate",
    "build_operation_inventory",
    "numerical_metrics",
]
