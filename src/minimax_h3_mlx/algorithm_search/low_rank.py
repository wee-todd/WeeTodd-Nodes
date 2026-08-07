"""Low-rank fitting helpers for offline H3 algorithm-search experiments."""

from __future__ import annotations

import numpy as np


def randomized_right_basis(
    matrix: np.ndarray,
    rank: int,
    *,
    oversample: int = 16,
    seed: int = 0,
) -> np.ndarray:
    """Approximate the leading right-singular basis without a full matrix SVD."""
    if matrix.ndim != 2:
        raise ValueError("matrix must be two-dimensional")
    if rank < 1 or rank > min(matrix.shape):
        raise ValueError("rank must be between one and the smaller matrix dimension")
    width = min(min(matrix.shape), rank + max(0, oversample))
    generator = np.random.default_rng(seed)
    omega = generator.standard_normal((matrix.shape[1], width), dtype=np.float32)
    sample = matrix @ omega
    q, _ = np.linalg.qr(sample, mode="reduced")
    compressed = q.T @ matrix
    _, _, right = np.linalg.svd(compressed, full_matrices=False)
    return np.ascontiguousarray(right[:rank].T, dtype=np.float32)


def fit_reduced_map(
    inputs: np.ndarray,
    targets: np.ndarray,
    input_basis: np.ndarray,
    target_basis: np.ndarray,
) -> np.ndarray:
    """Fit a least-squares map between fixed input and target subspaces."""
    if inputs.shape[0] != targets.shape[0]:
        raise ValueError("inputs and targets must have the same row count")
    reduced_inputs = inputs @ input_basis
    reduced_targets = targets @ target_basis
    mapping, _, _, _ = np.linalg.lstsq(reduced_inputs, reduced_targets, rcond=None)
    return np.ascontiguousarray(mapping, dtype=np.float32)


def supervised_input_basis(
    inputs: np.ndarray,
    targets: np.ndarray,
    target_basis: np.ndarray,
) -> np.ndarray:
    """Select input directions correlated with a fixed target subspace."""
    if inputs.shape[0] != targets.shape[0]:
        raise ValueError("inputs and targets must have the same row count")
    reduced_targets = targets @ target_basis
    cross_covariance = inputs.T @ reduced_targets
    basis, _ = np.linalg.qr(cross_covariance, mode="reduced")
    return np.ascontiguousarray(basis, dtype=np.float32)
