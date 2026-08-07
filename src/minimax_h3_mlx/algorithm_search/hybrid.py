"""Accounting helpers for selective cross-timestep hybrid blocks."""

from __future__ import annotations


def trajectory_state_bytes(
    *,
    predicted_rows: int,
    hidden_size: int,
    blocks: int,
    bytes_per_value: int = 2,
    tensors_per_block: int = 2,
) -> int:
    """Return steady bytes for previous input/output state retained at every block."""
    if min(predicted_rows, hidden_size, blocks, bytes_per_value, tensors_per_block) < 1:
        raise ValueError("trajectory-state dimensions must be positive")
    return predicted_rows * hidden_size * blocks * bytes_per_value * tensors_per_block


def hybrid_theoretical_fraction(
    *, total_rows: int, exact_rows: int, shared_projections: int
) -> float:
    """Compute relative projection work when shared projections still consume all rows."""
    if total_rows < 1 or not 0 <= exact_rows <= total_rows or shared_projections < 0:
        raise ValueError("invalid hybrid projection geometry")
    return (shared_projections * total_rows + exact_rows) / (
        (shared_projections + 1) * total_rows
    )
