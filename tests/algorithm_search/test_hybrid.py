import pytest

from minimax_h3_mlx.algorithm_search.hybrid import (
    hybrid_theoretical_fraction,
    trajectory_state_bytes,
)


def test_h3_text_audio_trajectory_state_accounting():
    assert trajectory_state_bytes(
        predicted_rows=597, hidden_size=5376, blocks=50
    ) == 641_894_400


def test_qkv_hybrid_work_fraction_keeps_all_keys_and_values():
    fraction = hybrid_theoretical_fraction(
        total_rows=9477, exact_rows=8880, shared_projections=2
    )
    assert fraction == pytest.approx((2 * 9477 + 8880) / (3 * 9477))
    assert fraction == pytest.approx(0.9790025, rel=1e-5)


def test_hybrid_accounting_rejects_invalid_geometry():
    with pytest.raises(ValueError, match="invalid"):
        hybrid_theoretical_fraction(total_rows=4, exact_rows=5, shared_projections=2)
