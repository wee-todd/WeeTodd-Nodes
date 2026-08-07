import numpy as np

from minimax_h3_mlx.algorithm_search.low_rank import (
    fit_reduced_map,
    randomized_right_basis,
    supervised_input_basis,
)


def test_randomized_bases_recover_low_rank_linear_residual():
    rng = np.random.default_rng(5)
    inputs, _ = np.linalg.qr(rng.standard_normal((128, 12), dtype=np.float32))
    inputs = inputs.astype(np.float32)
    left = rng.standard_normal((12, 3), dtype=np.float32)
    right = rng.standard_normal((3, 10), dtype=np.float32)
    targets = inputs @ left @ right
    target_basis = randomized_right_basis(targets, 3, seed=11)
    input_basis = supervised_input_basis(inputs, targets, target_basis)
    mapping = fit_reduced_map(inputs, targets, input_basis, target_basis)
    predicted = (inputs @ input_basis) @ mapping @ target_basis.T
    relative = np.linalg.norm(predicted - targets) / np.linalg.norm(targets)
    assert relative < 1e-5


def test_randomized_basis_rejects_invalid_rank():
    with np.testing.assert_raises_regex(ValueError, "rank"):
        randomized_right_basis(np.ones((4, 3), dtype=np.float32), 4)
