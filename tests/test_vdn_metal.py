import mlx.core as mx
import numpy as np
import pytest

from minimax_h3_mlx import vdn_metal
from minimax_h3_mlx.vdn import _factor
from minimax_h3_mlx.vdn_metal import VDNMatrixSolver, _metal_inverse


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16, mx.float32])
@pytest.mark.parametrize("frames", [1, 2, 7])
def test_fused_temporal_preserves_padding_tap_order_and_rounding(dtype, frames):
    from minimax_h3_mlx.vdn import _temporal_reference
    mx.random.seed(91)
    spatial = mx.random.normal((frames, 3, 5, 256)).astype(dtype)
    weight = mx.random.normal((256, 1, 5)).astype(dtype)
    expected = _temporal_reference(spatial, weight)
    actual = vdn_metal.temporal_five_tap(spatial, weight)
    np.testing.assert_allclose(np.array(actual.astype(mx.float32)),
                               np.array(expected.astype(mx.float32)), rtol=1e-6, atol=1e-6)


def test_temporal_kernel_verification_and_latched_fallback(monkeypatch):
    from minimax_h3_mlx.vdn import _temporal_reference
    runtime = vdn_metal.VDNFeatureKernels()
    spatial, weight = mx.ones((3, 2, 2, 128)), mx.ones((128, 1, 5))
    mx.eval(runtime.temporal(spatial, weight, _temporal_reference))
    assert runtime.report()["metal_calls"] == 1
    assert runtime.report()["verified_geometries"] == 1
    monkeypatch.setattr(vdn_metal, "temporal_five_tap", lambda *a: mx.zeros((4, 2, 2, 128)))
    spatial = mx.ones((4, 2, 2, 128))
    result = runtime.temporal(spatial, weight, _temporal_reference)
    assert mx.array_equal(result, _temporal_reference(spatial, weight)).item()
    assert runtime.report()["fallback_reason"]
    assert runtime.report()["reference_calls"] == 1


def test_warm_vdn_run_resets_execution_counts_without_rechecking_kernels():
    from minimax_h3_mlx.vdn import H3VDNRuntime, _temporal_reference
    runtime = H3VDNRuntime.__new__(H3VDNRuntime)
    runtime._configure_inference("verified")
    runtime.solver = VDNMatrixSolver()
    runtime.feature_kernels = vdn_metal.VDNFeatureKernels()
    matrix = mx.eye(128)[None]
    spatial, weight = mx.ones((3, 1, 1, 128)), mx.ones((128, 1, 5))
    mx.eval(runtime.solver.inverse(matrix))
    mx.eval(runtime.feature_kernels.temporal(spatial, weight, _temporal_reference))
    runtime.calls = 400
    runtime.begin_run()
    assert runtime.calls == runtime.solver.metal_calls == runtime.feature_kernels.metal_calls == 0
    assert len(runtime.solver.verified_shapes) == 1
    assert len(runtime.feature_kernels.verified_shapes) == 1


@pytest.mark.parametrize("rank,scale", [(1, 1), (16, 1), (128, 1), (1, 512), (4, 4096)])
def test_metal_inverse_matches_independent_spd_reference(rank, scale):
    rng = np.random.default_rng(81)
    vectors = rng.normal(size=(2, 3, 128, rank)).astype(np.float32)
    vectors *= np.sqrt(scale / np.sum(vectors ** 2, axis=(-1, -2), keepdims=True))
    a = vectors @ vectors.swapaxes(-1, -2) + np.eye(128, dtype=np.float32)
    reference = np.linalg.inv(a.astype(np.float64))
    actual = np.array(_metal_inverse(mx.array(a)))
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(actual, reference, rtol=2e-4, atol=2e-5)
    np.testing.assert_allclose(a @ actual, np.broadcast_to(np.eye(128), a.shape), atol=3e-4)


def test_solver_verifies_each_geometry_once_and_reports_metal(monkeypatch):
    solver = VDNMatrixSolver()
    matrix = mx.broadcast_to(mx.eye(128), (2, 128, 128))
    assert mx.array_equal(solver.inverse(matrix), matrix).item()
    assert solver.report()["verified_geometries"] == 1
    def no_cpu(*args, **kwargs):
        raise AssertionError("Verified geometry returned to CPU.")
    monkeypatch.setattr(mx.linalg, "inv", no_cpu)
    assert mx.array_equal(solver.inverse(matrix), matrix).item()
    assert solver.report()["metal_calls"] == 2
    assert solver.report()["cpu_calls"] == 0


@pytest.mark.parametrize("failure", ["compile", "numerical"])
def test_solver_latches_cpu_fallback_on_kernel_or_parity_failure(monkeypatch, failure):
    calls = []
    def broken(matrix):
        calls.append(1)
        if failure == "compile":
            raise RuntimeError("Metal unavailable")
        return mx.full(matrix.shape, float("nan"))
    monkeypatch.setattr(vdn_metal, "_metal_inverse", broken)
    solver = VDNMatrixSolver()
    matrix = mx.eye(128)[None]
    for _ in range(2):
        assert mx.array_equal(solver.inverse(matrix), matrix).item()
    assert len(calls) == 1
    assert solver.report()["metal_calls"] == 0
    assert solver.report()["cpu_calls"] == 2
    assert solver.report()["fallback_reason"]


def test_solver_unsupported_geometry_retains_cpu_reference():
    solver = VDNMatrixSolver()
    assert mx.array_equal(solver.inverse(mx.eye(4)), mx.eye(4)).item()
    assert solver.report()["cpu_calls"] == 1


def test_solver_reports_mixed_execution_after_later_fallback(monkeypatch):
    solver = VDNMatrixSolver()
    mx.eval(solver.inverse(mx.eye(128)[None]))
    monkeypatch.setattr(vdn_metal, "_metal_inverse", lambda a: mx.zeros_like(a))
    matrix = mx.broadcast_to(mx.eye(128), (2, 128, 128))
    assert mx.array_equal(solver.inverse(matrix), matrix).item()
    assert solver.report()["backend"] == "mixed_metal_cpu"
    assert solver.report()["metal_calls"] == 1
    assert solver.report()["cpu_calls"] == 1


def test_metal_inverse_accepts_strided_matrix_batches():
    mx.random.seed(20)
    x = mx.random.normal((2, 256, 8)) * 0.1
    large = x @ x.swapaxes(-1, -2) + mx.eye(256)
    matrix = large[:, ::2, ::2]
    actual = _metal_inverse(matrix)
    expected = mx.linalg.inv(matrix, stream=mx.cpu)
    np.testing.assert_allclose(np.array(actual), np.array(expected), rtol=2e-4, atol=2e-5)


def test_metal_factor_preserves_transition_and_injection():
    mx.random.seed(31)
    keys = mx.random.normal((3, 2, 128, 8)) * 0.1
    a = keys @ keys.swapaxes(-1, -2)
    b = mx.random.normal((3, 2, 128, 128))
    alpha = mx.sigmoid(mx.random.normal((3, 2, 128)))
    expected = _factor(a, b, alpha)
    solver = VDNMatrixSolver()
    actual = _factor(a, b, alpha, solver=solver)
    for left, right in zip(actual, expected, strict=True):
        np.testing.assert_allclose(np.array(left), np.array(right), rtol=2e-4, atol=1e-5)
    assert solver.report()["metal_calls"] == 1
