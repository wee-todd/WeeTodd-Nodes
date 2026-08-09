import mlx.core as mx
import numpy as np
import pytest

from minimax_h3_mlx.algorithm_search.sol_metal import (
    SolMetalConfig,
    sol_metal_attention,
    sol_metal_block_attention,
)
from minimax_h3_mlx.algorithm_search.sol_metal_global import (
    sol_metal_global_pool_block_attention,
)


def _qkv(rows=192, heads=1):
    mx.random.seed(43)
    shape = (1, heads, rows, 128)
    values = tuple(mx.random.normal(shape).astype(mx.bfloat16) for _ in range(3))
    mx.eval(*values)
    return values


@pytest.mark.skipif(not hasattr(mx.fast, "metal_kernel"), reason="custom Metal unavailable")
def test_metal_all_exact_route_matches_dense_attention_closely():
    q, k, v = _qkv()
    scale = 128**-0.5
    dense = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale)
    candidate = sol_metal_attention(
        q,
        k,
        v,
        scale=scale,
        config=SolMetalConfig(prefix_rows=64, force_all_exact=True),
    )
    mx.eval(dense, candidate)
    np.testing.assert_allclose(
        np.asarray(candidate.astype(mx.float32)),
        np.asarray(dense.astype(mx.float32)),
        rtol=0.03,
        atol=0.03,
    )


def test_metal_rejects_non_h3_shape_and_scale():
    q = mx.ones((1, 1, 16, 8), dtype=mx.bfloat16)
    with pytest.raises(ValueError, match="H3 shape"):
        sol_metal_attention(q, q, q, scale=8**-0.5, config=SolMetalConfig(prefix_rows=4))
    q, k, v = _qkv()
    with pytest.raises(ValueError, match="attention scale"):
        sol_metal_attention(q, k, v, scale=1.0, config=SolMetalConfig(prefix_rows=64))


@pytest.mark.skipif(not hasattr(mx.fast, "metal_kernel"), reason="custom Metal unavailable")
def test_block_metal_all_exact_route_matches_dense_and_reports_routes():
    q, k, v = _qkv(rows=224, heads=1)
    scale = 128**-0.5
    dense = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale)
    candidate, route_counts = sol_metal_block_attention(
        q,
        k,
        v,
        scale=scale,
        config=SolMetalConfig(prefix_rows=64, force_all_exact=True),
        return_route_counts=True,
    )
    mx.eval(dense, candidate, route_counts)
    np.testing.assert_allclose(
        np.asarray(candidate.astype(mx.float32)),
        np.asarray(dense.astype(mx.float32)),
        rtol=0.03,
        atol=0.03,
    )
    assert route_counts.shape == (1, 1, 3)
    np.testing.assert_array_equal(np.asarray(route_counts), np.full((1, 1, 3), 3))


@pytest.mark.skipif(not hasattr(mx.fast, "metal_kernel"), reason="custom Metal unavailable")
def test_block_metal_keeps_prefix_queries_dense():
    q, k, v = _qkv(rows=192, heads=1)
    scale = 128**-0.5
    dense = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale)
    candidate = sol_metal_block_attention(
        q,
        k,
        v,
        scale=scale,
        config=SolMetalConfig(prefix_rows=64, beta=0.75),
    )
    mx.eval(dense, candidate)
    np.testing.assert_array_equal(
        np.asarray(candidate[..., :64, :].astype(mx.float32)),
        np.asarray(dense[..., :64, :].astype(mx.float32)),
    )


@pytest.mark.parametrize("simdgroups", [8, 16])
@pytest.mark.skipif(not hasattr(mx.fast, "metal_kernel"), reason="custom Metal unavailable")
def test_global_pool_block_metal_all_exact_matches_dense(simdgroups):
    q, k, v = _qkv(rows=224, heads=1)
    scale = 128**-0.5
    dense = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale)
    candidate, route_counts = sol_metal_global_pool_block_attention(
        q,
        k,
        v,
        scale=scale,
        config=SolMetalConfig(prefix_rows=64, force_all_exact=True),
        simdgroups=simdgroups,
        return_route_counts=True,
    )
    mx.eval(dense, candidate, route_counts)
    np.testing.assert_allclose(
        np.asarray(candidate.astype(mx.float32)),
        np.asarray(dense.astype(mx.float32)),
        rtol=0.03,
        atol=0.03,
    )
    np.testing.assert_array_equal(np.asarray(route_counts), np.full((1, 1, 3), 3))


def test_global_pool_block_metal_rejects_invalid_simdgroup_count():
    q, k, v = _qkv()
    with pytest.raises(ValueError, match="8 or 16"):
        sol_metal_global_pool_block_attention(
            q,
            k,
            v,
            scale=128**-0.5,
            config=SolMetalConfig(prefix_rows=64),
            simdgroups=4,
        )
