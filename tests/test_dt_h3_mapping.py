import numpy as np
import pytest

from minimax_h3_mlx.dt_h3_checkpoint import fuse_qkv, unpair_rotary


def test_inverse_rotary_preserves_tail_and_heads():
    a = np.arange(2 * 8 * 3).reshape(16, 3)
    b = unpair_rotary(a, heads=2, head_dim=8, rotary_dim=6)
    expected = np.concatenate([a[:8][[0, 2, 4, 1, 3, 5, 6, 7]], a[8:][[0, 2, 4, 1, 3, 5, 6, 7]]])
    np.testing.assert_array_equal(b, expected)


def test_fused_qkv_head_order():
    q = np.arange(8).reshape(4, 2)
    k = q + 100
    v = q + 200
    actual = fuse_qkv(q, k, v, heads=2, head_dim=2)
    np.testing.assert_array_equal(
        actual, np.concatenate([q[:2], k[:2], v[:2], q[2:], k[2:], v[2:]])
    )


def test_bad_shape_rejected():
    with pytest.raises(ValueError):
        unpair_rotary(np.zeros((3, 2)), heads=2, head_dim=8, rotary_dim=6)
