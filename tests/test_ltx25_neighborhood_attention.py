import mlx.core as mx
import numpy as np

from ltx25_mlx.diffusion_vae import _apply_rope
from ltx25_mlx.neighborhood_attention import (
    metal_neighborhood_attention_3d_slice,
    metal_qk_rmsnorm_rope_3d,
    metal_rmsnorm_rope_3d_slice,
    neighborhood_attention_3d,
)


def _numpy_shifted_na3d(q, k, v, kernel):
    """Independent translation of the official eager NATTEN window contract."""
    batch, frames, height, width, heads, head_dim = q.shape
    dimensions = (frames, height, width)
    bounds = []
    for length, size in zip(dimensions, kernel, strict=True):
        size = min(size, length)
        low = length - size
        half = size // 2
        starts = [min(max(index - half, 0), low) for index in range(length)]
        bounds.append([(start, start + size) for start in starts])
    output = np.empty_like(v)
    scale = head_dim**-0.5
    for t in range(frames):
        for h in range(height):
            for w in range(width):
                slices = tuple(
                    slice(*axis[index]) for axis, index in zip(bounds, (t, h, w), strict=True)
                )
                keys = k[(slice(None), *slices, slice(None), slice(None))].reshape(
                    batch, -1, heads, head_dim
                )
                values = v[(slice(None), *slices, slice(None), slice(None))].reshape(
                    batch, -1, heads, head_dim
                )
                query = q[:, t, h, w]
                scores = np.einsum("bhd,bkhd->bhk", query, keys) * scale
                scores = scores - scores.max(axis=-1, keepdims=True)
                probabilities = np.exp(scores)
                probabilities /= probabilities.sum(axis=-1, keepdims=True)
                output[:, t, h, w] = np.einsum("bhk,bkhd->bhd", probabilities, values)
    return output


def test_neighborhood_attention_3d_masks_borders_and_chunks_queries():
    q = mx.zeros((1, 2, 2, 2, 1, 1), dtype=mx.float32)
    k = mx.zeros_like(q)
    v = mx.arange(8, dtype=mx.float32).reshape(1, 2, 2, 2, 1, 1)
    output = neighborhood_attention_3d(q, k, v, query_chunk_size=3)
    mx.eval(output)

    # A 3x3x3 neighborhood covers the complete 2x2x2 volume at every voxel. With zero Q/K,
    # softmax is uniform over the eight valid entries and padded entries must not contribute.
    np.testing.assert_allclose(np.asarray(output), np.full(output.shape, 3.5), atol=1e-6)


def test_neighborhood_attention_3d_rejects_even_kernel():
    q = mx.zeros((1, 1, 1, 1, 1, 1))
    try:
        neighborhood_attention_3d(q, q, q, kernel=(2, 3, 3))
    except ValueError as error:
        assert "positive odd" in str(error)
    else:
        raise AssertionError("even neighborhood kernel was accepted")


def test_neighborhood_attention_shift_keeps_complete_border_window():
    import mlx.core as mx

    q = mx.zeros((1, 3, 3, 3, 1, 1))
    k = mx.zeros_like(q)
    values = mx.arange(27, dtype=mx.float32).reshape(1, 3, 3, 3, 1, 1)
    output = neighborhood_attention_3d(
        q,
        k,
        values,
        kernel=(3, 3, 3),
        boundary_mode="shift",
        query_chunk_size=4,
    )
    mx.eval(output)
    expected = float(mx.mean(values))
    assert abs(float(output[0, 0, 0, 0, 0, 0]) - expected) < 1e-5
    assert abs(float(output[0, 2, 2, 2, 0, 0]) - expected) < 1e-5


def test_neighborhood_attention_shift_matches_official_eager_window_contract():
    rng = np.random.default_rng(20260814)
    shape = (1, 4, 5, 6, 2, 4)
    q = rng.normal(size=shape).astype(np.float32)
    k = rng.normal(size=shape).astype(np.float32)
    v = rng.normal(size=shape).astype(np.float32)
    expected = _numpy_shifted_na3d(q, k, v, (3, 3, 5))
    actual = neighborhood_attention_3d(
        mx.array(q),
        mx.array(k),
        mx.array(v),
        kernel=(3, 3, 5),
        boundary_mode="shift",
        query_chunk_size=7,
    )
    mx.eval(actual)
    np.testing.assert_allclose(np.asarray(actual), expected, rtol=2e-5, atol=2e-5)


def test_fused_sdpa_matches_shifted_einsum_attention():
    q = mx.random.normal((1, 3, 3, 3, 2, 4), key=mx.random.key(21), dtype=mx.float32)
    k = mx.random.normal((1, 3, 3, 3, 2, 4), key=mx.random.key(22), dtype=mx.float32)
    v = mx.random.normal((1, 3, 3, 3, 2, 4), key=mx.random.key(23), dtype=mx.float32)
    expected = neighborhood_attention_3d(
        q, k, v, kernel=(3, 3, 3), query_chunk_size=7, boundary_mode="shift"
    )
    actual = neighborhood_attention_3d(
        q,
        k,
        v,
        kernel=(3, 3, 3),
        query_chunk_size=7,
        boundary_mode="shift",
        backend="sdpa",
    )
    mx.eval(expected, actual)
    assert mx.allclose(expected, actual, rtol=2e-5, atol=2e-5)


def test_metal_na3d_matches_shifted_einsum_attention():
    shape = (1, 3, 3, 3, 2, 64)
    q = mx.random.normal(shape, key=mx.random.key(31), dtype=mx.bfloat16)
    k = mx.random.normal(shape, key=mx.random.key(32), dtype=mx.bfloat16)
    v = mx.random.normal(shape, key=mx.random.key(33), dtype=mx.bfloat16)
    expected = neighborhood_attention_3d(
        q, k, v, kernel=(3, 3, 3), query_chunk_size=7, boundary_mode="shift"
    )
    actual = neighborhood_attention_3d(
        q,
        k,
        v,
        kernel=(3, 3, 3),
        query_chunk_size=7,
        boundary_mode="shift",
        backend="metal",
    )
    mx.eval(expected, actual)
    assert mx.allclose(expected, actual, rtol=1e-2, atol=1e-2)


def test_metal_na3d_rejects_unsupported_contracts():
    q = mx.zeros((1, 3, 3, 3, 1, 64), dtype=mx.bfloat16)
    for kwargs, message in (
        ({"boundary_mode": "mask"}, "shifted"),
        ({"boundary_mode": "shift", "dilation": (1, 1, 2)}, "dilation-one"),
        ({"boundary_mode": "shift", "scale": 0.25}, "head scale"),
    ):
        try:
            neighborhood_attention_3d(q, q, q, backend="metal", **kwargs)
        except ValueError as error:
            assert message in str(error)
        else:
            raise AssertionError(f"unsupported Metal NA3D contract was accepted: {kwargs}")

    wrong_dtype = q.astype(mx.float32)
    try:
        neighborhood_attention_3d(
            wrong_dtype,
            wrong_dtype,
            wrong_dtype,
            boundary_mode="shift",
            backend="metal",
        )
    except TypeError as error:
        assert "BF16" in str(error)
    else:
        raise AssertionError("non-BF16 Metal NA3D input was accepted")


def test_metal_qk_rmsnorm_rope_matches_reference_bf16_path():
    shape = (1, 3, 4, 5, 2, 64)
    q = mx.random.normal(shape, key=mx.random.key(41), dtype=mx.bfloat16)
    k = mx.random.normal(shape, key=mx.random.key(42), dtype=mx.bfloat16)
    q_weight = mx.random.uniform(
        low=0.5, high=1.5, shape=(64,), key=mx.random.key(43), dtype=mx.bfloat16
    )
    k_weight = mx.random.uniform(
        low=0.5, high=1.5, shape=(64,), key=mx.random.key(44), dtype=mx.bfloat16
    )
    q_expected = _apply_rope(mx.fast.rms_norm(q, q_weight, 1e-6))
    k_expected = _apply_rope(mx.fast.rms_norm(k, k_weight, 1e-6))
    q_actual, k_actual = metal_qk_rmsnorm_rope_3d(q, k, q_weight, k_weight)
    mx.eval(q_expected, k_expected, q_actual, k_actual)
    assert mx.allclose(q_expected, q_actual, rtol=1e-2, atol=1e-2)
    assert mx.allclose(k_expected, k_actual, rtol=1e-2, atol=1e-2)


def test_metal_sliced_q_preparation_and_attention_match_complete_volume():
    shape = (1, 3, 4, 5, 2, 64)
    q = mx.random.normal(shape, key=mx.random.key(61), dtype=mx.bfloat16)
    k = mx.random.normal(shape, key=mx.random.key(62), dtype=mx.bfloat16)
    v = mx.random.normal(shape, key=mx.random.key(63), dtype=mx.bfloat16)
    q_weight = mx.random.uniform(
        low=0.5, high=1.5, shape=(64,), key=mx.random.key(64), dtype=mx.bfloat16
    )
    k_weight = mx.random.uniform(
        low=0.5, high=1.5, shape=(64,), key=mx.random.key(65), dtype=mx.bfloat16
    )
    full_q, full_k = metal_qk_rmsnorm_rope_3d(q, k, q_weight, k_weight)
    full = neighborhood_attention_3d(
        full_q,
        full_k,
        v,
        kernel=(3, 3, 3),
        boundary_mode="shift",
        backend="metal",
    )
    flat_q = q.reshape(-1, shape[-2], shape[-1])
    pieces = []
    for start in range(0, flat_q.shape[0], 7):
        stop = min(start + 7, flat_q.shape[0])
        prepared = metal_rmsnorm_rope_3d_slice(
            flat_q[start:stop], q_weight, full_shape=shape, query_start=start
        )
        pieces.append(
            metal_neighborhood_attention_3d_slice(
                prepared,
                full_k,
                v,
                query_start=start,
                kernel=(3, 3, 3),
                scale=0.125,
            )
        )
    sliced = mx.concatenate(pieces, axis=0).reshape(shape)
    mx.eval(full, sliced)
    assert mx.array_equal(full, sliced)
