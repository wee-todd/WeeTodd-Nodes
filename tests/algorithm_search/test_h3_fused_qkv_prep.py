import mlx.core as mx

from minimax_h3_mlx.algorithm_search.fused_qkv_prep import fused_qkv_prep
from minimax_h3_mlx.dit import apply_rotary


def _rms_norm(value, weight):
    fp32 = value.astype(mx.float32)
    normalized = fp32 * mx.rsqrt(mx.mean(fp32**2, axis=-1, keepdims=True) + 1e-5)
    return normalized.astype(mx.bfloat16) * weight


def test_fused_h3_qkv_prep_matches_reference_layout_and_values():
    qkv = mx.random.normal((1, 7, 2, 3, 128), key=mx.random.key(81), dtype=mx.bfloat16)
    q_weight = mx.random.normal((128,), key=mx.random.key(82), dtype=mx.bfloat16)
    k_weight = mx.random.normal((128,), key=mx.random.key(83), dtype=mx.bfloat16)
    cosine = mx.cos(mx.arange(7 * 96, dtype=mx.float32).reshape(7, 96) / 100)
    sine = mx.sin(mx.arange(7 * 96, dtype=mx.float32).reshape(7, 96) / 100)

    query, key, value = fused_qkv_prep(qkv, q_weight, k_weight, cosine, sine)
    reference_query = apply_rotary(
        _rms_norm(qkv[:, :, :, 0], q_weight).transpose(0, 2, 1, 3), cosine, sine
    )
    reference_key = apply_rotary(
        _rms_norm(qkv[:, :, :, 1], k_weight).transpose(0, 2, 1, 3), cosine, sine
    )
    reference_value = qkv[:, :, :, 2].transpose(0, 2, 1, 3)
    mx.eval(query, key, value, reference_query, reference_key, reference_value)

    assert query.shape == (1, 2, 7, 128)
    assert mx.allclose(query, reference_query, rtol=2e-2, atol=2e-2)
    assert mx.allclose(key, reference_key, rtol=2e-2, atol=2e-2)
    assert mx.array_equal(value, reference_value)


def test_fused_h3_qkv_prep_rejects_non_h3_head_width():
    qkv = mx.zeros((1, 3, 2, 3, 64), dtype=mx.bfloat16)
    weight = mx.ones((128,), dtype=mx.bfloat16)
    rotary = mx.ones((3, 48), dtype=mx.float32)
    try:
        fused_qkv_prep(qkv, weight, weight, rotary, rotary)
    except ValueError as error:
        assert "3, 128" in str(error)
    else:
        raise AssertionError("non-H3 head width must be rejected")
