import mlx.core as mx

from minimax_h3_mlx.algorithm_search.fused_mpp_qkv import fused_mpp_qkv
from minimax_h3_mlx.dit import apply_rotary


def _rms_norm(value, weight):
    fp32 = value.astype(mx.float32)
    normalized = fp32 * mx.rsqrt(mx.mean(fp32**2, axis=-1, keepdims=True) + 1e-5)
    return normalized.astype(mx.bfloat16) * weight


def _relative_l2(candidate, reference):
    difference = candidate.astype(mx.float32) - reference.astype(mx.float32)
    return mx.sqrt(mx.sum(difference**2) / mx.sum(reference.astype(mx.float32) ** 2))


def test_fused_mpp_qkv_has_bounded_qk_error_and_exact_value_projection():
    source = mx.random.normal((1, 7, 256), key=mx.random.key(101), dtype=mx.bfloat16)
    weight = mx.random.normal((2 * 3 * 128, 256), key=mx.random.key(102), dtype=mx.bfloat16)
    q_weight = mx.random.normal((128,), key=mx.random.key(103), dtype=mx.bfloat16)
    k_weight = mx.random.normal((128,), key=mx.random.key(104), dtype=mx.bfloat16)
    angles = mx.arange(7 * 48, dtype=mx.float32).reshape(7, 48) / 100
    cosine = mx.concatenate((mx.cos(angles), mx.cos(angles)), axis=-1)
    sine = mx.concatenate((mx.sin(angles), mx.sin(angles)), axis=-1)

    query, key, value = fused_mpp_qkv(
        source, weight, q_weight, k_weight, cosine, sine
    )
    projected = (source @ weight.T).reshape(1, 7, 2, 3, 128)
    reference_query = apply_rotary(
        _rms_norm(projected[:, :, :, 0], q_weight).transpose(0, 2, 1, 3), cosine, sine
    )
    reference_key = apply_rotary(
        _rms_norm(projected[:, :, :, 1], k_weight).transpose(0, 2, 1, 3), cosine, sine
    )
    reference_value = projected[:, :, :, 2].transpose(0, 2, 1, 3)
    mx.eval(query, key, value, reference_query, reference_key, reference_value)

    assert _relative_l2(query, reference_query).item() < 0.005
    assert _relative_l2(key, reference_key).item() < 0.005
    assert mx.array_equal(value, reference_value)


def test_fused_mpp_qkv_adds_a_bf16_lora_projection_before_preparation():
    source = mx.random.normal((1, 7, 256), key=mx.random.key(201), dtype=mx.bfloat16)
    weight = mx.random.normal((2 * 3 * 128, 256), key=mx.random.key(202), dtype=mx.bfloat16)
    q_weight = mx.ones((128,), dtype=mx.bfloat16)
    k_weight = mx.ones((128,), dtype=mx.bfloat16)
    lora_hidden = mx.random.normal((1, 7, 16), key=mx.random.key(203), dtype=mx.bfloat16)
    lora_weight = mx.random.normal((2 * 3 * 128, 16), key=mx.random.key(204), dtype=mx.bfloat16)
    cosine = mx.ones((7, 96), dtype=mx.float32)
    sine = mx.zeros((7, 96), dtype=mx.float32)

    query, key, value = fused_mpp_qkv(
        source,
        weight,
        q_weight,
        k_weight,
        cosine,
        sine,
        lora_hidden=lora_hidden,
        lora_weight=lora_weight,
    )
    projected = (source @ weight.T) + (lora_hidden @ lora_weight.T)
    projected = projected.reshape(1, 7, 2, 3, 128)
    reference_query = _rms_norm(projected[:, :, :, 0], q_weight).transpose(0, 2, 1, 3)
    reference_key = _rms_norm(projected[:, :, :, 1], k_weight).transpose(0, 2, 1, 3)
    reference_value = projected[:, :, :, 2].transpose(0, 2, 1, 3)
    mx.eval(query, key, value, reference_query, reference_key, reference_value)

    assert _relative_l2(query, reference_query).item() < 0.005
    assert _relative_l2(key, reference_key).item() < 0.005
    assert mx.array_equal(value, reference_value)
