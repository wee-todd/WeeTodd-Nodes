import mlx.core as mx
import pytest

from minimax_h3_mlx.algorithm_search.sdpa import head_grouped_sdpa, query_chunked_sdpa


def test_query_chunked_sdpa_matches_fused_baseline():
    mx.random.seed(7)
    q = mx.random.normal((1, 2, 9, 8)).astype(mx.bfloat16)
    k = mx.random.normal((1, 2, 11, 8)).astype(mx.bfloat16)
    v = mx.random.normal((1, 2, 11, 8)).astype(mx.bfloat16)
    scale = 8**-0.5
    baseline = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale)
    candidate = query_chunked_sdpa(q, k, v, scale=scale, chunk_size=4)
    mx.eval(baseline, candidate)
    assert mx.array_equal(baseline, candidate).item()


def test_query_chunked_sdpa_slices_query_mask():
    mx.random.seed(11)
    q = mx.random.normal((1, 2, 7, 8)).astype(mx.float32)
    k = mx.random.normal((1, 2, 7, 8)).astype(mx.float32)
    v = mx.random.normal((1, 2, 7, 8)).astype(mx.float32)
    mask = mx.triu(mx.full((1, 1, 7, 7), -1e9), k=1)
    scale = 8**-0.5
    baseline = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)
    candidate = query_chunked_sdpa(q, k, v, scale=scale, chunk_size=3, mask=mask)
    mx.eval(baseline, candidate)
    assert mx.allclose(baseline, candidate, rtol=1e-6, atol=1e-6).item()


def test_query_chunked_sdpa_rejects_invalid_chunk_size():
    values = mx.ones((1, 1, 2, 4))
    with pytest.raises(ValueError, match="positive"):
        query_chunked_sdpa(values, values, values, scale=0.5, chunk_size=0)


def test_head_grouped_sdpa_matches_fused_baseline():
    mx.random.seed(13)
    q = mx.random.normal((1, 5, 9, 8)).astype(mx.bfloat16)
    k = mx.random.normal((1, 5, 11, 8)).astype(mx.bfloat16)
    v = mx.random.normal((1, 5, 11, 8)).astype(mx.bfloat16)
    scale = 8**-0.5
    baseline = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale)
    candidate = head_grouped_sdpa(q, k, v, scale=scale, heads_per_group=2)
    mx.eval(baseline, candidate)
    assert mx.array_equal(baseline, candidate).item()


def test_head_grouped_sdpa_slices_per_head_mask():
    mx.random.seed(17)
    q = mx.random.normal((1, 4, 7, 8)).astype(mx.float32)
    k = mx.random.normal((1, 4, 7, 8)).astype(mx.float32)
    v = mx.random.normal((1, 4, 7, 8)).astype(mx.float32)
    mask = mx.zeros((1, 4, 7, 7))
    mask = mask.at[:, 1::2].add(mx.triu(mx.full((1, 2, 7, 7), -1e9), k=1))
    scale = 8**-0.5
    baseline = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)
    candidate = head_grouped_sdpa(q, k, v, scale=scale, heads_per_group=2, mask=mask)
    mx.eval(baseline, candidate)
    assert mx.allclose(baseline, candidate, rtol=1e-6, atol=1e-6).item()


def test_head_grouped_sdpa_rejects_mismatched_heads():
    q = mx.ones((1, 2, 3, 4))
    kv = mx.ones((1, 1, 3, 4))
    with pytest.raises(ValueError, match="same head count"):
        head_grouped_sdpa(q, kv, kv, scale=0.5, heads_per_group=1)
