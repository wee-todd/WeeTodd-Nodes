from __future__ import annotations

import mlx.core as mx
import pytest

from minimax_h3_mlx.algorithm_search.steel_attention import (
    SteelAttentionTile,
    steel_attention,
)


def test_steel_attention_tile_contract() -> None:
    SteelAttentionTile(32, 16, 4).validate()
    with pytest.raises(ValueError, match="eight rows per simdgroup"):
        SteelAttentionTile(64, 16, 4).validate()
    with pytest.raises(ValueError, match="divisible by eight"):
        SteelAttentionTile(32, 12, 4).validate()


def test_steel_attention_contract_rejects_non_h3_input() -> None:
    tile = SteelAttentionTile(32, 16, 4)
    query = mx.ones((1, 2, 32, 64), dtype=mx.bfloat16)
    with pytest.raises(ValueError, match=r"\[1, heads, rows, 128\]"):
        steel_attention(query, query, query, scale=0.125, tile=tile)


@pytest.mark.skipif(mx.default_device() == mx.cpu, reason="Metal is unavailable")
def test_steel_attention_default_tile_matches_mlx_layout() -> None:
    mx.random.seed(20260808)
    query = mx.random.uniform(-0.125, 0.125, (1, 2, 65, 128)).astype(mx.bfloat16)
    key = mx.random.uniform(-0.125, 0.125, (1, 2, 65, 128)).astype(mx.bfloat16)
    value = mx.random.uniform(-0.125, 0.125, (1, 2, 65, 128)).astype(mx.bfloat16)
    tile = SteelAttentionTile(32, 16, 4)

    actual = steel_attention(query, key, value, scale=128**-0.5, tile=tile)
    expected = mx.fast.scaled_dot_product_attention(query, key, value, scale=128**-0.5).transpose(
        0, 2, 1, 3
    )
    mx.eval(actual, expected)

    assert actual.shape == (1, 65, 2, 128)
    assert bool(mx.array_equal(actual, expected))
