import math

import mlx.core as mx
import pytest
from mlx.utils import tree_map

import minimax_h3_mlx.vsa_h3 as vsa_module
from minimax_h3_mlx.vsa_h3 import (
    FastH3VSAConfig,
    build_vsa_h3_geometry,
    vsa_h3_attention,
    vsa_h3_stack_order,
)


def test_vsa_gate_is_scoped_to_main_transformer_blocks():
    from minimax_h3_mlx.config import DiTConfig
    from minimax_h3_mlx.dit import TokenRefinerBlock, TransformerBlock

    config = DiTConfig(
        hidden_size=8,
        num_attention_heads=2,
        attention_head_dim=4,
        ffn_hidden_size=16,
        time_embed_dim=4,
        adaln_out_features=144,
        vsa_gate=True,
    )

    assert TransformerBlock(config).attn.gate_compress is not None
    assert TokenRefinerBlock(config).attn.gate_compress is None


def test_vsa_h3_geometry_matches_segment_pure_ragged_64_tile_contract():
    geometry = build_vsa_h3_geometry((70, 5, 130), (9, 10, 13))

    assert geometry.prefix_rows == 205
    assert geometry.prefix_tiles == 6
    assert geometry.video_tiles == 3 * 3 * 4
    assert geometry.variable_sizes[:6].tolist() == [64, 6, 5, 64, 64, 2]
    assert int(mx.sum(geometry.variable_sizes).item()) == 205 + 9 * 10 * 13
    assert int(mx.min(geometry.variable_sizes[6:]).item()) == 1 * 2 * 1
    assert int(mx.max(geometry.variable_sizes).item()) == 64
    sizes = geometry.variable_sizes.tolist()
    offsets = geometry.compact_tile_offsets.tolist()
    assert offsets[0] == 0
    assert all(
        next_offset == offset + size
        for offset, next_offset, size in zip(
            offsets[:-1], offsets[1:], sizes[:-1], strict=True
        )
    )
    assert geometry.compact_row_tiles.shape == (205 + 9 * 10 * 13,)
    assert geometry.compact_row_tiles[:64].tolist() == [0] * 64
    assert geometry.compact_row_tiles[64:70].tolist() == [1] * 6

    rows = mx.arange(9 * 10 * 13, dtype=mx.int32)
    row_t = rows // (10 * 13)
    row_h = (rows // 13) % 10
    row_w = rows % 13
    expected_tile = 6 + ((row_t // 4) * 3 + row_h // 4) * 4 + row_w // 4
    actual_tile = geometry.scatter_index[205:] // 64
    assert mx.array_equal(actual_tile, expected_tile)

    sequence = mx.arange(205 + 9 * 10 * 13, dtype=mx.int32)
    ordered = vsa_h3_stack_order(sequence, geometry, axis=0)
    restored = vsa_h3_stack_order(ordered, geometry, inverse=True, axis=0)
    assert mx.array_equal(restored, sequence)
    ordered_destinations = geometry.preordered_scatter_index[205:]
    assert ordered_destinations.tolist() == sorted(geometry.scatter_index[205:].tolist())


def test_vsa_h3_dense_route_and_zero_gate_match_dense_sdpa():
    mx.random.seed(3)
    rows = 8 + 4 * 4 * 4
    q = mx.random.normal((1, 1, rows, 128)).astype(mx.bfloat16)
    k = mx.random.normal((1, 1, rows, 128)).astype(mx.bfloat16)
    v = mx.random.normal((1, 1, rows, 128)).astype(mx.bfloat16)
    gate = mx.zeros(q.shape, dtype=mx.bfloat16)
    config = FastH3VSAConfig(
        sparsity=0.0,
        min_tokens=64,
        prefix_segments=(5, 3),
        video_grid=(4, 4, 4),
    )

    expected = mx.fast.scaled_dot_product_attention(q, k, v, scale=128**-0.5)
    actual, counts = vsa_h3_attention(q, k, v, gate, scale=128**-0.5, config=config)
    mx.eval(expected, actual, counts)

    assert mx.array_equal(actual, expected)
    assert counts.reshape(-1).tolist() == [9, 0]


def test_vsa_h3_sparse_route_keeps_prefix_queries_dense():
    mx.random.seed(5)
    prefix_rows = 64
    video_grid = (4, 4, 8)
    rows = prefix_rows + math.prod(video_grid)
    q = mx.random.normal((1, 1, rows, 128)).astype(mx.bfloat16)
    k = mx.random.normal((1, 1, rows, 128)).astype(mx.bfloat16)
    v = mx.random.normal((1, 1, rows, 128)).astype(mx.bfloat16)
    gate = mx.zeros(q.shape, dtype=mx.bfloat16)
    config = FastH3VSAConfig(
        sparsity=0.5,
        min_tokens=64,
        query_tile_batch=1,
        prefix_segments=(prefix_rows,),
        video_grid=video_grid,
    )

    dense = mx.fast.scaled_dot_product_attention(q, k, v, scale=128**-0.5)
    sparse, counts = vsa_h3_attention(q, k, v, gate, scale=128**-0.5, config=config)
    mx.eval(dense, sparse, counts)

    assert mx.array_equal(sparse[:, :, :prefix_rows], dense[:, :, :prefix_rows])
    assert not mx.array_equal(sparse[:, :, prefix_rows:], dense[:, :, prefix_rows:])
    assert counts.reshape(-1).tolist() == [7, 2]


def test_vsa_h3_block_stack_preorder_matches_legacy_consumer_exactly():
    mx.random.seed(17)
    prefix_rows = 67
    video_grid = (5, 6, 7)
    rows = prefix_rows + math.prod(video_grid)
    q = mx.random.normal((1, 2, rows, 128)).astype(mx.bfloat16)
    k = mx.random.normal((1, 2, rows, 128)).astype(mx.bfloat16)
    v = mx.random.normal((1, 2, rows, 128)).astype(mx.bfloat16)
    gate = mx.random.normal((1, 2, rows, 128)).astype(mx.bfloat16)
    legacy_config = FastH3VSAConfig(
        sparsity=0.5,
        min_tokens=64,
        query_tile_batch=2,
        prefix_segments=(64, 3),
        video_grid=video_grid,
        block_stack_preorder=False,
    )
    geometry = build_vsa_h3_geometry(legacy_config.prefix_segments, video_grid)
    expected, expected_counts = vsa_h3_attention(
        q, k, v, gate, scale=128**-0.5, config=legacy_config
    )

    config = FastH3VSAConfig(
        sparsity=0.5,
        min_tokens=64,
        query_tile_batch=2,
        prefix_segments=(64, 3),
        video_grid=video_grid,
        block_stack_preorder=True,
    )
    ordered = [vsa_h3_stack_order(value, geometry, axis=2) for value in (q, k, v, gate)]
    actual, actual_counts = vsa_h3_attention(
        *ordered, scale=128**-0.5, config=config
    )
    actual = vsa_h3_stack_order(actual, geometry, inverse=True, axis=2)
    mx.eval(expected, actual, expected_counts, actual_counts)

    assert mx.array_equal(actual, expected)
    assert mx.array_equal(actual_counts, expected_counts)


@pytest.mark.skipif(mx.default_device() == mx.cpu, reason="Metal is unavailable")
def test_vsa_h3_indexed_metal_matches_grouped_consumer():
    mx.random.seed(29)
    prefix_rows = 67
    video_grid = (5, 4, 7)
    rows = prefix_rows + math.prod(video_grid)
    values = [
        mx.random.uniform(-0.25, 0.25, (1, 2, rows, 128)).astype(mx.bfloat16)
        for _ in range(4)
    ]
    geometry = build_vsa_h3_geometry((64, 3), video_grid)
    ordered = [vsa_h3_stack_order(value, geometry, axis=2) for value in values]
    common = dict(
        sparsity=0.5,
        min_tokens=64,
        query_tile_batch=2,
        prefix_segments=(64, 3),
        video_grid=video_grid,
        block_stack_preorder=True,
    )
    expected, expected_counts = vsa_h3_attention(
        *ordered,
        scale=128**-0.5,
        config=FastH3VSAConfig(**common, consumer_backend="grouped_sdpa"),
    )
    actual, actual_counts = vsa_h3_attention(
        *ordered,
        scale=128**-0.5,
        config=FastH3VSAConfig(**common, consumer_backend="metal_indexed"),
    )
    mx.eval(expected, actual, expected_counts, actual_counts)

    delta = actual.astype(mx.float32) - expected.astype(mx.float32)
    relative_l2 = mx.sqrt(mx.sum(delta * delta) / mx.sum(expected.astype(mx.float32) ** 2))
    assert float(relative_l2.item()) < 1.0e-4
    assert mx.array_equal(actual_counts, expected_counts)


@pytest.mark.skipif(mx.default_device() == mx.cpu, reason="Metal is unavailable")
def test_vsa_h3_compact_summaries_match_ragged_preordered_tile_means():
    from minimax_h3_mlx.vsa_h3_metal import vsa_h3_compact_summaries

    mx.random.seed(31)
    geometry = build_vsa_h3_geometry((64, 3), (5, 4, 7))
    rows = geometry.prefix_rows + 5 * 4 * 7
    values = [
        mx.random.uniform(-0.25, 0.25, (1, 2, rows, 128)).astype(mx.bfloat16)
        for _ in range(3)
    ]
    ordered = [vsa_h3_stack_order(value, geometry, axis=2) for value in values]
    actual = vsa_h3_compact_summaries(
        *ordered,
        geometry.compact_tile_offsets,
        geometry.variable_sizes,
    )
    expected = []
    offsets = geometry.compact_tile_offsets.tolist()
    sizes = geometry.variable_sizes.tolist()
    for value in ordered:
        expected.append(
            mx.stack(
                [
                    mx.mean(value[:, :, start : start + size].astype(mx.float32), axis=2)
                    for start, size in zip(offsets, sizes, strict=True)
                ],
                axis=2,
            )
        )
    mx.eval(*actual, *expected)

    for compact, reference in zip(actual, expected, strict=True):
        assert mx.allclose(compact, reference, rtol=1.0e-5, atol=1.0e-5)


@pytest.mark.skipif(mx.default_device() == mx.cpu, reason="Metal is unavailable")
def test_vsa_h3_indexed_metal_does_not_materialize_padded_tiles(monkeypatch):
    mx.random.seed(37)
    geometry = build_vsa_h3_geometry((64, 3), (5, 4, 7))
    rows = geometry.prefix_rows + 5 * 4 * 7
    values = [
        mx.random.uniform(-0.25, 0.25, (1, 2, rows, 128)).astype(mx.bfloat16)
        for _ in range(4)
    ]
    ordered = [vsa_h3_stack_order(value, geometry, axis=2) for value in values]

    def reject_padded_materialization(*_args, **_kwargs):
        raise AssertionError("indexed Metal rematerialized padded tile tensors")

    monkeypatch.setattr(vsa_module, "_tile_rows", reject_padded_materialization)
    output, counts = vsa_h3_attention(
        *ordered,
        scale=128**-0.5,
        config=FastH3VSAConfig(
            sparsity=0.5,
            min_tokens=64,
            prefix_segments=(64, 3),
            video_grid=(5, 4, 7),
            block_stack_preorder=True,
            consumer_backend="metal_indexed",
        ),
    )
    mx.eval(output, counts)

    assert output.shape == ordered[0].shape
    assert counts.reshape(-1).tolist() == [56, 16]


def test_vsa_h3_indexed_metal_requires_stack_preordering():
    value = mx.zeros((1, 1, 128, 128), dtype=mx.bfloat16)
    with pytest.raises(ValueError, match="block-stack preordering"):
        vsa_h3_attention(
            value,
            value,
            value,
            value,
            scale=128**-0.5,
            config=FastH3VSAConfig(
                sparsity=0.5,
                min_tokens=64,
                prefix_segments=(64,),
                video_grid=(4, 4, 4),
                block_stack_preorder=False,
                consumer_backend="metal_indexed",
            ),
        )


def test_vsa_qkv_backend_contract_rejects_unknown_value():
    with pytest.raises(ValueError, match="qkv_prep_backend"):
        FastH3VSAConfig(qkv_prep_backend="unknown").validate()


def test_fused_metal_qkv_preparation_preserves_small_attention_fixture():
    from minimax_h3_mlx.config import DiTConfig
    from minimax_h3_mlx.dit import Attention

    attention = Attention(
        DiTConfig(hidden_size=128, num_attention_heads=1, attention_head_dim=128)
    )
    attention.update(tree_map(lambda value: value.astype(mx.bfloat16), attention.parameters()))
    source = mx.random.uniform(-0.125, 0.125, (1, 131, 128)).astype(mx.bfloat16)
    angles = mx.arange(131 * 96, dtype=mx.float32).reshape(131, 96) / 10_000
    rotary = (mx.cos(angles), mx.sin(angles))
    attention.vsa_h3_config = FastH3VSAConfig(qkv_prep_backend="mlx")
    expected = attention._normal(source, rotary, None, 0)
    attention.vsa_h3_config = FastH3VSAConfig(qkv_prep_backend="metal_fused")
    actual = attention._normal(source, rotary, None, 0)
    mx.eval(expected, actual)

    assert mx.array_equal(actual, expected)
