import mlx.core as mx
import mlx.nn as nn
import pytest

from minimax_h3_mlx.algorithm_search.block_quantization import (
    parse_block_bit_overrides,
    parse_module_bit_overrides,
    quantize_selected_blocks,
    quantize_selected_modules,
    selected_block_predicate,
)


class _Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv_proj = nn.Linear(64, 192, bias=False)
        self.out_proj = nn.Linear(64, 64, bias=False)


class _FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(64, 128, bias=False)
        self.fc2 = nn.Linear(64, 64, bias=False)


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = _Attention()
        self.mlp = _FeedForward()


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = [_Block(), _Block()]
        self.unrelated = nn.Linear(64, 64, bias=False)


def test_selected_block_quantization_changes_only_four_core_linears():
    model = _Model()
    summary = quantize_selected_blocks(model, [1], bits=5, group_size=64)

    assert summary["selected_paths"] == [
        "blocks.1.attn.out_proj",
        "blocks.1.attn.qkv_proj",
        "blocks.1.mlp.fc1",
        "blocks.1.mlp.fc2",
    ]
    assert summary["parameter_bytes_saved"] > 0
    assert isinstance(model.blocks[0].attn.qkv_proj, nn.Linear)
    assert isinstance(model.blocks[1].attn.qkv_proj, nn.QuantizedLinear)
    assert isinstance(model.unrelated, nn.Linear)


def test_selected_block_quantized_model_executes():
    model = _Model()
    quantize_selected_blocks(model, [0], bits=5, group_size=64)
    output = model.blocks[0].attn.qkv_proj(mx.ones((2, 64), dtype=mx.bfloat16))
    mx.eval(output)

    assert output.shape == (2, 192)


def test_parse_module_bit_overrides_requires_exact_core_projection_paths():
    values = parse_module_bit_overrides(
        ["blocks.1.mlp.fc1=8", "blocks.0.attn.qkv_proj=6"]
    )
    assert values == {"blocks.1.mlp.fc1": 8, "blocks.0.attn.qkv_proj": 6}
    with pytest.raises(ValueError, match="unsupported core projection"):
        parse_module_bit_overrides(["blocks.1.norm1=8"])


def test_selected_module_quantization_changes_only_exact_paths():
    model = _Model()
    report = quantize_selected_modules(
        model,
        {"blocks.1.mlp.fc1": 8, "blocks.1.attn.qkv_proj": 8},
        group_size=64,
    )

    assert report["selected_paths"] == [
        "blocks.1.attn.qkv_proj",
        "blocks.1.mlp.fc1",
    ]
    assert isinstance(model.blocks[1].mlp.fc1, nn.QuantizedLinear)
    assert isinstance(model.blocks[1].attn.qkv_proj, nn.QuantizedLinear)
    assert isinstance(model.blocks[1].mlp.fc2, nn.Linear)
    assert isinstance(model.blocks[0].mlp.fc1, nn.Linear)


@pytest.mark.parametrize(
    ("blocks", "bits", "group_size", "message"),
    [
        ([], 5, 64, "non-negative and non-empty"),
        ([0], 3, 64, "bits must be"),
        ([0], 5, 0, "positive"),
    ],
)
def test_selected_block_predicate_validates_configuration(blocks, bits, group_size, message):
    with pytest.raises(ValueError, match=message):
        selected_block_predicate(blocks, bits=bits, group_size=group_size)


def test_parse_block_bit_overrides():
    assert parse_block_bit_overrides(["0=6", "49=5"]) == {0: 6, 49: 5}


@pytest.mark.parametrize("value", ["bad", "-1=5", "0=3", "0=5,0=6"])
def test_parse_block_bit_overrides_rejects_invalid_values(value):
    with pytest.raises(ValueError):
        parse_block_bit_overrides([value])


def test_parse_block_bit_overrides_rejects_duplicates():
    with pytest.raises(ValueError, match="duplicate"):
        parse_block_bit_overrides(["0=5", "0=6"])
