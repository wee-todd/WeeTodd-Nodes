"""Opt-in H3 hot-path experiments; never retain expanded quantized weights."""

import math
from functools import partial

import mlx.core as mx
import mlx.nn as nn

POLICIES = ("off", "transient_q8", "compiled_adaln", "combined")


@partial(mx.compile, shapeless=True)
def compiled_scale_add(value, scale, shift):
    return value * scale + shift


class TransientQ8Linear(nn.Module):
    """Use dense GEMM only for wide, unadapted Q8 projections (FastVideo #1788)."""

    def __init__(self, base, row_floor=768):
        super().__init__()
        self.base = base
        self.row_floor = row_floor

    def __call__(self, x):
        if math.prod(x.shape[:-1]) < self.row_floor or self.base.scales.dtype != x.dtype:
            return self.base(x)
        dense = mx.dequantize(
            self.base.weight,
            self.base.scales,
            self.base.biases,
            group_size=self.base.group_size,
            bits=self.base.bits,
            mode=self.base.mode,
            dtype=x.dtype,
        )
        result = x @ dense.T
        if "bias" in self.base:
            result = result + self.base.bias
        return result


def configure_block(block, policy):
    if policy not in POLICIES:
        raise ValueError(f"Unknown H3 inference optimization: {policy}")
    block.compiled_adaln = policy in {"compiled_adaln", "combined"}
    if policy in {"transient_q8", "combined"}:
        for parent, name in (
            (block.attn, "qkv_proj"),
            (block.attn, "out_proj"),
            (block.mlp, "fc1"),
            (block.mlp, "fc2"),
        ):
            layer = getattr(parent, name)
            # Do not unwrap adapters or other backend wrappers.
            if type(layer) is nn.QuantizedLinear and layer.bits == 8:
                setattr(parent, name, TransientQ8Linear(layer))


def configure_inference_optimizations(dit, policy):
    if policy not in POLICIES:
        raise ValueError(f"Unknown H3 inference optimization: {policy}")
    pager = getattr(dit, "paged_blocks", None)
    if pager is not None:
        pager.inference_optimization = policy
    for block in getattr(dit, "blocks", ()):
        configure_block(block, policy)
    return {
        "policy": policy,
        "experimental": policy != "off",
        "q8_row_floor": 768,
        "expanded_weight_cache": False,
    }
