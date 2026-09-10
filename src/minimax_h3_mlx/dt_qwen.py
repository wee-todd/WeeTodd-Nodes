"""Sequential Qwen conditioning from DT-owned H3 weights (text path)."""

from __future__ import annotations

import gc
from contextlib import contextmanager

import numpy as np

from .dt_h3_checkpoint import unpair_rotary
from .dt_tensor_store import DTTensorStore


def language_weights(store, index):
    result = {}

    def read(role):
        return store.read(f"__text_model__[t-{role}-{index}-0]")

    for role in ("input_layernorm", "post_attention_layernorm"):
        result[f"{role}.weight"] = read(role).reshape(-1)
    for role, heads in (("q", 64), ("k", 8), ("v", 8)):
        value = read(f"{role}_proj")
        if role != "v":
            value = unpair_rotary(value, heads=heads, head_dim=128, rotary_dim=128)
        result[f"self_attn.{role}_proj.weight"] = value
    result["self_attn.o_proj.weight"] = read("out_proj")
    for role in ("q", "k"):
        result[f"self_attn.{role}_norm.weight"] = unpair_rotary(
            read(f"norm_{role}").reshape(-1), heads=1, head_dim=128, rotary_dim=128
        )
    for role in ("gate", "up", "down"):
        result[f"mlp.{role}_proj.weight"] = store.read(
            f"__text_model__[t-mlp-{index}-mlp_{role}_proj-0-0]"
        )
    return result


def install_dt_qwen(encoder, checkpoint):
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_flatten, tree_unflatten
    from mlx_vlm.models.qwen3_vl.language import Qwen3VLDecoderLayer

    if encoder._load_vision_enabled:
        raise ValueError(
            "DT direct Qwen visual conditioning has not been qualified yet; use text-to-video."
        )
    store = DTTensorStore(checkpoint)

    class TokenLookup(nn.Module):
        def __call__(self, ids):
            tokens = np.asarray(ids).astype(np.int64)
            unique, inverse = np.unique(tokens, return_inverse=True)
            rows = [
                store.read_rows("__text_model__[t-tok_embeddings-0-0]", int(i), 1) for i in unique
            ]
            matrix = np.concatenate(rows)[inverse].reshape(*tokens.shape, 5120)
            return mx.array(matrix).astype(encoder.dtype)

    class Layers:
        num_layers = 50
        peak_layer_bytes = 0
        layers_loaded = 0

        @contextmanager
        def layer(self, index):
            if not 0 <= index < self.num_layers:
                raise ValueError("DT Qwen layer is outside the supported range.")
            values = {}
            layer = None
            try:
                source = language_weights(store, index)
                for key in list(source):
                    values[key] = mx.array(source.pop(key)).astype(encoder.dtype)
                    mx.eval(values[key])
                layer = Qwen3VLDecoderLayer(encoder.text_config, index)
                expected = {k: v.shape for k, v in tree_flatten(layer.parameters())}
                if {k: v.shape for k, v in values.items()} != expected:
                    raise ValueError("DT Qwen tensor mapping does not match its architecture.")
                layer.update(tree_unflatten(list(values.items())))
                mx.eval(layer.parameters())
                self.peak_layer_bytes = max(
                    self.peak_layer_bytes, sum(v.nbytes for v in values.values())
                )
                self.layers_loaded += 1
                yield layer
            finally:
                values.clear()
                del layer
                gc.collect()
                mx.clear_cache()

        def close(self):
            store.close()

        def report(self):
            return {
                **store.report(),
                "format": "weetodd-h3-dt-qwen-v1",
                "layers_loaded": self.layers_loaded,
                "peak_layer_bytes": self.peak_layer_bytes,
                "fixed_bytes": 0,
            }

    try:
        from .dt_source import expected_inventory, validate_inventory

        validate_inventory(store.records, "text_encoder")
        for name in expected_inventory("text_encoder"):
            store.validate_tensor(name, row_access=name == "__text_model__[t-tok_embeddings-0-0]")
        if encoder.num_layers != 50 or encoder.text_config.hidden_size != 5120:
            raise ValueError("DT H3 Qwen requires the 50-layer 5120-wide architecture.")
        encoder.language.embed_tokens = TokenLookup()
        encoder.paged_layers = Layers()
        encoder.skipped_tensors = 0
    except BaseException:
        store.close()
        raise
