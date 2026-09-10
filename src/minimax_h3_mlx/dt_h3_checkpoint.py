"""H3 tensor-layout adapter for original Draw Things checkpoints.

Uses the existing H3 block executor and sampler. Source files remain read-only;
only the active block window is decoded. Native H3's BF16/FP32 arithmetic policy
is retained, so this is not a claim of Draw Things GPU numerical parity.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from .dt_tensor_store import DTTensorStore


def unpair_rotary(value, *, heads=56, head_dim=128, rotary_dim=96):
    if value.shape[0] != heads * head_dim or rotary_dim % 2 or rotary_dim > head_dim:
        raise ValueError("Invalid DT rotary tensor shape.")
    order = np.concatenate(
        (np.arange(0, rotary_dim, 2), np.arange(1, rotary_dim, 2), np.arange(rotary_dim, head_dim))
    )
    return value.reshape(heads, head_dim, *value.shape[1:])[:, order].reshape(value.shape)


def fuse_qkv(q, k, v, *, heads=56, head_dim=128):
    if q.shape != k.shape or q.shape != v.shape or q.shape[0] != heads * head_dim:
        raise ValueError("Invalid DT Q/K/V shapes.")
    return np.stack([a.reshape(heads, head_dim, *a.shape[1:]) for a in (q, k, v)], axis=1).reshape(
        -1, *q.shape[1:]
    )


class DTH3Mapping:
    def __init__(self, store):
        self.store = store
        self.consumed = set()

    def read(self, name, index=0, parameter=0):
        key = f"__dit__[t-{name}-{index}-{parameter}]"
        self.consumed.add(key)
        return self.store.read(key)

    def block(self, index, *, refiner=False, skip_adaln=False):
        prefix = "refiner_" if refiner else ""

        def read(name):
            return self.read(prefix + name, index)

        q, k, v = (read(name) for name in ("q", "k", "v"))
        qn, kn = (read(name).reshape(-1) for name in ("norm_q", "norm_k"))
        if not refiner:
            q, k = unpair_rotary(q), unpair_rotary(k)
            qn, kn = (unpair_rotary(a, heads=1) for a in (qn, kn))
        values = {
            "norm1.weight": read("norm1").reshape(-1),
            "norm2.weight": read("norm2").reshape(-1),
            "attn.qkv_proj.weight": fuse_qkv(q, k, v),
            "attn.q_norm.weight": qn,
            "attn.k_norm.weight": kn,
            "attn.out_proj.weight": read("o"),
            "mlp.fc1.weight": np.concatenate([read("gate"), read("up")]),
            "mlp.fc2.weight": read("down"),
        }
        if not refiner and not skip_adaln:
            for parameter, suffix in ((0, "weight"), (1, "bias")):
                values[f"adaln_proj.linear.{suffix}"] = np.concatenate(
                    [
                        self.read(f"adaln_{chunk}_{modality}", index, parameter)
                        for modality in range(3)
                        for chunk in range(6)
                    ]
                )
        return values

    def fixed(self):
        values = {}
        pairs = {
            "video_patch_proj": "proj_in",
            "audio_patch_proj": "audio_proj_in",
            "condition_proj": "context_embedder",
            "time_embedder.proj_in": "time_embedder_linear_1",
            "time_embedder.proj_out": "time_embedder_linear_2",
            "final_layer.video_out": "proj_out",
            "final_layer.audio_out": "audio_proj_out",
        }
        for target, source in pairs.items():
            for parameter, suffix in ((0, "weight"), (1, "bias")):
                values[f"{target}.{suffix}"] = self.read(source, parameter=parameter)
        for i in range(2):
            values.update(
                {f"token_refiner.blocks.{i}.{k}": v for k, v in self.block(i, refiner=True).items()}
            )
        values["token_refiner.final_norm.weight"] = self.read("refiner_final_norm").reshape(-1)
        values["final_layer.norm.weight"] = self.read("norm_out").reshape(-1)
        for parameter, suffix in ((0, "weight"), (1, "bias")):
            values[f"final_layer.adaln_proj.linear.{suffix}"] = np.concatenate(
                [
                    self.read("norm_out_shift", parameter=parameter),
                    self.read("norm_out_scale", parameter=parameter),
                ]
            )
        return values


def _native_arrays(values):
    import mlx.core as mx

    from .load import is_fp32_key

    out = {}
    for key in list(values):
        array = values.pop(key)
        out[key] = mx.array(array).astype(mx.float32 if is_fp32_key(key) else mx.bfloat16)
        mx.eval(out[key])
    return out


def load_dt_h3_dit(path: str | Path, *, window_size=1):
    """Load H3 fixed weights, leaving transformer blocks in their original DT file."""
    import mlx.core as mx
    from mlx.utils import tree_flatten, tree_unflatten

    from .config import DiTConfig
    from .dit import MiniMaxH3DiT
    from .paged_checkpoint import (
        PagedBlockExecutor,
        PagedCheckpointManifest,
        PagedTensorStore,
        PageRecord,
    )

    path = Path(path).expanduser().resolve(strict=True)
    describe_dt_h3(path)
    config = DiTConfig()
    # Logical records feed the existing executor; no page files are generated or opened.
    records = tuple(PageRecord(str(i), 0, 0, "") for i in range(config.num_layers))
    manifest = PagedCheckpointManifest(
        path.parent, config.num_layers, path.stat().st_size, PageRecord("fixed", 0, 0, ""), records
    )

    class DirectStore(PagedTensorStore):
        def __init__(self):
            super().__init__(manifest)
            self.source = DTTensorStore(path)
            self.mapping = DTH3Mapping(self.source)

        def _load_record(self, record, *, skip_adaln):
            self.source._check()
            if self._cache_enabled and record.file in self._retained:
                self.raw_cache_hits += 1
                return self._retained[record.file]
            started = time.perf_counter()
            before = self.source.payload_bytes_read
            if record.file == "fixed":
                values = self.mapping.fixed()
            else:
                index = int(record.file)
                values = {
                    f"blocks.{index}.{k}": v
                    for k, v in self.mapping.block(index, skip_adaln=skip_adaln).items()
                }
            values = _native_arrays(values)
            self.file_tensor_bytes += self.source.payload_bytes_read - before
            self.disk_load_seconds += time.perf_counter() - started
            self.disk_page_loads += 1
            size = sum(v.nbytes for v in values.values())
            if self._cache_enabled:
                self.raw_cache_misses += 1
                if record.file != "fixed" and self.retained_bytes + size <= self.cache_budget_bytes:
                    self._retained[record.file] = values
                    self._retained_adaln_bytes[record.file] = 0
                    self.retained_bytes += size
                    self.peak_retained_bytes = max(self.peak_retained_bytes, self.retained_bytes)
            return values

    class DirectExecutor(PagedBlockExecutor):
        def close(self):
            try:
                super().close()
                self.store.release()
            finally:
                self.store.source.close()

        def report(self):
            return {
                **super().report(),
                **self.store.source.report(),
                "execution_dtype": "native_bf16_fp32",
            }

    store = DirectStore()
    executor = None
    try:
        model = MiniMaxH3DiT(config)
        model.blocks = []
        expected = {k: v.shape for k, v in tree_flatten(model.parameters())}
        fixed = store.load_fixed()
        if {k: v.shape for k, v in fixed.items()} != expected:
            raise ValueError("DT H3 fixed tensor mapping does not match the native architecture.")
        model.update(tree_unflatten(list(fixed.items())))
        mx.eval(model.parameters())
        fixed.clear()
        store.release()
        executor = DirectExecutor(manifest, config, None, window_size, prefetch=False)
        executor.store = store
        model.paged_blocks = executor
        return model
    except BaseException:
        if executor is not None:
            executor.close()
        else:
            store.source.close()
        raise


def is_dt_checkpoint(path):
    path = Path(path)
    if not path.is_file():
        return False
    with path.open("rb") as handle:
        return handle.read(16) == b"SQLite format 3\0"


def describe_dt_h3(path):
    """Validate the supported FL2VA tensor inventory without loading weights."""
    names = set()
    fixed = set()
    roles = ("q", "k", "v", "o", "gate", "up", "down", "norm1", "norm2", "norm_q", "norm_k")
    for i in range(50):
        names.update(f"__dit__[t-{role}-{i}-0]" for role in roles)
        names.update(
            f"__dit__[t-adaln_{chunk}_{modality}-{i}-{param}]"
            for chunk in range(6)
            for modality in range(3)
            for param in (0, 1)
        )
    for i in range(2):
        fixed.update(f"__dit__[t-refiner_{role}-{i}-0]" for role in roles)
    for role in (
        "proj_in",
        "audio_proj_in",
        "context_embedder",
        "time_embedder_linear_1",
        "time_embedder_linear_2",
        "proj_out",
        "audio_proj_out",
        "norm_out_shift",
        "norm_out_scale",
    ):
        fixed.update(f"__dit__[t-{role}-0-{param}]" for param in (0, 1))
    fixed.update({"__dit__[t-norm_out-0-0]", "__dit__[t-refiner_final_norm-0-0]"})
    from .dt_source import validate_inventory

    with DTTensorStore(path) as store:
        validate_inventory(store.records, "transformer")
        if set(store.records) != names | fixed:
            raise ValueError("DT checkpoint is not the supported H3 FL2VA transformer inventory.")
        for name in store.records:
            r = store.validate_tensor(name)
            if r.codec & 0x10000000:
                store._span(r, store._inline(r))
        block_sizes = []
        for i in range(50):
            block_sizes.append(
                sum(
                    r.elements * 2
                    for n, r in store.records.items()
                    if n not in fixed and n.endswith((f"-{i}-0]", f"-{i}-1]"))
                )
            )
        return {
            "tensor_count": len(store.records),
            "tensor_bytes": sum(r.elements * 2 for r in store.records.values()),
            "fixed_bytes": sum(store.records[n].elements * 2 for n in fixed),
            "window_bytes": max(block_sizes),
            "adaln_bytes": sum(r.elements * 2 for n, r in store.records.items() if "t-adaln_" in n),
        }
