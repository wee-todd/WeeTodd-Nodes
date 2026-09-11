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


def unpair_rotary(value, *, heads=56, head_dim=128, rotary_dim=96, array_module=np):
    if value.shape[0] != heads * head_dim or rotary_dim % 2 or rotary_dim > head_dim:
        raise ValueError("Invalid DT rotary tensor shape.")
    order = array_module.array(
        np.concatenate(
            (
                np.arange(0, rotary_dim, 2),
                np.arange(1, rotary_dim, 2),
                np.arange(rotary_dim, head_dim),
            )
        )
    )
    return value.reshape(heads, head_dim, *value.shape[1:])[:, order].reshape(value.shape)


def fuse_qkv(q, k, v, *, heads=56, head_dim=128, array_module=np):
    if q.shape != k.shape or q.shape != v.shape or q.shape[0] != heads * head_dim:
        raise ValueError("Invalid DT Q/K/V shapes.")
    return array_module.stack(
        [a.reshape(heads, head_dim, *a.shape[1:]) for a in (q, k, v)], axis=1
    ).reshape(-1, *q.shape[1:])


class DTH3Mapping:
    def __init__(self, store, *, array_module=np):
        self.store = store
        self.xp = array_module
        self.consumed = set()

    def read(self, name, index=0, parameter=0):
        key = f"__dit__[t-{name}-{index}-{parameter}]"
        self.consumed.add(key)
        value = self.store.read(key)
        return value if self.xp is np else self.xp.array(value)

    def block(self, index, *, refiner=False, skip_adaln=False):
        prefix = "refiner_" if refiner else ""

        def read(name):
            return self.read(prefix + name, index)

        def native_group(roles, layout):
            reader = getattr(self.store, "read_native_group", None)
            if refiner or not skip_adaln or self.xp is np or reader is None:
                return None
            names = [f"__dit__[t-{role}-{index}-0]" for role in roles]
            value = reader(names, layout=layout)
            if value is not None:
                self.consumed.update(names)
            return value

        qkv = native_group(("q", "k", "v"), "qkv")
        if qkv is None:
            q, k, v = (read(name) for name in ("q", "k", "v"))
            if not refiner:
                q, k = (unpair_rotary(a, array_module=self.xp) for a in (q, k))
            qkv = fuse_qkv(q, k, v, array_module=self.xp)
        fc1 = native_group(("gate", "up"), "fc1")
        if fc1 is None:
            fc1 = self.xp.concatenate([read("gate"), read("up")])
        qn, kn = (read(name).reshape(-1) for name in ("norm_q", "norm_k"))
        if not refiner:
            qn, kn = (unpair_rotary(a, heads=1, array_module=self.xp) for a in (qn, kn))
        values = {
            "norm1.weight": read("norm1").reshape(-1),
            "norm2.weight": read("norm2").reshape(-1),
            "attn.qkv_proj.weight": qkv,
            "attn.q_norm.weight": qn,
            "attn.k_norm.weight": kn,
            "attn.out_proj.weight": read("o"),
            "mlp.fc1.weight": fc1,
            "mlp.fc2.weight": read("down"),
        }
        if not refiner and not skip_adaln:
            for parameter, suffix in ((0, "weight"), (1, "bias")):
                values[f"adaln_proj.linear.{suffix}"] = self.xp.concatenate(
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
            values[f"final_layer.adaln_proj.linear.{suffix}"] = self.xp.concatenate(
                [
                    self.read("norm_out_shift", parameter=parameter),
                    self.read("norm_out_scale", parameter=parameter),
                ]
            )
        return values


def _native_arrays(values, *, evaluate=True):
    import mlx.core as mx

    from .load import is_fp32_key

    out = {}
    for key in list(values):
        array = values.pop(key)
        out[key] = mx.array(array).astype(mx.float32 if is_fp32_key(key) else mx.bfloat16)
        if evaluate:
            mx.eval(out[key])
    return out


def _load_native_record(mapping, record_file, *, skip_adaln):
    import mlx.core as mx

    batch = mapping.xp is mx and record_file != "fixed" and skip_adaln

    def prepare():
        if record_file == "fixed":
            values = mapping.fixed()
        else:
            index = int(record_file)
            values = {
                f"blocks.{index}.{k}": v
                for k, v in mapping.block(index, skip_adaln=skip_adaln).items()
            }
        return _native_arrays(values, evaluate=not batch)

    return mapping.store.materialize(prepare) if batch else prepare()


def load_dt_h3_dit(path: str | Path, *, window_size=1, decode_backend="mlx"):
    """Load H3 fixed weights, leaving transformer blocks in their original DT file."""
    if decode_backend not in {"mlx", "reference"}:
        raise ValueError("DT weight decode backend must be mlx or reference.")
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
            if decode_backend == "mlx":
                from .dt_mlx_decode import DTMLXTensorStore

                self.source = DTMLXTensorStore(path)
            else:
                self.source = DTTensorStore(path)
            self.mapping = DTH3Mapping(
                self.source, array_module=mx if decode_backend == "mlx" else np
            )

        def _load_record(self, record, *, skip_adaln):
            self.source._check()
            if self._cache_enabled and record.file in self._retained:
                self.raw_cache_hits += 1
                return self._retained[record.file]
            started = time.perf_counter()
            before = self.source.payload_bytes_read
            values = _load_native_record(self.mapping, record.file, skip_adaln=skip_adaln)
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
                "weight_decode_backend": decode_backend,
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
