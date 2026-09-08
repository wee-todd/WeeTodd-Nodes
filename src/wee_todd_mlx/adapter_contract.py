"""Source-independent, payload-free inspection of supported low-rank adapter schemas.

This describes adapter structure, not architecture compatibility or application.
Original tensor names are retained so a backend need not rewrite/copy a checkpoint.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Collection

from .model_library import inspect_safetensors_header

PAIR_SUFFIXES = {
    ".lora_A.weight": ("a", "ab"),
    ".lora_B.weight": ("b", "ab"),
    ".lora_A.default.weight": ("a", "default"),
    ".lora_B.default.weight": ("b", "default"),
    ".lora_A.turbo.weight": ("a", "turbo"),
    ".lora_B.turbo.weight": ("b", "turbo"),
    ".lora_down.weight": ("a", "down_up"),
    ".lora_up.weight": ("b", "down_up"),
    ".lora_a.weight": ("a", "lowercase_ab"),
    ".lora_b.weight": ("b", "lowercase_ab"),
}

ALPHA_SUFFIXES = (".alpha", ".lora_alpha", ".alpha.weight")

_UNSPECIFIED_SCALING_VALUES = {
    "",
    "none",
    "null",
    "dynamic",
    "baked",
    "baked_scale",
}


def inspect_adapter_scaling(
    metadata: dict[str, str], *, adapter_label: str = "Adapter"
) -> dict[str, float | None]:
    """Normalize common exporter-level rank and alpha metadata.

    Dynamic/baked markers mean that scaling is already represented by the
    tensors. Numeric aliases must agree so a backend never chooses one source
    convention silently.
    """

    def consistent_number(
        names: tuple[str, ...], label: str, *, minimum: float
    ) -> float | None:
        values = []
        for name in names:
            raw = metadata.get(name)
            if raw is None or str(raw).strip().lower() in _UNSPECIFIED_SCALING_VALUES:
                continue
            try:
                value = float(raw)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{adapter_label} has invalid {label} metadata: {name}") from exc
            if not math.isfinite(value) or value < minimum:
                raise ValueError(f"{adapter_label} has invalid {label} metadata: {name}")
            values.append(value)
        if len(set(values)) > 1:
            raise ValueError(f"{adapter_label} has conflicting {label} metadata.")
        return values[0] if values else None

    return {
        "rank": consistent_number(
            ("lora_rank", "ss_network_dim", "network_dim"), "rank", minimum=1.0
        ),
        "alpha": consistent_number(
            ("lora_alpha", "ss_network_alpha", "network_alpha"),
            "alpha",
            minimum=0.0,
        ),
    }


def inspect_adapter(
    path,
    *,
    allow_exact_deltas=False,
    target_normalizer: Callable[[str], str | None] | None = None,
    allowed_auxiliary_names: Collection[str] = (),
):
    """Reject ambiguous pairs and unsupported fields rather than partially apply them.

    Alpha tensor values are deliberately not read here. The backend must read and
    validate finite scalar values, and match all targets against the selected base.
    No inference about Turbo schedules or QKV layout is made from a filename.
    """
    header = inspect_safetensors_header(path, include_tensors=True)
    groups, alphas, deltas, auxiliaries = {}, {}, {}, {}
    allowed_auxiliary_names = set(allowed_auxiliary_names)
    for name, tensor in header["tensors"].items():
        for suffix, (side, schema) in PAIR_SUFFIXES.items():
            if name.endswith(suffix):
                target = name[: -len(suffix)]
                group = groups.setdefault(target, {"schema": schema})
                if not target or group["schema"] != schema or side in group:
                    raise ValueError(f"Ambiguous adapter pair: {name}")
                if tensor["dtype"] not in {"F16", "BF16", "F32", "F64"}:
                    raise ValueError(f"Unsupported adapter dtype: {name}")
                group[side] = name
                break
        else:
            alpha_suffix = next(
                (suffix for suffix in ALPHA_SUFFIXES if name.endswith(suffix)), None
            )
            if alpha_suffix is not None:
                if math.prod(tensor["shape"]) != 1:
                    raise ValueError(f"Adapter alpha must be scalar: {name}")
                target = name[: -len(alpha_suffix)]
                if target in alphas:
                    raise ValueError(f"Multiple alpha tensors for adapter target: {target}")
                alphas[target] = name
            elif allow_exact_deltas and name.endswith((".diff", ".diff_b")):
                deltas[name] = tensor
            elif name in allowed_auxiliary_names:
                auxiliaries[name] = tensor
            else:
                raise ValueError(
                    f"Unsupported adapter tensor; refusing partial application: {name}"
                )
    if not groups:
        raise ValueError("No supported LoRA A/B pairs")
    orphan_alphas = set(alphas) - set(groups)
    if orphan_alphas:
        raise ValueError(f"Alpha without adapter pair: {sorted(orphan_alphas)}")
    pairs = []
    normalized_targets = set()
    for target, group in sorted(groups.items()):
        if "a" not in group or "b" not in group:
            raise ValueError(f"Incomplete adapter A/B pair: {target}")
        a, b = (header["tensors"][group[key]]["shape"] for key in ("a", "b"))
        if len(a) != 2 or len(b) != 2 or min(*a, *b) < 1 or a[0] != b[1]:
            raise ValueError(f"Invalid adapter rank or shape: {target}: A={a}, B={b}")
        normalized_target = target_normalizer(target) if target_normalizer else target
        if not normalized_target:
            raise ValueError(f"Adapter target is not supported by the selected backend: {target}")
        if normalized_target in normalized_targets:
            raise ValueError(
                "Multiple source adapter targets normalize to the same backend target: "
                f"{normalized_target}"
            )
        normalized_targets.add(normalized_target)
        pairs.append(
            {
                "target": target,
                "normalized_target": normalized_target,
                **group,
                "rank": a[0],
                "logical_shape": [b[0], a[1]],
                "alpha_tensor": alphas.get(target),
            }
        )
    fingerprint_payload = [
        {
            "target": pair["normalized_target"],
            "rank": pair["rank"],
            "shape": pair["logical_shape"],
        }
        for pair in pairs
    ]
    target_fingerprint = hashlib.sha256(
        json.dumps(fingerprint_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "format": "weetodd-adapter-contract-v2",
        "metadata": header["metadata"],
        "declared_scaling": inspect_adapter_scaling(header["metadata"]),
        "pairs": pairs,
        "pair_schemas": sorted({pair["schema"] for pair in pairs}),
        "ranks": sorted({pair["rank"] for pair in pairs}),
        "target_fingerprint": target_fingerprint,
        "exact_deltas": deltas,
        "auxiliary_tensors": auxiliaries,
        "tensor_count": header["tensor_count"],
        "compatibility": "requires_backend_target_validation",
        "scaling": "requires_backend_scaling_validation",
    }
