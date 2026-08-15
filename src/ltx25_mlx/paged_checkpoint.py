"""Streamed Q8 page conversion for LTX 2.5 transformer and Gemma checkpoints."""

from __future__ import annotations

import gc
import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import mlx.core as mx
from safetensors import safe_open

PAGED_TRANSFORMER_FORMAT = "weetodd-ltx25-transformer-paged-q8-v1"
PAGED_GEMMA_FORMAT = "weetodd-ltx25-gemma-paged-q8-v1"
PAGED_MANIFEST = "paged_manifest.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class PageRecord:
    file: str
    tensor_count: int
    tensor_bytes: int
    sha256: str


@dataclass(frozen=True)
class LTX25PagedManifest:
    root: Path
    format: str
    kind: Literal["transformer", "gemma"]
    num_layers: int
    group_size: int
    bits: int
    source_tensor_bytes: int
    output_tensor_bytes: int
    metadata: dict[str, Any]
    fixed: PageRecord
    layers: tuple[PageRecord, ...]

    @classmethod
    def load(cls, root: str | Path, *, verify_hashes: bool = False):
        root = Path(root).expanduser().resolve()
        path = root / PAGED_MANIFEST
        if not path.is_file():
            raise FileNotFoundError(f"LTX 2.5 paged manifest not found: {path}")
        raw = json.loads(path.read_text())
        kind = str(raw.get("kind"))
        expected_format = {
            "transformer": PAGED_TRANSFORMER_FORMAT,
            "gemma": PAGED_GEMMA_FORMAT,
        }.get(kind)
        if expected_format is None or raw.get("format") != expected_format:
            raise ValueError(
                f"Unsupported LTX 2.5 paged format {raw.get('format')!r} for {kind!r}."
            )
        layers = tuple(PageRecord(**item) for item in raw["layers"])
        num_layers = int(raw["num_layers"])
        if len(layers) != num_layers:
            raise ValueError(
                f"Paged LTX 2.5 manifest declares {num_layers} layers but lists "
                f"{len(layers)} pages."
            )
        manifest = cls(
            root=root,
            format=expected_format,
            kind=kind,
            num_layers=num_layers,
            group_size=int(raw["group_size"]),
            bits=int(raw["bits"]),
            source_tensor_bytes=int(raw["source_tensor_bytes"]),
            output_tensor_bytes=int(raw["output_tensor_bytes"]),
            metadata=dict(raw.get("metadata", {})),
            fixed=PageRecord(**raw["fixed"]),
            layers=layers,
        )
        manifest.validate_files(verify_hashes=verify_hashes)
        return manifest

    def validate_files(self, *, verify_hashes: bool = False) -> None:
        for record in (self.fixed, *self.layers):
            path = (self.root / record.file).resolve()
            try:
                path.relative_to(self.root)
            except ValueError as exc:
                raise ValueError(f"Paged LTX 2.5 page escapes its root: {record.file}") from exc
            if not path.is_file():
                raise FileNotFoundError(f"Paged LTX 2.5 page not found: {path}")
            if verify_hashes and _sha256(path) != record.sha256:
                raise ValueError(f"Paged LTX 2.5 page hash differs from manifest: {path}")

    @property
    def fixed_path(self) -> Path:
        return self.root / self.fixed.file

    @property
    def layer_paths(self) -> tuple[Path, ...]:
        return tuple(self.root / record.file for record in self.layers)


def _decoded_metadata(path: Path) -> dict[str, Any]:
    with safe_open(path, framework="numpy") as handle:
        raw = handle.metadata() or {}
    result: dict[str, Any] = {}
    for key, value in raw.items():
        try:
            result[key] = json.loads(value)
        except (json.JSONDecodeError, TypeError):
            result[key] = value
    return result


def _layer_index(key: str, kind: str) -> int | None:
    prefixes = (
        ("model.diffusion_model.transformer_blocks.", "diffusion_model.transformer_blocks."),
        ("model.layers.", "model.language_model.layers."),
    )[kind == "gemma"]
    for prefix in prefixes:
        if key.startswith(prefix):
            value = key.removeprefix(prefix).split(".", 1)[0]
            if not value.isdigit():
                raise ValueError(f"Malformed LTX 2.5 {kind} layer key: {key!r}")
            return int(value)
    return None


def _quantizable(key: str, value: mx.array, *, group_size: int) -> bool:
    return (
        key.endswith(".weight")
        and value.ndim == 2
        and int(value.shape[-1]) % group_size == 0
    )


def _quantize_page(
    source: Path,
    keys: list[str],
    *,
    group_size: int,
    bits: int,
    quantize: bool = True,
) -> dict[str, mx.array]:
    loaded = dict(mx.load(str(source)))
    output: dict[str, mx.array] = {}
    try:
        for key in keys:
            value = loaded[key]
            if quantize and _quantizable(key, value, group_size=group_size):
                packed, scales, biases = mx.quantize(
                    value,
                    group_size=group_size,
                    bits=bits,
                    mode="affine",
                )
                mx.eval(packed, scales, biases)
                module = key.removesuffix(".weight")
                output[key] = packed
                output[f"{module}.scales"] = scales
                output[f"{module}.biases"] = biases
            else:
                output[key] = value
        mx.eval(output)
        return output
    finally:
        loaded.clear()


def convert_to_paged_q8(
    source: str | Path,
    destination: str | Path,
    *,
    kind: Literal["transformer", "gemma"],
    group_size: int = 64,
    bits: int = 8,
    verify_output: bool = True,
) -> LTX25PagedManifest:
    """Quantize one layer at a time and publish a directly streamable page set."""
    source = Path(source).expanduser().resolve()
    destination = Path(destination).expanduser().resolve()
    if not source.is_file() or source.suffix != ".safetensors":
        raise FileNotFoundError(f"LTX 2.5 source is not a safetensors file: {source}")
    if destination.exists():
        raise FileExistsError(f"LTX 2.5 paged destination already exists: {destination}")
    if kind not in {"transformer", "gemma"}:
        raise ValueError(f"Unsupported LTX 2.5 page kind: {kind!r}")
    if bits != 8 or group_size < 1:
        raise ValueError("The initial LTX 2.5 paged recipe requires Q8 affine quantization.")

    with safe_open(source, framework="numpy") as handle:
        keys = list(handle.keys())
        source_tensor_bytes = 0
        for key in keys:
            tensor = handle.get_slice(key)
            shape = tensor.get_shape()
            dtype = str(tensor.get_dtype())
            bytes_per_value = {"BF16": 2, "F16": 2, "F32": 4, "U8": 1}.get(dtype, 0)
            count = 1
            for dimension in shape:
                count *= int(dimension)
            source_tensor_bytes += count * bytes_per_value
    metadata = _decoded_metadata(source)
    layers: dict[int, list[str]] = {}
    fixed: list[str] = []
    for key in keys:
        index = _layer_index(key, kind)
        if index is None:
            fixed.append(key)
        else:
            layers.setdefault(index, []).append(key)
    if not layers or sorted(layers) != list(range(max(layers) + 1)):
        raise ValueError(f"LTX 2.5 {kind} layer indexes are missing or non-contiguous.")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    records: list[PageRecord] = []
    try:
        pages = temporary / "pages"
        pages.mkdir()

        def write_page(
            name: str, selected_keys: list[str], *, quantize: bool = True
        ) -> PageRecord:
            values = _quantize_page(
                source,
                selected_keys,
                group_size=group_size,
                bits=bits,
                quantize=quantize,
            )
            target = pages / name
            mx.save_safetensors(str(target), values, metadata={"format": "mlx"})
            record = PageRecord(
                file=f"pages/{name}",
                tensor_count=len(values),
                tensor_bytes=sum(int(value.nbytes) for value in values.values()),
                sha256=_sha256(target),
            )
            values.clear()
            gc.collect()
            mx.clear_cache()
            return record

        fixed_record = write_page("fixed.safetensors", sorted(fixed), quantize=False)
        for index in range(len(layers)):
            records.append(write_page(f"layer-{index:03d}.safetensors", sorted(layers[index])))
        output_tensor_bytes = fixed_record.tensor_bytes + sum(item.tensor_bytes for item in records)
        raw = {
            "format": (
                PAGED_TRANSFORMER_FORMAT if kind == "transformer" else PAGED_GEMMA_FORMAT
            ),
            "kind": kind,
            "num_layers": len(records),
            "group_size": group_size,
            "bits": bits,
            "source": source.name,
            "source_tensor_bytes": source_tensor_bytes,
            "output_tensor_bytes": output_tensor_bytes,
            "metadata": metadata,
            "fixed": asdict(fixed_record),
            "layers": [asdict(record) for record in records],
        }
        (temporary / PAGED_MANIFEST).write_text(json.dumps(raw, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return LTX25PagedManifest.load(destination, verify_hashes=verify_output)


def fuse_paged_transformer_loras(
    source: str | Path,
    destination: str | Path,
    loras: tuple[tuple[str | Path, float], ...],
    *,
    verify_output: bool = True,
) -> LTX25PagedManifest:
    """Fuse fixed LoRAs into existing Q8 transformer pages once."""
    from ltx_core_mlx.loader.fuse_loras import apply_loras
    from ltx_core_mlx.loader.primitives import LoraStateDictWithStrength, StateDict

    from .transformer import (
        _fuse_non_block_loras,
        _remap_comfy_lora_weights,
        inspect_ltx25_lora,
        remap_comfy_transformer_key,
        remap_comfy_transformer_weights,
    )

    source = Path(source).expanduser().resolve()
    destination = Path(destination).expanduser().resolve()
    manifest = LTX25PagedManifest.load(source, verify_hashes=verify_output)
    if manifest.kind != "transformer":
        raise ValueError("Only LTX 2.5 transformer pages can bake transformer LoRAs.")
    if destination.exists():
        raise FileExistsError(f"LTX 2.5 fused destination already exists: {destination}")
    if not loras:
        raise ValueError("At least one LTX 2.5 LoRA is required for page fusion.")
    if manifest.metadata.get("weetodd_baked_loras"):
        raise ValueError(
            "Fuse LTX 2.5 LoRAs from the original Q8 pages. Sequential page fusion "
            "adds an extra requantization and can change the generated video."
        )

    loaded_loras: list[tuple[dict[str, mx.array], float]] = []
    baked_loras = []
    for value, strength in loras:
        adapter = Path(value).expanduser().resolve()
        report = inspect_ltx25_lora(adapter)
        if strength <= 0:
            raise ValueError("LTX 2.5 LoRA fusion strength must be positive.")
        loaded_loras.append(
            (_remap_comfy_lora_weights(dict(mx.load(str(adapter)))), float(strength))
        )
        baked_loras.append(
            {
                "file": adapter.name,
                "sha256": _sha256(adapter),
                "bytes": int(report["bytes"]),
                "strength": float(strength),
                "adapter_role": report["adapter_role"],
                "adapter_pairs": int(report["adapter_pairs"]),
                "lora_rank": report["lora_rank"],
                "lora_alpha": report["lora_alpha"],
            }
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    records: list[PageRecord] = []

    def write_fused_page(record: PageRecord, index: int | None) -> PageRecord:
        raw = dict(mx.load(str(source / record.file)))
        mapped = remap_comfy_transformer_weights(raw)
        reverse = {}
        for raw_key in raw:
            mapped_key = remap_comfy_transformer_key(raw_key)
            if mapped_key is not None:
                reverse[mapped_key] = raw_key
        if index is None:
            fused_items = _fuse_non_block_loras(mapped, loaded_loras)
        else:
            prefix = f"transformer_blocks.{index}."
            block_weights = {
                key.removeprefix(prefix): value
                for key, value in mapped.items()
                if key.startswith(prefix)
            }
            block_loras = []
            for adapter, strength in loaded_loras:
                values = {
                    key.removeprefix(prefix): value
                    for key, value in adapter.items()
                    if key.startswith(prefix)
                }
                if values:
                    block_loras.append(
                        LoraStateDictWithStrength(
                            StateDict(sd=values, size=0, dtype=set()), strength
                        )
                    )
            fused = apply_loras(
                StateDict(sd=block_weights, size=0, dtype=set()), block_loras
            )
            fused_items = [(f"{prefix}{key}", value) for key, value in fused.sd.items()]
        output = {
            raw_key: value
            for raw_key, value in raw.items()
            if remap_comfy_transformer_key(raw_key) is None
        }
        output.update({reverse[key]: value for key, value in fused_items})
        mx.eval(output)
        target = temporary / record.file
        target.parent.mkdir(parents=True, exist_ok=True)
        mx.save_safetensors(str(target), output, metadata={"format": "mlx"})
        result = PageRecord(
            file=record.file,
            tensor_count=len(output),
            tensor_bytes=sum(int(value.nbytes) for value in output.values()),
            sha256=_sha256(target),
        )
        raw.clear()
        mapped.clear()
        output.clear()
        gc.collect()
        mx.clear_cache()
        return result

    try:
        fixed = write_fused_page(manifest.fixed, None)
        for index, record in enumerate(manifest.layers):
            records.append(write_fused_page(record, index))
        metadata = dict(manifest.metadata)
        metadata["weetodd_baked_loras"] = baked_loras
        raw_manifest = {
            "format": PAGED_TRANSFORMER_FORMAT,
            "kind": "transformer",
            "num_layers": manifest.num_layers,
            "group_size": manifest.group_size,
            "bits": manifest.bits,
            "source": source.name,
            "source_tensor_bytes": manifest.source_tensor_bytes,
            "output_tensor_bytes": fixed.tensor_bytes
            + sum(item.tensor_bytes for item in records),
            "metadata": metadata,
            "fixed": asdict(fixed),
            "layers": [asdict(record) for record in records],
        }
        (temporary / PAGED_MANIFEST).write_text(
            json.dumps(raw_manifest, indent=2, sort_keys=True) + "\n"
        )
        os.replace(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    finally:
        loaded_loras.clear()
        mx.clear_cache()
    return LTX25PagedManifest.load(destination, verify_hashes=verify_output)


__all__ = [
    "LTX25PagedManifest",
    "PAGED_GEMMA_FORMAT",
    "PAGED_MANIFEST",
    "PAGED_TRANSFORMER_FORMAT",
    "convert_to_paged_q8",
    "fuse_paged_transformer_loras",
]
