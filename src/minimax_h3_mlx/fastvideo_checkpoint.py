"""Stream FastVideo's Diffusers-layout FastH3 checkpoint into native H3 pages.

The conversion is deliberately page-first: a 33B BF16 transformer is never resident as one
object and no full-size native intermediate is written.  Each H3 block is mapped, optionally
quantized, and saved independently for :func:`load_paged_dit`.
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

import mlx.core as mx

from .paged_checkpoint import (
    PAGED_FORMAT,
    PAGED_MANIFEST,
    PagedCheckpointManifest,
    PageRecord,
)
from .quantize import QuantConfig

FASTH3_DENSE_MODEL_ID = "FastVideo/FastVideo-FastH3-4-step-Preview-v1-Dense-DataFree"
FASTH3_DENSE_REVISION = "f624f08c6c279ab43534c003e556fc5b295b6558"
FASTH3_VSA_MODEL_ID = "FastVideo/FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree"
FASTH3_VSA_REVISION = "b65818d41939b5085451074fe8ca8b799f8d4921"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_index(root: Path) -> dict[str, str]:
    for name in (
        "diffusion_pytorch_model.safetensors.index.json",
        "model.safetensors.index.json",
    ):
        path = root / name
        if path.is_file():
            return json.loads(path.read_text())["weight_map"]
    shards = sorted(root.glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"No FastH3 SafeTensors files found in {root}.")
    result: dict[str, str] = {}
    for shard in shards:
        for key in mx.load(str(shard)):
            if key in result:
                raise ValueError(f"Duplicate FastH3 tensor key {key!r}.")
            result[key] = shard.name
    return result


def _native_config(raw: dict) -> dict:
    mapping = {
        "hidden_size": "hidden_size",
        "num_layers": "num_layers",
        "num_refiner_layers": "token_refiner_num_layers",
        "num_attention_heads": "num_attention_heads",
        "attention_head_dim": "attention_head_dim",
        "ffn_dim": "ffn_hidden_size",
        "in_channels": "latents_dim",
        "audio_in_channels": "audio_latents_dim",
        "patch_size": "patch_size",
        "text_dim": "text_dim",
        "freq_dim": "timestep_input_dim",
        "time_embed_hidden_dim": "time_embed_hidden_size",
        "time_embed_dim": "time_embed_dim",
        "rope_freq_dim": "rope_inv_freq_len",
        "rope_theta": "rope_theta",
        "norm_eps": "norm_eps",
        "qk_norm_eps": "qk_norm_eps",
        "final_norm_eps": "final_norm_eps",
    }
    return {target: raw[source] for source, target in mapping.items() if source in raw}


def _direct_name(name: str) -> str:
    replacements = (
        ("transformer_blocks.", "blocks."),
        ("token_refiner.refiner_blocks.", "token_refiner.blocks."),
        ("audio_proj_in", "audio_patch_proj"),
        ("audio_proj_out", "final_layer.audio_out"),
        ("context_embedder", "condition_proj"),
        ("norm_out.linear", "final_layer.adaln_proj.linear"),
        ("norm_out.norm", "final_layer.norm"),
        ("proj_in", "video_patch_proj"),
        ("proj_out", "final_layer.video_out"),
        ("time_embedder.linear_1", "time_embedder.proj_in"),
        ("time_embedder.linear_2", "time_embedder.proj_out"),
    )
    for source, target in replacements:
        if name == source or name.startswith(source if source.endswith(".") else source + "."):
            name = target + name[len(source) :]
            break
    name = name.replace(".attn.to_out.0.", ".attn.out_proj.")
    name = name.replace(".attn.norm_q.", ".attn.q_norm.")
    name = name.replace(".attn.norm_k.", ".attn.k_norm.")
    name = name.replace(".attn.to_gate_compress.", ".attn.gate_compress.")
    name = name.replace(".ff.net.2.", ".mlp.fc2.")
    name = name.replace(".ff.net.0.proj.", ".mlp.fc1.")
    return name


def _page_source_keys(weight_map: dict[str, str], block: int | None) -> list[str]:
    prefix = None if block is None else f"transformer_blocks.{block}."
    return [
        key
        for key in weight_map
        if (
            key.startswith(prefix)
            if prefix is not None
            else not key.startswith("transformer_blocks.")
        )
    ]


def _converted_key_names(keys: list[str]) -> set[str]:
    names: set[str] = set()
    qkv_stems: set[str] = set()
    for key in keys:
        if key.endswith(".attn.to_q.weight"):
            qkv_stems.add(key[: -len("to_q.weight")])
            continue
        if key.endswith((".attn.to_k.weight", ".attn.to_v.weight")):
            continue
        names.add(_direct_name(key))
    names.update(_direct_name(stem + "qkv_proj.weight") for stem in qkv_stems)
    return names


def _validate_source_layout(weight_map: dict[str, str], blocks: int) -> dict[str, int]:
    block_indexes = {
        int(key.split(".", 2)[1])
        for key in weight_map
        if key.startswith("transformer_blocks.")
    }
    if block_indexes != set(range(blocks)):
        raise ValueError(
            f"FastH3 config declares {blocks} blocks but the index contains "
            f"{sorted(block_indexes)}."
        )
    all_names: set[str] = set()
    for block in (None, *range(blocks)):
        keys = _page_source_keys(weight_map, block)
        key_set = set(keys)
        for q_key in (key for key in keys if key.endswith(".attn.to_q.weight")):
            stem = q_key[: -len("to_q.weight")]
            missing = {stem + "to_k.weight", stem + "to_v.weight"} - key_set
            if missing:
                raise KeyError(f"FastH3 index has incomplete QKV tensors: {sorted(missing)}.")
        names = _converted_key_names(keys)
        overlap = all_names & names
        if overlap:
            raise ValueError(f"FastH3 tensors map to duplicate native keys: {sorted(overlap)[:4]}.")
        all_names.update(names)
    return {"source_tensors": len(weight_map), "native_tensors": len(all_names)}


def _load_selected(root: Path, weight_map: dict[str, str], keys: list[str]) -> dict[str, mx.array]:
    by_shard: dict[str, list[str]] = {}
    for key in keys:
        by_shard.setdefault(weight_map[key], []).append(key)
    selected: dict[str, mx.array] = {}
    for filename, names in by_shard.items():
        shard = mx.load(str(root / filename))
        selected.update({name: shard[name] for name in names})
        del shard
    return selected


def _fuse_qkv(q: mx.array, k: mx.array, v: mx.array, heads: int, head_dim: int) -> mx.array:
    if q.shape != k.shape or q.shape != v.shape:
        raise ValueError(f"FastH3 Q/K/V shapes differ: {q.shape}, {k.shape}, {v.shape}.")
    if int(q.shape[0]) != heads * head_dim:
        raise ValueError(f"FastH3 Q rows {q.shape[0]} do not equal {heads} heads x {head_dim}.")
    # Diffusers stores three contiguous matrices, each ordered [head, channel].  The native H3
    # projection consumes [head, qkv, channel], so channel must remain inside each head while the
    # q/k/v axis moves in front of it.
    stacked = mx.stack((q, k, v), axis=1).reshape(heads, head_dim, 3, q.shape[1])
    return stacked.transpose(0, 2, 1, 3).reshape(3 * heads * head_dim, q.shape[1])


def _convert_values(
    values: dict[str, mx.array], *, heads: int, head_dim: int
) -> dict[str, mx.array]:
    converted: dict[str, mx.array] = {}
    consumed: set[str] = set()
    for key, value in values.items():
        if key in consumed or any(f".attn.to_{part}.weight" in key for part in ("q", "k", "v")):
            continue
        target = _direct_name(key)
        if ".mlp.fc1.weight" in target:
            if int(value.shape[0]) % 2:
                raise ValueError(f"FastH3 SwiGLU tensor {key!r} has an odd output width.")
            half = int(value.shape[0]) // 2
            value = mx.concatenate((value[half:], value[:half]), axis=0)
        converted[target] = value

    for q_key in sorted(key for key in values if key.endswith(".attn.to_q.weight")):
        stem = q_key[: -len("to_q.weight")]
        k_key, v_key = stem + "to_k.weight", stem + "to_v.weight"
        if k_key not in values or v_key not in values:
            raise KeyError(f"FastH3 split attention projection is incomplete at {stem!r}.")
        target = _direct_name(stem + "qkv_proj.weight")
        converted[target] = _fuse_qkv(values[q_key], values[k_key], values[v_key], heads, head_dim)
        consumed.update((q_key, k_key, v_key))
    return converted


def _quantize_values(
    values: dict[str, mx.array], config: QuantConfig | None
) -> dict[str, mx.array]:
    if config is None:
        return values
    result: dict[str, mx.array] = {}
    for key, value in values.items():
        path = key.removesuffix(".weight")
        bits = config.bits_for(path) if key.endswith(".weight") and value.ndim == 2 else None
        if bits is None or int(value.shape[-1]) % config.group_size:
            result[key] = value
            continue
        packed, scales, biases = mx.quantize(value, group_size=config.group_size, bits=bits)
        result[key] = packed
        result[path + ".scales"] = scales
        result[path + ".biases"] = biases
    return result


def convert_fastvideo_fasth3_to_paged(
    source: str | Path,
    destination: str | Path,
    *,
    quant_config: QuantConfig | None = None,
    verify_output: bool = True,
) -> PagedCheckpointManifest:
    """Convert an official dense FastH3 transformer directory to WeeTodd native pages."""
    source = Path(source).expanduser().resolve()
    destination = Path(destination).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"FastH3 destination already exists: {destination}")
    raw_config = json.loads((source / "config.json").read_text())
    native_config = _native_config(raw_config)
    blocks = int(native_config["num_layers"])
    heads = int(native_config["num_attention_heads"])
    head_dim = int(native_config["attention_head_dim"])
    weight_map = _source_index(source)
    has_vsa_gates = any(".attn.to_gate_compress.weight" in key for key in weight_map)
    native_config["vsa_gate"] = has_vsa_gates
    _validate_source_layout(weight_map, blocks)

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        pages_dir = temporary / "pages"
        pages_dir.mkdir()

        def write_page(name: str, source_keys: list[str]) -> PageRecord:
            values = _load_selected(source, weight_map, source_keys)
            converted = _quantize_values(
                _convert_values(values, heads=heads, head_dim=head_dim), quant_config
            )
            mx.eval(converted)
            path = pages_dir / name
            mx.save_safetensors(str(path), converted, metadata={"format": "mlx"})
            record = PageRecord(
                file=f"pages/{name}",
                tensor_count=len(converted),
                tensor_bytes=sum(value.nbytes for value in converted.values()),
                sha256=_sha256(path),
            )
            del values, converted
            gc.collect()
            mx.clear_cache()
            return record

        fixed = write_page("fixed.safetensors", _page_source_keys(weight_map, None))
        page_records = tuple(
            write_page(f"block-{index:03d}.safetensors", _page_source_keys(weight_map, index))
            for index in range(blocks)
        )
        (temporary / "config.json").write_text(json.dumps(native_config, indent=2) + "\n")
        if quant_config is not None:
            quant = {
                "bits": quant_config.bits,
                "group_size": quant_config.group_size,
                "quantize_core": quant_config.quantize_core,
                "quantize_adaln": quant_config.quantize_adaln,
                "adaln_bits": quant_config.adaln_bits if quant_config.quantize_adaln else None,
                "overrides": quant_config.overrides,
            }
            (temporary / "quant_config.json").write_text(json.dumps(quant, indent=2) + "\n")
        manifest = {
            "format": PAGED_FORMAT,
            "num_blocks": blocks,
            "source": FASTH3_VSA_MODEL_ID if has_vsa_gates else FASTH3_DENSE_MODEL_ID,
            "source_revision": FASTH3_VSA_REVISION if has_vsa_gates else FASTH3_DENSE_REVISION,
            "source_tensor_bytes": sum(
                path.stat().st_size for path in source.glob("*.safetensors")
            ),
            "fixed": fixed.__dict__,
            "blocks": [record.__dict__ for record in page_records],
            "sampling": {"schedule_points": 5, "transformer_evaluations": 4},
            "attention": "vsa_h3_64_90" if has_vsa_gates else "dense",
        }
        (temporary / PAGED_MANIFEST).write_text(json.dumps(manifest, indent=2) + "\n")
        os.replace(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return PagedCheckpointManifest.load(destination, verify_hashes=verify_output)


__all__ = [
    "FASTH3_DENSE_MODEL_ID",
    "FASTH3_DENSE_REVISION",
    "FASTH3_VSA_MODEL_ID",
    "FASTH3_VSA_REVISION",
    "convert_fastvideo_fasth3_to_paged",
]
