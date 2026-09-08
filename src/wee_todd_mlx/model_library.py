"""Read-only shared-model inventory, independent of MLX and ComfyUI.

File identity and architecture compatibility are different questions. This module
answers the first and safely summarizes safetensors headers; engine-specific
compatibility is deliberately not inferred from names or equal file sizes.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import struct
from collections import defaultdict
from pathlib import Path

WEIGHT_SUFFIXES = {".safetensors", ".gguf", ".ckpt", ".pt", ".pth", ".bin", ".npz"}
MAX_HEADER_BYTES = 16 * 1024 * 1024


def inspect_safetensors_header(filename: str | Path, *, include_tensors: bool = False) -> dict:
    """Read bounded metadata only, never tensors, pickle payloads, or remote code."""
    filename = Path(filename)
    size = filename.stat().st_size
    with filename.open("rb") as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ValueError("Missing safetensors header length")
        length = struct.unpack("<Q", prefix)[0]
        if not 2 <= length <= min(MAX_HEADER_BYTES, size - 8):
            raise ValueError("Safetensors header length is invalid or exceeds the inventory limit")
        header = json.loads(stream.read(length))
    if not isinstance(header, dict):
        raise ValueError("Safetensors header must be an object")
    metadata = header.pop("__metadata__", {})
    if metadata is None:
        metadata = {}  # MLX exports may encode absent metadata explicitly as null.
    if not isinstance(metadata, dict) or any(
        not isinstance(k, str) or not isinstance(v, str) for k, v in metadata.items()
    ):
        raise ValueError("Invalid safetensors metadata")
    payload_bytes = size - 8 - length
    dtypes = set()
    intervals = []
    dtype_sizes = {
        "BOOL": 1,
        "U8": 1,
        "I8": 1,
        "I16": 2,
        "U16": 2,
        "I32": 4,
        "U32": 4,
        "I64": 8,
        "U64": 8,
        "F16": 2,
        "BF16": 2,
        "F32": 4,
        "F64": 8,
        "F8_E4M3": 1,
        "F8_E5M2": 1,
    }
    for name, tensor in header.items():
        if not isinstance(tensor, dict):
            raise ValueError(f"Invalid tensor entry: {name}")
        shape, offsets, dtype = tensor.get("shape"), tensor.get("data_offsets"), tensor.get("dtype")
        if not isinstance(shape, list) or any(type(n) is not int or n < 0 for n in shape):
            raise ValueError(f"Invalid tensor shape: {name}")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or any(type(n) is not int for n in offsets)
            or not 0 <= offsets[0] <= offsets[1] <= payload_bytes
        ):
            raise ValueError(f"Invalid tensor payload offsets: {name}")
        if not isinstance(dtype, str) or dtype not in dtype_sizes:
            raise ValueError(f"Inventory does not recognize safetensors dtype {dtype!r}")
        if math.prod(shape) * dtype_sizes[dtype] != offsets[1] - offsets[0]:
            raise ValueError(f"Tensor shape/byte size mismatch: {name}")
        dtypes.add(dtype)
        intervals.append(tuple(offsets))
    end = 0
    for start, stop in sorted(intervals):
        if start != end:
            raise ValueError("Tensor payload contains overlapping ranges or gaps")
        end = stop
    if end != payload_bytes:
        raise ValueError("Safetensors payload has unaccounted bytes")
    result = {
        "tensor_count": len(header),
        "dtypes": sorted(dtypes),
        "metadata": metadata,
        "has_adapter_keys": any(".lora_" in name for name in header),
    }
    if include_tensors:
        result["tensors"] = {
            name: {"shape": tensor["shape"], "dtype": tensor["dtype"]}
            for name, tensor in header.items()
        }
    return result


def content_hash(filename: str | Path) -> str:
    """Hash a complete file and reject a detectable concurrent modification."""
    filename = Path(filename)
    with filename.open("rb") as stream:
        before = os.fstat(stream.fileno())
        digest = hashlib.sha256()
        while chunk := stream.read(4 * 1024 * 1024):
            digest.update(chunk)
        after = os.fstat(stream.fileno())

    def signature(stat):
        return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)

    if signature(before) != signature(after) or signature(after) != signature(filename.stat()):
        raise ValueError(f"File changed while hashing: {filename}")
    return digest.hexdigest()


def conversion_cache_key(source_sha256: str, converter: str, settings: dict) -> str:
    """A conversion is reusable only for identical source, converter version, and settings."""
    if not re.fullmatch(r"[0-9a-f]{64}", source_sha256) or not converter.strip():
        raise ValueError("A full source SHA-256 and versioned converter identity are required")
    identity = json.dumps(
        {"source": source_sha256, "converter": converter, "settings": settings},
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(identity.encode()).hexdigest()


def scan_model_library(roots, *, hash_duplicates=False):
    """Inventory explicitly supplied paths, following aliases without recursion cycles.

    Physical aliases are recognized cheaply. Different files of equal size are
    never called duplicates unless optional complete SHA-256 verification agrees.
    No model file is created, removed, moved, converted, loaded, or downloaded.
    """
    roots = [Path(root).expanduser().absolute() for root in roots]
    if not roots:
        raise ValueError("Specify at least one model root or file")
    assets, errors, directory_aliases = [], [], []
    directories, files, seen_paths = {}, {}, set()
    pending = list(reversed(roots))
    while pending:
        candidate = pending.pop()
        name = str(candidate)
        if name in seen_paths:
            continue
        seen_paths.add(name)
        try:
            stat = candidate.stat()
            identity = (stat.st_dev, stat.st_ino)
            if candidate.is_dir():
                if identity in directories:
                    directory_aliases.append(
                        {"path": name, "same_directory_as": directories[identity]}
                    )
                    continue
                directories[identity] = name
                pending.extend(
                    reversed(
                        sorted(
                            (
                                child
                                for child in candidate.iterdir()
                                if child.name not in {".git", ".cache"}
                            ),
                            key=str,
                        )
                    )
                )
                continue
            if not candidate.is_file() or candidate.name.startswith("._"):
                continue
            if candidate.suffix.lower() not in WEIGHT_SUFFIXES:
                continue
            if identity in files:
                files[identity]["paths"].append(name)
                continue
            record = {
                "paths": [name],
                "resolved_path": str(candidate.resolve()),
                "physical_identity": list(identity),
                "size_bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "compatibility": "not_evaluated",
                "format": candidate.suffix.lower().lstrip("."),
            }
            if candidate.suffix.lower() == ".safetensors":
                try:
                    record["header"] = inspect_safetensors_header(candidate)
                    record["header_valid"] = True
                except (ValueError, OSError) as exc:
                    record.update(header_valid=False, header_error=str(exc))
            assets.append(record)
            files[identity] = record
        except OSError as exc:
            errors.append({"path": name, "error": str(exc)})
    by_size = defaultdict(list)
    for asset in assets:
        by_size[asset["size_bytes"]].append(asset)
    duplicates = []
    if hash_duplicates:
        for candidates in by_size.values():
            if len(candidates) < 2:
                continue
            by_hash = defaultdict(list)
            for asset in candidates:
                try:
                    current = Path(asset["resolved_path"]).stat()
                    if (
                        [current.st_dev, current.st_ino] != asset["physical_identity"]
                        or current.st_size != asset["size_bytes"]
                        or current.st_mtime_ns != asset["mtime_ns"]
                    ):
                        raise ValueError(
                            "File changed since header inspection; rescan before hashing"
                        )
                    sha = content_hash(asset["resolved_path"])
                    asset["sha256"] = sha
                    by_hash[sha].append(asset)
                except (ValueError, OSError) as exc:
                    errors.append({"path": asset["resolved_path"], "error": str(exc)})
            for sha, group in by_hash.items():
                if len(group) > 1:
                    duplicates.append(
                        {
                            "sha256": sha,
                            "files": [asset["resolved_path"] for asset in group],
                            "duplicate_file_bytes": (len(group) - 1) * group[0]["size_bytes"],
                        }
                    )
    return {
        "format": "weetodd-model-library-inventory-v1",
        "roots": list(map(str, roots)),
        "read_only": True,
        "assets": assets,
        "directory_aliases": directory_aliases,
        "verified_content_duplicates": duplicates,
        "errors": errors,
        "distinct_physical_files": len(assets),
        "distinct_file_bytes": sum(asset["size_bytes"] for asset in assets),
        "hash_duplicates_requested": hash_duplicates,
        "storage_note": "Logical file bytes; APFS clones/compression may share physical storage",
    }
