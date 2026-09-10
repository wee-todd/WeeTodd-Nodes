"""Bounded, atomic, non-pickle text-feature cache. Media requests deliberately bypass it."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path
from threading import RLock

_LOCK = RLock()
_POLICY = "h3-text-only-v1"
MAX_BYTES = 1024**3


def default_cache_directory():
    try:
        import folder_paths

        return Path(folder_paths.get_user_directory()) / "weetodd" / "h3-conditioning-cache"
    except ImportError:
        return Path.home() / ".cache" / "weetodd" / "h3-conditioning"


def cache_key(spec, prompt, task):
    identity = {
        "policy": _POLICY,
        "mlx": version("mlx"),
        "prompt": prompt,
        "task": task,
        "spec": asdict(spec),
        "files": [],
    }
    for location in sorted(
        {spec.text_encoder, spec.processor, spec.tokenizer, spec.config_path} - {None}
    ):
        root = Path(location).expanduser().resolve()
        files = [root] if root.is_file() else sorted(root.rglob("*"))
        from minimax_h3_mlx.dt_source import dt_source_files

        files.extend(dt_source_files(root))
        for path in files:
            if path.is_file() and not path.name.startswith("."):
                stat = path.stat()
                item = [str(path), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns]
                # Hash small policy/tokenizer/config files, never rescan huge weights per prompt.
                if stat.st_size <= 4 * 1024**2 and path.suffix != ".safetensors":
                    item.append(hashlib.sha256(path.read_bytes()).hexdigest())
                identity["files"].append(item)
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def read_features(directory, key):
    import mlx.core as mx
    import numpy as np

    path = Path(directory) / f"{key}.safetensors"
    with _LOCK:
        if not path.is_file() or path.is_symlink() or path.stat().st_size > MAX_BYTES:
            return None
        values, metadata = mx.load(str(path), return_metadata=True)
        if metadata.get("key") != key or metadata.get("policy") != _POLICY:
            return None
        if set(values) != {"embeddings", "token_tags"}:
            return None
        embeddings, tags = values["embeddings"], values["token_tags"]
        if tags.ndim != 1 or embeddings.ndim not in {2, 3} or embeddings.shape[-2] != tags.shape[0]:
            return None
        mx.eval(embeddings, tags)
        if not bool(mx.all(mx.isfinite(embeddings)).item()):
            return None
        os.utime(path, None)
        return embeddings, np.asarray(tags)


def write_features(directory, key, embeddings, token_tags):
    import mlx.core as mx

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if embeddings.nbytes + token_tags.nbytes > MAX_BYTES:
        return
    with _LOCK:
        descriptor, temporary = tempfile.mkstemp(
            prefix=".features-", suffix=".safetensors", dir=directory
        )
        os.close(descriptor)
        try:
            mx.save_safetensors(
                temporary,
                {"embeddings": embeddings, "token_tags": mx.array(token_tags)},
                metadata={"key": key, "policy": _POLICY},
            )
            os.replace(temporary, directory / f"{key}.safetensors")
        finally:
            Path(temporary).unlink(missing_ok=True)
        # Only prune files owned by this cache, never arbitrary user files.
        files = [
            p
            for p in directory.glob("*.safetensors")
            if len(p.stem) == 64
            and all(c in "0123456789abcdef" for c in p.stem)
            and not p.is_symlink()
        ]
        files.sort(key=lambda p: p.stat().st_mtime_ns)
        total = sum(p.stat().st_size for p in files)
        for path in files:
            if total <= MAX_BYTES:
                break
            total -= path.stat().st_size
            path.unlink(missing_ok=True)
