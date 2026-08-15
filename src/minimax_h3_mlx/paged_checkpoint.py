"""Block-aligned, file-backed checkpoint support for low-memory H3 inference."""

from __future__ import annotations

import gc
import hashlib
import json
import os
import shutil
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
from mlx.utils import tree_flatten, tree_unflatten

from .load import shard_paths
from .page_prefetch import SequentialPagePrefetch

PAGED_FORMAT = "weetodd-h3-paged-v1"
PAGED_MANIFEST = "paged_manifest.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _block_index(key: str) -> int | None:
    if not key.startswith("blocks."):
        return None
    parts = key.split(".", 2)
    if len(parts) != 3 or not parts[1].isdigit():
        raise ValueError(f"Malformed H3 transformer block key: {key!r}.")
    return int(parts[1])


@dataclass(frozen=True)
class PageRecord:
    file: str
    tensor_count: int
    tensor_bytes: int
    sha256: str

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> PageRecord:
        return cls(
            file=str(value["file"]),
            tensor_count=int(value["tensor_count"]),
            tensor_bytes=int(value["tensor_bytes"]),
            sha256=str(value["sha256"]),
        )


@dataclass(frozen=True)
class PagedCheckpointManifest:
    root: Path
    num_blocks: int
    source_tensor_bytes: int
    fixed: PageRecord
    blocks: tuple[PageRecord, ...]

    @classmethod
    def load(cls, root: str | Path, *, verify_hashes: bool = False) -> PagedCheckpointManifest:
        root = Path(root).expanduser().resolve()
        manifest_path = root / PAGED_MANIFEST
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Paged H3 manifest not found: {manifest_path}")
        raw = json.loads(manifest_path.read_text())
        if raw.get("format") != PAGED_FORMAT:
            raise ValueError(
                f"Unsupported paged H3 format {raw.get('format')!r}; expected {PAGED_FORMAT!r}."
            )
        blocks = tuple(PageRecord.from_dict(item) for item in raw["blocks"])
        num_blocks = int(raw["num_blocks"])
        if len(blocks) != num_blocks:
            raise ValueError(
                f"Paged H3 manifest declares {num_blocks} blocks but lists {len(blocks)} pages."
            )
        manifest = cls(
            root=root,
            num_blocks=num_blocks,
            source_tensor_bytes=int(raw["source_tensor_bytes"]),
            fixed=PageRecord.from_dict(raw["fixed"]),
            blocks=blocks,
        )
        manifest.validate_files(verify_hashes=verify_hashes)
        return manifest

    def validate_files(self, *, verify_hashes: bool = False) -> None:
        for record in (self.fixed, *self.blocks):
            path = (self.root / record.file).resolve()
            try:
                path.relative_to(self.root)
            except ValueError as error:
                raise ValueError(
                    f"Paged H3 page escapes the checkpoint root: {record.file!r}."
                ) from error

            if not path.is_file():
                raise FileNotFoundError(f"Paged H3 page not found: {path}")
            if verify_hashes and _sha256(path) != record.sha256:
                raise ValueError(f"Paged H3 page hash differs from the manifest: {path}")


class PagedTensorStore:
    """Load one lazy page at a time and explicitly reclaim its materialized allocation."""

    def __init__(self, manifest: PagedCheckpointManifest):
        self.manifest = manifest
        self._active: dict[str, mx.array] | None = None
        self._active_name: str | None = None
        self.peak_page_bytes = 0
        self.pages_loaded = 0

    @property
    def active_page(self) -> str | None:
        return self._active_name

    def load_fixed(self) -> dict[str, mx.array]:
        return self._load(self.manifest.fixed)

    def load_block(self, index: int) -> dict[str, mx.array]:
        if not 0 <= index < self.manifest.num_blocks:
            upper = self.manifest.num_blocks - 1
            raise IndexError(f"H3 block index {index} is outside 0..{upper}.")
        return self._load(self.manifest.blocks[index])

    def load_block_window(self, start: int, size: int) -> dict[str, mx.array]:
        """Load consecutive block pages as one bounded materialization window."""
        if size < 1:
            raise ValueError("Paged H3 block window size must be positive.")
        stop = min(start + size, self.manifest.num_blocks)
        if not 0 <= start < stop:
            upper = self.manifest.num_blocks - 1
            raise IndexError(f"H3 block window start {start} is outside 0..{upper}.")
        return self._load_many(self.manifest.blocks[start:stop])

    def _load(self, record: PageRecord) -> dict[str, mx.array]:
        return self._load_many((record,))

    def _load_many(self, records: tuple[PageRecord, ...]) -> dict[str, mx.array]:
        if self._active is not None:
            raise RuntimeError(
                f"Paged H3 page {self._active_name!r} is still active; release it before loading "
                f"{records[0].file!r}."
            )
        values: dict[str, mx.array] = {}
        actual_bytes = 0
        for record in records:
            loaded = dict(mx.load(str(self.manifest.root / record.file)))
            if len(loaded) != record.tensor_count:
                raise ValueError(
                    f"Paged H3 page {record.file!r} contains {len(loaded)} tensors; "
                    f"the manifest declares {record.tensor_count}."
                )
            page_bytes = sum(value.nbytes for value in loaded.values())
            if page_bytes != record.tensor_bytes:
                raise ValueError(
                    f"Paged H3 page {record.file!r} describes {page_bytes} tensor bytes; "
                    f"the manifest declares {record.tensor_bytes}."
                )
            overlap = values.keys() & loaded.keys()
            if overlap:
                raise ValueError(
                    f"Duplicate tensors across paged H3 window: {sorted(overlap)[:4]}."
                )
            values.update(loaded)
            actual_bytes += page_bytes
        self._active = values
        self._active_name = ",".join(record.file for record in records)
        self.peak_page_bytes = max(self.peak_page_bytes, actual_bytes)
        self.pages_loaded += len(records)
        return values

    def release(self) -> None:
        self._active = None
        self._active_name = None
        gc.collect()
        mx.clear_cache()


class PagedBlockExecutor:
    """Construct and retire H3 transformer blocks in bounded consecutive windows."""

    def __init__(
        self,
        manifest: PagedCheckpointManifest,
        config,
        quant_config,
        window_size: int = 4,
        prefetch: bool | None = None,
    ):
        if window_size < 1:
            raise ValueError("Paged H3 block window size must be positive.")
        self.manifest = manifest
        self.config = config
        self.quant_config = quant_config
        self.window_size = int(window_size)
        self.query_chunk_size: int | None = None
        self.store = PagedTensorStore(manifest)
        if prefetch is None:
            value = os.environ.get("WEETODD_H3_TRANSFORMER_PREFETCH", "0").strip().lower()
            prefetch = value not in {"0", "false", "no", "off"}
        self.prefetch = SequentialPagePrefetch(
            manifest.root,
            manifest.blocks,
            enabled=prefetch,
            thread_name="h3-transformer-prefetch",
            backend="darwin_advisory",
        )
        self.windows_materialized = 0
        self.window_setup_seconds = 0.0
        self.window_compute_seconds = 0.0
        self.lora_requests: list[tuple[Any, mx.array | None]] = []
        self.lora_timesteps: mx.array | None = None
        self.projection_backend = "mlx"
        self.projection_wrapped_by_block: dict[int, tuple[int, int]] = {}

    @property
    def num_blocks(self) -> int:
        return self.manifest.num_blocks

    @contextmanager
    def window(self, start: int):
        """Yield materialized blocks and guarantee their release after the caller completes x."""
        from .dit import TransformerBlock
        from .quantize import apply_block_quantization_structure

        stop = min(start + self.window_size, self.num_blocks)
        size = stop - start
        self.prefetch.wait(start, size)
        setup_started = time.perf_counter()
        values = self.store.load_block_window(start, self.window_size)
        blocks = []
        try:
            for index in range(start, stop):
                block = TransformerBlock(self.config)
                if self.quant_config is not None:
                    apply_block_quantization_structure(block, index, self.quant_config)
                prefix = f"blocks.{index}."
                local = {
                    key[len(prefix) :]: value
                    for key, value in values.items()
                    if key.startswith(prefix)
                }
                expected = {key for key, _ in tree_flatten(block.parameters())}
                missing = sorted(expected - local.keys())
                unexpected = sorted(local.keys() - expected)
                if missing or unexpected:
                    raise KeyError(
                        f"Paged H3 block {index} mismatch: {len(missing)} missing "
                        f"(e.g. {missing[:4]}), {len(unexpected)} unexpected "
                        f"(e.g. {unexpected[:4]})."
                    )
                block.update(tree_unflatten(list(local.items())))
                block.attn.query_chunk_size = self.query_chunk_size
                if self.projection_backend == "mpp_experimental":
                    from .projection import configure_block_projection_backend

                    self.projection_wrapped_by_block[index] = (
                        configure_block_projection_backend(block)
                    )
                if self.lora_requests:
                    from .lora import apply_paged_loras_to_block

                    apply_paged_loras_to_block(
                        block,
                        index,
                        self.lora_requests,
                        self.lora_timesteps,
                    )
                blocks.append(block)
            mx.eval(tuple(block.parameters() for block in blocks))
            self.windows_materialized += 1
            self.window_setup_seconds += time.perf_counter() - setup_started
            if stop < self.num_blocks:
                self.prefetch.start(
                    stop, min(self.window_size, self.num_blocks - stop)
                )
            compute_started = time.perf_counter()
            yield blocks
            self.window_compute_seconds += time.perf_counter() - compute_started
        finally:
            blocks.clear()
            values.clear()
            self.store.release()

    def close(self) -> None:
        self.prefetch.close()

    def report(self) -> dict[str, int | float | bool | str]:
        return {
            "format": PAGED_FORMAT,
            "window_size": self.window_size,
            "pages_loaded": self.store.pages_loaded,
            "peak_window_bytes": self.store.peak_page_bytes,
            "lora_count": len(self.lora_requests),
            "windows_materialized": self.windows_materialized,
            "projection_backend": self.projection_backend,
            "projection_wrapped": sum(
                counts[0] for counts in self.projection_wrapped_by_block.values()
            ),
            "projection_skipped": sum(
                counts[1] for counts in self.projection_wrapped_by_block.values()
            ),
            "window_setup_seconds": self.window_setup_seconds,
            "window_compute_seconds": self.window_compute_seconds,
            **self.prefetch.report(),
        }


def load_paged_dit(
    model_dir: str | Path,
    *,
    window_size: int = 4,
    verify_hashes: bool = False,
    prefetch: bool | None = None,
):
    """Load fixed H3 tensors and attach a bounded block executor for direct inference."""
    from .config import DiTConfig
    from .dit import MiniMaxH3DiT
    from .load import SKIP_KEYS
    from .quantize import QuantConfig, apply_quantization_structure

    manifest = PagedCheckpointManifest.load(model_dir, verify_hashes=verify_hashes)
    config = DiTConfig.from_json(manifest.root / "config.json")
    if config.num_layers != manifest.num_blocks:
        raise ValueError(
            f"Paged H3 config declares {config.num_layers} blocks but the manifest has "
            f"{manifest.num_blocks}."
        )
    quant_config = None
    quant_path = manifest.root / "quant_config.json"
    if quant_path.is_file():
        recipe = json.loads(quant_path.read_text())
        quant_config = QuantConfig(
            bits=recipe["bits"],
            group_size=recipe["group_size"],
            quantize_adaln=recipe.get("quantize_adaln", False),
            adaln_bits=recipe.get("adaln_bits") or 8,
            overrides={
                str(path): int(bits) for path, bits in recipe.get("overrides", {}).items()
            },
            quantize_core=recipe.get("quantize_core", True),
        )

    model = MiniMaxH3DiT(config)
    if quant_config is not None:
        apply_quantization_structure(model, quant_config)
    expected = {
        key
        for key, _ in tree_flatten(model.parameters())
        if not key.startswith("blocks.") and key not in SKIP_KEYS
    }
    store = PagedTensorStore(manifest)
    fixed = store.load_fixed()
    try:
        missing = sorted(expected - fixed.keys())
        unexpected = sorted(fixed.keys() - expected - set(SKIP_KEYS))
        if missing or unexpected:
            raise KeyError(
                f"Paged H3 fixed tensors mismatch: {len(missing)} missing "
                f"(e.g. {missing[:4]}), {len(unexpected)} unexpected "
                f"(e.g. {unexpected[:4]})."
            )
        model.update(
            tree_unflatten([(key, value) for key, value in fixed.items() if key in expected])
        )
        model.blocks = []
        mx.eval(model.parameters())
    finally:
        fixed.clear()
        store.release()
    model.paged_blocks = PagedBlockExecutor(
        manifest, config, quant_config, window_size, prefetch
    )
    return model


def convert_to_paged_checkpoint(
    source: str | Path,
    destination: str | Path,
    *,
    verify_output: bool = True,
) -> PagedCheckpointManifest:
    """Rewrite one H3 transformer as fixed tensors plus one safetensors page per block."""
    source = Path(source).expanduser().resolve()
    destination = Path(destination).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"Paged H3 destination already exists: {destination}")

    sources: dict[str, Path] = {}
    tensor_bytes: dict[str, int] = {}
    block_keys: dict[int, list[str]] = {}
    fixed_keys: list[str] = []
    for shard in shard_paths(source):
        values = mx.load(str(shard))
        for key, value in values.items():
            if key in sources:
                raise ValueError(f"Duplicate tensor key across H3 source shards: {key!r}.")
            sources[key] = shard
            tensor_bytes[key] = value.nbytes
            index = _block_index(key)
            if index is None:
                fixed_keys.append(key)
            else:
                block_keys.setdefault(index, []).append(key)

    if not block_keys:
        raise ValueError("The source checkpoint contains no `blocks.<index>` tensors.")
    expected = list(range(max(block_keys) + 1))
    if sorted(block_keys) != expected:
        raise ValueError(
            f"The source checkpoint block indexes are not contiguous: {sorted(block_keys)}."
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        pages_dir = temporary / "pages"
        pages_dir.mkdir()

        def write_page(relative: Path, keys: list[str]) -> PageRecord:
            by_shard: dict[Path, list[str]] = {}
            for key in keys:
                by_shard.setdefault(sources[key], []).append(key)
            selected: dict[str, mx.array] = {}
            for shard, shard_keys in by_shard.items():
                loaded = mx.load(str(shard))
                selected.update({key: loaded[key] for key in shard_keys})
            path = temporary / relative
            mx.save_safetensors(str(path), selected)
            del selected
            gc.collect()
            mx.clear_cache()
            return PageRecord(
                file=str(relative),
                tensor_count=len(keys),
                tensor_bytes=sum(tensor_bytes[key] for key in keys),
                sha256=_sha256(path),
            )

        fixed = write_page(Path("pages/fixed.safetensors"), sorted(fixed_keys))
        blocks = tuple(
            write_page(Path(f"pages/block-{index:03d}.safetensors"), sorted(block_keys[index]))
            for index in expected
        )
        for name in ("config.json", "quant_config.json"):
            candidate = source / name
            if candidate.is_file():
                shutil.copy2(candidate, temporary / name)
        raw = {
            "format": PAGED_FORMAT,
            "num_blocks": len(blocks),
            "source": source.name,
            "source_tensor_bytes": sum(tensor_bytes.values()),
            "fixed": fixed.__dict__,
            "blocks": [record.__dict__ for record in blocks],
        }
        (temporary / PAGED_MANIFEST).write_text(json.dumps(raw, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return PagedCheckpointManifest.load(destination, verify_hashes=verify_output)
