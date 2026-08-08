"""File-backed sequential layer storage for the text-only H3 Qwen3-VL conditioner."""

from __future__ import annotations

import gc
import json
import os
import shutil
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten

from .paged_checkpoint import PagedTensorStore, PageRecord, _sha256

PAGED_QWEN_FORMAT = "weetodd-h3-qwen-paged-v1"
PAGED_QWEN_MANIFEST = "paged_text_encoder_manifest.json"


def _layer_index(key: str) -> int | None:
    prefix = "model.layers."
    if not key.startswith(prefix):
        return None
    rest = key[len(prefix) :]
    value = rest.split(".", 1)[0]
    if not value.isdigit():
        raise ValueError(f"Malformed Qwen text-layer tensor key: {key!r}.")
    return int(value)


@dataclass(frozen=True)
class PagedTextEncoderManifest:
    root: Path
    num_blocks: int
    source_tensor_bytes: int
    fixed: PageRecord
    layers: tuple[PageRecord, ...]
    skipped_visual_bytes: int = 0

    @property
    def blocks(self) -> tuple[PageRecord, ...]:
        """Expose the generic paged-store record contract."""
        return self.layers

    @classmethod
    def load(cls, root: str | Path, *, verify_hashes: bool = False):
        root = Path(root).expanduser().resolve()
        path = root / PAGED_QWEN_MANIFEST
        if not path.is_file():
            raise FileNotFoundError(f"Paged H3 text-encoder manifest not found: {path}")
        raw = json.loads(path.read_text())
        if raw.get("format") != PAGED_QWEN_FORMAT:
            raise ValueError(
                f"Unsupported paged H3 text format {raw.get('format')!r}; "
                f"expected {PAGED_QWEN_FORMAT!r}."
            )
        layers = tuple(PageRecord.from_dict(item) for item in raw["layers"])
        num_layers = int(raw["num_layers"])
        if len(layers) != num_layers:
            raise ValueError(
                f"Paged H3 text manifest declares {num_layers} layers but lists "
                f"{len(layers)} pages."
            )
        manifest = cls(
            root=root,
            num_blocks=num_layers,
            source_tensor_bytes=int(raw["source_tensor_bytes"]),
            fixed=PageRecord.from_dict(raw["fixed"]),
            layers=layers,
            skipped_visual_bytes=int(raw.get("skipped_visual_bytes", 0)),
        )
        for record in (manifest.fixed, *manifest.layers):
            page = (root / record.file).resolve()
            try:
                page.relative_to(root)
            except ValueError as error:
                raise ValueError(
                    f"Paged H3 text page escapes the checkpoint root: {record.file!r}."
                ) from error
            if not page.is_file():
                raise FileNotFoundError(f"Paged H3 text page not found: {page}")
            if verify_hashes and _sha256(page) != record.sha256:
                raise ValueError(f"Paged H3 text page hash differs from the manifest: {page}")
        return manifest


class PagedTextLayerExecutor:
    """Materialize and retire one truncated Qwen decoder layer at a time."""

    def __init__(self, manifest: PagedTextEncoderManifest, text_config):
        self.manifest = manifest
        self.text_config = text_config
        self.store = PagedTensorStore(manifest)

    @property
    def num_layers(self) -> int:
        return self.manifest.num_blocks

    @contextmanager
    def layer(self, index: int):
        from mlx_vlm.models.qwen3_vl.language import Qwen3VLDecoderLayer

        values = self.store.load_block(index)
        layer = Qwen3VLDecoderLayer(self.text_config, index)
        prefix = f"model.layers.{index}."
        local = {
            key[len(prefix) :]: value for key, value in values.items() if key.startswith(prefix)
        }
        quantized = {
            key[: -len(".scales")] for key in local if key.endswith(".scales")
        }
        if quantized:
            nn.quantize(
                layer,
                group_size=64,
                bits=8,
                mode="affine",
                class_predicate=lambda path, _module: path in quantized,
            )
        expected = {key for key, _ in tree_flatten(layer.parameters())}
        missing = sorted(expected - local.keys())
        unexpected = sorted(local.keys() - expected)
        try:
            if missing or unexpected:
                raise KeyError(
                    f"Paged H3 text layer {index} mismatch: {len(missing)} missing "
                    f"(e.g. {missing[:4]}), {len(unexpected)} unexpected "
                    f"(e.g. {unexpected[:4]})."
                )
            layer.update(tree_unflatten(list(local.items())))
            mx.eval(layer.parameters())
            yield layer
        finally:
            local.clear()
            values.clear()
            del layer
            self.store.release()

    def report(self) -> dict[str, int | str]:
        return {
            "format": PAGED_QWEN_FORMAT,
            "layers_loaded": self.store.pages_loaded,
            "peak_layer_bytes": self.store.peak_page_bytes,
            "fixed_bytes": self.manifest.fixed.tensor_bytes,
        }


def convert_to_paged_text_encoder(
    source: str | Path,
    destination: str | Path,
    *,
    num_layers: int = 50,
    verify_output: bool = True,
    architecture_config: str | Path | None = None,
) -> PagedTextEncoderManifest:
    """Write the text-only Qwen subset as one fixed page plus sequential layer pages."""
    source = Path(source).expanduser().resolve()
    destination = Path(destination).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"Paged H3 text destination already exists: {destination}")
    checkpoint = source / "text_encoder.safetensors" if source.is_dir() else source
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Compact H3 text encoder not found: {checkpoint}")

    scanned = dict(mx.load(str(checkpoint)))
    tensor_bytes = {key: value.nbytes for key, value in scanned.items()}
    keys = tuple(scanned)
    scanned.clear()
    layers: dict[int, list[str]] = {}
    fixed: list[str] = []
    skipped_visual_bytes = 0
    for key in keys:
        if key.startswith("visual."):
            skipped_visual_bytes += tensor_bytes[key]
            continue
        index = _layer_index(key)
        if index is not None:
            if index < num_layers:
                layers.setdefault(index, []).append(key)
            continue
        if key.startswith("model."):
            fixed.append(key)
    expected = list(range(num_layers))
    if sorted(layers) != expected:
        raise ValueError(
            f"The compact Qwen checkpoint does not contain layers 0..{num_layers - 1}: "
            f"found {sorted(layers)}."
        )

    config_source = source / "config.json" if source.is_dir() else source.parent / "config.json"
    if not config_source.is_file():
        raise FileNotFoundError(f"Paged H3 text conversion needs a model config: {config_source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        pages = temporary / "pages"
        pages.mkdir()

        def write_page(relative: Path, keys: list[str]) -> PageRecord:
            loaded = dict(mx.load(str(checkpoint)))
            selected = {key: loaded[key] for key in keys}
            page = temporary / relative
            mx.save_safetensors(str(page), selected)
            selected.clear()
            loaded.clear()
            gc.collect()
            mx.clear_cache()
            return PageRecord(
                file=str(relative),
                tensor_count=len(keys),
                tensor_bytes=sum(tensor_bytes[key] for key in keys),
                sha256=_sha256(page),
            )

        fixed_record = write_page(Path("pages/fixed.safetensors"), sorted(fixed))
        layer_records = tuple(
            write_page(Path(f"pages/layer-{index:03d}.safetensors"), sorted(layers[index]))
            for index in expected
        )
        shutil.copy2(config_source, temporary / "config.json")
        if architecture_config is not None:
            architecture_config = Path(architecture_config).expanduser().resolve()
            raw_config = json.loads(architecture_config.read_text())
            if "text_config" not in raw_config:
                raise ValueError(
                    "The H3 text architecture config must contain a `text_config` object."
                )
            shutil.copy2(architecture_config, temporary / "architecture_config.json")
        raw = {
            "format": PAGED_QWEN_FORMAT,
            "num_layers": num_layers,
            "source": checkpoint.name,
            "source_tensor_bytes": sum(tensor_bytes.values()),
            "skipped_visual_bytes": skipped_visual_bytes,
            "fixed": fixed_record.__dict__,
            "layers": [record.__dict__ for record in layer_records],
        }
        (temporary / PAGED_QWEN_MANIFEST).write_text(
            json.dumps(raw, indent=2, sort_keys=True) + "\n"
        )
        os.replace(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return PagedTextEncoderManifest.load(destination, verify_hashes=verify_output)
