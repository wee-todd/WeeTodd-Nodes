"""File-backed sequential layer storage for the H3 Qwen3-VL conditioner."""

from __future__ import annotations

import json
import mmap
import os
import shutil
import struct
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten
from safetensors import safe_open

from .page_prefetch import SequentialPagePrefetch
from .paged_checkpoint import PagedTensorStore, PageRecord, _sha256

PAGED_QWEN_FORMAT = "weetodd-h3-qwen-paged-v1"
PAGED_QWEN_VISION_FORMAT = "weetodd-h3-qwen-paged-v2"
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
    vision: PageRecord | None = None

    @property
    def supports_vision(self) -> bool:
        return self.vision is not None

    @property
    def format(self) -> str:
        return PAGED_QWEN_VISION_FORMAT if self.supports_vision else PAGED_QWEN_FORMAT

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
        if raw.get("format") not in {PAGED_QWEN_FORMAT, PAGED_QWEN_VISION_FORMAT}:
            raise ValueError(
                f"Unsupported paged H3 text format {raw.get('format')!r}; "
                f"expected {PAGED_QWEN_FORMAT!r}."
            )
        vision = raw.get("vision")
        if raw["format"] == PAGED_QWEN_VISION_FORMAT:
            if (
                not vision
                or int(vision.get("tensor_count", 0)) <= 0
                or int(vision.get("tensor_bytes", 0)) <= 0
            ):
                raise ValueError("Paged H3 v2 requires a nonempty vision page.")
        elif vision is not None:
            raise ValueError("Paged H3 v1 cannot contain a vision page; use v2.")
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
            vision=PageRecord.from_dict(vision) if vision is not None else None,
        )
        records = (manifest.fixed, *manifest.layers)
        if manifest.vision is not None:
            records += (manifest.vision,)
        for record in records:
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

    def __init__(
        self,
        manifest: PagedTextEncoderManifest,
        text_config,
        *,
        prefetch: bool | None = None,
        dtype: mx.Dtype | None = None,
    ):
        self.manifest = manifest
        self.text_config = text_config
        self.dtype = dtype
        self.store = PagedTensorStore(manifest)
        if prefetch is None:
            # Read-ahead improves the cold-storage case by overlapping the next page with
            # current-layer compute, but it adds memory-bandwidth work once macOS already has
            # every page cached. Keep it opt-in until the runtime can identify that state.
            value = os.environ.get("WEETODD_H3_QWEN_PREFETCH", "0").strip().lower()
            prefetch = value not in {"0", "false", "no", "off"}
        self.prefetch = SequentialPagePrefetch(
            manifest.root,
            manifest.layers,
            enabled=prefetch,
            thread_name="h3-qwen-prefetch",
        )

    @property
    def num_layers(self) -> int:
        return self.manifest.num_blocks

    @contextmanager
    def layer(self, index: int):
        from mlx_vlm.models.qwen3_vl.language import Qwen3VLDecoderLayer

        # A failed speculative read is not fatal. The normal mapped load below remains the source
        # of truth and reports any real checkpoint error with its existing detailed message.
        self.prefetch.wait(index)
        values = {}
        local = {}
        layer = None
        try:
            values = self.store.load_block(index)
            layer = Qwen3VLDecoderLayer(self.text_config, index)
            prefix = f"model.layers.{index}."
            local = {
                key[len(prefix) :]: value for key, value in values.items() if key.startswith(prefix)
            }
            quantized = {key[: -len(".scales")] for key in local if key.endswith(".scales")}
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
            if missing or unexpected:
                raise KeyError(
                    f"Paged H3 text layer {index} mismatch: {len(missing)} missing "
                    f"(e.g. {missing[:4]}), {len(unexpected)} unexpected "
                    f"(e.g. {unexpected[:4]})."
                )
            if self.dtype is not None:
                local = {
                    key: value if value.dtype == mx.uint32 else value.astype(self.dtype)
                    for key, value in local.items()
                }
            layer.update(tree_unflatten(list(local.items())))
            mx.eval(layer.parameters())
            self.prefetch.start(index + 1)
            yield layer
        finally:
            local.clear()
            values.clear()
            del layer
            self.store.release()

    def close(self) -> None:
        self.prefetch.close()

    def report(self) -> dict[str, int | float | bool | str]:
        prefetch = self.prefetch.report()
        prefetch["prefetch_max_page_bytes"] = prefetch.pop("prefetch_max_window_bytes")
        return {
            "format": self.manifest.format,
            "layers_loaded": self.store.pages_loaded,
            "peak_layer_bytes": self.store.peak_page_bytes,
            "fixed_bytes": self.manifest.fixed.tensor_bytes,
            **prefetch,
        }


def convert_to_paged_text_encoder(
    source: str | Path,
    destination: str | Path,
    *,
    num_layers: int = 50,
    verify_output: bool = True,
    architecture_config: str | Path | None = None,
    include_vision: bool = False,
) -> PagedTextEncoderManifest:
    """Copy compact Qwen tensors to bounded pages, optionally retaining the vision tower."""
    source = Path(source).expanduser().resolve()
    destination = Path(destination).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"Paged H3 text destination already exists: {destination}")
    checkpoint = source / "text_encoder.safetensors" if source.is_dir() else source
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Compact H3 text encoder not found: {checkpoint}")

    # Validate the safetensors index without materializing any source arrays. Copy mapped byte
    # ranges per page below; this preserves BF16 and packed Q8 storage exactly.
    with safe_open(checkpoint, framework="np") as handle:
        keys = tuple(handle.keys())
    with checkpoint.open("rb") as handle:
        header_size = struct.unpack("<Q", handle.read(8))[0]
        source_header = json.loads(handle.read(header_size))
    data_start = 8 + header_size
    tensor_bytes = {
        key: source_header[key]["data_offsets"][1] - source_header[key]["data_offsets"][0]
        for key in keys
    }
    layers: dict[int, list[str]] = {}
    fixed: list[str] = []
    visual: list[str] = []
    skipped_visual_bytes = 0
    for key in keys:
        if key.startswith("visual."):
            if include_vision:
                visual.append(key)
            else:
                skipped_visual_bytes += tensor_bytes[key]
            continue
        index = _layer_index(key)
        if index is not None:
            if index < num_layers:
                layers.setdefault(index, []).append(key)
            continue
        if key.startswith("model."):
            fixed.append(key)
    if include_vision and not visual:
        raise ValueError("Vision paging requires visual tensors in the source checkpoint.")
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
            page = temporary / relative
            header = {}
            offset = 0
            for key in keys:
                header[key] = {
                    **source_header[key],
                    "data_offsets": [offset, offset + tensor_bytes[key]],
                }
                offset += tensor_bytes[key]
            encoded = json.dumps(header, separators=(",", ":")).encode()
            encoded += b" " * (-len(encoded) % 8)
            with checkpoint.open("rb") as source_file, page.open("wb") as output:
                output.write(struct.pack("<Q", len(encoded)))
                output.write(encoded)
                with mmap.mmap(source_file.fileno(), 0, access=mmap.ACCESS_READ) as mapped:
                    for key in keys:
                        start, end = source_header[key]["data_offsets"]
                        for position in range(
                            data_start + start, data_start + end, 8 * 1024 * 1024
                        ):
                            output.write(
                                mapped[position : min(position + 8 * 1024 * 1024, data_start + end)]
                            )
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
        vision_record = (
            write_page(Path("pages/vision.safetensors"), sorted(visual)) if include_vision else None
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
            "format": PAGED_QWEN_VISION_FORMAT if include_vision else PAGED_QWEN_FORMAT,
            "num_layers": num_layers,
            "source": checkpoint.name,
            "source_tensor_bytes": sum(tensor_bytes.values()),
            "skipped_visual_bytes": skipped_visual_bytes,
            "fixed": fixed_record.__dict__,
            "layers": [record.__dict__ for record in layer_records],
        }
        if vision_record is not None:
            raw["vision"] = vision_record.__dict__
        (temporary / PAGED_QWEN_MANIFEST).write_text(
            json.dumps(raw, indent=2, sort_keys=True) + "\n"
        )
        os.replace(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return PagedTextEncoderManifest.load(destination, verify_hashes=verify_output)
