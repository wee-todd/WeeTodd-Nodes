"""Header-only MiniMax H3 component validation and memory estimation."""

from __future__ import annotations

import json
import math
import struct
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

SUPPORTED_TASKS = frozenset({"t2va", "fl2va", "ref2va"})
COMPONENT_NAMES = (
    "transformer",
    "text_encoder",
    "processor",
    "tokenizer",
    "video_vae",
    "audio_vae",
)
WEIGHT_COMPONENTS = frozenset({"transformer", "text_encoder", "video_vae", "audio_vae"})
DTYPE_BYTES = {
    "BOOL": 1,
    "I8": 1,
    "U8": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "I64": 8,
    "U64": 8,
    "F64": 8,
}


@dataclass(frozen=True)
class H3ComponentSetSpec:
    """Immutable locations for one complete MiniMax H3 pipeline."""

    checkpoint: str
    task: str = "t2va"
    transformer: str | None = None
    text_encoder: str | None = None
    processor: str | None = None
    tokenizer: str | None = None
    video_vae: str | None = None
    audio_vae: str | None = None

    def resolved_paths(self) -> dict[str, Path]:
        root = Path(self.checkpoint).expanduser()
        return {
            name: Path(getattr(self, name)).expanduser() if getattr(self, name) else root / name
            for name in COMPONENT_NAMES
        }


@dataclass(frozen=True)
class H3PreflightRequest:
    duration_seconds: float = 5.0
    steps: int = 16
    width: int = 640
    height: int = 384
    prompt_tokens: int = 512
    available_memory_gb: float = 0.0

    def validate(self) -> None:
        if not 5.0 <= self.duration_seconds <= 15.0:
            raise ValueError("Duration must be between 5 and 15 seconds.")
        if self.steps < 2:
            raise ValueError("Sampling steps must be at least 2.")
        if self.width < 32 or self.height < 32:
            raise ValueError("Width and height must be at least 32 pixels.")
        if self.width % 32 or self.height % 32:
            raise ValueError("Width and height must be divisible by 32.")
        if self.prompt_tokens < 1:
            raise ValueError("Prompt token estimate must be positive.")
        if self.available_memory_gb < 0:
            raise ValueError("Available memory must be zero or positive.")


@dataclass(frozen=True)
class SafetensorsHeader:
    tensor_count: int
    tensor_bytes: int
    dtypes: tuple[str, ...]
    metadata: dict[str, str]
    tensor_names: tuple[str, ...]
    adaln_bytes: int = 0


@dataclass(frozen=True)
class ComponentReport:
    name: str
    path: str
    files: tuple[str, ...]
    disk_bytes: int
    tensor_bytes: int
    tensor_count: int
    dtypes: tuple[str, ...]
    quantization: str
    adaln_bytes: int = 0


@dataclass(frozen=True)
class H3PreflightReport:
    task: str
    partition: str
    components: tuple[ComponentReport, ...]
    frames: int
    video_latent_frames: int
    audio_latent_frames: int
    prompt_rows: int
    video_rows: int
    audio_rows: int
    packed_rows: int
    adaln_cache_bytes: int
    packed_workspace_bytes: int
    video_decode_workspace_bytes: int
    qwen_stage_bytes: int
    transformer_load_stage_bytes: int
    transformer_sample_stage_bytes: int
    video_decode_stage_bytes: int
    audio_decode_stage_bytes: int
    staged_peak_bytes: int
    available_memory_bytes: int | None
    headroom_bytes: int | None
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)


def read_safetensors_header(path: str | Path) -> SafetensorsHeader:
    """Read only the JSON header of a safetensors file."""

    path = Path(path)
    size = path.stat().st_size
    with path.open("rb") as handle:
        prefix = handle.read(8)
        if len(prefix) != 8:
            raise ValueError(f"Safetensors file has no complete header prefix: {path}")
        header_size = struct.unpack("<Q", prefix)[0]
        if header_size < 2 or header_size > size - 8:
            raise ValueError(f"Safetensors file has an invalid header length: {path}")
        raw = handle.read(header_size)
    if len(raw) != header_size:
        raise ValueError(f"Safetensors file has a truncated header: {path}")
    try:
        header = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Safetensors file has invalid header JSON: {path}") from exc

    metadata = header.pop("__metadata__", {})
    if metadata is None:
        metadata = {}
    if not isinstance(metadata, dict):
        raise ValueError(f"Safetensors metadata must be an object: {path}")
    payload_capacity = size - 8 - header_size
    tensor_bytes = 0
    adaln_bytes = 0
    dtypes: set[str] = set()
    intervals: list[tuple[int, int, str]] = []
    for name, entry in header.items():
        if not isinstance(entry, dict):
            raise ValueError(f"Safetensors tensor entry is invalid for {name!r}: {path}")
        offsets = entry.get("data_offsets")
        if not isinstance(offsets, list) or len(offsets) != 2:
            raise ValueError(f"Safetensors tensor offsets are invalid for {name!r}: {path}")
        start, end = offsets
        if not isinstance(start, int) or not isinstance(end, int) or start < 0 or end < start:
            raise ValueError(f"Safetensors tensor offsets are invalid for {name!r}: {path}")
        payload_bytes = end - start
        if end > payload_capacity:
            raise ValueError(f"Safetensors tensor offsets exceed the file payload: {path}")
        shape = entry.get("shape")
        dtype = entry.get("dtype")
        if not isinstance(shape, list) or not all(
            isinstance(dimension, int) and dimension >= 0 for dimension in shape
        ):
            raise ValueError(f"Safetensors tensor shape is invalid for {name!r}: {path}")
        if isinstance(dtype, str) and dtype in DTYPE_BYTES:
            expected_bytes = math.prod(shape) * DTYPE_BYTES[dtype]
            if expected_bytes != payload_bytes:
                raise ValueError(f"Safetensors tensor byte count is invalid for {name!r}: {path}")
        tensor_bytes += payload_bytes
        intervals.append((start, end, name))
        if ".adaln_proj.linear." in name:
            adaln_bytes += payload_bytes
        if isinstance(dtype, str):
            dtypes.add(dtype)
    sorted_intervals = sorted(intervals)
    for index in range(1, len(sorted_intervals)):
        previous = sorted_intervals[index - 1]
        current = sorted_intervals[index]
        if current[0] < previous[1]:
            raise ValueError(
                f"Safetensors tensors {previous[2]!r} and {current[2]!r} overlap: {path}"
            )
    return SafetensorsHeader(
        tensor_count=len(header),
        tensor_bytes=tensor_bytes,
        dtypes=tuple(sorted(dtypes)),
        metadata={str(key): str(value) for key, value in metadata.items()},
        tensor_names=tuple(sorted(header)),
        adaln_bytes=adaln_bytes,
    )


def _read_json(path: Path, subject: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{subject} file not found: {path}")
    try:
        value = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"{subject} file contains invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{subject} file must contain a JSON object: {path}")
    return value


def _weight_files(path: Path, component: str) -> tuple[Path, ...]:
    if path.is_file():
        if path.suffix != ".safetensors":
            raise ValueError(f"{component} must use a safetensors file: {path}")
        return (path,)
    if not path.is_dir():
        raise FileNotFoundError(f"{component} path not found: {path}")

    # Compact MLX exports may colocate the Qwen encoder and both VAEs. Keep the
    # component report scoped to the file the compact text loader actually uses.
    compact_text_encoder = path / "text_encoder.safetensors"
    if component == "text_encoder" and compact_text_encoder.is_file():
        return (compact_text_encoder,)

    indexes = sorted(path.glob("*.safetensors.index.json"))
    if indexes:
        index = _read_json(indexes[0], f"{component} safetensors index")
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError(f"{component} safetensors index has no weight map: {indexes[0]}")
        names = sorted({str(name) for name in weight_map.values()})
        files = tuple(path / name for name in names)
        missing = [file.name for file in files if not file.is_file()]
        if missing:
            raise FileNotFoundError(
                f"{component} is missing indexed safetensors shards: {', '.join(missing)}"
            )
        return files

    files = tuple(sorted(path.rglob("*.safetensors")))
    if not files:
        raise FileNotFoundError(f"{component} has no safetensors weights: {path}")
    return files


def _validate_asset_directory(
    path: Path,
    component: str,
    *,
    allow_text_only_processor: bool = False,
) -> None:
    if not path.is_dir():
        raise FileNotFoundError(f"{component} directory not found: {path}")
    if component == "processor":
        candidates = ("preprocessor_config.json", "processor_config.json")
    else:
        candidates = ("tokenizer.json", "vocab.json")
    if not any((path / name).is_file() for name in candidates):
        if (
            component == "processor"
            and allow_text_only_processor
            and (path / "tokenizer.json").is_file()
        ):
            return
        joined = " or ".join(candidates)
        raise FileNotFoundError(f"{component} requires {joined}: {path}")


def _quantization(path: Path, headers: list[SafetensorsHeader], component: str) -> str:
    if path.is_dir() and (path / "quant_config.json").exists():
        recipe = _read_json(path / "quant_config.json", f"{component} quantization recipe")
        bits = recipe.get("bits")
        group_size = recipe.get("group_size")
        if bits not in {4, 5, 6, 8}:
            raise ValueError(f"{component} quantization bits must be 4, 5, 6, or 8, got {bits!r}.")
        if not isinstance(group_size, int) or group_size < 1:
            raise ValueError(f"{component} quantization group size must be positive.")
        return f"mlx-affine-{bits}bit-group-{group_size}"
    metadata = {key: value for header in headers for key, value in header.metadata.items()}
    for key in ("quantization", "quant_config", "format"):
        if key in metadata:
            return metadata[key]
    return "unquantized-or-self-describing"


def _validate_component_config(path: Path, component: str) -> None:
    if not path.is_dir():
        if component == "text_encoder":
            raise ValueError(
                "text_encoder must be a directory with its configuration and tokenizer context."
            )
        return
    config_path = path / "config.json"
    if component == "video_vae" and (path / "source" / "config.json").is_file():
        config_path = path / "source" / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"{component} config file not found: {config_path}")
    if component == "audio_vae" and not (path / "metadata.json").is_file():
        raise FileNotFoundError(f"audio_vae metadata file not found: {path / 'metadata.json'}")
    config = _read_json(config_path, f"{component} config")
    if component == "transformer":
        expected = {"latents_dim": 24, "audio_latents_dim": 32, "text_dim": 5120}
        for key, value in expected.items():
            if key in config and config[key] != value:
                raise ValueError(
                    f"transformer {key} must be {value}, got {config[key]!r}: {config_path}"
                )
    elif component == "video_vae":
        value = config.get("z_channels", config.get("latent_channels"))
        if value is not None and value != 24:
            raise ValueError(f"video_vae latent channels must be 24, got {value!r}.")
    elif component == "audio_vae":
        kwargs = config.get("kwargs", config)
        value = kwargs.get("vae_latent_channels", kwargs.get("latent_channels"))
        if value is not None and value != 32:
            raise ValueError(f"audio_vae latent channels must be 32, got {value!r}.")


def _component_report(
    name: str,
    path: Path,
    *,
    allow_text_only_processor: bool = False,
) -> ComponentReport:
    if name not in WEIGHT_COMPONENTS:
        _validate_asset_directory(
            path,
            name,
            allow_text_only_processor=allow_text_only_processor,
        )
        asset_names = {
            "processor": (
                "preprocessor_config.json",
                "processor_config.json",
                "tokenizer.json",
            ),
            "tokenizer": (
                "tokenizer.json",
                "tokenizer_config.json",
                "vocab.json",
                "merges.txt",
                "special_tokens_map.json",
                "added_tokens.json",
            ),
        }
        files = tuple(
            path / filename for filename in asset_names[name] if (path / filename).is_file()
        )
        return ComponentReport(
            name=name,
            path=str(path),
            files=tuple(file.name for file in files),
            disk_bytes=sum(file.stat().st_size for file in files),
            tensor_bytes=0,
            tensor_count=0,
            dtypes=(),
            quantization="not-applicable",
        )

    _validate_component_config(path, name)
    files = _weight_files(path, name)
    headers = [read_safetensors_header(file) for file in files]
    if path.is_file():
        metadata = {key: value for header in headers for key, value in header.metadata.items()}
        tensor_names = {tensor for header in headers for tensor in header.tensor_names}
        if name == "transformer" and "adaln_t_table" not in tensor_names:
            raise ValueError(
                "Single-file transformer is not an MLX-ready pruned H3 export: "
                "adaln_t_table is missing."
            )
        required_metadata = {
            "video_vae": "minimax_h3_video_vae",
            "audio_vae": "minimax_h3_audio_vae",
        }
        metadata_key = required_metadata.get(name)
        if metadata_key is not None and metadata_key not in metadata:
            raise ValueError(
                f"Single-file {name} is not self-describing: metadata key "
                f"{metadata_key!r} is missing."
            )
        if name == "video_vae" and metadata_key is not None:
            from minimax_h3_mlx.video_vae_checkpoint import validate_video_vae_wrapper

            try:
                wrapper = json.loads(metadata[metadata_key])
            except (TypeError, json.JSONDecodeError) as exc:
                raise ValueError("Single-file video_vae metadata is not valid JSON.") from exc
            validate_video_vae_wrapper(wrapper)
    return ComponentReport(
        name=name,
        path=str(path),
        files=tuple(file.name for file in files),
        disk_bytes=sum(file.stat().st_size for file in files),
        tensor_bytes=sum(header.tensor_bytes for header in headers),
        tensor_count=sum(header.tensor_count for header in headers),
        dtypes=tuple(sorted({dtype for header in headers for dtype in header.dtypes})),
        quantization=_quantization(path, headers, name),
        adaln_bytes=sum(header.adaln_bytes for header in headers),
    )


def _aligned_frames(duration_seconds: float) -> int:
    frames = int(round(duration_seconds * 24))
    while frames % 17 != 5:
        frames += 1
    return frames


def preflight_components(
    spec: H3ComponentSetSpec,
    request: H3PreflightRequest,
) -> H3PreflightReport:
    """Validate a complete component set and estimate staged memory without loading tensors."""

    request.validate()
    if spec.task not in SUPPORTED_TASKS:
        raise ValueError(
            f"MiniMax H3 task must be one of {sorted(SUPPORTED_TASKS)}, got {spec.task!r}."
        )
    root = Path(spec.checkpoint).expanduser()
    if not root.is_dir():
        raise FileNotFoundError(f"MiniMax H3 checkpoint directory not found: {root}")
    manifest = _read_json(root / "model_index.json", "MiniMax H3 model manifest")
    missing_manifest_components = [name for name in COMPONENT_NAMES if name not in manifest]
    if missing_manifest_components:
        raise ValueError(
            "MiniMax H3 model manifest is missing component declarations: "
            + ", ".join(missing_manifest_components)
        )
    metadata = manifest.get("_minimax_h3")
    if not isinstance(metadata, dict):
        raise ValueError("MiniMax H3 model manifest has no _minimax_h3 task metadata.")
    tasks = metadata.get("tasks")
    if not isinstance(tasks, list) or not all(isinstance(task, str) for task in tasks):
        raise ValueError("MiniMax H3 model manifest has invalid task metadata.")
    if spec.task not in tasks:
        raise ValueError(
            f"Checkpoint does not support task {spec.task!r}; supported tasks: {tasks}."
        )
    partition = metadata.get("partition")
    if not isinstance(partition, str) or not partition:
        raise ValueError("MiniMax H3 model manifest has no partition name.")
    expected_partition = "ref2va" if spec.task == "ref2va" else "fl2va"
    if partition != expected_partition:
        raise ValueError(
            f"Checkpoint partition must be {expected_partition!r} for task {spec.task!r}, "
            f"got {partition!r}."
        )

    paths = spec.resolved_paths()
    components = tuple(
        _component_report(
            name,
            paths[name],
            allow_text_only_processor=spec.task == "t2va",
        )
        for name in COMPONENT_NAMES
    )
    by_name = {component.name: component for component in components}

    frames = _aligned_frames(request.duration_seconds)
    video_latent_frames = (frames - 5) // 17 * 5 + 2
    audio_latent_frames = round(frames / 24 * 40)
    video_rows = video_latent_frames * (request.height // 32) * (request.width // 32)
    audio_rows = audio_latent_frames * 2
    packed_rows = request.prompt_tokens + video_rows + audio_rows

    distinct_timesteps = 2 * (request.steps - 1) + 1
    adaln_cache_bytes = 50 * 6 * distinct_timesteps * 3 * 5376 * 2
    packed_workspace_bytes = packed_rows * 5376 * 2 * 8
    video_decode_workspace_bytes = (
        frames * request.width * request.height * 3 * 4
        + video_latent_frames * (request.width // 16) * (request.height // 16) * 24 * 2
    )

    qwen_stage = by_name["text_encoder"].tensor_bytes + request.prompt_tokens * 5120 * 2 * 4
    transformer = by_name["transformer"]
    transformer_load_stage = transformer.tensor_bytes + adaln_cache_bytes + packed_workspace_bytes
    sample_resident = max(0, transformer.tensor_bytes - transformer.adaln_bytes)
    transformer_sample_stage = sample_resident + adaln_cache_bytes + packed_workspace_bytes
    video_decode_stage = by_name["video_vae"].tensor_bytes + video_decode_workspace_bytes
    audio_seconds = frames / 24
    audio_decode_workspace = math.ceil(audio_seconds * 32000 * 2 * 4 * 2)
    audio_decode_stage = by_name["audio_vae"].tensor_bytes + audio_decode_workspace
    staged_peak = max(
        qwen_stage,
        transformer_load_stage,
        transformer_sample_stage,
        video_decode_stage,
        audio_decode_stage,
    )

    available = (
        round(request.available_memory_gb * 1_000_000_000)
        if request.available_memory_gb > 0
        else None
    )
    headroom = available - staged_peak if available is not None else None
    warnings: list[str] = [
        "Memory values are header-based estimates; Metal kernels and allocator fragmentation "
        "can increase peak memory."
    ]
    if request.width < 768 and request.height < 768:
        warnings.append("The selected canvas is an off-distribution wiring-test size.")
    if headroom is not None and headroom < 0:
        warnings.append("Estimated staged peak memory exceeds the supplied available memory.")
    if transformer.adaln_bytes == 0:
        warnings.append(
            "The transformer header exposes no removable AdaLN projection bytes; verify whether "
            "it is already pruned."
        )
    processor_path = paths["processor"]
    if not any(
        (processor_path / filename).is_file()
        for filename in ("preprocessor_config.json", "processor_config.json")
    ):
        warnings.append(
            "No processor configuration is present; this component set is valid only for "
            "text-only T2VA conditioning."
        )

    return H3PreflightReport(
        task=spec.task,
        partition=partition,
        components=components,
        frames=frames,
        video_latent_frames=video_latent_frames,
        audio_latent_frames=audio_latent_frames,
        prompt_rows=request.prompt_tokens,
        video_rows=video_rows,
        audio_rows=audio_rows,
        packed_rows=packed_rows,
        adaln_cache_bytes=adaln_cache_bytes,
        packed_workspace_bytes=packed_workspace_bytes,
        video_decode_workspace_bytes=video_decode_workspace_bytes,
        qwen_stage_bytes=qwen_stage,
        transformer_load_stage_bytes=transformer_load_stage,
        transformer_sample_stage_bytes=transformer_sample_stage,
        video_decode_stage_bytes=video_decode_stage,
        audio_decode_stage_bytes=audio_decode_stage,
        staged_peak_bytes=staged_peak,
        available_memory_bytes=available,
        headroom_bytes=headroom,
        warnings=tuple(warnings),
    )
