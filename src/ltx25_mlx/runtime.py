"""Versioned LTX 2.5 split-component contracts and MLX lifecycle management."""

from __future__ import annotations

import gc
import hashlib
import inspect
import json
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from threading import RLock
from typing import Any

from safetensors import safe_open

from .gemma_pack import inspect_gemma4_pack

LTX25_CONFIG_MODES = ("distilled", "guided", "guided_hq")
LTX25_PROMPT_CONTEXTS = ("official_1024", "auto", "128", "256", "512", "1024")
LTX25_FEED_FORWARD_BACKENDS = (
    "reference_fp32",
    "mlx_fused_experimental",
    "bf16_mpp_experimental",
)
LTX25_DIFFVAE_OPTIMIZATIONS = (
    "combined",
    "metal_na3d_experimental",
    "metal_na3d_query_tiled_experimental",
    "deferred_stage4",
    "stage4_width_tiles",
)
LTX25_GENERATION_PRESETS = (
    "Custom",
    "Official parity — 768×512, 5 s, reference FP32, 8 ancestral + 3 deterministic",
    "High quality — 1920×1088, 5 s, reference FP32, 8 ancestral + 3 deterministic",
)
LTX25_DISTILLED_SIGMAS = (1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0)
LTX25_STAGE2_SIGMAS = (0.909375, 0.725, 0.421875, 0.0)
LTX25_CFG_PP_SCHEDULES = {
    "full": (0, 1, 2, 3, 4, 5, 6),
    "balanced": (0, 2, 4, 6),
    "speed": (0, 4),
}
LTX25_DEFAULT_NEGATIVE_PROMPT = (
    "blurry, out of focus, overexposed, underexposed, low contrast, washed out colors, "
    "excessive noise, grainy texture, poor lighting, flickering, motion blur, distorted "
    "proportions, deformed facial features, extra limbs, disfigured hands, inconsistent "
    "perspective, camera shake, cartoonish rendering, 3D CGI look, unrealistic materials, "
    "mismatched lip sync, silent or muted audio, distorted voice, robotic voice, echo, "
    "background noise, off-sync audio, incorrect dialogue, added dialogue, repetitive speech, "
    "jittery movement, awkward pauses, unnatural transitions, inconsistent framing, AI artifacts"
)


@lru_cache(maxsize=16)
def _checkpoint_sha256(path: str, size: int, mtime_ns: int) -> str:
    """Hash a stable checkpoint identity once per process."""
    del size, mtime_ns
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_ltx25_dfr_recipe(
    config: LTX25GenerationConfig, component_report: dict[str, object]
) -> str:
    """Resolve the sampler recipe from the validated adapter inventory.

    The official DFR stack is the development transformer plus the rank-450
    distilled adapter. Its two spatial stages are deterministic. The fused
    distilled checkpoint keeps the ordinary ancestral first stage.
    """
    if not config.dfr_enabled:
        return "disabled"
    components = component_report.get("components", ())
    if isinstance(components, list):
        for component in components:
            if not isinstance(component, dict):
                continue
            if (
                component.get("adapter_role") == "transformer_lora"
                and component.get("lora_rank") == 450
                and component.get("lora_alpha") == 450
            ):
                return "official_dev_distilled_lora"
    baked_loras = component_report.get("transformer_baked_loras", ())
    if isinstance(baked_loras, list):
        for adapter in baked_loras:
            if not isinstance(adapter, dict):
                continue
            if (
                adapter.get("adapter_role") == "transformer_lora"
                and adapter.get("lora_rank") == 450
                and adapter.get("lora_alpha") == 450
            ):
                return "official_dev_distilled_lora"
    return "fused_distilled_experimental"


def validate_ltx25_dfr_prebaked_pair(
    config: LTX25GenerationConfig, component_report: dict[str, object]
) -> None:
    """Require stage-one and stage-two pages to contain the same base adapter."""
    if not config.dfr_prebaked_transformer_path:
        return
    from .paged_checkpoint import LTX25PagedManifest

    stage1 = component_report.get("transformer_baked_loras", ())
    stage2 = LTX25PagedManifest.load(config.dfr_prebaked_transformer_path).metadata.get(
        "weetodd_baked_loras", ()
    )
    stage1_hashes = {
        item.get("sha256")
        for item in stage1
        if isinstance(item, dict) and item.get("adapter_role") == "transformer_lora"
    }
    stage2_hashes = {
        item.get("sha256")
        for item in stage2
        if isinstance(item, dict) and item.get("adapter_role") == "transformer_lora"
    }
    if not stage1_hashes or stage1_hashes.isdisjoint(stage2_hashes):
        raise ValueError(
            "The DFR stage-one and stage-two paged transformers must contain the same "
            "base transformer LoRA. Rebuild both page sets from the original Q8 base."
        )
    detail_path = Path(config.dfr_detailing_lora_path)
    detail_stat = detail_path.stat()
    detail_hashes = {
        item.get("sha256")
        for item in stage2
        if isinstance(item, dict) and item.get("adapter_role") == "ic_lora"
    }
    if _checkpoint_sha256(
        str(detail_path.resolve()), detail_stat.st_size, detail_stat.st_mtime_ns
    ) not in detail_hashes:
        raise ValueError(
            "The DFR stage-two paged transformer does not contain the selected "
            "Pixel-Spatial IC-LoRA. Rebuild the stage-two pages with that adapter."
        )


def _metadata(path: Path) -> dict[str, object]:
    """Read and JSON-decode a safetensors metadata header without loading weights."""
    if path.is_dir():
        from .paged_checkpoint import LTX25PagedManifest

        return LTX25PagedManifest.load(path).metadata
    with safe_open(path, framework="numpy") as handle:
        raw = handle.metadata() or {}
    parsed: dict[str, object] = {}
    for key, value in raw.items():
        try:
            parsed[key] = json.loads(value)
        except (json.JSONDecodeError, TypeError):
            parsed[key] = value
    return parsed


def _version_tuple(value: object) -> tuple[int, ...]:
    parts = []
    for item in str(value or "").replace("-", ".").split("."):
        if item.isdigit():
            parts.append(int(item))
        elif parts:
            break
    return tuple(parts)


def _scale_factors(video_vae_metadata: dict[str, object]) -> tuple[int, int, int]:
    config = video_vae_metadata.get("config")
    if not isinstance(config, dict):
        return (8, 32, 32)
    vae = config.get("vae")
    if not isinstance(vae, dict):
        return (8, 32, 32)
    blocks = vae.get("encoder_blocks") or vae.get("decoder_blocks")
    if not isinstance(blocks, list):
        return (8, 32, 32)
    spatial_steps = 0
    temporal_steps = 0
    for entry in blocks:
        if not isinstance(entry, (list, tuple)) or not entry:
            continue
        name = str(entry[0])
        if name.startswith(("compress_space", "compress_all")):
            spatial_steps += 1
        if name.startswith(("compress_time", "compress_all")):
            temporal_steps += 1
    patch_size = int(vae.get("patch_size", 4))
    spatial = patch_size * (2**spatial_steps)
    return (2**temporal_steps, spatial, spatial)


def _decoder_kind(video_vae_metadata: dict[str, object]) -> str:
    config = video_vae_metadata.get("config")
    vae = config.get("vae", {}) if isinstance(config, dict) else {}
    class_name = str(vae.get("_class_name", "")) if isinstance(vae, dict) else ""
    return "diffusion" if "diffusion" in class_name.lower() else "convolutional"


def _transformer_architecture(metadata: dict[str, object]) -> dict[str, object]:
    config = metadata.get("config")
    transformer = config.get("transformer", {}) if isinstance(config, dict) else {}
    if not isinstance(transformer, dict):
        transformer = {}
    return {
        "num_layers": int(transformer.get("num_layers", 48)),
        "use_prompt_adaln_single": bool(transformer.get("use_prompt_adaln_single", True)),
        "ff_bias": bool(transformer.get("ff_bias", True)),
        "audio_ff_bias": bool(transformer.get("audio_ff_bias", True)),
        "caption_proj_before_connector": bool(
            transformer.get("caption_proj_before_connector", False)
        ),
        "cross_attention_adaln": bool(transformer.get("cross_attention_adaln", False)),
        "use_keyframes_abs_pos_embedding": bool(
            transformer.get("use_keyframes_abs_pos_embedding", False)
        ),
        "frequencies_precision": str(transformer.get("frequencies_precision", "float32")),
    }


@dataclass(frozen=True)
class LTX25ComponentSpec:
    """Explicit LTX 2.5 split components; no implicit downloads are permitted."""

    transformer_path: str
    text_encoder_path: str
    video_vae_path: str
    audio_vae_path: str
    spatial_upscaler_path: str
    duration_head_path: str = ""
    distilled_lora_path: str = ""
    loras: tuple[tuple[str, float], ...] = ()
    ic_loras: tuple[tuple[str, float], ...] = ()

    def paths(self) -> dict[str, Path]:
        values = asdict(self)
        values.pop("loras", None)
        values.pop("ic_loras", None)
        return {name: Path(value).expanduser() for name, value in values.items() if value}

    def validate(
        self,
        mode: str = "distilled",
        *,
        require_spatial_upscaler: bool = True,
    ) -> dict[str, object]:
        if mode not in LTX25_CONFIG_MODES:
            raise ValueError(f"Unsupported LTX 2.5 pipeline mode: {mode!r}.")
        if self.ic_loras and mode != "distilled":
            raise ValueError(
                "LTX 2.5 IC-LoRA requires the distilled transformer. Select a distilled "
                "Quality Mode or remove the IC-LoRA."
            )
        if len(self.ic_loras) > 1:
            raise ValueError(
                "LTX 2.5 supports one active IC-LoRA adapter per generation. Remove the "
                "additional IC-LoRA loaders; standard style LoRAs may remain separate."
            )
        required = {
            "transformer_path",
            "text_encoder_path",
            "video_vae_path",
            "audio_vae_path",
        }
        if require_spatial_upscaler:
            required.add("spatial_upscaler_path")
        paths = self.paths()
        if mode == "distilled" and "distilled_lora_path" in paths:
            raise ValueError(
                "The Guided Model Loader selects a development transformer. Connect a guided "
                "Quality Mode, or use the Component Loader output directly for fast distilled "
                "generation."
            )
        if mode in {"guided", "guided_hq"}:
            distilled_lora = paths.get("distilled_lora_path")
            if distilled_lora is None or not distilled_lora.is_file():
                raise FileNotFoundError(
                    "Guided LTX 2.5 modes require the official rank-450 distilled LoRA."
                )
            from .transformer import inspect_ltx25_lora

            adapter = inspect_ltx25_lora(distilled_lora)
            if (
                adapter["adapter_role"] != "transformer_lora"
                or adapter["lora_rank"] != 450
                or adapter["lora_alpha"] != 450
            ):
                raise ValueError(
                    "Guided LTX 2.5 modes require the official rank-450/alpha-450 "
                    "distilled transformer LoRA."
                )
        for lora_path, strength in self.loras:
            resolved = Path(lora_path).expanduser()
            if not resolved.is_file() or resolved.suffix != ".safetensors":
                raise FileNotFoundError(f"LTX 2.5 LoRA file not found: {resolved}")
            if strength <= 0:
                raise ValueError("LTX 2.5 LoRA strength must be positive.")
        for lora_path, strength in self.ic_loras:
            resolved = Path(lora_path).expanduser()
            if not resolved.is_file() or resolved.suffix != ".safetensors":
                raise FileNotFoundError(f"LTX 2.5 IC-LoRA file not found: {resolved}")
            if strength <= 0:
                raise ValueError("LTX 2.5 IC-LoRA strength must be positive.")
            from .transformer import inspect_ltx25_lora

            ic_report = inspect_ltx25_lora(resolved)
            if ic_report["adapter_role"] != "ic_lora":
                raise ValueError(
                    f"The dedicated IC-LoRA stack requires reference metadata: {resolved}"
                )
            if ic_report["ic_lora_task"] == "pixel_spatial_upscaler":
                raise ValueError(
                    "The Pixel-Spatial IC-LoRA belongs to the Video Upscale / Refine node, "
                    "not the general video-reference stack."
                )
        missing = [
            name for name in sorted(required) if name not in paths or not paths[name].exists()
        ]
        if missing:
            raise FileNotFoundError(
                "LTX 2.5 split component files are missing: " + ", ".join(missing)
            )
        for name, path in paths.items():
            if path.is_dir() and name not in {"transformer_path", "text_encoder_path"}:
                raise ValueError(f"Only LTX 2.5 transformer and text encoder may be paged: {path}")
            if path.is_file() and path.suffix != ".safetensors":
                raise ValueError(f"LTX 2.5 {name} must be a .safetensors file: {path}")

        transformer_meta = _metadata(paths["transformer_path"])
        baked_ic_loras = [
            item
            for item in transformer_meta.get("weetodd_baked_loras", [])
            if item.get("adapter_role") == "ic_lora"
        ]
        if self.ic_loras and baked_ic_loras:
            raise ValueError(
                "The selected transformer already contains a baked IC-LoRA. Remove the "
                "separate IC-LoRA Loader to avoid applying the adapter twice."
            )
        if baked_ic_loras and mode != "distilled":
            raise ValueError(
                "A transformer with a baked IC-LoRA requires distilled generation mode."
            )
        if len(baked_ic_loras) > 1:
            raise ValueError(
                "The selected transformer contains more than one baked IC-LoRA. Use one "
                "task adapter per reference-conditioned generation."
            )
        for item in baked_ic_loras:
            if item.get("ic_lora_task") == "pixel_spatial_upscaler":
                raise ValueError(
                    "The selected transformer bakes the Pixel-Spatial Upscaler adapter, "
                    "which is not a general video/reference conditioning checkpoint."
                )
            for field in (
                "reference_downscale_factor",
                "reference_temporal_scale_factor",
            ):
                if int(item.get(field, 0)) < 1:
                    raise ValueError(
                        "The baked IC-LoRA manifest is missing valid reference-scale "
                        f"metadata: {field}. Rebuild it with the current fusion script."
                    )
        text_meta = _metadata(paths["text_encoder_path"])
        gemma_pack = inspect_gemma4_pack(paths["text_encoder_path"])
        video_vae_meta = _metadata(paths["video_vae_path"])
        if "duration_head_path" in paths:
            duration_meta = _metadata(paths["duration_head_path"])
            with safe_open(paths["duration_head_path"], framework="numpy") as handle:
                if not any(name.startswith("duration_head.") for name in handle.keys()):
                    raise ValueError("The selected LTX 2.5 duration head has no duration tensors.")
            if _version_tuple(duration_meta.get("model_version")) < (2, 5):
                raise ValueError("The selected duration head is not identified as LTX 2.5.")
        model_version = transformer_meta.get("model_version")
        if _version_tuple(model_version) < (2, 5):
            raise ValueError(
                "The selected transformer is not identified as LTX 2.5 or newer; "
                f"model_version={model_version!r}. Preserve upstream checkpoint metadata."
            )

        transformer_gemma = transformer_meta.get("gemma_source_checkpoint")
        text_gemma = text_meta.get("gemma_source_checkpoint")
        expected_gemma_version = (
            transformer_gemma.get("gemma_version") if isinstance(transformer_gemma, dict) else None
        )
        packed_gemma_version = gemma_pack.get("gemma_version")
        if expected_gemma_version and packed_gemma_version != expected_gemma_version:
            raise ValueError(
                "The LTX 2.5 transformer and Gemma 4 pack declare different Gemma versions: "
                f"{expected_gemma_version!r} != {packed_gemma_version!r}."
            )
        if transformer_gemma and text_gemma and transformer_gemma != text_gemma:
            raise ValueError(
                "The LTX 2.5 transformer and Gemma 4 component declare different sources."
            )
        architecture = _transformer_architecture(transformer_meta)
        incompatible = []
        if architecture["use_prompt_adaln_single"] is not True:
            incompatible.append("use_prompt_adaln_single=true")
        if architecture["ff_bias"] is not False:
            incompatible.append("ff_bias=false")
        if architecture["audio_ff_bias"] is not True:
            incompatible.append("audio_ff_bias=true")
        if architecture["cross_attention_adaln"] is not True:
            incompatible.append("cross_attention_adaln=true")
        if architecture["caption_proj_before_connector"] is not True:
            incompatible.append("caption_proj_before_connector=true")
        if architecture["use_keyframes_abs_pos_embedding"] is not True:
            incompatible.append("use_keyframes_abs_pos_embedding=true")
        if incompatible:
            raise ValueError(
                "The selected LTX 2.5 distilled transformer metadata does not declare the "
                "required architecture: " + ", ".join(incompatible)
            )

        scale_factors = _scale_factors(video_vae_meta)
        inventory = []
        total = 0
        for name, path in paths.items():
            size = (
                sum(file.stat().st_size for file in path.rglob("*") if file.is_file())
                if path.is_dir()
                else path.stat().st_size
            )
            total += size
            inventory.append({"component": name, "file": path.name, "bytes": size})
        if self.loras:
            from .transformer import inspect_ltx25_lora

            for index, (lora_path, strength) in enumerate(self.loras, start=1):
                report = inspect_ltx25_lora(lora_path)
                size = int(report["bytes"])
                total += size
                inventory.append(
                    {
                        "component": f"transformer_lora_{index}",
                        "file": Path(lora_path).name,
                        "bytes": size,
                        "strength": float(strength),
                        "adapter_role": report["adapter_role"],
                        "adapter_pairs": report["adapter_pairs"],
                        "compatibility": report["compatibility"],
                        "adapter_family": report["adapter_family"],
                        "lora_rank": report["lora_rank"],
                        "lora_alpha": report["lora_alpha"],
                    }
                )
        if self.ic_loras:
            from .transformer import inspect_ltx25_lora

            downscales: set[int] = set()
            temporal_scales: set[int] = set()
            for index, (lora_path, strength) in enumerate(self.ic_loras, start=1):
                report = inspect_ltx25_lora(lora_path)
                downscales.add(int(report["reference_downscale_factor"]))
                temporal_scales.add(int(report["reference_temporal_scale_factor"]))
                size = int(report["bytes"])
                total += size
                inventory.append(
                    {
                        "component": f"ic_lora_{index}",
                        "file": Path(lora_path).name,
                        "bytes": size,
                        "strength": float(strength),
                        "adapter_role": "ic_lora",
                        "adapter_pairs": report["adapter_pairs"],
                        "compatibility": report["compatibility"],
                        "adapter_family": report["adapter_family"],
                        "reference_downscale_factor": report[
                            "reference_downscale_factor"
                        ],
                        "reference_temporal_scale_factor": report[
                            "reference_temporal_scale_factor"
                        ],
                    }
                )
            if len(downscales) != 1 or len(temporal_scales) != 1:
                raise ValueError(
                    "Stacked LTX 2.5 IC-LoRAs must declare identical spatial and temporal "
                    "reference scale factors."
                )
        elif baked_ic_loras:
            baked = baked_ic_loras[0]
            downscales = {int(baked["reference_downscale_factor"])}
            temporal_scales = {int(baked["reference_temporal_scale_factor"])}
            inventory.append(
                {
                    "component": "baked_ic_lora",
                    "file": str(baked.get("file", "embedded adapter")),
                    "bytes": int(baked.get("bytes", 0)),
                    "strength": float(baked.get("strength", 1.0)),
                    "adapter_role": "ic_lora",
                    "adapter_family": baked.get("adapter_family"),
                    "reference_downscale_factor": next(iter(downscales)),
                    "reference_temporal_scale_factor": next(iter(temporal_scales)),
                }
            )
        return {
            "model_version": str(model_version),
            "gemma_source_checkpoint": transformer_gemma or text_gemma,
            "gemma_pack": gemma_pack,
            "video_scale_factors": list(scale_factors),
            "video_decoder": _decoder_kind(video_vae_meta),
            "transformer_architecture": architecture,
            "transformer_baked_loras": transformer_meta.get("weetodd_baked_loras", []),
            "components": inventory,
            "checkpoint_bytes": total,
            "ic_lora_reference_downscale_factor": (
                next(iter(downscales)) if self.ic_loras or baked_ic_loras else None
            ),
            "ic_lora_reference_temporal_scale_factor": (
                next(iter(temporal_scales)) if self.ic_loras or baked_ic_loras else None
            ),
        }


@dataclass(frozen=True)
class LTX25GenerationConfig:
    """Validated LTX 2.5 generation settings with a distilled default recipe."""

    pipeline_mode: str = "distilled"
    width: int = 768
    height: int = 512
    duration_seconds: float = 5.0
    duration_mode: str = "manual"
    auto_duration_min_seconds: float = 1.0
    auto_duration_max_seconds: float = 20.0
    frame_rate: float = 24.0
    seed: int = 0
    stage1_steps: int = 8
    stage2_steps: int = 3
    ic_lora_single_stage: bool = False
    stage1_sampler: str = "euler_ancestral"
    cfg_pp_batched: bool = False
    cfg_pp_schedule: str = "full"
    stage2_sampler: str = "euler"
    stage1_eta: float = 1.0
    stage1_s_noise: float = 1.0
    ancestral_seed_offset: int = 10000
    negative_prompt: str = LTX25_DEFAULT_NEGATIVE_PROMPT
    video_cfg_scale: float = 1.0
    audio_cfg_scale: float = 1.0
    stg_scale: float = 0.0
    video_rescale_scale: float = 0.0
    audio_rescale_scale: float = 0.0
    modality_scale: float = 1.0
    stg_blocks: tuple[int, ...] = ()
    low_memory: bool = True
    low_ram_streaming: bool = False
    prompt_context: str = "official_1024"
    feed_forward_backend: str = "reference_fp32"
    generated_keyframes: int = 0
    dfr_enabled: bool = False
    dfr_detailing_lora_path: str = ""
    dfr_detailing_lora_strength: float = 1.0
    dfr_prebaked_transformer_path: str = ""
    dfr_temporal_upsampler_path: str = ""
    dfr_temporal_rounds: int = 0
    diffvae_optimization: str = "combined"
    diffvae_query_chunk_size: int = 512
    diffvae_context_width_chunks: int = 4
    diffvae_stage4_tile_width: int = 0
    sol_attention_profile: str = "disabled"

    @property
    def num_frames(self) -> int:
        intervals = max(1, round(self.duration_seconds * self.frame_rate / 8.0))
        return intervals * 8 + 1

    @property
    def delivered_duration_seconds(self) -> float:
        return (self.num_frames - 1) / self.frame_rate

    @property
    def stage1_forward_passes(self) -> int:
        if self.stage1_sampler != "euler_ancestral_cfg_pp":
            return self.stage1_steps
        return self.stage1_steps + len(LTX25_CFG_PP_SCHEDULES[self.cfg_pp_schedule])

    @property
    def real_forward_passes(self) -> int:
        return self.stage1_forward_passes + self.stage2_steps

    def validate(
        self,
        *,
        scale_factors: tuple[int, int, int] = (8, 32, 32),
        reference_downscale_factor: int = 1,
    ) -> None:
        if self.pipeline_mode not in LTX25_CONFIG_MODES:
            raise ValueError(f"Unsupported LTX 2.5 pipeline mode: {self.pipeline_mode!r}.")
        if self.cfg_pp_batched and self.stage1_sampler != "euler_ancestral_cfg_pp":
            raise ValueError("Batched CFG++ requires the Euler ancestral CFG++ sampler.")
        if self.cfg_pp_schedule not in LTX25_CFG_PP_SCHEDULES:
            raise ValueError(f"Unsupported CFG++ schedule: {self.cfg_pp_schedule!r}.")
        if self.duration_mode not in {"manual", "automatic"}:
            raise ValueError("LTX 2.5 duration mode must be manual or automatic.")
        if not 0.25 <= self.auto_duration_min_seconds <= self.auto_duration_max_seconds <= 30.0:
            raise ValueError(
                "LTX 2.5 automatic duration bounds must satisfy "
                "0.25 <= minimum <= maximum <= 30 seconds."
            )
        if self.prompt_context not in LTX25_PROMPT_CONTEXTS:
            raise ValueError(f"Unsupported LTX 2.5 prompt context mode: {self.prompt_context!r}.")
        if self.feed_forward_backend not in LTX25_FEED_FORWARD_BACKENDS:
            raise ValueError(
                f"Unsupported LTX 2.5 feed-forward backend: {self.feed_forward_backend!r}."
            )
        if self.diffvae_optimization not in LTX25_DIFFVAE_OPTIMIZATIONS:
            raise ValueError(
                f"Unsupported LTX 2.5 Diffusion VAE optimization: {self.diffvae_optimization!r}."
            )
        if self.sol_attention_profile not in {"disabled", "quality", "balanced", "speed"}:
            raise ValueError("Unsupported LTX 2.5 MLX Sol Attention profile.")
        if self.sol_attention_profile != "disabled" and not self.ic_lora_single_stage:
            raise ValueError(
                "LTX 2.5 MLX Sol Attention currently requires full-resolution single-stage mode."
            )
        if self.sol_attention_profile != "disabled" and self.low_ram_streaming:
            raise ValueError(
                "LTX 2.5 MLX Sol Attention currently requires a resident transformer."
            )
        if self.diffvae_query_chunk_size < 1:
            raise ValueError("Diffusion VAE query chunk size must be positive.")
        if self.diffvae_context_width_chunks < 1:
            raise ValueError("Diffusion VAE context width chunks must be positive.")
        if self.diffvae_stage4_tile_width < 0:
            raise ValueError("Diffusion VAE stage-four tile width cannot be negative.")
        if self.diffvae_optimization == "stage4_width_tiles" and self.diffvae_stage4_tile_width < 1:
            raise ValueError("Stage-four width tiling requires a positive tile width.")
        if self.low_ram_streaming and self.feed_forward_backend != "reference_fp32":
            raise ValueError(
                "Experimental LTX 2.5 feed-forward modes are not compatible with low-RAM "
                "block streaming. Select reference_fp32 or disable low_ram_streaming."
            )
        temporal, spatial_h, spatial_w = scale_factors
        if reference_downscale_factor < 1:
            raise ValueError("LTX 2.5 IC-LoRA reference downscale factor must be positive.")
        spatial_multiplier = 1 if self.ic_lora_single_stage else 2
        modulus_h = spatial_h * spatial_multiplier * reference_downscale_factor
        modulus_w = spatial_w * spatial_multiplier * reference_downscale_factor
        if self.width < modulus_w or self.height < modulus_h:
            raise ValueError("LTX 2.5 distilled dimensions are below the two-stage latent grid.")
        if self.width % modulus_w or self.height % modulus_h:
            raise ValueError(
                f"LTX 2.5 distilled dimensions must be divisible by {modulus_w}x{modulus_h}."
            )
        if self.width > 1920 or self.height > 1920:
            raise ValueError("LTX 2.5 dimensions must not exceed 1920 pixels.")
        if not 0.25 <= self.duration_seconds <= 30.0:
            raise ValueError("LTX 2.5 duration must be between 0.25 and 30 seconds.")
        if not 0 <= self.generated_keyframes <= 8:
            raise ValueError("LTX 2.5 generated_keyframes must be between zero and eight.")
        if self.dfr_enabled:
            if self.generated_keyframes:
                raise ValueError("DFR derives its own segment-grid keyframes.")
            detail_path = Path(self.dfr_detailing_lora_path).expanduser()
            if not detail_path.is_file():
                raise FileNotFoundError(f"LTX 2.5 DFR detailing LoRA not found: {detail_path}")
            if self.dfr_detailing_lora_strength <= 0:
                raise ValueError("LTX 2.5 DFR detailing LoRA strength must be positive.")
            from .transformer import inspect_ltx25_ic_lora

            detail = inspect_ltx25_ic_lora(detail_path)
            if detail["reference_downscale_factor"] != 2:
                raise ValueError("LTX 2.5 DFR requires a 2x Pixel-Spatial IC-LoRA.")
            if self.dfr_prebaked_transformer_path:
                from .paged_checkpoint import LTX25PagedManifest

                prebaked = LTX25PagedManifest.load(self.dfr_prebaked_transformer_path)
                baked = prebaked.metadata.get("weetodd_baked_loras", [])
                roles = {
                    item.get("adapter_role")
                    for item in baked
                    if isinstance(item, dict)
                }
                if "transformer_lora" not in roles or "ic_lora" not in roles:
                    raise ValueError(
                        "The DFR prebaked transformer must contain the base transformer "
                        "LoRA and Pixel-Spatial IC-LoRA."
                    )
        if self.dfr_temporal_rounds not in {0, 1, 2}:
            raise ValueError("LTX 2.5 DFR temporal rounds must be zero, one, or two.")
        if self.dfr_temporal_rounds:
            if not self.dfr_enabled:
                raise ValueError("LTX 2.5 temporal rounds require DFR detailing.")
            temporal_path = Path(self.dfr_temporal_upsampler_path).expanduser()
            if not temporal_path.is_file() or temporal_path.suffix != ".safetensors":
                raise FileNotFoundError(
                    f"LTX 2.5 temporal upsampler not found: {temporal_path}"
                )
            from .components import inspect_ltx25_latent_upsampler

            temporal_report = inspect_ltx25_latent_upsampler(temporal_path)
            if temporal_report["spatial_upsample"] or not temporal_report[
                "temporal_upsample"
            ]:
                raise ValueError(
                    "LTX 2.5 temporal refinement requires a temporal-only upsampler."
                )
        if not 1.0 <= self.frame_rate <= 60.0:
            raise ValueError("LTX 2.5 frame rate must be between 1 and 60 fps.")
        if self.pipeline_mode == "distilled":
            if (
                self.stage1_steps != 8
                or self.stage2_steps != (0 if self.ic_lora_single_stage else 3)
                or self.stage1_sampler
                not in {"euler_ancestral", "euler_ancestral_cfg_pp"}
                or self.stage2_sampler != "euler"
            ):
                raise ValueError(
                    "The LTX 2.5 distilled path requires eight stage-one steps and either "
                    "zero single-stage or three two-stage refinement steps with Euler "
                    "ancestral or Euler ancestral CFG++ sampling."
                )
            if self.stage1_sampler == "euler_ancestral_cfg_pp" and not self.ic_lora_single_stage:
                raise ValueError(
                    "Euler ancestral CFG++ is currently validated for the official "
                    "single-stage IC-LoRA workflow only."
                )
            if self.stage1_eta != 1.0 or self.stage1_s_noise != 1.0:
                raise ValueError("LTX 2.5 distilled stage one requires eta=1.0 and s_noise=1.0.")
            if self.ancestral_seed_offset != 10000:
                raise ValueError(
                    "LTX 2.5 distilled requires the stage-one noise seed offset 10000."
                )
        elif self.pipeline_mode == "guided":
            if (
                self.stage1_steps != 30
                or self.stage2_steps != 3
                or self.stage1_sampler != "euler_guided"
                or self.stage2_sampler != "euler"
            ):
                raise ValueError("Production guided LTX 2.5 requires 30 guided Euler steps.")
            if not self.negative_prompt.strip() or self.video_cfg_scale <= 1.0:
                raise ValueError("Production guided LTX 2.5 requires CFG and a negative prompt.")
        else:
            if (
                self.stage1_steps != 15
                or self.stage2_steps != 3
                or self.stage1_sampler != "res_2s_guided"
                or self.stage2_sampler != "euler"
            ):
                raise ValueError("HQ guided LTX 2.5 requires 15 res_2s stage-one steps.")
            if not self.negative_prompt.strip() or self.video_cfg_scale <= 1.0:
                raise ValueError("HQ guided LTX 2.5 requires CFG and a negative prompt.")
        if self.dfr_enabled and self.pipeline_mode != "distilled":
            raise ValueError(
                "DFR currently uses its own distilled recipe and cannot be stacked on a "
                "guided mode."
            )
        if self.ic_lora_single_stage:
            if self.pipeline_mode != "distilled":
                raise ValueError("LTX 2.5 IC-LoRA single-stage mode requires distilled sampling.")
            if self.dfr_enabled or self.dfr_temporal_rounds:
                raise ValueError("LTX 2.5 IC-LoRA single-stage mode cannot be combined with DFR.")
        if (self.num_frames - 1) % temporal:
            raise ValueError(
                f"LTX 2.5 frame count must align to temporal compression factor {temporal}."
            )


def apply_ltx25_generation_preset(name: str, values: dict[str, object]) -> dict[str, object]:
    """Apply a named LTX 2.5 recipe without changing the selected seed."""
    if name not in LTX25_GENERATION_PRESETS:
        raise ValueError(f"Unsupported LTX 2.5 generation preset: {name!r}.")
    resolved = dict(values)
    if name in LTX25_GENERATION_PRESETS[1:]:
        high_resolution = name == LTX25_GENERATION_PRESETS[2]
        resolved.update(
            width=1920 if high_resolution else 768,
            height=1088 if high_resolution else 512,
            duration_seconds=5.0,
            frame_rate=24.0,
            low_memory=True,
            low_ram_streaming=False,
            prompt_context="official_1024",
            feed_forward_backend="reference_fp32",
        )
    return resolved


def _pipeline_class():
    from .pipeline import LTX25DistilledPipeline

    return LTX25DistilledPipeline


def backend_capability() -> dict[str, object]:
    """Report whether the installed optional backend exposes the versioned entry point."""
    try:
        pipeline = _pipeline_class()
    except ImportError as exc:
        try:
            from importlib.metadata import version

            installed = version("ltx-pipelines-mlx")
        except Exception:
            installed = None
        return {
            "ready": False,
            "installed_ltx_pipelines_mlx": installed,
            "project_ancestral_sampler": True,
            "project_gemma4_conditioner": True,
            "remaining_engine_gates": ["transformer", "video_vae", "audio_vae", "vocoder"],
            "reason": str(exc),
        }
    return {"ready": True, "pipeline_class": pipeline.__name__}


@contextmanager
def _comfy_sampler_progress(check_interrupted, step_callback, expected_steps: int):
    if check_interrupted is None and step_callback is None:
        yield
        return
    from ltx_pipelines_mlx.utils import samplers

    original = samplers.tqdm
    completed = 0

    def iter_with_callbacks(iterable, *_args, **_kwargs):
        nonlocal completed
        for item in iterable:
            if check_interrupted is not None:
                check_interrupted()
            yield item
            completed += 1
            if step_callback is not None:
                step_callback(completed, max(expected_steps, completed))

    samplers.tqdm = iter_with_callbacks
    try:
        yield
    finally:
        samplers.tqdm = original


class LTX25RuntimeCache:
    """One process-local LTX 2.5 pipeline with explicit staged release."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._key: tuple[object, ...] | None = None
        self._pipeline: Any = None
        self._previous_cache_limit: int | None = None

    @property
    def loaded(self) -> bool:
        with self._lock:
            return self._pipeline is not None

    def get(self, spec: LTX25ComponentSpec, config: LTX25GenerationConfig):
        report = spec.validate(
            config.pipeline_mode,
            require_spatial_upscaler=not config.ic_lora_single_stage,
        )
        scales = tuple(int(value) for value in report["video_scale_factors"])
        config.validate(
            scale_factors=scales,
            reference_downscale_factor=int(
                report.get("ic_lora_reference_downscale_factor") or 1
            ),
        )
        validate_ltx25_dfr_prebaked_pair(config, report)
        key = (
            spec,
            config.pipeline_mode,
            config.low_memory,
            config.low_ram_streaming,
            config.feed_forward_backend,
            config.diffvae_optimization,
            config.diffvae_query_chunk_size,
            config.diffvae_context_width_chunks,
            config.diffvae_stage4_tile_width,
            config.sol_attention_profile,
            config.dfr_temporal_upsampler_path,
            config.dfr_temporal_rounds,
            config.dfr_prebaked_transformer_path,
        )
        with self._lock:
            if self._pipeline is None or self._key != key:
                self._release_locked()
                pipeline_class = _pipeline_class()
                if config.low_ram_streaming:
                    import mlx.core as mx

                    self._previous_cache_limit = int(mx.set_cache_limit(0))
                kwargs = {
                    **{name: str(path) for name, path in spec.paths().items()},
                    "low_memory": config.low_memory,
                    "low_ram_streaming": config.low_ram_streaming,
                    "feed_forward_backend": config.feed_forward_backend,
                    "diffvae_optimization": config.diffvae_optimization,
                    "diffvae_query_chunk_size": config.diffvae_query_chunk_size,
                    "diffvae_context_width_chunks": config.diffvae_context_width_chunks,
                    "diffvae_stage4_tile_width": config.diffvae_stage4_tile_width,
                    "sol_attention_profile": config.sol_attention_profile,
                    "temporal_upsampler_path": config.dfr_temporal_upsampler_path,
                    "dfr_stage2_transformer_path": config.dfr_prebaked_transformer_path,
                    "loras": spec.loras,
                    "ic_loras": spec.ic_loras,
                }
                signature = inspect.signature(pipeline_class)
                accepted = {
                    key: value for key, value in kwargs.items() if key in signature.parameters
                }
                try:
                    self._pipeline = pipeline_class(**accepted)
                except BaseException:
                    if self._previous_cache_limit is not None:
                        mx.set_cache_limit(self._previous_cache_limit)
                        self._previous_cache_limit = None
                    raise
                self._key = key
            return self._pipeline

    def generate_to_file(
        self,
        spec: LTX25ComponentSpec,
        config: LTX25GenerationConfig,
        prompt: str,
        output_path: str | Path,
        *,
        image_path: str | None = None,
        image_inputs: list[dict[str, object]] | None = None,
        video_references: list[dict[str, object]] | None = None,
        audio_reference: dict[str, object] | None = None,
        unload_after: bool = True,
        check_interrupted=None,
        step_callback=None,
    ) -> dict[str, object]:
        if not prompt.strip():
            raise ValueError("LTX 2.5 prompt must not be empty.")
        report = spec.validate(
            config.pipeline_mode,
            require_spatial_upscaler=not config.ic_lora_single_stage,
        )
        scales = tuple(int(value) for value in report["video_scale_factors"])
        config.validate(
            scale_factors=scales,
            reference_downscale_factor=int(
                report.get("ic_lora_reference_downscale_factor") or 1
            ),
        )
        dfr_recipe = resolve_ltx25_dfr_recipe(config, report)
        if config.duration_mode == "automatic" and not spec.duration_head_path:
            raise ValueError(
                "Automatic LTX 2.5 duration requires a duration head in the component loader."
            )
        if check_interrupted is not None:
            check_interrupted()
        try:
            import mlx.core as mx

            mx.reset_peak_memory()
        except (ImportError, AttributeError):
            mx = None
        started = time.perf_counter()
        pipeline = self.get(spec, config)
        try:
            from .feed_forward import reset_feed_forward_runtime_status

            reset_feed_forward_runtime_status()
        except (ImportError, AttributeError):
            pass
        images = None
        if image_inputs:
            from ltx_pipelines_mlx.utils.args import ImageConditioningInput

            images = [
                ImageConditioningInput(
                    path=str(item["path"]),
                    frame_idx=int(item["frame_index"]),
                    strength=float(item["strength"]),
                )
                for item in image_inputs
            ]
        kwargs: dict[str, object] = {
            "prompt": prompt,
            "output_path": str(output_path),
            "height": config.height,
            "width": config.width,
            "num_frames": config.num_frames,
            "frame_rate": config.frame_rate,
            "seed": config.seed,
            "image": image_path,
            "images": images,
            "video_references": video_references or [],
            "audio_reference": audio_reference,
            "stage1_steps": config.stage1_steps,
            "stage2_steps": config.stage2_steps,
            "ic_lora_single_stage": config.ic_lora_single_stage,
            "stage1_sigmas": LTX25_DISTILLED_SIGMAS,
            "stage2_sigmas": LTX25_STAGE2_SIGMAS,
            "stage1_sampler": config.stage1_sampler,
            "cfg_pp_batched": config.cfg_pp_batched,
            "cfg_pp_schedule": config.cfg_pp_schedule,
            "stage2_sampler": config.stage2_sampler,
            "stage1_eta": config.stage1_eta,
            "stage1_s_noise": config.stage1_s_noise,
            "ancestral_noise_seed": config.seed + config.ancestral_seed_offset,
            "check_interrupted": check_interrupted,
            "step_callback": step_callback,
            "prompt_context": config.prompt_context,
            "generated_keyframes": config.generated_keyframes,
            "pipeline_mode": config.pipeline_mode,
            "negative_prompt": config.negative_prompt,
            "video_cfg_scale": config.video_cfg_scale,
            "audio_cfg_scale": config.audio_cfg_scale,
            "stg_scale": config.stg_scale,
            "video_rescale_scale": config.video_rescale_scale,
            "audio_rescale_scale": config.audio_rescale_scale,
            "modality_scale": config.modality_scale,
            "stg_blocks": config.stg_blocks,
            "dfr_enabled": config.dfr_enabled,
            "dfr_official_recipe": dfr_recipe == "official_dev_distilled_lora",
            "dfr_detailing_lora": (
                (config.dfr_detailing_lora_path, config.dfr_detailing_lora_strength)
                if config.dfr_enabled
                else None
            ),
            "temporal_upsample_rounds": config.dfr_temporal_rounds,
            "auto_duration": config.duration_mode == "automatic",
            "auto_duration_min_seconds": config.auto_duration_min_seconds,
            "auto_duration_max_seconds": config.auto_duration_max_seconds,
        }
        signature = inspect.signature(pipeline.generate_and_save)
        accepts_kwargs = any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        )
        accepted = {
            key: value
            for key, value in kwargs.items()
            if accepts_kwargs or key in signature.parameters
        }
        succeeded = False
        try:
            with (
                self._lock,
                _comfy_sampler_progress(
                    check_interrupted, step_callback, config.stage1_steps + config.stage2_steps
                ),
            ):
                result_path = pipeline.generate_and_save(**accepted)
            if check_interrupted is not None:
                check_interrupted()
            succeeded = True
            try:
                from .feed_forward import feed_forward_runtime_status

                feed_forward_runtime = feed_forward_runtime_status()
            except (ImportError, AttributeError):
                feed_forward_runtime = None
            resolved_num_frames = int(
                getattr(pipeline, "last_num_frames", None) or config.num_frames
            )
            resolved_frame_rate = float(
                getattr(pipeline, "last_output_frame_rate", None) or config.frame_rate
            )
            result = {
                "prompt": prompt,
                "video_path": str(result_path),
                "generation": asdict(config),
                "sampling": {
                    "stage1_steps": config.stage1_steps,
                    "stage1_real_forwards": config.stage1_forward_passes,
                    "stage2_steps": config.stage2_steps,
                    "total_real_forwards": config.real_forward_passes,
                    "stage1_sampler": config.stage1_sampler,
                },
                "num_frames": resolved_num_frames,
                "frame_rate": resolved_frame_rate,
                "delivered_duration_seconds": (resolved_num_frames - 1) / resolved_frame_rate,
                "predicted_duration_seconds": getattr(
                    pipeline, "last_predicted_duration_seconds", None
                ),
                "pipeline_mode": config.pipeline_mode,
                "resolved_dfr_recipe": dfr_recipe,
                "model_version": report["model_version"],
                "video_decoder": report["video_decoder"],
                "video_scale_factors": report["video_scale_factors"],
                "mlx_peak_bytes": int(mx.get_peak_memory()) if mx is not None else None,
                "total_seconds": time.perf_counter() - started,
                "stage_timings": getattr(pipeline, "last_timings", {}),
                "video_decode": getattr(
                    getattr(pipeline, "video_decoder_block", None),
                    "last_decode_report",
                    {},
                ),
                "resolved_prompt_context": getattr(pipeline, "last_prompt_context", None),
                "feed_forward_backend": getattr(pipeline, "feed_forward_report", None),
                "feed_forward_runtime": feed_forward_runtime,
                "paged_transformer": getattr(pipeline, "paged_transformer_report", None),
                "paged_text_encoder": getattr(
                    getattr(pipeline, "prompt_encoder", None),
                    "paged_checkpoint_report",
                    None,
                ),
                "runtime_cached": not unload_after,
                "conditioning": {
                    "video_reference_count": len(video_references or ()),
                    "audio_driven": audio_reference is not None,
                    "audio_output": (
                        "original_comfy_audio" if audio_reference is not None else "generated"
                    ),
                    "ic_lora_stage_scope": (
                        (
                            "single_stage_full_resolution"
                            if config.ic_lora_single_stage
                            else "stage_1_only"
                        )
                        if video_references
                        else None
                    ),
                },
            }
            try:
                from wee_todd_nodes.process_memory import complete_process_memory

                result.update(complete_process_memory())
            except (ImportError, AttributeError):
                result["complete_process_memory_scope"] = "unavailable"
            return result
        finally:
            if unload_after or not succeeded:
                self.unload()

    def generate_chain_to_file(
        self,
        spec: LTX25ComponentSpec,
        config: LTX25GenerationConfig,
        prompts: list[str],
        output_path: str | Path,
        *,
        window_count: int,
        overlap_frames: int,
        unload_after: bool = True,
        check_interrupted=None,
        step_callback=None,
    ) -> dict[str, object]:
        """Generate an exact latent-native LTX 2.5 chained timeline."""
        from .chaining import plan_ltx25_chain

        report = spec.validate(config.pipeline_mode)
        scales = tuple(int(value) for value in report["video_scale_factors"])
        config.validate(scale_factors=scales)
        if config.duration_mode != "manual":
            raise ValueError(
                "Automatic duration is only available for one-shot LTX 2.5 generation; "
                "set an explicit total duration for chained timelines."
            )
        if config.dfr_temporal_rounds:
            raise ValueError(
                "LTX 2.5 temporal DFR is not yet available for chained timelines."
            )
        plan = plan_ltx25_chain(
            total_frames=config.num_frames,
            window_count=window_count,
            overlap_frames=overlap_frames,
            frame_rate=config.frame_rate,
        )
        if len(prompts) != window_count or any(not prompt.strip() for prompt in prompts):
            raise ValueError("Every LTX 2.5 chained window requires a non-empty prompt.")
        if check_interrupted is not None:
            check_interrupted()
        try:
            import mlx.core as mx

            mx.reset_peak_memory()
        except (ImportError, AttributeError):
            mx = None
        started = time.perf_counter()
        pipeline = self.get(spec, config)
        try:
            from .feed_forward import reset_feed_forward_runtime_status

            reset_feed_forward_runtime_status()
        except (ImportError, AttributeError):
            pass
        succeeded = False
        try:
            with self._lock:
                result_path = pipeline.generate_chained_and_save(
                    prompts=prompts,
                    output_path=str(output_path),
                    height=config.height,
                    width=config.width,
                    total_frames=config.num_frames,
                    window_count=window_count,
                    overlap_frames=overlap_frames,
                    frame_rate=config.frame_rate,
                    seed=config.seed,
                    check_interrupted=check_interrupted,
                    step_callback=step_callback,
                    prompt_context=config.prompt_context,
                )
            if check_interrupted is not None:
                check_interrupted()
            succeeded = True
            try:
                from .feed_forward import feed_forward_runtime_status

                feed_forward_runtime = feed_forward_runtime_status()
            except (ImportError, AttributeError):
                feed_forward_runtime = None
            result = {
                "prompts": prompts,
                "video_path": str(result_path),
                "generation": asdict(config),
                "num_frames": config.num_frames,
                "delivered_duration_seconds": config.delivered_duration_seconds,
                "pipeline_mode": config.pipeline_mode,
                "model_version": report["model_version"],
                "video_decoder": report["video_decoder"],
                "video_scale_factors": report["video_scale_factors"],
                "chain_plan": plan.as_dict(),
                "mlx_peak_bytes": int(mx.get_peak_memory()) if mx is not None else None,
                "total_seconds": time.perf_counter() - started,
                "stage_timings": getattr(pipeline, "last_timings", {}),
                "video_decode": getattr(
                    getattr(pipeline, "video_decoder_block", None),
                    "last_decode_report",
                    {},
                ),
                "resolved_prompt_context": getattr(pipeline, "last_prompt_context", None),
                "feed_forward_backend": getattr(pipeline, "feed_forward_report", None),
                "feed_forward_runtime": feed_forward_runtime,
                "paged_transformer": getattr(pipeline, "paged_transformer_report", None),
                "paged_text_encoder": getattr(
                    getattr(pipeline, "prompt_encoder", None),
                    "paged_checkpoint_report",
                    None,
                ),
                "runtime_cached": not unload_after,
            }
            try:
                from wee_todd_nodes.process_memory import complete_process_memory

                result.update(complete_process_memory())
            except (ImportError, AttributeError):
                result["complete_process_memory_scope"] = "unavailable"
            return result
        finally:
            if unload_after or not succeeded:
                self.unload()

    def unload(self) -> None:
        with self._lock:
            self._release_locked()

    def _release_locked(self) -> None:
        self._pipeline = None
        self._key = None
        gc.collect()
        try:
            import mlx.core as mx

            mx.clear_cache()
            if self._previous_cache_limit is not None:
                mx.set_cache_limit(self._previous_cache_limit)
                self._previous_cache_limit = None
        except (ImportError, AttributeError):
            pass


RUNTIME = LTX25RuntimeCache()
