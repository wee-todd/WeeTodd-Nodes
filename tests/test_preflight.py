import json
import shutil
import struct
from pathlib import Path

import pytest

from wee_todd_nodes.preflight import (
    H3ComponentSetSpec,
    H3PreflightRequest,
    preflight_components,
    read_safetensors_header,
)


def _json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")


def _safetensors(
    path: Path,
    tensors: dict[str, tuple[str, list[int], int]],
    metadata: dict[str, str] | None = None,
) -> None:
    """Write a header-valid file with opaque payload bytes."""

    path.parent.mkdir(parents=True, exist_ok=True)
    offset = 0
    header = {"__metadata__": metadata} if metadata is not None else {}
    for name, (dtype, shape, size) in tensors.items():
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + size]}
        offset += size
    raw = json.dumps(header, separators=(",", ":")).encode()
    padding = (-len(raw)) % 8
    raw += b" " * padding
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + b"X" * offset)


def _component_tree(tmp_path: Path, tasks=("t2va", "fl2va")) -> Path:
    root = tmp_path / "FL2VA"
    _json(
        root / "model_index.json",
        {
            "transformer": ["diffusers", "MiniMaxH3DiTModel"],
            "text_encoder": ["transformers", "MiniMaxH3Qwen3VLHFEncoder"],
            "processor": ["transformers", "Qwen3VLProcessor"],
            "tokenizer": ["transformers", "Qwen2TokenizerFast"],
            "video_vae": ["diffusers", "MiniMaxH3VideoVAE"],
            "audio_vae": ["diffusers", "MiniMaxH3AudioVAE"],
            "_minimax_h3": {"partition": "fl2va", "tasks": list(tasks)},
        },
    )
    _json(
        root / "transformer" / "config.json",
        {"latents_dim": 24, "audio_latents_dim": 32, "text_dim": 5120},
    )
    _safetensors(
        root / "transformer" / "model.safetensors",
        {
            "blocks.0.adaln_proj.linear.weight": ("F16", [4, 4], 32),
            "video_patch_proj.weight": ("F16", [4, 4], 32),
        },
    )
    _json(root / "text_encoder" / "config.json", {"hidden_size": 5120})
    _safetensors(
        root / "text_encoder" / "model.safetensors",
        {"model.layers.0.weight": ("F16", [4, 4], 32)},
    )
    _json(root / "processor" / "preprocessor_config.json", {"patch_size": 16})
    _json(root / "tokenizer" / "tokenizer.json", {"version": "1.0"})
    _json(root / "video_vae" / "config.json", {"vae_clip_length": 17})
    _json(root / "video_vae" / "source" / "config.json", {"z_channels": 24})
    _safetensors(
        root / "video_vae" / "source" / "model.safetensors",
        {"decoder.weight": ("F16", [4, 4], 32)},
    )
    audio_config = {"kwargs": {"vae_latent_channels": 32}}
    _json(root / "audio_vae" / "config.json", audio_config)
    _json(root / "audio_vae" / "metadata.json", {"metadata": audio_config})
    _safetensors(
        root / "audio_vae" / "model.safetensors",
        {"decoder.weight": ("F32", [4, 4], 64)},
    )
    return root


def _portable_optimized_spec(tmp_path: Path) -> H3ComponentSetSpec:
    """Create the documented shared T2VA layout with tiny valid fixtures."""

    models = tmp_path / "MiniMax-H3"
    root = _component_tree(models)
    transformer = models / "transformers" / "q8_extended_paged"
    text_encoder = models / "text_encoders" / "q8-paged"
    video_vae = models / "vae" / "q8" / "video_vae_affine_q8.safetensors"

    _json(
        transformer / "config.json",
        {"latents_dim": 24, "audio_latents_dim": 32, "text_dim": 5120},
    )
    _safetensors(
        transformer / "pages" / "fixed.safetensors",
        {"video_patch_proj.weight": ("F16", [4, 4], 32)},
    )
    _safetensors(
        transformer / "pages" / "block-000.safetensors",
        {"blocks.0.attn.weight": ("F16", [4, 4], 32)},
    )
    _json(
        transformer / "paged_manifest.json",
        {
            "format": "weetodd-h3-paged-v1",
            "num_blocks": 1,
            "source_tensor_bytes": 64,
            "fixed": {
                "file": "pages/fixed.safetensors",
                "tensor_count": 1,
                "tensor_bytes": 32,
                "sha256": "0" * 64,
            },
            "blocks": [
                {
                    "file": "pages/block-000.safetensors",
                    "tensor_count": 1,
                    "tensor_bytes": 32,
                    "sha256": "0" * 64,
                }
            ],
        },
    )

    _json(text_encoder / "config.json", {"hidden_size": 5120})
    _safetensors(
        text_encoder / "pages" / "fixed.safetensors",
        {"model.embed_tokens.weight": ("F16", [4, 4], 32)},
    )
    _safetensors(
        text_encoder / "pages" / "layer-000.safetensors",
        {"model.layers.0.weight": ("F16", [4, 4], 32)},
    )
    _json(
        text_encoder / "paged_text_encoder_manifest.json",
        {
            "format": "weetodd-h3-qwen-paged-v1",
            "num_layers": 1,
            "source_tensor_bytes": 64,
            "fixed": {
                "file": "pages/fixed.safetensors",
                "tensor_count": 1,
                "tensor_bytes": 32,
                "sha256": "0" * 64,
            },
            "layers": [
                {
                    "file": "pages/layer-000.safetensors",
                    "tensor_count": 1,
                    "tensor_bytes": 32,
                    "sha256": "0" * 64,
                }
            ],
        },
    )

    wrapper = {
        "format": "minimax-h3-mlx-video-vae",
        "format_version": 1,
        "tensor_layout": "ODHWI",
    }
    _safetensors(
        video_vae,
        {"decoder.weight": ("F16", [4, 4], 32)},
        metadata={"minimax_h3_video_vae": json.dumps(wrapper)},
    )
    return H3ComponentSetSpec(
        checkpoint=str(root),
        task="t2va",
        transformer=str(transformer),
        text_encoder=str(text_encoder),
        processor=str(root / "processor"),
        tokenizer=str(root / "tokenizer"),
        video_vae=str(video_vae),
        audio_vae=str(root / "audio_vae"),
    )


def test_header_reader_does_not_interpret_tensor_payload(tmp_path: Path):
    path = tmp_path / "opaque.safetensors"
    _safetensors(path, {"weight": ("F16", [4, 4], 32)})

    header = read_safetensors_header(path)

    assert header.tensor_count == 1
    assert header.tensor_bytes == 32
    assert header.dtypes == ("F16",)
    assert header.tensor_names == ("weight",)


def test_preflight_validates_complete_stack_and_estimates_stages(tmp_path: Path):
    root = _component_tree(tmp_path)

    report = preflight_components(
        H3ComponentSetSpec(str(root), task="t2va"),
        H3PreflightRequest(
            duration_seconds=5.0,
            steps=8,
            width=640,
            height=384,
            prompt_tokens=64,
            available_memory_gb=1.0,
        ),
    )

    assert report.partition == "fl2va"
    assert report.frames == 124
    assert report.video_latent_frames == 37
    assert report.audio_latent_frames == 207
    assert report.packed_rows == 64 + 37 * 20 * 12 + 207 * 2
    assert report.staged_peak_bytes > 0
    assert report.headroom_bytes is not None
    assert {component.name for component in report.components} == {
        "transformer",
        "text_encoder",
        "processor",
        "tokenizer",
        "video_vae",
        "audio_vae",
    }


def test_preflight_accepts_experimental_two_and_a_half_second_window(tmp_path: Path):
    root = _component_tree(tmp_path)

    report = preflight_components(
        H3ComponentSetSpec(str(root), task="t2va"),
        H3PreflightRequest(
            duration_seconds=2.5,
            steps=8,
            width=640,
            height=384,
            prompt_tokens=64,
        ),
    )

    assert report.frames == 73


def test_preflight_validates_portable_optimized_component_layout(tmp_path: Path):
    spec = _portable_optimized_spec(tmp_path)

    report = preflight_components(spec, H3PreflightRequest())
    by_name = {component.name: component for component in report.components}

    assert by_name["transformer"].paging_format == "weetodd-h3-paged-v1"
    assert by_name["text_encoder"].paging_format == "weetodd-h3-qwen-paged-v1"
    assert by_name["processor"].path.endswith("FL2VA/processor")
    assert by_name["tokenizer"].path.endswith("FL2VA/tokenizer")
    assert by_name["video_vae"].files == ("video_vae_affine_q8.safetensors",)


def test_preflight_allows_one_native_asset_directory_for_processor_and_tokenizer(
    tmp_path: Path,
):
    root = _component_tree(tmp_path)
    assets = root / "tokenizer"

    report = preflight_components(
        H3ComponentSetSpec(
            str(root),
            processor=str(assets),
            tokenizer=str(assets),
        ),
        H3PreflightRequest(),
    )

    by_name = {component.name: component for component in report.components}
    assert by_name["processor"].path == str(assets)
    assert by_name["tokenizer"].path == str(assets)


def test_missing_native_fallback_requests_an_explicit_override(tmp_path: Path):
    root = _component_tree(tmp_path)
    shutil.rmtree(root / "transformer")

    with pytest.raises(FileNotFoundError) as caught:
        preflight_components(H3ComponentSetSpec(str(root)), H3PreflightRequest())

    message = str(caught.value)
    assert "No transformer override is selected" in message
    assert "native partition fallback" in message
    assert "shared physical folder named 'transformers'" in message


def test_nested_component_root_error_identifies_selectable_directory(tmp_path: Path):
    root = _component_tree(tmp_path)
    nested = root / "transformer" / "q8_extended_paged"
    nested.mkdir(parents=True)
    (root / "transformer" / "config.json").replace(nested / "config.json")
    (root / "transformer" / "model.safetensors").replace(nested / "model.safetensors")

    with pytest.raises(FileNotFoundError) as caught:
        preflight_components(H3ComponentSetSpec(str(root)), H3PreflightRequest())

    message = str(caught.value)
    assert "A nested component root was found" in message
    assert str(nested) in message


def test_vision_tasks_reject_text_only_paged_qwen_before_loading(tmp_path: Path):
    spec = _portable_optimized_spec(tmp_path)
    root = Path(spec.checkpoint)
    manifest = json.loads((root / "model_index.json").read_text())
    manifest["_minimax_h3"] = {"partition": "ref2va", "tasks": ["ref2va"]}
    _json(root / "model_index.json", manifest)

    with pytest.raises(ValueError, match="paged text_encoder is text-only"):
        preflight_components(
            H3ComponentSetSpec(**{**spec.__dict__, "task": "ref2va"}),
            H3PreflightRequest(),
        )


def test_preflight_rejects_missing_component_before_weight_loading(tmp_path: Path):
    root = _component_tree(tmp_path)
    (root / "audio_vae" / "model.safetensors").unlink()

    with pytest.raises(FileNotFoundError, match="audio_vae has no safetensors"):
        preflight_components(H3ComponentSetSpec(str(root)), H3PreflightRequest())


def test_preflight_rejects_wrong_task(tmp_path: Path):
    root = _component_tree(tmp_path)

    with pytest.raises(ValueError, match="does not support task 'ref2va'"):
        preflight_components(
            H3ComponentSetSpec(str(root), task="ref2va"),
            H3PreflightRequest(),
        )


def test_preflight_explicitly_allows_fl2va_weights_for_ref2va(tmp_path: Path):
    root = _component_tree(tmp_path)

    report = preflight_components(
        H3ComponentSetSpec(
            str(root),
            task="ref2va",
            allow_fl2va_weights_for_ref2va=True,
        ),
        H3PreflightRequest(),
    )

    assert report.task == "ref2va"
    assert report.partition == "fl2va"
    assert any("different learned tensor payloads" in warning for warning in report.warnings)


def test_preflight_rejects_unsupported_quantization(tmp_path: Path):
    root = _component_tree(tmp_path)
    _json(root / "transformer" / "quant_config.json", {"bits": 3, "group_size": 64})

    with pytest.raises(ValueError, match="quantization bits must be 4, 5, 6, or 8"):
        preflight_components(H3ComponentSetSpec(str(root)), H3PreflightRequest())


@pytest.mark.parametrize("bits", [5, 6])
def test_preflight_accepts_experimental_native_quantization_widths(tmp_path: Path, bits: int):
    root = _component_tree(tmp_path)
    _json(root / "transformer" / "quant_config.json", {"bits": bits, "group_size": 64})

    report = preflight_components(H3ComponentSetSpec(str(root)), H3PreflightRequest())

    transformer = next(item for item in report.components if item.name == "transformer")
    assert transformer.quantization == f"mlx-affine-{bits}bit-group-64"


def test_preflight_uses_bounded_paged_transformer_window(tmp_path: Path):
    root = _component_tree(tmp_path)
    transformer_path = root / "transformer"
    (transformer_path / "model.safetensors").unlink()
    _safetensors(
        transformer_path / "pages" / "fixed.safetensors",
        {"video_patch_proj.weight": ("F16", [4, 4], 32)},
    )
    blocks = []
    for index in range(8):
        relative = f"pages/block-{index:03d}.safetensors"
        _safetensors(
            transformer_path / relative,
            {f"blocks.{index}.attn.weight": ("F16", [4, 4], 32)},
        )
        blocks.append(
            {
                "file": relative,
                "tensor_count": 1,
                "tensor_bytes": 32,
                "sha256": "0" * 64,
            }
        )
    _json(
        transformer_path / "paged_manifest.json",
        {
            "format": "weetodd-h3-paged-v1",
            "num_blocks": 8,
            "source_tensor_bytes": 288,
            "fixed": {
                "file": "pages/fixed.safetensors",
                "tensor_count": 1,
                "tensor_bytes": 32,
                "sha256": "0" * 64,
            },
            "blocks": blocks,
        },
    )

    report = preflight_components(H3ComponentSetSpec(str(root)), H3PreflightRequest())
    transformer = next(item for item in report.components if item.name == "transformer")

    assert transformer.tensor_bytes == 288
    assert transformer.paging_format == "weetodd-h3-paged-v1"
    assert transformer.paging_fixed_bytes == 32
    assert transformer.paging_window_bytes == 128
    assert report.transformer_load_stage_bytes == (
        160 + report.adaln_cache_bytes + report.packed_workspace_bytes
    )


def test_preflight_uses_one_paged_qwen_layer(tmp_path: Path):
    root = _component_tree(tmp_path)
    encoder = root / "text_encoder"
    (encoder / "model.safetensors").unlink()
    _safetensors(
        encoder / "pages" / "fixed.safetensors",
        {"model.embed_tokens.weight": ("F16", [4, 4], 32)},
    )
    layers = []
    for index, size in enumerate((32, 64)):
        relative = f"pages/layer-{index:03d}.safetensors"
        _safetensors(
            encoder / relative,
            {f"model.layers.{index}.weight": ("F16", [size // 2], size)},
        )
        layers.append(
            {
                "file": relative,
                "tensor_count": 1,
                "tensor_bytes": size,
                "sha256": "0" * 64,
            }
        )
    _json(
        encoder / "paged_text_encoder_manifest.json",
        {
            "format": "weetodd-h3-qwen-paged-v1",
            "num_layers": 2,
            "source_tensor_bytes": 128,
            "fixed": {
                "file": "pages/fixed.safetensors",
                "tensor_count": 1,
                "tensor_bytes": 32,
                "sha256": "0" * 64,
            },
            "layers": layers,
        },
    )

    request = H3PreflightRequest(prompt_tokens=10)
    report = preflight_components(H3ComponentSetSpec(str(root)), request)
    text_encoder = next(item for item in report.components if item.name == "text_encoder")

    assert text_encoder.paging_format == "weetodd-h3-qwen-paged-v1"
    assert text_encoder.paging_fixed_bytes == 32
    assert text_encoder.paging_window_bytes == 64
    assert report.qwen_stage_bytes == 96 + 10 * 5120 * 2 * 4


def test_preflight_rejects_incompatible_transformer_config(tmp_path: Path):
    root = _component_tree(tmp_path)
    _json(
        root / "transformer" / "config.json",
        {"latents_dim": 16, "audio_latents_dim": 32, "text_dim": 5120},
    )

    with pytest.raises(ValueError, match="latents_dim must be 24"):
        preflight_components(H3ComponentSetSpec(str(root)), H3PreflightRequest())


def test_preflight_rejects_missing_indexed_shard(tmp_path: Path):
    root = _component_tree(tmp_path)
    (root / "transformer" / "model.safetensors").unlink()
    _json(
        root / "transformer" / "model.safetensors.index.json",
        {"weight_map": {"weight": "missing-00001-of-00001.safetensors"}},
    )

    with pytest.raises(FileNotFoundError, match="missing indexed safetensors shards"):
        preflight_components(H3ComponentSetSpec(str(root)), H3PreflightRequest())


def test_preflight_rejects_native_single_file_transformer_as_not_mlx_ready(tmp_path: Path):
    root = _component_tree(tmp_path)
    native_transformer = tmp_path / "native_transformer.safetensors"
    _safetensors(native_transformer, {"blocks.0.weight": ("F16", [4, 4], 32)})

    with pytest.raises(ValueError, match="adaln_t_table is missing"):
        preflight_components(
            H3ComponentSetSpec(str(root), transformer=str(native_transformer)),
            H3PreflightRequest(),
        )


def test_preflight_rejects_unvalidated_low_rank_adaln_curve(tmp_path: Path):
    root = _component_tree(tmp_path)
    transformer = tmp_path / "rank8_transformer.safetensors"
    _safetensors(
        transformer,
        {"adaln_t_table": ("F16", [1025, 8], 1025 * 8 * 2)},
    )

    with pytest.raises(ValueError, match="rank 8 is below the validated minimum 32"):
        preflight_components(
            H3ComponentSetSpec(str(root), transformer=str(transformer)),
            H3PreflightRequest(),
        )


def test_preflight_rejects_unknown_native_video_vae_version(tmp_path: Path):
    root = _component_tree(tmp_path)
    native_video_vae = tmp_path / "native_video_vae.safetensors"
    wrapper = {
        "format": "minimax-h3-mlx-video-vae",
        "format_version": 2,
        "tensor_layout": "ODHWI",
    }
    _safetensors(
        native_video_vae,
        {"decoder.weight": ("F16", [4, 4], 32)},
        metadata={"minimax_h3_video_vae": json.dumps(wrapper)},
    )

    with pytest.raises(ValueError, match="format version"):
        preflight_components(
            H3ComponentSetSpec(str(root), video_vae=str(native_video_vae)),
            H3PreflightRequest(),
        )


def test_preflight_reports_native_q8_video_vae(tmp_path: Path):
    root = _component_tree(tmp_path)
    native_video_vae = tmp_path / "video_vae_q8.safetensors"
    wrapper = {
        "format": "minimax-h3-mlx-video-vae",
        "format_version": 1,
        "tensor_layout": "ODHWI",
        "quantization": {
            "format": "mlx-affine",
            "bits": 8,
            "group_size": 64,
            "scope": "decoder-transformer-core",
            "quantized_layers": 144,
        },
    }
    _safetensors(
        native_video_vae,
        {"decoder.transformer_blocks.0.attn.to_qkv.weight": ("U32", [4, 4], 64)},
        metadata={"minimax_h3_video_vae": json.dumps(wrapper)},
    )

    report = preflight_components(
        H3ComponentSetSpec(str(root), video_vae=str(native_video_vae)),
        H3PreflightRequest(),
    )

    video_vae = next(component for component in report.components if component.name == "video_vae")
    assert video_vae.quantization == "mlx-affine-8bit-group-64"


def test_compact_text_encoder_report_excludes_colocated_vae_weights(tmp_path: Path):
    root = _component_tree(tmp_path)
    compact = tmp_path / "compact"
    _json(compact / "config.json", {"hidden_size": 5120})
    _safetensors(
        compact / "text_encoder.safetensors",
        {"model.layers.0.weight": ("F16", [4, 4], 32)},
    )
    _safetensors(
        compact / "video_vae.safetensors",
        {"decoder.weight": ("F16", [4, 4], 32)},
    )
    _safetensors(
        compact / "audio_vae.safetensors",
        {"decoder.weight": ("F32", [4, 4], 64)},
    )

    report = preflight_components(
        H3ComponentSetSpec(str(root), text_encoder=str(compact)),
        H3PreflightRequest(),
    )

    text_encoder = next(
        component for component in report.components if component.name == "text_encoder"
    )
    tokenizer = next(component for component in report.components if component.name == "tokenizer")
    assert text_encoder.files == ("text_encoder.safetensors",)
    assert text_encoder.tensor_bytes == 32
    assert "text_encoder.safetensors" not in tokenizer.files
    assert "video_vae.safetensors" not in tokenizer.files


def test_t2va_allows_tokenizer_only_processor_directory_with_warning(tmp_path: Path):
    root = _component_tree(tmp_path)
    processor_config = root / "processor" / "preprocessor_config.json"
    processor_config.unlink()
    _json(root / "processor" / "tokenizer.json", {"version": "1.0"})

    report = preflight_components(H3ComponentSetSpec(str(root)), H3PreflightRequest())

    assert any("valid only for text-only T2VA" in warning for warning in report.warnings)


def test_reference_task_requires_processor_configuration(tmp_path: Path):
    root = _component_tree(tmp_path, tasks=("ref2va",))
    manifest = json.loads((root / "model_index.json").read_text())
    manifest["_minimax_h3"]["partition"] = "ref2va"
    _json(root / "model_index.json", manifest)
    (root / "processor" / "preprocessor_config.json").unlink()
    _json(root / "processor" / "tokenizer.json", {"version": "1.0"})

    with pytest.raises(FileNotFoundError, match="processor requires"):
        preflight_components(
            H3ComponentSetSpec(str(root), task="ref2va"),
            H3PreflightRequest(),
        )
