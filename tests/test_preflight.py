import json
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


def _safetensors(path: Path, tensors: dict[str, tuple[str, list[int], int]]) -> None:
    """Write a header-valid file with opaque payload bytes."""

    path.parent.mkdir(parents=True, exist_ok=True)
    offset = 0
    header = {}
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


def test_preflight_rejects_unsupported_quantization(tmp_path: Path):
    root = _component_tree(tmp_path)
    _json(root / "transformer" / "quant_config.json", {"bits": 3, "group_size": 64})

    with pytest.raises(ValueError, match="quantization bits must be 4 or 8"):
        preflight_components(H3ComponentSetSpec(str(root)), H3PreflightRequest())


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
    tokenizer = next(
        component for component in report.components if component.name == "tokenizer"
    )
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
