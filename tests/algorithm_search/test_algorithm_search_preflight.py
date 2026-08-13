from pathlib import Path

import pytest

from minimax_h3_mlx.algorithm_search.preflight import validate_profile_components


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    return path


def test_profile_preflight_rejects_text_encoder_file_instead_of_directory(tmp_path):
    text_encoder_file = _touch(tmp_path / "text_encoder.safetensors")
    with pytest.raises(ValueError, match="text encoder must be an existing directory"):
        validate_profile_components(
            model_index=_touch(tmp_path / "model_index.json"),
            transformer=_touch(tmp_path / "transformer.safetensors"),
            text_encoder_directory=text_encoder_file,
            processor_directory=tmp_path,
            tokenizer_directory=tmp_path,
            prompt_file=_touch(tmp_path / "prompt.txt"),
        )


def test_profile_preflight_accepts_complete_component_contract(tmp_path):
    _touch(tmp_path / "text_encoder.safetensors")
    _touch(tmp_path / "tokenizer.json")
    validate_profile_components(
        model_index=_touch(tmp_path / "model_index.json"),
        transformer=_touch(tmp_path / "transformer.safetensors"),
        text_encoder_directory=tmp_path,
        processor_directory=tmp_path,
        tokenizer_directory=tmp_path,
        prompt_file=_touch(tmp_path / "prompt.txt"),
    )


def test_profile_preflight_accepts_sharded_transformer_directory(tmp_path):
    transformer = tmp_path / "transformer"
    _touch(transformer / "config.json")
    _touch(transformer / "model-00001.safetensors")
    _touch(tmp_path / "text_encoder.safetensors")
    _touch(tmp_path / "tokenizer.json")

    validate_profile_components(
        model_index=_touch(tmp_path / "model_index.json"),
        transformer=transformer,
        text_encoder_directory=tmp_path,
        processor_directory=tmp_path,
        tokenizer_directory=tmp_path,
        prompt_file=_touch(tmp_path / "prompt.txt"),
    )


def test_profile_preflight_accepts_paged_transformer_and_text_encoder(tmp_path):
    transformer = tmp_path / "transformer"
    text_encoder = tmp_path / "text_encoder"
    _touch(transformer / "config.json")
    _touch(transformer / "paged_manifest.json")
    _touch(text_encoder / "paged_text_encoder_manifest.json")
    _touch(tmp_path / "tokenizer.json")

    validate_profile_components(
        model_index=_touch(tmp_path / "model_index.json"),
        transformer=transformer,
        text_encoder_directory=text_encoder,
        processor_directory=tmp_path,
        tokenizer_directory=tmp_path,
        prompt_file=_touch(tmp_path / "prompt.txt"),
    )


def test_profile_preflight_rejects_incomplete_transformer_directory(tmp_path):
    transformer = tmp_path / "transformer"
    transformer.mkdir()
    _touch(tmp_path / "text_encoder.safetensors")
    _touch(tmp_path / "tokenizer.json")

    with pytest.raises(ValueError, match="lacks config.json"):
        validate_profile_components(
            model_index=_touch(tmp_path / "model_index.json"),
            transformer=transformer,
            text_encoder_directory=tmp_path,
            processor_directory=tmp_path,
            tokenizer_directory=tmp_path,
            prompt_file=_touch(tmp_path / "prompt.txt"),
        )
