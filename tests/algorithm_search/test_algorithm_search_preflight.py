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
