"""Path-contract validation for local algorithm-search probes."""

from __future__ import annotations

from pathlib import Path


def validate_profile_components(
    *,
    model_index: Path,
    transformer: Path,
    text_encoder_directory: Path,
    processor_directory: Path,
    tokenizer_directory: Path,
    prompt_file: Path,
    text_config: Path | None = None,
) -> None:
    """Reject incorrect file-versus-directory component arguments before model loading."""
    files = {
        "model index": model_index,
        "transformer": transformer,
        "prompt": prompt_file,
    }
    if text_config is not None:
        files["text config"] = text_config
    for label, path in files.items():
        if not path.is_file():
            raise ValueError(f"{label} must be an existing file: {path}")

    directories = {
        "text encoder": text_encoder_directory,
        "processor": processor_directory,
        "tokenizer": tokenizer_directory,
    }
    for label, path in directories.items():
        if not path.is_dir():
            raise ValueError(f"{label} must be an existing directory: {path}")
    if not (text_encoder_directory / "text_encoder.safetensors").is_file() and not list(
        text_encoder_directory.glob("*.safetensors")
    ):
        raise ValueError(
            f"text encoder directory contains no safetensors: {text_encoder_directory}"
        )
    if not (tokenizer_directory / "tokenizer.json").is_file():
        raise ValueError(f"tokenizer directory lacks tokenizer.json: {tokenizer_directory}")
