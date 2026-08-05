from pathlib import Path

import pytest

from wee_todd_nodes.runtime import H3GenerationConfig, H3ModelSpec


def test_config_accepts_small_wiring_canvas():
    H3GenerationConfig(width=640, height=384, steps=8).validate()


@pytest.mark.parametrize("width,height", [(641, 384), (640, 385)])
def test_config_rejects_unaligned_canvas(width, height):
    with pytest.raises(ValueError, match="divisible by 32"):
        H3GenerationConfig(width=width, height=height).validate()


def test_model_spec_rejects_missing_checkpoint(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        H3ModelSpec(str(tmp_path / "missing")).validate()
