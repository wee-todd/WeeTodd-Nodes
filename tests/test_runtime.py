from pathlib import Path

import pytest

from wee_todd_nodes.runtime import H3GenerationConfig, H3ModelSpec, H3RuntimeCache


def test_config_accepts_small_wiring_canvas():
    H3GenerationConfig(width=640, height=384, steps=8).validate()


def test_low_memory_mode_selects_query_chunks_without_quantization():
    config = H3GenerationConfig(memory_mode="low_memory_bf16")
    config.validate()
    assert config.attention_query_chunk_size == 512


@pytest.mark.parametrize("chunk", ["512", "1024", "2048"])
def test_low_memory_mode_accepts_explicit_query_chunk(chunk):
    config = H3GenerationConfig(memory_mode="low_memory_bf16", attention_chunk_size=chunk)
    config.validate()
    assert config.attention_query_chunk_size == int(chunk)


def test_normal_mode_ignores_attention_chunk_selection():
    config = H3GenerationConfig(memory_mode="normal", attention_chunk_size="2048")
    config.validate()
    assert config.attention_query_chunk_size is None


def test_config_rejects_unknown_memory_mode():
    with pytest.raises(ValueError, match="memory_mode"):
        H3GenerationConfig(memory_mode="tiny").validate()


def test_config_accepts_experimental_mpp_projection_backend():
    H3GenerationConfig(projection_backend="mpp_experimental").validate()


def test_config_rejects_unknown_projection_backend():
    with pytest.raises(ValueError, match="projection_backend"):
        H3GenerationConfig(projection_backend="unknown").validate()


def test_config_accepts_res_multistep_sampling():
    H3GenerationConfig(sampling_method="res_multistep").validate()


def test_config_rejects_unknown_sampling_method():
    with pytest.raises(ValueError, match="sampling_method"):
        H3GenerationConfig(sampling_method="unknown").validate()


@pytest.mark.parametrize("width,height", [(641, 384), (640, 385)])
def test_config_rejects_unaligned_canvas(width, height):
    with pytest.raises(ValueError, match="divisible by 32"):
        H3GenerationConfig(width=width, height=height).validate()


def test_config_rejects_nonpositive_canvas():
    with pytest.raises(ValueError, match="at least 32"):
        H3GenerationConfig(width=0, height=384).validate()


@pytest.mark.parametrize(("width", "height"), [(1952, 1088), (1088, 1952)])
def test_generation_config_rejects_dimensions_above_1920(width, height):
    with pytest.raises(ValueError, match="must not exceed 1920"):
        H3GenerationConfig(width=width, height=height).validate()


def test_model_spec_rejects_missing_checkpoint(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        H3ModelSpec(str(tmp_path / "missing")).validate()


def test_unload_forgets_pipeline_and_spec(tmp_path: Path):
    cache = H3RuntimeCache()
    cache._pipeline = object()
    cache._spec = H3ModelSpec(str(tmp_path))
    assert cache.loaded is True

    cache.unload()

    assert cache.loaded is False
    assert cache._pipeline is None
    assert cache._spec is None
