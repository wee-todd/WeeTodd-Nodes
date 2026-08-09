import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from minimax_h3_mlx.ref2va import PreparedReference
from minimax_h3_mlx.text_encoder import MiniMaxH3TextEncoder
from wee_todd_nodes.conditioning import H3TextEncoderCache, H3TextEncoderSpec
from wee_todd_nodes.preflight import H3ComponentSetSpec


class _Tags:
    shape = (3,)


class FakeEncoder:
    def __init__(self, spec, fail=False):
        self.spec = spec
        self.fail = fail
        self.prompts = []

    def encode(self, prompt):
        self.prompts.append(prompt)
        if self.fail:
            raise RuntimeError("synthetic encoder failure")
        return "live-mlx-embeddings", _Tags()


def test_ref2va_request_uses_independent_labels_and_ordered_visual_units():
    calls = []

    class Tokenizer:
        tokens = {
            "<|vision_start|>": 100,
            "<|vision_end|>": 101,
            "<|image_pad|>": 200,
            "<|video_pad|>": 201,
        }

        def convert_tokens_to_ids(self, value):
            return self.tokens[value]

        def __call__(self, value, add_special_tokens=False):
            calls.append(value)
            return {"input_ids": [1000 + len(calls)]}

    class ImageProcessor:
        merge_size = 2

        def __call__(self, **kwargs):
            return {
                "pixel_values": np.zeros((4, 3), dtype=np.float32),
                "image_grid_thw": np.array([[1, 2, 2]], dtype=np.int32),
            }

    class VideoProcessor:
        def __call__(self, **kwargs):
            return {
                "pixel_values_videos": np.zeros((4, 3), dtype=np.float32),
                "video_grid_thw": np.array([[1, 2, 2]], dtype=np.int32),
            }

    encoder = object.__new__(MiniMaxH3TextEncoder)
    encoder._tokenizer = Tokenizer()
    encoder._processor = SimpleNamespace(
        image_processor=ImageProcessor(), video_processor=VideoProcessor()
    )
    encoder.merge_size = 2
    encoder.image_token_id = 200
    references = [
        PreparedReference("audio", waveform=np.zeros((2, 3200), dtype=np.float32)),
        PreparedReference("image", image=np.zeros((32, 32, 3), dtype=np.uint8)),
        PreparedReference(
            "video",
            frames=np.zeros((13, 32, 32, 3), dtype=np.uint8),
            waveform=np.zeros((2, 3200), dtype=np.float32),
        ),
    ]

    _, tags, units = encoder._build_reference_request("prompt", references)

    assert calls == [
        "<Audio 1>: ",
        "<Picture 1>: ",
        "<Audio 2>: ",
        "<Video 1>: ",
        "<0.2 seconds>",
        "prompt",
    ]
    assert [unit[0] for unit in units] == [200, 201]
    assert np.count_nonzero(tags == 0) == 6


def _spec(tmp_path: Path, name="encoder") -> H3TextEncoderSpec:
    text_encoder = tmp_path / name
    processor = tmp_path / f"{name}-processor"
    tokenizer = tmp_path / f"{name}-tokenizer"
    for directory in (text_encoder, processor, tokenizer):
        directory.mkdir()
    (text_encoder / "config.json").write_text("{}\n")
    return H3TextEncoderSpec(
        text_encoder=str(text_encoder),
        processor=str(processor),
        tokenizer=str(tokenizer),
    )


def test_conditioning_cache_can_unload_after_encode(tmp_path: Path):
    created = []

    def factory(spec):
        encoder = FakeEncoder(spec)
        created.append(encoder)
        return encoder

    cache = H3TextEncoderCache(factory)
    conditioning = cache.encode(_spec(tmp_path), "A test prompt", unload_after=True)

    assert conditioning.token_count == 3
    assert conditioning.embeddings == "live-mlx-embeddings"
    assert cache.loaded is False
    assert len(created) == 1


def test_conditioning_retains_paged_encoder_report_after_unload(tmp_path: Path):
    def factory(spec):
        encoder = FakeEncoder(spec)
        encoder.paged_layers = SimpleNamespace(
            report=lambda: {"format": "weetodd-h3-qwen-paged-v1", "layers_loaded": 50}
        )
        return encoder

    conditioning = H3TextEncoderCache(factory).encode(
        _spec(tmp_path), "A test prompt", unload_after=True
    )

    assert conditioning.paging_report == {
        "format": "weetodd-h3-qwen-paged-v1",
        "layers_loaded": 50,
    }


def test_conditioning_cache_reuses_compatible_encoder(tmp_path: Path):
    created = []

    def factory(spec):
        encoder = FakeEncoder(spec)
        created.append(encoder)
        return encoder

    cache = H3TextEncoderCache(factory)
    spec = _spec(tmp_path)
    cache.encode(spec, "First prompt", unload_after=False)
    cache.encode(spec, "Second prompt", unload_after=False)

    assert cache.loaded is True
    assert len(created) == 1
    assert created[0].prompts == ["First prompt", "Second prompt"]


def test_conditioning_cache_replaces_incompatible_encoder(tmp_path: Path):
    created = []

    def factory(spec):
        encoder = FakeEncoder(spec)
        created.append(encoder)
        return encoder

    cache = H3TextEncoderCache(factory)
    cache.encode(_spec(tmp_path, "first"), "First prompt", unload_after=False)
    second = _spec(tmp_path, "second")
    cache.encode(second, "Second prompt", unload_after=False)

    assert len(created) == 2
    assert cache.spec == second


def test_conditioning_failure_releases_encoder(tmp_path: Path):
    cache = H3TextEncoderCache(lambda spec: FakeEncoder(spec, fail=True))

    with pytest.raises(RuntimeError, match="synthetic encoder failure"):
        cache.encode(_spec(tmp_path), "A test prompt", unload_after=False)

    assert cache.loaded is False
    assert cache.spec is None


def test_conditioning_rejects_empty_prompt_before_loading(tmp_path: Path):
    called = False

    def factory(spec):
        nonlocal called
        called = True
        return FakeEncoder(spec)

    cache = H3TextEncoderCache(factory)
    with pytest.raises(ValueError, match="Prompt must contain text"):
        cache.encode(_spec(tmp_path), "   ")

    assert called is False


def test_conditioning_rejects_empty_prompt_before_staged_release(tmp_path: Path):
    prepared = False

    def prepare_stage():
        nonlocal prepared
        prepared = True

    with pytest.raises(ValueError, match="Prompt must contain text"):
        H3TextEncoderCache().encode(
            _spec(tmp_path), "   ", prepare_stage=prepare_stage
        )

    assert prepared is False


def test_compact_encoder_uses_checkpoint_architecture_config(tmp_path: Path):
    root = tmp_path / "FL2VA"
    compact = tmp_path / "compact"
    for directory in (root / "text_encoder", compact, compact / "processor"):
        directory.mkdir(parents=True)
    (root / "text_encoder" / "config.json").write_text('{"text_config": {}}\n')
    (compact / "config.json").write_text('{"model_type": "minimax_h3"}\n')

    spec = H3TextEncoderSpec.from_components(
        H3ComponentSetSpec(
            checkpoint=str(root),
            text_encoder=str(compact),
            processor=str(compact / "processor"),
            tokenizer=str(compact),
        )
    )

    assert spec.config_path == str(root / "text_encoder" / "config.json")


def test_paged_encoder_prefers_packaged_architecture_config(tmp_path: Path):
    root = tmp_path / "FL2VA"
    paged = tmp_path / "paged"
    processor = tmp_path / "processor"
    tokenizer = tmp_path / "tokenizer"
    for directory in (root / "text_encoder", paged, processor, tokenizer):
        directory.mkdir(parents=True)
    (root / "text_encoder" / "config.json").write_text('{"text_config": {}}\n')
    (paged / "config.json").write_text('{"model_type": "minimax_h3"}\n')
    packaged = paged / "architecture_config.json"
    packaged.write_text('{"text_config": {}}\n')

    spec = H3TextEncoderSpec.from_components(
        H3ComponentSetSpec(
            checkpoint=str(root),
            text_encoder=str(paged),
            processor=str(processor),
            tokenizer=str(tokenizer),
        )
    )

    assert spec.config_path == str(packaged)


def test_node_import_does_not_import_mlx():
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(Path(__file__).parents[1] / "src")
    code = (
        "import sys; import wee_todd_nodes.nodes; "
        "assert 'mlx' not in sys.modules and 'mlx.core' not in sys.modules"
    )

    result = subprocess.run(
        [sys.executable, "-c", code],
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_comfy_entrypoint_imports_without_mlx():
    root = Path(__file__).parents[1]
    code = (
        "import importlib.util, pathlib, sys; "
        f"root=pathlib.Path({str(root)!r}); "
        "spec=importlib.util.spec_from_file_location('weetodd_custom_node', root/'__init__.py', "
        "submodule_search_locations=[str(root)]); "
        "module=importlib.util.module_from_spec(spec); "
        "sys.modules[spec.name]=module; spec.loader.exec_module(module); "
        "assert len(module.NODE_CLASS_MAPPINGS) == 31; "
        "assert 'mlx' not in sys.modules and 'mlx.core' not in sys.modules"
    )

    result = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
