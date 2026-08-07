from __future__ import annotations

import hashlib
import json
from dataclasses import asdict

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten, tree_unflatten

from minimax_h3_mlx.algorithm_search.block_quantization import quantize_selected_blocks
from minimax_h3_mlx.config import TAG_AUDIO, TAG_TEXT, TAG_VIDEO, DiTConfig
from minimax_h3_mlx.dit import MiniMaxH3DiT
from minimax_h3_mlx.load import load_dit
from minimax_h3_mlx.mixed_checkpoint import (
    MIXED_CHECKPOINT_FORMAT,
    Q8_CONSERVATIVE_PROFILE,
    Q8_EXTENDED_PROFILE,
    accepted_q8_blocks_38_49_recipe,
    block_core_paths,
    convert_mixed_checkpoint,
    extended_q8_mlp_recipe,
    q8_profile_info,
    validate_named_q8_checkpoint,
)
from minimax_h3_mlx.quantize import QuantConfig


def _config() -> DiTConfig:
    hidden = 256
    return DiTConfig(
        hidden_size=hidden,
        num_layers=2,
        token_refiner_num_layers=1,
        num_attention_heads=4,
        attention_head_dim=64,
        ffn_hidden_size=128,
        latents_dim=4,
        audio_latents_dim=8,
        text_dim=128,
        timestep_input_dim=16,
        time_embed_hidden_size=hidden,
        time_embed_dim=64,
        adaln_out_features=6 * 3 * hidden,
        final_adaln_out_features=2 * hidden,
        rope_inv_freq_len=4,
    )


def _source(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    config = _config()
    model = MiniMaxH3DiT(config)
    mx.eval(model.parameters())
    weights = dict(tree_flatten(model.parameters()))
    mx.save_safetensors(str(source / "model.safetensors"), weights)
    (source / "config.json").write_text(json.dumps(asdict(config)))
    return source, config, weights


def _inputs(config: DiTConfig):
    text_rows, video_rows, audio_rows = 4, 8, 4
    rows = text_rows + video_rows + audio_rows
    generator = np.random.default_rng(7)
    tags = np.concatenate(
        [
            np.full(text_rows, TAG_TEXT),
            np.full(video_rows, TAG_VIDEO),
            np.full(audio_rows, TAG_AUDIO),
        ]
    ).astype(np.int32)
    timestep_indices = np.concatenate(
        [np.zeros(text_rows), np.ones(video_rows), np.zeros(audio_rows)]
    ).astype(np.int32)
    positions = np.stack(
        [np.arange(rows) % 3, np.arange(rows) % 5, np.arange(rows) % 7], axis=-1
    ).astype(np.float32)
    return (
        mx.array(generator.standard_normal((1, video_rows, config.video_patch_dim))),
        mx.array(generator.standard_normal((1, audio_rows, config.audio_latents_dim))),
        mx.array(generator.standard_normal((1, text_rows, config.text_dim))),
        mx.array(np.array([0.0, 0.6], np.float32)),
        mx.array(timestep_indices),
        mx.array(tags),
        mx.array(positions),
        mx.array(np.arange(text_rows, text_rows + video_rows, dtype=np.int32)),
        mx.array(np.arange(text_rows + video_rows, rows, dtype=np.int32)),
        mx.array(np.arange(text_rows, dtype=np.int32)),
    )


def test_accepted_recipe_selects_only_late_core_blocks():
    recipe = accepted_q8_blocks_38_49_recipe()
    assert recipe.quantize_core is False
    assert len(recipe.overrides) == 48
    assert set(recipe.overrides.values()) == {8}
    assert min(int(path.split(".")[1]) for path in recipe.overrides) == 38
    assert max(int(path.split(".")[1]) for path in recipe.overrides) == 49


def test_extended_recipe_adds_only_middle_mlp_projections():
    conservative = accepted_q8_blocks_38_49_recipe()
    extended = extended_q8_mlp_recipe()

    assert len(conservative.overrides) == 48
    assert len(extended.overrides) == 82
    added = set(extended.overrides).difference(conservative.overrides)
    assert len(added) == 34
    assert all(21 <= int(path.split(".")[1]) <= 37 for path in added)
    assert all(path.endswith((".mlp.fc1", ".mlp.fc2")) for path in added)
    assert q8_profile_info(Q8_EXTENDED_PROFILE)["parameter_bytes_saved"] == 8_020_131_840


def test_streamed_checkpoint_loads_directly_and_matches_in_memory_quantization(tmp_path):
    source, config, weights = _source(tmp_path)
    output = tmp_path / "mixed"
    selected = block_core_paths([1])
    recipe = QuantConfig(
        bits=8,
        group_size=64,
        overrides={path: 8 for path in selected},
        quantize_core=False,
    )

    report = convert_mixed_checkpoint(source, output, recipe, max_shard_bytes=64 * 1024)

    assert report["format"] == MIXED_CHECKPOINT_FORMAT
    assert report["selected_modules"] == 4
    assert report["shards"] > 1
    assert report["peak_buffered_output_bytes"] < sum(value.nbytes for value in weights.values())
    quant_config = json.loads((output / "quant_config.json").read_text())
    assert quant_config["quantize_core"] is False
    assert quant_config["overrides"] == dict(sorted(recipe.overrides.items()))
    assert quant_config["profile"] is None
    assert (
        quant_config["source"][0]["sha256"]
        == hashlib.sha256((source / "model.safetensors").read_bytes()).hexdigest()
    )

    expected = MiniMaxH3DiT(config)
    expected.update(tree_unflatten(list(weights.items())))
    quantize_selected_blocks(expected, [1], bits=8, group_size=64)
    loaded = load_dit(output)

    expected_keys = sorted(key for key, _ in tree_flatten(expected.parameters()))
    loaded_keys = sorted(key for key, _ in tree_flatten(loaded.parameters()))
    assert loaded_keys == expected_keys
    arguments = _inputs(config)
    expected_video, expected_audio = expected(*arguments)
    loaded_video, loaded_audio = loaded(*arguments)
    mx.eval(expected_video, expected_audio, loaded_video, loaded_audio)
    assert mx.array_equal(expected_video, loaded_video).item()
    assert mx.array_equal(expected_audio, loaded_audio).item()


def test_named_checkpoint_validation_accepts_exact_profile(tmp_path):
    directory = tmp_path / "q8"
    directory.mkdir()
    recipe = accepted_q8_blocks_38_49_recipe()
    (directory / "config.json").write_text("{}\n")
    (directory / "model.safetensors.index.json").write_text("{}\n")
    (directory / "quant_config.json").write_text(
        json.dumps(
            {
                "format": MIXED_CHECKPOINT_FORMAT,
                "format_version": 1,
                "profile": Q8_CONSERVATIVE_PROFILE,
                "bits": 8,
                "group_size": 64,
                "quantize_core": False,
                "quantize_adaln": False,
                "overrides": recipe.overrides,
            }
        )
    )

    info = validate_named_q8_checkpoint(directory, Q8_CONSERVATIVE_PROFILE)

    assert info["selected_modules"] == 48
    with pytest.raises(ValueError, match="does not match"):
        validate_named_q8_checkpoint(directory, Q8_EXTENDED_PROFILE)


def test_streamed_checkpoint_cleans_partial_output_on_recipe_mismatch(tmp_path):
    source, _, _ = _source(tmp_path)
    output = tmp_path / "mixed"
    recipe = QuantConfig(
        bits=8,
        group_size=64,
        overrides={path: 8 for path in block_core_paths([99])},
        quantize_core=False,
    )

    with pytest.raises(KeyError, match="did not match"):
        convert_mixed_checkpoint(source, output, recipe, max_shard_bytes=64 * 1024)

    assert not output.exists()
    assert not list(tmp_path.glob(".mixed.*.tmp"))
