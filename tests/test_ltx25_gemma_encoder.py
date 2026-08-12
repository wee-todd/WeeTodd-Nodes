import json

import mlx.core as mx
import numpy as np
import pytest
from ltx_core_mlx.text_encoders.gemma.feature_extractor import GemmaFeaturesExtractorV2
from mlx.utils import tree_flatten
from mlx_lm.models.gemma4 import Model, ModelArgs
from safetensors.numpy import save_file
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from ltx25_mlx.gemma_encoder import (
    LTX25Gemma4Conditioner,
    collect_gemma4_hidden_states,
    load_gemma4_backbone,
    load_gemma4_feature_extractor,
    load_gemma4_tokenizer,
    tokenize_gemma4,
)
from ltx25_mlx.gemma_pack import gemma4_mlx_model_config


def _tiny_config():
    return {
        "model_type": "gemma4_unified",
        "gemma_version": "tiny-test",
        "text_config": {
            "vocab_size": 32,
            "hidden_size": 8,
            "num_hidden_layers": 1,
            "intermediate_size": 16,
            "num_attention_heads": 1,
            "num_key_value_heads": 1,
            "num_global_key_value_heads": 1,
            "head_dim": 8,
            "global_head_dim": 8,
            "layer_types": ["full_attention"],
            "num_kv_shared_layers": 0,
            "use_double_wide_mlp": False,
            "max_position_embeddings": 32,
            "sliding_window": 16,
            "rope_parameters": {
                "full_attention": {"rope_type": "default", "rope_theta": 10000.0},
                "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
            },
        },
    }


def _tiny_pack(path):
    config = _tiny_config()
    model = Model(ModelArgs.from_dict(gemma4_mlx_model_config(config)))
    tensors = {
        "model.language_model." + key.removeprefix("language_model.model."): np.array(value)
        for key, value in tree_flatten(model.parameters())
    }
    feature = GemmaFeaturesExtractorV2(
        caption_channels=8,
        num_gemma_layers=2,
        video_dim=8,
        audio_dim=8,
        num_heads=1,
        video_head_dim=8,
        audio_head_dim=8,
        num_connector_layers=1,
        num_registers=2,
    )
    for key, value in tree_flatten(feature.parameters()):
        if key.startswith("connector.text_embedding_projection."):
            packed_key = key.removeprefix("connector.")
        elif key.startswith("connector.video_embeddings_connector."):
            packed_key = "model.diffusion_model." + key.removeprefix("connector.")
        elif key.startswith("connector.audio_embeddings_connector."):
            packed_key = "model.diffusion_model." + key.removeprefix("connector.")
        else:
            raise AssertionError(f"Unexpected feature key: {key}")
        tensors[packed_key] = np.array(value)
    tokenizer = Tokenizer(
        WordLevel(
            {"<pad>": 0, "<bos>": 1, "<eos>": 2, "hello": 3, "world": 4, "<unk>": 5},
            unk_token="<unk>",
        )
    )
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer_config = {
        "pad_token": "<pad>",
        "bos_token": "<bos>",
        "eos_token": "<eos>",
        "unk_token": "<unk>",
    }
    tensors.update(
        {
            "tokenizer_json": np.frombuffer(tokenizer.to_str().encode(), dtype=np.uint8),
            "hf_asset__tokenizer_config.json": np.frombuffer(
                json.dumps(tokenizer_config).encode(), dtype=np.uint8
            ),
            "hf_asset__processor_config.json": np.frombuffer(b"{}", dtype=np.uint8),
        }
    )
    save_file(tensors, path, metadata={"gemma_config": json.dumps(config)})
    return model, feature


def test_direct_gemma4_backbone_loads_strictly_and_matches_tiny_forward(tmp_path):
    pack = tmp_path / "gemma4.safetensors"
    expected, _feature = _tiny_pack(pack)
    loaded, config, report = load_gemma4_backbone(pack)

    tokens = mx.array([[1, 2, 3]])
    expected_output = expected(tokens)
    loaded_output = loaded(tokens)
    mx.eval(expected_output, loaded_output)

    assert mx.array_equal(expected_output, loaded_output).item()
    assert config["text_config"]["hidden_size_per_layer_input"] == 0
    assert report["weight_layout"] == "huggingface_unified"


def test_gemma4_hidden_state_collection_reports_layers_and_honors_padding(tmp_path):
    pack = tmp_path / "gemma4.safetensors"
    _tiny_pack(pack)
    loaded, _config, _report = load_gemma4_backbone(pack)
    progress = []
    states, mask = collect_gemma4_hidden_states(
        loaded,
        mx.array([[0, 1, 2]]),
        mx.array([[0, 1, 1]]),
        progress_callback=lambda current, total: progress.append((current, total)),
    )
    mx.eval(*states)
    assert len(states) == 2
    assert states[0].shape == (1, 3, 8)
    assert mask.tolist() == [[0, 1, 1]]
    assert progress == [(1, 1)]


def test_gemma4_hidden_state_collection_cancels_before_layer(tmp_path):
    pack = tmp_path / "gemma4.safetensors"
    _tiny_pack(pack)
    loaded, _config, _report = load_gemma4_backbone(pack)
    with pytest.raises(InterruptedError, match="cancelled"):
        collect_gemma4_hidden_states(loaded, mx.array([[1, 2]]), is_cancelled=lambda: True)


def test_direct_gemma4_feature_extractor_matches_tiny_pack(tmp_path):
    pack = tmp_path / "gemma4.safetensors"
    model, expected = _tiny_pack(pack)
    loaded = load_gemma4_feature_extractor(
        pack,
        num_heads=1,
        video_head_dim=8,
        audio_head_dim=8,
        num_connector_layers=1,
        num_registers=2,
    )
    states, mask = collect_gemma4_hidden_states(model, mx.array([[1, 2, 3, 4]]))
    expected_video, expected_audio = expected(states, attention_mask=mask)
    loaded_video, loaded_audio = loaded(states, attention_mask=mask)
    mx.eval(expected_video, expected_audio, loaded_video, loaded_audio)
    assert mx.array_equal(expected_video, loaded_video).item()
    assert mx.array_equal(expected_audio, loaded_audio).item()


def test_embedded_gemma4_tokenizer_uses_left_padding_and_bos(tmp_path):
    pack = tmp_path / "gemma4.safetensors"
    _tiny_pack(pack)
    tokenizer = load_gemma4_tokenizer(pack, max_length=4)
    token_ids, mask = tokenize_gemma4(tokenizer, "hello world", max_length=4)
    assert token_ids.tolist() == [[0, 1, 3, 4]]
    assert mask.tolist() == [[0, 1, 1, 1]]


def test_gemma4_conditioner_runs_and_frees_all_owned_components(tmp_path):
    pack = tmp_path / "gemma4.safetensors"
    _tiny_pack(pack)
    conditioner = LTX25Gemma4Conditioner(
        pack,
        max_length=4,
        feature_kwargs={
            "num_heads": 1,
            "video_head_dim": 8,
            "audio_head_dim": 8,
            "num_connector_layers": 1,
            "num_registers": 2,
        },
    )
    video, audio, mask = conditioner.encode("hello world")
    assert video.shape == (1, 4, 8)
    assert audio.shape == (1, 4, 8)
    assert mask.tolist() == [[0, 1, 1, 1]]
    conditioner.free()
    assert conditioner.model is None
    assert conditioner.feature_extractor is None
    assert conditioner.tokenizer is None
