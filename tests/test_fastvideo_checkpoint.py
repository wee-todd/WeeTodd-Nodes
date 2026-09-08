import json

import mlx.core as mx

from minimax_h3_mlx.fastvideo_checkpoint import _quantize_values, convert_fastvideo_fasth3_to_paged
from minimax_h3_mlx.paged_checkpoint import PagedCheckpointManifest
from minimax_h3_mlx.quantize import QuantConfig


def test_fastvideo_dense_checkpoint_converts_directly_to_native_pages(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    config = {
        "hidden_size": 2,
        "num_layers": 1,
        "num_refiner_layers": 0,
        "num_attention_heads": 2,
        "attention_head_dim": 2,
        "ffn_dim": 2,
        "in_channels": 1,
        "audio_in_channels": 1,
        "patch_size": [1, 1, 1],
        "text_dim": 2,
        "freq_dim": 2,
        "time_embed_hidden_dim": 2,
        "time_embed_dim": 2,
        "rope_freq_dim": 1,
    }
    (source / "config.json").write_text(json.dumps(config))
    q = mx.arange(1, 9, dtype=mx.float32).reshape(4, 2)
    k = mx.arange(11, 19, dtype=mx.float32).reshape(4, 2)
    v = mx.arange(21, 29, dtype=mx.float32).reshape(4, 2)
    fc1 = mx.array([[1, 1], [2, 2], [3, 3], [4, 4]], dtype=mx.float32)
    tensors = {
        "proj_in.weight": mx.ones((2, 1)),
        "transformer_blocks.0.attn.to_q.weight": q,
        "transformer_blocks.0.attn.to_k.weight": k,
        "transformer_blocks.0.attn.to_v.weight": v,
        "transformer_blocks.0.attn.to_out.0.weight": mx.ones((2, 2)),
        "transformer_blocks.0.attn.norm_q.weight": mx.ones((2,)),
        "transformer_blocks.0.attn.norm_k.weight": mx.ones((2,)),
        "transformer_blocks.0.ff.net.0.proj.weight": fc1,
        "transformer_blocks.0.ff.net.2.weight": mx.ones((2, 2)),
        "transformer_blocks.0.norm1.weight": mx.ones((2,)),
    }
    shard = "diffusion_pytorch_model-00001-of-00001.safetensors"
    mx.save_safetensors(source / shard, tensors)
    (source / "diffusion_pytorch_model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {key: shard for key in tensors}})
    )

    output = tmp_path / "native"
    manifest = convert_fastvideo_fasth3_to_paged(source, output)
    block = mx.load(output / manifest.blocks[0].file)
    fixed = mx.load(output / manifest.fixed.file)

    expected_qkv = mx.array(
        [
            [1, 2], [3, 4], [11, 12], [13, 14], [21, 22], [23, 24],
            [5, 6], [7, 8], [15, 16], [17, 18], [25, 26], [27, 28],
        ],
        dtype=mx.float32,
    )
    assert mx.array_equal(block["blocks.0.attn.qkv_proj.weight"], expected_qkv)
    assert mx.array_equal(block["blocks.0.mlp.fc1.weight"], fc1[[2, 3, 0, 1]])
    assert "blocks.0.attn.out_proj.weight" in block
    assert "blocks.0.attn.q_norm.weight" in block
    assert "blocks.0.attn.k_norm.weight" in block
    assert "video_patch_proj.weight" in fixed
    assert PagedCheckpointManifest.load(output, verify_hashes=True).num_blocks == 1
    native_config = json.loads((output / "config.json").read_text())
    assert native_config["token_refiner_num_layers"] == 0
    assert native_config["vsa_gate"] is False


def test_fastvideo_vsa_checkpoint_preserves_trained_gate_in_native_page(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    config = {
        "hidden_size": 2,
        "num_layers": 1,
        "num_refiner_layers": 0,
        "num_attention_heads": 2,
        "attention_head_dim": 2,
        "ffn_dim": 2,
        "in_channels": 1,
        "audio_in_channels": 1,
        "patch_size": [1, 1, 1],
        "text_dim": 2,
        "freq_dim": 2,
        "time_embed_hidden_dim": 2,
        "time_embed_dim": 2,
        "rope_freq_dim": 1,
    }
    (source / "config.json").write_text(json.dumps(config))
    tensors = {
        "proj_in.weight": mx.ones((2, 1)),
        "transformer_blocks.0.attn.to_q.weight": mx.ones((4, 2)),
        "transformer_blocks.0.attn.to_k.weight": mx.ones((4, 2)),
        "transformer_blocks.0.attn.to_v.weight": mx.ones((4, 2)),
        "transformer_blocks.0.attn.to_gate_compress.weight": mx.arange(
            8, dtype=mx.float32
        ).reshape(4, 2),
    }
    shard = "diffusion_pytorch_model-00001-of-00001.safetensors"
    mx.save_safetensors(source / shard, tensors)
    (source / "diffusion_pytorch_model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {key: shard for key in tensors}})
    )

    output = tmp_path / "native"
    manifest = convert_fastvideo_fasth3_to_paged(source, output)
    block = mx.load(output / manifest.blocks[0].file)
    native_config = json.loads((output / "config.json").read_text())
    raw_manifest = json.loads((output / "paged_manifest.json").read_text())

    assert native_config["vsa_gate"] is True
    assert "blocks.0.attn.gate_compress.weight" in block
    assert raw_manifest["attention"] == "vsa_h3_64_90"


def test_fastvideo_page_quantization_writes_native_mlx_affine_tensors():
    weight = mx.arange(4 * 32, dtype=mx.float32).reshape(4, 32)
    values = _quantize_values(
        {"blocks.0.attn.out_proj.weight": weight},
        QuantConfig(bits=8, group_size=32),
    )

    assert values["blocks.0.attn.out_proj.weight"].dtype == mx.uint32
    assert values["blocks.0.attn.out_proj.weight"].shape == (4, 8)
    assert values["blocks.0.attn.out_proj.scales"].shape == (4, 1)
    assert values["blocks.0.attn.out_proj.biases"].shape == (4, 1)
