from dataclasses import dataclass, replace
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from ltx25_mlx.conv_vae_acceleration import StagedSmallDepthConv3d
from ltx25_mlx.dfr import (
    dfr_conditioning_fps,
    extract_dfr_temporal_image_anchors,
    frozen_audio_for_tile,
    plan_dfr_temporal_tiles,
    scale_dfr_temporal_image_anchors,
    stitch_dfr_temporal_tiles,
)
from ltx25_mlx.sampling import euler_ancestral_denoise_loop
from minimax_h3_mlx.inference_optimizations import (
    TransientQ8Linear,
    compiled_scale_add,
    configure_block,
)
from wee_todd_nodes.conditioning import H3TextEncoderCache, H3TextEncoderSpec
from wee_todd_nodes.conditioning_cache import cache_key
from wee_todd_nodes.phase_memory import measured_phase
from wee_todd_nodes.runtime import H3GenerationConfig


@pytest.mark.parametrize("rows", [256, 768, 1024])
def test_transient_q8_bounded_rounding_error_and_no_expanded_cache(rows):
    source = nn.Linear(128, 256, bias=True)
    source.set_dtype(mx.bfloat16)
    base = nn.QuantizedLinear.from_linear(source, group_size=64, bits=8)
    candidate = TransientQ8Linear(base)
    x = mx.random.normal((1, rows, 128)).astype(mx.bfloat16)
    reference, actual = base(x).astype(mx.float32), candidate(x).astype(mx.float32)
    if rows < 768:
        assert mx.array_equal(reference, actual).item()
    else:
        # Generic QMM/dense kernels need not round identically. These remain opt-in.
        relative = mx.sqrt(mx.mean((reference - actual) ** 2) / mx.mean(reference**2))
        assert relative.item() < 0.01
    assert set(candidate.parameters()) == {"base"}


@pytest.mark.parametrize("rows", [1, 63, 768])
def test_compiled_adaln_strided_bf16_exact(rows):
    x, scale, shift = [
        mx.random.normal((1, rows * 2, 64)).astype(mx.bfloat16)[:, ::2] for _ in range(3)
    ]
    assert mx.array_equal(x * scale + shift, compiled_scale_add(x, scale, shift)).item()


def test_optimization_defaults_and_adapter_exclusion():
    assert H3GenerationConfig().inference_optimization == "off"
    with pytest.raises(ValueError, match="optimization"):
        replace(H3GenerationConfig(), inference_optimization="invalid").validate()
    adapter = SimpleNamespace(base=nn.QuantizedLinear(64, 64, bits=8))
    block = SimpleNamespace(
        attn=SimpleNamespace(qkv_proj=adapter, out_proj=nn.Linear(64, 64)),
        mlp=SimpleNamespace(fc1=adapter, fc2=adapter),
    )
    configure_block(block, "combined")
    assert block.compiled_adaln
    assert block.attn.qkv_proj is adapter


def test_existing_nodes_append_optimization_and_cache_controls():
    from wee_todd_nodes.nodes import WeeToddH3GenerationConfig, WeeToddH3TextEncode

    config_inputs = WeeToddH3GenerationConfig.INPUT_TYPES()["optional"]
    assert list(config_inputs)[-1] == "inference_optimization"
    assert config_inputs["inference_optimization"][1]["default"] == "off"
    text_inputs = WeeToddH3TextEncode.INPUT_TYPES()["optional"]
    assert text_inputs["persistent_cache"][1]["default"] is True


@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
@pytest.mark.parametrize("batch,stride,padding", [(1, 1, 0), (2, 2, 1)])
def test_conv_depth_windows_uneven_tail_and_spatial_geometry(dtype, batch, stride, padding):
    conv = nn.Conv3d(16, 32, (3, 3, 3), stride=(1, stride, stride), padding=(0, padding, padding))
    conv.set_dtype(dtype)
    x = mx.random.normal((batch, 9, 8, 10, 16)).astype(dtype)
    baseline = StagedSmallDepthConv3d(conv, max_window_elements=10**9)(x)
    windowed = StagedSmallDepthConv3d(conv, max_window_elements=batch * 8 * 10 * 32 * 2)(x)
    mx.eval(baseline, windowed)
    np.testing.assert_allclose(
        np.asarray(windowed.astype(mx.float32)),
        np.asarray(baseline.astype(mx.float32)),
        rtol=0.02,
        atol=0.02,
    )
    assert windowed.shape[1] == 7


def spec_fixture(tmp_path):
    root = tmp_path / "encoder"
    root.mkdir()
    (root / "config.json").write_text("{}")
    return H3TextEncoderSpec(str(root), str(root), str(root))


def test_persistent_features_survive_runtime_recreation_and_invalidate(tmp_path):
    spec = spec_fixture(tmp_path)
    calls = []

    class Encoder:
        def encode(self, prompt):
            calls.append(prompt)
            return mx.ones((3, 8), dtype=mx.bfloat16), np.array([0, 0, 0], dtype=np.int32)

    directory = tmp_path / "cache"
    a = H3TextEncoderCache(lambda _: Encoder()).encode(spec, "hello", cache_directory=directory)
    b = H3TextEncoderCache(lambda _: Encoder()).encode(spec, "hello", cache_directory=directory)
    assert calls == ["hello"]
    assert a.cache_report["status"] == "miss" and b.cache_report["status"] == "hit"
    assert mx.array_equal(a.embeddings, b.embeddings).item()
    assert a.phase_memory["run_id"] != b.phase_memory["run_id"]
    old_key = cache_key(spec, "hello", "t2va")
    assert old_key != cache_key(spec, "hello", "i2va")
    (tmp_path / "encoder" / "config.json").write_text('{"updated":true}')
    assert old_key != cache_key(spec, "hello", "t2va")
    H3TextEncoderCache(lambda _: Encoder()).encode(spec, "hello", cache_directory=directory)
    assert calls == ["hello", "hello"]


def test_corrupt_cache_fails_open_and_releases_encoder(tmp_path):
    spec = spec_fixture(tmp_path)
    directory = tmp_path / "cache"
    directory.mkdir()
    key = cache_key(spec, "hello", "t2va")
    (directory / f"{key}.safetensors").write_bytes(b"broken")
    runtime = H3TextEncoderCache(
        lambda _: SimpleNamespace(encode=lambda _: (mx.ones((2, 4)), mx.array([0, 0])))
    )
    value = runtime.encode(spec, "hello", cache_directory=directory)
    assert value.cache_report["status"] == "read_error"
    assert not runtime.loaded
    assert runtime.encode(spec, "hello", cache_directory=directory).cache_report["status"] == "hit"


def test_phase_peak_resets_and_aggregates_without_previous_jobs(monkeypatch):
    counter = iter([1000, 20, 35])
    monkeypatch.setattr(mx, "get_peak_memory", lambda: next(counter))

    @dataclass
    class Result:
        phase_memory: dict | None = None

    @measured_phase("text_encoder")
    def encode():
        return Result()

    @measured_phase("transformer")
    def sample(conditioning):
        return Result()

    previous = encode()
    current = encode()
    sampled = sample(current)
    assert previous.phase_memory["run_peak_bytes"] == 1000
    assert sampled.phase_memory["run_peak_bytes"] == 35
    assert current.phase_memory["run_peak_bytes"] == 20


@pytest.mark.parametrize("fps,expected", [(24, 24), (30, 30), (48, 60), (96, 60)])
def test_dfr_conditioning_timebase(fps, expected):
    assert dfr_conditioning_fps(fps) == expected


def test_cached_previous_prompt_phase_is_not_counted_in_current_job(monkeypatch):
    import wee_todd_nodes.phase_memory as memory

    monkeypatch.setattr(memory, "current_prompt_id", lambda: "new-prompt")
    monkeypatch.setattr(mx, "get_peak_memory", lambda: 10)

    @dataclass
    class Result:
        phase_memory: dict | None = None

    old = Result({"prompt_id": "old-prompt", "phases": [], "run_peak_bytes": 9999})

    @measured_phase("transformer")
    def sample(conditioning):
        return Result()

    result = sample(old)
    assert result.phase_memory["run_peak_bytes"] == 10
    assert old.phase_memory["run_peak_bytes"] == 9999


def test_dfr_nonzero_latent_indices_convert_to_pixel_grid_and_scale_without_flooring():
    from ltx_core_mlx.conditioning.types.latent_cond import VideoConditionByLatentIndex

    source = VideoConditionByLatentIndex([2], mx.ones((1, 1, 4)))
    anchors = extract_dfr_temporal_image_anchors([source], latent_h=1, latent_w=1)
    assert anchors[0].pixel_frame == 16
    unaligned = anchors[0]._replace(pixel_frame=17, replace=False)
    assert scale_dfr_temporal_image_anchors([unaligned])[0].pixel_frame == 34


def test_dfr_audio_window_wall_clock_and_fixed_source_across_rounds():
    source = mx.arange(100).reshape(1, 100, 1).astype(mx.float32)
    first = frozen_audio_for_tile(
        source, pixel_start=48, tile_frames=49, playback_fps=48, source_seconds=4
    )
    second = frozen_audio_for_tile(
        source, pixel_start=96, tile_frames=97, playback_fps=96, source_seconds=4
    )
    assert first.latent.shape[1] == round(49 / 60 * 25)
    assert second.latent.shape[1] == round(97 / 60 * 25)
    assert first.latent[0, 0, 0].item() == second.latent[0, 0, 0].item() == 25
    assert mx.all(first.denoise_mask == 0).item()


def test_dfr_joint_denoising_preserves_audio_but_video_depends_on_it():
    from ltx_core_mlx.conditioning.types.latent_cond import LatentState

    video = LatentState(mx.zeros((1, 3, 1)), mx.zeros((1, 3, 1)), mx.ones((1, 3, 1)))
    audio = frozen_audio_for_tile(
        mx.full((1, 25, 1), 2.0), pixel_start=0, tile_frames=49, playback_fps=48, source_seconds=1
    )
    calls = []

    def model(**kwargs):
        calls.append(kwargs)
        assert mx.all(kwargs["audio_timesteps"] == 0).item()
        assert mx.array_equal(kwargs["audio_latent"], audio.latent).item()
        return mx.ones_like(kwargs["video_latent"]) * mx.mean(
            kwargs["audio_latent"]
        ), mx.zeros_like(audio.latent)

    result = euler_ancestral_denoise_loop(
        model,
        video,
        audio,
        mx.zeros((1, 1)),
        mx.zeros((1, 1)),
        sigmas=(0.5, 0.25, 0),
        noise_seed=1,
        freeze_audio=True,
    )
    assert len(calls) == 2
    assert mx.array_equal(result.audio_latent, audio.latent).item()
    assert mx.all(result.video_latent == 2).item()


def test_dfr_seam_ownership_rejects_missing_rows():
    tiles = list(plan_dfr_temporal_tiles((48, 96), 97, 2))
    latents = [mx.zeros((1, 1, t.latent_end_exclusive - t.latent_start, 1, 1)) for t in tiles]
    tiles[1] = tiles[1]._replace(drop_latent_prefix=6)
    with pytest.raises(ValueError, match="ownership"):
        stitch_dfr_temporal_tiles(latents, tiles)


def test_dfr_active_temporal_pipeline_uses_joint_audio_every_round(monkeypatch):
    import ltx25_mlx.frozen_audio as models
    import ltx25_mlx.generated_keyframes as markers
    from ltx25_mlx.pipeline import LTX25DistilledPipeline

    observed = []

    def joint(**kwargs):
        assert kwargs["audio_latent"] is not None
        assert mx.all(kwargs["audio_timesteps"] == 0).item()
        observed.append(kwargs["audio_latent"].shape[1])
        return kwargs["video_latent"], kwargs["audio_latent"]

    monkeypatch.setattr(models, "DFRFrozenAudioX0Model", lambda _: joint)
    monkeypatch.setattr(markers, "set_generated_keyframe_marker", lambda *_: None)
    pipeline = object.__new__(LTX25DistilledPipeline)
    pipeline.low_memory = False
    pipeline._loaded_loras = pipeline.loras = ()
    pipeline.dit = object()
    pipeline._temporal_upsample = lambda x: mx.concatenate(
        [mx.repeat(x[:, :, :-1], 2, axis=2), x[:, :, -1:]], axis=2
    )
    timings = {}
    video, frames, fps = pipeline._run_dfr_temporal_rounds(
        video_latent=mx.ones((1, 128, 7, 1, 1)),
        stage1_audio=mx.ones((1, 51, 128)),
        source_seconds=49 / 24,
        carry_frames=(24, 48),
        carry_keyframes=mx.ones((1, 128, 2, 1, 1)),
        num_frames=49,
        requested_num_frames=49,
        frame_rate=24,
        rounds=2,
        latent_h=1,
        latent_w=1,
        video_embeds=mx.ones((1, 1, 4)),
        audio_embeds=mx.ones((1, 1, 4)),
        seed=1,
        check_interrupted=None,
        step_callback=None,
        timings=timings,
    )
    assert frames == 193 and fps == 96 and video.shape[2] == 25
    assert observed
    assert [item["conditioning_fps"] for item in timings["temporal_rounds"]] == [60, 60]
    assert all(
        tile["audio_source_seconds"] == 49 / 24
        for item in timings["temporal_rounds"]
        for tile in item["tiles"]
    )


def test_frozen_audio_routes_scalar_and_per_token_adaln_and_restores_on_failure():
    from ltx25_mlx.frozen_audio import frozen_audio_timestep_routing

    calls = {}
    names = (
        "audio_prompt_adaln_single",
        "av_ca_video_scale_shift_adaln_single",
        "av_ca_a2v_gate_adaln_single",
        "av_ca_audio_scale_shift_adaln_single",
    )

    class Modulation(nn.Module):
        def __init__(self, name):
            super().__init__()
            self.name = name

        def __call__(self, x):
            calls[self.name] = x
            return x, x

    owner = SimpleNamespace(**{name: Modulation(name) for name in names})
    owner._embed_timestep_scalar = lambda x: mx.repeat(x[:, None], 4, axis=1)
    originals = {name: getattr(owner, name) for name in names}

    class StreamingProxy:
        def __init__(self, inner):
            self.inner = inner

        def __getattr__(self, name):
            return getattr(self.inner, name)

    with pytest.raises(RuntimeError, match="cancelled"):
        with frozen_audio_timestep_routing(StreamingProxy(owner), mx.array([0.5, 0.25])):
            for name in names:
                getattr(owner, name)(mx.ones((6, 4)))
            raise RuntimeError("cancelled")
    assert all(getattr(owner, name) is originals[name] for name in names)
    assert mx.all(calls["audio_prompt_adaln_single"] == 0).item()
    assert mx.all(calls["av_ca_video_scale_shift_adaln_single"] == 0).item()
    assert mx.all(calls["av_ca_a2v_gate_adaln_single"] == 0).item()
    assert calls["av_ca_audio_scale_shift_adaln_single"][:, 0].tolist() == [0.5] * 3 + [0.25] * 3


def test_frozen_audio_wrapper_runs_real_joint_transformer_and_audio_affects_video():
    from ltx_core_mlx.model.transformer.model import LTXModel, LTXModelConfig

    from ltx25_mlx.frozen_audio import DFRFrozenAudioX0Model

    mx.random.seed(32)
    config = LTXModelConfig(
        num_layers=1,
        video_dim=32,
        audio_dim=16,
        video_num_heads=2,
        audio_num_heads=2,
        video_head_dim=16,
        audio_head_dim=8,
        av_cross_num_heads=2,
        av_cross_head_dim=8,
        timestep_embedding_dim=16,
        ff_mult=2,
    )
    transformer = LTXModel(config)
    transformer.set_dtype(mx.bfloat16)
    model = DFRFrozenAudioX0Model(transformer)
    kwargs = dict(
        video_latent=mx.random.normal((1, 3, 128)).astype(mx.bfloat16),
        audio_latent=mx.random.normal((1, 5, 128)).astype(mx.bfloat16),
        sigma=mx.array([0.5], dtype=mx.bfloat16),
        video_timesteps=mx.array([[0, 0.5, 0.5]], dtype=mx.bfloat16),
        audio_timesteps=mx.zeros((1, 5), dtype=mx.bfloat16),
        video_text_embeds=mx.ones((1, 2, 32), dtype=mx.bfloat16),
        audio_text_embeds=mx.ones((1, 2, 16), dtype=mx.bfloat16),
    )
    original_prompt = transformer.audio_prompt_adaln_single
    video, audio = model(**kwargs)
    changed_video, _ = model(**{**kwargs, "audio_latent": -kwargs["audio_latent"]})
    mx.eval(video, audio, changed_video)
    assert transformer.audio_prompt_adaln_single is original_prompt
    assert mx.array_equal(audio, kwargs["audio_latent"]).item()
    assert mx.all(mx.isfinite(video)).item()
    assert not mx.array_equal(video[:, 1:], changed_video[:, 1:]).item()
