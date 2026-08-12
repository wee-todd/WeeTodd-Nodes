import json
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import pytest

from minimax_h3_mlx.blockcache import H3BlockCacheConfig
from minimax_h3_mlx.trajectory_forecast import H3TrajectoryForecastConfig
from wee_todd_nodes.conditioning import H3Conditioning, H3TextEncoderSpec
from wee_todd_nodes.continuation import H3ContinuationContext
from wee_todd_nodes.lora import H3LoRASpec, H3LoRAStack
from wee_todd_nodes.runtime import H3GenerationConfig
from wee_todd_nodes.sampling import H3Latents, H3TransformerCache, H3TransformerSpec


class FakeDiT:
    def __init__(self):
        self.query_chunk_size = None

    def set_attention_query_chunk_size(self, value):
        self.query_chunk_size = value


class FakeSampler:
    def __init__(self, spec, fail=False):
        self.spec = spec
        self.fail = fail
        self.calls = []
        self.dit = FakeDiT()

    def sample_latents(self, embeddings, token_tags, **kwargs):
        self.calls.append((embeddings, token_tags, kwargs))
        if self.fail:
            raise RuntimeError("synthetic sampling failure")
        callback = kwargs.get("step_callback")
        if callback is not None:
            callback(0, 2)
            callback(1, 2)
            callback(2, 2)
        return SimpleNamespace(
            video_latents="live-video-latents",
            audio_latents="live-audio-latents",
            num_frames=124,
            width=kwargs["width"],
            height=kwargs["height"],
            fps=24,
            sample_rate=32000,
            transformer_evaluations=2,
            seconds_per_evaluation=1.25,
            total_seconds=2.5,
        )


def _spec(tmp_path: Path, task="t2va") -> H3TransformerSpec:
    root = tmp_path / task
    transformer = root / "transformer"
    transformer.mkdir(parents=True)
    partition = "ref2va" if task == "ref2va" else "fl2va"
    (root / "model_index.json").write_text(
        json.dumps({"_minimax_h3": {"partition": partition, "tasks": [task]}}) + "\n"
    )
    return H3TransformerSpec(
        checkpoint=str(root),
        transformer=str(transformer),
        text_encoder=str(root / "text_encoder"),
        processor=str(root / "processor"),
        tokenizer=str(root / "tokenizer"),
        video_vae=str(root / "video_vae"),
        audio_vae=str(root / "audio_vae"),
        task=task,
    )


def test_transformer_spec_requires_explicit_cross_partition_ref2va(tmp_path: Path):
    spec = _spec(tmp_path, task="fl2va")
    ref_spec = H3TransformerSpec(
        **{
            **spec.__dict__,
            "task": "ref2va",
            "allow_fl2va_weights_for_ref2va": True,
        }
    )

    ref_spec.validate()

    strict_spec = H3TransformerSpec(
        **{
            **ref_spec.__dict__,
            "allow_fl2va_weights_for_ref2va": False,
        }
    )
    with pytest.raises(ValueError, match="does not support task 'ref2va'"):
        strict_spec.validate()


def _conditioning(
    spec: H3TransformerSpec,
    load_vision=False,
    *,
    task="t2va",
    condition_video_rows=None,
    condition_audio_rows=None,
    keyframe_anchors=(),
    references=(),
) -> H3Conditioning:
    root = Path(spec.checkpoint)
    for name in ("text_encoder", "processor", "tokenizer"):
        (root / name).mkdir(exist_ok=True)
    (root / "text_encoder" / "config.json").write_text("{}\n")
    return H3Conditioning(
        embeddings="live-conditioning",
        token_tags="text-tags",
        token_count=3,
        prompt="A test prompt",
        load_vision=load_vision,
        encoder_spec=H3TextEncoderSpec(
            text_encoder=str(root / "text_encoder"),
            processor=str(root / "processor"),
            tokenizer=str(root / "tokenizer"),
            load_vision=load_vision,
        ),
        task=task,
        condition_video_rows=condition_video_rows,
        condition_audio_rows=condition_audio_rows,
        keyframe_anchors=keyframe_anchors,
        references=references,
    )


def test_transformer_cache_samples_and_unloads(tmp_path: Path):
    created = []

    def factory(spec):
        sampler = FakeSampler(spec)
        created.append(sampler)
        return sampler

    progress = []
    cache = H3TransformerCache(factory)
    spec = _spec(tmp_path)
    latents = cache.sample(
        spec,
        _conditioning(spec),
        H3GenerationConfig(steps=3),
        unload_after=True,
        step_callback=lambda completed, total: progress.append((completed, total)),
    )

    assert latents.video == "live-video-latents"
    assert latents.audio == "live-audio-latents"
    assert latents.transformer_evaluations == 2
    assert progress == [(0, 2), (1, 2), (2, 2)]
    assert cache.loaded is False
    assert len(created) == 1


def test_transformer_cache_forwards_dense_continuation(tmp_path: Path):
    created = []

    def factory(spec):
        sampler = FakeSampler(spec)
        created.append(sampler)
        return sampler

    spec = _spec(tmp_path)
    context = H3ContinuationContext(
        video="context-video",
        audio="context-audio",
        context_frames=22,
        width=640,
        height=384,
        fps=24,
        sample_rate=32000,
        transformer_checkpoint=spec.checkpoint,
        transformer_path=spec.transformer,
    )

    H3TransformerCache(factory).sample(
        spec,
        _conditioning(spec),
        H3GenerationConfig(steps=3, width=640, height=384),
        continuation=context,
    )

    kwargs = created[0].calls[0][2]
    assert kwargs["continuation_video_latents"] == "context-video"
    assert kwargs["continuation_audio_latents"] == "context-audio"
    assert kwargs["continuation_frames"] == 22


def test_transformer_cache_forwards_fl2va_timed_rows_with_continuation(tmp_path: Path):
    created = []

    def factory(spec):
        sampler = FakeSampler(spec)
        created.append(sampler)
        return sampler

    spec = _spec(tmp_path, task="fl2va")
    context = H3ContinuationContext(
        video="context-video",
        audio="context-audio",
        context_frames=22,
        width=640,
        height=384,
        fps=24,
        sample_rate=32000,
        transformer_checkpoint=spec.checkpoint,
        transformer_path=spec.transformer,
    )
    conditioning = _conditioning(
        spec,
        load_vision=True,
        task="fl2va",
        condition_video_rows="timed-rows",
        keyframe_anchors=(36, 84),
    )

    H3TransformerCache(factory).sample(
        spec,
        conditioning,
        H3GenerationConfig(steps=3, width=640, height=384),
        continuation=context,
    )

    kwargs = created[0].calls[0][2]
    assert kwargs["condition_video_rows"] == "timed-rows"
    assert kwargs["keyframe_anchors"] == (36, 84)
    assert kwargs["continuation_frames"] == 22


def test_transformer_cache_forwards_ref2va_rows_with_continuation(tmp_path: Path):
    created = []

    def factory(spec):
        sampler = FakeSampler(spec)
        created.append(sampler)
        return sampler

    spec = _spec(tmp_path, task="ref2va")
    context = H3ContinuationContext(
        video="context-video",
        audio="context-audio",
        context_frames=22,
        width=640,
        height=384,
        fps=24,
        sample_rate=32000,
        transformer_checkpoint=spec.checkpoint,
        transformer_path=spec.transformer,
    )
    conditioning = _conditioning(
        spec,
        load_vision=True,
        task="ref2va",
        condition_video_rows="reference-video-rows",
        condition_audio_rows="reference-audio-rows",
        references=("picture-1", "video-2"),
    )

    H3TransformerCache(factory).sample(
        spec,
        conditioning,
        H3GenerationConfig(steps=3, width=640, height=384),
        continuation=context,
    )

    kwargs = created[0].calls[0][2]
    assert kwargs["condition_video_rows"] == "reference-video-rows"
    assert kwargs["condition_audio_rows"] == "reference-audio-rows"
    assert kwargs["references"] == ("picture-1", "video-2")
    assert kwargs["continuation_frames"] == 22


def test_transformer_cache_accepts_text_only_fl2va_after_first_window(tmp_path: Path):
    created = []

    def factory(spec):
        sampler = FakeSampler(spec)
        created.append(sampler)
        return sampler

    spec = _spec(tmp_path, task="fl2va")
    context = H3ContinuationContext(
        video="context-video",
        audio="context-audio",
        context_frames=22,
        width=640,
        height=384,
        fps=24,
        sample_rate=32000,
        transformer_checkpoint=spec.checkpoint,
        transformer_path=spec.transformer,
    )
    conditioning = _conditioning(spec, load_vision=False, task="fl2va")

    H3TransformerCache(factory).sample(
        spec,
        conditioning,
        H3GenerationConfig(steps=3, width=640, height=384),
        continuation=context,
    )

    kwargs = created[0].calls[0][2]
    assert kwargs["condition_video_rows"] is None
    assert kwargs["keyframe_anchors"] == ()
    assert kwargs["continuation_frames"] == 22


def test_transformer_cache_forwards_h3_native_refinement_and_preserved_audio(tmp_path: Path):
    from minimax_h3_mlx.hires_fix import resize_video_latents_bicubic

    created = []

    def factory(spec):
        sampler = FakeSampler(spec)
        created.append(sampler)
        return sampler

    spec = _spec(tmp_path)
    source_config = H3GenerationConfig(steps=8, width=640, height=384)
    source_video = mx.arange(24 * 37 * 24 * 40, dtype=mx.float32).reshape(1, 24, 37, 24, 40)
    source = H3Latents(
        video=source_video,
        audio=mx.zeros((2, 32, 207)),
        num_frames=124,
        width=640,
        height=384,
        fps=24,
        sample_rate=32000,
        transformer_evaluations=7,
        seconds_per_evaluation=1.0,
        total_seconds=7.0,
        transformer_spec=spec,
        generation_config=source_config,
    )

    H3TransformerCache(factory).sample(
        spec,
        _conditioning(spec),
        H3GenerationConfig(steps=5, width=960, height=576),
        refinement_source=source,
        refinement_strength=0.35,
        refinement_resize_method="bicubic",
    )

    kwargs = created[0].calls[0][2]
    assert tuple(kwargs["initial_video_latents"].shape) == (1, 24, 37, 36, 60)
    expected_video = resize_video_latents_bicubic(source_video, 36, 60)
    mx.eval(kwargs["initial_video_latents"], expected_video)
    assert bool(mx.allclose(kwargs["initial_video_latents"], expected_video).item())
    assert kwargs["initial_audio_latents"] is source.audio
    assert kwargs["refinement_strength"] == 0.35
    assert kwargs["preserve_initial_audio"] is True


def test_transformer_cache_accepts_target_only_forecast_with_continuation(tmp_path: Path):
    created = []

    def factory(spec):
        sampler = FakeSampler(spec)
        created.append(sampler)
        return sampler

    spec = _spec(tmp_path)
    context = H3ContinuationContext(
        video="context-video",
        audio="context-audio",
        context_frames=22,
        width=640,
        height=384,
        fps=24,
        sample_rate=32000,
        transformer_checkpoint=spec.checkpoint,
        transformer_path=spec.transformer,
    )
    forecast = H3TrajectoryForecastConfig(
        mode="automatic_speed",
        offline_smoothing_replay=True,
        conditioned_row_policy="target_only",
    )

    H3TransformerCache(factory).sample(
        spec,
        _conditioning(spec),
        H3GenerationConfig(steps=20, width=640, height=384),
        continuation=context,
        trajectory_forecast=forecast,
    )

    kwargs = created[0].calls[0][2]
    assert kwargs["continuation_frames"] == 22
    assert kwargs["trajectory_forecast_config"] is forecast


def test_transformer_cache_reloads_for_continuation_schedule_when_kept_warm(tmp_path: Path):
    created = []

    def factory(spec):
        sampler = FakeSampler(spec)
        created.append(sampler)
        return sampler

    cache = H3TransformerCache(factory)
    spec = _spec(tmp_path)
    config = H3GenerationConfig(steps=20, width=640, height=384)
    cache.sample(spec, _conditioning(spec), config, unload_after=False)
    context = H3ContinuationContext(
        video="context-video",
        audio="context-audio",
        context_frames=22,
        width=640,
        height=384,
        fps=24,
        sample_rate=32000,
        transformer_checkpoint=spec.checkpoint,
        transformer_path=spec.transformer,
    )
    cache.sample(
        spec,
        _conditioning(spec),
        config,
        unload_after=False,
        continuation=context,
    )

    assert len(created) == 2


def test_transformer_cache_rejects_step_cache_with_continuation(tmp_path: Path):
    spec = _spec(tmp_path)
    context = H3ContinuationContext(
        video="context-video",
        audio="context-audio",
        context_frames=22,
        width=640,
        height=384,
        fps=24,
        sample_rate=32000,
        transformer_checkpoint=spec.checkpoint,
        transformer_path=spec.transformer,
    )

    with pytest.raises(ValueError, match="supports dense sampling or Trajectory Forecast"):
        H3TransformerCache(lambda value: FakeSampler(value)).sample(
            spec,
            _conditioning(spec),
            H3GenerationConfig(steps=20, width=640, height=384),
            continuation=context,
            blockcache=H3BlockCacheConfig(),
        )


def test_transformer_cache_forwards_reference_strengths(tmp_path: Path):
    created = []

    def factory(spec):
        sampler = FakeSampler(spec)
        created.append(sampler)
        return sampler

    spec = _spec(tmp_path, task="fl2va")
    conditioning = _conditioning(
        spec,
        load_vision=True,
        task="fl2va",
        condition_video_rows="rows",
        keyframe_anchors=("first",),
    )
    conditioning = H3Conditioning(
        **{
            **conditioning.__dict__,
            "visual_condition_strength": 0.7,
            "audio_condition_strength": 0.9,
        }
    )

    H3TransformerCache(factory).sample(
        spec, conditioning, H3GenerationConfig(steps=3), unload_after=True
    )
    kwargs = created[0].calls[0][2]

    assert kwargs["visual_condition_strength"] == 0.7
    assert kwargs["audio_condition_strength"] == 0.9


def test_transformer_cache_reuses_only_equal_schedule(tmp_path: Path):
    created = []

    def factory(spec):
        sampler = FakeSampler(spec)
        created.append(sampler)
        return sampler

    cache = H3TransformerCache(factory)
    spec = _spec(tmp_path)
    conditioning = _conditioning(spec)
    cache.sample(spec, conditioning, H3GenerationConfig(steps=3), unload_after=False)
    cache.sample(spec, conditioning, H3GenerationConfig(steps=3), unload_after=False)
    cache.sample(spec, conditioning, H3GenerationConfig(steps=4), unload_after=False)

    assert len(created) == 2
    assert cache.loaded is True


def test_low_memory_mode_forces_transformer_unload(tmp_path: Path):
    cache = H3TransformerCache(lambda spec: FakeSampler(spec))
    spec = _spec(tmp_path)
    cache.sample(
        spec,
        _conditioning(spec),
        H3GenerationConfig(steps=3, memory_mode="low_memory_bf16"),
        unload_after=False,
    )
    assert cache.loaded is False


def _lora_stack(tmp_path: Path) -> H3LoRAStack:
    path = tmp_path / "generic.safetensors"
    mx.save_safetensors(
        path,
        {
            "blocks.0.attn.out_proj.lora_A.weight": mx.zeros((2, 4)),
            "blocks.0.attn.out_proj.lora_B.weight": mx.zeros((4, 2)),
        },
        metadata={"base_model": "MiniMax-H3"},
    )
    return H3LoRAStack().append(H3LoRASpec(str(path), profile="standard"))


def _turbo_lora_stack(tmp_path: Path) -> H3LoRAStack:
    stack = _lora_stack(tmp_path)
    return H3LoRAStack((H3LoRASpec(stack.adapters[0].path, profile="turbo"),))


def _staged_turbo_lora_stack(tmp_path: Path) -> H3LoRAStack:
    stack = _lora_stack(tmp_path)
    return H3LoRAStack(
        (
            H3LoRASpec(
                stack.adapters[0].path,
                profile="turbo",
                start_after_evaluations=2,
            ),
        )
    )


def test_transformer_cache_reloads_when_lora_stack_changes(tmp_path: Path, monkeypatch):
    created = []
    applied = []

    def factory(spec):
        sampler = FakeSampler(spec)
        created.append(sampler)
        return sampler

    monkeypatch.setattr(
        "minimax_h3_mlx.lora.apply_lora_stack",
        lambda dit, requests: applied.append((dit, requests)) or (),
    )
    cache = H3TransformerCache(factory)
    spec = _spec(tmp_path)
    conditioning = _conditioning(spec)
    config = H3GenerationConfig(steps=5)
    cache.sample(spec, conditioning, config, unload_after=False)
    stack = _lora_stack(tmp_path)
    cache.sample(spec, conditioning, config, unload_after=False, loras=stack)
    cache.sample(spec, conditioning, config, unload_after=False, loras=stack)

    assert len(created) == 2
    assert len(applied) == 1


def test_lora_application_failure_releases_transformer(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        "minimax_h3_mlx.lora.apply_lora_stack",
        lambda dit, requests: (_ for _ in ()).throw(RuntimeError("synthetic LoRA failure")),
    )
    cache = H3TransformerCache(lambda spec: FakeSampler(spec))
    spec = _spec(tmp_path)

    with pytest.raises(RuntimeError, match="synthetic LoRA failure"):
        cache.sample(
            spec,
            _conditioning(spec),
            H3GenerationConfig(steps=5),
            unload_after=False,
            loras=_lora_stack(tmp_path),
        )

    assert not cache.loaded


def test_turbo_blockcache_requires_explicit_experimental_opt_in(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("minimax_h3_mlx.lora.apply_lora_stack", lambda dit, requests: ())
    cache = H3TransformerCache(lambda spec: FakeSampler(spec))
    spec = _spec(tmp_path)
    conditioning = _conditioning(spec)
    config = H3GenerationConfig(steps=5)
    turbo = _turbo_lora_stack(tmp_path)

    with pytest.raises(ValueError, match="explicit experimental opt-in"):
        cache.sample(spec, conditioning, config, loras=turbo, blockcache=H3BlockCacheConfig())

    allowed = H3BlockCacheConfig(allow_turbo_experimental=True)
    result = cache.sample(spec, conditioning, config, loras=turbo, blockcache=allowed)

    assert result.transformer_evaluations == 2
    assert cache.loaded is False


def test_turbo_trajectory_forecast_is_supported_and_exclusive(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("minimax_h3_mlx.lora.apply_lora_stack", lambda dit, requests: ())
    cache = H3TransformerCache(lambda spec: FakeSampler(spec))
    spec = _spec(tmp_path)
    conditioning = _conditioning(spec)
    config = H3GenerationConfig(steps=5)
    forecast = H3TrajectoryForecastConfig()

    result = cache.sample(
        spec,
        conditioning,
        config,
        loras=_turbo_lora_stack(tmp_path),
        trajectory_forecast=forecast,
    )
    assert result.transformer_evaluations == 2

    with pytest.raises(ValueError, match="mutually exclusive"):
        cache.sample(
            spec,
            conditioning,
            config,
            trajectory_forecast=forecast,
            blockcache=H3BlockCacheConfig(),
        )


def test_staged_turbo_requires_dense_evaluations(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("minimax_h3_mlx.lora.apply_lora_stack", lambda dit, requests: ())
    cache = H3TransformerCache(lambda spec: FakeSampler(spec))
    spec = _spec(tmp_path)
    conditioning = _conditioning(spec)
    config = H3GenerationConfig(steps=7)
    staged = _staged_turbo_lora_stack(tmp_path)

    with pytest.raises(ValueError, match="requires dense transformer evaluations"):
        cache.sample(
            spec,
            conditioning,
            config,
            loras=staged,
            trajectory_forecast=H3TrajectoryForecastConfig(),
        )

    result = cache.sample(spec, conditioning, config, loras=staged)
    assert result.transformer_evaluations == 2


def test_transformer_failure_releases_sampler(tmp_path: Path):
    cache = H3TransformerCache(lambda spec: FakeSampler(spec, fail=True))

    spec = _spec(tmp_path)
    with pytest.raises(RuntimeError, match="synthetic sampling failure"):
        cache.sample(
            spec,
            _conditioning(spec),
            H3GenerationConfig(steps=3),
            unload_after=False,
        )

    assert cache.loaded is False


def test_transformer_sampler_rejects_non_text_conditioning(tmp_path: Path):
    called = False

    def factory(spec):
        nonlocal called
        called = True
        return FakeSampler(spec)

    cache = H3TransformerCache(factory)
    spec = _spec(tmp_path)
    with pytest.raises(ValueError, match="text-only conditioning"):
        cache.sample(
            spec,
            _conditioning(spec, load_vision=True),
            H3GenerationConfig(steps=3),
        )

    assert called is False


def test_transformer_sampler_accepts_prepared_ref2va_conditioning(tmp_path: Path):
    created = []

    def factory(spec):
        sampler = FakeSampler(spec)
        created.append(sampler)
        return sampler

    cache = H3TransformerCache(factory)
    spec = _spec(tmp_path, task="ref2va")
    reference = SimpleNamespace(kind="image")
    conditioning = _conditioning(
        spec,
        load_vision=True,
        task="ref2va",
        condition_video_rows="reference-video-rows",
        condition_audio_rows="reference-audio-rows",
        references=(reference,),
    )

    result = cache.sample(spec, conditioning, H3GenerationConfig(steps=3))

    assert result.transformer_evaluations == 2
    kwargs = created[0].calls[0][2]
    assert kwargs["condition_video_rows"] == "reference-video-rows"
    assert kwargs["condition_audio_rows"] == "reference-audio-rows"
    assert kwargs["references"] == (reference,)


def test_transformer_sampler_accepts_ref2va_trajectory_forecast(tmp_path: Path):
    created = []

    def factory(spec):
        sampler = FakeSampler(spec)
        created.append(sampler)
        return sampler

    cache = H3TransformerCache(factory)
    spec = _spec(tmp_path, task="ref2va")
    reference = SimpleNamespace(kind="image")
    conditioning = _conditioning(
        spec,
        load_vision=True,
        task="ref2va",
        condition_video_rows="reference-video-rows",
        references=(reference,),
    )
    trajectory = H3TrajectoryForecastConfig(
        mode="automatic_speed",
        offline_smoothing_replay=True,
    )

    cache.sample(
        spec,
        conditioning,
        H3GenerationConfig(steps=3),
        trajectory_forecast=trajectory,
    )

    assert created[0].calls[0][2]["trajectory_forecast_config"] is trajectory


def test_transformer_sampler_rejects_ref2va_blockcache_until_validated(tmp_path: Path):
    cache = H3TransformerCache(lambda spec: FakeSampler(spec))
    spec = _spec(tmp_path, task="ref2va")
    conditioning = _conditioning(
        spec,
        load_vision=True,
        task="ref2va",
        condition_video_rows="reference-video-rows",
        references=(SimpleNamespace(kind="image"),),
    )

    with pytest.raises(ValueError, match="supports Trajectory Forecast only"):
        cache.sample(
            spec,
            conditioning,
            H3GenerationConfig(steps=3),
            blockcache=H3BlockCacheConfig(),
        )


def test_transformer_sampler_accepts_prepared_fl2va_conditioning(tmp_path: Path):
    created = []

    def factory(spec):
        sampler = FakeSampler(spec)
        created.append(sampler)
        return sampler

    cache = H3TransformerCache(factory)
    spec = _spec(tmp_path, task="fl2va")
    conditioning = _conditioning(
        spec,
        load_vision=True,
        task="fl2va",
        condition_video_rows="encoded-keyframe-rows",
        keyframe_anchors=("first", "last"),
    )

    result = cache.sample(spec, conditioning, H3GenerationConfig(steps=3))

    assert result.transformer_evaluations == 2
    assert created[0].calls[0][2]["condition_video_rows"] == "encoded-keyframe-rows"
    assert created[0].calls[0][2]["keyframe_anchors"] == ("first", "last")


def test_fl2va_baseline_rejects_acceleration_until_validated(tmp_path: Path):
    cache = H3TransformerCache(lambda spec: FakeSampler(spec))
    spec = _spec(tmp_path, task="fl2va")
    conditioning = _conditioning(
        spec,
        load_vision=True,
        task="fl2va",
        condition_video_rows="encoded-keyframe-rows",
        keyframe_anchors=("first",),
    )

    with pytest.raises(ValueError, match="does not support cache or trajectory"):
        cache.sample(
            spec,
            conditioning,
            H3GenerationConfig(steps=3),
            blockcache=H3BlockCacheConfig(),
        )


def test_transformer_sampler_rejects_conditioning_from_other_components(tmp_path: Path):
    spec = _spec(tmp_path, task="t2va")
    other_root = tmp_path / "other"
    other_root.mkdir()
    other_spec = H3TransformerSpec(
        checkpoint=str(other_root),
        transformer=str(other_root),
        text_encoder=str(other_root / "text_encoder"),
        processor=str(other_root / "processor"),
        tokenizer=str(other_root / "tokenizer"),
        video_vae=str(other_root / "video_vae"),
        audio_vae=str(other_root / "audio_vae"),
        task="t2va",
    )
    conditioning = _conditioning(other_spec)
    cache = H3TransformerCache(lambda value: FakeSampler(value))

    with pytest.raises(ValueError, match="different Qwen3-VL component specification"):
        cache.sample(spec, conditioning, H3GenerationConfig(steps=3))
