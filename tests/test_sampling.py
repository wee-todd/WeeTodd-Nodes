import json
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import pytest

from minimax_h3_mlx.blockcache import H3BlockCacheConfig
from minimax_h3_mlx.trajectory_forecast import H3TrajectoryForecastConfig
from wee_todd_nodes.conditioning import H3Conditioning, H3TextEncoderSpec
from wee_todd_nodes.lora import H3LoRASpec, H3LoRAStack
from wee_todd_nodes.runtime import H3GenerationConfig
from wee_todd_nodes.sampling import H3TransformerCache, H3TransformerSpec


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
    (root / "model_index.json").write_text(
        json.dumps({"_minimax_h3": {"partition": "fl2va", "tasks": [task]}}) + "\n"
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
