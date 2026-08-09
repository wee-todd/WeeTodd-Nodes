"""The MiniMax-H3 text/keyframe -> video+audio pipeline in MLX.

Modified by WeeTodd Nodes to expose host-neutral progress and cancellation callbacks.

One packed sequence carries text, keyframe conditioning, audio and video rows at once, and a single
transformer forward per step predicts the velocity of every row — video and audio are denoised
*jointly*, on two schedules with different sigma shifts (12.0 and 3.0). The checkpoint is
CFG-distilled, so there is no unconditional pass and no guidance scale.

Conditioning rows are re-imposed by construction rather than by masking: only the generated rows are
ever written back, so keyframe anchors survive the whole loop untouched.

The AdaLN modulation cache is built once over the union of every timestep the run will present, and
the 13B of `adaln_proj` is then dropped — see :mod:`minimax_h3_mlx.adaln`.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import numpy as np

from .adaln import ModulationCache, drop_adaln_weights
from .config import TAG_TEXT, PipelineConfig
from .packing import (
    AUDIO_CHANNELS,
    FPS,
    KEYFRAME_NOISE_AUG,
    PIXEL_MEAN,
    PIXEL_STD,
    align_num_frames,
    audio_latent_num_frames,
    build_packed_sequence,
    build_row_timesteps,
    patchify_video_latents,
    resolve_canvas_size,
    unpack_audio_tokens,
    unpatchify_video_tokens,
    video_latent_num_frames,
)
from .scheduler import MiniMaxH3Scheduler


def encode_keyframe_rows(
    video_vae,
    images: list,
    height: int,
    width: int,
    patch_size: tuple[int, int, int],
) -> mx.array:
    """Encode ``fl2va`` keyframes into packed conditioning rows.

    Module-level so a staged runner — which only ever holds one large model at a time and therefore
    cannot construct a whole :class:`MiniMaxH3Pipeline` — can encode a first frame with just the
    video VAE resident.

    Keyframes are single frames, so they go through the video VAE's **spatial** encoder only —
    none of its 17-frame temporal chunking applies. Two details of the reference are load-bearing
    and easy to miss:

    * the posterior is **sampled**, not taken at its mode, under a generator seeded with 42
      independently of the request seed;
    * the sampled latent is **rounded through float16** before normalization, which is about 11
      bits of every conditioning latent — the released model's conditioning cannot be reproduced
      without it.

    MLX's RNG differs from torch's, so the seed-42 draw is not bit-identical to the reference's;
    the distribution and every other step are.
    """
    from .packing import KEYFRAME_ENCODE_SEED, prepare_keyframe_image

    cfg = video_vae.config
    latents_mean = mx.array(np.array(cfg.latents_mean, np.float32)).reshape(1, -1, 1, 1, 1)
    latents_std = mx.array(np.array(cfg.latents_std, np.float32)).reshape(1, -1, 1, 1, 1)
    pixel_mean = np.array(PIXEL_MEAN, np.float32).reshape(1, 3, 1, 1, 1)
    pixel_std = np.array(PIXEL_STD, np.float32).reshape(1, 3, 1, 1, 1)

    mx.random.seed(KEYFRAME_ENCODE_SEED)
    rows = []
    for index, image in enumerate(images):
        prepared = prepare_keyframe_image(image, height, width, stretch=index == 0)
        pixels = np.asarray(prepared, dtype=np.float32).transpose(2, 0, 1)[None, :, None]
        pixels = (pixels / 255.0 - pixel_mean) / pixel_std

        # (1, 3, 1, H, W) -> channels-last for the spatial encoder.
        moments = video_vae._encode_clip(mx.array(pixels).transpose(0, 2, 3, 4, 1))
        channels = cfg.latent_channels
        mean, logvar = moments[..., :channels], moments[..., channels:]
        logvar = mx.clip(logvar, -30.0, 20.0)
        std = mx.exp(0.5 * logvar)
        latent = mean + std * mx.random.normal(mean.shape)
        # -> (1, C, 1, H', W'), then the float16 round trip the reference relies on.
        latent = latent.transpose(0, 4, 1, 2, 3).astype(mx.float16).astype(mx.float32)
        normalized = (latent - latents_mean) / latents_std
        rows.append(patchify_video_latents(normalized, patch_size))
    return mx.concatenate(rows)


@dataclass
class GenerationResult:
    video: np.ndarray  # (frames, height, width, 3) uint8
    audio: np.ndarray  # (2, samples) float32, in [-1, 1]
    sample_rate: int
    fps: int = FPS
    seconds_per_step: float = 0.0
    total_seconds: float = 0.0


@dataclass
class LatentResult:
    """Synchronized normalized H3 latents produced before either VAE decode."""

    video_latents: mx.array  # (1, 24, latent_frames, height/16, width/16)
    audio_latents: mx.array  # (2, 32, audio_latent_frames)
    num_frames: int
    width: int
    height: int
    fps: int = FPS
    sample_rate: int = 32000
    transformer_evaluations: int = 0
    easycache_skipped_steps: int = 0
    easycache_resolved_threshold: float | None = None
    blockcache_hits: int = 0
    blockcache_resolved_threshold: float | None = None
    blockcache_cache_bytes: int = 0
    blockcache_segment_hits: tuple[int, ...] = ()
    blockcache_segment_thresholds: tuple[float | None, ...] = ()
    blockcache_executed_blocks: int = 0
    blockcache_skipped_blocks: int = 0
    trajectory_forecasts: int = 0
    trajectory_bootstrap_forecasts: int = 0
    trajectory_fallbacks: int = 0
    trajectory_history_bytes: int = 0
    trajectory_offline_replay: bool = False
    trajectory_replay_steps: int = 0
    trajectory_replay_anchor_steps: int = 0
    trajectory_replay_smoothed_steps: int = 0
    trajectory_capture_seconds: float = 0.0
    trajectory_replay_seconds: float = 0.0
    trajectory_replay_fallback_reason: str | None = None
    seconds_per_evaluation: float = 0.0
    total_seconds: float = 0.0


class MiniMaxH3Pipeline:
    """Joint video + audio generation."""

    def __init__(
        self,
        dit,
        text_encoder,
        video_vae,
        audio_vae,
        config: PipelineConfig | None = None,
    ):
        self.dit = dit
        self.text_encoder = text_encoder
        self.video_vae = video_vae
        self.audio_vae = audio_vae
        self.config = config or PipelineConfig()
        self._cache: ModulationCache | None = None
        self._cache_timesteps: tuple[float, ...] | None = None

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_dir: str | Path,
        transformer_dir: str | Path | None = None,
        dtype: mx.Dtype = mx.bfloat16,
        load_vision: bool = False,
        verbose: bool = True,
    ) -> MiniMaxH3Pipeline:
        """Load a released ``FL2VA/`` (or ``Ref2VA/``) directory.

        Args:
            checkpoint_dir: the upstream release, which supplies the VAEs and the text encoder.
            transformer_dir: load the DiT from here instead of ``<checkpoint_dir>/transformer``.
                This is how a published quant is used: the quantized repository holds only the
                transformer, and everything else still comes from upstream. ``load_dit`` picks up
                the recorded recipe from its ``quant_config.json`` automatically.
        """
        from .load import load_audio_vae, load_dit, load_video_vae
        from .text_encoder import MiniMaxH3TextEncoder

        root = Path(checkpoint_dir)
        dit_path = Path(transformer_dir) if transformer_dir else root / "transformer"
        config = PipelineConfig.from_model_index(root / "model_index.json")

        def step(label, fn):
            started = time.perf_counter()
            out = fn()
            if verbose:
                print(f"  {label}: {time.perf_counter() - started:.1f}s")
            return out

        if verbose:
            print(f"loading MiniMax-H3 from {root}")
        text_encoder = step(
            "text encoder",
            lambda: MiniMaxH3TextEncoder(
                root / "text_encoder", dtype=dtype, load_vision=load_vision
            ),
        )
        dit = step(f"transformer ({dit_path.name})", lambda: load_dit(dit_path))
        video_vae = step("video vae", lambda: load_video_vae(root / "video_vae"))
        audio_vae = step("audio vae", lambda: load_audio_vae(root / "audio_vae"))
        return cls(dit, text_encoder, video_vae, audio_vae, config)

    # -- schedule -----------------------------------------------------------------------------

    def _build_schedules(self, num_inference_steps: int):
        video = MiniMaxH3Scheduler(shift=self.config.sigma_shift_video)
        audio = MiniMaxH3Scheduler(shift=self.config.sigma_shift_audio)
        video.set_timesteps(num_inference_steps)
        audio.set_timesteps(num_inference_steps)
        return video, audio

    def _row_timestep_plan(self, layout, video_timesteps, audio_timesteps):
        """Per-step ``(timestep_indices,)`` against one global timestep table.

        The transformer is handed the same table at every step, so a single
        :class:`ModulationCache` covers the whole run. Conditioning video rows sit at
        ``max(t, 0.999)`` and reference audio rows at ``1.0``, matching the reference.
        """
        per_step = []
        for t, at in zip(video_timesteps.tolist(), audio_timesteps.tolist(), strict=True):
            distinct, inverse = build_row_timesteps(
                layout, float(t), float(at), max(float(t), KEYFRAME_NOISE_AUG), 1.0
            )
            per_step.append((np.array(distinct), np.array(inverse)))

        table = sorted({float(v) for distinct, _ in per_step for v in distinct})
        lookup = {v: i for i, v in enumerate(table)}
        plan = []
        for distinct, inverse in per_step:
            remap = np.array([lookup[float(v)] for v in distinct], dtype=np.int32)
            plan.append(mx.array(remap[inverse].astype(np.int32)))
        return mx.array(np.array(table, dtype=np.float32)), plan

    def _ensure_cache(self, timesteps: mx.array, drop_adaln: bool, verbose: bool):
        key = tuple(round(float(v), 9) for v in timesteps.tolist())
        if self._cache is not None and self._cache_timesteps == key:
            return
        started = time.perf_counter()
        from .lora import prepare_lora_timesteps

        prepare_lora_timesteps(self.dit, timesteps)
        self._cache = ModulationCache.build(self.dit, timesteps, dtype=mx.bfloat16)
        self._cache_timesteps = key
        if verbose:
            print(
                f"  adaln cache: {len(key)} timesteps, {self._cache.nbytes() / 1e6:.0f} MB "
                f"in {time.perf_counter() - started:.1f}s"
            )
        if drop_adaln:
            freed = drop_adaln_weights(self.dit)
            mx.eval(self.dit.parameters())
            if verbose:
                print(f"  dropped adaln projections, freeing {freed / 1e9:.1f} GB")

    # -- keyframe conditioning ----------------------------------------------------------------

    def _encode_keyframes(self, images: list, height: int, width: int) -> mx.array:
        return encode_keyframe_rows(
            self.video_vae, images, height, width, self.dit.config.patch_size
        )

    def sample_latents(
        self,
        prompt_embeds: mx.array,
        text_token_tags: np.ndarray,
        *,
        duration_seconds: float = 5.0,
        num_inference_steps: int = 16,
        seed: int = 0,
        height: int = 384,
        width: int = 640,
        drop_adaln: bool = True,
        verbose: bool = True,
        step_callback: Callable[[int, int], None] | None = None,
        easycache_config=None,
        blockcache_config=None,
        trajectory_forecast_config=None,
        diagnostics=None,
    ) -> LatentResult:
        """Sample synchronized text-only video and audio latents without loading either VAE."""
        run_started = time.perf_counter()
        if not 5.0 <= duration_seconds <= 15.0:
            raise ValueError("`duration_seconds` must be between 5 and 15 seconds.")
        if num_inference_steps < 2:
            raise ValueError("`num_inference_steps` must be at least 2.")
        if height < 32 or width < 32 or height % 32 or width % 32:
            raise ValueError(
                f"`height` and `width` must be positive multiples of 32, got {height}x{width}."
            )

        tags = np.asarray(text_token_tags, dtype=np.int32)
        if tags.ndim != 1 or tags.size == 0:
            raise ValueError(
                "`text_token_tags` must contain one modality tag per conditioning row."
            )
        if np.any(tags != TAG_TEXT):
            raise ValueError("Text-only sampling accepts text modality tags only.")
        if prompt_embeds.ndim != 3 or prompt_embeds.shape[0] != 1:
            raise ValueError("`prompt_embeds` must have shape (1, tokens, text_dim).")
        if prompt_embeds.shape[1] != tags.size:
            raise ValueError(
                "Prompt embedding rows and text modality tags must have equal lengths."
            )
        if prompt_embeds.shape[2] != self.dit.config.text_dim:
            raise ValueError(
                f"Prompt embedding width must be {self.dit.config.text_dim}, "
                f"got {prompt_embeds.shape[2]}."
            )

        num_frames = align_num_frames(int(round(duration_seconds * FPS)))
        num_latent_frames = video_latent_num_frames(num_frames)
        latent_height, latent_width = height // 16, width // 16
        num_audio_latents = audio_latent_num_frames(num_frames)
        patch_size = self.dit.config.patch_size
        layout = build_packed_sequence(
            tags,
            num_latent_frames,
            latent_height,
            latent_width,
            num_audio_latents,
            patch_size,
        )

        mx.random.seed(seed)
        video_latents = mx.random.normal(
            (
                1,
                self.dit.config.latents_dim,
                num_latent_frames,
                latent_height,
                latent_width,
            )
        ).astype(mx.float32)
        video_rows = patchify_video_latents(video_latents, patch_size)
        audio_rows = mx.random.normal(
            (num_audio_latents * AUDIO_CHANNELS, self.dit.config.audio_latents_dim)
        ).astype(mx.float32)

        video_sched, audio_sched = self._build_schedules(num_inference_steps)
        timestep_table, plan = self._row_timestep_plan(
            layout, video_sched.timesteps, audio_sched.timesteps
        )
        self._ensure_cache(timestep_table, drop_adaln, verbose)
        embeds = prompt_embeds.astype(mx.bfloat16)

        accelerators = sum(
            value is not None
            for value in (easycache_config, blockcache_config, trajectory_forecast_config)
        )
        if accelerators > 1:
            raise ValueError(
                "EasyCache, BlockCache, and Trajectory Forecast are mutually exclusive."
            )
        easycache = None
        if easycache_config is not None:
            from .easycache import H3EasyCacheState

            easycache = H3EasyCacheState(easycache_config)
        blockcache = None
        if blockcache_config is not None:
            from .blockcache import H3BlockCacheState, H3HierarchicalBlockCacheState

            state_type = (
                H3HierarchicalBlockCacheState
                if hasattr(blockcache_config, "segments")
                else H3BlockCacheState
            )
            blockcache = state_type(blockcache_config)
        trajectory_forecast = None
        if trajectory_forecast_config is not None:
            from .trajectory_forecast import (
                H3OfflineReplayError,
                H3TrajectoryForecastState,
            )

            trajectory_forecast = H3TrajectoryForecastState(trajectory_forecast_config)

        total_steps = len(video_sched.timesteps)
        offline_replay = bool(
            trajectory_forecast is not None and trajectory_forecast.config.offline_smoothing_replay
        )
        if offline_replay:
            initial_video_rows = video_rows + mx.zeros((), dtype=video_rows.dtype)
            initial_audio_rows = audio_rows + mx.zeros((), dtype=audio_rows.dtype)
            mx.eval(initial_video_rows, initial_audio_rows)
            trajectory_forecast.begin_capture(total_steps)
        else:
            initial_video_rows = video_rows
            initial_audio_rows = audio_rows

        step_times = []
        transformer_times = []
        transformer_evaluations = 0
        trajectory_capture_seconds = 0.0
        trajectory_replay_seconds = 0.0

        def run_pass(start_video_rows, start_audio_rows, progress_offset, progress_total):
            nonlocal transformer_evaluations

            current_video_rows = start_video_rows
            current_audio_rows = start_audio_rows
            for index, timestep in enumerate(video_sched.timesteps.tolist()):
                if step_callback is not None:
                    step_callback(progress_offset + index, progress_total)
                started = time.perf_counter()
                video_input = current_video_rows[None].astype(mx.bfloat16)
                audio_input = current_audio_rows[None].astype(mx.bfloat16)
                reused = (
                    easycache.try_reuse(video_input, audio_input, index, total_steps)
                    if easycache is not None
                    else None
                )
                blockcache_hit = False
                hierarchical_blockcache = blockcache is not None and hasattr(
                    blockcache, "segment_hits"
                )
                replaying = bool(trajectory_forecast is not None and trajectory_forecast.replaying)
                if reused is None:
                    if diagnostics is not None and hasattr(diagnostics, "begin_evaluation"):
                        diagnostics.begin_evaluation(
                            index,
                            timestep=float(timestep),
                            audio_timestep=float(audio_sched.timesteps[index].item()),
                        )
                    video_pred, audio_pred = self.dit(
                        video_input,
                        audio_input,
                        embeds,
                        timestep_table,
                        plan[index],
                        layout.token_tags,
                        layout.position_ids,
                        layout.video_indices,
                        layout.audio_indices,
                        layout.text_indices,
                        modulation_cache=self._cache,
                        blockcache=blockcache,
                        trajectory_forecast=trajectory_forecast,
                        forecast_coordinate=float(timestep),
                        step_index=index,
                        total_steps=total_steps,
                        diagnostics=diagnostics,
                    )
                    actual_transformer = not replaying and (
                        hierarchical_blockcache
                        or (
                            (blockcache is None or not blockcache.last_was_hit)
                            and (
                                trajectory_forecast is None
                                or not trajectory_forecast.last_was_forecast
                            )
                        )
                    )
                    transformer_evaluations += int(actual_transformer)
                    blockcache_hit = blockcache is not None and blockcache.last_was_hit
                    if easycache is not None:
                        easycache.update(video_input, audio_input, video_pred, audio_pred)
                else:
                    video_pred, audio_pred = reused
                current_video_rows = video_sched.step(
                    video_pred[0].astype(mx.float32),
                    float(timestep),
                    current_video_rows,
                )
                current_audio_rows = audio_sched.step(
                    audio_pred[0].astype(mx.float32),
                    float(audio_sched.timesteps[index].item()),
                    current_audio_rows,
                )
                mx.eval(current_video_rows, current_audio_rows)
                elapsed = time.perf_counter() - started
                step_times.append(elapsed)
                forecast_hit = bool(
                    trajectory_forecast is not None and trajectory_forecast.last_was_forecast
                )
                if (
                    not replaying
                    and reused is None
                    and (not blockcache_hit or hierarchical_blockcache)
                    and not forecast_hit
                ):
                    transformer_times.append(elapsed)
                if verbose:
                    completed = progress_offset + index + 1
                    mean = sum(step_times) / len(step_times)
                    eta = mean * (progress_total - completed)
                    action = (
                        "replay-smooth  "
                        if replaying and forecast_hit
                        else "replay-anchor  "
                        if replaying
                        else "forecast  "
                        if forecast_hit
                        else "cached  "
                        if reused is not None or blockcache_hit
                        else ""
                    )
                    print(
                        f"  step {completed}/{progress_total}  {elapsed:.1f}s  "
                        f"{action}eta {eta / 60:.1f} min",
                        flush=True,
                    )
            return current_video_rows, current_audio_rows

        progress_total = total_steps * (2 if offline_replay else 1)
        try:
            capture_started = time.perf_counter()
            capture_video_rows, capture_audio_rows = run_pass(
                video_rows,
                audio_rows,
                0,
                progress_total,
            )
            if offline_replay:
                trajectory_capture_seconds = time.perf_counter() - capture_started
                if trajectory_forecast.complete_capture():
                    try:
                        trajectory_forecast.begin_replay()
                        video_sched.set_timesteps(num_inference_steps)
                        audio_sched.set_timesteps(num_inference_steps)
                        replay_started = time.perf_counter()
                        video_rows, audio_rows = run_pass(
                            initial_video_rows,
                            initial_audio_rows,
                            total_steps,
                            progress_total,
                        )
                        trajectory_replay_seconds = time.perf_counter() - replay_started
                    except H3OfflineReplayError as exc:
                        trajectory_forecast.mark_replay_fallback(str(exc))
                        video_rows, audio_rows = capture_video_rows, capture_audio_rows
                else:
                    trajectory_forecast.mark_replay_fallback(
                        trajectory_forecast.replay_fallback_reason
                        or "Offline capture validation failed."
                    )
                    video_rows, audio_rows = capture_video_rows, capture_audio_rows
            else:
                video_rows, audio_rows = capture_video_rows, capture_audio_rows
            if step_callback is not None:
                step_callback(progress_total, progress_total)

            trajectory_forecasts = (
                trajectory_forecast.forecasts if trajectory_forecast is not None else 0
            )
            trajectory_bootstrap_forecasts = (
                trajectory_forecast.bootstrap_forecasts if trajectory_forecast is not None else 0
            )
            trajectory_fallbacks = (
                trajectory_forecast.fallbacks if trajectory_forecast is not None else 0
            )
            trajectory_history_bytes = (
                trajectory_forecast.history_bytes if trajectory_forecast is not None else 0
            )
            trajectory_replay_steps = (
                trajectory_forecast.replay_steps if trajectory_forecast is not None else 0
            )
            trajectory_replay_anchor_steps = (
                trajectory_forecast.replay_anchor_steps if trajectory_forecast is not None else 0
            )
            trajectory_replay_smoothed_steps = (
                trajectory_forecast.replay_smoothed_steps if trajectory_forecast is not None else 0
            )
            trajectory_replay_fallback_reason = (
                trajectory_forecast.replay_fallback_reason
                if trajectory_forecast is not None
                else None
            )
        finally:
            if trajectory_forecast is not None:
                trajectory_forecast.release()

        if diagnostics is not None:
            diagnostics.write_metadata()

        video_latents = unpatchify_video_tokens(
            video_rows,
            num_latent_frames,
            latent_height,
            latent_width,
            self.dit.config.latents_dim,
            patch_size,
        )
        audio_latents = unpack_audio_tokens(audio_rows, num_audio_latents)
        mx.eval(video_latents, audio_latents)
        return LatentResult(
            video_latents=video_latents,
            audio_latents=audio_latents,
            num_frames=num_frames,
            width=width,
            height=height,
            transformer_evaluations=transformer_evaluations,
            easycache_skipped_steps=easycache.skipped_steps if easycache is not None else 0,
            easycache_resolved_threshold=(
                easycache.resolved_threshold if easycache is not None else None
            ),
            blockcache_hits=blockcache.hits if blockcache is not None else 0,
            blockcache_resolved_threshold=(
                blockcache.resolved_threshold
                if blockcache is not None and not hasattr(blockcache, "segment_hits")
                else None
            ),
            blockcache_cache_bytes=blockcache.cache_bytes if blockcache is not None else 0,
            blockcache_segment_hits=(
                blockcache.segment_hits if hasattr(blockcache, "segment_hits") else ()
            ),
            blockcache_segment_thresholds=(
                blockcache.resolved_threshold if hasattr(blockcache, "segment_hits") else ()
            ),
            blockcache_executed_blocks=(
                blockcache.executed_blocks
                if hasattr(blockcache, "executed_blocks")
                else transformer_evaluations * 50
            ),
            blockcache_skipped_blocks=(
                blockcache.skipped_blocks if hasattr(blockcache, "skipped_blocks") else 0
            ),
            trajectory_forecasts=trajectory_forecasts,
            trajectory_bootstrap_forecasts=trajectory_bootstrap_forecasts,
            trajectory_fallbacks=trajectory_fallbacks,
            trajectory_history_bytes=trajectory_history_bytes,
            trajectory_offline_replay=offline_replay,
            trajectory_replay_steps=trajectory_replay_steps,
            trajectory_replay_anchor_steps=trajectory_replay_anchor_steps,
            trajectory_replay_smoothed_steps=trajectory_replay_smoothed_steps,
            trajectory_capture_seconds=trajectory_capture_seconds,
            trajectory_replay_seconds=trajectory_replay_seconds,
            trajectory_replay_fallback_reason=trajectory_replay_fallback_reason,
            seconds_per_evaluation=(sum(transformer_times) / max(len(transformer_times), 1)),
            total_seconds=time.perf_counter() - run_started,
        )

    # -- generation ---------------------------------------------------------------------------

    def __call__(
        self,
        prompt: str,
        duration_seconds: float = 5.0,
        aspect: tuple[int, int] = (16, 9),
        num_inference_steps: int = 16,
        seed: int = 0,
        images: list | None = None,
        keyframe_anchors: tuple[str, ...] = (),
        height: int | None = None,
        width: int | None = None,
        drop_adaln: bool = True,
        verbose: bool = True,
        step_callback: Callable[[int, int], None] | None = None,
    ) -> GenerationResult:
        """Generate a clip.

        Args:
            duration_seconds: 5 to 15; snapped up to the ``17n + 5`` frame grid the VAE encodes.
            num_inference_steps: the weights are CFG-distilled, so each step is one forward.
            keyframe_anchors: ``"first"`` / ``"last"`` per conditioning keyframe, in packed order.
            height, width: override the canvas ``aspect`` would resolve to. Both must be multiples
                of 32. H3 was released for a 768-pixel short edge only, so anything else is
                off-distribution — useful for exercising the pipeline, not for quality.
            step_callback: called between denoising steps with ``(completed, total)``. The host may
                raise from this callback to cancel without coupling the MLX engine to ComfyUI.
        """
        run_started = time.perf_counter()

        # 1. Text conditioning. Keyframe vision blocks come back tagged as *video* rows.
        prompt_embeds, text_token_tags = self.text_encoder.encode(prompt, images)

        # 2. Geometry.
        if height is None or width is None:
            height, width = resolve_canvas_size(*aspect)
        elif height % 32 or width % 32:
            raise ValueError(f"`height` and `width` must be multiples of 32, got {height}x{width}.")
        num_frames = align_num_frames(int(round(duration_seconds * FPS)))
        num_latent_frames = video_latent_num_frames(num_frames)
        ratio = self.video_vae.config.spatial_compression_ratio
        latent_height, latent_width = height // ratio, width // ratio
        num_audio_latents = audio_latent_num_frames(num_frames)
        patch_size = self.dit.config.patch_size

        layout = build_packed_sequence(
            text_token_tags,
            num_latent_frames,
            latent_height,
            latent_width,
            num_audio_latents,
            patch_size,
            keyframe_anchors,
        )
        if verbose:
            print(
                f"canvas {width}x{height}, {num_frames} frames ({num_latent_frames} latent), "
                f"{num_audio_latents} audio latents"
            )
            print(
                f"packed sequence: {layout.sequence_length:,} rows "
                f"({len(text_token_tags):,} text, {layout.num_condition_video_rows:,} condition)"
            )

        # 3. Keyframe conditioning rows, encoded before any request noise is drawn.
        condition_rows = None
        if images:
            condition_rows = self._encode_keyframes(images, height, width)

        # 4. Initial noise. Draw order matches the reference — the conditioning noise comes off the
        #    request generator first, then video, then audio — so a seed reproduces the same run.
        mx.random.seed(seed)
        if condition_rows is not None:
            condition_noise = mx.random.normal(condition_rows.shape).astype(mx.float32)
            # Anchors are not fully clean: they are noised to t = 0.999 and held there every step.
            condition_rows = MiniMaxH3Scheduler(shift=self.config.sigma_shift_video).scale_noise(
                condition_rows, KEYFRAME_NOISE_AUG, condition_noise
            )

        latents = mx.random.normal(
            (
                1,
                self.video_vae.config.latent_channels,
                num_latent_frames,
                latent_height,
                latent_width,
            )
        ).astype(mx.float32)
        video_rows = patchify_video_latents(latents, patch_size)
        audio_rows = mx.random.normal(
            (num_audio_latents * AUDIO_CHANNELS, self.audio_vae.config.latent_channels)
        ).astype(mx.float32)
        if condition_rows is not None:
            video_rows = mx.concatenate([condition_rows, video_rows])

        # 5. Two schedules over one shared forward.
        video_sched, audio_sched = self._build_schedules(num_inference_steps)
        timestep_table, plan = self._row_timestep_plan(
            layout, video_sched.timesteps, audio_sched.timesteps
        )
        self._ensure_cache(timestep_table, drop_adaln, verbose)

        n_cond_v = layout.num_condition_video_rows
        n_cond_a = layout.num_condition_audio_rows
        embeds = prompt_embeds.astype(mx.bfloat16)

        # 6. Denoise. One forward per step; only generated rows are written back, so the
        #    conditioning anchors survive without any masking.
        step_times = []
        total_steps = len(video_sched.timesteps)
        for i, t in enumerate(video_sched.timesteps.tolist()):
            if step_callback is not None:
                step_callback(i, total_steps)
            started = time.perf_counter()
            video_pred, audio_pred = self.dit(
                video_rows[None].astype(mx.bfloat16),
                audio_rows[None].astype(mx.bfloat16),
                embeds,
                timestep_table,
                plan[i],
                layout.token_tags,
                layout.position_ids,
                layout.video_indices,
                layout.audio_indices,
                layout.text_indices,
                modulation_cache=self._cache,
            )
            # Rebind rather than assign into a slice: the stepped result is a lazy graph reading the
            # very rows it would overwrite, and with conditioning rows present the two halves must
            # stay distinct. Concatenating is unambiguous and costs nothing next to the forward.
            stepped_video = video_sched.step(
                video_pred[0, n_cond_v:].astype(mx.float32), float(t), video_rows[n_cond_v:]
            )
            stepped_audio = audio_sched.step(
                audio_pred[0, n_cond_a:].astype(mx.float32),
                float(audio_sched.timesteps[i].item()),
                audio_rows[n_cond_a:],
            )
            video_rows = (
                mx.concatenate([video_rows[:n_cond_v], stepped_video])
                if n_cond_v
                else stepped_video
            )
            audio_rows = (
                mx.concatenate([audio_rows[:n_cond_a], stepped_audio])
                if n_cond_a
                else stepped_audio
            )
            mx.eval(video_rows, audio_rows)
            step_times.append(time.perf_counter() - started)
            if verbose:
                done = i + 1
                mean = sum(step_times) / len(step_times)
                eta = mean * (len(video_sched.timesteps) - done)
                print(
                    f"  step {done}/{len(video_sched.timesteps)}  "
                    f"{step_times[-1]:.1f}s  eta {eta / 60:.1f} min",
                    flush=True,
                )
        if step_callback is not None:
            step_callback(total_steps, total_steps)

        # 7. Decode both modalities.
        video = self._decode_video(
            video_rows[n_cond_v:], num_latent_frames, latent_height, latent_width
        )
        audio = self._decode_audio(audio_rows[n_cond_a:], num_audio_latents)
        total = time.perf_counter() - run_started
        return GenerationResult(
            video=video,
            audio=audio,
            sample_rate=self.audio_vae.config.sampling_rate,
            seconds_per_step=sum(step_times) / max(len(step_times), 1),
            total_seconds=total,
        )

    # -- decoding -----------------------------------------------------------------------------

    def _decode_video(self, rows, num_latent_frames, latent_height, latent_width) -> np.ndarray:
        cfg = self.video_vae.config
        latents = unpatchify_video_tokens(
            rows,
            num_latent_frames,
            latent_height,
            latent_width,
            cfg.latent_channels,
            self.dit.config.patch_size,
        )
        mean = mx.array(np.array(cfg.latents_mean, np.float32)).reshape(1, -1, 1, 1, 1)
        std = mx.array(np.array(cfg.latents_std, np.float32)).reshape(1, -1, 1, 1, 1)
        latents = latents * std + mean

        frames = np.array(self.video_vae.decode(latents.astype(mx.float32)))
        # The VAE decodes into ImageNet-normalized RGB over a [0, 1] base range.
        pixel_mean = np.array(PIXEL_MEAN, np.float32).reshape(1, 3, 1, 1, 1)
        pixel_std = np.array(PIXEL_STD, np.float32).reshape(1, 3, 1, 1, 1)
        frames = frames * pixel_std + pixel_mean
        frames = np.clip(frames, 0.0, 1.0)[0].transpose(1, 2, 3, 0)  # -> (F, H, W, 3)
        return (frames * 255.0 + 0.5).astype(np.uint8)

    def _decode_audio(self, rows, num_audio_latents) -> np.ndarray:
        cfg = self.audio_vae.config
        latents = unpack_audio_tokens(rows, num_audio_latents)
        mean = mx.array(np.array(cfg.latents_mean, np.float32)).reshape(1, -1, 1)
        std = mx.array(np.array(cfg.latents_std, np.float32)).reshape(1, -1, 1)
        latents = latents * std + mean
        waveform = np.array(self.audio_vae.decode(latents.astype(mx.float32)))
        return waveform[:, 0, :].astype(np.float32)  # (2, samples), one row per stereo channel
