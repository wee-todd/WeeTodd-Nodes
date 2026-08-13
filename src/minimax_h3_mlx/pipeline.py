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

import hashlib
import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import numpy as np

from .adaln import ModulationCache, drop_adaln_weights
from .config import TAG_TEXT, TAG_VIDEO, PipelineConfig
from .packing import (
    AUDIO_CHANNELS,
    FPS,
    KEYFRAME_NOISE_AUG,
    MAX_DURATION,
    MIN_DURATION,
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
from .scheduler import MiniMaxH3ResMultistepScheduler, MiniMaxH3Scheduler


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
    easycache_reuse_strategy: str | None = None
    easycache_cache_bytes: int = 0
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
    trajectory_conditioned_row_policy: str | None = None
    trajectory_excluded_video_rows: int = 0
    trajectory_excluded_audio_rows: int = 0
    refinement_strength: float | None = None
    refinement_audio_preserved: bool = False
    prepared_state_cache_hits: int = 0
    prepared_state_cache_builds: int = 0
    prepared_state_cache_bytes: int = 0
    prepared_state_build_seconds: float = 0.0
    prepared_state_key: str | None = None
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
        self._cache_hits = 0
        self._cache_builds = 0
        self._cache_build_seconds = 0.0
        self._cache_key_digest: str | None = None

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

    def _build_schedules(self, num_inference_steps: int, sampling_method: str = "euler"):
        if sampling_method not in {"euler", "res_multistep"}:
            raise ValueError("`sampling_method` must be 'euler' or 'res_multistep'.")
        scheduler_type = (
            MiniMaxH3ResMultistepScheduler
            if sampling_method == "res_multistep"
            else MiniMaxH3Scheduler
        )
        video = scheduler_type(shift=self.config.sigma_shift_video)
        audio = scheduler_type(shift=self.config.sigma_shift_audio)
        video.set_timesteps(num_inference_steps)
        audio.set_timesteps(num_inference_steps)
        return video, audio

    def _row_timestep_plan(
        self,
        layout,
        video_timesteps,
        audio_timesteps,
        visual_condition_strength: float = KEYFRAME_NOISE_AUG,
        audio_condition_strength: float = 1.0,
    ):
        """Per-step ``(timestep_indices,)`` against one global timestep table.

        The transformer is handed the same table at every step, so a single
        :class:`ModulationCache` covers the whole run. Conditioning video rows sit at
        ``max(t, visual strength)`` and reference audio rows at
        ``max(audio t, audio strength)``. Defaults preserve the released behavior.
        """
        per_step = []
        for t, at in zip(video_timesteps.tolist(), audio_timesteps.tolist(), strict=True):
            distinct, inverse = build_row_timesteps(
                layout,
                float(t),
                float(at),
                max(float(t), visual_condition_strength),
                max(float(at), audio_condition_strength),
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
            self._cache_hits += 1
            return
        started = time.perf_counter()
        from .lora import prepare_lora_timesteps

        prepare_lora_timesteps(self.dit, timesteps)
        self._cache = ModulationCache.build(self.dit, timesteps, dtype=mx.bfloat16)
        self._cache_timesteps = key
        self._cache_key_digest = hashlib.sha256(
            np.asarray(key, dtype=np.float64).tobytes()
        ).hexdigest()[:16]
        self._cache_builds += 1
        self._cache_build_seconds += time.perf_counter() - started
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
        latent_preview_callback: Callable[[int, int, mx.array], None] | None = None,
        easycache_config=None,
        blockcache_config=None,
        trajectory_forecast_config=None,
        diagnostics=None,
        condition_video_rows: mx.array | None = None,
        condition_audio_rows: mx.array | None = None,
        continuation_video_latents: mx.array | None = None,
        continuation_audio_latents: mx.array | None = None,
        continuation_frames: int = 0,
        keyframe_anchors: tuple[str | int, ...] = (),
        references: tuple | list = (),
        sampling_method: str = "euler",
        visual_condition_strength: float = KEYFRAME_NOISE_AUG,
        audio_condition_strength: float = 1.0,
        terminal_target_only: bool = False,
        initial_video_latents: mx.array | None = None,
        initial_audio_latents: mx.array | None = None,
        refinement_strength: float = 1.0,
        preserve_initial_audio: bool = False,
    ) -> LatentResult:
        """Sample synchronized T2VA, FL2VA, or Ref2VA latents without loading a VAE."""
        run_started = time.perf_counter()
        if not MIN_DURATION <= duration_seconds <= MAX_DURATION:
            raise ValueError(
                f"`duration_seconds` must be between {MIN_DURATION:g} and "
                f"{MAX_DURATION:g} seconds. Durations below 5 seconds are experimental."
            )
        if num_inference_steps < 2:
            raise ValueError("`num_inference_steps` must be at least 2.")
        if height < 32 or width < 32 or height % 32 or width % 32:
            raise ValueError(
                f"`height` and `width` must be positive multiples of 32, got {height}x{width}."
            )
        for name, value in (
            ("visual_condition_strength", visual_condition_strength),
            ("audio_condition_strength", audio_condition_strength),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"`{name}` must be between 0 and 1, got {value}.")
        has_initial_latents = initial_video_latents is not None or initial_audio_latents is not None
        if has_initial_latents and (
            initial_video_latents is None or initial_audio_latents is None
        ):
            raise ValueError("H3 refinement requires both initial video and audio latents.")
        if has_initial_latents and not 0.0 < refinement_strength <= 1.0:
            raise ValueError("`refinement_strength` must be greater than 0 and no more than 1.")
        if preserve_initial_audio and not has_initial_latents:
            raise ValueError("Preserving initial audio requires H3 refinement latents.")

        tags = np.asarray(text_token_tags, dtype=np.int32)
        if tags.ndim != 1 or tags.size == 0:
            raise ValueError(
                "`text_token_tags` must contain one modality tag per conditioning row."
            )
        has_visual_conditioning = condition_video_rows is not None
        allowed_tags = (
            {int(TAG_VIDEO), int(TAG_TEXT)} if has_visual_conditioning else {int(TAG_TEXT)}
        )
        if any(int(tag) not in allowed_tags for tag in tags):
            raise ValueError("H3 conditioning contains unsupported modality tags.")
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
        has_continuation = continuation_frames > 0
        if has_initial_latents and has_continuation:
            raise ValueError("H3 refinement cannot be combined with motion continuation.")
        expected_initial_video_shape = (
            1,
            self.dit.config.latents_dim,
            num_latent_frames,
            latent_height,
            latent_width,
        )
        expected_initial_audio_shape = (
            AUDIO_CHANNELS,
            self.dit.config.audio_latents_dim,
            num_audio_latents,
        )
        if initial_video_latents is not None and tuple(initial_video_latents.shape) != (
            expected_initial_video_shape
        ):
            raise ValueError(
                "Initial H3 video latent shape does not match the refinement canvas: "
                f"expected {expected_initial_video_shape}, got "
                f"{tuple(initial_video_latents.shape)}."
            )
        if initial_audio_latents is not None and tuple(initial_audio_latents.shape) != (
            expected_initial_audio_shape
        ):
            raise ValueError(
                "Initial H3 audio latent shape does not match the refinement duration: "
                f"expected {expected_initial_audio_shape}, got "
                f"{tuple(initial_audio_latents.shape)}."
            )
        if has_continuation:
            if continuation_video_latents is None or continuation_audio_latents is None:
                raise ValueError("Continuation requires both video and audio latent streams.")
            expected_video_frames = video_latent_num_frames(continuation_frames)
            expected_audio_frames = audio_latent_num_frames(continuation_frames)
            expected_video_shape = (
                1,
                self.dit.config.latents_dim,
                expected_video_frames,
                latent_height,
                latent_width,
            )
            if tuple(continuation_video_latents.shape) != expected_video_shape:
                raise ValueError(
                    "Continuation video latent shape does not match the target canvas and context: "
                    f"expected {expected_video_shape}, got "
                    f"{tuple(continuation_video_latents.shape)}."
                )
            expected_audio_shape = (
                AUDIO_CHANNELS,
                self.dit.config.audio_latents_dim,
                expected_audio_frames,
            )
            if tuple(continuation_audio_latents.shape) != expected_audio_shape:
                raise ValueError(
                    "Continuation audio latent shape does not match the context: "
                    f"expected {expected_audio_shape}, got "
                    f"{tuple(continuation_audio_latents.shape)}."
                )
        elif continuation_video_latents is not None or continuation_audio_latents is not None:
            raise ValueError(
                "Continuation latent streams require a positive continuation frame count."
            )
        layout_started = time.perf_counter()
        if references:
            if keyframe_anchors:
                raise ValueError("Ref2VA references cannot be combined with FL2VA keyframes.")
            if condition_video_rows is None:
                raise ValueError("Ref2VA requires encoded image or video reference rows.")
            from .ref2va import validate_reference_set

            validate_reference_set(references, patch_size)
            if condition_video_rows.ndim != 2 or int(condition_video_rows.shape[1]) != (
                self.dit.config.video_patch_dim
            ):
                raise ValueError("Ref2VA video rows must have shape (rows, video_patch_dim).")
            if condition_audio_rows is not None and (
                condition_audio_rows.ndim != 2
                or int(condition_audio_rows.shape[1]) != self.dit.config.audio_latents_dim
            ):
                raise ValueError("Ref2VA audio rows must have shape (rows, audio_latents_dim).")
            expected_video_rows = sum(reference.video_rows(patch_size) for reference in references)
            expected_audio_rows = sum(reference.audio_rows for reference in references)
            if int(condition_video_rows.shape[0]) != expected_video_rows:
                raise ValueError(
                    "Ref2VA encoded video row count does not match the prepared references."
                )
            actual_audio_rows = (
                0 if condition_audio_rows is None else int(condition_audio_rows.shape[0])
            )
            if actual_audio_rows != expected_audio_rows:
                raise ValueError(
                    "Ref2VA encoded audio row count does not match the prepared references."
                )
        elif condition_video_rows is None:
            if keyframe_anchors:
                raise ValueError("FL2VA keyframe anchors require encoded condition video rows.")
            if condition_audio_rows is not None:
                raise ValueError("Reference audio rows require a Ref2VA reference set.")
        else:
            if not keyframe_anchors or len(keyframe_anchors) > 8:
                raise ValueError("FL2VA requires between one and eight keyframe anchors.")
            for anchor in keyframe_anchors:
                if anchor in {"first", "last"}:
                    continue
                if not isinstance(anchor, int) or isinstance(anchor, bool):
                    raise ValueError(
                        "FL2VA keyframe anchors must be 'first', 'last', or zero-based "
                        "pixel-frame indices."
                    )
                if not 0 <= anchor < num_frames:
                    raise ValueError(
                        f"FL2VA timed keyframe {anchor} is outside 0..{num_frames - 1}."
                    )
            if condition_video_rows.ndim != 2:
                raise ValueError(
                    "FL2VA condition video rows must have shape (rows, video_patch_dim)."
                )
            if int(condition_video_rows.shape[1]) != self.dit.config.video_patch_dim:
                raise ValueError(
                    f"FL2VA condition row width must be {self.dit.config.video_patch_dim}."
                )
            _, patch_height, patch_width = patch_size
            rows_per_frame = (latent_height // patch_height) * (latent_width // patch_width)
            expected_rows = len(keyframe_anchors) * rows_per_frame
            if int(condition_video_rows.shape[0]) != expected_rows:
                raise ValueError(
                    f"FL2VA encoded {int(condition_video_rows.shape[0])} condition rows; "
                    f"the target canvas and anchors require {expected_rows}."
                )
        if references:
            from .ref2va import build_ref2va_packed_sequence

            layout = build_ref2va_packed_sequence(
                tags,
                references,
                num_latent_frames,
                latent_height,
                latent_width,
                num_audio_latents,
                patch_size,
                continuation_video_frames=(
                    video_latent_num_frames(continuation_frames) if has_continuation else 0
                ),
                continuation_audio_latents=(
                    audio_latent_num_frames(continuation_frames) if has_continuation else 0
                ),
            )
        else:
            layout = build_packed_sequence(
                tags,
                num_latent_frames,
                latent_height,
                latent_width,
                num_audio_latents,
                patch_size,
                keyframe_anchors,
                continuation_video_frames=(
                    video_latent_num_frames(continuation_frames) if has_continuation else 0
                ),
                continuation_audio_latents=(
                    audio_latent_num_frames(continuation_frames) if has_continuation else 0
                ),
            )
        if diagnostics is not None and hasattr(diagnostics, "record_external"):
            diagnostics.record_external(
                "packing.layout",
                time.perf_counter() - layout_started,
                metadata={
                    "sequence_rows": int(layout.sequence_length),
                    "condition_video_rows": int(layout.num_condition_video_rows),
                    "condition_audio_rows": int(layout.num_condition_audio_rows),
                },
            )

        video_sched, audio_sched = self._build_schedules(
            num_inference_steps, sampling_method=sampling_method
        )
        if has_initial_latents and refinement_strength < 1.0:
            full_steps = len(video_sched.timesteps)
            active_steps = max(1, min(full_steps, int(math.ceil(full_steps * refinement_strength))))
            video_sigmas = video_sched.sigmas.tolist()[-(active_steps + 1) :]
            audio_sigmas = audio_sched.sigmas.tolist()[-(active_steps + 1) :]
            video_sched.set_timesteps(sigmas=video_sigmas)
            audio_sched.set_timesteps(sigmas=audio_sigmas)

        mx.random.seed(seed)
        keyframe_video_rows = condition_video_rows
        continuation_video_rows = None
        continuation_audio_rows = None
        if has_continuation:
            continuation_video_rows = patchify_video_latents(
                continuation_video_latents.astype(mx.float32), patch_size
            )
            continuation_audio_rows = (
                continuation_audio_latents.astype(mx.float32)
                .transpose(0, 2, 1)
                .reshape(-1, self.dit.config.audio_latents_dim)
            )
        if keyframe_video_rows is not None:
            keyframe_video_rows = keyframe_video_rows.astype(mx.float32)
            condition_noise = mx.random.normal(keyframe_video_rows.shape).astype(mx.float32)
            keyframe_video_rows = MiniMaxH3Scheduler(
                shift=self.config.sigma_shift_video
            ).scale_noise(keyframe_video_rows, visual_condition_strength, condition_noise)
        packed_condition_video_rows = [
            rows for rows in (continuation_video_rows, keyframe_video_rows) if rows is not None
        ]
        condition_video_rows = (
            mx.concatenate(packed_condition_video_rows)
            if len(packed_condition_video_rows) > 1
            else (packed_condition_video_rows[0] if packed_condition_video_rows else None)
        )
        video_noise = mx.random.normal(expected_initial_video_shape).astype(mx.float32)
        video_latents = video_noise
        if initial_video_latents is not None:
            video_latents = video_sched.scale_noise(
                initial_video_latents.astype(mx.float32),
                float(video_sched.timesteps[0].item()),
                video_noise,
            )
        video_rows = patchify_video_latents(video_latents, patch_size)
        if condition_video_rows is not None:
            video_rows = mx.concatenate([condition_video_rows, video_rows])
        audio_noise = mx.random.normal(expected_initial_audio_shape).astype(mx.float32)
        audio_latents_working = audio_noise
        if initial_audio_latents is not None:
            audio_latents_working = audio_sched.scale_noise(
                initial_audio_latents.astype(mx.float32),
                float(audio_sched.timesteps[0].item()),
                audio_noise,
            )
        audio_rows = audio_latents_working.transpose(0, 2, 1).reshape(
            num_audio_latents * AUDIO_CHANNELS, self.dit.config.audio_latents_dim
        )
        if condition_audio_rows is not None:
            condition_audio_rows = condition_audio_rows.astype(mx.float32)
            if audio_condition_strength < 1.0:
                audio_noise = mx.random.normal(
                    condition_audio_rows.shape,
                    key=mx.random.key(int(seed) + 1),
                ).astype(mx.float32)
                condition_audio_rows = MiniMaxH3Scheduler(
                    shift=self.config.sigma_shift_audio
                ).scale_noise(
                    condition_audio_rows,
                    audio_condition_strength,
                    audio_noise,
                )
        if continuation_audio_rows is not None:
            condition_audio_rows = (
                mx.concatenate([continuation_audio_rows, condition_audio_rows])
                if condition_audio_rows is not None
                else continuation_audio_rows
            )
        if condition_audio_rows is not None:
            audio_rows = mx.concatenate([condition_audio_rows, audio_rows])

        timestep_table, plan = self._row_timestep_plan(
            layout,
            video_sched.timesteps,
            audio_sched.timesteps,
            visual_condition_strength,
            audio_condition_strength,
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
        core_reuse_marker = None
        if easycache_config is not None:
            from .easycache import CORE_RESIDUAL_REUSE, H3EasyCacheState

            easycache = H3EasyCacheState(easycache_config)
            core_reuse_marker = CORE_RESIDUAL_REUSE
            if terminal_target_only and easycache.uses_core_residual:
                raise ValueError(
                    "Core-residual EasyCache is incompatible with terminal target-only research."
                )
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
        condition_video_count = layout.num_condition_video_rows
        condition_audio_count = layout.num_condition_audio_rows
        target_only_forecast = bool(
            trajectory_forecast is not None
            and trajectory_forecast.config.conditioned_row_policy == "target_only"
        )
        trajectory_video_row_start = condition_video_count if target_only_forecast else 0
        trajectory_audio_row_start = condition_audio_count if target_only_forecast else 0

        def run_pass(start_video_rows, start_audio_rows, progress_offset, progress_total):
            nonlocal transformer_evaluations

            from .lora import lora_evaluation

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
                core_reuse = core_reuse_marker is not None and reused is core_reuse_marker
                blockcache_hit = False
                hierarchical_blockcache = blockcache is not None and hasattr(
                    blockcache, "segment_hits"
                )
                replaying = bool(trajectory_forecast is not None and trajectory_forecast.replaying)
                if reused is None or core_reuse:
                    if diagnostics is not None and hasattr(diagnostics, "begin_evaluation"):
                        diagnostics.begin_evaluation(
                            index,
                            timestep=float(timestep),
                            audio_timestep=float(audio_sched.timesteps[index].item()),
                        )
                    with lora_evaluation(index, total_steps):
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
                            easycache_core=(
                                easycache
                                if easycache is not None and easycache.uses_core_residual
                                else None
                            ),
                            trajectory_forecast=trajectory_forecast,
                            forecast_coordinate=float(timestep),
                            trajectory_video_row_start=trajectory_video_row_start,
                            trajectory_audio_row_start=trajectory_audio_row_start,
                            terminal_target_only=terminal_target_only,
                            terminal_video_row_start=condition_video_count,
                            terminal_audio_row_start=condition_audio_count,
                            step_index=index,
                            total_steps=total_steps,
                            diagnostics=diagnostics,
                        )
                    actual_transformer = not core_reuse and not replaying and (
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
                    if easycache is not None and not core_reuse:
                        easycache.update(video_input, audio_input, video_pred, audio_pred)
                else:
                    video_pred, audio_pred = reused
                scheduler_started = time.perf_counter()
                preview_latents = None
                if latent_preview_callback is not None:
                    sigma_from_timestep = float(
                        np.float32(1.0) - np.float32(timestep)
                    )
                    denoised_video_rows = (
                        current_video_rows[condition_video_count:]
                        + sigma_from_timestep
                        * video_pred[0, condition_video_count:].astype(mx.float32)
                    )
                    preview_latents = unpatchify_video_tokens(
                        denoised_video_rows,
                        num_latent_frames,
                        latent_height,
                        latent_width,
                        self.dit.config.latents_dim,
                        patch_size,
                    )
                stepped_video = video_sched.step(
                    video_pred[0, condition_video_count:].astype(mx.float32),
                    float(timestep),
                    current_video_rows[condition_video_count:],
                )
                stepped_audio = audio_sched.step(
                    audio_pred[0, condition_audio_count:].astype(mx.float32),
                    float(audio_sched.timesteps[index].item()),
                    current_audio_rows[condition_audio_count:],
                )
                current_video_rows = (
                    mx.concatenate([current_video_rows[:condition_video_count], stepped_video])
                    if condition_video_count
                    else stepped_video
                )
                current_audio_rows = (
                    mx.concatenate([current_audio_rows[:condition_audio_count], stepped_audio])
                    if condition_audio_count
                    else stepped_audio
                )
                if preview_latents is not None:
                    mx.eval(current_video_rows, current_audio_rows, preview_latents)
                else:
                    mx.eval(current_video_rows, current_audio_rows)
                if diagnostics is not None and hasattr(diagnostics, "record_external"):
                    diagnostics.record_external(
                        "scheduler.step_and_repack",
                        time.perf_counter() - scheduler_started,
                        metadata={
                            "condition_video_rows": int(condition_video_count),
                            "condition_audio_rows": int(condition_audio_count),
                        },
                    )
                if preview_latents is not None:
                    latent_preview_callback(
                        progress_offset + index + 1,
                        progress_total,
                        preview_latents,
                    )
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
            video_rows[condition_video_count:],
            num_latent_frames,
            latent_height,
            latent_width,
            self.dit.config.latents_dim,
            patch_size,
        )
        audio_latents = unpack_audio_tokens(audio_rows[condition_audio_count:], num_audio_latents)
        if preserve_initial_audio:
            audio_latents = initial_audio_latents
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
            easycache_reuse_strategy=(
                easycache.config.reuse_strategy if easycache is not None else None
            ),
            easycache_cache_bytes=easycache.cache_bytes if easycache is not None else 0,
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
            trajectory_conditioned_row_policy=(
                trajectory_forecast_config.conditioned_row_policy
                if trajectory_forecast_config is not None
                else None
            ),
            trajectory_excluded_video_rows=trajectory_video_row_start,
            trajectory_excluded_audio_rows=trajectory_audio_row_start,
            refinement_strength=(refinement_strength if has_initial_latents else None),
            refinement_audio_preserved=bool(preserve_initial_audio),
            prepared_state_cache_hits=self._cache_hits,
            prepared_state_cache_builds=self._cache_builds,
            prepared_state_cache_bytes=(self._cache.nbytes() if self._cache is not None else 0),
            prepared_state_build_seconds=self._cache_build_seconds,
            prepared_state_key=self._cache_key_digest,
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
            duration_seconds: 2.5 to 15; snapped up to the ``17n + 5`` frame grid the VAE encodes.
                Durations below 5 seconds are experimental.
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
