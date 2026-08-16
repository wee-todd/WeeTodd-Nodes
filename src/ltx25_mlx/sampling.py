"""LTX 2.5 distilled sampling primitives for MLX."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

import mlx.core as mx


@dataclass(frozen=True)
class LTX25DenoiseOutput:
    """Final video and audio latent tensors from one distilled sampling stage."""

    video_latent: mx.array | None
    audio_latent: mx.array | None


def euler_ancestral_step(
    sample: mx.array,
    denoised_sample: mx.array,
    sigma: float,
    sigma_next: float,
    *,
    noise: mx.array | None,
    eta: float = 1.0,
    s_noise: float = 1.0,
) -> mx.array:
    """Advance one variance-preserving rectified-flow Euler ancestral step."""
    if sigma_next == 0.0:
        return denoised_sample.astype(sample.dtype)
    if eta > 0.0 and noise is None:
        raise ValueError("LTX 2.5 Euler ancestral sampling requires noise when eta is positive.")
    if sigma <= 0.0:
        raise ValueError("LTX 2.5 Euler ancestral sampling requires a positive current sigma.")

    source_dtype = sample.dtype
    source = sample.astype(mx.float32)
    denoised = denoised_sample.astype(mx.float32)
    downstep_ratio = 1.0 + (sigma_next / sigma - 1.0) * eta
    sigma_down = sigma_next * downstep_ratio
    sigma_down_ratio = sigma_down / sigma
    next_sample = sigma_down_ratio * source + (1.0 - sigma_down_ratio) * denoised

    if eta > 0.0:
        alpha_next = 1.0 - sigma_next
        alpha_down = 1.0 - sigma_down
        variance = sigma_next**2 - sigma_down**2 * alpha_next**2 / alpha_down**2
        renoise = max(variance, 0.0) ** 0.5
        next_sample = (
            alpha_next / alpha_down * next_sample + noise.astype(mx.float32) * s_noise * renoise
        )
    return next_sample.astype(source_dtype)


def _masked_clean(denoised: mx.array, state: Any) -> mx.array:
    return denoised * state.denoise_mask + state.clean_latent * (1.0 - state.denoise_mask)


def _uniform(mask: mx.array) -> bool:
    return bool(mx.all(mask == 1.0).item())


def _timesteps(sigma: float, mask: mx.array) -> mx.array:
    return (mask * sigma).squeeze(-1)


def euler_ancestral_denoise_loop(
    model: Callable[..., tuple[mx.array | None, mx.array | None]],
    video_state: Any | None,
    audio_state: Any | None,
    video_text_embeds: mx.array,
    audio_text_embeds: mx.array,
    *,
    sigmas: tuple[float, ...] | list[float],
    noise_seed: int,
    eta: float = 1.0,
    s_noise: float = 1.0,
    check_interrupted: Callable[[], None] | None = None,
    step_callback: Callable[[int, int], None] | None = None,
    evaluation_timing_callback: Callable[[int, float], None] | None = None,
) -> LTX25DenoiseOutput:
    """Run the LTX 2.5 distilled ancestral stage over joint audio/video latents."""
    if video_state is None and audio_state is None:
        raise ValueError("LTX 2.5 sampling requires a video or audio latent state.")
    if len(sigmas) < 2:
        raise ValueError("LTX 2.5 sampling requires at least two sigma points.")

    states = {"video": video_state, "audio": audio_state}
    video_uniform = video_state is None or _uniform(video_state.denoise_mask)
    audio_uniform = audio_state is None or _uniform(audio_state.denoise_mask)
    random_key = mx.random.key(noise_seed)
    total = len(sigmas) - 1

    for index, (sigma, sigma_next) in enumerate(zip(sigmas[:-1], sigmas[1:], strict=True)):
        evaluation_started = time.perf_counter()
        if check_interrupted is not None:
            check_interrupted()
        current_video = states["video"]
        current_audio = states["audio"]
        representative = current_video if current_video is not None else current_audio
        batch = representative.latent.shape[0]
        sigma_array = mx.broadcast_to(mx.array([sigma], dtype=mx.bfloat16), (batch,))
        kwargs: dict[str, Any] = {
            "video_latent": current_video.latent if current_video is not None else None,
            "audio_latent": current_audio.latent if current_audio is not None else None,
            "sigma": sigma_array,
            "video_text_embeds": video_text_embeds,
            "audio_text_embeds": audio_text_embeds,
            "video_positions": getattr(current_video, "positions", None),
            "audio_positions": getattr(current_audio, "positions", None),
            "video_attention_mask": getattr(current_video, "attention_mask", None),
            "audio_attention_mask": getattr(current_audio, "attention_mask", None),
        }
        if current_video is not None and not video_uniform:
            kwargs["video_timesteps"] = _timesteps(sigma, current_video.denoise_mask)
        if current_audio is not None and not audio_uniform:
            kwargs["audio_timesteps"] = _timesteps(sigma, current_audio.denoise_mask)

        from .sol_attention import set_ltx25_sol_context

        set_ltx25_sol_context(model, step_index=index, total_steps=total)
        video_denoised, audio_denoised = model(**kwargs)
        for modality, state, denoised in (
            ("video", current_video, video_denoised),
            ("audio", current_audio, audio_denoised),
        ):
            if state is None or denoised is None:
                continue
            clean_prediction = _masked_clean(denoised.astype(mx.float32), state)
            if sigma_next == 0.0:
                next_latent = clean_prediction.astype(state.latent.dtype)
            else:
                noise = None
                if eta > 0.0:
                    split_keys = mx.random.split(random_key, 2)
                    random_key, draw_key = split_keys[0], split_keys[1]
                    noise = mx.random.normal(state.latent.shape, key=draw_key)
                next_latent = euler_ancestral_step(
                    state.latent,
                    clean_prediction,
                    sigma,
                    sigma_next,
                    noise=noise,
                    eta=eta,
                    s_noise=s_noise,
                )
                next_latent = _masked_clean(next_latent, state).astype(state.latent.dtype)
            states[modality] = replace(state, latent=next_latent)

        mx.async_eval(
            *(state.latent for state in states.values() if state is not None),
        )
        if evaluation_timing_callback is not None:
            mx.eval(*(state.latent for state in states.values() if state is not None))
            evaluation_timing_callback(index + 1, time.perf_counter() - evaluation_started)
        if step_callback is not None:
            step_callback(index + 1, total)
        if sigma_next == 0.0:
            break

    return LTX25DenoiseOutput(
        video_latent=states["video"].latent if states["video"] is not None else None,
        audio_latent=states["audio"].latent if states["audio"] is not None else None,
    )


def euler_ancestral_cfg_pp_denoise_loop(
    model: Callable[..., tuple[mx.array | None, mx.array | None]],
    video_state: Any | None,
    audio_state: Any | None,
    video_text_embeds: mx.array,
    audio_text_embeds: mx.array,
    negative_video_text_embeds: mx.array,
    negative_audio_text_embeds: mx.array,
    *,
    sigmas: tuple[float, ...] | list[float],
    noise_seed: int,
    eta: float = 1.0,
    s_noise: float = 1.0,
    check_interrupted: Callable[[], None] | None = None,
    step_callback: Callable[[int, int], None] | None = None,
    evaluation_timing_callback: Callable[[int, float], None] | None = None,
    forward_timing_callback: Callable[[int, str, float], None] | None = None,
    batch_branches: bool = False,
    cfg_pp_step_indices: tuple[int, ...] | None = None,
) -> LTX25DenoiseOutput:
    """Run Comfy-compatible Euler ancestral CFG++ sampling.

    CFG++ requires both conditional and unconditional predictions at every
    non-final sigma, even when CFG is 1.0. Eight displayed sampler steps therefore
    execute sixteen transformer forwards. The conditioned prediction remains the
    denoising target while the unconditional prediction supplies the ODE direction.
    """
    from ltx_core_mlx.components.diffusion_steps import EulerCfgPpDiffusionStep

    if video_state is None and audio_state is None:
        raise ValueError("LTX 2.5 CFG++ sampling requires a video or audio latent state.")
    if len(sigmas) < 2:
        raise ValueError("LTX 2.5 CFG++ sampling requires at least two sigma points.")

    states = {"video": video_state, "audio": audio_state}
    video_uniform = video_state is None or _uniform(video_state.denoise_mask)
    audio_uniform = audio_state is None or _uniform(audio_state.denoise_mask)
    sigma_values = mx.array(sigmas, dtype=mx.float32)
    stepper = EulerCfgPpDiffusionStep(eta=eta, s_noise=s_noise)
    random_key = mx.random.key(noise_seed)
    total = len(sigmas) - 1
    selected_cfg_steps = (
        frozenset(range(total))
        if cfg_pp_step_indices is None
        else frozenset(int(value) for value in cfg_pp_step_indices)
    )
    if any(value < 0 or value >= total for value in selected_cfg_steps):
        raise ValueError("CFG++ step indices must address an existing sampler step.")
    forward_index = 0

    for index, (sigma, sigma_next) in enumerate(zip(sigmas[:-1], sigmas[1:], strict=True)):
        step_started = time.perf_counter()
        if check_interrupted is not None:
            check_interrupted()
        current_video = states["video"]
        current_audio = states["audio"]
        representative = current_video if current_video is not None else current_audio
        batch = representative.latent.shape[0]
        sigma_array = mx.broadcast_to(mx.array([sigma], dtype=mx.bfloat16), (batch,))
        shared: dict[str, Any] = {
            "video_latent": current_video.latent if current_video is not None else None,
            "audio_latent": current_audio.latent if current_audio is not None else None,
            "sigma": sigma_array,
            "video_positions": getattr(current_video, "positions", None),
            "audio_positions": getattr(current_audio, "positions", None),
            "video_attention_mask": getattr(current_video, "attention_mask", None),
            "audio_attention_mask": getattr(current_audio, "attention_mask", None),
        }
        if current_video is not None and not video_uniform:
            shared["video_timesteps"] = _timesteps(sigma, current_video.denoise_mask)
        if current_audio is not None and not audio_uniform:
            shared["audio_timesteps"] = _timesteps(sigma, current_audio.denoise_mask)
        # The unconditional prediction is not consumed by the terminal update.
        use_cfg_pp = sigma_next != 0.0 and index in selected_cfg_steps
        from .sol_attention import set_ltx25_sol_context

        set_ltx25_sol_context(model, step_index=index, total_steps=total)

        if batch_branches and use_cfg_pp:
            batched_started = time.perf_counter()

            def pair(value, expected_batch=batch):
                if value is None or value.shape[0] != expected_batch:
                    return value
                return mx.concatenate([value, value], axis=0)

            video_pair, audio_pair = model(
                **{key: pair(value) for key, value in shared.items()},
                video_text_embeds=mx.concatenate(
                    [video_text_embeds, negative_video_text_embeds], axis=0
                ),
                audio_text_embeds=mx.concatenate(
                    [audio_text_embeds, negative_audio_text_embeds], axis=0
                ),
            )
            mx.eval(*(value for value in (video_pair, audio_pair) if value is not None))
            video_cond = video_pair[:batch] if video_pair is not None else None
            video_uncond = video_pair[batch:] if video_pair is not None else None
            audio_cond = audio_pair[:batch] if audio_pair is not None else None
            audio_uncond = audio_pair[batch:] if audio_pair is not None else None
            forward_index += 1
            if forward_timing_callback is not None:
                forward_timing_callback(
                    forward_index,
                    "batched_conditional_unconditional",
                    time.perf_counter() - batched_started,
                )
        else:
            conditional_started = time.perf_counter()
            video_cond, audio_cond = model(
                **shared,
                video_text_embeds=video_text_embeds,
                audio_text_embeds=audio_text_embeds,
            )
            mx.eval(*(value for value in (video_cond, audio_cond) if value is not None))
            forward_index += 1
            if forward_timing_callback is not None:
                forward_timing_callback(
                    forward_index, "conditional", time.perf_counter() - conditional_started
                )

            video_uncond = None
            audio_uncond = None
            if use_cfg_pp:
                if check_interrupted is not None:
                    check_interrupted()
                unconditional_started = time.perf_counter()
                video_uncond, audio_uncond = model(
                    **shared,
                    video_text_embeds=negative_video_text_embeds,
                    audio_text_embeds=negative_audio_text_embeds,
                )
                mx.eval(*(value for value in (video_uncond, audio_uncond) if value is not None))
                forward_index += 1
                if forward_timing_callback is not None:
                    forward_timing_callback(
                        forward_index,
                        "unconditional",
                        time.perf_counter() - unconditional_started,
                    )

        for modality, state, conditioned, unconditioned in (
            ("video", current_video, video_cond, video_uncond),
            ("audio", current_audio, audio_cond, audio_uncond),
        ):
            if state is None or conditioned is None:
                continue
            cond_clean = _masked_clean(conditioned.astype(mx.float32), state)
            if sigma_next == 0.0:
                next_latent = cond_clean.astype(state.latent.dtype)
            else:
                noise = None
                if eta > 0.0 and s_noise > 0.0:
                    split_keys = mx.random.split(random_key, 2)
                    random_key, draw_key = split_keys[0], split_keys[1]
                    noise = mx.random.normal(state.latent.shape, key=draw_key)
                if use_cfg_pp:
                    if unconditioned is None:
                        raise RuntimeError("CFG++ selected a step without an unconditional output.")
                    uncond_clean = _masked_clean(unconditioned.astype(mx.float32), state)
                    next_latent = stepper.step(
                        state.latent,
                        cond_clean,
                        sigma_values,
                        index,
                        uncond_denoised=uncond_clean,
                        noise=noise,
                    )
                else:
                    next_latent = euler_ancestral_step(
                        state.latent,
                        cond_clean,
                        sigma,
                        sigma_next,
                        noise=noise,
                        eta=eta,
                        s_noise=s_noise,
                    )
                next_latent = _masked_clean(next_latent, state).astype(state.latent.dtype)
            states[modality] = replace(state, latent=next_latent)

        mx.async_eval(*(state.latent for state in states.values() if state is not None))
        if evaluation_timing_callback is not None:
            mx.eval(*(state.latent for state in states.values() if state is not None))
            evaluation_timing_callback(index + 1, time.perf_counter() - step_started)
        if step_callback is not None:
            step_callback(index + 1, total)

    return LTX25DenoiseOutput(
        video_latent=states["video"].latent if states["video"] is not None else None,
        audio_latent=states["audio"].latent if states["audio"] is not None else None,
    )
