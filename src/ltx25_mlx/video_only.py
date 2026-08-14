"""Video-only LTX 2.5 transformer execution for temporal refinement."""

from __future__ import annotations

from contextlib import nullcontext

import mlx.core as mx
import mlx.nn as nn


class _VideoOnlyBlock(nn.Module):
    """Execute only the video branches of one joint AV transformer block."""

    def __init__(self, block) -> None:
        super().__init__()
        self.block = block

    def __call__(
        self,
        video_hidden: mx.array,
        video_adaln_params: mx.array,
        video_prompt_adaln_params: mx.array,
        video_text_embeds: mx.array | None,
        video_rope_freqs: mx.array | None,
        video_attention_mask: mx.array | None,
        video_cross_attention_mask: mx.array | None,
    ) -> mx.array:
        block = self.block
        dim = video_hidden.shape[-1]
        (
            shift_sa,
            scale_sa,
            gate_sa,
            shift_ff,
            scale_ff,
            gate_ff,
            shift_ca,
            scale_ca,
            gate_ca,
        ) = block._unpack_adaln(
            video_adaln_params,
            block.scale_shift_table,
            9,
            dim,
        )

        normed = block._rms_norm(video_hidden) * (1.0 + scale_sa) + shift_sa
        video_hidden = video_hidden + block.attn1(
            normed,
            rope_freqs=video_rope_freqs,
            attention_mask=video_attention_mask,
        ) * gate_sa

        if video_text_embeds is not None:
            normed = block._rms_norm(video_hidden) * (1.0 + scale_ca) + shift_ca
            prompt_shift, prompt_scale = block._unpack_adaln(
                video_prompt_adaln_params,
                block.prompt_scale_shift_table,
                2,
                dim,
            )
            text_scaled = video_text_embeds * (1.0 + prompt_scale) + prompt_shift
            video_hidden = video_hidden + block.attn2(
                normed,
                encoder_hidden_states=text_scaled,
                attention_mask=video_cross_attention_mask,
            ) * gate_ca

        normed = block._rms_norm(video_hidden) * (1.0 + scale_ff) + shift_ff
        return video_hidden + block.ff(normed) * gate_ff


def _unwrap(transformer):
    current = transformer
    visited: set[int] = set()
    while id(current) not in visited:
        visited.add(id(current))
        inner = getattr(current, "inner", None)
        if inner is None or inner is current:
            return current
        current = inner
    raise ValueError("LTX 2.5 transformer wrappers contain a cycle.")


class LTX25VideoOnlyX0Model:
    """Predict video x0 without materializing or evaluating the audio branches."""

    def __init__(self, transformer) -> None:
        self.transformer = transformer
        self.inner = _unwrap(transformer)
        self._resident_adapters: dict[int, _VideoOnlyBlock] = {}
        self._streaming_adapters: tuple[_VideoOnlyBlock, ...] | None = None
        self._compiled_streaming_adapters = None

    def _streaming_provider(self):
        wrapper = self.transformer
        streamer = getattr(wrapper, "_streamer", None)
        if streamer is None:
            return None, nullcontext(), 0

        shared_blocks = getattr(wrapper, "_shared_blocks", None)
        if shared_blocks is None:
            shared_blocks = (wrapper._shared_block,)
        if self._streaming_adapters is None:
            self._streaming_adapters = tuple(_VideoOnlyBlock(block) for block in shared_blocks)
            self._compiled_streaming_adapters = tuple(
                mx.compile(adapter, inputs=adapter) for adapter in self._streaming_adapters
            )
        adapters = self._compiled_streaming_adapters
        lora_sources = getattr(wrapper, "_lora_sources", ())
        previous: list[int | None] = [None] * len(shared_blocks)

        def provider(index: int):
            slot = index % len(shared_blocks)
            streamer.bind(
                shared_blocks[slot],
                index,
                evict_previous=previous[slot],
                lora_sources=lora_sources or None,
            )
            previous[slot] = index
            return adapters[slot]

        try:
            from .transformer import _STREAMING_EVAL_LOCK

            lock = _STREAMING_EVAL_LOCK
        except ImportError:  # pragma: no cover - defensive package fallback
            lock = nullcontext()
        return provider, lock, len(shared_blocks)

    def _resident_adapter(self, block) -> _VideoOnlyBlock:
        key = id(block)
        adapter = self._resident_adapters.get(key)
        if adapter is None:
            adapter = _VideoOnlyBlock(block)
            self._resident_adapters[key] = adapter
        return adapter

    def __call__(
        self,
        *,
        video_latent: mx.array,
        audio_latent: None,
        sigma: mx.array,
        video_text_embeds: mx.array | None = None,
        audio_text_embeds: mx.array | None = None,
        video_positions: mx.array | None = None,
        audio_positions: None = None,
        video_attention_mask: mx.array | None = None,
        audio_attention_mask: None = None,
        video_cross_attention_mask: mx.array | None = None,
        video_timesteps: mx.array | None = None,
        audio_timesteps: None = None,
        **_kwargs,
    ) -> tuple[mx.array, None]:
        del (
            audio_latent,
            audio_text_embeds,
            audio_positions,
            audio_attention_mask,
            audio_timesteps,
        )
        model = self.inner
        video_latent = video_latent.astype(mx.bfloat16)
        if video_text_embeds is not None:
            video_text_embeds = video_text_embeds.astype(mx.bfloat16)
        video_hidden = model.patchify_proj(video_latent)
        timestep = sigma.astype(mx.bfloat16)
        timestep_embedding = model._embed_timestep_scalar(timestep)
        if video_timesteps is not None:
            token_embedding = model._embed_timestep_per_token(video_timesteps)
            video_adaln, embedded_timestep = model._adaln_per_token(
                model.adaln_single,
                token_embedding,
            )
        else:
            video_adaln, embedded_timestep = model.adaln_single(timestep_embedding)
        video_prompt_adaln, _ = model.prompt_adaln_single(timestep_embedding)
        video_rope = None
        if video_positions is not None:
            video_rope = model._compute_rope_freqs(
                video_positions,
                model.config.video_num_heads,
                model.config.video_head_dim,
            )

        provider, lock, streaming_window = self._streaming_provider()
        with lock:
            for block_index in range(model.config.num_layers):
                if provider is None:
                    block = model.transformer_blocks[block_index]
                    adapter = self._resident_adapter(block)
                else:
                    adapter = provider(block_index)
                video_hidden = adapter(
                    video_hidden,
                    video_adaln,
                    video_prompt_adaln,
                    video_text_embeds,
                    video_rope,
                    video_attention_mask,
                    video_cross_attention_mask,
                )
                if provider is not None and (block_index + 1) % streaming_window == 0:
                    mx.eval(video_hidden)
            if provider is not None and model.config.num_layers % streaming_window:
                mx.eval(video_hidden)

        velocity = model._output_block(
            video_hidden,
            embedded_timestep,
            model.scale_shift_table,
            model.proj_out,
        )
        if video_timesteps is not None:
            video_sigma = video_timesteps[:, :, None].astype(mx.float32)
        else:
            video_sigma = sigma[:, None, None].astype(mx.float32)
        video_x0 = (
            video_latent.astype(mx.float32) - video_sigma * velocity.astype(mx.float32)
        ).astype(video_latent.dtype)
        return video_x0, None


__all__ = ["LTX25VideoOnlyX0Model"]
