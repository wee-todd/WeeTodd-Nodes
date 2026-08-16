"""MLX port of ``MiniMaxH3DiTModel`` — the 33B MiniMax-H3 diffusion transformer.

The module tree reproduces the *original* checkpoint names, so the released safetensors load
1:1 with no weight surgery:

    video_patch_proj / audio_patch_proj / condition_proj
    time_embedder.proj_in / .proj_out
    token_refiner.blocks.{i}.{norm1,norm2,attn.*,mlp.*} + token_refiner.final_norm
    blocks.{i}.{norm1,norm2,attn.*,mlp.*,adaln_proj.linear}
    final_layer.{norm,adaln_proj.linear,video_out,audio_out}

Two raw-checkpoint layout quirks are handled by reshape in the forward pass rather than by
rewriting weights:

* ``attn.qkv_proj`` rows are **per-head interleaved** — ``[h0: q,k,v][h1: q,k,v]...`` — so the
  projection output reshapes to ``(..., heads, 3, head_dim)`` and indexes q/k/v on axis -2.
* ``mlp.fc1`` is a fused **``[gate; value]``** SwiGLU projection; the reference computes
  ``fc2(silu(gate) * value)``.

MiniMax-H3 runs one stack over a single packed 1-D sequence holding text, conditioning and
target video rows, and audio rows. Attention is full self-attention over that sequence; there
is no cross-attention and no per-modality block weights. Modality-specific behaviour comes only
from the two input patch projections, the per-row AdaLN modality tag, and the two output heads.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import mlx.core as mx
import mlx.nn as nn

from wee_todd_mlx.execution_evidence import ExecutionEvidence

if TYPE_CHECKING:
    from .adaln import ModulationCache

from .config import MODALITY_NUM, DiTConfig


def _diagnostic_run(
    diagnostics,
    name: str,
    fn,
    *,
    block: int | None = None,
    metadata=None,
    capture_as: str | None = None,
):
    """Run directly unless an opt-in research diagnostic session was supplied."""
    if diagnostics is None:
        return fn()
    return diagnostics.measure(
        name,
        fn,
        block=block,
        metadata=metadata,
        capture_as=capture_as,
    )


def param_dtype(layer: nn.Module) -> mx.Dtype:
    """The dtype a layer's *activations* should be aligned to.

    The reference casts each input to its projection's parameter dtype, and reading
    ``layer.weight.dtype`` is the obvious way to do that — but it is wrong the moment the layer is
    quantized. ``QuantizedLinear.weight`` is **packed uint32 storage**, so casting activations to it
    truncates them to integers, silently and identically at every bit width. The scales carry the
    real compute dtype.
    """
    while hasattr(layer, "base"):
        layer = layer.base
    scales = getattr(layer, "scales", None)
    return scales.dtype if scales is not None else layer.weight.dtype


def projection_weight_shape(layer: nn.Module) -> list[int]:
    """Return the stored projection shape through activation-space adapter wrappers."""
    while hasattr(layer, "base"):
        layer = layer.base
    return list(layer.weight.shape)


def timestep_embedding(
    timesteps: mx.array,
    dim: int,
    max_period: float = 10000.0,
    flip_sin_to_cos: bool = True,
    downscale_freq_shift: float = 0.0,
) -> mx.array:
    """Sinusoidal timestep embedding matching diffusers ``Timesteps`` / ``get_timestep_embedding``.

    Timesteps are consumed unscaled in ``[0, 1]``.
    """
    half_dim = dim // 2
    exponent = -math.log(max_period) * mx.arange(half_dim, dtype=mx.float32)
    exponent = exponent / (half_dim - downscale_freq_shift)
    emb = mx.exp(exponent)
    emb = timesteps.astype(mx.float32)[:, None] * emb[None, :]
    # diffusers builds [sin, cos] then swaps the halves when `flip_sin_to_cos`.
    if flip_sin_to_cos:
        return mx.concatenate([mx.cos(emb), mx.sin(emb)], axis=-1)
    return mx.concatenate([mx.sin(emb), mx.cos(emb)], axis=-1)


class TimestepEmbedder(nn.Module):
    """The timestep MLP shared by every AdaLN projection (``proj_in`` -> silu -> ``proj_out``).

    Kept in float32: it is a float32 module in the released mixed-precision checkpoint, and every
    block reads the same ``temb``, so a rounding applied here biases every block's modulation
    identically at every sampling step and accumulates coherently over the denoising trajectory.
    """

    def __init__(self, config: DiTConfig):
        super().__init__()
        self.proj_in = nn.Linear(
            config.timestep_input_dim, config.time_embed_hidden_size, bias=True
        )
        self.proj_out = nn.Linear(config.time_embed_hidden_size, config.time_embed_dim, bias=True)

    def __call__(self, sinusoid: mx.array) -> mx.array:
        return self.proj_out(nn.silu(self.proj_in(sinusoid)))


class RotaryPosEmbed3D:
    """3-axis rotary embedding over the ``(t, h, w)`` coordinates of the packed sequence.

    One ``inv_freq`` buffer of ``rope_inv_freq_len`` frequencies is shared by the three axes.
    Each axis contributes that many angles; the three blocks are concatenated to
    ``3 * rope_inv_freq_len`` and then concatenated with themselves, so the rotate-half convention
    rotates ``2 * 3 * rope_inv_freq_len`` of the head channels and passes the rest through.
    """

    def __init__(self, config: DiTConfig):
        n = config.rope_inv_freq_len
        self.inv_freq = 1.0 / (
            config.rope_theta ** (mx.arange(0, 2 * n, 2, dtype=mx.float32) / (2 * n))
        )

    def __call__(self, position_ids: mx.array) -> tuple[mx.array, mx.array]:
        # position_ids: (seq_len, 3) -> cos/sin: (seq_len, 2 * 3 * inv_freq_len)
        pos = position_ids.astype(mx.float32)
        freqs = pos[..., None] * self.inv_freq.reshape(1, 1, -1)  # (seq, 3, n)
        freqs = mx.concatenate([freqs[:, 0], freqs[:, 1], freqs[:, 2]], axis=-1)
        freqs = mx.concatenate([freqs, freqs], axis=-1)
        return mx.cos(freqs), mx.sin(freqs)


def apply_rotary(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    """Rotate the leading ``rotary_dim`` channels of every head, pass the rest through.

    ``x`` is ``(batch, heads, seq, head_dim)``; ``cos``/``sin`` are ``(seq, rotary_dim)``.
    """
    rotary_dim = cos.shape[-1]
    x_rot, x_pass = x[..., :rotary_dim], x[..., rotary_dim:]
    cos = cos.astype(x.dtype)[None, None, :, :]
    sin = sin.astype(x.dtype)[None, None, :, :]
    half = rotary_dim // 2
    x1, x2 = x_rot[..., :half], x_rot[..., half:]
    rotated = mx.concatenate([-x2, x1], axis=-1)
    out = x_rot * cos + rotated * sin
    if x_pass.shape[-1] == 0:
        return out
    return mx.concatenate([out, x_pass], axis=-1)


class Attention(nn.Module):
    """Full self-attention over the packed sequence, with per-head q/k RMSNorm."""

    def __init__(self, config: DiTConfig):
        super().__init__()
        self.heads = config.num_attention_heads
        self.head_dim = config.attention_head_dim
        self.scale = self.head_dim**-0.5

        self.qkv_proj = nn.Linear(config.hidden_size, 3 * config.inner_dim, bias=False)
        self.q_norm = nn.RMSNorm(config.attention_head_dim, eps=config.qk_norm_eps)
        self.k_norm = nn.RMSNorm(config.attention_head_dim, eps=config.qk_norm_eps)
        self.out_proj = nn.Linear(config.inner_dim, config.hidden_size, bias=False)
        self.query_chunk_size: int | None = None
        self.head_chunk_size: int | None = None
        self.sol_config = None
        self.sol_evidence: ExecutionEvidence | None = None
        self.sol_route_records: list[tuple[int, int, mx.array]] | None = None
        self.sol_step_index = 0
        self.sol_total_steps = 1

    def _attend(self, q, k, v, mask, block_index):
        """Run SDPA with optional independent head groups.

        Attention heads do not interact until the output projection.  Splitting that axis bounds
        the SDPA workspace without changing the mathematical operation or the packed sequence.
        """
        config = self.sol_config
        evidence = self.sol_evidence
        if config is not None:
            if evidence is not None:
                evidence.record_call()
            if config.active(
                step_index=self.sol_step_index,
                total_steps=self.sol_total_steps,
                block_index=int(block_index or 0),
            ):
                from .sol_attention import sol_attention, supports_sol_attention

                if evidence is not None:
                    evidence.record_observed(dtype=q.dtype, shape=q.shape)
                if q.shape != k.shape or q.shape != v.shape:
                    if evidence is not None:
                        evidence.record_fallback("non_self_attention_shape")
                elif not supports_sol_attention(q, mask, config):
                    if evidence is not None:
                        reason = (
                            "attention_mask"
                            if mask is not None
                            else "unsupported_dtype"
                            if q.dtype != mx.bfloat16
                            else "unsupported_shape"
                        )
                        evidence.record_fallback(reason)
                else:
                    if evidence is not None:
                        evidence.record_eligible()
                        evidence.record_executed(work_units=0)
                    output, route_counts = sol_attention(
                        q,
                        k,
                        v,
                        scale=self.scale,
                        config=config,
                        return_route_counts=True,
                    )
                    if self.sol_route_records is not None:
                        self.sol_route_records.append(
                            (self.sol_step_index, int(block_index or 0), route_counts)
                        )
                    return output
            elif evidence is not None:
                evidence.record_dense_policy()

        chunk = self.head_chunk_size
        if chunk is None or q.shape[1] <= chunk:
            return mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask)
        pieces = []
        for start in range(0, q.shape[1], chunk):
            stop = min(start + chunk, q.shape[1])
            piece = mx.fast.scaled_dot_product_attention(
                q[:, start:stop],
                k[:, start:stop],
                v[:, start:stop],
                scale=self.scale,
                mask=mask,
            )
            mx.eval(piece)
            pieces.append(piece)
        return mx.concatenate(pieces, axis=1)

    def _normal(self, x, rotary, mask, block_index):
        """Original inference path, kept free of diagnostic callables and synchronization."""
        batch, sequence, _ = x.shape
        qkv = self.qkv_proj(x).reshape(batch, sequence, self.heads, 3, self.head_dim)
        q, k, v = qkv[:, :, :, 0], qkv[:, :, :, 1], qkv[:, :, :, 2]
        q = self.q_norm(q).transpose(0, 2, 1, 3)
        k = self.k_norm(k).transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)
        if rotary is not None:
            q = apply_rotary(q, *rotary)
            k = apply_rotary(k, *rotary)
        chunk = self.query_chunk_size
        if chunk is None or q.shape[-2] <= chunk:
            out = self._attend(q, k, v, mask, block_index)
        else:
            pieces = []
            for start in range(0, q.shape[-2], chunk):
                stop = min(start + chunk, q.shape[-2])
                chunk_mask = mask
                if mask is not None and mask.shape[-2] == q.shape[-2]:
                    chunk_mask = mask[..., start:stop, :]
                piece = self._attend(
                    q[..., start:stop, :], k, v, chunk_mask, block_index
                )
                mx.eval(piece)
                pieces.append(piece)
            out = mx.concatenate(pieces, axis=-2)
        out = out.transpose(0, 2, 1, 3).reshape(batch, sequence, self.heads * self.head_dim)
        return self.out_proj(out.astype(x.dtype))

    def selected_queries(self, x, query_indices, rotary, mask):
        """Attend selected queries to full-sequence K/V for terminal-block research."""
        batch, sequence, _ = x.shape
        qkv = self.qkv_proj(x).reshape(batch, sequence, self.heads, 3, self.head_dim)
        q = mx.take(qkv[:, :, :, 0], query_indices, axis=1)
        k = qkv[:, :, :, 1]
        v = qkv[:, :, :, 2]
        q = self.q_norm(q).transpose(0, 2, 1, 3)
        k = self.k_norm(k).transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)
        if rotary is not None:
            cos, sin = rotary
            q = apply_rotary(
                q,
                mx.take(cos, query_indices, axis=0),
                mx.take(sin, query_indices, axis=0),
            )
            k = apply_rotary(k, cos, sin)
        query_mask = mask
        if mask is not None and mask.shape[-2] == sequence:
            query_mask = mx.take(mask, query_indices, axis=-2)
        out = mx.fast.scaled_dot_product_attention(
            q,
            k,
            v,
            scale=self.scale,
            mask=query_mask,
        )
        out = out.transpose(0, 2, 1, 3).reshape(
            batch, query_indices.shape[0], self.heads * self.head_dim
        )
        return self.out_proj(out.astype(x.dtype))

    def __call__(
        self,
        x: mx.array,
        rotary: tuple[mx.array, mx.array] | None = None,
        mask: mx.array | None = None,
        diagnostics=None,
        block_index: int | None = None,
        module_prefix: str = "blocks",
    ) -> mx.array:
        if diagnostics is None:
            return self._normal(x, rotary, mask, block_index)
        B, S, _ = x.shape
        prefix = f"{module_prefix}.{block_index}.attn"
        # Raw-checkpoint QKV rows are per-head interleaved: (..., heads, 3, head_dim).
        qkv = _diagnostic_run(
            diagnostics,
            f"{prefix}.qkv_proj",
            lambda: self.qkv_proj(x).reshape(B, S, self.heads, 3, self.head_dim),
            block=block_index,
            metadata={
                "input_shape": list(x.shape),
                "weight_shape": projection_weight_shape(self.qkv_proj),
            },
            capture_as="qkv_output",
        )
        q, k, v = qkv[:, :, :, 0], qkv[:, :, :, 1], qkv[:, :, :, 2]

        if diagnostics is not None:
            diagnostics.capture("q_output", q, block=block_index)
            diagnostics.capture("k_output", k, block=block_index)
            diagnostics.capture("v_output", v, block=block_index)

        q, k = _diagnostic_run(
            diagnostics,
            f"{prefix}.qk_norm",
            lambda: (
                self.q_norm(q).transpose(0, 2, 1, 3),
                self.k_norm(k).transpose(0, 2, 1, 3),
            ),
            block=block_index,
        )
        v = v.transpose(0, 2, 1, 3)

        if rotary is not None:
            q, k = _diagnostic_run(
                diagnostics,
                f"{prefix}.rotary",
                lambda: (apply_rotary(q, *rotary), apply_rotary(k, *rotary)),
                block=block_index,
            )

        chunk = self.query_chunk_size
        if chunk is None or q.shape[-2] <= chunk:
            out = _diagnostic_run(
                diagnostics,
                f"{prefix}.sdpa",
                lambda: mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask),
                block=block_index,
                metadata={"query_rows": int(q.shape[-2]), "key_rows": int(k.shape[-2])},
            )
        else:
            pieces = []
            for start in range(0, q.shape[-2], chunk):
                stop = min(start + chunk, q.shape[-2])
                chunk_mask = mask
                if mask is not None and mask.shape[-2] == q.shape[-2]:
                    chunk_mask = mask[..., start:stop, :]
                piece = mx.fast.scaled_dot_product_attention(
                    q[..., start:stop, :], k, v, scale=self.scale, mask=chunk_mask
                )
                mx.eval(piece)
                pieces.append(piece)
            out = mx.concatenate(pieces, axis=-2)
        out = out.transpose(0, 2, 1, 3).reshape(B, S, self.heads * self.head_dim)
        return _diagnostic_run(
            diagnostics,
            f"{prefix}.out_proj",
            lambda: self.out_proj(out.astype(x.dtype)),
            block=block_index,
            metadata={"weight_shape": projection_weight_shape(self.out_proj)},
            capture_as="attention_output",
        )


class FeedForward(nn.Module):
    """SwiGLU feed-forward. ``fc1`` is the fused ``[gate; value]`` projection."""

    def __init__(self, config: DiTConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, 2 * config.ffn_hidden_size, bias=False)
        self.fc2 = nn.Linear(config.ffn_hidden_size, config.hidden_size, bias=False)
        self._ffn = config.ffn_hidden_size
        self.row_chunk_size: int | None = None

    def __call__(
        self,
        x: mx.array,
        diagnostics=None,
        block_index: int | None = None,
        module_prefix: str = "blocks",
    ) -> mx.array:
        if diagnostics is None:
            chunk = self.row_chunk_size
            if chunk is None or x.shape[-2] <= chunk:
                fused = self.fc1(x)
                gate, value = fused[..., : self._ffn], fused[..., self._ffn :]
                return self.fc2(nn.silu(gate) * value)
            pieces = []
            for start in range(0, x.shape[-2], chunk):
                stop = min(start + chunk, x.shape[-2])
                fused = self.fc1(x[..., start:stop, :])
                gate, value = fused[..., : self._ffn], fused[..., self._ffn :]
                piece = self.fc2(nn.silu(gate) * value)
                mx.eval(piece)
                pieces.append(piece)
            return mx.concatenate(pieces, axis=-2)
        prefix = f"{module_prefix}.{block_index}.mlp"
        fused = _diagnostic_run(
            diagnostics,
            f"{prefix}.fc1",
            lambda: self.fc1(x),
            block=block_index,
            metadata={
                "input_shape": list(x.shape),
                "weight_shape": projection_weight_shape(self.fc1),
            },
        )
        activated = _diagnostic_run(
            diagnostics,
            f"{prefix}.swiglu",
            lambda: nn.silu(fused[..., : self._ffn]) * fused[..., self._ffn :],
            block=block_index,
        )
        return _diagnostic_run(
            diagnostics,
            f"{prefix}.fc2",
            lambda: self.fc2(activated),
            block=block_index,
            metadata={"weight_shape": projection_weight_shape(self.fc2)},
            capture_as="mlp_output",
        )


class AdaLayerNormModulation(nn.Module):
    """Projects the shared timestep embedding into one block's six modulation parameters.

    ``(num_timesteps, time_embed_dim)`` -> six tensors of shape
    ``(num_timesteps * MODALITY_NUM, hidden_size)`` in
    ``shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp`` order. Row layout is
    ``[t0_mod0, t0_mod1, t0_mod2, t1_mod0, ...]``, which
    ``timestep_indices * MODALITY_NUM + token_tags`` addresses.

    One projection is shared by ``norm1`` and ``norm2`` and by the three modalities, so it cannot
    be folded into either norm.
    """

    def __init__(self, config: DiTConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.apply_silu = config.adaln_curve_grid is None
        self.linear = nn.Linear(config.time_embed_dim, config.adaln_out_features, bias=True)

    def __call__(self, temb: mx.array) -> tuple[mx.array, ...]:
        # Activate at `temb`'s own (float32) precision, cast to the projection's dtype after.
        h = nn.silu(temb) if self.apply_silu else temb
        h = h.astype(param_dtype(self.linear))
        h = self.linear(h).reshape(-1, 6 * self.hidden_size)
        return tuple(h[..., i * self.hidden_size : (i + 1) * self.hidden_size] for i in range(6))


class AdaLayerNormModulationOut(nn.Module):
    """``final_layer.adaln_proj`` — the ``linear`` sub-name matches the checkpoint."""

    def __init__(self, config: DiTConfig):
        super().__init__()
        self.apply_silu = config.adaln_curve_grid is None
        self.linear = nn.Linear(config.time_embed_dim, config.final_adaln_out_features, bias=True)


class FinalLayer(nn.Module):
    """Checkpoint's ``final_layer``: shared modulated norm plus the two output heads.

    The modulation table holds one row per *timestep* and is addressed per row of the packed
    sequence. The two halves of the projection are ``shift`` then ``scale``.
    """

    def __init__(self, config: DiTConfig):
        super().__init__()
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.final_norm_eps)
        self.adaln_proj = AdaLayerNormModulationOut(config)
        self.video_out = nn.Linear(config.hidden_size, config.video_patch_dim, bias=True)
        self.audio_out = nn.Linear(config.hidden_size, config.audio_latents_dim, bias=True)
        self.hidden_size = config.hidden_size

    def norm_out(self, x: mx.array, temb: mx.array, timestep_indices: mx.array) -> mx.array:
        h = nn.silu(temb) if self.adaln_proj.apply_silu else temb
        h = self.adaln_proj.linear(h.astype(param_dtype(self.adaln_proj.linear)))
        shift, scale = h[..., : self.hidden_size], h[..., self.hidden_size :]
        x = self.norm(x)
        return x * (1.0 + scale[timestep_indices]) + shift[timestep_indices]


class TokenRefinerBlock(nn.Module):
    """Plain pre-norm block used to refine the projected text stream. No AdaLN, no rotary."""

    def __init__(self, config: DiTConfig):
        super().__init__()
        self.norm1 = nn.RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.attn = Attention(config)
        self.norm2 = nn.RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.mlp = FeedForward(config)

    def __call__(self, x: mx.array, diagnostics=None, block_index: int | None = None) -> mx.array:
        if diagnostics is None:
            x = x + self.attn(self.norm1(x))
            return x + self.mlp(self.norm2(x))
        h = self.norm1(x)
        x = x + self.attn(
            h,
            diagnostics=diagnostics,
            block_index=block_index,
            module_prefix="token_refiner.blocks",
        )
        return x + self.mlp(
            self.norm2(x),
            diagnostics=diagnostics,
            block_index=block_index,
            module_prefix="token_refiner.blocks",
        )


class TokenRefiner(nn.Module):
    def __init__(self, config: DiTConfig):
        super().__init__()
        self.blocks = [TokenRefinerBlock(config) for _ in range(config.token_refiner_num_layers)]
        self.final_norm = nn.RMSNorm(config.hidden_size, eps=config.final_norm_eps)

    def __call__(self, x: mx.array, diagnostics=None) -> mx.array:
        if diagnostics is None:
            for block in self.blocks:
                x = block(x)
            return self.final_norm(x)
        for index, block in enumerate(self.blocks):
            if diagnostics is not None:
                diagnostics.prepare_block(x, index)
            x = block(x, diagnostics=diagnostics, block_index=index)
        return self.final_norm(x)


class TransformerBlock(nn.Module):
    """Pre-norm attention and feed-forward, each modulated by AdaLN parameters selected per row."""

    def __init__(self, config: DiTConfig):
        super().__init__()
        self.norm1 = nn.RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.attn = Attention(config)
        self.norm2 = nn.RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.mlp = FeedForward(config)
        self.adaln_proj = AdaLayerNormModulation(config)

    def __call__(
        self,
        x: mx.array,
        modulation: tuple[mx.array, ...],
        adaln_indices: mx.array,
        rotary: tuple[mx.array, mx.array],
        mask: mx.array | None = None,
        diagnostics=None,
        block_index: int | None = None,
    ) -> mx.array:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = modulation

        if diagnostics is None:
            h = self.norm1(x)
            h = h * (1.0 + scale_msa[adaln_indices]) + shift_msa[adaln_indices]
            x = x + gate_msa[adaln_indices] * self.attn(
                h, rotary, mask, block_index=block_index
            )
            h = self.norm2(x)
            h = h * (1.0 + scale_mlp[adaln_indices]) + shift_mlp[adaln_indices]
            return x + gate_mlp[adaln_indices] * self.mlp(h)

        block_input = x
        hybrid_result = diagnostics.try_hybrid_block(
            self,
            block_index,
            x,
            modulation=modulation,
            adaln_indices=adaln_indices,
            rotary=rotary,
            mask=mask,
        )
        if hybrid_result is not None:
            diagnostics.capture("residual_output", hybrid_result, block=block_index)
            return hybrid_result

        diagnostics.capture("block_input", x, block=block_index)

        h = _diagnostic_run(
            diagnostics,
            f"blocks.{block_index}.norm1_adaln",
            lambda: self.norm1(x) * (1.0 + scale_msa[adaln_indices]) + shift_msa[adaln_indices],
            block=block_index,
            capture_as="normalized_block_input",
        )
        attention = self.attn(
            h,
            rotary,
            mask,
            diagnostics=diagnostics,
            block_index=block_index,
        )
        x = _diagnostic_run(
            diagnostics,
            f"blocks.{block_index}.attention_residual",
            lambda: x + gate_msa[adaln_indices] * attention,
            block=block_index,
        )

        h = _diagnostic_run(
            diagnostics,
            f"blocks.{block_index}.norm2_adaln",
            lambda: self.norm2(x) * (1.0 + scale_mlp[adaln_indices]) + shift_mlp[adaln_indices],
            block=block_index,
            capture_as="mlp_input",
        )
        mlp = self.mlp(h, diagnostics=diagnostics, block_index=block_index)
        result = _diagnostic_run(
            diagnostics,
            f"blocks.{block_index}.mlp_residual",
            lambda: x + gate_mlp[adaln_indices] * mlp,
            block=block_index,
            capture_as="residual_output",
        )
        diagnostics.observe_hybrid_block(block_index, block_input, result)
        return result

    def selected_terminal_queries(
        self,
        x: mx.array,
        modulation: tuple[mx.array, ...],
        adaln_indices: mx.array,
        rotary: tuple[mx.array, mx.array],
        query_indices: mx.array,
        mask: mx.array | None = None,
    ) -> mx.array:
        """Return terminal-block target rows while retaining full-sequence K/V."""
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = modulation
        h = self.norm1(x)
        h = h * (1.0 + scale_msa[adaln_indices]) + shift_msa[adaln_indices]
        target_adaln = adaln_indices[query_indices]
        target = mx.take(x, query_indices, axis=1)
        target = target + gate_msa[target_adaln] * self.attn.selected_queries(
            h,
            query_indices,
            rotary,
            mask,
        )
        h = self.norm2(target)
        h = h * (1.0 + scale_mlp[target_adaln]) + shift_mlp[target_adaln]
        return target + gate_mlp[target_adaln] * self.mlp(h)


class MiniMaxH3DiT(nn.Module):
    """The MiniMax-H3 joint video + audio diffusion transformer.

    The caller builds the packed layout: patchified video latents, row order, the ``(t, h, w)``
    position grid, per-row modality tags and per-row timestep indices. See ``packing.py``.
    """

    def __init__(self, config: DiTConfig):
        super().__init__()
        self.config = config

        # 1. Per-modality input projections.
        self.video_patch_proj = nn.Linear(config.video_patch_dim, config.hidden_size, bias=True)
        self.audio_patch_proj = nn.Linear(config.audio_latents_dim, config.hidden_size, bias=True)
        self.condition_proj = nn.Linear(config.text_dim, config.hidden_size, bias=True)

        # 2. Timestep embedding, shared by every AdaLN projection. Pruned inference checkpoints
        #    carry a sampled low-rank curve instead of the original 13B-parameter projection path.
        if config.adaln_curve_grid is None:
            self.time_embedder = TimestepEmbedder(config)
        else:
            self.adaln_t_table = mx.zeros(
                (config.adaln_curve_grid, config.time_embed_dim), dtype=mx.float32
            )

        # 3. Text stream refiner.
        self.token_refiner = TokenRefiner(config)

        # 4. The block stack.
        self.blocks = [TransformerBlock(config) for _ in range(config.num_layers)]

        # 5. Shared output norm and the two per-modality heads. Both heads run over every row;
        #    the rows of each modality are selected afterwards.
        self.final_layer = FinalLayer(config)

        # Rotary is a computed buffer, not a parameter (`rope.inv_freq` is recomputed bit-exactly).
        self.rope = RotaryPosEmbed3D(config)
        self.sol_attention_config = None
        self.last_sol_attention_config = None
        self.sol_attention_evidence: ExecutionEvidence | None = None
        self.sol_route_records: list[tuple[int, int, mx.array]] = []
        self.sol_route_report: dict[str, object] | None = None

    def set_attention_query_chunk_size(self, chunk_size: int | None) -> None:
        """Select dense attention or memory-bounded query chunks for every attention layer."""
        if chunk_size is not None and chunk_size < 1:
            raise ValueError("attention query chunk size must be positive")
        for block in self.token_refiner.blocks:
            block.attn.query_chunk_size = chunk_size
        for block in self.blocks:
            block.attn.query_chunk_size = chunk_size
        paged = getattr(self, "paged_blocks", None)
        if paged is not None:
            paged.query_chunk_size = chunk_size

    def set_attention_head_chunk_size(self, chunk_size: int | None) -> None:
        """Bound SDPA workspace by processing independent attention-head groups."""
        if chunk_size is not None and chunk_size < 1:
            raise ValueError("attention head chunk size must be positive")
        for block in self.token_refiner.blocks:
            block.attn.head_chunk_size = chunk_size
        for block in self.blocks:
            block.attn.head_chunk_size = chunk_size

    def set_ffn_row_chunk_size(self, chunk_size: int | None) -> None:
        """Bound the SwiGLU intermediate by processing independent packed rows."""
        if chunk_size is not None and chunk_size < 1:
            raise ValueError("FFN row chunk size must be positive")
        for block in self.token_refiner.blocks:
            block.mlp.row_chunk_size = chunk_size
        for block in self.blocks:
            block.mlp.row_chunk_size = chunk_size

    def set_sol_attention_config(self, config) -> None:
        """Install or clear the opt-in Sol-style sparse-attention policy."""
        if config is not None:
            config.validate()
        self.sol_attention_config = config
        self.last_sol_attention_config = config
        self.sol_attention_evidence = (
            ExecutionEvidence(
                requested_backend="sol_attention",
                resolved_backend="mlx_fused_sol_bf16",
                scope="H3 packed transformer self-attention",
            )
            if config is not None
            else None
        )
        self.sol_route_records = []
        self.sol_route_report = None
        for block in self.blocks:
            block.attn.sol_config = config
            block.attn.sol_evidence = self.sol_attention_evidence
            block.attn.sol_route_records = self.sol_route_records
        paged = getattr(self, "paged_blocks", None)
        if paged is not None:
            paged.sol_config = config
            paged.sol_evidence = self.sol_attention_evidence
            paged.sol_route_records = self.sol_route_records

    def sol_attention_report(self) -> dict[str, object] | None:
        """Return the resolved H3 Sol policy and actual generation dispatch evidence."""

        config = self.last_sol_attention_config
        evidence = self.sol_attention_evidence
        if config is None or evidence is None:
            return None
        if self.sol_route_report is None and self.sol_route_records:
            from .sol_attention import route_telemetry_report

            self.sol_route_report = route_telemetry_report(self.sol_route_records)
            evidence.work_units_processed += int(
                self.sol_route_report["processed_key_row_units"]
            )
            evidence.work_units_avoided += int(
                self.sol_route_report["avoided_key_row_units"]
            )
        snapshot = evidence.snapshot()
        return {
            **config.__dict__,
            **snapshot,
            "attention_calls": snapshot["total_calls"],
            "sparse_kernel_calls": snapshot["executed_calls"],
            "fallback_calls": snapshot["fallback_calls"],
            **(self.sol_route_report or {}),
        }

    def embed_timesteps(self, timesteps: mx.array) -> mx.array:
        """Return original MLP embeddings or linearly interpolated pruned AdaLN coordinates."""
        if self.config.adaln_curve_grid is None:
            return self.time_embedder(timestep_embedding(timesteps, self.config.timestep_input_dim))

        grid = self.config.adaln_curve_grid
        position = mx.clip(timesteps.astype(mx.float32), 0.0, 1.0) * (grid - 1)
        lower = mx.minimum(mx.floor(position).astype(mx.int32), grid - 2)
        fraction = (position - lower.astype(mx.float32))[:, None]
        return (
            self.adaln_t_table[lower] * (1.0 - fraction) + self.adaln_t_table[lower + 1] * fraction
        )

    def pack_inputs(
        self,
        video_latents: mx.array,
        audio_latents: mx.array,
        text_embeds: mx.array,
        timestep: mx.array,
        timestep_indices: mx.array,
        token_tags: mx.array,
        position_ids: mx.array,
        video_indices: mx.array,
        audio_indices: mx.array,
        text_indices: mx.array,
        diagnostics=None,
    ) -> tuple[mx.array, mx.array, mx.array, tuple[mx.array, mx.array]]:
        """Everything the block stack needs before its first block: ``(x, temb, adaln_rows, rope)``.

        Split out of :meth:`__call__` so a step-skipping sampler can evaluate the block-0 modulated
        input — its skip indicator — without running the 50-block stack. The projections here are
        ~0.03% of a forward, so recomputing them on a probe is free relative to what a skip saves.
        """
        seq_len = position_ids.shape[0]
        if position_ids.ndim != 2 or position_ids.shape[-1] != 3:
            raise ValueError(f"`position_ids` must be (seq_len, 3), got {position_ids.shape}.")
        if token_tags.shape != (seq_len,) or timestep_indices.shape != (seq_len,):
            raise ValueError(
                "`token_tags` and `timestep_indices` must be (seq_len,) matching "
                "`position_ids`, got "
                f"{token_tags.shape} and {timestep_indices.shape} for seq_len={seq_len}."
            )

        rotary = _diagnostic_run(
            diagnostics,
            "input.rotary",
            lambda: self.rope(position_ids),
            metadata={"sequence_rows": int(seq_len)},
        )

        # 1. Project each modality and scatter the rows into the packed buffer. The text stream
        #    sets the dtype of the packed sequence.
        video_embeds = _diagnostic_run(
            diagnostics,
            "input.video_projection",
            lambda: self.video_patch_proj(video_latents.astype(param_dtype(self.video_patch_proj))),
            metadata={"rows": int(video_latents.shape[1])},
        )
        audio_embeds = _diagnostic_run(
            diagnostics,
            "input.audio_projection",
            lambda: self.audio_patch_proj(audio_latents.astype(param_dtype(self.audio_patch_proj))),
            metadata={"rows": int(audio_latents.shape[1])},
        )
        text = _diagnostic_run(
            diagnostics,
            "input.text_projection",
            lambda: self.condition_proj(text_embeds.astype(param_dtype(self.condition_proj))),
            metadata={"rows": int(text_embeds.shape[1])},
        )
        text = self.token_refiner(text, diagnostics=diagnostics)

        B = text.shape[0]

        def scatter_inputs():
            packed = mx.zeros((B, seq_len, text.shape[-1]), dtype=text.dtype)
            packed[:, text_indices] = text
            packed[:, video_indices] = video_embeds.astype(text.dtype)
            packed[:, audio_indices] = audio_embeds.astype(text.dtype)
            return packed

        x = _diagnostic_run(
            diagnostics,
            "input.scatter",
            scatter_inputs,
            metadata={"sequence_rows": int(seq_len)},
        )

        # 2. One timestep embedding per distinct noise level, shared by all AdaLN projections.
        temb = _diagnostic_run(
            diagnostics,
            "input.timestep_embedding",
            lambda: self.embed_timesteps(timestep),
            metadata={"distinct_timesteps": int(timestep.shape[0])},
        )

        # 3. Row -> AdaLN table row. `maximum(tags, 0)` mirrors the reference clamp: padding rows
        #    carry tag -1 and must not index backwards. They never reach the outputs.
        adaln_indices = timestep_indices * MODALITY_NUM + mx.maximum(token_tags, 0)
        return x, temb, adaln_indices, rotary

    def skip_indicator(self, x: mx.array, adaln_indices: mx.array, modulation) -> mx.array:
        """The block-0 AdaLN-modulated input — the quantity a TeaCache-style skip test watches.

        The modulation carries the timestep, so this tensor moves both with the trajectory and with
        the schedule; its step-to-step relative L1 is the standard proxy for how much the stack's
        output would have moved.
        """
        block = self.blocks[0]
        shift_msa, scale_msa = modulation[0], modulation[1]
        h = block.norm1(x)
        return h * (1.0 + scale_msa[adaln_indices]) + shift_msa[adaln_indices]

    def __call__(
        self,
        video_latents: mx.array,
        audio_latents: mx.array,
        text_embeds: mx.array,
        timestep: mx.array,
        timestep_indices: mx.array,
        token_tags: mx.array,
        position_ids: mx.array,
        video_indices: mx.array,
        audio_indices: mx.array,
        text_indices: mx.array,
        modulation_cache: ModulationCache | None = None,
        mask: mx.array | None = None,
        blockcache=None,
        easycache_core=None,
        trajectory_forecast=None,
        forecast_coordinate: float | None = None,
        trajectory_video_row_start: int = 0,
        trajectory_audio_row_start: int = 0,
        terminal_target_only: bool = False,
        terminal_video_row_start: int = 0,
        terminal_audio_row_start: int = 0,
        step_index: int = 0,
        total_steps: int = 1,
        diagnostics=None,
    ) -> tuple[mx.array, mx.array]:
        """Predict the video and audio velocity for one packed sequence.

        Args:
            video_latents: ``(B, num_video_tokens, video_patch_dim)`` patchified rows, ordered to
                match ``video_indices`` (conditioning rows included).
            audio_latents: ``(B, num_audio_tokens, audio_latents_dim)`` ordered as
                ``audio_indices``.
            text_embeds: ``(B, num_text_tokens, text_dim)`` ordered as ``text_indices``.
            timestep: ``(num_timesteps,)`` the *distinct* noise levels present, unscaled in [0, 1].
            timestep_indices: ``(seq_len,)`` index into ``timestep`` for every row.
            token_tags: ``(seq_len,)`` modality per row (0 video, 1 text, 2 audio, -1 padding).
            position_ids: ``(seq_len, 3)`` the ``(t, h, w)`` rotary coordinates per row.
            modulation_cache: optional precomputed AdaLN table; when supplied the ``adaln_proj``
                weights are never read, which is what lets them be dropped at inference time.

        Returns:
            ``(video_velocity, audio_velocity)`` in the row order of ``video_indices`` /
            ``audio_indices``.
        """
        if trajectory_forecast is not None and forecast_coordinate is not None:
            predicted = trajectory_forecast.try_predict(
                forecast_coordinate, step_index, total_steps
            )
            if predicted is not None:
                temb = self.embed_timesteps(timestep)
                video_result, audio_result = self._project_target_features(
                    predicted[0],
                    predicted[1],
                    temb,
                    timestep_indices[video_indices[trajectory_video_row_start:]],
                    timestep_indices[audio_indices[trajectory_audio_row_start:]],
                )
                if trajectory_video_row_start:
                    video_result = mx.concatenate(
                        [
                            mx.zeros(
                                (
                                    video_result.shape[0],
                                    trajectory_video_row_start,
                                    video_result.shape[2],
                                ),
                                dtype=video_result.dtype,
                            ),
                            video_result,
                        ],
                        axis=1,
                    )
                if trajectory_audio_row_start:
                    audio_result = mx.concatenate(
                        [
                            mx.zeros(
                                (
                                    audio_result.shape[0],
                                    trajectory_audio_row_start,
                                    audio_result.shape[2],
                                ),
                                dtype=audio_result.dtype,
                            ),
                            audio_result,
                        ],
                        axis=1,
                    )
                return video_result, audio_result

        x, temb, adaln_indices, rotary = self.pack_inputs(
            video_latents,
            audio_latents,
            text_embeds,
            timestep,
            timestep_indices,
            token_tags,
            position_ids,
            video_indices,
            audio_indices,
            text_indices,
            diagnostics,
        )
        core_input = x
        if easycache_core is not None and easycache_core.last_was_core_reuse:
            x = easycache_core.reuse_core(x)
            return self._project_packed_features(
                x,
                temb,
                timestep_indices,
                video_indices,
                audio_indices,
                diagnostics,
            )

        paged = getattr(self, "paged_blocks", None)
        if terminal_target_only and (
            paged is not None or blockcache is not None or trajectory_forecast is not None
        ):
            raise ValueError(
                "Terminal target-only research requires resident dense transformer blocks."
            )
        target_video_indices = None
        target_audio_indices = None
        terminal_query_indices = None
        if terminal_target_only:
            target_video_indices = video_indices[terminal_video_row_start:]
            target_audio_indices = audio_indices[terminal_audio_row_start:]
            terminal_query_indices = mx.concatenate((target_video_indices, target_audio_indices))
        sol_config = self.sol_attention_config
        if sol_config is not None:
            # H3 packs target video last. Everything before its first row is the multimodal
            # prefix and must remain an exact K/V sink with exact query rows.
            prefix_rows = int(video_indices[terminal_video_row_start].item())
            active_config = sol_config.with_prefix(prefix_rows)
            self.last_sol_attention_config = active_config
            for block in self.blocks:
                block.attn.sol_config = active_config
                block.attn.sol_step_index = step_index
                block.attn.sol_total_steps = total_steps
            paged_context = getattr(self, "paged_blocks", None)
            if paged_context is not None:
                paged_context.sol_config = active_config
                paged_context.sol_step_index = step_index
                paged_context.sol_total_steps = total_steps
        if paged is not None:
            if blockcache is not None and hasattr(blockcache, "segment_start"):
                raise ValueError(
                    "Hierarchical BlockCache is not yet compatible with paged H3 weights. "
                    "Use standard BlockCache or disconnect the cache node."
                )
            x, before_block_zero, after_block_zero = self._run_paged_blocks(
                x,
                temb,
                adaln_indices,
                rotary,
                video_indices,
                audio_indices,
                modulation_cache,
                mask,
                blockcache,
                step_index,
                total_steps,
                diagnostics,
            )
        elif blockcache is not None and hasattr(blockcache, "segment_start"):
            blockcache.begin_step()
            i = 0
            current_segment = None
            before_anchor = None
            after_anchor = None
            while i < len(self.blocks):
                segment_index = blockcache.segment_start(i)
                if segment_index is not None:
                    current_segment = segment_index
                    before_anchor = x
                block = self.blocks[i]
                if diagnostics is not None:
                    diagnostics.prepare_block(x, i)
                modulation = (
                    modulation_cache.get(i)
                    if modulation_cache is not None
                    else block.adaln_proj(temb)
                )
                x = block(
                    x,
                    modulation,
                    adaln_indices,
                    rotary,
                    mask,
                    diagnostics=diagnostics,
                    block_index=i,
                )
                if segment_index is not None:
                    after_anchor = x
                    reused = blockcache.try_reuse_segment(
                        segment_index,
                        before_anchor,
                        after_anchor,
                        video_indices,
                        audio_indices,
                        step_index,
                        total_steps,
                    )
                    if reused is not None:
                        x = reused
                        i = blockcache.segment_end(segment_index) + 1
                        continue
                if current_segment is not None and i == blockcache.segment_end(current_segment):
                    blockcache.update_segment(
                        current_segment,
                        before_anchor,
                        after_anchor,
                        x,
                        video_indices,
                        audio_indices,
                    )
                    current_segment = None
                    before_anchor = None
                    after_anchor = None
                i += 1
        else:
            before_block_zero = x
            for i, block in enumerate(self.blocks):
                if diagnostics is not None:
                    diagnostics.prepare_block(x, i)
                modulation = (
                    modulation_cache.get(i)
                    if modulation_cache is not None
                    else block.adaln_proj(temb)
                )
                if terminal_target_only and i == len(self.blocks) - 1:
                    x = block.selected_terminal_queries(
                        x,
                        modulation,
                        adaln_indices,
                        rotary,
                        terminal_query_indices,
                        mask,
                    )
                else:
                    x = block(
                        x,
                        modulation,
                        adaln_indices,
                        rotary,
                        mask,
                        diagnostics=diagnostics,
                        block_index=i,
                    )
                if i == 0 and blockcache is not None:
                    after_block_zero = x
                    reused = blockcache.try_reuse(
                        before_block_zero,
                        after_block_zero,
                        video_indices,
                        audio_indices,
                        step_index,
                        total_steps,
                    )
                    if reused is not None:
                        x = reused
                        break

        if (
            blockcache is not None
            and not hasattr(blockcache, "segment_start")
            and not blockcache.last_was_hit
        ):
            blockcache.update(
                before_block_zero,
                after_block_zero,
                x,
                video_indices,
                audio_indices,
            )

        if easycache_core is not None:
            easycache_core.update_core(core_input, x)

        if trajectory_forecast is not None and forecast_coordinate is not None:
            trajectory_forecast.update(
                forecast_coordinate,
                x[:, video_indices[trajectory_video_row_start:]],
                x[:, audio_indices[trajectory_audio_row_start:]],
            )

        if terminal_target_only:
            num_target_video = int(target_video_indices.shape[0])
            video_hidden = x[:, :num_target_video]
            audio_hidden = x[:, num_target_video:]
            video = self.final_layer.norm_out(
                video_hidden,
                temb,
                timestep_indices[target_video_indices],
            )
            audio = self.final_layer.norm_out(
                audio_hidden,
                temb,
                timestep_indices[target_audio_indices],
            )
            video_result = self.final_layer.video_out(
                video.astype(param_dtype(self.final_layer.video_out))
            )
            # The narrow audio head can select a different Metal GEMM kernel when its row count
            # shrinks. Filler rows restore the original M dimension and bit-exact target audio;
            # each output row is independent, and this projection is tiny relative to one block.
            audio_rows = int(audio.shape[1])
            if audio_rows < int(timestep_indices.shape[0]):
                audio = mx.concatenate(
                    (
                        audio,
                        mx.zeros(
                            (
                                audio.shape[0],
                                int(timestep_indices.shape[0]) - audio_rows,
                                audio.shape[2],
                            ),
                            dtype=audio.dtype,
                        ),
                    ),
                    axis=1,
                )
            audio_result = self.final_layer.audio_out(
                audio.astype(param_dtype(self.final_layer.audio_out))
            )[:, :audio_rows]
            if terminal_video_row_start:
                video_result = mx.concatenate(
                    (
                        mx.zeros(
                            (
                                video_result.shape[0],
                                terminal_video_row_start,
                                video_result.shape[2],
                            ),
                            dtype=video_result.dtype,
                        ),
                        video_result,
                    ),
                    axis=1,
                )
            if terminal_audio_row_start:
                audio_result = mx.concatenate(
                    (
                        mx.zeros(
                            (
                                audio_result.shape[0],
                                terminal_audio_row_start,
                                audio_result.shape[2],
                            ),
                            dtype=audio_result.dtype,
                        ),
                        audio_result,
                    ),
                    axis=1,
                )
            return video_result, audio_result

        return self._project_packed_features(
            x,
            temb,
            timestep_indices,
            video_indices,
            audio_indices,
            diagnostics,
        )

    def _project_packed_features(
        self,
        x,
        temb,
        timestep_indices,
        video_indices,
        audio_indices,
        diagnostics=None,
    ):
        """Run current-timestep output normalization and heads over packed features."""
        x = _diagnostic_run(
            diagnostics,
            "output.norm",
            lambda: self.final_layer.norm_out(x, temb, timestep_indices),
            metadata={"sequence_rows": int(x.shape[1])},
        )
        video_out = _diagnostic_run(
            diagnostics,
            "output.video_projection",
            lambda: self.final_layer.video_out(x.astype(param_dtype(self.final_layer.video_out))),
            metadata={"sequence_rows": int(x.shape[1])},
        )
        audio_out = _diagnostic_run(
            diagnostics,
            "output.audio_projection",
            lambda: self.final_layer.audio_out(x.astype(param_dtype(self.final_layer.audio_out))),
            metadata={"sequence_rows": int(x.shape[1])},
        )
        video_result = video_out[:, video_indices]
        audio_result = audio_out[:, audio_indices]
        if diagnostics is not None:
            diagnostics.capture("video_output", video_result)
            diagnostics.capture("audio_output", audio_result)
        return video_result, audio_result

    def _run_paged_blocks(
        self,
        x,
        temb,
        adaln_indices,
        rotary,
        video_indices,
        audio_indices,
        modulation_cache,
        mask,
        blockcache,
        step_index,
        total_steps,
        diagnostics,
    ):
        """Run bounded weight windows and complete activations before retiring each window."""
        paged = self.paged_blocks
        before_block_zero = x
        after_block_zero = None
        cache_hit = False
        for start in range(0, paged.num_blocks, paged.window_size):
            with paged.window(start) as blocks:
                for offset, block in enumerate(blocks):
                    index = start + offset
                    if diagnostics is not None:
                        diagnostics.prepare_block(x, index)
                    modulation = (
                        modulation_cache.get(index)
                        if modulation_cache is not None
                        else block.adaln_proj(temb)
                    )
                    x = block(
                        x,
                        modulation,
                        adaln_indices,
                        rotary,
                        mask,
                        diagnostics=diagnostics,
                        block_index=index,
                    )
                    if index == 0:
                        after_block_zero = x
                        if blockcache is not None:
                            reused = blockcache.try_reuse(
                                before_block_zero,
                                after_block_zero,
                                video_indices,
                                audio_indices,
                                step_index,
                                total_steps,
                            )
                            if reused is not None:
                                x = reused
                                cache_hit = True
                                break
                # MLX is lazy: this is the lifetime boundary that prevents retired page weights
                # from remaining captured by the hidden-state graph.
                mx.eval(x)
            if cache_hit:
                break
        if after_block_zero is None:
            raise RuntimeError("Paged H3 execution did not evaluate transformer block zero.")
        return x, before_block_zero, after_block_zero

    def _project_target_features(
        self,
        video_hidden,
        audio_hidden,
        temb,
        video_timestep_indices,
        audio_timestep_indices,
    ):
        """Run current exact output modulation and heads over compact target features."""
        video = self.final_layer.norm_out(video_hidden, temb, video_timestep_indices)
        audio = self.final_layer.norm_out(audio_hidden, temb, audio_timestep_indices)
        video = self.final_layer.video_out(video.astype(param_dtype(self.final_layer.video_out)))
        audio = self.final_layer.audio_out(audio.astype(param_dtype(self.final_layer.audio_out)))
        return video, audio
