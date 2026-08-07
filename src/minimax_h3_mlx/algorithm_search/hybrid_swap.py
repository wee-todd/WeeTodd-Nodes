"""Research-only selective block substitution for trajectory experiments."""

from __future__ import annotations

import time

import mlx.core as mx
import mlx.nn as nn

from minimax_h3_mlx.dit import apply_rotary


class SelectiveHybridBlockController:
    """Fit two exact evaluations, then predict text/audio rows in one selected block."""

    def __init__(
        self,
        *,
        block_index: int,
        text_rows: int,
        audio_rows: int,
        apply_evaluation: int = 2,
    ) -> None:
        if min(block_index + 1, text_rows, audio_rows, apply_evaluation) < 1:
            raise ValueError("hybrid block configuration values must be positive")
        self.block_index = block_index
        self.text_rows = text_rows
        self.audio_rows = audio_rows
        self.predicted_rows = text_rows + audio_rows
        self.apply_evaluation = apply_evaluation
        self._previous_input: mx.array | None = None
        self._previous_output: mx.array | None = None
        self._parameters: list[tuple[mx.array, mx.array]] | None = None
        self._block_started: float | None = None
        self.history: list[dict] = []

    def _regions(self) -> tuple[slice, slice]:
        return slice(0, self.text_rows), slice(self.text_rows, self.predicted_rows)

    def observe(
        self,
        block_index: int,
        block_input: mx.array,
        block_output: mx.array,
        *,
        evaluation_index: int | None,
    ) -> None:
        if block_index != self.block_index or evaluation_index is None:
            return
        current_input = block_input[:, : self.predicted_rows].astype(mx.bfloat16)
        current_output = block_output[:, : self.predicted_rows].astype(mx.bfloat16)
        mx.eval(current_input, current_output)
        duration = (
            None
            if self._block_started is None
            else time.perf_counter() - self._block_started
        )
        peak_memory = int(mx.get_peak_memory()) if self._block_started is not None else None
        if self._previous_input is not None and self._previous_output is not None:
            parameters = []
            for region in self._regions():
                dx = current_input[:, region].astype(mx.float32) - self._previous_input[
                    :, region
                ].astype(mx.float32)
                dy = current_output[:, region].astype(mx.float32) - self._previous_output[
                    :, region
                ].astype(mx.float32)
                denominator = mx.maximum(mx.sum(dx * dx, axis=(0, 1)), 1e-12)
                scale = mx.sum(dx * dy, axis=(0, 1)) / denominator
                bias = mx.mean(dy - scale * dx, axis=(0, 1))
                parameters.append((scale.astype(mx.bfloat16), bias.astype(mx.bfloat16)))
            mx.eval(*(value for pair in parameters for value in pair))
            self._parameters = parameters
        self._previous_input = current_input
        self._previous_output = current_output
        self.history.append(
            {
                "evaluation": evaluation_index,
                "action": "observe_exact",
                "block": block_index,
                "duration_seconds": duration,
                "peak_memory_bytes": peak_memory,
            }
        )
        self._block_started = None

    def _predict_prefix(self, x: mx.array) -> mx.array:
        if (
            self._parameters is None
            or self._previous_input is None
            or self._previous_output is None
        ):
            raise RuntimeError("hybrid block requires two observed exact evaluations")
        pieces = []
        for region, (scale, bias) in zip(self._regions(), self._parameters, strict=True):
            current = x[:, region].astype(mx.bfloat16)
            delta = current - self._previous_input[:, region]
            pieces.append(self._previous_output[:, region] + scale * delta + bias)
        return mx.concatenate(pieces, axis=1)

    def _exact_video(
        self,
        block,
        x: mx.array,
        *,
        modulation,
        adaln_indices: mx.array,
        rotary,
        mask,
    ) -> mx.array:
        prefix = self.predicted_rows
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = modulation
        h = block.norm1(x)
        h = h * (1.0 + scale_msa[adaln_indices]) + shift_msa[adaln_indices]
        attention = block.attn
        heads, head_dim = attention.heads, attention.head_dim
        weight = attention.qkv_proj.weight.reshape(heads, 3, head_dim, x.shape[-1])
        q_weight = weight[:, 0].reshape(heads * head_dim, x.shape[-1])
        kv_weight = weight[:, 1:].reshape(2 * heads * head_dim, x.shape[-1])
        q = (h[:, prefix:] @ q_weight.T).reshape(
            x.shape[0], x.shape[1] - prefix, heads, head_dim
        )
        kv = (h @ kv_weight.T).reshape(x.shape[0], x.shape[1], heads, 2, head_dim)
        k, v = kv[:, :, :, 0], kv[:, :, :, 1]
        q = attention.q_norm(q).transpose(0, 2, 1, 3)
        k = attention.k_norm(k).transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)
        if rotary is not None:
            q = apply_rotary(q, rotary[0][prefix:], rotary[1][prefix:])
            k = apply_rotary(k, *rotary)
        video_mask = mask
        if mask is not None and int(mask.shape[-2]) == int(x.shape[1]):
            video_mask = mask[..., prefix:, :]
        attended = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=attention.scale, mask=video_mask
        )
        attended = attended.transpose(0, 2, 1, 3).reshape(
            x.shape[0], x.shape[1] - prefix, heads * head_dim
        )
        attended = attention.out_proj(attended.astype(x.dtype))
        indices = adaln_indices[prefix:]
        video = x[:, prefix:] + gate_msa[indices] * attended
        h_video = block.norm2(video)
        h_video = h_video * (1.0 + scale_mlp[indices]) + shift_mlp[indices]
        fused = block.mlp.fc1(h_video)
        gate, value = fused[..., : block.mlp._ffn], fused[..., block.mlp._ffn :]
        mlp = block.mlp.fc2(nn.silu(gate) * value)
        return video + gate_mlp[indices] * mlp

    def try_apply(
        self,
        block,
        block_index: int,
        x: mx.array,
        *,
        evaluation_index: int | None,
        modulation,
        adaln_indices,
        rotary,
        mask,
    ) -> mx.array | None:
        if block_index != self.block_index:
            return None
        mx.eval(x)
        mx.reset_peak_memory()
        started = time.perf_counter()
        self._block_started = started
        if evaluation_index != self.apply_evaluation:
            return None
        predicted = self._predict_prefix(x)
        exact_video = self._exact_video(
            block,
            x,
            modulation=modulation,
            adaln_indices=adaln_indices,
            rotary=rotary,
            mask=mask,
        )
        result = mx.concatenate([predicted, exact_video], axis=1)
        mx.eval(result)
        self.history.append(
            {
                "evaluation": evaluation_index,
                "action": "apply_hybrid",
                "block": block_index,
                "duration_seconds": time.perf_counter() - started,
                "peak_memory_bytes": int(mx.get_peak_memory()),
            }
        )
        self._block_started = None
        return result
