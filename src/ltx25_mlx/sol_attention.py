"""LTX 2.5 video-self-attention adapter for the shared MLX Sol kernel."""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
from ltx_core_mlx.model.transformer.attention import Attention
from ltx_core_mlx.model.transformer.rope import apply_rope_interleaved, apply_rope_split

from wee_todd_mlx.execution_evidence import ExecutionEvidence
from wee_todd_mlx.sol_attention import (
    SolAttentionConfig,
    sol_attention,
    supports_sol_attention,
)


@dataclass
class _LTX25SolState:
    config: SolAttentionConfig
    evidence: ExecutionEvidence
    step_index: int = 0
    total_steps: int = 1
    exact_suffix_rows: int = 0
    bf16_projection_cast_calls: int = 0
    observed_projected_dtype: str | None = None
    observed_kernel_dtype: str | None = None


class _LTX25SolVideoAttention(Attention):
    """Preserve the upstream parameter layout while replacing eligible attention calls."""

    def __call__(
        self,
        x: mx.array,
        encoder_hidden_states: mx.array | None = None,
        rope_freqs: mx.array | None = None,
        rope_freqs_k: mx.array | None = None,
        attention_mask: mx.array | None = None,
        perturbation_mask: mx.array | None = None,
    ) -> mx.array:
        state = self._weetodd_sol_state
        evidence = state.evidence
        evidence.record_call()
        config = state.config.with_exact_rows(suffix=state.exact_suffix_rows)
        if (
            encoder_hidden_states is not None
            or attention_mask is not None
            or perturbation_mask is not None
        ):
            reason = (
                "non_self_attention"
                if encoder_hidden_states is not None
                else "attention_mask"
                if attention_mask is not None
                else "perturbation_mask"
            )
            evidence.record_fallback(reason)
            return super().__call__(
                x,
                encoder_hidden_states=encoder_hidden_states,
                rope_freqs=rope_freqs,
                rope_freqs_k=rope_freqs_k,
                attention_mask=attention_mask,
                perturbation_mask=perturbation_mask,
            )
        if not config.active(
            step_index=state.step_index,
            total_steps=state.total_steps,
            block_index=self._weetodd_sol_block_index,
        ):
            evidence.record_dense_policy()
            return super().__call__(
                x,
                encoder_hidden_states=encoder_hidden_states,
                rope_freqs=rope_freqs,
                rope_freqs_k=rope_freqs_k,
                attention_mask=attention_mask,
                perturbation_mask=perturbation_mask,
            )

        batch, tokens, _ = x.shape
        q = self.q_norm(self.to_q(x))
        k = self.k_norm(self.to_k(x))
        v = self.to_v(x)
        q = q.reshape(batch, tokens, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(batch, tokens, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(batch, tokens, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        if self.use_rope and rope_freqs is not None:
            cos_f, sin_f, rope_type = rope_freqs
            apply_rope = apply_rope_split if rope_type == "split" else apply_rope_interleaved
            q = apply_rope(q, cos_f, sin_f)
            k = apply_rope(k, cos_f, sin_f)

        projected_dtype = str(q.dtype)
        state.observed_projected_dtype = projected_dtype
        evidence.record_observed(dtype=q.dtype, shape=q.shape)
        if q.dtype == mx.float32 and k.dtype == mx.float32 and v.dtype == mx.float32:
            q = q.astype(mx.bfloat16)
            k = k.astype(mx.bfloat16)
            v = v.astype(mx.bfloat16)
            state.bf16_projection_cast_calls += 1
        state.observed_kernel_dtype = str(q.dtype)
        evidence.record_observed(dtype=q.dtype, shape=q.shape)

        if not supports_sol_attention(q, None, config):
            reason = "unsupported_dtype" if q.dtype != mx.bfloat16 else "unsupported_shape"
            evidence.record_fallback(reason)
            out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale)
        else:
            evidence.record_eligible()
            evidence.record_executed(work_units=int(q.shape[-2]))
            out = sol_attention(q, k, v, scale=self.scale, config=config)

        if self.to_gate_logits is not None:
            gate = 2.0 * mx.sigmoid(self.to_gate_logits(x))
            out = out * gate.transpose(0, 2, 1)[:, :, :, None]
        out = out.transpose(0, 2, 1, 3).reshape(
            batch,
            tokens,
            self.num_heads * self.head_dim,
        )
        return self.to_out(out)


def configure_ltx25_sol_attention(model, config: SolAttentionConfig | None) -> dict[str, object]:
    """Install the adapter on resident LTX video self-attention modules."""

    if config is None or not config.enabled:
        return {"enabled": False, "patched_video_self_attention": 0}
    config.validate()
    if getattr(model, "_streamer", None) is not None or getattr(model, "_shared_blocks", None):
        raise ValueError(
            "LTX 2.5 MLX Sol Attention currently requires a resident transformer. "
            "Disable low-RAM streaming for this experimental backend."
        )
    inner = getattr(model, "inner", model)
    blocks = getattr(inner, "transformer_blocks", None)
    if blocks is None:
        raise ValueError("The LTX 2.5 transformer has no resident block stack.")
    report: dict[str, object] = {
        "enabled": True,
        "patched_video_self_attention": len(blocks),
        "min_tokens": config.min_tokens,
        "tau": config.tau,
        "start_percent": config.start_percent,
        "end_percent": config.end_percent,
        "dense_blocks": config.dense_blocks,
        "scope": "unmasked resident video self-attention only",
    }
    evidence = ExecutionEvidence(
        requested_backend="sol_attention",
        resolved_backend="mlx_fused_sol_bf16",
        scope="unmasked resident video self-attention only",
    )
    state = _LTX25SolState(config=config, evidence=evidence)
    for block_index, block in enumerate(blocks):
        attention = block.attn1
        if attention.head_dim != 128:
            raise ValueError("LTX 2.5 MLX Sol Attention requires 128-wide video heads.")
        attention.__class__ = _LTX25SolVideoAttention
        attention._weetodd_sol_state = state
        attention._weetodd_sol_block_index = block_index
    model._weetodd_sol_state = state
    return report


def ltx25_sol_attention_report(model, policy: dict[str, object] | None = None) -> dict[str, object]:
    """Return the resolved policy plus generation-scoped execution evidence."""

    state = getattr(model, "_weetodd_sol_state", None)
    if state is None:
        return dict(policy or {"enabled": False, "patched_video_self_attention": 0})
    evidence = state.evidence.snapshot()
    return {
        **(policy or {}),
        **evidence,
        "attention_calls": evidence["total_calls"],
        "sparse_kernel_calls": evidence["executed_calls"],
        "unsupported_fallback_calls": evidence["fallback_calls"],
        "bf16_projection_cast_calls": state.bf16_projection_cast_calls,
        "observed_projected_dtype": state.observed_projected_dtype,
        "observed_kernel_dtype": state.observed_kernel_dtype,
        "observed_query_tokens": (
            max((shape[-2] for shape in state.evidence.observed_shapes), default=None)
        ),
    }


def set_ltx25_sol_context(
    model,
    *,
    step_index: int,
    total_steps: int,
    exact_suffix_rows: int | None = None,
) -> None:
    """Update schedule and exact reference-row state before one transformer evaluation."""

    inner = getattr(model, "model", model)
    state = getattr(inner, "_weetodd_sol_state", None)
    if state is None:
        return
    state.step_index = int(step_index)
    state.total_steps = max(1, int(total_steps))
    if exact_suffix_rows is not None:
        state.exact_suffix_rows = max(0, int(exact_suffix_rows))


__all__ = [
    "configure_ltx25_sol_attention",
    "ltx25_sol_attention_report",
    "set_ltx25_sol_context",
]
