"""LTX 2.5 video-self-attention adapter for the shared MLX Sol kernel."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import mlx.core as mx
from ltx_core_mlx.model.transformer.attention import Attention
from ltx_core_mlx.model.transformer.rope import apply_rope_interleaved, apply_rope_split

from wee_todd_mlx.execution_evidence import ExecutionEvidence
from wee_todd_mlx.sol_attention import (
    SolAttentionConfig,
    route_telemetry_report,
    sol_attention,
    supports_sol_attention,
)


@dataclass
class _LTX25SolState:
    config: SolAttentionConfig
    evidence: ExecutionEvidence
    force_dense_bf16: bool = False
    step_index: int = 0
    total_steps: int = 1
    exact_suffix_rows: int = 0
    bf16_projection_cast_calls: int = 0
    observed_projected_dtype: str | None = None
    observed_kernel_dtype: str | None = None
    route_records: list[tuple[int, int, mx.array]] = field(default_factory=list)
    route_report: dict[str, object] | None = None
    streaming_compiled: bool = False
    streaming_casts_bf16: bool = False
    last_route_counts: mx.array | None = None
    reinstall_compiled: Callable[[], None] | None = None
    compiled_exact_suffix_rows: int = 0
    last_step_index: int | None = None


class _CompiledSolBlock:
    """Keep paged blocks compiled while surfacing fused route counters."""

    def __init__(self, block, state: _LTX25SolState) -> None:
        self.block = block
        self.state = state

        def run(*args, **kwargs):
            state.last_route_counts = mx.zeros((1, 1, 0, 2), dtype=mx.uint32)
            video, audio = block(*args, **kwargs)
            return video, audio, state.last_route_counts

        self.compiled = mx.compile(run, inputs=block)

    def __call__(self, *args, **kwargs):
        video, audio, route_counts = self.compiled(*args, **kwargs)
        block_index = int(kwargs.get("block_idx", 0))
        evidence = self.state.evidence
        evidence.record_call()
        if route_counts.size:
            evidence.record_eligible()
            evidence.record_executed(work_units=0)
            self.state.route_records.append(
                (self.state.step_index, block_index, route_counts)
            )
        else:
            evidence.record_fallback("unsupported_compiled_shape")
        if self.state.streaming_casts_bf16:
            self.state.bf16_projection_cast_calls += 1
        return video, audio


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
        if not state.streaming_compiled:
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
            if state.streaming_compiled:
                state.streaming_casts_bf16 = True
            else:
                state.bf16_projection_cast_calls += 1
        state.observed_kernel_dtype = str(q.dtype)
        evidence.record_observed(dtype=q.dtype, shape=q.shape)

        if state.force_dense_bf16:
            if not state.streaming_compiled:
                evidence.record_eligible()
                evidence.record_executed(work_units=0)
            out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale)
        elif not supports_sol_attention(q, None, config):
            reason = "unsupported_dtype" if q.dtype != mx.bfloat16 else "unsupported_shape"
            evidence.record_fallback(reason)
            out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale)
        else:
            if not state.streaming_compiled:
                evidence.record_eligible()
                evidence.record_executed(work_units=0)
            out, route_counts = sol_attention(
                q,
                k,
                v,
                scale=self.scale,
                config=config,
                return_route_counts=True,
            )
            if state.streaming_compiled:
                state.last_route_counts = route_counts
            else:
                state.route_records.append(
                    (state.step_index, self._weetodd_sol_block_index, route_counts)
                )

        if self.to_gate_logits is not None:
            gate = 2.0 * mx.sigmoid(self.to_gate_logits(x))
            out = out * gate.transpose(0, 2, 1)[:, :, :, None]
        out = out.transpose(0, 2, 1, 3).reshape(
            batch,
            tokens,
            self.num_heads * self.head_dim,
        )
        return self.to_out(out)


def configure_ltx25_sol_attention(
    model,
    config: SolAttentionConfig | None,
    *,
    force_dense_bf16: bool = False,
) -> dict[str, object]:
    """Install the adapter on resident LTX video self-attention modules."""

    if config is None or not config.enabled:
        return {"enabled": False, "patched_video_self_attention": 0}
    config.validate()
    try:
        shared_block = object.__getattribute__(model, "_shared_block")
    except AttributeError:
        shared_block = None
    try:
        shared_blocks = object.__getattribute__(model, "_shared_blocks")
    except AttributeError:
        shared_blocks = None
    streaming = shared_block is not None or bool(shared_blocks)
    if streaming and (
        config.start_percent != 0.0
        or config.end_percent != 1.0
        or config.dense_blocks != 0
    ):
        raise ValueError(
            "Compiled paged Sol Attention requires an always-active policy "
            "(start_percent=0, end_percent=1, dense_blocks=0)."
        )
    inner = getattr(model, "inner", model)
    blocks = tuple(shared_blocks or ((shared_block,) if shared_block is not None else ()))
    if not blocks:
        blocks = tuple(getattr(inner, "transformer_blocks", ()) or ())
    if not blocks:
        raise ValueError("The LTX 2.5 transformer has no resident block stack.")
    patched_count = (
        int(getattr(getattr(inner, "config", None), "num_layers", len(blocks)))
        if streaming
        else len(blocks)
    )
    report: dict[str, object] = {
        "enabled": True,
        "patched_video_self_attention": patched_count,
        "min_tokens": config.min_tokens,
        "tau": config.tau,
        "start_percent": config.start_percent,
        "end_percent": config.end_percent,
        "dense_blocks": config.dense_blocks,
        "scope": (
            "unmasked compiled paged video self-attention only"
            if streaming
            else "unmasked resident video self-attention only"
        ),
        "kernel": "dense_bf16_control" if force_dense_bf16 else "sol_sparse_bf16",
        "streaming_compiled": streaming,
    }
    evidence = ExecutionEvidence(
        requested_backend="sol_attention",
        resolved_backend=(
            "mlx_dense_bf16_control" if force_dense_bf16 else "mlx_fused_sol_bf16"
        ),
        scope=(
            "unmasked compiled paged video self-attention only"
            if streaming
            else "unmasked resident video self-attention only"
        ),
    )
    state = _LTX25SolState(
        config=config,
        evidence=evidence,
        force_dense_bf16=force_dense_bf16,
        streaming_compiled=streaming,
    )
    for block_index, block in enumerate(blocks):
        attention = block.attn1
        if attention.head_dim != 128:
            raise ValueError("LTX 2.5 MLX Sol Attention requires 128-wide video heads.")
        attention.__class__ = _LTX25SolVideoAttention
        attention._weetodd_sol_state = state
        attention._weetodd_sol_block_index = block_index
    if streaming:
        def reinstall_compiled() -> None:
            if shared_block is not None:
                object.__setattr__(
                    model,
                    "_compiled_block",
                    _CompiledSolBlock(shared_block, state),
                )
            else:
                object.__setattr__(
                    model,
                    "_compiled_blocks",
                    tuple(_CompiledSolBlock(block, state) for block in shared_blocks),
                )
            state.compiled_exact_suffix_rows = state.exact_suffix_rows

        state.reinstall_compiled = reinstall_compiled
        reinstall_compiled()
    model._weetodd_sol_state = state
    return report


def ltx25_sol_attention_report(model, policy: dict[str, object] | None = None) -> dict[str, object]:
    """Return the resolved policy plus generation-scoped execution evidence."""

    state = getattr(model, "_weetodd_sol_state", None)
    if state is None:
        return dict(policy or {"enabled": False, "patched_video_self_attention": 0})
    if state.route_report is None and state.route_records:
        state.route_report = route_telemetry_report(state.route_records)
        state.evidence.work_units_processed += int(
            state.route_report["processed_key_row_units"]
        )
        state.evidence.work_units_avoided += int(state.route_report["avoided_key_row_units"])
    evidence = state.evidence.snapshot()
    observed_query_tokens = max(
        (shape[-2] for shape in state.evidence.observed_shapes), default=None
    )
    return {
        **(policy or {}),
        **evidence,
        "attention_calls": evidence["total_calls"],
        "sparse_kernel_calls": 0 if state.force_dense_bf16 else evidence["executed_calls"],
        "dense_bf16_control_calls": (
            evidence["executed_calls"] if state.force_dense_bf16 else 0
        ),
        "unsupported_fallback_calls": evidence["fallback_calls"],
        "bf16_projection_cast_calls": state.bf16_projection_cast_calls,
        "dense_bf16_control": state.force_dense_bf16,
        "observed_projected_dtype": state.observed_projected_dtype,
        "observed_kernel_dtype": state.observed_kernel_dtype,
        "observed_query_tokens": observed_query_tokens,
        "exact_suffix_rows": state.exact_suffix_rows,
        "approximation_candidate_rows": (
            None
            if observed_query_tokens is None
            else max(0, int(observed_query_tokens) - state.exact_suffix_rows)
        ),
        "compiled_exact_suffix_rows": (
            state.compiled_exact_suffix_rows if state.streaming_compiled else None
        ),
        **(state.route_report or {}),
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
    next_step = int(step_index)
    if next_step == 0 and state.last_step_index not in {None, 0}:
        previous = state.evidence
        state.evidence = ExecutionEvidence(
            requested_backend=previous.requested_backend,
            resolved_backend=previous.resolved_backend,
            scope=previous.scope,
        )
        state.route_records.clear()
        state.route_report = None
        state.bf16_projection_cast_calls = 0
    state.step_index = next_step
    state.total_steps = max(1, int(total_steps))
    if exact_suffix_rows is not None:
        resolved_suffix = max(0, int(exact_suffix_rows))
        state.exact_suffix_rows = resolved_suffix
        if (
            state.streaming_compiled
            and resolved_suffix != state.compiled_exact_suffix_rows
            and state.reinstall_compiled is not None
        ):
            state.reinstall_compiled()
    state.last_step_index = next_step


__all__ = [
    "configure_ltx25_sol_attention",
    "ltx25_sol_attention_report",
    "set_ltx25_sol_context",
]
