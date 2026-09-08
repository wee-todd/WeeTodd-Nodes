"""MLX reference runtime for OpenVDN's MiniMax-H3 hybrid attention.

Independent MLX implementation of the published model contract, with grouped window
attention and a verified FP32 Metal solver for the linear recurrence.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn


@dataclass(frozen=True)
class VDNLayout:
    sequence: int
    video_start: int
    frames: int
    tokens_per_frame: int
    frame_height: int
    frame_width: int
    text_start: int
    text_length: int

    @property
    def video_end(self) -> int:
        return self.video_start + self.frames * self.tokens_per_frame


def _linear(value, weight, bias=None):
    result = value.astype(weight.dtype) @ weight.T
    if bias is not None:
        result = result + bias
    return result


def _l2_normalize(value):
    scale = mx.rsqrt(
        mx.maximum(mx.sum(value.astype(mx.float32) ** 2, axis=-1, keepdims=True), 1e-12)
    )
    return (value * scale.astype(value.dtype)).astype(value.dtype)


def _frame_alpha(frame_mean, weights, heads, head_dim):
    # The released recurrence explicitly promotes BOTH operands before either
    # projection. Downcasting the mean to BF16 first irreversibly changes retention.
    hidden = (
        frame_mean.astype(mx.float32)
        @ weights["linear_attention.alpha.down.weight"].astype(mx.float32).T
    )
    delta = hidden @ weights["linear_attention.alpha.up.weight"].astype(mx.float32).T
    delta = delta + weights["linear_attention.alpha.dt_bias"].astype(mx.float32)
    delta = delta.reshape(-1, heads, head_dim)
    scale = mx.exp(weights["linear_attention.alpha.A_log"].astype(mx.float32))[None, :, None]
    return mx.exp(-scale * mx.logaddexp(delta, 0.0))


def _rms_norm(value, weight, eps=1e-6):
    mean_square = mx.mean(value.astype(mx.float32) ** 2, axis=-1, keepdims=True)
    return value * mx.rsqrt(mean_square + eps).astype(value.dtype) * weight.astype(value.dtype)


def _window_bounds(frames: int) -> tuple[tuple[int, int], ...]:
    # The release uses chunk-aligned c1 windows with five latent frames per chunk.
    return tuple((((frame // 5) - 1) * 5, ((frame // 5) + 2) * 5 - 1) for frame in range(frames))


def _temporal_reference(spatial, temporal_weight):
    frames = spatial.shape[0]
    temporal_padded = mx.pad(spatial, ((2, 2), (0, 0), (0, 0), (0, 0)))
    temporal = mx.zeros_like(spatial)
    for offset in range(5):
        coefficient = temporal_weight[:, 0, offset].astype(spatial.dtype)
        temporal = temporal + temporal_padded[offset : offset + frames] * coefficient
    return temporal


def _depthwise_short_conv(
    tokens, spatial_weight, temporal_weight, layout: VDNLayout, *, feature_kernels=None
):
    channels = int(tokens.shape[-2] * tokens.shape[-1])
    volume = tokens.reshape(layout.frames, layout.frame_height, layout.frame_width, channels)
    # Match the published grouped Conv2d: accumulate spatial taps in the native
    # convolution, not a BF16 elementwise chain that rounds after every tap.
    spatial = mx.conv2d(
        volume,
        spatial_weight.transpose(0, 2, 3, 1).astype(tokens.dtype),
        padding=2,
        groups=channels,
    )
    temporal = (
        _temporal_reference(spatial, temporal_weight)
        if feature_kernels is None
        else feature_kernels.temporal(spatial, temporal_weight, _temporal_reference)
    )
    return temporal.reshape(-1, tokens.shape[-2], tokens.shape[-1])


class H3VDNRuntime:
    """File-backed VDN branch weights plus the active packed-sequence layout."""

    def __init__(
        self,
        checkpoint: str | Path,
        model_spec: str | Path,
        linear_branch: str | Path,
        *,
        inference_backend="verified",
    ):
        self.checkpoint = str(Path(checkpoint).expanduser())
        payload = json.loads(Path(model_spec).read_text(encoding="utf-8"))
        transform = next(
            item for item in payload["transforms"] if item["type"] == "hybrid_attention"
        )
        self.config = transform["config"]
        self.weights = dict(mx.load(str(Path(linear_branch).expanduser())))
        self.layout: VDNLayout | None = None
        self.calls = 0
        from .vdn_metal import VDNFeatureKernels, VDNMatrixSolver

        self.solver = VDNMatrixSolver()
        self.feature_kernels = VDNFeatureKernels()
        self._configure_inference(inference_backend)

    def _configure_inference(self, inference_backend):
        if inference_backend not in {"reference", "verified", "indexed_experimental"}:
            raise ValueError("Unknown VDN inference backend.")
        from .vdn_inference import VDNInferenceKernels

        self.inference_backend = inference_backend
        self.inference = VDNInferenceKernels(enabled=inference_backend != "reference")
        self._blocks = {}
        self.indexed_verified = set()
        self.indexed_fallback = None
        self.indexed_calls = 0

    def begin_run(self) -> None:
        """Reset execution counters while preserving warm verification/fallback verdicts."""
        self.calls = 0
        self.solver.metal_calls = 0
        self.solver.cpu_calls = 0
        self.feature_kernels.metal_calls = 0
        self.feature_kernels.reference_calls = 0
        self.inference.begin_run()
        self.indexed_calls = 0

    def set_layout(self, video_indices, text_indices, position_ids) -> None:
        video = video_indices.tolist()
        text = text_indices.tolist()
        if not video or video != list(range(video[0], video[0] + len(video))):
            raise ValueError("VDN-H3 requires contiguous target-video rows.")
        positions = position_ids[video_indices].tolist()
        frame_values = sorted({float(row[0]) for row in positions})
        height_values = sorted({float(row[1]) for row in positions})
        width_values = sorted({float(row[2]) for row in positions})
        frames = len(frame_values)
        frame_height = len(height_values)
        frame_width = len(width_values)
        per_frame = frame_height * frame_width
        if frames * per_frame != len(video):
            raise ValueError("VDN-H3 target-video rows do not form one dense temporal grid.")
        expected_positions = [
            [t, h, w] for t in frame_values for h in height_values for w in width_values
        ]
        if positions != expected_positions:
            raise ValueError("VDN-H3 target-video rows must use dense frame-major grid order.")
        if not text or text != list(range(text[0], text[0] + len(text))):
            raise ValueError("VDN-H3 requires contiguous prompt rows.")
        self.layout = VDNLayout(
            sequence=int(position_ids.shape[0]),
            video_start=video[0],
            frames=frames,
            tokens_per_frame=per_frame,
            frame_height=frame_height,
            frame_width=frame_width,
            text_start=text[0],
            text_length=len(text),
        )

    def block(self, index: int) -> dict[str, mx.array]:
        if index in self._blocks:
            return self._blocks[index]
        prefix = f"transformer_blocks.{index}.attn."
        self._blocks[index] = {
            key[len(prefix) :]: value
            for key, value in self.weights.items()
            if key.startswith(prefix)
        }
        return self._blocks[index]

    def report(self) -> dict[str, object]:
        return {
            "checkpoint": Path(self.checkpoint).name,
            "backend": (
                "mlx_metal_solve"
                if self.solver.metal_calls and not self.solver.cpu_calls
                else "mlx_cpu_solve_fallback"
            ),
            "solver": self.solver.report(),
            "feature_kernels": self.feature_kernels.report(),
            "inference_kernels": self.inference.report(),
            "inference_backend": self.inference_backend,
            "indexed_attention_calls": self.indexed_calls,
            "indexed_attention_fallback": self.indexed_fallback,
            "execution_counter_scope": "current_sampling_run",
            "verification_scope": "resident_runtime_lifetime",
            "lora_execution": "sequential_reference",
            "window_dispatch": "static_indexed" if self.indexed_calls else "grouped_equal_windows",
            "attention_calls": self.calls,
            "hybrid_attention_version": 2,
            "anchor_frames": self.config["anchor_frames"],
            "softmax_attention": self.config["softmax_attention"],
            "linear_attention": self.config["linear_attention"],
        }


def _softmax_branch(attn, x, q, k, v, weights, layout: VDNLayout):
    runtime = getattr(attn, "vdn_runtime", None)
    indexed = runtime is not None and runtime.inference_backend == "indexed_experimental"
    if indexed and runtime.indexed_fallback is None:
        try:
            from .vdn_attention import indexed_attention

            softmax = indexed_attention(q, k, v, layout, attn.scale)
            key = (layout, q.shape, q.dtype)
            if key not in runtime.indexed_verified:
                reference = _softmax_values(attn, q, k, v, layout)
                difference = softmax.astype(mx.float32) - reference.astype(mx.float32)
                relative = mx.sqrt(
                    mx.mean(difference**2)
                    / mx.maximum(mx.mean(reference.astype(mx.float32) ** 2), 1e-20)
                )
                valid = mx.all(mx.isfinite(softmax)) & (relative < 0.01)
                mx.eval(valid)
                if not valid.item():
                    raise ValueError("Indexed VDN attention failed numerical verification.")
                runtime.indexed_verified.add(key)
            runtime.indexed_calls += 1
        except (RuntimeError, ValueError) as exc:
            runtime.indexed_fallback = f"{type(exc).__name__}: {str(exc)[:240]}"
            softmax = _softmax_values(attn, q, k, v, layout)
    else:
        softmax = _softmax_values(attn, q, k, v, layout)
    softmax = softmax.transpose(0, 2, 1, 3)
    gate = mx.sigmoid(
        _linear(x[0], weights["softmax_gate.up.weight"], weights["softmax_gate.up.bias"])
    ).reshape(layout.sequence, attn.heads, 1)
    flat = (softmax[0] * gate).reshape(layout.sequence, attn.heads * attn.head_dim)
    return attn.out_proj(flat[None].astype(x.dtype))


def _softmax_values(attn, q, k, v, layout: VDNLayout):
    def attend(query, keys, values):
        head_chunk = getattr(attn, "head_chunk_size", None) or query.shape[1]
        row_chunk = getattr(attn, "query_chunk_size", None) or query.shape[2]
        groups = []
        for head in range(0, query.shape[1], head_chunk):
            rows = []
            for row in range(0, query.shape[2], row_chunk):
                rows.append(
                    mx.fast.scaled_dot_product_attention(
                        query[:, head : head + head_chunk, row : row + row_chunk],
                        keys[:, head : head + head_chunk],
                        values[:, head : head + head_chunk],
                        scale=attn.scale,
                    )
                )
            groups.append(mx.concatenate(rows, axis=2))
        return mx.concatenate(groups, axis=1)

    video_q = q[:, :, layout.video_start : layout.video_end]
    video_k = k[:, :, layout.video_start : layout.video_end]
    video_v = v[:, :, layout.video_start : layout.video_end]
    outputs = []
    if layout.video_start:
        outputs.append(attend(q[:, :, : layout.video_start], k, v))
    global_k = mx.concatenate((k[:, :, : layout.video_start], k[:, :, layout.video_end :]), axis=2)
    global_v = mx.concatenate((v[:, :, : layout.video_start], v[:, :, layout.video_end :]), axis=2)
    bounds = _window_bounds(layout.frames)
    frame = 0
    while frame < layout.frames:
        lower, upper = bounds[frame]
        stop_frame = frame + 1
        if frame not in (0, layout.frames - 1):
            while stop_frame < layout.frames - 1 and bounds[stop_frame] == (lower, upper):
                stop_frame += 1
        query = video_q[
            :, :, frame * layout.tokens_per_frame : stop_frame * layout.tokens_per_frame
        ]
        if frame in (0, layout.frames - 1):
            outputs.append(attend(query, k, v))
            frame = stop_frame
            continue
        lower = max(lower, 0)
        upper = min(upper, layout.frames - 1)
        # Each chunk shares exactly the same key rows. Preserve their ordering,
        # but gather the contiguous local window once instead of once per frame.
        start = lower * layout.tokens_per_frame
        stop = (upper + 1) * layout.tokens_per_frame
        key_parts = [global_k, video_k[:, :, start:stop]]
        value_parts = [global_v, video_v[:, :, start:stop]]
        for anchor in (0, layout.frames - 1):
            if not lower <= anchor <= upper:
                start = anchor * layout.tokens_per_frame
                stop = start + layout.tokens_per_frame
                key_parts.append(video_k[:, :, start:stop])
                value_parts.append(video_v[:, :, start:stop])
        outputs.append(
            attend(query, mx.concatenate(key_parts, axis=2), mx.concatenate(value_parts, axis=2))
        )
        frame = stop_frame
    if layout.video_end < layout.sequence:
        outputs.append(attend(q[:, :, layout.video_end :], k, v))
    return mx.concatenate(outputs, axis=2)


def _features(tokens, projection, weights, layout: VDNLayout, *, video: bool, feature_kernels=None):
    if video and projection in {"k", "v"}:
        tokens = _depthwise_short_conv(
            tokens,
            weights[f"linear_attention.short_conv.{projection}_sp.weight"],
            weights[f"linear_attention.short_conv.{projection}_tm.weight"],
            layout,
            feature_kernels=feature_kernels,
        )
    activated = nn.silu(tokens)
    return _l2_normalize(activated) if projection != "v" else activated


def _frame_statistics(key, value, beta):
    scaled_key = key.astype(mx.float32) * beta[..., None].astype(mx.float32)
    a = mx.matmul(scaled_key.transpose(0, 1, 3, 2), key.astype(mx.float32))
    a = 0.5 * (a + a.transpose(0, 1, 3, 2))
    b = mx.matmul((value * beta[..., None].astype(value.dtype)).transpose(0, 1, 3, 2), key).astype(
        mx.float32
    )
    return a, b


def _factor(a, b, alpha, *, solver=None):
    identity = mx.eye(a.shape[-1], dtype=mx.float32)
    matrix = a + identity
    inverse = mx.linalg.inv(matrix, stream=mx.cpu) if solver is None else solver.inverse(matrix)
    return alpha[..., None] * inverse, mx.matmul(b, inverse)


def _scan(transitions, injections, start):
    state = start
    prefix = []
    for frame in range(transitions.shape[0]):
        state = mx.matmul(state, transitions[frame]) + injections[frame]
        prefix.append(state)
    state = start
    suffix = [None] * transitions.shape[0]
    for frame in range(transitions.shape[0] - 1, -1, -1):
        state = mx.matmul(state, transitions[frame]) + injections[frame]
        suffix[frame] = state
    return mx.stack(prefix), mx.stack(suffix)


def _gather_state_reference(prefix, suffix, alpha, text_state, bounds):
    inner_frames = alpha.shape[0]
    log_prefix = mx.concatenate(
        (mx.zeros_like(alpha[:1]), mx.cumsum(mx.log(mx.maximum(alpha, 1e-12)), axis=0))
    )
    states = []
    for frame, (lower, upper) in enumerate(bounds):
        before = text_state if lower <= 0 else prefix[lower - 1]
        after = text_state if upper >= inner_frames - 1 else suffix[upper + 1]
        before_decay = mx.exp(log_prefix[frame + 1] - log_prefix[max(lower, 0)])
        after_decay = mx.exp(log_prefix[min(upper + 1, inner_frames)] - log_prefix[frame])
        states.append(before * before_decay[:, None, :] + after * after_decay[:, None, :])
    return mx.stack(states)


def _linear_branch(
    x, raw, weights, layout: VDNLayout, *, solver=None, feature_kernels=None, inference=None
):
    # Anchor frames are exact softmax rows/columns, so the linear complement owns only
    # frames 1..F-2 and its local bounds are rebased to that sliced sequence.
    if layout.frames <= 2:
        return mx.zeros((layout.video_end - layout.video_start, x.shape[-1]), dtype=x.dtype)
    per_frame = layout.tokens_per_frame
    inner = slice(layout.video_start + per_frame, layout.video_end - per_frame)
    inner_frames = layout.frames - 2
    inner_layout = VDNLayout(
        sequence=inner_frames * per_frame,
        video_start=0,
        frames=inner_frames,
        tokens_per_frame=per_frame,
        frame_height=layout.frame_height,
        frame_width=layout.frame_width,
        text_start=0,
        text_length=layout.text_length,
    )
    q_raw, k_raw, v_raw = (item[0, inner] for item in raw)
    query = _features(q_raw, "q", weights, inner_layout, video=True)
    key = _features(k_raw, "k", weights, inner_layout, video=True, feature_kernels=feature_kernels)
    value = _features(
        v_raw, "v", weights, inner_layout, video=True, feature_kernels=feature_kernels
    )
    shape = (inner_frames, per_frame, q_raw.shape[-2], q_raw.shape[-1])
    query_frame = query.reshape(shape)
    key_frame = key.reshape(shape).transpose(0, 2, 1, 3)
    value_frame = value.reshape(shape).transpose(0, 2, 1, 3)
    xv = x[0, inner]
    beta = mx.sigmoid(_linear(xv, weights["linear_attention.beta_proj.weight"]))
    beta = beta.reshape(inner_frames, per_frame, q_raw.shape[-2]).transpose(0, 2, 1)
    a, b = _frame_statistics(key_frame, value_frame, beta)

    frame_mean = mx.mean(xv.astype(mx.float32).reshape(inner_frames, per_frame, -1), axis=1)
    alpha = _frame_alpha(frame_mean, weights, q_raw.shape[-2], q_raw.shape[-1])

    text_slice = slice(layout.text_start, layout.text_start + layout.text_length)
    text_x = x[0, text_slice]
    text_key = _features(raw[1][0, text_slice], "k", weights, layout, video=False)
    text_value = _features(raw[2][0, text_slice], "v", weights, layout, video=False)
    text_beta = mx.sigmoid(_linear(text_x, weights["linear_attention.beta_proj.weight"]))
    text_key = text_key.transpose(1, 0, 2)[None]
    text_value = text_value.transpose(1, 0, 2)[None]
    text_beta = text_beta.transpose(1, 0)[None]
    text_a, text_b = _frame_statistics(text_key, text_value, text_beta)
    ones = mx.ones((1, q_raw.shape[-2], q_raw.shape[-1]), dtype=mx.float32)
    _, text_injection = _factor(text_a, text_b, ones, solver=solver)
    text_state = 0.5 * text_injection[0]

    transitions, injections = _factor(a, b, alpha, solver=solver)
    prefix, suffix = (
        _scan(transitions, injections, text_state)
        if inference is None
        else inference.scan(transitions, injections, text_state, _scan)
    )
    bounds = tuple((lo - 1, hi - 1) for lo, hi in _window_bounds(layout.frames)[1:-1])
    state = (
        _gather_state_reference(prefix, suffix, alpha, text_state, bounds)
        if inference is None
        else inference.gather(prefix, suffix, alpha, text_state, bounds, _gather_state_reference)
    ).astype(query.dtype)
    readout = mx.einsum("fhvk,fshk->fshv", state, query_frame)
    readout = _rms_norm(readout, weights["linear_attention.norm.weight"])
    gate_hidden = (
        _linear(xv, weights["linear_attention.output_gate.down.weight"])
        if inference is None
        else inference.project(xv, weights, "linear_attention.output_gate.down.weight")
    )
    gate = mx.sigmoid(
        _linear(
            gate_hidden,
            weights["linear_attention.output_gate.up.weight"],
            weights["linear_attention.output_gate.up.bias"],
        )
    ).reshape(inner_frames, per_frame, q_raw.shape[-2], q_raw.shape[-1])
    gated = (readout * gate).reshape(inner_frames * per_frame, -1)
    projected = (
        _linear(gated, weights["to_out_linear.weight"])
        if inference is None
        else inference.project(gated, weights, "to_out_linear.weight")
    )
    zeros = mx.zeros((per_frame, projected.shape[-1]), dtype=projected.dtype)
    return mx.concatenate((zeros, projected, zeros), axis=0)


def vdn_attention(attn, x, rotary, mask, block_index: int):
    runtime = attn.vdn_runtime
    layout = runtime.layout
    if layout is None:
        raise RuntimeError("VDN-H3 packed layout was not prepared before attention.")
    if mask is not None:
        raise ValueError("VDN-H3 MLX currently supports unmasked T2VA attention only.")
    weights = runtime.block(int(block_index))
    expected = 16
    if len(weights) != expected:
        raise ValueError(
            f"VDN-H3 block {block_index} has {len(weights)} branch tensors; expected {expected}."
        )
    batch, sequence, _ = x.shape
    if batch != 1 or sequence != layout.sequence:
        raise ValueError("VDN-H3 requires one sample matching the prepared packed layout.")
    qkv = attn.qkv_proj(x).reshape(batch, sequence, attn.heads, 3, attn.head_dim)
    q_raw, k_raw, v_raw = qkv[:, :, :, 0], qkv[:, :, :, 1], qkv[:, :, :, 2]
    q = attn.q_norm(q_raw).transpose(0, 2, 1, 3)
    k = attn.k_norm(k_raw).transpose(0, 2, 1, 3)
    v = v_raw.transpose(0, 2, 1, 3)
    if rotary is not None:
        from .dit import apply_rotary

        q = apply_rotary(q, *rotary)
        k = apply_rotary(k, *rotary)
    output = _softmax_branch(attn, x, q, k, v, weights, layout)
    linear = _linear_branch(
        x,
        (q_raw, k_raw, v_raw),
        weights,
        layout,
        solver=runtime.solver,
        feature_kernels=runtime.feature_kernels,
        inference=runtime.inference,
    )
    video = output[:, layout.video_start : layout.video_end] + linear[None].astype(output.dtype)
    output = mx.concatenate(
        (output[:, : layout.video_start], video, output[:, layout.video_end :]), axis=1
    )
    runtime.calls += 1
    return output


def load_vdn_runtime(request: dict[str, object]) -> H3VDNRuntime:
    return H3VDNRuntime(
        request["checkpoint"],
        request["model_spec"],
        request["linear_branch"],
        inference_backend=request.get("inference_backend", "verified"),
    )
