"""MLX implementation of the official LTX 2.5 prompt-duration head."""

from __future__ import annotations

import json
import math
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from safetensors import safe_open


class _CrossAttention(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.in_proj_weight = mx.zeros((hidden_dim * 3, hidden_dim), dtype=mx.bfloat16)
        self.in_proj_bias = mx.zeros((hidden_dim * 3,), dtype=mx.bfloat16)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    def __call__(self, queries: mx.array, tokens: mx.array) -> mx.array:
        width = self.hidden_dim
        q_weight, k_weight, v_weight = mx.split(self.in_proj_weight, 3, axis=0)
        q_bias, k_bias, v_bias = mx.split(self.in_proj_bias, 3, axis=0)
        query = queries @ q_weight.T + q_bias
        key = tokens @ k_weight.T + k_bias
        value = tokens @ v_weight.T + v_bias
        batch, query_count, _ = query.shape
        token_count = key.shape[1]
        head_dim = width // self.num_heads
        query = query.reshape(batch, query_count, self.num_heads, head_dim).transpose(0, 2, 1, 3)
        key = key.reshape(batch, token_count, self.num_heads, head_dim).transpose(0, 2, 1, 3)
        value = value.reshape(batch, token_count, self.num_heads, head_dim).transpose(0, 2, 1, 3)
        scores = (query * (1.0 / math.sqrt(head_dim))) @ key.transpose(0, 1, 3, 2)
        pooled = mx.softmax(scores, axis=-1) @ value
        pooled = pooled.transpose(0, 2, 1, 3).reshape(batch, query_count, width)
        return self.out_proj(pooled)


class _AttentionPooler(nn.Module):
    def __init__(self, hidden_dim: int, num_queries: int, num_heads: int) -> None:
        super().__init__()
        self.query_tokens = mx.zeros((num_queries, hidden_dim), dtype=mx.bfloat16)
        self.cross_attn = _CrossAttention(hidden_dim, num_heads)

    def __call__(self, tokens: mx.array) -> mx.array:
        queries = mx.broadcast_to(
            self.query_tokens[None, :, :],
            (tokens.shape[0], self.query_tokens.shape[0], self.query_tokens.shape[1]),
        )
        return self.cross_attn(queries, tokens)


class LTX25DurationHead(nn.Module):
    """Predict seconds from the trained video and audio connector token streams."""

    def __init__(
        self,
        *,
        video_dim: int = 4096,
        audio_dim: int = 2048,
        hidden_dim: int = 256,
        num_queries: int = 1,
        num_heads: int = 4,
        mlp_hidden: int = 256,
    ) -> None:
        super().__init__()
        self.video_input_proj = nn.Linear(video_dim, hidden_dim)
        self.video_modality_emb = mx.zeros((hidden_dim,), dtype=mx.bfloat16)
        self.audio_input_proj = nn.Linear(audio_dim, hidden_dim)
        self.audio_modality_emb = mx.zeros((hidden_dim,), dtype=mx.bfloat16)
        self.attention_pooler = _AttentionPooler(hidden_dim, num_queries, num_heads)
        self.mlp_hidden = nn.Linear(hidden_dim * num_queries, mlp_hidden)
        self.mlp_out = nn.Linear(mlp_hidden, 1)

    def __call__(
        self,
        video_tokens: mx.array | None = None,
        audio_tokens: mx.array | None = None,
    ) -> mx.array:
        if video_tokens is None and audio_tokens is None:
            raise ValueError(
                "LTX 2.5 duration prediction requires video or audio connector tokens."
            )
        groups = []
        if video_tokens is not None:
            groups.append(self.video_input_proj(video_tokens) + self.video_modality_emb)
        if audio_tokens is not None:
            groups.append(self.audio_input_proj(audio_tokens) + self.audio_modality_emb)
        pooled = self.attention_pooler(mx.concatenate(groups, axis=1))
        # The official model uses tanh-approximate GELU.
        hidden = self.mlp_hidden(pooled.reshape(pooled.shape[0], -1))
        hidden = 0.5 * hidden * (
            1.0
            + mx.tanh(
                math.sqrt(2.0 / math.pi) * (hidden + 0.044715 * hidden * hidden * hidden)
            )
        )
        return mx.exp(self.mlp_out(hidden).squeeze(-1))


def _configuration(path: Path) -> dict[str, object]:
    with safe_open(path, framework="numpy") as handle:
        metadata = handle.metadata() or {}
    try:
        config = json.loads(metadata.get("config", "{}"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid LTX 2.5 duration-head metadata: {path}") from exc
    return config if isinstance(config, dict) else {}


def load_ltx25_duration_head(path: str | Path) -> LTX25DurationHead:
    """Strictly load the small official BF16 duration head."""
    source = Path(path).expanduser()
    config = _configuration(source)
    transformer = config.get("transformer", {})
    head_config = config.get("duration_head", {})
    if not isinstance(transformer, dict) or not isinstance(head_config, dict):
        raise ValueError(f"Invalid LTX 2.5 duration-head configuration: {source}")
    model = LTX25DurationHead(
        video_dim=int(transformer.get("cross_attention_dim", 4096)),
        audio_dim=int(transformer.get("audio_cross_attention_dim", 2048)),
        hidden_dim=int(head_config.get("pooler_hidden_dim", 256)),
        num_queries=int(head_config.get("num_queries", 1)),
        num_heads=int(head_config.get("num_pooler_heads", 4)),
        mlp_hidden=int(head_config.get("mlp_hidden", 256)),
    )
    weights = {
        key.removeprefix("duration_head."): value
        for key, value in dict(mx.load(str(source))).items()
        if key.startswith("duration_head.")
    }
    if not weights:
        raise ValueError(f"No duration_head tensors found in {source}")
    model.load_weights(list(weights.items()), strict=True)
    mx.eval(model.parameters())
    return model


def seconds_to_ltx25_frames(
    seconds: float,
    *,
    frame_rate: float,
    min_seconds: float = 1.0,
    max_seconds: float = 20.0,
) -> int:
    """Clamp a predicted duration and floor it to the official ``8k + 1`` grid."""
    if min_seconds <= 0 or max_seconds < min_seconds:
        raise ValueError("Automatic duration bounds must satisfy 0 < minimum <= maximum.")
    raw_frames = round(max(min_seconds, min(float(seconds), max_seconds)) * frame_rate)
    min_frames = round(min_seconds * frame_rate)
    max_frames = round(max_seconds * frame_rate)
    raw_frames = max(min_frames, min(raw_frames, max_frames))
    frames = ((raw_frames - 1) // 8) * 8 + 1
    if frames < min_frames:
        frames = min(-(-(min_frames - 1) // 8) * 8 + 1, max_frames)
    return frames


__all__ = ["LTX25DurationHead", "load_ltx25_duration_head", "seconds_to_ltx25_frames"]
