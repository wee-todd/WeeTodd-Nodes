"""Compatibility exports for the shared WeeTodd MLX Sol-attention kernel."""

from wee_todd_mlx.sol_attention import (
    SolAttentionConfig,
    route_telemetry_report,
    sol_attention,
    supports_sol_attention,
)

__all__ = [
    "SolAttentionConfig",
    "route_telemetry_report",
    "sol_attention",
    "supports_sol_attention",
]
