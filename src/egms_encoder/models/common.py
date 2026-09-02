from __future__ import annotations

import torch
from torch import nn


def make_mlp(input_dim: int, output_dim: int, hidden_dim: int | None = None, dropout: float = 0.0) -> nn.Module:
    hidden = hidden_dim or output_dim
    return nn.Sequential(
        nn.LayerNorm(input_dim),
        nn.Linear(input_dim, hidden),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden, output_dim),
    )


def validate_series(
    series: torch.Tensor,
    input_length: int,
    max_points: int | None = None,
) -> tuple[int, int, int]:
    if series.ndim != 3:
        raise ValueError(f"Expected series shape [batch, points, time], got {tuple(series.shape)}")
    batch_size, num_points, time_steps = series.shape
    if max_points is not None and num_points > max_points:
        raise ValueError(f"Expected at most {max_points} points per tile, got {num_points}")
    if time_steps != input_length:
        raise ValueError(f"Expected {input_length} time steps, got {time_steps}")
    return batch_size, num_points, time_steps


def validate_optional_features(
    coords: torch.Tensor | None,
    static: torch.Tensor | None,
    point_mask: torch.Tensor | None,
    batch_size: int,
    num_points: int,
    static_dim: int,
) -> None:
    if coords is not None and coords.shape[:2] != (batch_size, num_points):
        raise ValueError(f"Expected coords shape [B, N, 2], got {tuple(coords.shape)}")
    if coords is not None and coords.shape[-1] != 2:
        raise ValueError(f"Expected coords last dimension 2, got {coords.shape[-1]}")
    if static is not None and static.shape[:2] != (batch_size, num_points):
        raise ValueError(f"Expected static shape [B, N, F], got {tuple(static.shape)}")
    if static is not None and static.shape[-1] != static_dim:
        raise ValueError(f"Expected static_dim={static_dim}, got {static.shape[-1]}")
    if point_mask is not None and point_mask.shape != (batch_size, num_points):
        raise ValueError(f"Expected point_mask shape [B, N], got {tuple(point_mask.shape)}")


def mask_points(values: torch.Tensor, point_mask: torch.Tensor | None) -> torch.Tensor:
    if point_mask is None:
        return values
    view_shape = [point_mask.shape[0], point_mask.shape[1]] + [1] * (values.ndim - 2)
    return values * point_mask.reshape(view_shape).to(dtype=values.dtype, device=values.device)
