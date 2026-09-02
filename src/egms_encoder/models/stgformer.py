from __future__ import annotations

import math

import torch
from torch import nn

from egms_encoder.models.common import make_mlp, mask_points, validate_optional_features, validate_series


def build_knn_adjacency(
    coords: torch.Tensor,
    k: int,
    point_mask: torch.Tensor | None = None,
    include_self: bool = True,
    symmetric: bool = True,
) -> torch.Tensor:
    """Build a dense boolean kNN adjacency matrix from tile-local coordinates."""

    if coords.ndim != 3 or coords.shape[-1] != 2:
        raise ValueError(f"Expected coords shape [B, N, 2], got {tuple(coords.shape)}")
    batch_size, num_points, _ = coords.shape
    if k < 1:
        raise ValueError("k must be >= 1")

    valid = (
        point_mask.to(dtype=torch.bool, device=coords.device)
        if point_mask is not None
        else torch.ones(batch_size, num_points, dtype=torch.bool, device=coords.device)
    )
    distances = torch.cdist(coords, coords)
    invalid_keys = ~valid[:, None, :]
    distances = distances.masked_fill(invalid_keys, float("inf"))
    if not include_self:
        eye = torch.eye(num_points, dtype=torch.bool, device=coords.device).unsqueeze(0)
        distances = distances.masked_fill(eye, float("inf"))

    neighbors = min(num_points, k + int(include_self))
    indices = distances.topk(neighbors, dim=-1, largest=False).indices
    adjacency = torch.zeros(batch_size, num_points, num_points, dtype=torch.bool, device=coords.device)
    adjacency.scatter_(dim=-1, index=indices, value=True)
    adjacency = adjacency & valid[:, :, None] & valid[:, None, :]
    if include_self:
        eye = torch.eye(num_points, dtype=torch.bool, device=coords.device).unsqueeze(0)
        adjacency = adjacency | (eye & valid[:, :, None])
    if symmetric:
        adjacency = adjacency | adjacency.transpose(-1, -2)
    return adjacency


def build_deformation_adjacency(
    series: torch.Tensor,
    k: int,
    point_mask: torch.Tensor | None = None,
    include_self: bool = True,
    symmetric: bool = True,
) -> torch.Tensor:
    """Build a dense boolean kNN graph from time-series shape similarity."""

    if series.ndim != 3:
        raise ValueError(f"Expected series shape [B, N, T], got {tuple(series.shape)}")
    batch_size, num_points, _ = series.shape
    if k < 1:
        raise ValueError("k must be >= 1")

    valid = (
        point_mask.to(dtype=torch.bool, device=series.device)
        if point_mask is not None
        else torch.ones(batch_size, num_points, dtype=torch.bool, device=series.device)
    )
    centered = series - series.mean(dim=-1, keepdim=True)
    scale = centered.square().mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-6)
    normalized = torch.nan_to_num(centered / scale, nan=0.0, posinf=0.0, neginf=0.0)
    similarity = torch.matmul(normalized, normalized.transpose(-1, -2)) / max(series.shape[-1], 1)
    similarity = similarity.masked_fill(~valid[:, None, :], float("-inf"))
    if not include_self:
        eye = torch.eye(num_points, dtype=torch.bool, device=series.device).unsqueeze(0)
        similarity = similarity.masked_fill(eye, float("-inf"))

    neighbors = min(num_points, k + int(include_self))
    indices = similarity.topk(neighbors, dim=-1, largest=True).indices
    adjacency = torch.zeros(batch_size, num_points, num_points, dtype=torch.bool, device=series.device)
    adjacency.scatter_(dim=-1, index=indices, value=True)
    adjacency = adjacency & valid[:, :, None] & valid[:, None, :]
    if include_self:
        eye = torch.eye(num_points, dtype=torch.bool, device=series.device).unsqueeze(0)
        adjacency = adjacency | (eye & valid[:, :, None])
    if symmetric:
        adjacency = adjacency | adjacency.transpose(-1, -2)
    return adjacency


class GraphSpatialAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        hidden: torch.Tensor,
        adjacency: torch.Tensor | None,
        point_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        batch_size, num_points, time_steps, d_model = hidden.shape
        x = hidden.permute(0, 2, 1, 3)
        qkv = self.qkv(x).view(batch_size, time_steps, num_points, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=3)
        q = q.permute(0, 1, 3, 2, 4)
        k = k.permute(0, 1, 3, 2, 4)
        v = v.permute(0, 1, 3, 2, 4)

        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        if adjacency is not None:
            allowed = adjacency.to(dtype=torch.bool, device=hidden.device)
            scores = scores.masked_fill(~allowed[:, None, None, :, :], float("-inf"))
        if point_mask is not None:
            valid = point_mask.to(dtype=torch.bool, device=hidden.device)
            scores = scores.masked_fill(~valid[:, None, None, None, :], float("-inf"))

        attention = torch.softmax(scores, dim=-1)
        attention = torch.nan_to_num(attention, nan=0.0)
        attention = self.dropout(attention)
        out = torch.matmul(attention, v)
        out = out.permute(0, 1, 3, 2, 4).reshape(batch_size, time_steps, num_points, d_model)
        out = self.out_proj(out).permute(0, 2, 1, 3)
        return mask_points(out, point_mask)


class STGformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dim_feedforward: int, dropout: float) -> None:
        super().__init__()
        self.temporal_norm = nn.LayerNorm(d_model)
        self.temporal_attention = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.spatial_norm = nn.LayerNorm(d_model)
        self.spatial_attention = GraphSpatialAttention(d_model, num_heads, dropout)
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        hidden: torch.Tensor,
        adjacency: torch.Tensor | None,
        point_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        batch_size, num_points, time_steps, d_model = hidden.shape

        temporal_input = self.temporal_norm(hidden).reshape(batch_size * num_points, time_steps, d_model)
        temporal_out, _ = self.temporal_attention(
            temporal_input,
            temporal_input,
            temporal_input,
            need_weights=False,
        )
        temporal_out = temporal_out.reshape(batch_size, num_points, time_steps, d_model)
        hidden = hidden + self.dropout(temporal_out)
        hidden = mask_points(hidden, point_mask)

        spatial_out = self.spatial_attention(self.spatial_norm(hidden), adjacency, point_mask)
        hidden = hidden + self.dropout(spatial_out)
        hidden = mask_points(hidden, point_mask)

        hidden = hidden + self.dropout(self.ffn(self.ffn_norm(hidden)))
        hidden = mask_points(hidden, point_mask)
        return hidden


class STGformer(nn.Module):
    """EGMS-adapted spatiotemporal graph Transformer.

    This implementation uses factorized temporal attention and graph-masked
    spatial attention. It is intended as a project-local STGformer-style main
    model rather than a byte-for-byte copy of a traffic forecasting repository.

    Tiles may contain different numbers of points. Batch them with padding
    and pass point_mask so padded points are excluded from kNN graph building,
    spatial attention, and outputs. Set output_length=input_length for masked
    reconstruction, or a different output_length for forecasting/probing heads.
    """

    def __init__(
        self,
        input_length: int,
        output_length: int | None = None,
        forecast_horizon: int | None = None,
        static_dim: int = 0,
        d_model: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        dim_feedforward: int | None = None,
        dropout: float = 0.1,
        knn_k: int | None = 8,
        deformation_k: int | None = None,
        use_coords: bool = True,
        use_static: bool = True,
        max_points: int | None = None,
    ) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.input_length = input_length
        if output_length is None:
            output_length = forecast_horizon if forecast_horizon is not None else input_length
        self.output_length = output_length
        self.forecast_horizon = output_length
        self.static_dim = static_dim
        self.knn_k = knn_k
        self.deformation_k = deformation_k
        self.use_coords = use_coords
        self.use_static = use_static and static_dim > 0
        self.max_points = max_points

        self.value_embedding = nn.Linear(1, d_model)
        self.time_embedding = nn.Parameter(torch.zeros(1, 1, input_length, d_model))
        nn.init.trunc_normal_(self.time_embedding, std=0.02)
        self.coord_embedding = make_mlp(2, d_model, dropout=dropout) if use_coords else None
        self.static_embedding = make_mlp(static_dim, d_model, dropout=dropout) if self.use_static else None

        feedforward = dim_feedforward or d_model * 4
        self.blocks = nn.ModuleList(
            [STGformerBlock(d_model, num_heads, feedforward, dropout) for _ in range(num_layers)]
        )
        self.output_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, output_length),
        )

    def forward(
        self,
        series: torch.Tensor,
        coords: torch.Tensor | None = None,
        static: torch.Tensor | None = None,
        point_mask: torch.Tensor | None = None,
        adjacency: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        batch_size, num_points, _ = validate_series(series, self.input_length, self.max_points)
        validate_optional_features(coords, static, point_mask, batch_size, num_points, self.static_dim)

        hidden = self.value_embedding(series.unsqueeze(-1)) + self.time_embedding
        if self.coord_embedding is not None:
            if coords is None:
                raise ValueError("coords are required when use_coords=True")
            hidden = hidden + self.coord_embedding(coords).unsqueeze(2)
        if self.static_embedding is not None:
            if static is None:
                raise ValueError("static features are required when use_static=True")
            hidden = hidden + self.static_embedding(static).unsqueeze(2)
        hidden = mask_points(hidden, point_mask)

        if adjacency is None and self.knn_k is not None:
            if coords is None:
                raise ValueError("coords are required to build kNN adjacency")
            adjacency = build_knn_adjacency(coords, k=self.knn_k, point_mask=point_mask)
        if self.deformation_k is not None:
            deformation_adjacency = build_deformation_adjacency(series, k=self.deformation_k, point_mask=point_mask)
            adjacency = deformation_adjacency if adjacency is None else adjacency | deformation_adjacency
        if adjacency is not None and adjacency.shape != (batch_size, num_points, num_points):
            raise ValueError(f"Expected adjacency shape [B, N, N], got {tuple(adjacency.shape)}")

        for block in self.blocks:
            hidden = block(hidden, adjacency, point_mask)

        point_embedding = hidden[:, :, -1, :]
        reconstruction = self.output_head(point_embedding)
        reconstruction = mask_points(reconstruction, point_mask)
        return {
            "prediction": reconstruction,
            "reconstruction": reconstruction,
            "embedding": point_embedding,
            "adjacency": adjacency,
        }
