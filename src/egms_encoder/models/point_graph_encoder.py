from __future__ import annotations

import math

import torch
from torch import nn

from egms_encoder.models.common import mask_points, validate_series


def build_sparse_knn_indices(
    coords: torch.Tensor,
    k: int,
    point_mask: torch.Tensor | None = None,
    include_self: bool = False,
) -> torch.Tensor:
    """Return per-point coordinate nearest-neighbor indices shaped [B, N, K]."""

    if coords.ndim != 3 or coords.shape[-1] != 2:
        raise ValueError(f"Expected coords shape [B, N, 2], got {tuple(coords.shape)}")
    batch_size, num_points, _ = coords.shape
    if k < 1:
        raise ValueError("k must be >= 1")
    if num_points < 1:
        raise ValueError("coords must include at least one point")

    valid = (
        point_mask.to(dtype=torch.bool, device=coords.device)
        if point_mask is not None
        else torch.ones(batch_size, num_points, dtype=torch.bool, device=coords.device)
    )
    distances = torch.cdist(coords, coords)
    distances = distances.masked_fill(~valid[:, None, :], float("inf"))
    if not include_self and num_points > 1:
        eye = torch.eye(num_points, dtype=torch.bool, device=coords.device).unsqueeze(0)
        distances = distances.masked_fill(eye, float("inf"))

    neighbors = min(num_points, k + int(not include_self and num_points == 1))
    indices = distances.topk(neighbors, dim=-1, largest=False).indices
    if neighbors < k:
        pad = indices[..., -1:].expand(batch_size, num_points, k - neighbors)
        indices = torch.cat([indices, pad], dim=-1)
    return indices


class SparseGraphAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden: torch.Tensor, neighbor_indices: torch.Tensor) -> torch.Tensor:
        if hidden.ndim != 3:
            raise ValueError(f"Expected hidden shape [B, N, D], got {tuple(hidden.shape)}")
        if neighbor_indices.ndim != 3:
            raise ValueError(f"Expected neighbor_indices shape [B, N, K], got {tuple(neighbor_indices.shape)}")
        batch_size, num_points, d_model = hidden.shape
        if neighbor_indices.shape[:2] != (batch_size, num_points):
            raise ValueError(f"Expected neighbor_indices shape [B, N, K], got {tuple(neighbor_indices.shape)}")

        qkv = self.qkv(hidden).view(batch_size, num_points, 3, self.num_heads, self.head_dim)
        q, keys, values = qkv.unbind(dim=2)

        batch_index = torch.arange(batch_size, device=hidden.device)[:, None, None]
        neighbor_keys = keys[batch_index, neighbor_indices]
        neighbor_values = values[batch_index, neighbor_indices]

        scores = (q[:, :, None, :, :] * neighbor_keys).sum(dim=-1) / math.sqrt(self.head_dim)
        attention = torch.softmax(scores, dim=2)
        attention = self.dropout(attention)
        out = (attention[..., None] * neighbor_values).sum(dim=2)
        out = out.reshape(batch_size, num_points, d_model)
        return self.out_proj(out)


class PointGraphEncoderBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dim_feedforward: int, dropout: float) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(d_model)
        self.attention = SparseGraphAttention(d_model, num_heads, dropout)
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden: torch.Tensor, neighbor_indices: torch.Tensor, point_mask: torch.Tensor | None) -> torch.Tensor:
        hidden = hidden + self.dropout(self.attention(self.attention_norm(hidden), neighbor_indices))
        hidden = mask_points(hidden, point_mask)
        hidden = hidden + self.dropout(self.ffn(self.ffn_norm(hidden)))
        return mask_points(hidden, point_mask)


class PointGraphEncoder(nn.Module):
    """Point-level deformation encoder with sparse coordinate-neighbor attention."""

    def __init__(
        self,
        input_length: int,
        output_length: int | None = None,
        d_model: int = 128,
        num_layers: int = 4,
        num_heads: int = 4,
        dim_feedforward: int | None = None,
        dropout: float = 0.1,
        num_neighbors: int = 16,
        max_points: int | None = None,
    ) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        if num_neighbors < 1:
            raise ValueError("num_neighbors must be >= 1")
        self.input_length = input_length
        self.output_length = output_length or input_length
        self.num_neighbors = num_neighbors
        self.max_points = max_points

        self.series_embedding = nn.Sequential(
            nn.LayerNorm(input_length),
            nn.Linear(input_length, d_model),
        )
        feedforward = dim_feedforward or d_model * 4
        self.blocks = nn.ModuleList(
            [PointGraphEncoderBlock(d_model, num_heads, feedforward, dropout) for _ in range(num_layers)]
        )
        self.reconstruction_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, self.output_length),
        )

    def forward(
        self,
        series: torch.Tensor,
        coords: torch.Tensor | None = None,
        static: torch.Tensor | None = None,
        point_mask: torch.Tensor | None = None,
        neighbor_indices: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        del static
        batch_size, num_points, _ = validate_series(series, self.input_length, self.max_points)
        if point_mask is not None and point_mask.shape != (batch_size, num_points):
            raise ValueError(f"Expected point_mask shape [B, N], got {tuple(point_mask.shape)}")
        if neighbor_indices is None:
            if coords is None:
                raise ValueError("coords are required when neighbor_indices is not provided")
            neighbor_indices = build_sparse_knn_indices(coords, self.num_neighbors, point_mask=point_mask)
        if neighbor_indices.shape != (batch_size, num_points, self.num_neighbors):
            raise ValueError(
                f"Expected neighbor_indices shape {(batch_size, num_points, self.num_neighbors)}, "
                f"got {tuple(neighbor_indices.shape)}"
            )
        neighbor_indices = neighbor_indices.to(dtype=torch.long, device=series.device)

        embedding = mask_points(self.series_embedding(series), point_mask)
        for block in self.blocks:
            embedding = block(embedding, neighbor_indices, point_mask)

        reconstruction = self.reconstruction_head(embedding)
        reconstruction = mask_points(reconstruction, point_mask)
        return {
            "prediction": reconstruction,
            "reconstruction": reconstruction,
            "embedding": embedding,
            "neighbor_indices": neighbor_indices,
        }
