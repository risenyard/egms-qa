"""Dense-attention Tile Encoder for EGMS deformation data.

每个 tile (~1000 个点) 足够小，可以使用 full dense attention
代替 sparse kNN attention，让每个点直接看到同 tile 内的所有其他点。
多个 tile 通过 padding 并行处理，充分利用 GPU。
"""

from __future__ import annotations

import torch
from torch import nn

from egms_encoder.models.common import mask_points, validate_series


class TileEncoderBlock(nn.Module):
    """Single transformer block with dense multi-head self-attention."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dim_feedforward: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(d_model)
        self.attention = nn.MultiheadAttention(
            d_model,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
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
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass with dense attention.

        Parameters
        ----------
        hidden : [B, N, D]
        key_padding_mask : [B, N] bool, True = padding (will be masked out)
        """
        normed = self.attention_norm(hidden)
        attn_out, _ = self.attention(
            normed, normed, normed,
            key_padding_mask=key_padding_mask,
        )
        hidden = hidden + self.dropout(attn_out)
        hidden = hidden + self.dropout(self.ffn(self.ffn_norm(hidden)))
        return hidden


class TileEncoder(nn.Module):
    """Tile-level deformation encoder with dense self-attention.

    Designed for EGMS tiles where N ~ 1000 points, making full O(N²)
    attention feasible and more expressive than sparse kNN attention.
    Multiple tiles are batched together (B = number of tiles).
    """

    def __init__(
        self,
        input_length: int,
        output_length: int | None = None,
        d_model: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        dim_feedforward: int | None = None,
        dropout: float = 0.1,
        max_points: int | None = None,
    ) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.input_length = input_length
        self.output_length = output_length or input_length
        self.max_points = max_points

        # Project T-dim time series → d_model
        self.series_embedding = nn.Sequential(
            nn.LayerNorm(input_length),
            nn.Linear(input_length, d_model),
        )

        # Optional coordinate embedding
        self.coord_embedding = nn.Sequential(
            nn.Linear(2, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        feedforward = dim_feedforward or d_model * 4
        self.blocks = nn.ModuleList(
            [TileEncoderBlock(d_model, num_heads, feedforward, dropout) for _ in range(num_layers)]
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
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        """Forward pass.

        Parameters
        ----------
        series : [B, N, T] time series (possibly masked)
        coords : [B, N, 2] easting/northing coordinates
        static : unused, kept for API compatibility
        point_mask : [B, N] bool, True = real point, False = padding
        """
        del static
        batch_size, num_points, _ = validate_series(series, self.input_length, self.max_points)
        if point_mask is not None and point_mask.shape != (batch_size, num_points):
            raise ValueError(f"Expected point_mask shape [B, N], got {tuple(point_mask.shape)}")

        # Build embedding: time series + spatial coordinates
        embedding = self.series_embedding(series)
        if coords is not None:
            embedding = embedding + self.coord_embedding(coords)
        embedding = mask_points(embedding, point_mask)

        # key_padding_mask for nn.MultiheadAttention: True = ignore this position
        key_padding_mask = ~point_mask if point_mask is not None else None

        for block in self.blocks:
            embedding = block(embedding, key_padding_mask=key_padding_mask)
            embedding = mask_points(embedding, point_mask)

        reconstruction = self.reconstruction_head(embedding)
        reconstruction = mask_points(reconstruction, point_mask)

        return {
            "prediction": reconstruction,
            "reconstruction": reconstruction,
            "embedding": embedding,
        }
