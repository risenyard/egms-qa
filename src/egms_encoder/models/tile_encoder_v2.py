"""TileEncoder V2: Patch-based Temporal Transformer + Spatial Dense Attention.

V1 用单个 Linear(T→256) 压缩时间维度 — 太粗糙。
V2 引入 PatchTST 风格的时间 Transformer:
  T 步 → 分成若干 patch → 时间 attention → pool → 256维
然后再做空间 dense attention (与 V1 相同)。

配合同步遮挡 (sync_mask)，迫使模型真正学习时间规律。
"""

from __future__ import annotations

import math

import torch
from torch import nn

from egms_encoder.models.common import mask_points, validate_series


class SinusoidalPosEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 512) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, L, D]"""
        return x + self.pe[:, : x.size(1), :]


class TemporalPatchEncoder(nn.Module):
    """Encode a time series via patching + Transformer.

    T steps → patch into chunks of patch_size → linear project → Transformer → mean pool
    """

    def __init__(
        self,
        input_length: int,
        d_model: int,
        patch_size: int = 16,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = math.ceil(input_length / patch_size)
        self.padded_length = self.num_patches * patch_size

        # Project each patch to d_model
        self.patch_embed = nn.Linear(patch_size, d_model)
        self.pos_encoding = SinusoidalPosEncoding(d_model, max_len=self.num_patches + 1)
        self.norm_in = nn.LayerNorm(d_model)

        # Temporal Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm_out = nn.LayerNorm(d_model)

    def forward(self, series: torch.Tensor) -> torch.Tensor:
        """
        series: [B, N, T] → output: [B, N, d_model]

        Internally reshapes to [B*N, num_patches, d_model] for temporal attention,
        then pools back to [B*N, d_model] and reshapes to [B, N, d_model].
        """
        B, N, T = series.shape

        # Pad time series to multiple of patch_size
        if T < self.padded_length:
            pad = torch.zeros(B, N, self.padded_length - T, device=series.device, dtype=series.dtype)
            series = torch.cat([series, pad], dim=2)

        # Reshape to patches: [B, N, num_patches, patch_size]
        x = series.reshape(B, N, self.num_patches, self.patch_size)

        # Flatten batch and points: [B*N, num_patches, patch_size]
        x = x.reshape(B * N, self.num_patches, self.patch_size)

        # Project patches: [B*N, num_patches, d_model]
        x = self.patch_embed(x)
        x = self.norm_in(x)
        x = self.pos_encoding(x)

        # Temporal Transformer
        x = self.transformer(x)
        x = self.norm_out(x)

        # Mean pool over patches: [B*N, d_model]
        x = x.mean(dim=1)

        # Reshape back: [B, N, d_model]
        return x.reshape(B, N, -1)


class SpatialBlock(nn.Module):
    """Single transformer block with dense multi-head self-attention (spatial)."""

    def __init__(self, d_model: int, num_heads: int, dim_feedforward: int, dropout: float) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(d_model)
        self.attention = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True,
        )
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        normed = self.attention_norm(hidden)
        attn_out, _ = self.attention(normed, normed, normed, key_padding_mask=key_padding_mask)
        hidden = hidden + self.dropout(attn_out)
        hidden = hidden + self.dropout(self.ffn(self.ffn_norm(hidden)))
        return hidden


class TileEncoderV2(nn.Module):
    """Tile-level encoder with Temporal Patch Transformer + Spatial Dense Attention.

    Architecture:
      1. Temporal: T steps → patch → Transformer(2L) → mean pool → [N, d_model]
      2. Spatial:  + coord_embedding → Dense Attention(6L) → [N, d_model]
      3. Output:   Linear(d_model → T) for reconstruction
    """

    def __init__(
        self,
        input_length: int,
        output_length: int | None = None,
        d_model: int = 256,
        # Temporal Transformer params
        patch_size: int = 16,
        temporal_layers: int = 2,
        temporal_heads: int = 4,
        # Spatial Transformer params
        spatial_layers: int = 6,
        spatial_heads: int = 8,
        dim_feedforward: int | None = None,
        dropout: float = 0.1,
        max_points: int | None = None,
        coord_scale: float | None = None,
    ) -> None:
        super().__init__()
        if d_model % spatial_heads != 0:
            raise ValueError("d_model must be divisible by spatial_heads")
        if d_model % temporal_heads != 0:
            raise ValueError("d_model must be divisible by temporal_heads")

        self.input_length = input_length
        self.output_length = output_length or input_length
        self.max_points = max_points
        # Divisor applied to centered coords before coord_embedding. None keeps the
        # legacy behavior (raw centered metres, ~+-3700 m for a 7 km tile), which
        # makes the coord branch ~200x the time branch at init. Set to the tile
        # half-width (e.g. 3500) to bring centered coords to ~+-1 and balance the
        # two additive branches.
        self.coord_scale = float(coord_scale) if coord_scale else None

        # Temporal encoder: patches + transformer
        self.temporal_encoder = TemporalPatchEncoder(
            input_length=input_length,
            d_model=d_model,
            patch_size=patch_size,
            num_layers=temporal_layers,
            num_heads=temporal_heads,
            dropout=dropout,
        )

        # Coordinate embedding
        self.coord_embedding = nn.Sequential(
            nn.Linear(2, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        # Spatial transformer blocks
        feedforward = dim_feedforward or d_model * 4
        self.spatial_blocks = nn.ModuleList(
            [SpatialBlock(d_model, spatial_heads, feedforward, dropout) for _ in range(spatial_layers)]
        )

        # Reconstruction head
        self.reconstruction_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, self.output_length),
        )

        total = sum(p.numel() for p in self.parameters())
        print(f"TileEncoderV2: {total:,} parameters ({total/1e6:.1f}M)")
        print(f"  Temporal: patch_size={patch_size}, {temporal_layers} layers, {temporal_heads} heads")
        print(f"  Spatial:  {spatial_layers} layers, {spatial_heads} heads")

    def forward(
        self,
        series: torch.Tensor,
        coords: torch.Tensor | None = None,
        static: torch.Tensor | None = None,
        point_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        """
        series:     [B, N, T]
        coords:     [B, N, 2]
        point_mask: [B, N] bool (True = real point)
        """
        del static
        batch_size, num_points, _ = validate_series(series, self.input_length, self.max_points)
        if point_mask is not None and point_mask.shape != (batch_size, num_points):
            raise ValueError(f"Expected point_mask shape [B, N], got {tuple(point_mask.shape)}")

        # 1. Temporal encoding: [B, N, T] → [B, N, d_model]
        embedding = self.temporal_encoder(series)

        # 2. Add coordinate embedding
        if coords is not None:
            # Training centers coords per tile (~ km half-size); raw EPSG:3035 values
            # are ~1e6, which silently swamps the temporal embedding through this
            # additive residual. Guard against forgetting to center.
            coord_max = float(coords.detach().abs().amax())
            if coord_max > 1e5:
                raise ValueError(
                    f"coords appear uncentered: max(|coords|)={coord_max:.3g}. "
                    "Center per tile before forward: coords -= coords.mean(dim=-2, keepdim=True)."
                )
            if self.coord_scale is not None:
                coords = coords / self.coord_scale
            embedding = embedding + self.coord_embedding(coords)
        embedding = mask_points(embedding, point_mask)

        # 3. Spatial attention
        key_padding_mask = ~point_mask if point_mask is not None else None
        for block in self.spatial_blocks:
            embedding = block(embedding, key_padding_mask=key_padding_mask)
            embedding = mask_points(embedding, point_mask)

        # 4. Reconstruction
        reconstruction = self.reconstruction_head(embedding)
        reconstruction = mask_points(reconstruction, point_mask)

        return {
            "prediction": reconstruction,
            "reconstruction": reconstruction,
            "embedding": embedding,
        }
