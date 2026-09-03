"""EGMS tile encoder.

Maps a variable-size tile of persistent-scatterer displacement histories to a
fixed set of per-point contextual features, trained self-supervised with masked
reconstruction. Architecture:

  1. Temporal: each point's history is patchified along time and encoded by a
     patch Transformer, then mean-pooled to one feature per point.
  2. Spatial:  a per-point coordinate embedding is added, then a stack of dense
     self-attention blocks exchanges information across points within the tile.
  3. Heads:    a linear reconstruction head, plus an additive residual head that
     refines the trend-free component of the reconstruction.
"""

from __future__ import annotations

import math

import torch
from torch import nn


def validate_series(series: torch.Tensor, input_length: int, max_points: int | None = None) -> tuple[int, int, int]:
    if series.ndim != 3:
        raise ValueError(f"Expected series shape [batch, points, time], got {tuple(series.shape)}")
    batch_size, num_points, time_steps = series.shape
    if max_points is not None and num_points > max_points:
        raise ValueError(f"Expected at most {max_points} points per tile, got {num_points}")
    if time_steps != input_length:
        raise ValueError(f"Expected {input_length} time steps, got {time_steps}")
    return batch_size, num_points, time_steps


def mask_points(values: torch.Tensor, point_mask: torch.Tensor | None) -> torch.Tensor:
    if point_mask is None:
        return values
    view_shape = [point_mask.shape[0], point_mask.shape[1]] + [1] * (values.ndim - 2)
    return values * point_mask.reshape(view_shape).to(dtype=values.dtype, device=values.device)


def linear_detrend_tensor(x: torch.Tensor) -> torch.Tensor:
    """Remove the best fixed linear fit along the last dimension."""
    t = torch.linspace(-1.0, 1.0, x.shape[-1], device=x.device, dtype=x.dtype)
    t = t - t.mean()
    denom = (t * t).sum().clamp_min(torch.finfo(x.dtype).eps)
    intercept = x.mean(dim=-1, keepdim=True)
    slope = (x * t).sum(dim=-1, keepdim=True) / denom
    return x - (intercept + slope * t)


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

    T steps -> patch into chunks of patch_size -> linear project -> Transformer -> mean pool.
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

        self.patch_embed = nn.Linear(patch_size, d_model)
        self.pos_encoding = SinusoidalPosEncoding(d_model, max_len=self.num_patches + 1)
        self.norm_in = nn.LayerNorm(d_model)

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
        """series: [B, N, T] -> [B, N, d_model]."""
        B, N, T = series.shape
        if T < self.padded_length:
            pad = torch.zeros(B, N, self.padded_length - T, device=series.device, dtype=series.dtype)
            series = torch.cat([series, pad], dim=2)
        x = series.reshape(B, N, self.num_patches, self.patch_size)
        x = x.reshape(B * N, self.num_patches, self.patch_size)
        x = self.patch_embed(x)
        x = self.norm_in(x)
        x = self.pos_encoding(x)
        x = self.transformer(x)
        x = self.norm_out(x)
        x = x.mean(dim=1)
        return x.reshape(B, N, -1)


class SpatialBlock(nn.Module):
    """Single transformer block with dense multi-head self-attention (spatial)."""

    def __init__(self, d_model: int, num_heads: int, dim_feedforward: int, dropout: float) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(d_model)
        self.attention = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
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


class TileEncoder(nn.Module):
    """Temporal patch Transformer + spatial dense attention, with an additive
    residual correction head over the reconstruction.

    Submodule names are stable so a saved checkpoint loads back into this class.
    """

    def __init__(
        self,
        input_length: int,
        output_length: int | None = None,
        d_model: int = 256,
        patch_size: int = 16,
        temporal_layers: int = 2,
        temporal_heads: int = 4,
        spatial_layers: int = 6,
        spatial_heads: int = 8,
        dim_feedforward: int | None = None,
        dropout: float = 0.1,
        max_points: int | None = None,
        residual_scale: float = 1.0,
        residual_head_mode: str = "additive",
        coord_scale: float | None = None,
    ) -> None:
        super().__init__()
        if d_model % spatial_heads != 0:
            raise ValueError("d_model must be divisible by spatial_heads")
        if d_model % temporal_heads != 0:
            raise ValueError("d_model must be divisible by temporal_heads")
        if residual_head_mode not in {"additive", "aux_only"}:
            raise ValueError(f"Unknown residual_head_mode={residual_head_mode!r}")

        self.input_length = input_length
        self.output_length = output_length or input_length
        self.max_points = max_points
        # Divisor applied to centered coords before coord_embedding. Set to the
        # tile half-width (e.g. 3500) to bring centered coords to ~+-1 and balance
        # the temporal and coordinate branches.
        self.coord_scale = float(coord_scale) if coord_scale else None
        self.residual_head_mode = residual_head_mode

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

        # Base reconstruction head
        self.reconstruction_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, self.output_length),
        )

        # Additive residual correction head (trend-free refinement)
        self.register_buffer(
            "residual_scale",
            torch.tensor(float(max(residual_scale, 1e-6)), dtype=torch.float32),
            persistent=True,
        )
        self.residual_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, self.output_length),
        )
        self._zero_init_residual_output()

        total = sum(p.numel() for p in self.parameters())
        print(
            f"TileEncoder: {total:,} parameters ({total/1e6:.1f}M), "
            f"residual_mode={residual_head_mode}, residual_scale={float(self.residual_scale):.6f}"
        )

    def _zero_init_residual_output(self) -> None:
        last = self.residual_head[-1]
        if isinstance(last, nn.Linear):
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)

    def _encode(
        self,
        series: torch.Tensor,
        coords: torch.Tensor | None,
        point_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Temporal + spatial backbone -> (embedding, base reconstruction)."""
        batch_size, num_points, _ = validate_series(series, self.input_length, self.max_points)
        if point_mask is not None and point_mask.shape != (batch_size, num_points):
            raise ValueError(f"Expected point_mask shape [B, N], got {tuple(point_mask.shape)}")

        embedding = self.temporal_encoder(series)

        if coords is not None:
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

        key_padding_mask = ~point_mask if point_mask is not None else None
        for block in self.spatial_blocks:
            embedding = block(embedding, key_padding_mask=key_padding_mask)
            embedding = mask_points(embedding, point_mask)

        base_reconstruction = self.reconstruction_head(embedding)
        base_reconstruction = mask_points(base_reconstruction, point_mask)
        return embedding, base_reconstruction

    def forward(
        self,
        series: torch.Tensor,
        coords: torch.Tensor | None = None,
        static: torch.Tensor | None = None,
        point_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        """series: [B, N, T], coords: [B, N, 2], point_mask: [B, N] bool."""
        del static, kwargs
        embedding, base_reconstruction = self._encode(series, coords, point_mask)

        residual_raw = self.residual_head(embedding)
        residual_prediction = linear_detrend_tensor(residual_raw)
        residual_delta = residual_prediction * self.residual_scale.to(
            device=residual_prediction.device, dtype=residual_prediction.dtype
        )
        residual_delta = mask_points(residual_delta, point_mask)

        if self.residual_head_mode == "additive":
            reconstruction = base_reconstruction + residual_delta
        else:
            reconstruction = base_reconstruction
        reconstruction = mask_points(reconstruction, point_mask)

        return {
            "prediction": reconstruction,
            "reconstruction": reconstruction,
            "embedding": embedding,
            "base_reconstruction": base_reconstruction,
            "residual_delta": residual_delta,
            "residual_prediction": residual_prediction,
            "residual_raw": residual_raw,
        }
