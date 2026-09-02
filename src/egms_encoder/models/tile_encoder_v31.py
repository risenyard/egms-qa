"""TileEncoder V3.1: V2 backbone with an additive residual correction head."""

from __future__ import annotations

import torch
from torch import nn

from egms_encoder.models.common import mask_points
from egms_encoder.models.tile_encoder_v2 import TileEncoderV2


class TileEncoderV31(TileEncoderV2):
    """Residual-focused V3.1 model.

    The temporal/spatial backbone and base reconstruction head keep the V2/V3
    key names so V3 checkpoints can warm-start with only residual head keys
    missing.
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
        super().__init__(
            input_length=input_length,
            output_length=output_length,
            d_model=d_model,
            patch_size=patch_size,
            temporal_layers=temporal_layers,
            temporal_heads=temporal_heads,
            spatial_layers=spatial_layers,
            spatial_heads=spatial_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            max_points=max_points,
            coord_scale=coord_scale,
        )
        if residual_head_mode not in {"additive", "aux_only"}:
            raise ValueError(f"Unknown residual_head_mode={residual_head_mode!r}")
        self.residual_head_mode = residual_head_mode
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
            f"TileEncoderV31: {total:,} parameters ({total/1e6:.1f}M), "
            f"residual_mode={residual_head_mode}, residual_scale={float(self.residual_scale):.6f}"
        )

    def _zero_init_residual_output(self) -> None:
        last = self.residual_head[-1]
        if isinstance(last, nn.Linear):
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)

    def forward(
        self,
        series: torch.Tensor,
        coords: torch.Tensor | None = None,
        static: torch.Tensor | None = None,
        point_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        base = super().forward(series, coords=coords, static=static, point_mask=point_mask, **kwargs)
        embedding = base["embedding"]
        base_reconstruction = base["reconstruction"]

        residual_raw = self.residual_head(embedding)
        residual_prediction = linear_detrend_tensor(residual_raw)
        residual_delta = residual_prediction * self.residual_scale.to(
            device=residual_prediction.device,
            dtype=residual_prediction.dtype,
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


def linear_detrend_tensor(x: torch.Tensor) -> torch.Tensor:
    """Remove the best fixed linear fit along the last dimension."""
    t = torch.linspace(-1.0, 1.0, x.shape[-1], device=x.device, dtype=x.dtype)
    t = t - t.mean()
    denom = (t * t).sum().clamp_min(torch.finfo(x.dtype).eps)
    intercept = x.mean(dim=-1, keepdim=True)
    slope = (x * t).sum(dim=-1, keepdim=True) / denom
    return x - (intercept + slope * t)
