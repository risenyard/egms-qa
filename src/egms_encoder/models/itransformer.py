from __future__ import annotations

import torch
from torch import nn

from egms_encoder.models.common import make_mlp, mask_points, validate_optional_features, validate_series


class ITransformer(nn.Module):
    """iTransformer-style baseline for EGMS tile representation learning.

    Each point is treated as one variable token. The token embedding is built
    from that point's full historical time series, with optional coordinate
    and static-feature embeddings.

    Tiles may contain different numbers of points. Batch them with padding
    and pass point_mask so padded points do not contribute to attention or
    outputs. Set output_length=input_length for masked reconstruction, or a
    different output_length for forecasting/probing heads.
    """

    def __init__(
        self,
        input_length: int,
        output_length: int | None = None,
        forecast_horizon: int | None = None,
        static_dim: int = 0,
        d_model: int = 128,
        num_layers: int = 3,
        num_heads: int = 4,
        dim_feedforward: int | None = None,
        dropout: float = 0.1,
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
        self.use_coords = use_coords
        self.use_static = use_static and static_dim > 0
        self.max_points = max_points

        self.series_embedding = nn.Sequential(
            nn.LayerNorm(input_length),
            nn.Linear(input_length, d_model),
        )
        self.coord_embedding = make_mlp(2, d_model, dropout=dropout) if use_coords else None
        self.static_embedding = make_mlp(static_dim, d_model, dropout=dropout) if self.use_static else None

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward or d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )
        self.prediction_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, output_length),
        )

    def forward(
        self,
        series: torch.Tensor,
        coords: torch.Tensor | None = None,
        static: torch.Tensor | None = None,
        point_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        batch_size, num_points, _ = validate_series(series, self.input_length, self.max_points)
        validate_optional_features(coords, static, point_mask, batch_size, num_points, self.static_dim)

        hidden = self.series_embedding(series)
        if self.coord_embedding is not None:
            if coords is None:
                raise ValueError("coords are required when use_coords=True")
            hidden = hidden + self.coord_embedding(coords)
        if self.static_embedding is not None:
            if static is None:
                raise ValueError("static features are required when use_static=True")
            hidden = hidden + self.static_embedding(static)

        key_padding_mask = None
        if point_mask is not None:
            key_padding_mask = ~point_mask.to(dtype=torch.bool, device=series.device)

        embedding = self.encoder(hidden, src_key_padding_mask=key_padding_mask)
        embedding = mask_points(embedding, point_mask)
        reconstruction = self.prediction_head(embedding)
        reconstruction = mask_points(reconstruction, point_mask)
        return {
            "prediction": reconstruction,
            "reconstruction": reconstruction,
            "embedding": embedding,
        }
