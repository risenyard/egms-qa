from __future__ import annotations

import pytest
import torch

from egms_encoder.models.tile_encoder import SinusoidalPosEncoding, TileEncoder


def test_encoder_forward_shapes_and_padding_mask() -> None:
    model = TileEncoder(
        input_length=24,
        d_model=32,
        patch_size=8,
        temporal_layers=1,
        temporal_heads=4,
        spatial_layers=1,
        spatial_heads=4,
        dropout=0.0,
        coord_scale=3500.0,
    ).eval()
    series = torch.randn(2, 5, 24)
    coords = torch.randn(2, 5, 2)
    point_mask = torch.tensor(
        [[True, True, True, True, True], [True, True, True, False, False]]
    )
    with torch.no_grad():
        output = model(series, coords=coords, point_mask=point_mask)

    assert output["embedding"].shape == (2, 5, 32)
    assert output["reconstruction"].shape == (2, 5, 24)
    assert torch.count_nonzero(output["embedding"][1, 3:]) == 0
    assert torch.count_nonzero(output["reconstruction"][1, 3:]) == 0


def test_encoder_rejects_unscaled_absolute_coordinates() -> None:
    model = TileEncoder(
        input_length=8,
        d_model=16,
        temporal_layers=1,
        spatial_layers=1,
        spatial_heads=4,
        temporal_heads=4,
    )
    with pytest.raises(ValueError, match="uncentered"):
        model(torch.zeros(1, 2, 8), coords=torch.full((1, 2, 2), 1_000_000.0))


def test_positional_encoding_rejects_odd_dimension() -> None:
    with pytest.raises(ValueError, match="must be even"):
        SinusoidalPosEncoding(15)
