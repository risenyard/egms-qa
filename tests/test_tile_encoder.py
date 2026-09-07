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


def test_encode_matches_forward_without_reconstruction_outputs() -> None:
    model = TileEncoder(
        input_length=16,
        d_model=32,
        patch_size=8,
        temporal_layers=1,
        temporal_heads=4,
        spatial_layers=1,
        spatial_heads=4,
        dropout=0.0,
        coord_scale=3500.0,
    ).eval()
    series = torch.randn(1, 4, 16)
    coords = torch.randn(1, 4, 2)
    mask = torch.ones(1, 4, dtype=torch.bool)
    with torch.no_grad():
        embedding = model.encode(series, coords=coords, point_mask=mask)
        output = model(series, coords=coords, point_mask=mask)
    assert isinstance(embedding, torch.Tensor)
    assert embedding.shape == (1, 4, 32)
    torch.testing.assert_close(embedding, output["embedding"], rtol=0, atol=0)


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
