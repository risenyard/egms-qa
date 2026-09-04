from __future__ import annotations

import numpy as np
import pytest

from egms_encoder.extract_tokens import parse_args, pool_to_spatial_tokens


def test_spatial_pooling_uses_summary_token_and_row_major_cells() -> None:
    embedding = np.asarray([[1.0, 3.0], [5.0, 7.0]], dtype=np.float32)
    coords = np.asarray([[-2.0, -2.0], [2.0, 2.0]], dtype=np.float32)
    tokens, mask, counts = pool_to_spatial_tokens(
        embedding, coords, grid_size=2, tile_size=8.0
    )
    np.testing.assert_array_equal(tokens[0], embedding.mean(axis=0))
    np.testing.assert_array_equal(tokens[1], embedding[0])
    np.testing.assert_array_equal(tokens[4], embedding[1])
    np.testing.assert_array_equal(mask, [True, True, False, False, True])
    np.testing.assert_array_equal(counts, [1, 0, 0, 1])


def test_extract_defaults_target_hf_installed_inputs() -> None:
    args = parse_args([])
    assert args.checkpoint.endswith("data/encoder/checkpoint/encoder.safetensors")
    assert args.model_config.endswith("data/encoder/checkpoint/config.json")
    assert args.normalization.endswith("data/encoder/checkpoint/normalization.json")
    assert args.manifest.endswith("data/encoder/manifest/split.parquet")
    assert args.data_config.endswith("data/encoder/manifest/data_config.json")
    assert args.output_name == ""


def test_spatial_pooling_validates_shapes() -> None:
    with pytest.raises(ValueError, match="centered_coords"):
        pool_to_spatial_tokens(
            np.zeros((2, 4), dtype=np.float32),
            np.zeros((1, 2), dtype=np.float32),
            grid_size=8,
            tile_size=7000.0,
        )
