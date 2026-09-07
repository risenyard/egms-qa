from __future__ import annotations

import numpy as np
import pytest

from egms_encoder.data.tile_batching import iter_tile_batches, sample_tile_points
from egms_encoder.data.tile_store import TimeWindow


class SyntheticStore:
    def __init__(self) -> None:
        self.time_window = TimeWindow(0, 294, stored_steps=294)
        # Materialized TileStore rows contain 10 feature columns + 294 steps.
        self.tiles = [
            np.arange(3 * 304, dtype=np.float32).reshape(3, 304),
            np.arange(5 * 304, dtype=np.float32).reshape(5, 304),
        ]

    def split_tile_indices(self, split: str) -> np.ndarray:
        assert split == "train"
        return np.asarray([0, 1], dtype=np.int64)

    def get_tile(self, tile_index: int) -> np.ndarray:
        return self.tiles[tile_index]


def test_batching_uses_store_time_window_by_default() -> None:
    batch = next(iter_tile_batches(
        SyntheticStore(), 2, split="train", rng=np.random.default_rng(7)
    ))
    assert batch["series"].shape == (2, 5, 294)
    assert batch["coords"].shape == (2, 5, 2)
    assert batch["point_mask"].sum() == 8


def test_residual_sampling_rejects_invalid_weight() -> None:
    # 10 feature columns + a direct 294-step model window.
    tile = np.zeros((10, 304), dtype=np.float32)
    with pytest.raises(ValueError, match="must be in"):
        sample_tile_points(
            tile,
            5,
            np.random.default_rng(1),
            feature_columns_count=10,
            input_length=294,
            point_sampling="residual_weighted",
            residual_sampling_alpha=1.1,
        )
