"""Tile batching: assemble padded training batches from a tile store.

Points inside a 7km tile are grouped, subsampled when a tile is oversized, and
padded into fixed-shape arrays for the encoder. The store is a
``TileStore`` (see ``tile_store.py``); only its ``split_tile_indices`` and
``get_tile`` methods are used here, so any store with that interface works.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable

import numpy as np

if TYPE_CHECKING:
    from egms_encoder.data.tile_store import TileStore


def iter_tile_batches(
    store: "TileStore",
    tiles_per_batch: int,
    *,
    split: str,
    rng: np.random.Generator,
    max_batches: int | None = None,
    max_points: int | None = None,
    feature_columns_count: int = 10,
    input_length: int = 302,
    point_sampling: str = "uniform",
    residual_sampling_alpha: float = 0.5,
) -> Iterable[dict]:
    """Yield padded tile batches for training.

    Each yielded dict contains:
      - series:     np.ndarray [B, N_max, T]   (masked time series)
      - coords:     np.ndarray [B, N_max, 2]   (easting, northing)
      - point_mask: np.ndarray [B, N_max] bool  (True = real point)
      - num_points: np.ndarray [B]              (actual point count per tile)

    where B = tiles_per_batch and N_max = max points in this batch.
    max_points truncates tiles exceeding this size (for dense attention O(N^2)).
    """
    tile_indices = store.split_tile_indices(split)
    if tile_indices.size == 0:
        raise ValueError(f"No tiles available for split={split}")

    if max_batches is not None:
        # Sample a fixed number of batches (for validation)
        for _ in range(max_batches):
            selected = rng.choice(tile_indices, size=min(tiles_per_batch, len(tile_indices)), replace=False)
            yield _build_tile_batch(
                store, selected, feature_columns_count, input_length, max_points, rng,
                point_sampling=point_sampling,
                residual_sampling_alpha=residual_sampling_alpha,
            )
        return

    # Shuffle tile order for each epoch
    shuffled = tile_indices.copy()
    rng.shuffle(shuffled)

    for start in range(0, len(shuffled), tiles_per_batch):
        batch_indices = shuffled[start : start + tiles_per_batch]
        if len(batch_indices) == 0:
            continue
        yield _build_tile_batch(
            store, batch_indices, feature_columns_count, input_length, max_points, rng,
            point_sampling=point_sampling,
            residual_sampling_alpha=residual_sampling_alpha,
        )


def _build_tile_batch(
    store: "TileStore",
    tile_indices: np.ndarray,
    feature_columns_count: int,
    input_length: int,
    max_points_cap: int | None = None,
    rng: np.random.Generator | None = None,
    point_sampling: str = "uniform",
    residual_sampling_alpha: float = 0.5,
) -> dict:
    """Assemble a padded batch from multiple tiles."""
    tiles_data = [store.get_tile(int(idx)) for idx in tile_indices]
    # Random subsampling for oversized tiles (each epoch sees different subset)
    if max_points_cap is not None:
        subsampled = []
        for t in tiles_data:
            if t.shape[0] > max_points_cap:
                if rng is not None:
                    idx = sample_tile_points(
                        t,
                        max_points_cap,
                        rng,
                        feature_columns_count=feature_columns_count,
                        input_length=input_length,
                        point_sampling=point_sampling,
                        residual_sampling_alpha=residual_sampling_alpha,
                    )
                    idx.sort()  # preserve spatial ordering
                    subsampled.append(t[idx])
                else:
                    subsampled.append(t[:max_points_cap])
            else:
                subsampled.append(t)
        tiles_data = subsampled
    point_counts = [t.shape[0] for t in tiles_data]
    max_points = max(point_counts)
    batch_size = len(tiles_data)

    # Allocate padded arrays
    series = np.zeros((batch_size, max_points, input_length), dtype=np.float32)
    coords = np.zeros((batch_size, max_points, 2), dtype=np.float32)
    point_mask = np.zeros((batch_size, max_points), dtype=bool)

    for i, tile in enumerate(tiles_data):
        n = tile.shape[0]
        # Columns: [easting, northing, static..., time_series...]
        series[i, :n, :] = tile[:, feature_columns_count : feature_columns_count + input_length]
        coords[i, :n, :] = tile[:, :2]
        point_mask[i, :n] = True

    return {
        "series": series,
        "coords": coords,
        "point_mask": point_mask,
        "num_points": np.array(point_counts, dtype=np.int64),
        "tile_indices": tile_indices,
    }


def sample_tile_points(
    tile: np.ndarray,
    max_points: int,
    rng: np.random.Generator,
    *,
    feature_columns_count: int,
    input_length: int,
    point_sampling: str,
    residual_sampling_alpha: float,
) -> np.ndarray:
    """Sample points from an oversized tile."""
    n_points = tile.shape[0]
    if point_sampling == "uniform":
        return rng.choice(n_points, max_points, replace=False)
    if point_sampling != "residual_weighted":
        raise ValueError(f"Unknown point_sampling={point_sampling!r}")

    alpha = float(np.clip(residual_sampling_alpha, 0.0, 1.0))
    weighted_n = int(round(max_points * alpha))
    uniform_n = max_points - weighted_n
    series = tile[:, feature_columns_count : feature_columns_count + input_length]
    scores = residual_rms(series)
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    if weighted_n <= 0 or float(scores.sum()) <= 0.0:
        return rng.choice(n_points, max_points, replace=False)

    probs = scores + 1e-6
    probs = probs / probs.sum()
    weighted = rng.choice(n_points, weighted_n, replace=False, p=probs)
    if uniform_n <= 0:
        return weighted
    remaining_mask = np.ones(n_points, dtype=bool)
    remaining_mask[weighted] = False
    remaining = np.flatnonzero(remaining_mask)
    uniform = rng.choice(remaining, uniform_n, replace=False)
    return np.concatenate([weighted, uniform])


def residual_rms(series: np.ndarray) -> np.ndarray:
    """Per-point linear-detrended residual RMS, robust to occasional NaNs."""
    values = series.astype(np.float64, copy=False)
    finite = np.isfinite(values)
    counts = finite.sum(axis=1)
    valid = counts > 1
    rms = np.zeros(values.shape[0], dtype=np.float64)
    if not valid.any():
        return rms.astype(np.float32)

    t = np.linspace(-1.0, 1.0, values.shape[1], dtype=np.float64)
    sub = values[valid]
    sub_finite = finite[valid]
    sub_counts = counts[valid].astype(np.float64)
    y_sum = np.where(sub_finite, sub, 0.0).sum(axis=1, keepdims=True)
    t_sum = np.where(sub_finite, t, 0.0).sum(axis=1, keepdims=True)
    y_mean = y_sum / sub_counts[:, None]
    t_mean = t_sum / sub_counts[:, None]
    centered_t = np.where(sub_finite, t - t_mean, 0.0)
    centered_y = np.where(sub_finite, sub - y_mean, 0.0)
    denom = np.square(centered_t).sum(axis=1, keepdims=True)
    slope = np.divide(
        (centered_y * centered_t).sum(axis=1, keepdims=True),
        denom,
        out=np.zeros_like(denom),
        where=denom > 1e-12,
    )
    residual = np.where(sub_finite, sub - (y_mean + slope * (t - t_mean)), 0.0)
    rms[valid] = np.sqrt(np.square(residual).sum(axis=1) / np.maximum(sub_counts, 1.0))
    return rms.astype(np.float32)
