"""Tile-aware data store: groups EGMS points into 7km spatial tiles.

每个 tile 对应真实世界中一个 7km×7km 的空间块，
内部的所有测量点构成一个完整、干净的空间图。
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pyarrow.parquet as pq


def _safe_nanmedian(values: np.ndarray) -> float:
    finite = np.isfinite(values)
    if not finite.any():
        return 0.0
    return float(np.median(values[finite]))


def _quantile_bin(values: np.ndarray, bins: int) -> np.ndarray:
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        return np.zeros(len(values), dtype=np.int64)
    quantiles = np.quantile(finite_values, np.linspace(0.0, 1.0, bins + 1)[1:-1])
    quantiles = np.unique(quantiles)
    clean = np.nan_to_num(values, nan=float(np.median(finite_values)))
    return np.searchsorted(quantiles, clean, side="right").astype(np.int64)


def _proportional_counts(counts: np.ndarray, target: int) -> np.ndarray:
    if target <= 0 or counts.sum() <= 0:
        return np.zeros_like(counts)
    target = min(int(target), int(counts.sum()))
    raw = counts.astype(np.float64) * (target / float(counts.sum()))
    allocated = np.floor(raw).astype(np.int64)
    remainder = target - int(allocated.sum())
    if remainder > 0:
        order = np.argsort(-(raw - allocated))
        for idx in order:
            if remainder <= 0:
                break
            if allocated[idx] < counts[idx]:
                allocated[idx] += 1
                remainder -= 1
    return allocated


class TileStore:
    """Load all EGMS points and group them by spatial tile.

    Parameters
    ----------
    data : np.ndarray
        Full dataset shaped [total_rows, features].  The first two columns
        must be easting and northing (EPSG:3035 metres).
    tile_size : float
        Side length of square tiles in metres (default 7000 = 7 km).
    min_points : int
        Discard tiles with fewer points than this threshold.
    """

    def __init__(
        self,
        data: np.ndarray,
        tile_size: float = 7000.0,
        min_points: int = 200,
    ) -> None:
        if data.ndim != 2 or data.shape[1] < 3:
            raise ValueError("data must be shaped [rows, features] with at least 3 columns")

        self.data = data.astype(np.float32, copy=False)
        self.tile_size = float(tile_size)

        # Extract coordinates (columns 0 and 1: easting, northing)
        coords = np.nan_to_num(self.data[:, :2], nan=0.0, posinf=0.0, neginf=0.0)
        self.origin = coords.min(axis=0)

        # Assign each point a tile ID based on spatial grid
        tile_x = ((coords[:, 0] - self.origin[0]) / self.tile_size).astype(np.int32)
        tile_y = ((coords[:, 1] - self.origin[1]) / self.tile_size).astype(np.int32)
        tile_ids = tile_x.astype(np.int64) * 100_000 + tile_y.astype(np.int64)

        # Group row indices by tile, filtering out small tiles
        unique_tiles, inverse = np.unique(tile_ids, return_inverse=True)
        self.tiles: list[np.ndarray] = []
        self.tile_metadata: list[dict] = []
        for tile_idx, tile_id in enumerate(unique_tiles):
            row_indices = np.flatnonzero(inverse == tile_idx)
            if len(row_indices) < min_points:
                continue
            center_x = float(coords[row_indices, 0].mean())
            center_y = float(coords[row_indices, 1].mean())
            self.tiles.append(row_indices)
            self.tile_metadata.append({
                "tile_id": int(tile_id),
                "num_points": len(row_indices),
                "center_easting": center_x,
                "center_northing": center_y,
            })

        if not self.tiles:
            raise ValueError(
                f"No tiles with >= {min_points} points. "
                f"Total points: {len(data)}, tile_size: {tile_size}"
            )

        self.num_tiles = len(self.tiles)
        counts = [len(t) for t in self.tiles]
        print(
            f"TileStore: {self.num_tiles} tiles from {len(data):,} points "
            f"(tile_size={tile_size:.0f}m, min_points={min_points}), "
            f"points/tile: min={min(counts)}, median={int(np.median(counts))}, "
            f"max={max(counts)}",
            flush=True,
        )

    @classmethod
    def from_parquet(
        cls,
        parquet_path: str,
        columns: list[str],
        tile_size: float = 7000.0,
        min_points: int = 200,
    ) -> "TileStore":
        """Build a TileStore by reading all rows from a parquet file."""
        pf = pq.ParquetFile(parquet_path)
        arrays = []
        for record_batch in pf.iter_batches(batch_size=65536, columns=columns):
            batch = np.column_stack([
                record_batch.column(name).to_numpy(zero_copy_only=False).astype(np.float32, copy=False)
                for name in columns
            ])
            arrays.append(batch)
        if not arrays:
            raise ValueError("No rows loaded from parquet file")
        data = np.concatenate(arrays, axis=0)
        print(f"loaded {data.shape[0]:,} rows for tile-aware training", flush=True)
        return cls(data, tile_size=tile_size, min_points=min_points)

    def get_tile(self, tile_index: int) -> np.ndarray:
        """Return all rows for a given tile as [N_tile, features]."""
        return self.data[self.tiles[tile_index]]

    def split_tile_indices(
        self,
        split: str,
        val_fraction: float,
        split_seed: int,
        test_fraction: float = 0.0,
        split_strategy: str = "random",
        stratify_bins: int = 3,
    ) -> np.ndarray:
        """Deterministically split tile indices into train/val sets.

        Uses tile-level splitting: entire tiles go to train or val,
        never split within a tile.
        """
        if split not in {"all", "train", "val", "test"}:
            raise ValueError(f"Unknown split: {split}")
        all_indices = np.arange(self.num_tiles, dtype=np.int64)
        if split == "all":
            return all_indices
        if val_fraction <= 0.0 and test_fraction <= 0.0:
            if split == "val":
                return np.empty(0, dtype=np.int64)
            if split == "test":
                return np.empty(0, dtype=np.int64)
            return all_indices

        if split_strategy == "stratified":
            train_idx, val_idx, test_idx = self._stratified_split_indices(
                val_fraction=val_fraction,
                test_fraction=test_fraction,
                split_seed=split_seed,
                stratify_bins=stratify_bins,
            )
        elif split_strategy == "random":
            train_idx, val_idx, test_idx = self._random_split_indices(
                val_fraction=val_fraction,
                test_fraction=test_fraction,
                split_seed=split_seed,
            )
        else:
            raise ValueError(f"Unknown split_strategy: {split_strategy}")

        if split == "train":
            return train_idx
        if split == "val":
            return val_idx
        return test_idx

    def _random_split_indices(
        self,
        *,
        val_fraction: float,
        test_fraction: float,
        split_seed: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Random tile-level split with exact global val/test counts."""
        rng = np.random.default_rng(split_seed)
        all_indices = np.arange(self.num_tiles, dtype=np.int64)
        shuffled = all_indices.copy()
        rng.shuffle(shuffled)
        n_test = int(round(self.num_tiles * max(0.0, test_fraction)))
        n_val = int(round(self.num_tiles * max(0.0, val_fraction)))
        n_test = min(n_test, self.num_tiles)
        n_val = min(n_val, self.num_tiles - n_test)
        test_idx = np.sort(shuffled[:n_test])
        val_idx = np.sort(shuffled[n_test:n_test + n_val])
        train_idx = np.sort(shuffled[n_test + n_val:])
        return train_idx, val_idx, test_idx

    def _stratified_split_indices(
        self,
        *,
        val_fraction: float,
        test_fraction: float,
        split_seed: int,
        stratify_bins: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Tile-level split stratified by deformation descriptors.

        Strata use tile-level medians of mean velocity, acceleration and
        seasonality. The feature positions match the standard tile-aware
        training column order:
        [easting, northing, height, rmse, mean_velocity, mean_velocity_std,
        acceleration, acceleration_std, seasonality, seasonality_std, ...].
        """
        bins = max(2, int(stratify_bins))
        descriptors = np.array([
            [
                _safe_nanmedian(self.data[idx, 4]),
                _safe_nanmedian(self.data[idx, 6]),
                _safe_nanmedian(self.data[idx, 8]),
            ]
            for idx in self.tiles
        ], dtype=np.float64)
        labels = np.column_stack([
            _quantile_bin(descriptors[:, col], bins)
            for col in range(descriptors.shape[1])
        ])
        stratum_keys, inverse = np.unique(labels, axis=0, return_inverse=True)
        stratum_indices = [np.flatnonzero(inverse == i).astype(np.int64) for i in range(len(stratum_keys))]
        counts = np.array([len(idx) for idx in stratum_indices], dtype=np.int64)

        n_total = int(counts.sum())
        n_test = int(round(n_total * max(0.0, test_fraction)))
        n_val = int(round(n_total * max(0.0, val_fraction)))
        n_test = min(n_test, n_total)
        n_val = min(n_val, n_total - n_test)
        test_counts = _proportional_counts(counts, n_test)
        remaining = counts - test_counts
        val_counts = _proportional_counts(remaining, n_val)

        rng = np.random.default_rng(split_seed)
        train_parts: list[np.ndarray] = []
        val_parts: list[np.ndarray] = []
        test_parts: list[np.ndarray] = []
        for idx, n_t, n_v in zip(stratum_indices, test_counts, val_counts):
            shuffled = idx.copy()
            rng.shuffle(shuffled)
            test_parts.append(shuffled[:n_t])
            val_parts.append(shuffled[n_t:n_t + n_v])
            train_parts.append(shuffled[n_t + n_v:])

        train_idx = np.sort(np.concatenate(train_parts)) if train_parts else np.empty(0, dtype=np.int64)
        val_idx = np.sort(np.concatenate(val_parts)) if val_parts else np.empty(0, dtype=np.int64)
        test_idx = np.sort(np.concatenate(test_parts)) if test_parts else np.empty(0, dtype=np.int64)
        return train_idx, val_idx, test_idx


def iter_tile_batches(
    store: TileStore,
    tiles_per_batch: int,
    *,
    split: str,
    val_fraction: float,
    split_seed: int,
    rng: np.random.Generator,
    test_fraction: float = 0.0,
    split_strategy: str = "random",
    stratify_bins: int = 3,
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
    tile_indices = store.split_tile_indices(
        split,
        val_fraction,
        split_seed,
        test_fraction=test_fraction,
        split_strategy=split_strategy,
        stratify_bins=stratify_bins,
    )
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
    store: TileStore,
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
