"""LazyTileStore — per-tile npz reader with precomputed split.


Constructed from a manifest parquet (one row per tile, with ``path``,
``tile_id``, ``centroid_x``, ``centroid_y``, ``n_points`` …) and an optional
spatial blocked split JSON. Each tile's full feature row is materialised
on demand by reading the npz; nothing is loaded eagerly.

Tile row layout matches ``TileStore.get_tile``:
    [easting, northing,
     height, rmse,
     mean_velocity, mean_velocity_std,
     acceleration, acceleration_std,
     seasonality, seasonality_std,
     time_series(T) ...]

The time series is sliced to ``[t_start, t_end)``; the resulting
window is guaranteed NaN-free by the pool filter (verified by
``scripts/v4_verify_trim_zero_nan.py``).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

FEATURE_COLUMNS_COUNT = 10  # easting, northing + 8 static descriptors
STATIC_KEYS = (
    "height", "rmse",
    "mean_velocity", "mean_velocity_std",
    "acceleration", "acceleration_std",
    "seasonality", "seasonality_std",
)


@dataclass(frozen=True)
class TimeWindow:
    t_start: int
    t_end: int  # exclusive

    @property
    def input_length(self) -> int:
        return self.t_end - self.t_start


class LazyTileStore:
    """Per-tile npz reader with precomputed metadata + split.

    Parameters
    ----------
    manifest : pd.DataFrame
        Must contain columns: ``tile_id``, ``path``, ``n_points``,
        ``centroid_x``, ``centroid_y``. Row order defines tile indices.
    time_window : TimeWindow
        Time-axis slice applied to every loaded tile.
    split_assignments : dict[str, str] | None
        Optional ``tile_id -> {"train","val","test"}`` mapping.
    feature_columns_count : int
        Kept at 10 to match the tile row layout.
    """

    def __init__(
        self,
        manifest: pd.DataFrame,
        time_window: TimeWindow,
        split_assignments: dict[str, str] | None = None,
        feature_columns_count: int = FEATURE_COLUMNS_COUNT,
    ) -> None:
        required = {"tile_id", "path", "n_points", "centroid_x", "centroid_y"}
        missing = required - set(manifest.columns)
        if missing:
            raise ValueError(f"manifest missing required columns: {missing}")

        self.manifest = manifest.reset_index(drop=True).copy()
        self.time_window = time_window
        self.feature_columns_count = int(feature_columns_count)
        self.num_tiles = len(self.manifest)

        self.tile_metadata: list[dict] = [
            {
                "tile_id": str(row["tile_id"]),
                "num_points": int(row["n_points"]),
                "center_easting": float(row["centroid_x"]),
                "center_northing": float(row["centroid_y"]),
                "path": str(row["path"]),
            }
            for _, row in self.manifest.iterrows()
        ]

        if split_assignments is None:
            self._split_idx: dict[str, np.ndarray] = {
                "all": np.arange(self.num_tiles, dtype=np.int64),
            }
        else:
            id_to_idx = {m["tile_id"]: i for i, m in enumerate(self.tile_metadata)}
            buckets: dict[str, list[int]] = {"train": [], "val": [], "test": []}
            for tile_id, split in split_assignments.items():
                idx = id_to_idx.get(tile_id)
                if idx is None or split not in buckets:
                    continue
                buckets[split].append(idx)
            self._split_idx = {
                "train": np.sort(np.asarray(buckets["train"], dtype=np.int64)),
                "val":   np.sort(np.asarray(buckets["val"],   dtype=np.int64)),
                "test":  np.sort(np.asarray(buckets["test"],  dtype=np.int64)),
                "all":   np.arange(self.num_tiles, dtype=np.int64),
            }

        counts = [m["num_points"] for m in self.tile_metadata]
        print(
            f"LazyTileStore: {self.num_tiles} tiles  "
            f"(input_length={time_window.input_length}, t=[{time_window.t_start},{time_window.t_end}))  "
            f"points/tile: min={min(counts)}, median={int(np.median(counts))}, max={max(counts)}",
            flush=True,
        )

    @classmethod
    def from_config(cls, config_path: str | Path) -> "LazyTileStore":
        """Build directly from ``data/processed/v4/v4_data_config.json``.

        The config is the single source of truth for paths, time window,
        and split file.
        """
        config_path = Path(config_path)
        with open(config_path) as f:
            cfg = json.load(f)
        repo_root = _infer_repo_root(config_path)

        sample_path = repo_root / cfg["files"]["sample_10k"]
        manifest = pd.read_parquet(sample_path)
        window = TimeWindow(
            t_start=int(cfg["time_window"]["t_start"]),
            t_end=int(cfg["time_window"]["t_end"]),
        )

        split_path = repo_root / "data/processed/v4/v4_split.json"
        split_assignments: dict[str, str] | None = None
        if split_path.exists():
            with open(split_path) as f:
                split_doc = json.load(f)
            split_assignments = {}
            for name in ("train", "val", "test"):
                for tid in split_doc.get(name, []):
                    split_assignments[str(tid)] = name

        return cls(
            manifest=manifest,
            time_window=window,
            split_assignments=split_assignments,
            feature_columns_count=int(cfg["tile_field_layout"]["feature_columns_count"]),
        )

    def get_tile(self, tile_index: int) -> np.ndarray:
        """Return ``[N, feature_columns_count + input_length]`` row matrix
        with the standard tile layout."""
        meta = self.tile_metadata[tile_index]
        z = np.load(meta["path"])
        coords = z["coords"]                  # (N, 2) easting, northing
        ts = z["time_series"][:, self.time_window.t_start:self.time_window.t_end]
        n = coords.shape[0]

        static = np.empty((n, len(STATIC_KEYS)), dtype=np.float32)
        for j, key in enumerate(STATIC_KEYS):
            if key in z.files:
                static[:, j] = z[key].astype(np.float32, copy=False)
            else:
                static[:, j] = 0.0

        out = np.empty((n, self.feature_columns_count + ts.shape[1]), dtype=np.float32)
        out[:, 0:2] = coords.astype(np.float32, copy=False)
        out[:, 2:self.feature_columns_count] = static
        out[:, self.feature_columns_count:] = ts.astype(np.float32, copy=False)
        return out

    def split_tile_indices(self, split: str, *args, **kwargs) -> np.ndarray:
        """Ignores legacy split-config args (val_fraction, split_seed,
        test_fraction, split_strategy, stratify_bins) because the split is
        precomputed and loaded from disk. Extra args are accepted only for
        signature compatibility with iter_tile_batches.
        """
        if split not in self._split_idx:
            raise ValueError(
                f"split={split!r} not available; have {list(self._split_idx)}"
            )
        return self._split_idx[split]


def _infer_repo_root(config_path: Path) -> Path:
    """Walk up from config until we find a directory containing 'src/'."""
    p = config_path.resolve().parent
    for parent in [p, *p.parents]:
        if (parent / "src").is_dir() and (parent / "data").is_dir():
            return parent
    return p
