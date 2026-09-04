"""TileStore — per-tile NPZ reader with precomputed split.

Constructed from a released split manifest parquet (one row per tile, with
``path``, ``tile_id``, ``centroid_x``, ``centroid_y``, ``n_points``, ``split``).
Each tile's full feature row is materialised on demand by reading the NPZ;
nothing is loaded eagerly.

Tile row layout:
    [easting, northing,
     height, rmse,
     mean_velocity, mean_velocity_std,
     acceleration, acceleration_std,
     seasonality, seasonality_std,
     time_series(T) ...]

The time series is sliced to ``[t_start, t_end)``; the window is guaranteed
NaN-free by the pool filter used to build the released tile set.
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
    stored_steps: int | None = None

    def __post_init__(self) -> None:
        if self.t_start < 0 or self.t_end <= self.t_start:
            raise ValueError(
                f"invalid time window [{self.t_start},{self.t_end})"
            )
        if self.stored_steps is not None and self.t_end > self.stored_steps:
            raise ValueError(
                f"time window [{self.t_start},{self.t_end}) exceeds "
                f"stored_steps={self.stored_steps}"
            )

    @property
    def input_length(self) -> int:
        return self.t_end - self.t_start

    @classmethod
    def from_config(cls, config: dict) -> "TimeWindow":
        """Read and validate the stored-axis contract from ``data_config``.

        ``stored_steps`` is required; the public release has no implicit or
        compatibility time-axis fallback.
        """
        try:
            raw = config["time_window"]
            t_start = int(raw["t_start"])
            t_end = int(raw["t_end"])
            stored_steps = int(raw["stored_steps"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "data_config.json must define time_window.t_start, t_end, "
                "and stored_steps"
            ) from exc
        window = cls(t_start=t_start, t_end=t_end, stored_steps=stored_steps)
        declared_length = int(raw.get("input_length", window.input_length))
        if declared_length != window.input_length:
            raise ValueError(
                f"time_window.input_length={declared_length} does not match "
                f"[{t_start},{t_end}) ({window.input_length})"
            )
        return window


class TileStore:
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
            f"TileStore: {self.num_tiles} tiles  "
            f"(input_length={time_window.input_length}, t=[{time_window.t_start},{time_window.t_end}))  "
            f"points/tile: min={min(counts)}, median={int(np.median(counts))}, max={max(counts)}",
            flush=True,
        )

    @classmethod
    def from_manifest(cls, manifest_path: str | Path, data_config_path: str | Path) -> "TileStore":
        """Build from a released split manifest parquet.

        The manifest has one row per tile with ``tile_id``, ``path`` (relative to
        the checkout root), ``n_points``, centroids, and a ``split`` column. The
        time window and tile-row layout are read from ``data_config.json``.
        There is intentionally no implicit time-axis default: the same code can
        read the model-ready release only when its current contract is supplied.
        """
        manifest = pd.read_parquet(manifest_path)
        config_path = Path(data_config_path)
        if not config_path.is_file():
            raise FileNotFoundError(f"data config is required: {config_path}")
        with config_path.open(encoding="utf-8") as f:
            cfg = json.load(f)
        time_window = TimeWindow.from_config(cfg)
        feature_columns_count = int(cfg["tile_field_layout"]["feature_columns_count"])
        split_assignments = None
        if "split" in manifest.columns:
            split_assignments = dict(
                zip(manifest["tile_id"].astype(str), manifest["split"].astype(str))
            )
        return cls(
            manifest=manifest,
            time_window=time_window,
            split_assignments=split_assignments,
            feature_columns_count=feature_columns_count,
        )

    def get_tile(self, tile_index: int) -> np.ndarray:
        """Return ``[N, feature_columns_count + input_length]`` row matrix
        with the standard tile layout."""
        meta = self.tile_metadata[tile_index]
        with np.load(meta["path"], allow_pickle=False) as z:
            coords = z["coords"]                  # (N, 2) easting, northing
            full_ts = z["time_series"]
            if full_ts.ndim != 2:
                raise ValueError(
                    f"{meta['path']}: time_series must be 2-D, got {full_ts.shape}"
                )
            expected_steps = self.time_window.stored_steps
            if expected_steps is not None and full_ts.shape[1] != expected_steps:
                raise ValueError(
                    f"{meta['path']}: stored time_series has {full_ts.shape[1]} steps; "
                    f"data_config requires {expected_steps}. Refusing implicit or "
                    "repeated cropping."
                )
            ts = full_ts[:, self.time_window.t_start:self.time_window.t_end]
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

    def split_tile_indices(self, split: str) -> np.ndarray:
        """Return the tile indices for a split. The train/val/test assignment is
        precomputed and loaded from the manifest, so no split config is needed."""
        if split not in self._split_idx:
            raise ValueError(
                f"split={split!r} not available; have {list(self._split_idx)}"
            )
        return self._split_idx[split]
