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
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)

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

    def __post_init__(self) -> None:
        if self.t_start < 0:
            raise ValueError("t_start must be non-negative")
        if self.t_end <= self.t_start:
            raise ValueError("t_end must be greater than t_start")

    @property
    def input_length(self) -> int:
        return self.t_end - self.t_start


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
    data_root : path-like, optional
        Root used to resolve relative paths from the manifest. Defaults to the
        current working directory, matching the release installer layout.
    """

    def __init__(
        self,
        manifest: pd.DataFrame,
        time_window: TimeWindow,
        split_assignments: dict[str, str] | None = None,
        feature_columns_count: int = FEATURE_COLUMNS_COUNT,
        data_root: str | Path | None = None,
        source_time_steps: int | None = None,
    ) -> None:
        required = {"tile_id", "path", "n_points", "centroid_x", "centroid_y"}
        missing = required - set(manifest.columns)
        if missing:
            raise ValueError(f"manifest missing required columns: {missing}")

        if manifest.empty:
            raise ValueError("manifest contains no tiles")
        tile_ids = manifest["tile_id"].astype(str)
        if tile_ids.duplicated().any():
            duplicate = tile_ids[tile_ids.duplicated()].iloc[0]
            raise ValueError(f"manifest contains duplicate tile_id: {duplicate}")
        if feature_columns_count != 2 + len(STATIC_KEYS):
            raise ValueError(
                f"feature_columns_count must be {2 + len(STATIC_KEYS)}, got {feature_columns_count}"
            )
        if (manifest["n_points"].astype(int) <= 0).any():
            raise ValueError("manifest n_points values must be positive")

        self.manifest = manifest.reset_index(drop=True).copy()
        self.time_window = time_window
        self.feature_columns_count = int(feature_columns_count)
        self.num_tiles = len(self.manifest)
        self.data_root = Path(data_root) if data_root is not None else Path.cwd()
        self.source_time_steps = int(source_time_steps) if source_time_steps is not None else None

        self.tile_metadata: list[dict] = [
            {
                "tile_id": str(row["tile_id"]),
                "num_points": int(row["n_points"]),
                "center_easting": float(row["centroid_x"]),
                "center_northing": float(row["centroid_y"]),
                "path": self._resolve_path(row["path"]),
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
            normalized = {
                str(tile_id): "val" if str(split) == "validation" else str(split)
                for tile_id, split in split_assignments.items()
            }
            unknown_ids = set(normalized) - set(id_to_idx)
            missing_ids = set(id_to_idx) - set(normalized)
            invalid_splits = sorted(set(normalized.values()) - set(buckets))
            if unknown_ids:
                raise ValueError(f"split assignments contain unknown tile_id: {sorted(unknown_ids)[0]}")
            if missing_ids:
                raise ValueError(f"split assignment missing tile_id: {sorted(missing_ids)[0]}")
            if invalid_splits:
                raise ValueError(f"invalid split labels: {invalid_splits}")
            for tile_id, split in normalized.items():
                idx = id_to_idx[tile_id]
                buckets[split].append(idx)
            self._split_idx = {
                "train": np.sort(np.asarray(buckets["train"], dtype=np.int64)),
                "val":   np.sort(np.asarray(buckets["val"],   dtype=np.int64)),
                "test":  np.sort(np.asarray(buckets["test"],  dtype=np.int64)),
                "all":   np.arange(self.num_tiles, dtype=np.int64),
            }

        counts = [m["num_points"] for m in self.tile_metadata]
        LOGGER.info(
            f"TileStore: {self.num_tiles} tiles  "
            f"(input_length={time_window.input_length}, t=[{time_window.t_start},{time_window.t_end}))  "
            f"points/tile: min={min(counts)}, median={int(np.median(counts))}, max={max(counts)}"
        )

    def _resolve_path(self, value: object) -> Path:
        path = Path(str(value))
        return path if path.is_absolute() else self.data_root / path

    @classmethod
    def from_manifest(
        cls,
        manifest_path: str | Path,
        data_config_path: str | Path,
        data_root: str | Path | None = None,
    ) -> "TileStore":
        """Build from a released split manifest parquet.

        The manifest has one row per tile with ``tile_id``, ``path`` (relative to
        the checkout root), ``n_points``, centroids, and a ``split`` column. The
        time window and tile-row layout are read from the HF-provided
        ``data_config.json``.
        """
        manifest_path = Path(manifest_path)
        data_config_path = Path(data_config_path)
        if not manifest_path.is_file():
            raise FileNotFoundError(f"manifest does not exist: {manifest_path}")
        if not data_config_path.is_file():
            raise FileNotFoundError(f"data config does not exist: {data_config_path}")
        manifest = pd.read_parquet(manifest_path)
        with data_config_path.open(encoding="utf-8") as handle:
            config = json.load(handle)
        if config.get("schema_version") != "egms-qa-data-config-1.0":
            raise ValueError(f"unsupported data config schema in {data_config_path}")
        time_config = config["time_window"]
        t_start = int(time_config["t_start"])
        t_end = int(time_config["t_end"])
        if int(time_config["input_length"]) != t_end - t_start:
            raise ValueError("data config input_length does not match [t_start,t_end)")
        if time_config.get("end_is_exclusive") is not True:
            raise ValueError("data config must declare an exclusive t_end")
        source_time_steps = int(time_config["source_steps"])
        if t_end > source_time_steps:
            raise ValueError("data config time window exceeds source_steps")
        feature_columns_count = int(config["tile_field_layout"]["feature_columns_count"])
        if "split" not in manifest.columns:
            raise ValueError("released manifest is missing the split column")
        split_assignments = dict(
            zip(manifest["tile_id"].astype(str), manifest["split"].astype(str))
        )
        return cls(
            manifest=manifest,
            time_window=TimeWindow(t_start=t_start, t_end=t_end),
            split_assignments=split_assignments,
            feature_columns_count=feature_columns_count,
            data_root=data_root,
            source_time_steps=source_time_steps,
        )

    def get_tile(self, tile_index: int) -> np.ndarray:
        """Return ``[N, feature_columns_count + input_length]`` row matrix
        with the standard tile layout."""
        meta = self.tile_metadata[tile_index]
        path = Path(meta["path"])
        if not path.is_file():
            raise FileNotFoundError(f"tile file does not exist: {path}")
        with np.load(path, allow_pickle=False) as archive:
            required = {"coords", "time_series", *STATIC_KEYS}
            missing = required - set(archive.files)
            if missing:
                raise ValueError(f"{meta['tile_id']}: NPZ missing fields {sorted(missing)}")
            coords = np.asarray(archive["coords"], dtype=np.float32)
            source_series = np.asarray(archive["time_series"], dtype=np.float32)
            static_values = {
                key: np.asarray(archive[key], dtype=np.float32) for key in STATIC_KEYS
            }

        if coords.ndim != 2 or coords.shape[1] != 2:
            raise ValueError(f"{meta['tile_id']}: coords must have shape [N,2], got {coords.shape}")
        n = coords.shape[0]
        if n != meta["num_points"]:
            raise ValueError(
                f"{meta['tile_id']}: manifest n_points={meta['num_points']}, NPZ has {n}"
            )
        if source_series.ndim != 2 or source_series.shape[0] != n:
            raise ValueError(
                f"{meta['tile_id']}: time_series must have shape [N,T], got {source_series.shape}"
            )
        if self.source_time_steps is not None and source_series.shape[1] != self.source_time_steps:
            raise ValueError(
                f"{meta['tile_id']}: expected {self.source_time_steps} source steps, "
                f"got {source_series.shape[1]}"
            )
        if source_series.shape[1] < self.time_window.t_end:
            raise ValueError(
                f"{meta['tile_id']}: time_series has {source_series.shape[1]} steps, "
                f"cannot read [0,{self.time_window.t_end})"
            )
        ts = source_series[:, self.time_window.t_start : self.time_window.t_end]
        if not np.isfinite(ts).all():
            raise ValueError(f"{meta['tile_id']}: configured time window contains non-finite values")

        static = np.empty((n, len(STATIC_KEYS)), dtype=np.float32)
        for j, key in enumerate(STATIC_KEYS):
            values = static_values[key]
            if values.shape != (n,):
                raise ValueError(f"{meta['tile_id']}: {key} must have shape [N], got {values.shape}")
            static[:, j] = values

        out = np.empty((n, self.feature_columns_count + ts.shape[1]), dtype=np.float32)
        out[:, 0:2] = coords.astype(np.float32, copy=False)
        out[:, 2:self.feature_columns_count] = static
        out[:, self.feature_columns_count:] = ts.astype(np.float32, copy=False)
        return out

    def split_tile_indices(self, split: str) -> np.ndarray:
        """Return the tile indices for a split. The train/val/test assignment is
        precomputed and loaded from the manifest, so no split config is needed."""
        split = "val" if split == "validation" else split
        if split not in self._split_idx:
            raise ValueError(
                f"split={split!r} not available; have {list(self._split_idx)}"
            )
        return self._split_idx[split]
