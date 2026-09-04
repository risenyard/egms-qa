"""Per-tile NPZ reader for the released EGMS-QA data contract.

Each materialized row has the layout ``[x, y, eight static descriptors,
time_series]``. The time-axis contract comes exclusively from
``data_config.json``; no implicit crop is applied.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)

FEATURE_COLUMNS_COUNT = 10
DATA_CONFIG_SCHEMA = "egms-qa-data-config-1.1"
RELEASE_STORED_STEPS = 294
STATIC_KEYS = (
    "height",
    "rmse",
    "mean_velocity",
    "mean_velocity_std",
    "acceleration",
    "acceleration_std",
    "seasonality",
    "seasonality_std",
)


@dataclass(frozen=True)
class TimeWindow:
    t_start: int
    t_end: int  # exclusive
    stored_steps: int | None = None

    def __post_init__(self) -> None:
        if self.t_start < 0 or self.t_end <= self.t_start:
            raise ValueError(f"invalid time window [{self.t_start},{self.t_end})")
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
        """Read and validate the stored-axis contract from ``data_config``."""
        try:
            raw = config["time_window"]
            t_start = int(raw["t_start"])
            t_end = int(raw["t_end"])
            stored_steps = int(
                raw["stored_steps"] if "stored_steps" in raw else raw["source_steps"]
            )
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
        if "end_is_exclusive" in raw and raw["end_is_exclusive"] is not True:
            raise ValueError("data config must declare an exclusive t_end")
        return window


class TileStore:
    """Lazy NPZ store backed by a split manifest."""

    def __init__(
        self,
        manifest: pd.DataFrame,
        time_window: TimeWindow,
        split_assignments: dict[str, str] | None = None,
        feature_columns_count: int = FEATURE_COLUMNS_COUNT,
        data_root: str | Path | None = None,
        data_config: dict | None = None,
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
                f"feature_columns_count must be {2 + len(STATIC_KEYS)}, "
                f"got {feature_columns_count}"
            )
        if (manifest["n_points"].astype(int) <= 0).any():
            raise ValueError("manifest n_points values must be positive")

        self.manifest = manifest.reset_index(drop=True).copy()
        self.time_window = time_window
        self.data_config = dict(data_config or {})
        self.feature_columns_count = int(feature_columns_count)
        self.num_tiles = len(self.manifest)
        self.data_root = Path(data_root) if data_root is not None else Path.cwd()
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
                "all": np.arange(self.num_tiles, dtype=np.int64)
            }
        else:
            id_to_idx = {item["tile_id"]: i for i, item in enumerate(self.tile_metadata)}
            buckets: dict[str, list[int]] = {"train": [], "val": [], "test": []}
            normalized = {
                str(tile_id): "val" if str(split) == "validation" else str(split)
                for tile_id, split in split_assignments.items()
            }
            unknown_ids = set(normalized) - set(id_to_idx)
            missing_ids = set(id_to_idx) - set(normalized)
            invalid_splits = sorted(set(normalized.values()) - set(buckets))
            if unknown_ids:
                raise ValueError(
                    f"split assignments contain unknown tile_id: {sorted(unknown_ids)[0]}"
                )
            if missing_ids:
                raise ValueError(f"split assignment missing tile_id: {sorted(missing_ids)[0]}")
            if invalid_splits:
                raise ValueError(f"invalid split labels: {invalid_splits}")
            for tile_id, split in normalized.items():
                buckets[split].append(id_to_idx[tile_id])
            self._split_idx = {
                key: np.sort(np.asarray(values, dtype=np.int64))
                for key, values in buckets.items()
            }
            self._split_idx["all"] = np.arange(self.num_tiles, dtype=np.int64)

        counts = [item["num_points"] for item in self.tile_metadata]
        LOGGER.info(
            "TileStore: %d tiles (input_length=%d, t=[%d,%d)); "
            "points/tile: min=%d, median=%d, max=%d",
            self.num_tiles,
            time_window.input_length,
            time_window.t_start,
            time_window.t_end,
            min(counts),
            int(np.median(counts)),
            max(counts),
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
        """Build a store from the released manifest and data configuration."""
        manifest_path = Path(manifest_path)
        config_path = Path(data_config_path)
        if not manifest_path.is_file():
            raise FileNotFoundError(f"manifest does not exist: {manifest_path}")
        if not config_path.is_file():
            raise FileNotFoundError(f"data config is required: {config_path}")

        manifest = pd.read_parquet(manifest_path)
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if config.get("schema_version") != DATA_CONFIG_SCHEMA:
            raise ValueError(
                f"unsupported data config schema {config.get('schema_version')!r}; "
                f"expected {DATA_CONFIG_SCHEMA!r}"
            )
        time_window = TimeWindow.from_config(config)
        if (
            time_window.stored_steps,
            time_window.t_start,
            time_window.t_end,
            time_window.input_length,
        ) != (RELEASE_STORED_STEPS, 0, RELEASE_STORED_STEPS, RELEASE_STORED_STEPS):
            raise ValueError(
                "released tiles must store the model-ready [0,294) window directly; "
                f"found stored_steps={time_window.stored_steps}, "
                f"window=[{time_window.t_start},{time_window.t_end})"
            )
        try:
            feature_columns_count = int(
                config["tile_field_layout"]["feature_columns_count"]
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "data_config.json must define tile_field_layout.feature_columns_count"
            ) from exc
        if "split" not in manifest.columns:
            raise ValueError("released manifest is missing the split column")
        split_assignments = dict(
            zip(manifest["tile_id"].astype(str), manifest["split"].astype(str))
        )
        store = cls(
            manifest=manifest,
            time_window=time_window,
            split_assignments=split_assignments,
            feature_columns_count=feature_columns_count,
            data_root=data_root,
            data_config=config,
        )
        return store

    def get_tile(self, tile_index: int) -> np.ndarray:
        """Return one ``[N, feature_columns_count + input_length]`` tile."""
        meta = self.tile_metadata[tile_index]
        path = Path(meta["path"])
        if not path.is_file():
            raise FileNotFoundError(f"tile file does not exist: {path}")

        with np.load(path, allow_pickle=False) as archive:
            basic_missing = {"coords", "time_series"} - set(archive.files)
            if basic_missing:
                raise ValueError(
                    f"{meta['tile_id']}: NPZ missing fields {sorted(basic_missing)}"
                )
            coords = np.asarray(archive["coords"], dtype=np.float32)
            series = np.asarray(archive["time_series"], dtype=np.float32)

            if coords.ndim != 2 or coords.shape[1] != 2:
                raise ValueError(
                    f"{meta['tile_id']}: coords must have shape [N,2], got {coords.shape}"
                )
            n_points = coords.shape[0]
            if n_points != meta["num_points"]:
                raise ValueError(
                    f"{meta['tile_id']}: manifest n_points={meta['num_points']}, "
                    f"NPZ has {n_points}"
                )
            if series.ndim != 2 or series.shape[0] != n_points:
                raise ValueError(
                    f"{meta['tile_id']}: time_series must have shape [N,T], "
                    f"got {series.shape}"
                )
            expected_steps = self.time_window.stored_steps
            if expected_steps is not None and series.shape[1] != expected_steps:
                raise ValueError(
                    f"{path}: stored time_series has {series.shape[1]} steps; "
                    f"data_config requires {expected_steps}. Refusing implicit or "
                    "repeated cropping."
                )

            static_missing = set(STATIC_KEYS) - set(archive.files)
            if static_missing:
                raise ValueError(
                    f"{meta['tile_id']}: NPZ missing fields {sorted(static_missing)}"
                )
            static_values = {
                key: np.asarray(archive[key], dtype=np.float32) for key in STATIC_KEYS
            }

        time_series = series[:, self.time_window.t_start : self.time_window.t_end]
        if not np.isfinite(time_series).all():
            raise ValueError(
                f"{meta['tile_id']}: configured time window contains non-finite values"
            )

        static = np.empty((n_points, len(STATIC_KEYS)), dtype=np.float32)
        for column, key in enumerate(STATIC_KEYS):
            values = static_values[key]
            if values.shape != (n_points,):
                raise ValueError(
                    f"{meta['tile_id']}: {key} must have shape [N], got {values.shape}"
                )
            static[:, column] = values

        output = np.empty(
            (n_points, self.feature_columns_count + time_series.shape[1]),
            dtype=np.float32,
        )
        output[:, :2] = coords
        output[:, 2 : self.feature_columns_count] = static
        output[:, self.feature_columns_count :] = time_series
        return output

    def split_tile_indices(self, split: str) -> np.ndarray:
        """Return indices for ``train``, ``validation``/``val``, ``test``, or ``all``."""
        split = "val" if split == "validation" else split
        if split not in self._split_idx:
            raise ValueError(f"split={split!r} not available; have {list(self._split_idx)}")
        return self._split_idx[split]
