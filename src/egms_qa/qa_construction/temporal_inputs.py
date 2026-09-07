"""Read the published model window and reconstruct its physical time axis."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd

YEAR_DAYS = 365.25


@dataclass(frozen=True)
class TimeAxis:
    stored_steps: int
    t_start: int
    t_end: int
    original_index_offset: int
    original_epoch_year: float
    cadence_days: float

    @classmethod
    def from_file(cls, path: str | Path) -> "TimeAxis":
        config = json.loads(Path(path).read_text(encoding="utf-8"))
        if config.get("schema_version") != "egms-qa-data-config-1.1":
            raise ValueError("expected the published egms-qa-data-config-1.1 contract")
        raw = config["time_window"]
        axis = cls(**{key: raw[key] for key in cls.__dataclass_fields__})
        for name in ("stored_steps", "t_start", "t_end", "original_index_offset"):
            value = getattr(axis, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an integer")
        if (axis.stored_steps, axis.t_start, axis.t_end) != (294, 0, 294):
            raise ValueError("temporal tasks require the stored 294-step [0,294) window")
        if raw.get("input_length") != axis.t_end - axis.t_start:
            raise ValueError("input_length does not match the stored time window")
        if axis.cadence_days <= 0 or not np.isfinite(axis.cadence_days):
            raise ValueError("cadence_days must be finite and positive")
        if axis.original_index_offset < 0 or not np.isfinite(axis.original_epoch_year):
            raise ValueError("invalid source time origin")
        if (raw["original_t_start"] != axis.original_index_offset
                or raw["original_t_end"] != axis.original_index_offset + axis.t_end - axis.t_start
                or raw["original_t_end"] > raw["original_source_steps"]):
            raise ValueError("source-window indices do not match the stored window and source offset")
        return axis

    @property
    def start_year(self) -> float:
        return self.original_epoch_year + self.original_index_offset * self.cadence_days / YEAR_DAYS

    @property
    def delta_years(self) -> float:
        return self.cadence_days / YEAR_DAYS

    @property
    def years(self) -> np.ndarray:
        return self.start_year + np.arange(self.t_end - self.t_start, dtype=np.float64) * self.delta_years

    def tile_median(self, path: str | Path) -> np.ndarray:
        with np.load(path, allow_pickle=False) as arrays:
            series = arrays["time_series"]
            if series.ndim != 2 or series.shape[1] != self.stored_steps or not len(series):
                raise ValueError(f"expected nonempty time_series [N,{self.stored_steps}] in {path}")
            window = series[:, self.t_start:self.t_end].astype(np.float64, copy=False)
        return np.nanmedian(window, axis=0)


def read_tile_manifest(path: str | Path, source_tiles_root: str | Path | None = None) -> pd.DataFrame:
    """Resolve runtime or release-relative paths without changing the manifest."""
    frame = pd.read_parquet(path)
    required = {"tile_id", "split", "path"}
    if not required <= set(frame):
        raise ValueError(f"manifest is missing {sorted(required - set(frame))}")
    if frame.empty or frame["tile_id"].duplicated().any():
        raise ValueError("manifest must contain unique tile IDs and at least one tile")
    if frame[["tile_id", "split", "path"]].isna().any().any():
        raise ValueError("manifest keys and paths must not be missing")
    if not set(frame["split"]) <= {"train", "val", "test"}:
        raise ValueError("manifest splits must be train, val or test")
    if source_tiles_root is not None:
        def resolve(value: str) -> str:
            tile = Path(value)
            if tile.is_absolute():
                return str(tile)
            if tile.parts[:2] not in {("data", "tiles"), ("artifacts", "source_tiles")}:
                raise ValueError(f"unrecognized source tile path: {tile}")
            return str(Path(source_tiles_root).joinpath(*tile.parts[2:]))
        frame = frame.copy()
        frame["path"] = frame["path"].map(resolve)
    return frame
