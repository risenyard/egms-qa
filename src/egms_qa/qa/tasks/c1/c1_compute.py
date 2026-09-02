"""Compute the EGMS-QA C1 moving-fraction family table.

C1 contains:
  - C11 scalar:
  fraction of points whose absolute mean velocity is at least one RMSE.
  - C12 class:
  corpus-relative moving extent class from C11.
  - C13 location:
  8x8 bin with the largest mean absolute velocity.

Formula:
  C11_noise_aware_moving_fraction = mean(abs(mean_velocity) / rmse >= 1.0)
"""
from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(".")
MANIFEST = ROOT / "data/encoder/manifest/split.parquet"
OUT_DIR = ROOT / "outputs/tasks/c1"
EPS = 1e-12
GRID = 8
TILE = 7000.0
HALF = TILE / 2.0
MIN_BIN_POINTS = 5
C12_LABELS = ["limited", "partial", "broad", "widespread"]
C12_QUANTILES = [0.10, 0.50, 0.90]


def _train_thresholds(df: pd.DataFrame, column: str, quantiles: list[float]) -> list[float]:
    train = df.loc[df["split"].astype(str).eq("train"), column].astype(float)
    if train.empty:
        raise ValueError("train split has no rows for threshold fitting")
    return [float(train.quantile(q)) for q in quantiles]


def _add_moving_extent_class(df: pd.DataFrame) -> tuple[pd.DataFrame, list[float]]:
    df = df.copy()
    thresholds = _train_thresholds(df, "C11_noise_aware_moving_fraction", C12_QUANTILES)
    df["C12_motion_extent_class"] = pd.cut(
        df["C11_noise_aware_moving_fraction"].astype(float),
        bins=[-np.inf, *thresholds, np.inf],
        labels=C12_LABELS,
        include_lowest=True,
    ).astype(str)
    return df, thresholds


def _bin_index(coords: np.ndarray) -> np.ndarray:
    coords64 = coords.astype(np.float64, copy=False)
    centered = coords64 - coords64.mean(0, keepdims=True)
    bx = np.clip(np.floor((centered[:, 0] + HALF) / TILE * GRID).astype(np.int64), 0, GRID - 1)
    by = np.clip(np.floor((centered[:, 1] + HALF) / TILE * GRID).astype(np.int64), 0, GRID - 1)
    return by * GRID + bx


def _format_bin(bin_id: int | None) -> str:
    if bin_id is None:
        return "none"
    row, col = divmod(int(bin_id), GRID)
    return f"r{row}c{col}"


def _dominant_velocity_bin(coords: np.ndarray, mv: np.ndarray) -> str:
    finite = np.isfinite(mv) & np.isfinite(coords).all(axis=1)
    if int(finite.sum()) < MIN_BIN_POINTS:
        return "none"

    coords_finite = coords[finite]
    abs_velocity = np.abs(mv[finite])
    bidx = _bin_index(coords_finite)

    best_bin: int | None = None
    best_mean_abs_velocity = -np.inf
    best_n_points = -1
    for b in range(GRID * GRID):
        sel = bidx == b
        n_points = int(sel.sum())
        if n_points < MIN_BIN_POINTS:
            continue
        mean_abs_velocity = float(np.nanmean(abs_velocity[sel]))
        if mean_abs_velocity > best_mean_abs_velocity or (
            mean_abs_velocity == best_mean_abs_velocity and n_points > best_n_points
        ):
            best_bin = b
            best_mean_abs_velocity = mean_abs_velocity
            best_n_points = n_points

    if best_bin is None:
        return "none"
    return _format_bin(best_bin)


def _read_one(row: tuple[str, str, str]) -> dict[str, object]:
    tile_id, split, path = row
    with np.load(path) as z:
        coords = z["coords"]
        mv = z["mean_velocity"].astype(np.float64, copy=False)
        rmse = z["rmse"].astype(np.float64, copy=False)
        snr = np.abs(mv) / (rmse + EPS)
        value = float(np.nanmean(snr >= 1.0))
        return {
            "tile_id": str(tile_id),
            "split": str(split),
            "C11_noise_aware_moving_fraction": value,
            "C13_moving_bin_location": _dominant_velocity_bin(coords, mv),
        }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(MANIFEST))
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))
    ap.add_argument("--chunksize", type=int, default=32)
    ap.add_argument("--max-tiles", type=int, default=0)
    args = ap.parse_args()

    manifest = pd.read_parquet(args.manifest)
    if args.max_tiles:
        manifest = manifest.iloc[: args.max_tiles].copy()
    rows = [(str(r.tile_id), str(r.split), str(r.path)) for r in manifest.itertuples(index=False)]
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        records = list(ex.map(_read_one, rows, chunksize=args.chunksize))

    df = pd.DataFrame.from_records(records)
    df, thresholds = _add_moving_extent_class(df)
    df = df[
        [
            "tile_id",
            "split",
            "C11_noise_aware_moving_fraction",
            "C12_motion_extent_class",
            "C13_moving_bin_location",
        ]
    ]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "c1_final_table.csv"
    df.to_csv(out_path, index=False)

    print(f"wrote {out_path}")
    print(f"n_tiles={len(df)}")
    print(f"C12 train thresholds p10/p50/p90={thresholds}")
    print(df["C11_noise_aware_moving_fraction"].describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).to_string())
    print(df["C12_motion_extent_class"].value_counts().to_string())
    print(df["C13_moving_bin_location"].value_counts().head(12).to_string())


if __name__ == "__main__":
    main()
