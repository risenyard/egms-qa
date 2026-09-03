"""Compute the EGMS-QA C3 deformation-front family table.

Primary scalar:
  C31_deformation_front_strength_mm_yr =
      p90(abs(mean_velocity difference between adjacent valid 8x8 bins))

Location:
  C32_front_location records the adjacent bin pair with the maximum velocity jump,
  formatted as r{row_a}c{col_a}-r{row_b}c{col_b}.

Class:
  C33_deformation_front_strength_class is a corpus-relative class from C31.
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
OUT_DIR = ROOT / "outputs/tasks/c3"
GRID = 8
TILE = 7000.0
HALF = TILE / 2.0
MIN_BIN_POINTS = 5
C33_LABELS = ["weak", "moderate", "strong", "very_sharp"]
C33_QUANTILES = [0.10, 0.50, 0.90]


def _train_thresholds(df: pd.DataFrame, column: str, quantiles: list[float]) -> list[float]:
    train = df.loc[df["split"].astype(str).eq("train"), column].astype(float)
    if train.empty:
        raise ValueError("train split has no rows for threshold fitting")
    return [float(train.quantile(q)) for q in quantiles]


def _add_front_strength_class(df: pd.DataFrame) -> tuple[pd.DataFrame, list[float]]:
    df = df.copy()
    thresholds = _train_thresholds(df, "C31_deformation_front_strength_mm_yr", C33_QUANTILES)
    df["C33_deformation_front_strength_class"] = pd.cut(
        df["C31_deformation_front_strength_mm_yr"].astype(float),
        bins=[-np.inf, *thresholds, np.inf],
        labels=C33_LABELS,
        include_lowest=True,
    ).astype(str)
    return df, thresholds


def _bin_index(coords: np.ndarray) -> np.ndarray:
    coords64 = coords.astype(np.float64, copy=False)
    centered = coords64 - coords64.mean(0, keepdims=True)
    bx = np.clip(np.floor((centered[:, 0] + HALF) / TILE * GRID).astype(np.int64), 0, GRID - 1)
    by = np.clip(np.floor((centered[:, 1] + HALF) / TILE * GRID).astype(np.int64), 0, GRID - 1)
    return by * GRID + bx


def _neighbor_diffs(grid: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int, int, int]]]:
    diffs = []
    pairs: list[tuple[int, int, int, int]] = []
    for y in range(GRID):
        for x in range(GRID):
            v = grid[y, x]
            if not np.isfinite(v):
                continue
            if x + 1 < GRID and np.isfinite(grid[y, x + 1]):
                diffs.append(abs(v - grid[y, x + 1]))
                pairs.append((y, x, y, x + 1))
            if y + 1 < GRID and np.isfinite(grid[y + 1, x]):
                diffs.append(abs(v - grid[y + 1, x]))
                pairs.append((y, x, y + 1, x))
    return np.asarray(diffs, dtype=np.float64), pairs


def _format_pair(pair: tuple[int, int, int, int] | None) -> str:
    if pair is None:
        return "none"
    y0, x0, y1, x1 = pair
    return f"r{y0}c{x0}-r{y1}c{x1}"


def _read_one(row: tuple[str, str, str]) -> dict[str, object]:
    tile_id, split, path = row
    with np.load(path) as z:
        coords = z["coords"]
        mv = z["mean_velocity"].astype(np.float64, copy=False)

    bidx = _bin_index(coords)
    grid = np.full((GRID, GRID), np.nan, dtype=np.float64)
    valid_bins = 0
    for b in range(GRID * GRID):
        sel = bidx == b
        if int(sel.sum()) >= MIN_BIN_POINTS:
            y, x = divmod(b, GRID)
            grid[y, x] = float(np.nanmean(mv[sel]))
            valid_bins += 1

    diffs, pairs = _neighbor_diffs(grid)
    if diffs.size == 0:
        p90 = 0.0
        max_pair = None
    else:
        p90 = float(np.nanpercentile(diffs, 90))
        max_pair = pairs[int(np.nanargmax(diffs))]

    return {
        "tile_id": str(tile_id),
        "split": str(split),
        "C31_deformation_front_strength_mm_yr": p90,
        "C32_front_location": _format_pair(max_pair),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(MANIFEST))
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))
    ap.add_argument("--chunksize", type=int, default=32)
    args = ap.parse_args()

    manifest = pd.read_parquet(args.manifest)
    rows = [(str(r.tile_id), str(r.split), str(r.path)) for r in manifest.itertuples(index=False)]
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        records = list(ex.map(_read_one, rows, chunksize=args.chunksize))

    df = pd.DataFrame.from_records(records)
    df, thresholds = _add_front_strength_class(df)
    df = df[
        [
            "tile_id",
            "split",
            "C31_deformation_front_strength_mm_yr",
            "C32_front_location",
            "C33_deformation_front_strength_class",
        ]
    ]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "c3_final_table.csv"
    df.to_csv(out_path, index=False)

    print(f"wrote {out_path}")
    print(f"n_tiles={len(df)}")
    print(f"C33 train thresholds p10/p50/p90={thresholds}")
    print(df["C31_deformation_front_strength_mm_yr"].describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).to_string())
    print(df["C32_front_location"].value_counts().head(12).to_string())
    print(df["C33_deformation_front_strength_class"].value_counts().to_string())


if __name__ == "__main__":
    main()
