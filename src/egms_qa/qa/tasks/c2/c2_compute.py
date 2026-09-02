"""Compute the EGMS-QA C2 spatial-concentration family table.

C2 contains:
  - C21 scalar:
  spatial concentration of raw motion magnitude.
  - C22 class:
  corpus-relative spatial concentration class from C21.

For each valid 8x8 spatial bin:
  bin_mean_abs_velocity = mean(abs(point mean_velocity))

For each tile:
  C21_spatial_concentration_score = Gini(bin_mean_abs_velocity over valid bins)
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
OUT_DIR = ROOT / "outputs/tasks/c2"
GRID = 8
TILE = 7000.0
HALF = TILE / 2.0
MIN_BIN_POINTS = 5
C22_LABELS = ["diffuse", "mildly_concentrated", "concentrated", "highly_concentrated"]
C22_QUANTILES = [0.10, 0.50, 0.90]


def _train_thresholds(df: pd.DataFrame, column: str, quantiles: list[float]) -> list[float]:
    train = df.loc[df["split"].astype(str).eq("train"), column].astype(float)
    if train.empty:
        raise ValueError("train split has no rows for threshold fitting")
    return [float(train.quantile(q)) for q in quantiles]


def _add_concentration_class(df: pd.DataFrame) -> tuple[pd.DataFrame, list[float]]:
    df = df.copy()
    thresholds = _train_thresholds(df, "C21_spatial_concentration_score", C22_QUANTILES)
    df["C22_spatial_concentration_class"] = pd.cut(
        df["C21_spatial_concentration_score"].astype(float),
        bins=[-np.inf, *thresholds, np.inf],
        labels=C22_LABELS,
        include_lowest=True,
    ).astype(str)
    return df, thresholds


def _gini(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = np.clip(values[np.isfinite(values)], 0.0, None)
    if values.size < 2:
        return 0.0
    total = float(values.sum())
    if total <= 0:
        return 0.0
    x = np.sort(values)
    idx = np.arange(1, x.size + 1, dtype=np.float64)
    return float((2.0 * np.sum(idx * x)) / (x.size * total) - (x.size + 1.0) / x.size)


def _bin_index(coords: np.ndarray) -> np.ndarray:
    coords64 = coords.astype(np.float64, copy=False)
    centered = coords64 - coords64.mean(0, keepdims=True)
    bx = np.clip(np.floor((centered[:, 0] + HALF) / TILE * GRID).astype(np.int64), 0, GRID - 1)
    by = np.clip(np.floor((centered[:, 1] + HALF) / TILE * GRID).astype(np.int64), 0, GRID - 1)
    return by * GRID + bx


def _read_one(row: tuple[str, str, str]) -> dict[str, object]:
    tile_id, split, path = row
    with np.load(path) as z:
        coords = z["coords"]
        mv = z["mean_velocity"].astype(np.float64, copy=False)

    motion = np.abs(mv)
    bidx = _bin_index(coords)

    bin_values = []
    for b in range(GRID * GRID):
        sel = bidx == b
        if int(sel.sum()) >= MIN_BIN_POINTS:
            bin_values.append(float(np.mean(motion[sel])))

    score = _gini(np.asarray(bin_values, dtype=np.float64))
    return {
        "tile_id": str(tile_id),
        "split": str(split),
        "C21_spatial_concentration_score": score,
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
    df, thresholds = _add_concentration_class(df)
    df = df[["tile_id", "split", "C21_spatial_concentration_score", "C22_spatial_concentration_class"]]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "c2_final_table.csv"
    df.to_csv(out_path, index=False)

    print(f"wrote {out_path}")
    print(f"n_tiles={len(df)}")
    print(f"C22 train thresholds p10/p50/p90={thresholds}")
    print(df["C21_spatial_concentration_score"].describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).to_string())
    print(df["C22_spatial_concentration_class"].value_counts().to_string())


if __name__ == "__main__":
    main()
