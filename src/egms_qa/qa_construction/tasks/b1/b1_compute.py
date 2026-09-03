"""Compute the EGMS-QA B11/B12 final table for the 10k EGMS encoder tiles.

B11/B12 ask whether the tile has clear average subsidence relative to observation
noise. It intentionally does not split uplift; positive or near-zero motion is
`no_clear_subsidence`.

This script only reads the two required arrays from each tile npz:
`mean_velocity` and `rmse`.
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
OUT_DIR = ROOT / "outputs/tasks/b1"
EPS = 1e-12


def _read_one(row: tuple[str, str, str]) -> dict[str, object]:
    tile_id, split, path = row
    with np.load(path) as z:
        mv = z["mean_velocity"].astype(np.float64, copy=False)
        rmse = z["rmse"].astype(np.float64, copy=False)

        v_mean = float(np.nanmean(mv))
        rmse_median = float(np.nanmedian(rmse))
        subsidence_snr = -v_mean / (rmse_median + EPS)

        return {
            "tile_id": str(tile_id),
            "split": str(split),
            "B21_mean_velocity_mm_yr": v_mean,
            "A41_median_rmse_mm": rmse_median,
            "B11_subsidence_snr": subsidence_snr,
        }


def _classify_subsidence_snr(x: np.ndarray, tau: float) -> np.ndarray:
    return np.where(x >= tau, "clear_subsidence", "no_clear_subsidence")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(MANIFEST))
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--chunksize", type=int, default=32)
    args = ap.parse_args()

    manifest = pd.read_parquet(args.manifest)
    required = {"tile_id", "split", "path"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"manifest missing columns: {sorted(missing)}")

    rows = [(str(r.tile_id), str(r.split), str(r.path)) for r in manifest.itertuples(index=False)]
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        records = list(ex.map(_read_one, rows, chunksize=args.chunksize))

    df = pd.DataFrame.from_records(records)
    snr = df["B11_subsidence_snr"].to_numpy(np.float64)
    df["B12_clear_subsidence_class"] = _classify_subsidence_snr(snr, args.tau)
    df = df[
        [
            "tile_id",
            "split",
            "B21_mean_velocity_mm_yr",
            "A41_median_rmse_mm",
            "B11_subsidence_snr",
            "B12_clear_subsidence_class",
        ]
    ]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "b1_final_table.csv"
    df.to_csv(csv_path, index=False)

    print(f"wrote {csv_path}")
    print(f"n_tiles={len(df)} tau={args.tau}")
    print(df["B12_clear_subsidence_class"].value_counts().to_string())
    print(pd.crosstab(df["split"], df["B12_clear_subsidence_class"]).to_string())


if __name__ == "__main__":
    main()
