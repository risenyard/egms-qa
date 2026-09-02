"""Compute the EGMS-QA B3 velocity-tail family table.

B3 contains:
  - B31_velocity_p10_mm_yr: sinking-side velocity tail.
  - B32_velocity_p90_mm_yr: upper-side velocity tail.
  - B33_vel_abs_p90_mm_yr: absolute velocity tail.
  - B34_uplift_protected_direction: uplift/non-uplift direction.
  - B35_worst_point_significance: local worst-point significance band.
  - B36_european_velocity_typicality: European velocity typicality band.
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
OUT_DIR = ROOT / "outputs/tasks/b3"


def _direction_from_tails(p10: float, p90: float) -> str:
    return "uplift" if p90 > abs(p10) else "non_uplift"


def _worst_point_significance(vel_abs_p90: float) -> str:
    if vel_abs_p90 < 1.5:
        return "very_low"
    if vel_abs_p90 < 1.9:
        return "low"
    if vel_abs_p90 < 2.24:
        return "moderate"
    if vel_abs_p90 < 2.9:
        return "high"
    return "very_high"


def _velocity_typicality(vel_abs_p90: float) -> str:
    if vel_abs_p90 <= 1.367:
        return "low"
    if vel_abs_p90 <= 2.075:
        return "typ_low"
    if vel_abs_p90 <= 3.149:
        return "typ_high"
    if vel_abs_p90 <= 4.781:
        return "high"
    return "extreme"


def _read_one(row: tuple[str, str, str]) -> dict[str, object]:
    tile_id, split, path = row
    with np.load(path) as z:
        velocity = z["mean_velocity"].astype(np.float64, copy=False)
        p10 = float(np.nanpercentile(velocity, 10))
        p90 = float(np.nanpercentile(velocity, 90))
        vel_abs_p90 = float(np.nanpercentile(np.abs(velocity), 90))
    return {
        "tile_id": str(tile_id),
        "split": str(split),
        "B31_velocity_p10_mm_yr": p10,
        "B32_velocity_p90_mm_yr": p90,
        "B33_vel_abs_p90_mm_yr": vel_abs_p90,
        "B34_uplift_protected_direction": _direction_from_tails(p10, p90),
        "B35_worst_point_significance": _worst_point_significance(vel_abs_p90),
        "B36_european_velocity_typicality": _velocity_typicality(vel_abs_p90),
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
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "b3_final_table.csv"
    df.to_csv(out_path, index=False)
    print(f"wrote {out_path}")
    print(f"n_tiles={len(df)}")
    for col in ["B31_velocity_p10_mm_yr", "B32_velocity_p90_mm_yr", "B33_vel_abs_p90_mm_yr"]:
        print(f"\n[{col}]")
        print(df[col].describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).to_string())
    for col in ["B34_uplift_protected_direction", "B35_worst_point_significance", "B36_european_velocity_typicality"]:
        print(f"\n[{col}]")
        print(df[col].value_counts().to_string())


if __name__ == "__main__":
    main()
