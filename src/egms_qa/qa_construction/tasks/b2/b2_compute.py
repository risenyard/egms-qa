"""Compute the EGMS-QA B2 mean-velocity family table.

B2 contains:
  - B21_mean_velocity_mm_yr: tile mean of point-level mean_velocity.
  - B22_mean_subsidence_intensity_band: European relative intensity band.

The B22 uplift override uses the same uplift-protected direction rule as B34.
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
OUT_DIR = ROOT / "outputs/tasks/b2"


def _direction_from_tails(p10: float, p90: float) -> str:
    return "uplift" if p90 > abs(p10) else "non_uplift"


def _intensity_band(mean_velocity: float, direction: str) -> str:
    if direction == "uplift":
        return "uplift"
    if mean_velocity <= -1.47:
        return "high"
    if mean_velocity <= -1.215:
        return "high_mid"
    if mean_velocity <= -0.971:
        return "mid"
    if mean_velocity <= -0.657:
        return "low_mid"
    return "low"


def _read_one(row: tuple[str, str, str]) -> dict[str, object]:
    tile_id, split, path = row
    with np.load(path) as z:
        velocity = z["mean_velocity"].astype(np.float64, copy=False)
        mean_velocity = float(np.nanmean(velocity))
        p10 = float(np.nanpercentile(velocity, 10))
        p90 = float(np.nanpercentile(velocity, 90))
        direction = _direction_from_tails(p10, p90)
    return {
        "tile_id": str(tile_id),
        "split": str(split),
        "B21_mean_velocity_mm_yr": mean_velocity,
        "B34_uplift_protected_direction": direction,
        "B22_mean_subsidence_intensity_band": _intensity_band(mean_velocity, direction),
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
    out_path = out_dir / "b2_final_table.csv"
    df.to_csv(out_path, index=False)
    print(f"wrote {out_path}")
    print(f"n_tiles={len(df)}")
    print(df["B21_mean_velocity_mm_yr"].describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).to_string())
    print(df["B22_mean_subsidence_intensity_band"].value_counts().to_string())


if __name__ == "__main__":
    main()
