"""Compute the EGMS-QA B4 acceleration-strength family table.

B4 contains:
  - B41_acc_abs_p90: p90 of absolute point-level acceleration.
  - B42_european_acceleration_typicality: European acceleration typicality band.
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
OUT_DIR = ROOT / "outputs/tasks/b4"


def _acceleration_typicality(acc_abs_p90: float) -> str:
    if acc_abs_p90 <= 0.491:
        return "low"
    if acc_abs_p90 <= 0.697:
        return "typ_low"
    if acc_abs_p90 <= 0.989:
        return "typ_high"
    if acc_abs_p90 <= 1.404:
        return "high"
    return "extreme"


def _read_one(row: tuple[str, str, str]) -> dict[str, object]:
    tile_id, split, path = row
    with np.load(path) as z:
        acc = z["acceleration"].astype(np.float64, copy=False)
        value = float(np.nanpercentile(np.abs(acc), 90))
    return {
        "tile_id": str(tile_id),
        "split": str(split),
        "B41_acc_abs_p90": value,
        "B42_european_acceleration_typicality": _acceleration_typicality(value),
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
    out_path = out_dir / "b4_final_table.csv"
    df.to_csv(out_path, index=False)
    print(f"wrote {out_path}")
    print(f"n_tiles={len(df)}")
    print(df["B41_acc_abs_p90"].describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).to_string())
    print(df["B42_european_acceleration_typicality"].value_counts().to_string())


if __name__ == "__main__":
    main()
