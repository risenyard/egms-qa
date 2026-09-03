"""A41/A42 measurement noise outputs for EGMS-QA.

A41/A42 are intentionally simple for QA generation:
  - scalar: tile median of point-level EGMS RMSE
  - label: fixed absolute-mm noise class
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(".")
ENCODER_DATA = ROOT / "data/encoder"

MANIFEST = ENCODER_DATA / "manifest/split.parquet"
DATA_CONFIG = ENCODER_DATA / "manifest/data_config.json"
OUT_PATH = ROOT / "outputs/tasks/a4/a4_final_table.csv"
COL_RMSE = 3


def noise_class(median_rmse_mm: float) -> str:
    if median_rmse_mm < 1.0:
        return "low_noise"
    if median_rmse_mm < 1.5:
        return "moderate_noise"
    if median_rmse_mm < 2.0:
        return "high_noise"
    return "very_high_noise"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(MANIFEST))
    ap.add_argument("--data-config", default=str(DATA_CONFIG))
    ap.add_argument("--out-path", default=str(OUT_PATH))
    ap.add_argument("--log-every", type=int, default=1000)
    args = ap.parse_args()

    from egms_encoder.data.lazy_tile_store import LazyTileStore, TimeWindow

    cfg = json.load(open(args.data_config))
    tw = TimeWindow(
        t_start=int(cfg["time_window"]["t_start"]),
        t_end=int(cfg["time_window"]["t_end"]),
    )
    manifest = pd.read_parquet(args.manifest)
    split_assignments = dict(zip(manifest["tile_id"].astype(str), manifest["split"].astype(str)))
    store = LazyTileStore(manifest=manifest, time_window=tw, split_assignments=split_assignments)

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    class_counts: Counter[str] = Counter()
    t0 = time.monotonic()
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "tile_id",
                "split",
                "A41_median_rmse_mm",
                "A42_noise_level_class",
            ],
        )
        writer.writeheader()
        for i, row in enumerate(manifest.itertuples(index=False)):
            tile = store.get_tile(i)
            rmse = tile[:, COL_RMSE].astype(np.float64)
            rmse = rmse[np.isfinite(rmse)]
            if rmse.size == 0:
                median_rmse = float("nan")
                label = "unknown"
            else:
                median_rmse = float(np.median(rmse))
                label = noise_class(median_rmse)
            class_counts[label] += 1
            writer.writerow({
                "tile_id": str(row.tile_id),
                "split": str(row.split),
                "A41_median_rmse_mm": f"{median_rmse:.6f}",
                "A42_noise_level_class": label,
            })
            if args.log_every and ((i + 1) % args.log_every == 0 or i + 1 == len(manifest)):
                dt = time.monotonic() - t0
                print(f"  {i+1}/{len(manifest)} tiles  {dt:.1f}s  {(i+1)/max(dt, 1e-6):.1f} tiles/s", flush=True)

    summary = {
        "task": "A41",
        "n_tiles": int(len(manifest)),
        "scalar": "A41_median_rmse_mm",
        "label": "A42_noise_level_class",
        "class_counts": dict(class_counts),
        "output": str(out_path),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
