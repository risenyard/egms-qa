"""Combine EGMS-QA A11 global-instability shard outputs."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

CLASS_LABELS = ["stable", "mildly_sensitive", "highly_sensitive", "extreme"]


def threshold_mask(tile: pd.DataFrame, threshold_pool: str) -> pd.Series:
    split = tile["split"].astype(str)
    if threshold_pool == "train":
        return split.eq("train")
    if threshold_pool == "train_val":
        return split.isin(["train", "val"])
    raise ValueError(f"unknown threshold_pool={threshold_pool!r}")


def add_tail_strata(tile: pd.DataFrame, threshold_pool: str) -> tuple[pd.DataFrame, dict]:
    tile = tile.copy()
    drift = tile["A11_global_angular_drift"].astype(float)
    pool = threshold_mask(tile, threshold_pool)
    if not pool.any():
        raise ValueError(f"threshold pool {threshold_pool!r} has no rows")
    threshold_drift = drift[pool]
    q75, q95, q99 = [float(threshold_drift.quantile(q)) for q in (0.75, 0.95, 0.99)]
    tile["A11_global_drift_deg"] = drift * 180.0
    tile["A12_representation_stability_class"] = pd.cut(
        drift,
        bins=[-np.inf, q75, q95, q99, np.inf],
        labels=CLASS_LABELS,
        include_lowest=True,
    ).astype(str)
    counts = (
        tile["A12_representation_stability_class"]
        .value_counts()
        .reindex(CLASS_LABELS)
        .fillna(0)
        .astype(int)
    )
    class_summary = {
        "scheme": "corpus_relative_tail_strata",
        "threshold_source": "empirical train-split A11_global_angular_drift percentiles",
        "threshold_pool": threshold_pool,
        "threshold_pool_n": int(pool.sum()),
        "thresholds": {
            "stable_max_p75": q75,
            "mildly_sensitive_max_p95": q95,
            "highly_sensitive_max_p99": q99,
        },
        "thresholds_degrees": {
            "stable_max_p75": q75 * 180.0,
            "mildly_sensitive_max_p95": q95 * 180.0,
            "highly_sensitive_max_p99": q99 * 180.0,
        },
        "classes": {
            label: {
                "count": int(counts[label]),
                "fraction": float(counts[label] / len(tile)),
            }
            for label in CLASS_LABELS
        },
        "class_definitions": {
            "stable": "drift <= p75",
            "mildly_sensitive": "p75 < drift <= p95",
            "highly_sensitive": "p95 < drift <= p99",
            "extreme": "drift > p99",
        },
    }
    return tile, class_summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", default="outputs/tasks/a1/work")
    ap.add_argument("--out-path", default="outputs/tasks/a1/a1_final_table.csv")
    ap.add_argument("--num-shards", type=int, default=10)
    ap.add_argument("--threshold-pool", default="train", choices=["train", "train_val"])
    ap.add_argument("--keep-work", action="store_true")
    args = ap.parse_args()

    base = Path(args.base_dir)
    out_path = Path(args.out_path)
    tile_parts = []
    for shard in range(args.num_shards):
        shard_dir = base / "shards" / f"shard_{shard}"
        tile_path = shard_dir / "a1_global_instability_by_tile.csv"
        if not tile_path.exists():
            raise FileNotFoundError(f"missing output for shard {shard}: {shard_dir}")
        tile_parts.append(pd.read_csv(tile_path))

    tile = pd.concat(tile_parts, ignore_index=True).sort_values("tile_id").reset_index(drop=True)
    if tile["tile_id"].duplicated().any():
        dup = tile.loc[tile["tile_id"].duplicated(), "tile_id"].head().tolist()
        raise ValueError(f"duplicate tile ids in combined output, examples={dup}")
    tile, class_summary = add_tail_strata(tile, args.threshold_pool)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tile.to_csv(out_path, index=False)

    summary = {
        "task": "A11",
        "n_tiles": int(len(tile)),
        "target": "A11_global_angular_drift",
        "classification": class_summary,
        "output": str(out_path),
    }
    if not args.keep_work and base.name == "work" and base.exists():
        shutil.rmtree(base)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
