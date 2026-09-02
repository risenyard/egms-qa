"""Combine A21 shard outputs into the final tile table."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


CLASS_LABELS = ["reconstructable", "mildly_hard", "high_error", "unreliable"]


def threshold_mask(table: pd.DataFrame, threshold_pool: str) -> pd.Series:
    split = table["split"].astype(str)
    if threshold_pool == "train":
        return split.eq("train")
    if threshold_pool == "train_val":
        return split.isin(["train", "val"])
    raise ValueError(f"unknown threshold_pool={threshold_pool!r}")


def add_tail_classes(table: pd.DataFrame, threshold_pool: str) -> tuple[pd.DataFrame, dict]:
    table = table.copy()
    score = table["A21_masked_global_mse_z"].astype(float)
    pool = threshold_mask(table, threshold_pool)
    if not pool.any():
        raise ValueError(f"threshold pool {threshold_pool!r} has no rows")
    threshold_score = score[pool]
    threshold_rmse_mm = table.loc[pool, "A21_masked_global_rmse_mm"].astype(float)
    q75, q95, q99 = [float(threshold_score.quantile(q)) for q in (0.75, 0.95, 0.99)]
    r75, r95, r99 = [float(threshold_rmse_mm.quantile(q)) for q in (0.75, 0.95, 0.99)]
    table["A22_reconstruction_reliability_class"] = pd.cut(
        score,
        bins=[-np.inf, q75, q95, q99, np.inf],
        labels=CLASS_LABELS,
        include_lowest=True,
    ).astype(str)
    counts = (
        table["A22_reconstruction_reliability_class"]
        .value_counts()
        .reindex(CLASS_LABELS)
        .fillna(0)
        .astype(int)
    )
    class_summary = {
        "scheme": "corpus_relative_tail_strata",
        "threshold_pool": threshold_pool,
        "threshold_pool_n": int(pool.sum()),
        "target": "A21_masked_global_mse_z",
        "thresholds": {
            "reconstructable_max_p75": q75,
            "mildly_hard_max_p95": q95,
            "high_error_max_p99": q99,
        },
        "thresholds_rmse_z": {
            "reconstructable_max_p75": float(np.sqrt(q75)),
            "mildly_hard_max_p95": float(np.sqrt(q95)),
            "high_error_max_p99": float(np.sqrt(q99)),
        },
        "thresholds_rmse_mm": {
            "reconstructable_max_p75": r75,
            "mildly_hard_max_p95": r95,
            "high_error_max_p99": r99,
        },
        "classes": {
            label: {
                "count": int(counts[label]),
                "fraction": float(counts[label] / len(table)),
            }
            for label in CLASS_LABELS
        },
        "class_definitions": {
            "reconstructable": "mse <= p75",
            "mildly_hard": "p75 < mse <= p95",
            "high_error": "p95 < mse <= p99",
            "unreliable": "mse > p99",
        },
    }
    return table, class_summary


def summarize(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    return {
        "mean": float(np.nanmean(values)),
        "std": float(np.nanstd(values)),
        "min": float(np.nanmin(values)),
        "p25": float(np.nanpercentile(values, 25)),
        "p50": float(np.nanpercentile(values, 50)),
        "p75": float(np.nanpercentile(values, 75)),
        "p90": float(np.nanpercentile(values, 90)),
        "p95": float(np.nanpercentile(values, 95)),
        "p99": float(np.nanpercentile(values, 99)),
        "max": float(np.nanmax(values)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", default="outputs/tasks/a2/work")
    ap.add_argument("--out-path", default="outputs/tasks/a2/a2_final_table.csv")
    ap.add_argument("--num-shards", type=int, default=20)
    ap.add_argument("--threshold-pool", default="train", choices=["train", "train_val"])
    ap.add_argument("--keep-work", action="store_true")
    args = ap.parse_args()

    base = Path(args.base_dir)
    parts = []
    summaries = []
    for shard in range(args.num_shards):
        shard_dir = base / "shards" / f"shard_{shard}"
        table_path = shard_dir / "a2_by_tile.csv"
        summary_path = shard_dir / "a2_summary.json"
        if not table_path.exists():
            raise FileNotFoundError(f"missing shard table: {table_path}")
        parts.append(pd.read_csv(table_path))
        if summary_path.exists():
            summaries.append(json.loads(summary_path.read_text()))

    table = pd.concat(parts, ignore_index=True).sort_values("tile_idx").reset_index(drop=True)
    if table["tile_id"].duplicated().any():
        dup = table.loc[table["tile_id"].duplicated(), "tile_id"].head().tolist()
        raise ValueError(f"duplicate tile ids in combined output, examples={dup}")
    table, class_summary = add_tail_classes(table, args.threshold_pool)

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_path, index=False)

    summary = {
        "task": "A21",
        "n_tiles": int(len(table)),
        "target": "A21_masked_global_mse_z",
        "metric_diagnostics": {
            "A21_masked_global_mse_z": summarize(table["A21_masked_global_mse_z"].to_numpy(dtype=float)),
            "A21_masked_global_rmse_mm": summarize(table["A21_masked_global_rmse_mm"].to_numpy(dtype=float)),
        },
        "classification": class_summary,
        "output": str(out_path),
    }
    if summaries:
        first = summaries[0]
        summary.update({
            "method": first.get("method"),
            "checkpoint": first.get("checkpoint"),
            "data_config": first.get("data_config"),
            "mask_start": first.get("mask_start"),
            "mask_end": first.get("mask_end"),
            "mask_ratio": first.get("mask_ratio"),
            "max_tile_points": first.get("max_tile_points"),
        })
    if not args.keep_work and base.name == "work" and base.exists():
        shutil.rmtree(base)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
