"""Compute EGMS-QA D4 temporal composition family.

D4 summarizes which temporal process is dominant at tile level.

Delivered targets:
  - D41_temporal_dominant_process
  - D42_temporal_evolution_archetype

Inputs are existing B-family primitive strengths:
  - trend strength: B33_vel_abs_p90_mm_yr
  - seasonal strength: B51_seasonality_p90
  - acceleration strength: B41_acc_abs_p90

Because these inputs have different units, D41 compares their train-fitted
corpus percentile ranks rather than their raw values.

D42 is a readable archetype derived from D41 plus already delivered temporal
labels/scalars. It introduces no new measurement and no new threshold.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(".")
OUT_DIR = ROOT / "outputs/tasks/d4"
B3_TABLE = ROOT / "outputs/tasks/b3/b3_final_table.csv"
B4_TABLE = ROOT / "outputs/tasks/b4/b4_final_table.csv"
B5_TABLE = ROOT / "outputs/tasks/b5/b5_final_table.csv"
D1_TABLE = ROOT / "outputs/tasks/d1/d1_final_table.csv"
D2_TABLE = ROOT / "outputs/tasks/d2/d2_final_table.csv"
D3_TABLE = ROOT / "outputs/tasks/d3/d3_final_table.csv"

QUIET_TOP_RANK_MAX = 0.30
DOMINANCE_MARGIN_MIN = 0.15

BASE_INPUTS = {
    "trend": "B33_vel_abs_p90_mm_yr",
    "seasonal": "B51_seasonality_p90",
    "acceleration": "B41_acc_abs_p90",
}
CLASS_ORDER = [
    "low_activity",
    "trend_dominant",
    "seasonal_dominant",
    "acceleration_dominant",
    "mixed",
]
D42_CLASS_ORDER = [
    "low_activity",
    "linear_trend_dominated",
    "curved_trend_dominated",
    "regime_change_trend_dominated",
    "coherent_seasonal_dominated",
    "incoherent_seasonal_dominated",
    "intensifying_acceleration_dominated",
    "weakening_acceleration_dominated",
    "uncertain_direction_acceleration_dominated",
    "trend_seasonal_mixed",
    "trend_acceleration_mixed",
    "seasonal_acceleration_mixed",
]


def _classify(top_process: str, top_rank: float, margin: float) -> str:
    if not np.isfinite(top_rank) or top_rank < QUIET_TOP_RANK_MAX:
        return "low_activity"
    if np.isfinite(margin) and margin >= DOMINANCE_MARGIN_MIN:
        return f"{top_process}_dominant"
    return "mixed"


def _top_two_process_pair(row: pd.Series) -> str:
    values = {
        "trend": row["D41_trend_rank"],
        "seasonal": row["D41_seasonal_rank"],
        "acceleration": row["D41_acceleration_rank"],
    }
    top_two = set(sorted(values, key=values.get, reverse=True)[:2])
    if top_two == {"trend", "seasonal"}:
        return "trend_seasonal"
    if top_two == {"trend", "acceleration"}:
        return "trend_acceleration"
    if top_two == {"seasonal", "acceleration"}:
        return "seasonal_acceleration"
    return "unknown_pair"


def _d42_archetype(row: pd.Series) -> str:
    d41 = row["D41_temporal_dominant_process"]

    if d41 == "low_activity":
        return "low_activity"

    if d41 == "trend_dominant":
        shape = row["D11_long_term_trend_shape"]
        if shape == "linear_trend":
            return "linear_trend_dominated"
        if shape == "curved_trend":
            return "curved_trend_dominated"
        return "regime_change_trend_dominated"

    if d41 == "seasonal_dominant":
        peak = row["D21_dominant_seasonal_peak"]
        if isinstance(peak, str) and peak != "no_clear_seasonal_peak":
            return "coherent_seasonal_dominated"
        return "incoherent_seasonal_dominated"

    if d41 == "acceleration_dominant":
        d31 = row["D31_motion_intensification_mm_yr2"]
        if not np.isfinite(d31) or d31 == 0:
            return "uncertain_direction_acceleration_dominated"
        if d31 > 0:
            return "intensifying_acceleration_dominated"
        return "weakening_acceleration_dominated"

    if d41 == "mixed":
        return f"{_top_two_process_pair(row)}_mixed"

    return "unknown"


def _train_percentile_rank(values: pd.Series, split: pd.Series) -> np.ndarray:
    arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=np.float64)
    split_arr = split.astype(str).to_numpy()
    train = np.sort(arr[(split_arr == "train") & np.isfinite(arr)])
    if train.size == 0:
        raise ValueError(f"no finite train rows for {values.name}")
    out = np.full(arr.shape, np.nan, dtype=np.float64)
    finite = np.isfinite(arr)
    left = np.searchsorted(train, arr[finite], side="left")
    right = np.searchsorted(train, arr[finite], side="right")
    ranks = right.astype(np.float64) / float(train.size)
    tie = right > left
    ranks[tie] = (left[tie].astype(np.float64) + 1.0 + right[tie].astype(np.float64)) / (2.0 * train.size)
    out[finite] = ranks
    return out


def _plot_distribution(df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(18, 9), constrained_layout=True)

    ax = axes[0, 0]
    counts = df["D41_temporal_dominant_process"].value_counts().reindex(CLASS_ORDER).fillna(0)
    ax.barh(counts.index, counts.values, color=["#9d9d9d", "#4c78a8", "#54a24b", "#f58518", "#b279a2"])
    ax.set_xlabel("tiles")
    ax.set_ylabel("D41 class")

    ax = axes[0, 1]
    ax.hist(df["D41_top_rank"], bins=60, color="#4c78a8")
    ax.axvline(QUIET_TOP_RANK_MAX, color="#222222", linestyle="--", linewidth=1.0)
    ax.set_xlabel("D41 top process rank")
    ax.set_ylabel("tiles")

    ax = axes[1, 0]
    ax.hist(df["D41_dominance_margin"], bins=60, color="#f58518")
    ax.axvline(DOMINANCE_MARGIN_MIN, color="#222222", linestyle="--", linewidth=1.0)
    ax.set_xlabel("D41 dominance margin")
    ax.set_ylabel("tiles")

    ax = axes[1, 1]
    top_counts = df["D41_top_process_candidate"].value_counts().reindex(BASE_INPUTS.keys()).fillna(0)
    ax.bar(top_counts.index, top_counts.values, color=["#4c78a8", "#54a24b", "#f58518"])
    ax.set_xlabel("top process before quiet/mixed rule")
    ax.set_ylabel("tiles")

    ax = axes[0, 2]
    d42_counts = (
        df["D42_temporal_evolution_archetype"].value_counts().reindex(D42_CLASS_ORDER).fillna(0)
    )
    ax.barh(d42_counts.index, d42_counts.values, color="#72b7b2")
    ax.set_xlabel("tiles")
    ax.set_ylabel("D42 archetype")

    ax = axes[1, 2]
    pair_counts = (
        df.loc[df["D41_temporal_dominant_process"] == "mixed", "D42_temporal_evolution_archetype"]
        .value_counts()
        .reindex(["trend_seasonal_mixed", "trend_acceleration_mixed", "seasonal_acceleration_mixed"])
        .fillna(0)
    )
    ax.barh(pair_counts.index, pair_counts.values, color=["#4c78a8", "#f58518", "#54a24b"])
    ax.set_xlabel("tiles")
    ax.set_ylabel("mixed top-two archetype")

    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--b3-table", default=str(B3_TABLE))
    ap.add_argument("--b4-table", default=str(B4_TABLE))
    ap.add_argument("--b5-table", default=str(B5_TABLE))
    ap.add_argument("--d1-table", default=str(D1_TABLE))
    ap.add_argument("--d2-table", default=str(D2_TABLE))
    ap.add_argument("--d3-table", default=str(D3_TABLE))
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    args = ap.parse_args()

    b3 = pd.read_csv(args.b3_table)[["tile_id", "split", BASE_INPUTS["trend"]]]
    b4 = pd.read_csv(args.b4_table)[["tile_id", BASE_INPUTS["acceleration"]]]
    b5 = pd.read_csv(args.b5_table)[["tile_id", BASE_INPUTS["seasonal"]]]
    d1 = pd.read_csv(args.d1_table)[["tile_id", "D11_long_term_trend_shape"]]
    d2 = pd.read_csv(args.d2_table)[["tile_id", "D21_dominant_seasonal_peak"]]
    d3 = pd.read_csv(args.d3_table)[["tile_id", "D31_motion_intensification_mm_yr2"]]
    df = b3.merge(b4, on="tile_id", validate="one_to_one").merge(
        b5, on="tile_id", validate="one_to_one"
    )
    df = (
        df.merge(d1, on="tile_id", validate="one_to_one")
        .merge(d2, on="tile_id", validate="one_to_one")
        .merge(d3, on="tile_id", validate="one_to_one")
    )

    for process, col in BASE_INPUTS.items():
        df[f"D41_{process}_rank"] = _train_percentile_rank(df[col], df["split"])

    rank_cols = [f"D41_{process}_rank" for process in BASE_INPUTS]
    ranks = df[rank_cols].to_numpy(dtype=np.float64)
    processes = np.asarray(list(BASE_INPUTS.keys()), dtype=object)
    order = np.argsort(-ranks, axis=1)
    df["D41_top_process_candidate"] = processes[order[:, 0]]
    df["D41_top_rank"] = ranks[np.arange(len(df)), order[:, 0]]
    df["D41_second_rank"] = ranks[np.arange(len(df)), order[:, 1]]
    df["D41_dominance_margin"] = df["D41_top_rank"] - df["D41_second_rank"]
    df["D41_temporal_dominant_process"] = [
        _classify(process, top, margin)
        for process, top, margin in zip(
            df["D41_top_process_candidate"],
            df["D41_top_rank"],
            df["D41_dominance_margin"],
        )
    ]
    df["D42_temporal_evolution_archetype"] = df.apply(_d42_archetype, axis=1)

    final_cols = [
        "tile_id",
        "split",
        "D41_temporal_dominant_process",
        "D42_temporal_evolution_archetype",
        "D41_trend_rank",
        "D41_seasonal_rank",
        "D41_acceleration_rank",
        "D41_top_process_candidate",
        "D41_top_rank",
        "D41_second_rank",
        "D41_dominance_margin",
        "D11_long_term_trend_shape",
        "D21_dominant_seasonal_peak",
        "D31_motion_intensification_mm_yr2",
        BASE_INPUTS["trend"],
        BASE_INPUTS["seasonal"],
        BASE_INPUTS["acceleration"],
    ]
    final = df[final_cols].copy()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    table_path = out_dir / "d4_final_table.csv"
    summary_path = out_dir / "d4_final_summary.json"
    plot_path = out_dir / "d4_final_distribution.png"
    final.to_csv(table_path, index=False)

    score_cols = [
        BASE_INPUTS["trend"],
        BASE_INPUTS["seasonal"],
        BASE_INPUTS["acceleration"],
        "D41_trend_rank",
        "D41_seasonal_rank",
        "D41_acceleration_rank",
        "D41_top_rank",
        "D41_second_rank",
        "D41_dominance_margin",
    ]
    summary = {
        "n_tiles": int(len(final)),
        "tasks": ["D41_temporal_dominant_process", "D42_temporal_evolution_archetype"],
        "base_inputs": BASE_INPUTS,
        "rank_method": "train-split empirical percentile rank applied to all 10k tiles",
        "rank_fit_split": "train",
        "rank_fit_n": int(df["split"].astype(str).eq("train").sum()),
        "D41_rule": {
            "low_activity": f"D41_top_rank < {QUIET_TOP_RANK_MAX}",
            "dominant": f"D41_dominance_margin >= {DOMINANCE_MARGIN_MIN}",
            "mixed": "not low_activity and margin below dominant threshold",
            "threshold_type": "corpus-relative",
        },
        "D42_rule": {
            "description": "Readable archetype derived from D41 plus D11/D21/D31; no new measurement or threshold.",
            "low_activity": "D41 == low_activity",
            "trend_dominant": "D11 linear/curved/regime-change-or-complex trend shape",
            "seasonal_dominant": "D21 clear vs no-clear seasonal peak",
            "acceleration_dominant": "sign of D31 motion intensification",
            "mixed": "unordered top-two pair among D41 trend/seasonal/acceleration ranks",
        },
        "class_order": CLASS_ORDER,
        "D41_class_counts": final["D41_temporal_dominant_process"]
        .value_counts()
        .reindex(CLASS_ORDER)
        .fillna(0)
        .astype(int)
        .to_dict(),
        "D42_class_order": D42_CLASS_ORDER,
        "D42_class_counts": final["D42_temporal_evolution_archetype"]
        .value_counts()
        .reindex(D42_CLASS_ORDER)
        .fillna(0)
        .astype(int)
        .to_dict(),
        "top_process_candidate_counts": final["D41_top_process_candidate"].value_counts().to_dict(),
        "score_summary": final[score_cols]
        .describe(percentiles=[0.01, 0.05, 0.1, 0.2, 0.25, 0.3, 0.5, 0.75, 0.9, 0.95, 0.99])
        .to_dict(),
    }
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)
    _plot_distribution(final, plot_path)

    print(f"wrote {table_path}")
    print(f"wrote {summary_path}")
    print(f"wrote {plot_path}")
    print("D41")
    print(final["D41_temporal_dominant_process"].value_counts().reindex(CLASS_ORDER).fillna(0).astype(int).to_string())
    print("D42")
    print(final["D42_temporal_evolution_archetype"].value_counts().reindex(D42_CLASS_ORDER).fillna(0).astype(int).to_string())
    print(final[score_cols].describe(
        percentiles=[0.01, 0.05, 0.1, 0.2, 0.25, 0.3, 0.5, 0.75, 0.9, 0.95, 0.99]
    ).to_string())


if __name__ == "__main__":
    main()
