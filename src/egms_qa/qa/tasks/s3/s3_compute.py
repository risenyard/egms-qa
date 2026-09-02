"""Compute S3 representation-monitoring consistency tasks.

S31 compares how rare a tile looks in encoder representation space against how
rare it looks under the A/B/C/D monitoring scalar system. Both sides are mapped
to the same train-percentile p0-p99 scale before subtraction.

S32 is the train z-score five-class relation derived from S31.
S33 is the monitoring-side most distinctive dimension.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(".")
OUT_DIR = ROOT / "outputs/tasks/s3"

SOURCES = [
    (ROOT / "outputs/tasks/s2/s2_final_table.csv", ["S21_local_isolation_score"]),
    (ROOT / "outputs/tasks/a4/a4_final_table.csv", ["A41_median_rmse_mm"]),
    (ROOT / "outputs/tasks/b3/b3_final_table.csv", ["B33_vel_abs_p90_mm_yr"]),
    (ROOT / "outputs/tasks/b4/b4_final_table.csv", ["B41_acc_abs_p90"]),
    (ROOT / "outputs/tasks/b5/b5_final_table.csv", ["B51_seasonality_p90"]),
    (ROOT / "outputs/tasks/c1/c1_final_table.csv", ["C11_noise_aware_moving_fraction"]),
    (ROOT / "outputs/tasks/c2/c2_final_table.csv", ["C21_spatial_concentration_score"]),
    (ROOT / "outputs/tasks/c3/c3_final_table.csv", ["C31_deformation_front_strength_mm_yr"]),
    (ROOT / "outputs/tasks/c4/c4_final_table.csv", ["C41_fast_tail_bin_fraction"]),
    (ROOT / "outputs/tasks/d1/d1_final_table.csv", ["D12_trend_order_mean", "D13_top_changepoint_probability"]),
    (ROOT / "outputs/tasks/d2/d2_final_table.csv", ["D22_phase_coherence"]),
    (ROOT / "outputs/tasks/d3/d3_final_table.csv", ["D31_motion_intensification_mm_yr2"]),
]

AXES = {
    "quality": ["A41_median_rmse_mm"],
    "motion": ["B33_vel_abs_p90_mm_yr", "B41_acc_abs_p90", "B51_seasonality_p90"],
    "spatial": [
        "C11_noise_aware_moving_fraction",
        "C21_spatial_concentration_score",
        "C31_deformation_front_strength_mm_yr",
        "C41_fast_tail_bin_fraction",
    ],
    "temporal": [
        "D12_trend_order_mean",
        "D13_top_changepoint_probability",
        "D22_phase_coherence",
        "D31_motion_intensification_mm_yr2",
    ],
}

S31_COL = "S31_representation_monitoring_rarity_gap_p"
S32_COL = "S32_representation_monitoring_rarity_relation"
S33_COL = "S33_monitoring_distinctive_dimension"

S32_CLASS_ORDER = [
    "strong_monitoring_excess",
    "moderate_monitoring_excess",
    "aligned",
    "moderate_encoder_excess",
    "strong_encoder_excess",
]

S33_CLASS_ORDER = ["quality", "motion", "spatial", "temporal"]


def merge_sources() -> pd.DataFrame:
    base = None
    for path, cols in SOURCES:
        missing_path = not path.exists()
        if missing_path:
            raise FileNotFoundError(path)
        df = pd.read_csv(path, usecols=["tile_id", "split", *cols])
        if base is None:
            base = df
        else:
            base = base.merge(df, on=["tile_id", "split"], how="left", validate="one_to_one")
    if base is None:
        raise RuntimeError("No source tables configured.")
    return base


def empirical_percentile(values: np.ndarray, train_values: np.ndarray) -> np.ndarray:
    train_values = np.sort(train_values[np.isfinite(train_values)])
    out = np.full(len(values), np.nan, dtype=np.float64)
    finite = np.isfinite(values)
    if len(train_values) == 0:
        return out
    left = np.searchsorted(train_values, values[finite], side="left")
    right = np.searchsorted(train_values, values[finite], side="right")
    out[finite] = ((left + right) / 2.0 + 0.5) / len(train_values)
    return np.clip(out, 0.0, 1.0)


def train_percentile_p99(values: np.ndarray, train_mask: np.ndarray) -> np.ndarray:
    return 99.0 * empirical_percentile(values, values[train_mask])


def compute_s31(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_mask = df["split"].astype(str).to_numpy() == "train"
    work = df.copy()

    s21 = work["S21_local_isolation_score"].to_numpy(dtype=np.float64)
    work["embedding_rarity_p"] = train_percentile_p99(s21, train_mask)

    scalar_p_cols: dict[str, str] = {}
    for cols in AXES.values():
        for col in cols:
            values = work[col].to_numpy(dtype=np.float64)
            p_col = f"{col}__p"
            work[p_col] = train_percentile_p99(values, train_mask)
            scalar_p_cols[col] = p_col

    axis_cols = []
    for axis_name, cols in AXES.items():
        p_cols = [scalar_p_cols[col] for col in cols]
        axis_col = f"{axis_name}_axis_max_p"
        work[axis_col] = np.nanmax(work[p_cols].to_numpy(dtype=np.float64), axis=1)
        axis_cols.append(axis_col)

    work["monitoring_axis_mean_raw_p"] = np.nanmean(work[axis_cols].to_numpy(dtype=np.float64), axis=1)
    raw = work["monitoring_axis_mean_raw_p"].to_numpy(dtype=np.float64)
    work["monitoring_rarity_p"] = train_percentile_p99(raw, train_mask)
    work[S31_COL] = work["embedding_rarity_p"] - work["monitoring_rarity_p"]

    final = work[["tile_id", "split", S31_COL]].copy()
    final, s32_thresholds = add_s32(final)

    diagnostics = work[
        [
            "tile_id",
            "split",
            "embedding_rarity_p",
            "monitoring_rarity_p",
            "quality_axis_max_p",
            "motion_axis_max_p",
            "spatial_axis_max_p",
            "temporal_axis_max_p",
            S31_COL,
        ]
    ].copy()
    diagnostics.attrs["s32_thresholds"] = s32_thresholds
    final = add_s33(final, diagnostics)
    return final, diagnostics


def add_s32(final: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    train_gap = final.loc[final["split"].astype(str) == "train", S31_COL].to_numpy(dtype=np.float64)
    train_mean = float(np.mean(train_gap))
    train_std = float(np.std(train_gap, ddof=1))
    thresholds = {
        "train_mean": train_mean,
        "train_std": train_std,
        "z_neg_1p96": train_mean - 1.96 * train_std,
        "z_neg_1": train_mean - train_std,
        "z_pos_1": train_mean + train_std,
        "z_pos_1p96": train_mean + 1.96 * train_std,
    }

    gap = final[S31_COL].to_numpy(dtype=np.float64)
    relation = np.full(len(final), "aligned", dtype=object)
    relation[(gap <= thresholds["z_neg_1"]) & (gap > thresholds["z_neg_1p96"])] = "moderate_monitoring_excess"
    relation[gap <= thresholds["z_neg_1p96"]] = "strong_monitoring_excess"
    relation[(gap >= thresholds["z_pos_1"]) & (gap < thresholds["z_pos_1p96"])] = "moderate_encoder_excess"
    relation[gap >= thresholds["z_pos_1p96"]] = "strong_encoder_excess"

    final = final.copy()
    final[S32_COL] = relation
    return final, thresholds


def add_s33(final: pd.DataFrame, diagnostics: pd.DataFrame) -> pd.DataFrame:
    axis_cols = [f"{axis}_axis_max_p" for axis in S33_CLASS_ORDER]
    axis_scores = diagnostics[axis_cols].to_numpy(dtype=np.float64)
    top_idx = np.nanargmax(axis_scores, axis=1)
    labels = np.asarray(S33_CLASS_ORDER, dtype=object)[top_idx]

    final = final.copy()
    final[S33_COL] = labels
    return final


def quantiles(values: pd.Series) -> dict[str, float]:
    finite = values.dropna().to_numpy(dtype=np.float64)
    qs = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    return {f"p{int(q * 100):02d}": float(np.quantile(finite, q)) for q in qs}


def summarize(final: pd.DataFrame, diagnostics: pd.DataFrame) -> dict[str, object]:
    gap = final[S31_COL]
    embedding = diagnostics["embedding_rarity_p"]
    monitoring = diagnostics["monitoring_rarity_p"]
    s32_class_counts = final[S32_COL].value_counts().reindex(S32_CLASS_ORDER, fill_value=0)
    s32_split_class_counts = (
        final.groupby("split")[S32_COL]
        .value_counts()
        .unstack(fill_value=0)
        .reindex(columns=S32_CLASS_ORDER, fill_value=0)
    )
    s33_class_counts = final[S33_COL].value_counts().reindex(S33_CLASS_ORDER, fill_value=0)
    s33_split_class_counts = (
        final.groupby("split")[S33_COL]
        .value_counts()
        .unstack(fill_value=0)
        .reindex(columns=S33_CLASS_ORDER, fill_value=0)
    )
    summary = {
        "tasks": ["S31", "S32", "S33"],
        "output_columns": [S31_COL, S32_COL, S33_COL],
        "n_tiles": int(len(final)),
        "n_missing": final[[S31_COL, S32_COL, S33_COL]].isna().sum().astype(int).to_dict(),
        "scale": "train percentile p0-p99",
        "s31_formula": "embedding_rarity_p - monitoring_rarity_p",
        "monitoring_axes": AXES,
        "embedding_rarity_p": quantiles(embedding),
        "monitoring_rarity_p": quantiles(monitoring),
        "s31_gap_p": quantiles(gap),
        "s32_thresholds_train_zscore_s31_gap_p": diagnostics.attrs["s32_thresholds"],
        "s32_class_counts": s32_class_counts.astype(int).to_dict(),
        "s32_class_fractions": (s32_class_counts / len(final)).astype(float).to_dict(),
        "s32_class_counts_by_split": {
            str(idx): row.astype(int).to_dict() for idx, row in s32_split_class_counts.iterrows()
        },
        "s33_formula": "argmax(quality_axis_max_p, motion_axis_max_p, spatial_axis_max_p, temporal_axis_max_p)",
        "s33_class_counts": s33_class_counts.astype(int).to_dict(),
        "s33_class_fractions": (s33_class_counts / len(final)).astype(float).to_dict(),
        "s33_class_counts_by_split": {
            str(idx): row.astype(int).to_dict() for idx, row in s33_split_class_counts.iterrows()
        },
        "mean": float(gap.mean()),
        "std": float(gap.std(ddof=1)),
        "skew": float(((gap - gap.mean()) ** 3).mean() / (gap.std(ddof=0) ** 3)),
        "excess_kurtosis": float(((gap - gap.mean()) ** 4).mean() / (gap.std(ddof=0) ** 4) - 3.0),
        "diagnostic_counts_not_labels": {
            "embedding_more_rare_by_gt_20p": int((gap > 20.0).sum()),
            "near_aligned_abs_gap_le_20p": int((gap.abs() <= 20.0).sum()),
            "monitoring_more_rare_by_gt_20p": int((gap < -20.0).sum()),
        },
        "correlation_embedding_vs_monitoring": {
            "pearson": float(diagnostics[["embedding_rarity_p", "monitoring_rarity_p"]].corr(method="pearson").iloc[0, 1]),
            "spearman": float(diagnostics[["embedding_rarity_p", "monitoring_rarity_p"]].corr(method="spearman").iloc[0, 1]),
        },
    }
    return summary


def plot_distribution(final: pd.DataFrame, diagnostics: pd.DataFrame) -> None:
    gap = final[S31_COL].dropna()
    thresholds = diagnostics.attrs["s32_thresholds"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))

    axes[0].hist(gap, bins=80, color="#4c78a8", alpha=0.85)
    axes[0].axvline(0, color="black", linewidth=1)
    for key in ["z_neg_1p96", "z_neg_1", "z_pos_1", "z_pos_1p96"]:
        axes[0].axvline(thresholds[key], color="#e45756", linestyle="--", linewidth=1)
    axes[0].set_title("S31 representation-monitoring rarity gap")
    axes[0].set_xlabel("encoder p - monitoring p")
    axes[0].set_ylabel("tile count")

    axes[1].scatter(
        diagnostics["monitoring_rarity_p"],
        diagnostics["embedding_rarity_p"],
        s=6,
        alpha=0.25,
        color="#4c78a8",
    )
    axes[1].plot([0, 99], [0, 99], color="black", linewidth=1)
    axes[1].set_title("Encoder rarity vs monitoring-system rarity")
    axes[1].set_xlabel("A/B/C/D monitoring rarity p")
    axes[1].set_ylabel("encoder representation rarity p")

    s33_counts = final[S33_COL].value_counts().reindex(S33_CLASS_ORDER, fill_value=0)
    axes[2].bar(s33_counts.index, s33_counts.values, color="#f58518")
    axes[2].set_title("S33 monitoring distinctive dimension")
    axes[2].set_xlabel("dimension")
    axes[2].set_ylabel("tile count")
    axes[2].tick_params(axis="x", rotation=25)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "s3_distribution.png", dpi=180)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    final, diagnostics = compute_s31(merge_sources())
    final.to_csv(OUT_DIR / "s3_final_table.csv", index=False)
    summary = summarize(final, diagnostics)
    (OUT_DIR / "s3_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    plot_distribution(final, diagnostics)


if __name__ == "__main__":
    main()
