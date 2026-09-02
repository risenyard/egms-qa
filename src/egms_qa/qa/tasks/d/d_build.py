"""Build the EGMS-QA D-group delivery table.

The D group is delivered as a group-level convenience table assembled from the
already delivered D1-D4 family tables. It does not recompute the underlying
tasks.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(".")
OUT_DIR = ROOT / "outputs/tasks/d"
SOURCE_TABLES = {
    "d1": ROOT / "outputs/tasks/d1/d1_final_table.csv",
    "d2": ROOT / "outputs/tasks/d2/d2_final_table.csv",
    "d3": ROOT / "outputs/tasks/d3/d3_final_table.csv",
    "d4": ROOT / "outputs/tasks/d4/d4_final_table.csv",
}

D1_COLUMNS = [
    "tile_id",
    "split",
    "D11_long_term_trend_shape",
    "D12_trend_order_mean",
    "D13_top_changepoint_probability",
    "D14_dominant_changepoint_time_year",
]
D2_COLUMNS = [
    "tile_id",
    "split",
    "D21_dominant_seasonal_peak",
    "D22_phase_coherence",
    "D23_phase_dispersion_days",
    "D24_seasonal_amplitude_change_mm",
]
D3_COLUMNS = [
    "tile_id",
    "split",
    "D31_motion_intensification_mm_yr2",
    "D32_acceleration_support_fraction",
    "D33_intensification_spread_mm_yr2",
    "D34_intensification_hotspot_strength_mm_yr2",
    "D35_intensification_hotspot_location",
]
D4_COLUMNS = [
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
]


def _read_selected(path: Path, columns: list[str]) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=columns)
    if df["tile_id"].duplicated().any():
        raise ValueError(f"duplicate tile_id in {path}")
    return df


def _merge_sources(source_tables: dict[str, Path]) -> pd.DataFrame:
    d1 = _read_selected(source_tables["d1"], D1_COLUMNS)
    d2 = _read_selected(source_tables["d2"], D2_COLUMNS)
    d3 = _read_selected(source_tables["d3"], D3_COLUMNS)
    d4 = _read_selected(source_tables["d4"], D4_COLUMNS)

    out = d1.merge(d2, on=["tile_id", "split"], validate="one_to_one")
    out = out.merge(d3, on=["tile_id", "split"], validate="one_to_one")
    out = out.merge(d4, on=["tile_id", "split"], validate="one_to_one")
    return out


def _counts(series: pd.Series) -> dict[str, int]:
    return series.value_counts(dropna=False).astype(int).to_dict()


def _scalar_summary(df: pd.DataFrame, columns: list[str]) -> dict[str, dict[str, float | int]]:
    summary: dict[str, dict[str, float | int]] = {}
    for col in columns:
        vals = pd.to_numeric(df[col], errors="coerce")
        summary[col] = {
            "defined": int(vals.notna().sum()),
            "mean": float(vals.mean(skipna=True)),
            "p25": float(vals.quantile(0.25)),
            "p50": float(vals.quantile(0.50)),
            "p75": float(vals.quantile(0.75)),
            "p95": float(vals.quantile(0.95)),
        }
    return summary


def _plot(df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), constrained_layout=True)
    plots = [
        ("D11_long_term_trend_shape", "D11 trend shape"),
        ("D21_dominant_seasonal_peak", "D21 seasonal peak"),
        ("D41_temporal_dominant_process", "D41 dominant process"),
        ("D42_temporal_evolution_archetype", "D42 evolution archetype"),
    ]
    for ax, (col, title) in zip(axes.flat, plots):
        counts = df[col].value_counts()
        ax.barh(counts.index, counts.values)
        ax.set_title(title)
        ax.set_xlabel("tiles")
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    table_path = out_dir / "d_final_table.csv"
    summary_path = out_dir / "d_final_summary.json"
    plot_path = out_dir / "d_final_distribution.png"

    df = _merge_sources(SOURCE_TABLES)
    if len(df) != 10000:
        raise ValueError(f"expected 10000 rows, got {len(df)}")

    df.to_csv(table_path, index=False)

    scalar_columns = [
        "D12_trend_order_mean",
        "D13_top_changepoint_probability",
        "D14_dominant_changepoint_time_year",
        "D22_phase_coherence",
        "D23_phase_dispersion_days",
        "D24_seasonal_amplitude_change_mm",
        "D31_motion_intensification_mm_yr2",
        "D32_acceleration_support_fraction",
        "D33_intensification_spread_mm_yr2",
        "D34_intensification_hotspot_strength_mm_yr2",
        "D41_top_rank",
        "D41_dominance_margin",
    ]
    summary = {
        "n_tiles": int(len(df)),
        "n_columns": int(len(df.columns)),
        "delivered_tasks": [
            "D11",
            "D12",
            "D13",
            "D14",
            "D21",
            "D22",
            "D23",
            "D24",
            "D31",
            "D32",
            "D33",
            "D34",
            "D35",
            "D41",
            "D42",
        ],
        "source_tables": {k: str(v) for k, v in SOURCE_TABLES.items()},
        "class_counts": {
            "D11_long_term_trend_shape": _counts(df["D11_long_term_trend_shape"]),
            "D21_dominant_seasonal_peak": _counts(df["D21_dominant_seasonal_peak"]),
            "D35_intensification_hotspot_location": _counts(df["D35_intensification_hotspot_location"]),
            "D41_temporal_dominant_process": _counts(df["D41_temporal_dominant_process"]),
            "D42_temporal_evolution_archetype": _counts(df["D42_temporal_evolution_archetype"]),
        },
        "scalar_summary": _scalar_summary(df, scalar_columns),
    }
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)
    _plot(df, plot_path)

    print(f"wrote {table_path}")
    print(f"wrote {summary_path}")
    print(f"wrote {plot_path}")
    print(f"shape: {len(df)} rows x {len(df.columns)} columns")
    print("D42 counts")
    print(df["D42_temporal_evolution_archetype"].value_counts().to_string())


if __name__ == "__main__":
    main()
