"""Summarize the canonical D1-D4 temporal task tables without recomputing targets."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


from egms_qa.paths import OUTPUTS_DIR, TASKS_DIR
from egms_qa.qa_construction.tables import align_family_to_base, read_family
from egms_qa.qa_construction.task_specs import TASK_SPECS

OUT_DIR = OUTPUTS_DIR / "summaries/temporal"


D_COLUMNS = {
    family: ["tile_id", "split"] + [
        spec["target_column"] for spec in TASK_SPECS
        if spec["source_family"] == family
    ]
    for family in ("d1", "d2", "d3", "d4")
}
D_COLUMNS["d4"] += [
    "D41_trend_rank", "D41_seasonal_rank", "D41_acceleration_rank",
    "D41_top_process_candidate", "D41_top_rank", "D41_second_rank",
    "D41_dominance_margin",
]


def merge_temporal_tables(tasks_root: Path) -> pd.DataFrame:
    """Join complete, unique D-family tables in D1 tile order."""
    out = None
    for family, columns in D_COLUMNS.items():
        table = read_family(tasks_root, family)
        missing = set(columns) - set(table.columns)
        if missing:
            raise ValueError(f"{family} missing temporal summary columns: {sorted(missing)}")
        table = table[columns]
        if out is None:
            out = table.copy()
            out[["tile_id", "split"]] = out[["tile_id", "split"]].astype(str)
        else:
            index = pd.MultiIndex.from_frame(out[["tile_id", "split"]])
            aligned = align_family_to_base(table, index, family)
            out = pd.concat([out, aligned.drop(columns=["tile_id", "split"])], axis=1)
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
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

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
    ap.add_argument("--tasks-root", type=Path, default=TASKS_DIR)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--plot", action="store_true", help="Also write a distribution plot (requires tasks extra)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    table_path = out_dir / "d_final_table.csv"
    summary_path = out_dir / "d_final_summary.json"
    plot_path = out_dir / "d_final_distribution.png"

    df = merge_temporal_tables(args.tasks_root)
    if len(df) != 10000:
        raise ValueError(f"expected 10000 rows, got {len(df)}")

    df.to_csv(table_path, index=False)

    scalar_columns = [
        "D12_curvature_strength",
        "D13_changepoint_strength",
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
        "source_tables": {family: str(args.tasks_root / family / f"{family}_final_table.csv") for family in D_COLUMNS},
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
    if args.plot:
        _plot(df, plot_path)

    print(f"wrote {table_path}")
    print(f"wrote {summary_path}")
    if args.plot:
        print(f"wrote {plot_path}")
    print(f"shape: {len(df)} rows x {len(df.columns)} columns")
    print("D42 counts")
    print(df["D42_temporal_evolution_archetype"].value_counts().to_string())


if __name__ == "__main__":
    main()
