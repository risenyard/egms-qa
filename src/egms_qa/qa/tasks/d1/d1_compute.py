"""Canonicalize the deterministic D1 geometry table.

The heavy all-10k geometry fit is performed upstream; this delivery script
promotes its output to the canonical D1 column names used by EGMS-QA task tables.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


ROOT = Path(".")
DEFAULT_SOURCE = ROOT / "outputs/tasks/d1_exp/d1_exp_final_table.csv"
DEFAULT_SOURCE_SUMMARY = ROOT / "outputs/tasks/d1_exp/d1_exp_summary.json"
DEFAULT_OUT = ROOT / "outputs/tasks/d1"

RENAME = {
    "D1_exp_n_valid_epochs": "D1_n_valid_epochs",
    "D12_exp_curvature_strength": "D12_curvature_strength",
    "D12_exp_curvature_gain": "D12_curvature_gain",
    "D12_exp_delta_bic_quadratic": "D12_delta_bic_quadratic",
    "D12_exp_signed_quadratic_coef": "D12_signed_quadratic_coef",
    "D13_exp_changepoint_strength": "D13_changepoint_strength",
    "D13_exp_break_gain": "D13_break_gain",
    "D13_exp_delta_bic_piecewise": "D13_delta_bic_piecewise",
    "D13_exp_slope_delta_effect": "D13_slope_delta_effect",
    "D14_exp_dominant_changepoint_time_year": "D14_candidate_changepoint_time_year",
    "D14_exp_bic_selected_changepoint_time_year": "D14_bic_selected_changepoint_time_year",
    "D1_exp_bic_linear": "D1_bic_linear",
    "D1_exp_bic_quadratic": "D1_bic_quadratic",
    "D1_exp_bic_piecewise": "D1_bic_piecewise",
    "D1_exp_bic_complex": "D1_bic_complex",
    "D1_exp_best_bic_model": "D1_best_bic_model",
    "D1_exp_delta_bic_best_vs_linear": "D1_delta_bic_best_vs_linear",
    "D1_exp_linear_sse": "D1_linear_sse",
    "D1_exp_quadratic_sse": "D1_quadratic_sse",
    "D1_exp_piecewise_sse": "D1_piecewise_sse",
    "D1_exp_complex_sse": "D1_complex_sse",
    "D11_exp_has_break": "D11_has_break",
    "D11_exp_is_curved": "D11_is_curved",
    "D11_exp_trend_shape": "D11_long_term_trend_shape",
    "D11_exp_confident_trend_shape": "D11_confident_trend_shape",
    "D14_exp_strong_break_time_year": "D14_dominant_changepoint_time_year",
    "D14_exp_strong_break_month": "D14_dominant_changepoint_month",
    "D14_exp_strong_break_month_index": "D14_dominant_changepoint_month_index",
    "D14_exp_strong_break_time_bin8": "D14_dominant_changepoint_time_bin8",
}

COLUMNS = [
    "tile_id",
    "split",
    "D1_n_valid_epochs",
    "D11_long_term_trend_shape",
    "D11_confident_trend_shape",
    "D11_has_break",
    "D11_is_curved",
    "D12_curvature_strength",
    "D12_curvature_gain",
    "D12_delta_bic_quadratic",
    "D12_signed_quadratic_coef",
    "D13_changepoint_strength",
    "D13_break_gain",
    "D13_delta_bic_piecewise",
    "D13_slope_delta_effect",
    "D14_dominant_changepoint_time_year",
    "D14_dominant_changepoint_month",
    "D14_dominant_changepoint_month_index",
    "D14_dominant_changepoint_time_bin8",
    "D14_candidate_changepoint_time_year",
    "D14_bic_selected_changepoint_time_year",
    "D1_bic_linear",
    "D1_bic_quadratic",
    "D1_bic_piecewise",
    "D1_bic_complex",
    "D1_best_bic_model",
    "D1_delta_bic_best_vs_linear",
    "D1_linear_sse",
    "D1_quadratic_sse",
    "D1_piecewise_sse",
    "D1_complex_sse",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--source", default=str(DEFAULT_SOURCE))
    p.add_argument("--source-summary", default=str(DEFAULT_SOURCE_SUMMARY))
    p.add_argument("--out-dir", default=str(DEFAULT_OUT))
    return p.parse_args()


def counts(series: pd.Series) -> dict[str, int]:
    return {str(k): int(v) for k, v in series.value_counts(dropna=False).items()}


def main() -> None:
    args = parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.source).rename(columns=RENAME)
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing canonical columns: {missing}")
    df = df[COLUMNS]
    if len(df) != 10000:
        raise ValueError(f"Expected 10000 rows, found {len(df)}")
    if df["tile_id"].duplicated().any():
        raise ValueError("Duplicate tile_id values in D1 table")

    table_path = out / "d1_final_table.csv"
    df.to_csv(table_path, index=False)

    source_summary = json.loads(Path(args.source_summary).read_text(encoding="utf-8"))
    summary = {
        "n_tiles": int(len(df)),
        "source_table": str(Path(args.source).resolve()),
        "thresholds_fit_on_train": source_summary.get("thresholds_fit_on_train", {}),
        "primary_targets": {
            "D11": "D11_long_term_trend_shape",
            "D12": "D12_curvature_strength",
            "D13": "D13_changepoint_strength",
            "D14": "D14_dominant_changepoint_time_year",
        },
        "split_counts": counts(df["split"]),
        "class_counts": {
            "D11_long_term_trend_shape": counts(df["D11_long_term_trend_shape"]),
            "D11_has_break": counts(df["D11_has_break"]),
            "D11_is_curved": counts(df["D11_is_curved"]),
            "D14_dominant_changepoint_time_bin8": counts(df["D14_dominant_changepoint_time_bin8"]),
        },
    }
    (out / "d1_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {table_path}")
    print(f"wrote {out / 'd1_summary.json'}")


if __name__ == "__main__":
    main()
