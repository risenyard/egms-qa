"""Compute the canonical D1 temporal geometry targets from model-ready NPZ tiles."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

from typing import Any

import numpy as np
import pandas as pd

from egms_qa.qa_construction.reference import reference_table
from egms_qa.paths import DATA_DIR, OUTPUTS_DIR, SPLIT_MANIFEST
from egms_qa.qa_construction.inputs import read_tile_manifest
from egms_qa.qa_construction.temporal_inputs import TimeAxis, YEAR_DAYS


MIN_VALID_EPOCHS = 200
EPS = 1e-10
STRONG_QUANTILE = 0.85
CP_BINS = 8
TIME_AXIS: TimeAxis | None = None

COLUMNS = ['tile_id', 'split', 'D1_n_valid_epochs', 'D11_long_term_trend_shape', 'D11_confident_trend_shape', 'D11_has_break', 'D11_is_curved', 'D12_curvature_strength', 'D12_curvature_gain', 'D12_delta_bic_quadratic', 'D12_signed_quadratic_coef', 'D13_changepoint_strength', 'D13_break_gain', 'D13_delta_bic_piecewise', 'D13_slope_delta_effect', 'D14_dominant_changepoint_time_year', 'D14_dominant_changepoint_month', 'D14_dominant_changepoint_month_index', 'D14_dominant_changepoint_time_bin8', 'D14_candidate_changepoint_time_year', 'D14_bic_selected_changepoint_time_year', 'D1_bic_linear', 'D1_bic_quadratic', 'D1_bic_piecewise', 'D1_bic_complex', 'D1_best_bic_model', 'D1_delta_bic_best_vs_linear', 'D1_linear_sse', 'D1_quadratic_sse', 'D1_piecewise_sse', 'D1_complex_sse']


def _season_terms(years: np.ndarray) -> np.ndarray:
    phase = 2.0 * np.pi * years
    return np.column_stack(
        [
            np.sin(phase),
            np.cos(phase),
            np.sin(2.0 * phase),
            np.cos(2.0 * phase),
        ]
    )


def _configure_axis(axis: TimeAxis) -> None:
    global TIME_AXIS, YEARS, X_TIME, SEASON, X_LINEAR, X_QUADRATIC
    global CANDIDATE_INDEX, CANDIDATE_X, CANDIDATE_YEARS
    TIME_AXIS = axis
    YEARS = axis.years
    X_TIME = (YEARS - YEARS.mean()) / ((YEARS.max() - YEARS.min()) / 2.0)
    SEASON = _season_terms(YEARS)
    X_LINEAR = np.column_stack([np.ones_like(X_TIME), X_TIME, SEASON])
    X_QUADRATIC = np.column_stack([X_LINEAR, X_TIME**2])
    CANDIDATE_INDEX = np.unique(np.linspace(36, len(YEARS) - 37, 52).round().astype(int))
    CANDIDATE_X = X_TIME[CANDIDATE_INDEX]
    CANDIDATE_YEARS = YEARS[CANDIDATE_INDEX]


def _fit_robust(y: np.ndarray, design: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    yv = y[valid].astype(np.float64, copy=False)
    xv = design[valid].astype(np.float64, copy=False)
    if len(yv) <= xv.shape[1]:
        return np.full(xv.shape[1], np.nan), np.nan, np.full(len(yv), np.nan)
    weights = np.ones(len(yv), dtype=np.float64)
    beta = np.full(xv.shape[1], np.nan, dtype=np.float64)
    for _ in range(4):
        sw = np.sqrt(weights)
        beta, *_ = np.linalg.lstsq(xv * sw[:, None], yv * sw, rcond=None)
        resid = yv - xv @ beta
        center = np.nanmedian(resid)
        scale = 1.4826 * np.nanmedian(np.abs(resid - center)) + EPS
        weights = np.minimum(1.0, (1.345 * scale) / (np.abs(resid) + EPS))
    resid = yv - xv @ beta
    sse = float(np.sum(resid**2))
    return beta, sse, resid


def _bic(sse: float, n: int, k: int) -> float:
    if n <= k or not np.isfinite(sse) or sse <= EPS:
        return np.nan
    return float(n * np.log(sse / n) + k * np.log(n))


def _read_one(row: tuple[str, str, str]) -> dict[str, object]:
    tile_id, split, path = row
    if TIME_AXIS is None:
        raise RuntimeError("D1 time axis is not configured")
    y = TIME_AXIS.tile_median(path)
    valid = np.isfinite(y)
    n_valid = int(valid.sum())
    base = {
        "tile_id": str(tile_id),
        "split": str(split),
        "D1_n_valid_epochs": n_valid,
    }
    if n_valid < MIN_VALID_EPOCHS:
        return {
            **base,
            "D12_curvature_strength": np.nan,
            "D12_curvature_gain": np.nan,
            "D12_delta_bic_quadratic": np.nan,
            "D12_signed_quadratic_coef": np.nan,
            "D13_changepoint_strength": np.nan,
            "D13_break_gain": np.nan,
            "D13_delta_bic_piecewise": np.nan,
            "D13_slope_delta_effect": np.nan,
            "D14_candidate_changepoint_time_year": np.nan,
            "D14_bic_selected_changepoint_time_year": np.nan,
            "D1_bic_linear": np.nan,
            "D1_bic_quadratic": np.nan,
            "D1_bic_piecewise": np.nan,
            "D1_bic_complex": np.nan,
            "D1_best_bic_model": pd.NA,
            "D1_delta_bic_best_vs_linear": np.nan,
            "D1_linear_sse": np.nan,
            "D1_quadratic_sse": np.nan,
            "D1_piecewise_sse": np.nan,
            "D1_complex_sse": np.nan,
        }

    lin_beta, lin_sse, lin_resid = _fit_robust(y, X_LINEAR, valid)
    quad_beta, quad_sse, _ = _fit_robust(y, X_QUADRATIC, valid)
    resid_std = float(np.sqrt(lin_sse / max(n_valid - X_LINEAR.shape[1], 1))) if np.isfinite(lin_sse) else np.nan
    bic_linear = _bic(lin_sse, n_valid, X_LINEAR.shape[1])
    bic_quadratic = _bic(quad_sse, n_valid, X_QUADRATIC.shape[1])
    delta_bic_quadratic = (
        float(bic_linear - bic_quadratic)
        if np.isfinite(bic_linear) and np.isfinite(bic_quadratic)
        else np.nan
    )

    curvature_gain = 0.0
    curvature_effect = 0.0
    signed_quad = np.nan
    if np.isfinite(lin_sse) and lin_sse > EPS and np.isfinite(quad_sse):
        curvature_gain = float(np.clip(1.0 - quad_sse / lin_sse, 0.0, 1.0))
        signed_quad = float(quad_beta[-1])
        x2_scale = float(np.nanstd((X_TIME[valid] ** 2)))
        if np.isfinite(resid_std) and resid_std > EPS:
            curvature_effect = float(abs(signed_quad) * x2_scale / resid_std)
    curvature_strength = float(curvature_gain * np.log1p(max(curvature_effect, 0.0)))

    best_sse = np.inf
    best_year = np.nan
    best_hinge_coef = np.nan
    best_hinge_scale = np.nan
    best_complex_sse = np.inf
    best_complex_year = np.nan
    for tau, year in zip(CANDIDATE_X, CANDIDATE_YEARS):
        hinge = np.maximum(0.0, X_TIME - tau)
        design = np.column_stack([X_LINEAR, hinge])
        beta, sse, _ = _fit_robust(y, design, valid)
        if np.isfinite(sse) and sse < best_sse:
            best_sse = float(sse)
            best_year = float(year)
            best_hinge_coef = float(beta[-1])
            best_hinge_scale = float(np.nanstd(hinge[valid]))
        complex_design = np.column_stack([X_QUADRATIC, hinge])
        _, complex_sse, _ = _fit_robust(y, complex_design, valid)
        if np.isfinite(complex_sse) and complex_sse < best_complex_sse:
            best_complex_sse = float(complex_sse)
            best_complex_year = float(year)
    break_gain = 0.0
    slope_effect = 0.0
    if np.isfinite(lin_sse) and lin_sse > EPS and np.isfinite(best_sse):
        break_gain = float(np.clip(1.0 - best_sse / lin_sse, 0.0, 1.0))
        if np.isfinite(resid_std) and resid_std > EPS and np.isfinite(best_hinge_coef):
            slope_effect = float(abs(best_hinge_coef) * best_hinge_scale / resid_std)
    cp_strength = float(break_gain * np.log1p(max(slope_effect, 0.0)))
    # Penalize the searched breakpoint as one additional effective parameter.
    bic_piecewise = _bic(best_sse, n_valid, X_LINEAR.shape[1] + 2)
    bic_complex = _bic(best_complex_sse, n_valid, X_LINEAR.shape[1] + 3)
    delta_bic_piecewise = (
        float(bic_linear - bic_piecewise)
        if np.isfinite(bic_linear) and np.isfinite(bic_piecewise)
        else np.nan
    )
    bic_values = {
        "linear": bic_linear,
        "quadratic": bic_quadratic,
        "piecewise": bic_piecewise,
        "complex": bic_complex,
    }
    finite_bic = {k: v for k, v in bic_values.items() if np.isfinite(v)}
    best_bic_model = min(finite_bic, key=finite_bic.get) if finite_bic else pd.NA
    best_bic = finite_bic.get(str(best_bic_model), np.nan)
    delta_bic_best = (
        float(bic_linear - best_bic)
        if np.isfinite(bic_linear) and np.isfinite(best_bic) and str(best_bic_model) != "linear"
        else 0.0
    )
    selected_cp_time = best_complex_year if str(best_bic_model) == "complex" else best_year

    return {
        **base,
        "D12_curvature_strength": curvature_strength,
        "D12_curvature_gain": curvature_gain,
        "D12_delta_bic_quadratic": delta_bic_quadratic,
        "D12_signed_quadratic_coef": signed_quad,
        "D13_changepoint_strength": cp_strength,
        "D13_break_gain": break_gain,
        "D13_delta_bic_piecewise": delta_bic_piecewise,
        "D13_slope_delta_effect": slope_effect,
        "D14_candidate_changepoint_time_year": best_year,
        "D14_bic_selected_changepoint_time_year": selected_cp_time,
        "D1_bic_linear": bic_linear,
        "D1_bic_quadratic": bic_quadratic,
        "D1_bic_piecewise": bic_piecewise,
        "D1_bic_complex": bic_complex,
        "D1_best_bic_model": best_bic_model,
        "D1_delta_bic_best_vs_linear": delta_bic_best,
        "D1_linear_sse": lin_sse,
        "D1_quadratic_sse": quad_sse,
        "D1_piecewise_sse": best_sse,
        "D1_complex_sse": best_complex_sse,
    }


def _trend_shape(has_break: bool, is_curved: bool) -> str:
    if has_break and is_curved:
        return "complex_trend"
    if has_break:
        return "stage_change"
    if is_curved:
        return "curved_trend"
    return "linear_trend"


def _decimal_year_to_month(value: float) -> tuple[str, float]:
    year = int(np.floor(value))
    offset_days = float(value - year) * YEAR_DAYS
    timestamp = pd.Timestamp(year=year, month=1, day=1) + pd.to_timedelta(offset_days, unit="D")
    month_label = f"{timestamp.year:04d}-{timestamp.month:02d}"
    month_index = float(timestamp.year * 12 + timestamp.month - 1)
    return month_label, month_index


def _month_labels(values: pd.Series, strong: pd.Series) -> tuple[pd.Series, pd.Series]:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    strong_arr = strong.to_numpy(dtype=bool)
    labels: list[Any] = []
    indices: list[float] = []
    for value, ok in zip(numeric, strong_arr):
        if ok and np.isfinite(value):
            label, idx = _decimal_year_to_month(float(value))
            labels.append(label)
            indices.append(idx)
        else:
            labels.append(pd.NA)
            indices.append(np.nan)
    return pd.Series(labels, index=values.index, dtype="string"), pd.Series(indices, index=values.index, dtype="float64")


def _bin_labels(values: pd.Series, strong: pd.Series, train: pd.Series, n_bins: int, reference_fit=None) -> tuple[pd.Series, list[float]]:
    fit = values[strong & train].dropna().astype(float).to_numpy() if reference_fit is None else reference_fit
    if len(fit) < n_bins * 5:
        out = pd.Series(pd.NA, index=values.index, dtype="string")
        return out, []
    edges = np.quantile(fit, np.linspace(0.0, 1.0, n_bins + 1))
    if len(np.unique(edges)) < n_bins + 1:
        edges = np.linspace(float(np.nanmin(fit)), float(np.nanmax(fit)), n_bins + 1)
    edges[0] = -np.inf
    edges[-1] = np.inf
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    idx = np.digitize(numeric, edges[1:-1], right=False)
    labels = np.array([f"cp_bin_{i:02d}" for i in range(n_bins)], dtype=object)
    out = pd.Series(pd.NA, index=values.index, dtype="string")
    valid = strong.to_numpy() & np.isfinite(numeric)
    out.loc[valid] = labels[idx[valid]]
    serial_edges = [float(x) for x in edges]
    serial_edges[0] = float(np.nanmin(fit))
    serial_edges[-1] = float(np.nanmax(fit))
    return out, serial_edges


def classify_tiles(df: pd.DataFrame, reference: pd.DataFrame | None = None) -> tuple[pd.DataFrame, dict]:
    df = df.copy()
    train = df["split"].astype(str).eq("train")
    valid = df["D1_n_valid_epochs"].ge(MIN_VALID_EPOCHS)
    d12 = pd.to_numeric(df["D12_curvature_strength"], errors="coerce")
    d13 = pd.to_numeric(df["D13_changepoint_strength"], errors="coerce")
    fit = df if reference is None else reference
    fit_d12 = pd.to_numeric(fit["D12_curvature_strength"], errors="coerce")
    fit_d13 = pd.to_numeric(fit["D13_changepoint_strength"], errors="coerce")
    fit_mask = fit["split"].eq("train") & fit["D1_n_valid_epochs"].ge(MIN_VALID_EPOCHS) & fit_d12.notna() & fit_d13.notna()
    if int(fit_mask.sum()) < 100:
        raise RuntimeError("Not enough train rows to fit D1 reference thresholds.")
    d12_threshold = float(fit_d12[fit_mask].quantile(STRONG_QUANTILE))
    d13_threshold = float(fit_d13[fit_mask].quantile(STRONG_QUANTILE))
    thresholds = {
        "threshold_mode": "train_p85_primitives",
        "d12_strong_quantile": STRONG_QUANTILE,
        "d12_strong_threshold": d12_threshold,
        "d13_strong_quantile": STRONG_QUANTILE,
        "d13_strong_threshold": d13_threshold,
        "d14_time_source": "D14_candidate_changepoint_time_year",
        "cp_bins": CP_BINS,
    }
    has_break = valid & d13.ge(d13_threshold)
    is_curved = valid & d12.ge(d12_threshold)

    df["D11_has_break"] = np.where(has_break, "has_break", "no_strong_break")
    df["D11_is_curved"] = np.where(is_curved, "curved", "not_curved")
    df["D11_long_term_trend_shape"] = [
        _trend_shape(bool(b), bool(c)) if bool(v) else pd.NA
        for b, c, v in zip(has_break, is_curved, valid)
    ]
    df["D11_confident_trend_shape"] = pd.Series(df["D11_long_term_trend_shape"], dtype="string")
    df["D14_dominant_changepoint_time_year"] = pd.to_numeric(df["D14_candidate_changepoint_time_year"], errors="coerce").where(has_break)
    month_label, month_index = _month_labels(
        pd.to_numeric(df["D14_candidate_changepoint_time_year"], errors="coerce"),
        has_break,
    )
    df["D14_dominant_changepoint_month"] = month_label
    df["D14_dominant_changepoint_month_index"] = month_index
    bins, bin_edges = _bin_labels(
        pd.to_numeric(df["D14_candidate_changepoint_time_year"], errors="coerce"),
        has_break,
        train,
        CP_BINS,
        reference_fit=None if reference is None else pd.to_numeric(
            fit.loc[fit["split"].eq("train") & fit["D1_n_valid_epochs"].ge(MIN_VALID_EPOCHS)
                    & fit_d13.ge(d13_threshold), "D14_candidate_changepoint_time_year"], errors="coerce"
        ).dropna().to_numpy(dtype=float),
    )
    df["D14_dominant_changepoint_time_bin8"] = bins

    summary = {
        "n_tiles": len(df),
        "thresholds_fit_on_train": thresholds,
        "cp_bin_edges_fit_on_train_strong_break": bin_edges,
        "split_counts": {str(k): int(v) for k, v in df["split"].value_counts().items()},
        "class_counts": {col: {str(k): int(v) for k, v in df[col].value_counts(dropna=False).items()}
                         for col in ["D11_long_term_trend_shape", "D11_has_break", "D11_is_curved"]},
    }
    return df[COLUMNS], summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=SPLIT_MANIFEST)
    parser.add_argument("--data-config", type=Path, default=DATA_DIR / "encoder/manifest/data_config.json")
    parser.add_argument("--source-tiles-root", type=Path)
    parser.add_argument("--out-dir", type=Path, default=OUTPUTS_DIR / "tasks-rebuilt/d1")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--reference-state", type=Path)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    axis = TimeAxis.from_file(args.data_config)
    frame = read_tile_manifest(args.manifest, args.source_tiles_root)
    rows = list(frame[["tile_id", "split", "path"]].itertuples(index=False, name=None))
    if args.workers == 1:
        _configure_axis(axis)
        records = list(map(_read_one, rows))
    else:
        with ProcessPoolExecutor(max_workers=args.workers, initializer=_configure_axis, initargs=(axis,)) as executor:
            records = list(executor.map(_read_one, rows, chunksize=8))
    table, summary = classify_tiles(pd.DataFrame.from_records(records), reference_table(args.reference_state, "d1"))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out_dir / "d1_final_table.csv", index=False)
    summary["time_axis"] = {"stored_window": [axis.t_start, axis.t_end], "start_year": axis.start_year, "cadence_days": axis.cadence_days}
    (args.out_dir / "d1_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(table)} D1 rows to {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
