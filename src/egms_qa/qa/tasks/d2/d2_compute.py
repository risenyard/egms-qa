"""Compute EGMS-QA D2 seasonal-phase family targets.

Current targets:
  - D21_dominant_seasonal_peak
  - D22_phase_coherence
  - D23_phase_dispersion_days
  - D24_seasonal_amplitude_change_mm

D21-D24 use the public, standard annual-harmonic idea rather than a custom
time-series classifier. For every point in a tile, linearly detrend the
displacement series, project the residual onto annual sine/cosine components,
and summarize phase alignment or amplitude change.
"""
from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(".")
MANIFEST = ROOT / "data/encoder/manifest/split.parquet"
B51_TABLE = ROOT / "outputs/tasks/b5/b5_final_table.csv"
OUT_DIR = ROOT / "outputs/tasks/d2"

T_START = 8
T_END = 302
CADENCE_DAYS = 6.0
YEAR_DAYS = 365.25
START_YEAR = 2019.0 + T_START * CADENCE_DAYS / YEAR_DAYS
DELTA_T = CADENCE_DAYS / YEAR_DAYS
MIN_VALID_EPOCHS = 50
D24_MIN_VALID_WINDOW_EPOCHS = 50
MIN_VALID_POINTS = 30
EPS = 1e-9
D21_B51_MIN = 1.0
D21_COHERENCE_MIN = 0.20

TIME_YEARS = START_YEAR + np.arange(T_END - T_START, dtype=np.float64) * DELTA_T
ANNUAL_COS = np.cos(2.0 * np.pi * TIME_YEARS)
ANNUAL_SIN = np.sin(2.0 * np.pi * TIME_YEARS)
D24_MID = (T_END - T_START) // 2


def _annual_phasors(ts: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y = ts.astype(np.float64, copy=False)
    valid = np.isfinite(y)
    count = valid.sum(axis=1).astype(np.float64)
    keep = count >= MIN_VALID_EPOCHS
    if not keep.any():
        empty = np.full(y.shape[0], np.nan, dtype=np.float64)
        return empty, empty, empty

    yy = np.where(valid, y, 0.0)
    x = TIME_YEARS[None, :]
    x_sum = (valid * x).sum(axis=1)
    y_sum = yy.sum(axis=1)
    x_mean = np.divide(x_sum, count, out=np.full_like(count, np.nan), where=count > 0)
    y_mean = np.divide(y_sum, count, out=np.full_like(count, np.nan), where=count > 0)

    xc = x - x_mean[:, None]
    yc = np.where(valid, y - y_mean[:, None], 0.0)
    denom = (valid * xc * xc).sum(axis=1)
    numer = (valid * xc * yc).sum(axis=1)
    slope = np.divide(numer, denom, out=np.full_like(denom, np.nan), where=denom > 0)
    fit = y_mean[:, None] + slope[:, None] * xc
    resid = np.where(valid, y - fit, np.nan)

    a = np.nansum(resid * ANNUAL_COS[None, :], axis=1) / np.maximum(count, 1.0)
    b = np.nansum(resid * ANNUAL_SIN[None, :], axis=1) / np.maximum(count, 1.0)
    amp = np.sqrt(a * a + b * b)
    a[~keep] = np.nan
    b[~keep] = np.nan
    amp[~keep] = np.nan
    return a, b, amp


def _phase_coherence(ts: np.ndarray) -> tuple[float, int, float]:
    a, b, amp = _annual_phasors(ts)
    keep = np.isfinite(a) & np.isfinite(b) & np.isfinite(amp)
    n_valid = int(keep.sum())
    if n_valid < MIN_VALID_POINTS:
        return np.nan, n_valid, np.nan
    z = a[keep] + 1j * b[keep]
    amp_sum = float(np.sum(np.abs(z)))
    if amp_sum <= EPS:
        return np.nan, n_valid, float(np.nanmedian(amp[keep]))
    coherence = float(np.abs(np.sum(z)) / (amp_sum + EPS))
    return coherence, n_valid, float(np.nanmedian(amp[keep]))


def _phase_dispersion_days(ts: np.ndarray) -> float:
    a, b, amp = _annual_phasors(ts)
    keep = np.isfinite(a) & np.isfinite(b) & np.isfinite(amp)
    if int(keep.sum()) < MIN_VALID_POINTS:
        return np.nan
    amp_keep = amp[keep]
    amp_min = float(np.nanpercentile(amp_keep, 10.0))
    keep2 = keep.copy()
    keep2[keep] = amp_keep >= amp_min
    if int(keep2.sum()) < 10:
        keep2 = keep
    z = a[keep2] + 1j * b[keep2]
    z_abs = np.abs(z)
    if float(np.nansum(z_abs)) <= EPS:
        return np.nan
    unit = z / (z_abs + EPS)
    weights = amp[keep2]
    rbar = float(np.abs(np.sum(weights * unit)) / (np.sum(weights) + EPS))
    sigma = np.sqrt(-2.0 * np.log(np.clip(rbar, EPS, 1.0)))
    return float(sigma / (2.0 * np.pi) * YEAR_DAYS)


def _window_annual_amplitude(ts: np.ndarray, time_years: np.ndarray) -> np.ndarray:
    y = ts.astype(np.float64, copy=False)
    valid = np.isfinite(y)
    count = valid.sum(axis=1).astype(np.float64)
    keep = count >= D24_MIN_VALID_WINDOW_EPOCHS
    if not keep.any():
        return np.full(y.shape[0], np.nan, dtype=np.float64)

    yy = np.where(valid, y, 0.0)
    x = time_years[None, :]
    x_sum = (valid * x).sum(axis=1)
    y_sum = yy.sum(axis=1)
    x_mean = np.divide(x_sum, count, out=np.full_like(count, np.nan), where=count > 0)
    y_mean = np.divide(y_sum, count, out=np.full_like(count, np.nan), where=count > 0)

    xc = x - x_mean[:, None]
    yc = np.where(valid, y - y_mean[:, None], 0.0)
    denom = (valid * xc * xc).sum(axis=1)
    numer = (valid * xc * yc).sum(axis=1)
    slope = np.divide(numer, denom, out=np.full_like(denom, np.nan), where=denom > EPS)
    resid = np.where(valid, y - (y_mean[:, None] + slope[:, None] * xc), 0.0)

    c = np.cos(2.0 * np.pi * time_years)[None, :]
    s = np.sin(2.0 * np.pi * time_years)[None, :]
    scc = (valid * c * c).sum(axis=1)
    sss = (valid * s * s).sum(axis=1)
    scs = (valid * c * s).sum(axis=1)
    ycc = (resid * c).sum(axis=1)
    yss = (resid * s).sum(axis=1)
    det = scc * sss - scs * scs
    beta_c = np.divide(ycc * sss - yss * scs, det, out=np.full_like(count, np.nan), where=det > EPS)
    beta_s = np.divide(yss * scc - ycc * scs, det, out=np.full_like(count, np.nan), where=det > EPS)

    amp = np.sqrt(beta_c * beta_c + beta_s * beta_s)
    amp[~keep] = np.nan
    return amp


def _seasonal_amplitude_change(ts: np.ndarray) -> tuple[float, int, float, float, float, float]:
    early = _window_annual_amplitude(ts[:, :D24_MID], TIME_YEARS[:D24_MID])
    late = _window_annual_amplitude(ts[:, D24_MID:], TIME_YEARS[D24_MID:])
    delta = late - early
    keep = np.isfinite(delta)
    n_valid = int(keep.sum())
    if n_valid < MIN_VALID_POINTS:
        return np.nan, n_valid, np.nan, np.nan, np.nan, np.nan
    delta_keep = delta[keep]
    return (
        float(np.nanmedian(delta_keep)),
        n_valid,
        float(np.nanmedian(early[keep])),
        float(np.nanmedian(late[keep])),
        float(np.nanpercentile(delta_keep, 10.0)),
        float(np.nanpercentile(delta_keep, 90.0)),
    )


def _season_from_phase(phase: float) -> str:
    if not np.isfinite(phase):
        return "nan"
    day = float(np.mod(phase, 1.0) * YEAR_DAYS)
    if day < 59.0 or day >= 334.0:
        return "winter_peak"
    if day < 151.0:
        return "spring_peak"
    if day < 243.0:
        return "summer_peak"
    return "autumn_peak"


def _dominant_peak_phase(ts: np.ndarray) -> tuple[float, float, str]:
    a, b, amp = _annual_phasors(ts)
    keep = np.isfinite(a) & np.isfinite(b) & np.isfinite(amp)
    if int(keep.sum()) < MIN_VALID_POINTS:
        return np.nan, np.nan, "nan"
    z = a[keep] + 1j * b[keep]
    if float(np.nansum(np.abs(z))) <= EPS:
        return np.nan, np.nan, "nan"
    phase = float(np.mod(np.angle(np.sum(z)) / (2.0 * np.pi), 1.0))
    return phase, phase * YEAR_DAYS, _season_from_phase(phase)


def _read_one(row: tuple[str, str, str]) -> dict[str, object]:
    tile_id, split, path = row
    with np.load(path) as z:
        ts = z["time_series"][:, T_START:T_END]
    coherence, valid_points, median_amp = _phase_coherence(ts)
    dispersion = _phase_dispersion_days(ts)
    peak_phase, peak_day, peak_season = _dominant_peak_phase(ts)
    d24_delta, d24_valid, d24_early, d24_late, d24_p10, d24_p90 = _seasonal_amplitude_change(ts)
    return {
        "tile_id": str(tile_id),
        "split": str(split),
        "D21_raw_peak_phase_fraction": peak_phase,
        "D21_raw_peak_day_of_year": peak_day,
        "D21_raw_peak_season": peak_season,
        "D22_phase_coherence": coherence,
        "D23_phase_dispersion_days": dispersion,
        "D24_seasonal_amplitude_change_mm": d24_delta,
        "D22_valid_phase_point_count": valid_points,
        "D22_median_annual_amplitude_mm": median_amp,
        "D24_valid_amplitude_point_count": d24_valid,
        "D24_early_annual_amplitude_median_mm": d24_early,
        "D24_late_annual_amplitude_median_mm": d24_late,
        "D24_point_delta_p10_mm": d24_p10,
        "D24_point_delta_p90_mm": d24_p90,
    }


def _plot_distribution(df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    axes = axes.ravel()
    axes[0].hist(df["D22_phase_coherence"].dropna(), bins=60, color="#4c78a8")
    axes[0].set_xlabel("D22 phase coherence")
    axes[0].set_ylabel("tiles")
    axes[0].set_xlim(0, 1)

    axes[1].hist(df["D23_phase_dispersion_days"].dropna(), bins=60, color="#f58518")
    axes[1].set_xlabel("D23 phase dispersion (days)")
    axes[1].set_ylabel("tiles")

    axes[2].scatter(
        df["D22_phase_coherence"],
        df["D23_phase_dispersion_days"],
        s=7,
        alpha=0.35,
        color="#4c78a8",
    )
    axes[2].set_xlabel("D22 phase coherence")
    axes[2].set_ylabel("D23 dispersion (days)")

    axes[3].hist(df["D24_seasonal_amplitude_change_mm"].dropna(), bins=80, color="#54a24b")
    axes[3].axvline(0.0, color="#222222", linewidth=1)
    axes[3].set_xlabel("D24 seasonal amplitude change (mm)")
    axes[3].set_ylabel("tiles")
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _apply_d21_gate(row: object) -> tuple[str, str]:
    weak = (not np.isfinite(row.B51_seasonality_p90)) or row.B51_seasonality_p90 < D21_B51_MIN
    coherence_min = D21_COHERENCE_MIN
    incoherent = (not np.isfinite(row.D22_phase_coherence)) or row.D22_phase_coherence < coherence_min
    invalid_phase = str(row.D21_raw_peak_season) == "nan"
    if invalid_phase:
        return "no_clear_seasonal_peak", "invalid_phase"
    if weak and incoherent:
        return "no_clear_seasonal_peak", "weak_seasonality_and_low_coherence"
    if weak:
        return "no_clear_seasonal_peak", "weak_seasonality"
    if incoherent:
        return "no_clear_seasonal_peak", "low_phase_coherence"
    return str(row.D21_raw_peak_season), "clear"


def _add_d21_labels(df_full: pd.DataFrame) -> pd.DataFrame:
    if not B51_TABLE.exists():
        raise FileNotFoundError(f"Missing B51 table: {B51_TABLE}")
    b51 = pd.read_csv(B51_TABLE)
    out = df_full.merge(b51, on=["tile_id", "split"], how="left")
    labels = [_apply_d21_gate(row) for row in out.itertuples(index=False)]
    out["D21_dominant_seasonal_peak"] = [x[0] for x in labels]
    out["D21_gate_reason"] = [x[1] for x in labels]
    keep = [
        "tile_id",
        "split",
        "B51_seasonality_p90",
        "D21_raw_peak_phase_fraction",
        "D21_raw_peak_day_of_year",
        "D21_raw_peak_season",
        "D21_dominant_seasonal_peak",
        "D21_gate_reason",
        "D22_phase_coherence",
        "D23_phase_dispersion_days",
        "D24_seasonal_amplitude_change_mm",
    ]
    return out[keep].copy()


def _plot_d21_distribution(df: pd.DataFrame, out_path: Path) -> None:
    order = ["no_clear_seasonal_peak", "winter_peak", "spring_peak", "summer_peak", "autumn_peak"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    axes = axes.ravel()

    counts = df["D21_dominant_seasonal_peak"].value_counts().reindex(order).fillna(0)
    axes[0].bar(counts.index, counts.values, color="#4c78a8")
    axes[0].set_xlabel("D21 final class")
    axes[0].set_ylabel("tiles")
    axes[0].tick_params(axis="x", rotation=30)

    reasons = df["D21_gate_reason"].value_counts().sort_values()
    axes[1].barh(reasons.index, reasons.values, color="#f58518")
    axes[1].set_xlabel("tiles")
    axes[1].set_ylabel("D21 gate reason")

    axes[2].hist(df["D21_raw_peak_day_of_year"].dropna(), bins=60, color="#54a24b")
    axes[2].set_xlabel("raw peak day of year")
    axes[2].set_ylabel("tiles")

    axes[3].hist(df["D22_phase_coherence"].dropna(), bins=60, color="#b279a2")
    axes[3].axvline(D21_COHERENCE_MIN, color="#e45756", linewidth=2, label="D21 gate")
    axes[3].set_xlabel("D22 phase coherence")
    axes[3].set_ylabel("tiles")
    axes[3].set_xlim(0, 1)
    axes[3].legend(frameon=False)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(MANIFEST))
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))
    ap.add_argument("--chunksize", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--reuse-diagnostics", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "sample" if args.limit else "final"
    table_path = out_dir / f"d2_{suffix}_table.csv"
    detail_path = out_dir / f"d2_{suffix}_diagnostics.csv"
    summary_path = out_dir / f"d2_{suffix}_summary.json"
    plot_path = out_dir / f"d2_{suffix}_distribution.png"
    d21_plot_path = out_dir / f"d2_{suffix}_d21_distribution.png"

    if args.reuse_diagnostics:
        if not detail_path.exists():
            raise FileNotFoundError(f"Missing diagnostics for reuse: {detail_path}")
        df_full = pd.read_csv(detail_path)
    else:
        manifest = pd.read_parquet(args.manifest)
        if args.limit:
            manifest = manifest.head(args.limit).copy()
        rows = [(str(r.tile_id), str(r.split), str(r.path)) for r in manifest.itertuples(index=False)]

        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            records = list(ex.map(_read_one, rows, chunksize=args.chunksize))
        df_full = pd.DataFrame.from_records(records)

    df = _add_d21_labels(df_full)

    df.to_csv(table_path, index=False)
    df_full.to_csv(detail_path, index=False)
    summary = {
        "n_tiles": int(len(df)),
        "tasks": [
            "D21_dominant_seasonal_peak",
            "D22_phase_coherence",
            "D23_phase_dispersion_days",
            "D24_seasonal_amplitude_change_mm",
        ],
        "formula": {
            "D21_dominant_seasonal_peak": "raw peak season from angle(sum_i z_i), set to no_clear_seasonal_peak when B51_seasonality_p90 < 1.0 or D22_phase_coherence < 0.20",
            "D22_phase_coherence": "|sum_i z_i| / (sum_i |z_i| + eps), z_i = annual residual phasor per valid point",
            "D23_phase_dispersion_days": "sqrt(-2 ln Rbar) / (2pi) * 365.25, Rbar = amplitude-weighted circular resultant after excluding the lowest-amplitude 10% of valid points",
            "D24_seasonal_amplitude_change_mm": "median_i(late annual amplitude_i - early annual amplitude_i), with early=time_series[:, 8:155] and late=time_series[:, 155:302]",
        },
        "d21_gates": {
            "B51_seasonality_p90_min": D21_B51_MIN,
            "D22_phase_coherence_min": D21_COHERENCE_MIN,
        },
        "d21_class_counts": df["D21_dominant_seasonal_peak"].value_counts().to_dict(),
        "d21_gate_reason_counts": df["D21_gate_reason"].value_counts().to_dict(),
        "d21_raw_peak_season_counts": df["D21_raw_peak_season"].value_counts().to_dict(),
        "time_window": {
            "t_start": T_START,
            "t_end": T_END,
            "start_year": START_YEAR,
            "cadence_days": CADENCE_DAYS,
        },
        "validity": {
            "min_valid_epochs_per_point": MIN_VALID_EPOCHS,
            "d24_min_valid_epochs_per_window": D24_MIN_VALID_WINDOW_EPOCHS,
            "min_valid_points_per_tile": MIN_VALID_POINTS,
        },
        "score_summary": df_full[
            [
                "D22_phase_coherence",
                "D23_phase_dispersion_days",
                "D24_seasonal_amplitude_change_mm",
                "D22_valid_phase_point_count",
                "D22_median_annual_amplitude_mm",
                "D24_valid_amplitude_point_count",
                "D24_early_annual_amplitude_median_mm",
                "D24_late_annual_amplitude_median_mm",
            ]
        ]
        .describe(percentiles=[0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
        .to_dict(),
    }
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)
    _plot_distribution(df_full, plot_path)
    _plot_d21_distribution(df, d21_plot_path)

    print(f"wrote {table_path}")
    print(f"wrote {detail_path}")
    print(f"wrote {summary_path}")
    print(f"wrote {plot_path}")
    print(f"wrote {d21_plot_path}")
    print("D21 raw peak season counts")
    print(df["D21_raw_peak_season"].value_counts().to_string())
    print("D21 final class counts")
    print(df["D21_dominant_seasonal_peak"].value_counts().to_string())
    print("D21 gate reason counts")
    print(df["D21_gate_reason"].value_counts().to_string())
    print(df_full[[
        "D22_phase_coherence",
        "D23_phase_dispersion_days",
        "D24_seasonal_amplitude_change_mm",
        "D22_median_annual_amplitude_mm",
    ]].describe(
        percentiles=[0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
    ).to_string())


if __name__ == "__main__":
    main()
