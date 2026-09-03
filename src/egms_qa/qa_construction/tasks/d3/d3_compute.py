"""Compute EGMS-QA D3 acceleration family targets.

Current targets:
  - D31_motion_intensification_mm_yr2
  - D32_acceleration_support_fraction
  - D33_intensification_spread_mm_yr2
  - D34_intensification_hotspot_strength_mm_yr2
  - D35_intensification_hotspot_location

D3 is a point-level acceleration-field story. It uses point-level motion
direction, point-level acceleration, and point-level RMSE. It does not use a
tile-level direction to flip point-level quantities.
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
OUT_DIR = ROOT / "outputs/tasks/d3"
EPS = 1e-9
MOTION_SNR_MIN = 1.0
MIN_VALID_MOTION_POINTS = 100
MIN_VALID_MOTION_FRACTION = 0.10
GRID = 8
TILE = 7000.0
HALF = TILE / 2.0
MIN_BIN_POINTS = 5


def _bin_index(coords: np.ndarray) -> np.ndarray:
    coords64 = coords.astype(np.float64, copy=False)
    centered = coords64 - coords64.mean(0, keepdims=True)
    bx = np.clip(np.floor((centered[:, 0] + HALF) / TILE * GRID).astype(np.int64), 0, GRID - 1)
    by = np.clip(np.floor((centered[:, 1] + HALF) / TILE * GRID).astype(np.int64), 0, GRID - 1)
    return by * GRID + bx


def _format_bin(bin_id: int | None) -> str:
    if bin_id is None:
        return "none"
    row, col = divmod(int(bin_id), GRID)
    return f"r{row}c{col}"


def _hotspot(
    point_intensification: np.ndarray,
    bidx: np.ndarray,
) -> tuple[float, str, float, int, int]:
    best_bin: int | None = None
    best_score = -np.inf
    best_signed_mean = np.nan
    best_n_points = -1
    valid_bin_count = 0
    for b in range(GRID * GRID):
        sel = bidx == b
        n_points = int(sel.sum())
        if n_points < MIN_BIN_POINTS:
            continue
        valid_bin_count += 1
        values = point_intensification[sel]
        score = float(np.nanmean(np.abs(values)))
        signed_mean = float(np.nanmean(values))
        if score > best_score or (score == best_score and n_points > best_n_points):
            best_bin = b
            best_score = score
            best_signed_mean = signed_mean
            best_n_points = n_points

    if best_bin is None:
        return np.nan, "none", np.nan, valid_bin_count, 0
    return best_score, _format_bin(best_bin), best_signed_mean, valid_bin_count, best_n_points


def _read_one(row: tuple[str, str, str]) -> dict[str, object]:
    tile_id, split, path = row
    with np.load(path) as z:
        coords = z["coords"]
        velocity = z["mean_velocity"].astype(np.float64, copy=False)
        acceleration = z["acceleration"].astype(np.float64, copy=False)
        rmse = z["rmse"].astype(np.float64, copy=False)

    finite_coords = np.isfinite(coords).all(axis=1)
    finite = finite_coords & np.isfinite(velocity) & np.isfinite(acceleration) & np.isfinite(rmse)
    motion_snr = np.divide(
        np.abs(velocity),
        rmse + EPS,
        out=np.full_like(velocity, np.nan, dtype=np.float64),
        where=finite,
    )
    valid_motion = finite & (motion_snr >= MOTION_SNR_MIN) & (velocity != 0)
    n_total = int(finite.sum())
    n_valid = int(valid_motion.sum())
    valid_fraction = float(n_valid / n_total) if n_total else np.nan

    if n_valid == 0:
        return {
            "tile_id": str(tile_id),
            "split": str(split),
            "D31_motion_intensification_mm_yr2": np.nan,
            "D32_acceleration_support_fraction": np.nan,
            "D33_intensification_spread_mm_yr2": np.nan,
            "D34_intensification_hotspot_strength_mm_yr2": np.nan,
            "D35_intensification_hotspot_location": "none",
            "D3_acceleration_validity_reason": "no_valid_motion_points",
            "D3_passes_valid_motion_gate": False,
            "D32_raw_support_reason": "no_valid_motion_points",
            "D31_raw_motion_intensification_mm_yr2": np.nan,
            "D32_raw_acceleration_support_fraction": np.nan,
            "D33_raw_intensification_spread_mm_yr2": np.nan,
            "D34_raw_intensification_hotspot_strength_mm_yr2": np.nan,
            "D35_raw_intensification_hotspot_location": "none",
            "D35_raw_hotspot_signed_mean_intensification_mm_yr2": np.nan,
            "D35_hotspot_valid_bin_count": 0,
            "D35_hotspot_point_count": 0,
            "D31_raw_acceleration_median_mm_yr2": np.nan,
            "D31_point_intensification_mean_mm_yr2": np.nan,
            "D31_point_intensification_p10_mm_yr2": np.nan,
            "D31_point_intensification_p90_mm_yr2": np.nan,
            "D31_acceleration_abs_p90_mm_yr2": np.nan,
            "D31_valid_motion_point_count": 0,
            "D31_finite_point_count": n_total,
            "D31_valid_motion_point_fraction": valid_fraction,
        }

    point_intensification = np.sign(velocity[valid_motion]) * acceleration[valid_motion]
    bidx = _bin_index(coords)[valid_motion]
    raw_d31 = float(np.nanmedian(point_intensification))
    if not np.isfinite(raw_d31) or raw_d31 == 0:
        raw_d32 = np.nan
        reason = "zero_or_undefined_d31_direction"
    elif raw_d31 > 0:
        raw_d32 = float(np.mean(point_intensification > 0))
        reason = "defined"
    else:
        raw_d32 = float(np.mean(point_intensification < 0))
        reason = "defined"

    raw_d33 = float(
        np.nanpercentile(point_intensification, 90)
        - np.nanpercentile(point_intensification, 10)
    )
    raw_d34, raw_d35, raw_d35_signed, valid_bin_count, hotspot_point_count = _hotspot(
        point_intensification,
        bidx,
    )
    passes_gate = (
        n_valid >= MIN_VALID_MOTION_POINTS
        and np.isfinite(valid_fraction)
        and valid_fraction >= MIN_VALID_MOTION_FRACTION
    )
    if not passes_gate:
        d31 = np.nan
        d32 = np.nan
        d33 = np.nan
        d34 = np.nan
        d35 = "none"
        validity_reason = "insufficient_valid_motion_points"
    else:
        d31 = raw_d31
        d32 = raw_d32
        d33 = raw_d33
        d34 = raw_d34
        d35 = raw_d35
        validity_reason = "valid" if reason == "defined" else reason

    return {
        "tile_id": str(tile_id),
        "split": str(split),
        "D31_motion_intensification_mm_yr2": d31,
        "D32_acceleration_support_fraction": d32,
        "D33_intensification_spread_mm_yr2": d33,
        "D34_intensification_hotspot_strength_mm_yr2": d34,
        "D35_intensification_hotspot_location": d35,
        "D3_acceleration_validity_reason": validity_reason,
        "D3_passes_valid_motion_gate": passes_gate,
        "D32_raw_support_reason": reason,
        "D31_raw_motion_intensification_mm_yr2": raw_d31,
        "D32_raw_acceleration_support_fraction": raw_d32,
        "D33_raw_intensification_spread_mm_yr2": raw_d33,
        "D34_raw_intensification_hotspot_strength_mm_yr2": raw_d34,
        "D35_raw_intensification_hotspot_location": raw_d35,
        "D35_raw_hotspot_signed_mean_intensification_mm_yr2": raw_d35_signed,
        "D35_hotspot_valid_bin_count": valid_bin_count,
        "D35_hotspot_point_count": hotspot_point_count,
        "D31_raw_acceleration_median_mm_yr2": float(np.nanmedian(acceleration[valid_motion])),
        "D31_point_intensification_mean_mm_yr2": float(np.nanmean(point_intensification)),
        "D31_point_intensification_p10_mm_yr2": float(np.nanpercentile(point_intensification, 10)),
        "D31_point_intensification_p90_mm_yr2": float(np.nanpercentile(point_intensification, 90)),
        "D31_acceleration_abs_p90_mm_yr2": float(np.nanpercentile(np.abs(acceleration[valid_motion]), 90)),
        "D31_valid_motion_point_count": n_valid,
        "D31_finite_point_count": n_total,
        "D31_valid_motion_point_fraction": valid_fraction,
    }


def _plot_distribution(df: pd.DataFrame, out_path: Path) -> None:
    d31 = df["D31_motion_intensification_mm_yr2"].dropna()
    d32 = df["D32_acceleration_support_fraction"].dropna()
    d33 = df["D33_intensification_spread_mm_yr2"].dropna()
    d34 = df["D34_intensification_hotspot_strength_mm_yr2"].dropna()
    lo, hi = d31.quantile([0.01, 0.99]) if len(d31) else (np.nan, np.nan)
    if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
        lo, hi = (-1.0, 1.0) if not len(d31) else (float(d31.min()), float(d31.max()))
    d33_lo, d33_hi = d33.quantile([0.01, 0.99]) if len(d33) else (np.nan, np.nan)
    if not np.isfinite(d33_lo) or not np.isfinite(d33_hi) or d33_lo >= d33_hi:
        d33_lo, d33_hi = (0.0, 3.0) if not len(d33) else (float(d33.min()), float(d33.max()))
    d34_lo, d34_hi = d34.quantile([0.01, 0.99]) if len(d34) else (np.nan, np.nan)
    if not np.isfinite(d34_lo) or not np.isfinite(d34_hi) or d34_lo >= d34_hi:
        d34_lo, d34_hi = (0.0, 3.0) if not len(d34) else (float(d34.min()), float(d34.max()))

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)

    ax = axes[0, 0]
    ax.hist(d31, bins=80, range=(lo, hi), color="#4c78a8")
    ax.axvline(0.0, color="#222222", linestyle="--", linewidth=1.0)
    ax.set_xlabel("D31 motion intensification (mm/yr^2)")
    ax.set_ylabel("tiles")

    ax = axes[0, 1]
    ax.hist(d32, bins=60, range=(0.5, 1.0), color="#f58518")
    ax.set_xlabel("D32 acceleration support fraction")
    ax.set_ylabel("tiles")

    ax = axes[1, 0]
    ax.hist(d33, bins=80, range=(d33_lo, d33_hi), color="#b279a2")
    ax.set_xlabel("D33 intensification spread (mm/yr^2)")
    ax.set_ylabel("tiles")

    ax = axes[1, 1]
    ax.hist(d34, bins=80, range=(d34_lo, d34_hi), color="#72b7b2")
    ax.set_xlabel("D34 hotspot strength (mm/yr^2)")
    ax.set_ylabel("tiles")

    location_counts = np.zeros((GRID, GRID), dtype=np.int64)
    for loc, count in df["D35_intensification_hotspot_location"].value_counts().items():
        if not isinstance(loc, str) or loc == "none":
            continue
        try:
            row_s, col_s = loc[1:].split("c", 1)
            location_counts[int(row_s), int(col_s)] = int(count)
        except (ValueError, IndexError):
            continue
    ax = axes[0, 2]
    im = ax.imshow(location_counts, origin="upper", cmap="viridis")
    ax.set_xlabel("D35 hotspot col")
    ax.set_ylabel("D35 hotspot row")
    ax.set_xticks(range(GRID))
    ax.set_yticks(range(GRID))
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    reason_counts = df["D3_acceleration_validity_reason"].value_counts().sort_values()
    ax = axes[1, 2]
    ax.barh(reason_counts.index, reason_counts.values, color="#54a24b")
    ax.set_xlabel("tiles")
    ax.set_ylabel("D3 validity reason")

    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(MANIFEST))
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))
    ap.add_argument("--chunksize", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    manifest = pd.read_parquet(args.manifest)
    if args.limit:
        manifest = manifest.head(args.limit).copy()
    rows = [(str(r.tile_id), str(r.split), str(r.path)) for r in manifest.itertuples(index=False)]

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        records = list(ex.map(_read_one, rows, chunksize=args.chunksize))

    diagnostics = pd.DataFrame.from_records(records)
    final = diagnostics[
        [
            "tile_id",
            "split",
            "D31_motion_intensification_mm_yr2",
            "D32_acceleration_support_fraction",
            "D33_intensification_spread_mm_yr2",
            "D34_intensification_hotspot_strength_mm_yr2",
            "D35_intensification_hotspot_location",
            "D3_acceleration_validity_reason",
        ]
    ].copy()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "sample" if args.limit else "final"
    final_path = out_dir / f"d3_{suffix}_table.csv"
    diagnostics_path = out_dir / f"d3_{suffix}_diagnostics.csv"
    summary_path = out_dir / f"d3_{suffix}_summary.json"
    plot_path = out_dir / f"d3_{suffix}_distribution.png"

    final.to_csv(final_path, index=False)
    diagnostics.to_csv(diagnostics_path, index=False)
    score_cols = [
        "D31_motion_intensification_mm_yr2",
        "D32_acceleration_support_fraction",
        "D33_intensification_spread_mm_yr2",
        "D34_intensification_hotspot_strength_mm_yr2",
        "D31_raw_motion_intensification_mm_yr2",
        "D32_raw_acceleration_support_fraction",
        "D33_raw_intensification_spread_mm_yr2",
        "D34_raw_intensification_hotspot_strength_mm_yr2",
        "D35_raw_hotspot_signed_mean_intensification_mm_yr2",
        "D35_hotspot_valid_bin_count",
        "D35_hotspot_point_count",
        "D31_raw_acceleration_median_mm_yr2",
        "D31_point_intensification_mean_mm_yr2",
        "D31_point_intensification_p10_mm_yr2",
        "D31_point_intensification_p90_mm_yr2",
        "D31_acceleration_abs_p90_mm_yr2",
        "D31_valid_motion_point_count",
        "D31_valid_motion_point_fraction",
    ]
    summary = {
        "n_tiles": int(len(final)),
        "tasks": [
            "D31_motion_intensification_mm_yr2",
            "D32_acceleration_support_fraction",
            "D33_intensification_spread_mm_yr2",
            "D34_intensification_hotspot_strength_mm_yr2",
            "D35_intensification_hotspot_location",
        ],
        "formula": {
            "valid_motion_point": "finite(mean_velocity, acceleration, rmse) and abs(mean_velocity)/(rmse+eps) >= 1",
            "valid_tile_gate": "D31_valid_motion_point_count >= 100 and D31_valid_motion_point_fraction >= 0.10",
            "point_intensification": "sign(mean_velocity_i) * acceleration_i",
            "D31_motion_intensification_mm_yr2": "median_i(point_intensification_i over valid motion points); NaN when the valid tile gate fails",
            "D32_acceleration_support_fraction": "fraction of valid motion points with the same nonzero sign as D31; NaN when D31 has no nonzero direction or the valid tile gate fails",
            "D33_intensification_spread_mm_yr2": "p90(point_intensification_i) - p10(point_intensification_i) over valid motion points; NaN when the valid tile gate fails",
            "D34_intensification_hotspot_strength_mm_yr2": "max over 8x8 bins of mean(abs(point_intensification_i)); each bin requires at least 5 valid motion points; NaN when the valid tile gate fails",
            "D35_intensification_hotspot_location": "r{row}c{col} location of the D34 max bin; none when the valid tile gate fails or no valid bin exists",
        },
        "interpretation": {
            "D31_positive": "dominant motion is intensifying",
            "D31_negative": "dominant motion is weakening",
            "D31_near_zero": "dominant motion has weak central acceleration change",
            "D32_high": "the D31 direction is spatially consistent across valid moving points",
            "D32_near_half": "the D31 direction has weak spatial support",
            "D33_high": "direction-aware acceleration varies strongly across valid moving points",
            "D33_low": "direction-aware acceleration is more spatially compact or uniform",
            "D34_high": "the strongest 8x8 local bin has strong direction-aware acceleration activity",
            "D35_location": "local 8x8 bin where the D34 hotspot occurs",
        },
        "validity": {
            "motion_snr_min": MOTION_SNR_MIN,
            "min_valid_motion_points": MIN_VALID_MOTION_POINTS,
            "min_valid_motion_fraction": MIN_VALID_MOTION_FRACTION,
            "D31_is_nan_when": "no valid motion points exist or the valid tile gate fails",
            "D32_is_nan_when": "D31 is zero, undefined, no valid motion points exist, or the valid tile gate fails",
            "D33_is_nan_when": "no valid motion points exist or the valid tile gate fails",
            "D34_is_nan_when": "no valid motion points exist, the valid tile gate fails, or no 8x8 bin has at least 5 valid motion points",
            "D35_is_none_when": "no valid motion points exist, the valid tile gate fails, or no 8x8 bin has at least 5 valid motion points",
            "hotspot_grid": f"{GRID}x{GRID}",
            "hotspot_min_bin_points": MIN_BIN_POINTS,
        },
        "intentional_exclusions": [
            "D3 does not use a tile-level B34 direction to flip point-level acceleration.",
            "D31/D32 are not acceleration-strength tasks; B41_acc_abs_p90 already measures strength.",
            "D33 is not raw acceleration strength; it is the spread of direction-aware point intensification.",
            "D34 is not raw acceleration strength; it is the strongest local bin of direction-aware point intensification.",
            "D31/D32/D33/D34 are scalar-only and are not classified into hard bands.",
        ],
        "score_summary": diagnostics[score_cols]
        .describe(percentiles=[0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
        .to_dict(),
        "d3_validity_reason_counts": diagnostics["D3_acceleration_validity_reason"].value_counts(dropna=False).to_dict(),
        "d35_location_counts": diagnostics["D35_intensification_hotspot_location"].value_counts(dropna=False).head(20).to_dict(),
        "d32_raw_support_reason_counts": diagnostics["D32_raw_support_reason"].value_counts(dropna=False).to_dict(),
    }
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)
    _plot_distribution(diagnostics, plot_path)

    print(f"wrote {final_path}")
    print(f"wrote {diagnostics_path}")
    print(f"wrote {summary_path}")
    print(f"wrote {plot_path}")
    print(diagnostics[score_cols].describe(
        percentiles=[0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
    ).to_string())
    print(diagnostics["D3_acceleration_validity_reason"].value_counts(dropna=False).to_string())
    print(diagnostics["D35_intensification_hotspot_location"].value_counts(dropna=False).head(20).to_string())
    print(diagnostics["D32_raw_support_reason"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
