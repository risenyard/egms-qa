"""Compute the EGMS-QA C4 fast-tail spatial-extent family.

The family has two stages:

1. reference:
   Build a European reference distribution over valid-bin
   ``bin_abs_velocity_p90`` values from the full V4 candidate pool.

2. final:
   Use a frozen ``T_fast`` from that reference distribution to compute
   C41/C42 for the 10k VQA tiles.
"""
from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(".")
VQA_MANIFEST = ROOT / "data/encoder/manifest/split.parquet"
OUT_DIR = ROOT / "outputs/tasks/c4"

GRID = 8
TILE = 7000.0
HALF = TILE / 2.0
MIN_BIN_POINTS = 5
EPS = 1e-6

REFERENCE_JSON = "c4_bin_level_reference_thresholds.json"


def _bin_index(coords: np.ndarray) -> np.ndarray:
    coords64 = coords.astype(np.float64, copy=False)
    centered = coords64 - coords64.mean(0, keepdims=True)
    bx = np.clip(np.floor((centered[:, 0] + HALF) / TILE * GRID).astype(np.int64), 0, GRID - 1)
    by = np.clip(np.floor((centered[:, 1] + HALF) / TILE * GRID).astype(np.int64), 0, GRID - 1)
    return by * GRID + bx


def _tile_bin_abs_velocity_p90(path: str) -> tuple[np.ndarray, int, str]:
    try:
        with np.load(path) as z:
            coords = z["coords"]
            mv = z["mean_velocity"].astype(np.float64, copy=False)
    except Exception as exc:
        return np.asarray([], dtype=np.float32), 0, f"read_error:{type(exc).__name__}"

    if coords.shape[0] != mv.shape[0]:
        return np.asarray([], dtype=np.float32), 0, "shape_mismatch"

    finite = np.isfinite(mv) & np.isfinite(coords).all(axis=1)
    if int(finite.sum()) < MIN_BIN_POINTS:
        return np.asarray([], dtype=np.float32), 0, "too_few_points"

    coords = coords[finite]
    abs_mv = np.abs(mv[finite])
    bidx = _bin_index(coords)

    values: list[float] = []
    for b in range(GRID * GRID):
        sel = bidx == b
        if int(sel.sum()) >= MIN_BIN_POINTS:
            values.append(float(np.nanpercentile(abs_mv[sel], 90)))

    return np.asarray(values, dtype=np.float32), len(values), "ok"


def _reference_one(row: tuple[str, str]) -> dict[str, object]:
    tile_id, path = row
    values, valid_bins, status = _tile_bin_abs_velocity_p90(path)
    return {
        "tile_id": tile_id,
        "valid_bin_count_8x8": int(valid_bins),
        "status": status,
        "values": values,
    }


def _quantiles(values: np.ndarray, probs: list[float]) -> dict[str, float]:
    qs = np.nanpercentile(values, probs)
    return {f"p{p:g}": float(v) for p, v in zip(probs, qs)}


def _safe_log_values(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values) & (values > EPS)]
    return np.log(values)


def _write_reference_plot(values: np.ndarray, out_path: Path, thresholds: dict[str, float]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[plot] skipped: {type(exc).__name__}: {exc}", flush=True)
        return

    x = values[np.isfinite(values)]
    x = x[(x >= 0.0) & (x <= np.nanpercentile(x, 99.5))]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(x, bins=120, color="#557A95", alpha=0.82)
    for key, color in [("p90", "#D1495B"), ("p95", "#EDAE49"), ("p99", "#00798C")]:
        if key in thresholds:
            ax.axvline(thresholds[key], color=color, lw=2, label=f"{key}={thresholds[key]:.3f}")
    ax.set_title("C4 full-Europe valid-bin abs-velocity p90 distribution")
    ax.set_xlabel("bin_abs_velocity_p90 (mm/yr)")
    ax.set_ylabel("valid-bin count")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def run_reference(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_parquet(args.manifest)
    if args.max_tiles:
        manifest = manifest.iloc[: args.max_tiles].copy()
    rows = [(str(r.tile_id), str(r.path)) for r in manifest.itertuples(index=False)]
    print(f"[reference] tiles={len(rows)} manifest={args.manifest}", flush=True)

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        records = list(ex.map(_reference_one, rows, chunksize=args.chunksize))

    status = pd.Series([r["status"] for r in records], dtype="object").value_counts()
    counts = np.asarray([r["valid_bin_count_8x8"] for r in records], dtype=np.int32)
    values = np.concatenate([r["values"] for r in records if len(r["values"]) > 0]).astype(np.float64)
    if values.size == 0:
        raise RuntimeError("no valid bin values computed")

    probs = [0.1, 1, 5, 10, 25, 50, 75, 90, 95, 97.5, 99, 99.5, 99.9]
    raw_quantiles = _quantiles(values, probs)

    log_values = _safe_log_values(values)
    lo, hi = np.nanpercentile(log_values, [1, 99])
    bulk = log_values[(log_values >= lo) & (log_values <= hi)]
    log_mu = float(np.nanmean(bulk))
    log_sigma = float(np.nanstd(bulk))
    log_z_thresholds = {
        "z1": float(np.exp(log_mu + log_sigma)),
        "z1.5": float(np.exp(log_mu + 1.5 * log_sigma)),
        "z2": float(np.exp(log_mu + 2.0 * log_sigma)),
        "z2.5": float(np.exp(log_mu + 2.5 * log_sigma)),
    }

    summary = {
        "manifest": str(Path(args.manifest).resolve()),
        "n_tiles": int(len(rows)),
        "n_ok_tiles": int(status.get("ok", 0)),
        "n_valid_bins": int(values.size),
        "grid": GRID,
        "min_bin_points": MIN_BIN_POINTS,
        "bin_scalar": "p90(abs(point mean_velocity)) within each valid 8x8 bin",
        "raw_quantiles_mm_yr": raw_quantiles,
        "log_bulk_fit": {
            "trim": "log values between p1 and p99",
            "mu": log_mu,
            "sigma": log_sigma,
            "raw_thresholds_mm_yr": log_z_thresholds,
        },
        "status_counts": {str(k): int(v) for k, v in status.items()},
        "valid_bin_count_8x8": {
            "mean": float(np.nanmean(counts)),
            "p10": float(np.nanpercentile(counts, 10)),
            "p50": float(np.nanpercentile(counts, 50)),
            "p90": float(np.nanpercentile(counts, 90)),
        },
    }

    with (out_dir / REFERENCE_JSON).open("w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    pd.DataFrame(
        {
            "quantile": list(raw_quantiles.keys()),
            "bin_abs_velocity_p90_mm_yr": list(raw_quantiles.values()),
        }
    ).to_csv(out_dir / "c4_bin_level_reference_quantiles.csv", index=False)

    pd.DataFrame(
        {
            "threshold": list(log_z_thresholds.keys()),
            "bin_abs_velocity_p90_mm_yr": list(log_z_thresholds.values()),
        }
    ).to_csv(out_dir / "c4_bin_level_reference_logz_thresholds.csv", index=False)

    pd.DataFrame(
        {
            "tile_id": [r["tile_id"] for r in records],
            "valid_bin_count_8x8": [r["valid_bin_count_8x8"] for r in records],
            "status": [r["status"] for r in records],
        }
    ).to_csv(out_dir / "c4_bin_level_reference_tile_status.csv", index=False)

    np.savez_compressed(
        out_dir / "c4_bin_level_reference_values_sample.npz",
        bin_abs_velocity_p90=values
        if values.size <= args.sample_cap
        else np.random.default_rng(args.seed).choice(values, size=args.sample_cap, replace=False),
    )

    plot_thresholds = {
        "p90": raw_quantiles["p90"],
        "p95": raw_quantiles["p95"],
        "p99": raw_quantiles["p99"],
    }
    _write_reference_plot(values, out_dir / "c4_bin_level_reference_distribution.png", plot_thresholds)

    print(f"[reference] n_valid_bins={values.size:,}", flush=True)
    print("[reference] raw quantiles", flush=True)
    for key, value in raw_quantiles.items():
        print(f"  {key:>5}: {value:.6f}", flush=True)
    print("[reference] log-z raw thresholds", flush=True)
    for key, value in log_z_thresholds.items():
        print(f"  {key:>5}: {value:.6f}", flush=True)
    print(f"[reference] wrote {out_dir}", flush=True)


def _extent_class(fraction: float, fast_bins: int) -> str:
    if fast_bins <= 0:
        return "none"
    if fast_bins <= 2:
        return "sparse"
    if fraction < 0.25:
        return "localized"
    return "extensive"


def _final_one(row: tuple[str, str, str], threshold: float) -> dict[str, object]:
    tile_id, split, path = row
    values, valid_bins, status = _tile_bin_abs_velocity_p90(path)
    if status != "ok" or valid_bins <= 0:
        fast_bins = 0
        fraction = 0.0
    else:
        fast_bins = int(np.sum(values >= threshold))
        fraction = float(fast_bins / valid_bins)
    return {
        "tile_id": tile_id,
        "split": split,
        "valid_bin_count_8x8": int(valid_bins),
        "fast_tail_bin_count": int(fast_bins),
        "C41_fast_tail_bin_fraction": fraction,
    }


def _resolve_threshold(args: argparse.Namespace) -> float:
    if args.fast_threshold is not None:
        return float(args.fast_threshold)

    ref_path = Path(args.reference_json)
    with ref_path.open() as f:
        ref = json.load(f)

    key = args.threshold_key
    if key.startswith("p"):
        q = ref["raw_quantiles_mm_yr"]
        if key not in q:
            raise KeyError(f"{key} not found in raw_quantiles_mm_yr")
        return float(q[key])

    z = ref["log_bulk_fit"]["raw_thresholds_mm_yr"]
    if key not in z:
        raise KeyError(f"{key} not found in log_bulk_fit.raw_thresholds_mm_yr")
    return float(z[key])


def run_final(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    threshold = _resolve_threshold(args)

    manifest = pd.read_parquet(args.manifest)
    if args.max_tiles:
        manifest = manifest.iloc[: args.max_tiles].copy()
    rows = [(str(r.tile_id), str(r.split), str(r.path)) for r in manifest.itertuples(index=False)]
    print(f"[final] tiles={len(rows)} threshold={threshold:.6f}", flush=True)

    worker = partial(_final_one, threshold=threshold)
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        records = list(ex.map(worker, rows, chunksize=args.chunksize))

    df = pd.DataFrame.from_records(records)
    df["C42_fast_tail_extent_class"] = [
        _extent_class(float(frac), int(n_fast))
        for frac, n_fast in zip(df["C41_fast_tail_bin_fraction"], df["fast_tail_bin_count"])
    ]
    df["T_fast_bin_abs_velocity_p90_mm_yr"] = threshold
    df.to_csv(out_dir / "c4_final_table.csv", index=False)

    summary = {
        "manifest": str(Path(args.manifest).resolve()),
        "n_tiles": int(len(df)),
        "T_fast_bin_abs_velocity_p90_mm_yr": threshold,
        "threshold_key": args.threshold_key if args.fast_threshold is None else "manual",
        "C42_rule": {
            "none": "fast_tail_bin_count = 0",
            "sparse": "fast_tail_bin_count in [1, 2]",
            "localized": "fast_tail_bin_count >= 3 and C41_fast_tail_bin_fraction < 0.25",
            "extensive": "fast_tail_bin_count >= 3 and C41_fast_tail_bin_fraction >= 0.25",
        },
        "C41_quantiles": _quantiles(
            df["C41_fast_tail_bin_fraction"].to_numpy(dtype=np.float64),
            [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100],
        ),
        "C42_counts": {
            str(k): int(v) for k, v in df["C42_fast_tail_extent_class"].value_counts().items()
        },
    }
    with (out_dir / "c4_final_summary.json").open("w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    print(f"[final] wrote {out_dir / 'c4_final_table.csv'}", flush=True)
    print(df["C41_fast_tail_bin_fraction"].describe(percentiles=[0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]).to_string(), flush=True)
    print(df["C42_fast_tail_extent_class"].value_counts().to_string(), flush=True)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    ref = sub.add_parser("reference")
    ref.add_argument("--manifest", required=True,
                     help="Full European candidate-pool manifest (parquet). Not shipped; "
                          "only needed to RE-derive the threshold. The released reference JSON "
                          "(outputs/tasks/c4/c4_bin_level_reference_thresholds.json) already carries it.")
    ref.add_argument("--out-dir", default=str(OUT_DIR))
    ref.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "16")))
    ref.add_argument("--chunksize", type=int, default=32)
    ref.add_argument("--sample-cap", type=int, default=2_000_000)
    ref.add_argument("--seed", type=int, default=0)
    ref.add_argument("--max-tiles", type=int, default=0)
    ref.set_defaults(func=run_reference)

    final = sub.add_parser("final")
    final.add_argument("--manifest", default=str(VQA_MANIFEST))
    final.add_argument("--out-dir", default=str(OUT_DIR))
    final.add_argument("--reference-json", default=str(OUT_DIR / REFERENCE_JSON))
    final.add_argument("--threshold-key", default="p95")
    final.add_argument("--fast-threshold", type=float, default=None)
    final.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "16")))
    final.add_argument("--chunksize", type=int, default=32)
    final.add_argument("--max-tiles", type=int, default=0)
    final.set_defaults(func=run_final)

    return ap


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
