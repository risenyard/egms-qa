"""Refit the BEAST temporal inputs used by S3 from model-ready NPZ tiles.

Published S3 targets use the frozen posterior input table in the Dataset.
MCMC estimates can vary across Rbeast builds and hardware even with fixed seeds.
Use the frozen inputs for exact corpus reproduction and this command for refits.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
from importlib.metadata import version
import json
from pathlib import Path

import numpy as np
import pandas as pd

from egms_qa.paths import DATA_DIR, OUTPUTS_DIR, SPLIT_MANIFEST
from egms_qa.qa_construction.temporal_inputs import TimeAxis, read_tile_manifest

PARAMETERS = {
    "period": 1.0, "season": "harmonic", "scp_minmax": [0, 2],
    "sorder_minmax": [1, 3], "tcp_minmax": [0, 3], "torder_minmax": [0, 2],
    "mcmc_burbin": 200, "mcmc_samples": 800, "mcmc_chains": 2, "mcmc_thin": 5,
    "quiet": True, "print_param": False, "print_progress": False, "print_warning": False,
}
TIME_AXIS: TimeAxis | None = None


def _configure_axis(axis: TimeAxis) -> None:
    global TIME_AXIS
    TIME_AXIS = axis


def seed_from_tile(tile_id: str) -> int:
    digest = hashlib.blake2s(tile_id.encode("utf-8"), digest_size=4).digest()
    return int.from_bytes(digest, "little", signed=False) % (2**31 - 1)


def compute_tile(row: tuple[str, str, str]) -> dict[str, object]:
    import Rbeast as rb

    if TIME_AXIS is None:
        raise RuntimeError("S3 temporal input time axis is not configured")
    tile_id, split, path = row
    y = TIME_AXIS.tile_median(path)
    base = {"tile_id": str(tile_id), "split": str(split)}
    if np.isfinite(y).sum() < 200:
        return {**base, "S3_trend_order_mean": np.nan, "S3_top_changepoint_probability": np.nan}
    output = rb.beast(y, start=TIME_AXIS.start_year, deltat=TIME_AXIS.delta_years,
                      mcmc_seed=seed_from_tile(str(tile_id)), **PARAMETERS)
    order = np.asarray(getattr(output.trend, "order", []), dtype=np.float64).ravel()
    cp_probability = np.asarray(getattr(output.trend, "cpPr", []), dtype=np.float64).ravel()
    return {
        **base,
        "S3_trend_order_mean": float(np.nanmean(order)) if order.size and np.isfinite(order).any() else np.nan,
        "S3_top_changepoint_probability": float(cp_probability[int(np.nanargmax(cp_probability))]) if cp_probability.size and np.isfinite(cp_probability).any() else np.nan,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=SPLIT_MANIFEST)
    parser.add_argument("--data-config", type=Path, default=DATA_DIR / "encoder/manifest/data_config.json")
    parser.add_argument("--source-tiles-root", type=Path)
    parser.add_argument("--out-dir", type=Path, default=OUTPUTS_DIR / "s3-temporal-refit")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-tiles", type=int, default=0)
    args = parser.parse_args()
    if args.workers < 1 or args.max_tiles < 0:
        parser.error("--workers must be positive and --max-tiles must be non-negative")
    axis = TimeAxis.from_file(args.data_config)
    manifest = read_tile_manifest(args.manifest, args.source_tiles_root)
    if args.max_tiles:
        manifest = manifest.head(args.max_tiles)
    rows = list(manifest[["tile_id", "split", "path"]].itertuples(index=False, name=None))
    if args.workers == 1:
        _configure_axis(axis)
        records = list(map(compute_tile, rows))
    else:
        with ProcessPoolExecutor(max_workers=args.workers, initializer=_configure_axis, initargs=(axis,)) as executor:
            records = list(executor.map(compute_tile, rows, chunksize=2))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame.from_records(records)
    output = args.out_dir / "s3_temporal_inputs.csv"
    frame.to_csv(output, index=False)
    metadata = {
        "schema_version": "egms-qa-s3-temporal-inputs-1.0", "rows": len(frame),
        "estimator": "Rbeast", "estimator_version": version("Rbeast"),
        "parameters": PARAMETERS,
        "seed_rule": "BLAKE2s(tile_id, digest_size=4), unsigned little-endian modulo (2**31-1)",
        "time_axis": {"stored_window": [axis.t_start, axis.t_end], "start_year": axis.start_year, "cadence_days": axis.cadence_days},
        "table_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "role": "refitted posterior inputs; use the Dataset's frozen table to reproduce published S3 targets",
    }
    (args.out_dir / "s3_temporal_inputs_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(frame)} refitted S3 temporal input rows to {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
