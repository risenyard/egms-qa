"""A31/A32 spatial observation coverage outputs for EGMS-QA.

A31/A32 are intentionally simple for QA generation:
  - scalar: fraction of 8x8 spatial bins that contain at least one observation
  - label: fixed structural coverage class from that fraction
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import torch


ROOT = Path(".")
TOKEN_CACHE = ROOT / "data/encoder/tokens/egms_tokens_10k.pt"
OUT_PATH = ROOT / "outputs/tasks/a3/a3_final_table.csv"
GRID_BINS = 64


def coverage_class(valid_fraction: float) -> str:
    if valid_fraction >= 0.75:
        return "well_spread"
    if valid_fraction >= 0.50:
        return "moderate_gaps"
    if valid_fraction >= 0.25:
        return "sparse"
    return "highly_fragmented"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--token-cache", default=str(TOKEN_CACHE))
    ap.add_argument("--out-path", default=str(OUT_PATH))
    args = ap.parse_args()

    cache = torch.load(args.token_cache, map_location="cpu", weights_only=False)
    counts = cache["point_count_per_bin"]
    tile_ids = [str(x) for x in cache["tile_ids"]]
    splits = [str(x) for x in cache["splits"]]
    if counts.ndim != 2 or counts.shape[1] != GRID_BINS:
        raise ValueError(f"expected point_count_per_bin shape [N,{GRID_BINS}], got {tuple(counts.shape)}")
    if len(tile_ids) != counts.shape[0] or len(splits) != counts.shape[0]:
        raise ValueError("token cache tile_id/split length mismatch")

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    class_counts: Counter[str] = Counter()
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "tile_id",
                "split",
                "A31_valid_bin_fraction_8x8",
                "A32_spatial_coverage_class",
            ],
        )
        writer.writeheader()
        for tile_id, split, row in zip(tile_ids, splits, counts):
            valid_fraction = float((row > 0).sum().item() / GRID_BINS)
            label = coverage_class(valid_fraction)
            class_counts[label] += 1
            writer.writerow({
                "tile_id": tile_id,
                "split": split,
                "A31_valid_bin_fraction_8x8": f"{valid_fraction:.6f}",
                "A32_spatial_coverage_class": label,
            })

    summary = {
        "task": "A31",
        "n_tiles": int(counts.shape[0]),
        "scalar": "A31_valid_bin_fraction_8x8",
        "label": "A32_spatial_coverage_class",
        "class_counts": dict(class_counts),
        "output": str(out_path),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
