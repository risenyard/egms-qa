"""A51/A52 monitoring usability outputs for EGMS-QA.

A51/A52 are deterministic roll-ups of A12/A22/A32/A42. They are not new raw measurements or
encoder probes; they convert the A-group quality checks into a QA-friendly gate.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


ROOT = Path(".")
A1_PATH = ROOT / "outputs/tasks/a1/a1_final_table.csv"
A2_PATH = ROOT / "outputs/tasks/a2/a2_final_table.csv"
A3_PATH = ROOT / "outputs/tasks/a3/a3_final_table.csv"
A4_PATH = ROOT / "outputs/tasks/a4/a4_final_table.csv"
OUT_PATH = ROOT / "outputs/tasks/a5/a5_final_table.csv"


def read_by_tile(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="") as f:
        return {row["tile_id"]: row for row in csv.DictReader(f)}


def classify(row: dict[str, str]) -> tuple[str, str]:
    severe: list[str] = []
    caution: list[str] = []

    if row["A12_representation_stability_class"] == "extreme":
        severe.append("unstable_representation")
    elif row["A12_representation_stability_class"] == "highly_sensitive":
        caution.append("sensitive_representation")

    if row["A22_reconstruction_reliability_class"] == "unreliable":
        severe.append("poor_reconstruction")
    elif row["A22_reconstruction_reliability_class"] == "high_error":
        caution.append("high_reconstruction_error")

    if row["A32_spatial_coverage_class"] == "highly_fragmented":
        severe.append("fragmented_coverage")
    elif row["A32_spatial_coverage_class"] == "sparse":
        caution.append("sparse_coverage")

    if row["A42_noise_level_class"] == "very_high_noise":
        severe.append("very_high_noise")
    elif row["A42_noise_level_class"] == "high_noise":
        caution.append("high_noise")

    if severe:
        if len(severe) > 1 or caution:
            return "unreliable", "multiple_quality_issues"
        return "unreliable", severe[0]
    if caution:
        if len(caution) > 1:
            return "caution", "multiple_minor_issues"
        return "caution", caution[0]
    return "usable", "stable_inputs"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a1-path", default=str(A1_PATH))
    ap.add_argument("--a2-path", default=str(A2_PATH))
    ap.add_argument("--a3-path", default=str(A3_PATH))
    ap.add_argument("--a4-path", default=str(A4_PATH))
    ap.add_argument("--out-path", default=str(OUT_PATH))
    args = ap.parse_args()

    a11 = read_by_tile(Path(args.a1_path))
    a21 = read_by_tile(Path(args.a2_path))
    a31 = read_by_tile(Path(args.a3_path))
    a41 = read_by_tile(Path(args.a4_path))
    tile_ids = sorted(set(a11) & set(a21) & set(a31) & set(a41))
    if len(tile_ids) != len(a11) or len(tile_ids) != len(a21) or len(tile_ids) != len(a31) or len(tile_ids) != len(a41):
        raise ValueError("A11/A21/A31/A41 tile_id sets do not match exactly")

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    class_counts: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "tile_id",
                "split",
                "A51_monitoring_usability_class",
                "A52_monitoring_usability_reason",
                "A12_representation_stability_class",
                "A22_reconstruction_reliability_class",
                "A32_spatial_coverage_class",
                "A42_noise_level_class",
            ],
        )
        writer.writeheader()
        for tile_id in tile_ids:
            row = {
                "tile_id": tile_id,
                "split": a11[tile_id]["split"],
                "A12_representation_stability_class": a11[tile_id]["A12_representation_stability_class"],
                "A22_reconstruction_reliability_class": a21[tile_id]["A22_reconstruction_reliability_class"],
                "A32_spatial_coverage_class": a31[tile_id]["A32_spatial_coverage_class"],
                "A42_noise_level_class": a41[tile_id]["A42_noise_level_class"],
            }
            label, reason = classify(row)
            row["A51_monitoring_usability_class"] = label
            row["A52_monitoring_usability_reason"] = reason
            class_counts[label] += 1
            reason_counts[reason] += 1
            writer.writerow(row)

    print(json.dumps({
        "task": "A51",
        "n_tiles": len(tile_ids),
        "class_counts": dict(class_counts),
        "reason_counts": dict(reason_counts),
        "output": str(out_path),
    }, indent=2))


if __name__ == "__main__":
    main()
