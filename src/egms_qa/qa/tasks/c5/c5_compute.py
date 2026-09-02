"""Compute C5 spatial monitoring context tasks.

C5 is a composite family. It does not introduce new physical thresholds; it
routes existing B/C labels into monitoring-context answers.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


ROOT = Path(".")
DEFAULT_OUT_DIR = ROOT / "outputs/tasks/c5"
DEFAULT_B2 = ROOT / "outputs/tasks/b2/b2_final_table.csv"
DEFAULT_B3 = ROOT / "outputs/tasks/b3/b3_final_table.csv"
DEFAULT_B6 = ROOT / "outputs/tasks/b6/b6_final_table.csv"
DEFAULT_C3 = ROOT / "outputs/tasks/c3/c3_final_table.csv"

C51_COL = "C51_monitoring_priority"
C52_COL = "C52_hidden_local_risk"


def read_required(path: Path, columns: list[str]) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")
    return df[columns].copy()


def build_table(b2_path: Path, b3_path: Path, b6_path: Path, c3_path: Path) -> pd.DataFrame:
    b6 = read_required(
        b6_path,
        ["tile_id", "split", "B61_monitoring_trigger"],
    )
    c3 = read_required(
        c3_path,
        ["tile_id", "split", "C33_deformation_front_strength_class"],
    )
    b2 = read_required(
        b2_path,
        ["tile_id", "split", "B22_mean_subsidence_intensity_band"],
    )
    b3 = read_required(
        b3_path,
        ["tile_id", "split", "B35_worst_point_significance"],
    )

    out = (
        b6.merge(c3, on=["tile_id", "split"], how="inner", validate="one_to_one")
        .merge(b2, on=["tile_id", "split"], how="inner", validate="one_to_one")
        .merge(b3, on=["tile_id", "split"], how="inner", validate="one_to_one")
    )
    if len(out) != len(b6):
        raise ValueError(f"merged row count {len(out)} does not match B6 row count {len(b6)}")

    out[C51_COL] = "none"
    triggered = out["B61_monitoring_trigger"].astype(str) == "yes"
    sharp_front = out["C33_deformation_front_strength_class"].astype(str) == "very_sharp"
    out.loc[triggered & sharp_front, C51_COL] = "high"
    out.loc[triggered & ~sharp_front, C51_COL] = "standard"

    mean_not_high = out["B22_mean_subsidence_intensity_band"].isin(["low", "low_mid"])
    local_high = out["B35_worst_point_significance"].isin(["high", "very_high"])
    out[C52_COL] = "no"
    out.loc[mean_not_high & local_high, C52_COL] = "yes"

    return out[
        [
            "tile_id",
            "split",
            "B61_monitoring_trigger",
            "C33_deformation_front_strength_class",
            C51_COL,
            "B22_mean_subsidence_intensity_band",
            "B35_worst_point_significance",
            C52_COL,
        ]
    ]


def summarize(final: pd.DataFrame, out_dir: Path, inputs: dict[str, str]) -> None:
    c51_counts = final[C51_COL].value_counts().reindex(["none", "standard", "high"], fill_value=0)
    c52_counts = final[C52_COL].value_counts().reindex(["no", "yes"], fill_value=0)
    summary = {
        "family": "C5",
        "computed_tasks": [C51_COL, C52_COL],
        "n_tiles": int(len(final)),
        "inputs": inputs,
        "rules": {
            C51_COL: "none if B61=no; high if B61=yes and C33=very_sharp; standard otherwise",
            C52_COL: "yes if B22 in {low, low_mid} and B35 in {high, very_high}; no otherwise",
        },
        "c51_counts": c51_counts.astype(int).to_dict(),
        "c51_fractions": (c51_counts / len(final)).astype(float).to_dict(),
        "c52_counts": c52_counts.astype(int).to_dict(),
        "c52_fractions": (c52_counts / len(final)).astype(float).to_dict(),
        "split_counts": final["split"].value_counts().sort_index().astype(int).to_dict(),
        "split_c51_counts": {
            str(split): group[C51_COL].value_counts().reindex(["none", "standard", "high"], fill_value=0).astype(int).to_dict()
            for split, group in final.groupby("split")
        },
        "split_c52_counts": {
            str(split): group[C52_COL].value_counts().reindex(["no", "yes"], fill_value=0).astype(int).to_dict()
            for split, group in final.groupby("split")
        },
    }
    (out_dir / "c5_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def build_outputs(
    out_dir: Path,
    b2_path: Path,
    b3_path: Path,
    b6_path: Path,
    c3_path: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    final = build_table(b2_path, b3_path, b6_path, c3_path)
    final.to_csv(out_dir / "c5_final_table.csv", index=False)
    summarize(
        final,
        out_dir,
        {
            "b2": str(b2_path),
            "b3": str(b3_path),
            "b6": str(b6_path),
            "c3": str(c3_path),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--b2", type=Path, default=DEFAULT_B2)
    parser.add_argument("--b3", type=Path, default=DEFAULT_B3)
    parser.add_argument("--b6", type=Path, default=DEFAULT_B6)
    parser.add_argument("--c3", type=Path, default=DEFAULT_C3)
    args = parser.parse_args()
    build_outputs(args.out_dir, args.b2, args.b3, args.b6, args.c3)


if __name__ == "__main__":
    main()
