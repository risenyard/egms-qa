"""Compute the EGMS-QA B61 monitoring trigger table.

B61 is derived from B36 and B42:
  yes if velocity or acceleration typicality is extreme, otherwise no.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(".")
B3_TABLE = ROOT / "outputs/tasks/b3/b3_final_table.csv"
B4_TABLE = ROOT / "outputs/tasks/b4/b4_final_table.csv"
OUT_DIR = ROOT / "outputs/tasks/b6"


def _monitoring_trigger(row: pd.Series) -> str:
    if row["B36_european_velocity_typicality"] == "extreme":
        return "yes"
    if row["B42_european_acceleration_typicality"] == "extreme":
        return "yes"
    return "no"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--b3-table", default=str(B3_TABLE))
    ap.add_argument("--b4-table", default=str(B4_TABLE))
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    args = ap.parse_args()

    b3 = pd.read_csv(args.b3_table)[
        ["tile_id", "split", "B36_european_velocity_typicality"]
    ]
    b4 = pd.read_csv(args.b4_table)[
        ["tile_id", "split", "B42_european_acceleration_typicality"]
    ]
    df = b3.merge(b4, on=["tile_id", "split"], how="inner", validate="one_to_one")
    df["B61_monitoring_trigger"] = df.apply(_monitoring_trigger, axis=1)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "b6_final_table.csv"
    df.to_csv(out_path, index=False)
    print(f"wrote {out_path}")
    print(f"n_tiles={len(df)}")
    print(df["B61_monitoring_trigger"].value_counts().to_string())


if __name__ == "__main__":
    main()
