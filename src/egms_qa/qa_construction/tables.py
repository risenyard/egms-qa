"""Read canonical task tables and align their tile/split keys."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def read_family(root: Path, family: str) -> pd.DataFrame:
    path = root / family / f"{family}_final_table.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    if "tile_id" not in df.columns or "split" not in df.columns:
        raise ValueError(f"{path} must contain tile_id and split")
    if df["tile_id"].duplicated().any():
        dupes = df.loc[df["tile_id"].duplicated(), "tile_id"].head().tolist()
        raise ValueError(f"{path} has duplicate tile_id values: {dupes}")
    if len(df) != 10000:
        raise ValueError(f"{path} expected 10000 rows, found {len(df)}")
    return df


def align_family_to_base(table: pd.DataFrame, base_index: pd.MultiIndex, family: str) -> pd.DataFrame:
    keyed = table.copy()
    keyed["tile_id"] = keyed["tile_id"].astype(str)
    keyed["split"] = keyed["split"].astype(str)
    keyed = keyed.set_index(["tile_id", "split"], verify_integrity=True)
    missing = base_index.difference(keyed.index)
    extra = keyed.index.difference(base_index)
    if len(missing) or len(extra):
        raise ValueError(
            f"{family} tile_id/split index mismatch; "
            f"missing={list(missing[:5])} extra={list(extra[:5])}"
        )
    return keyed.loc[base_index].reset_index()
