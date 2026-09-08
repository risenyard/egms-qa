"""Read canonical task tables and align their tile/split keys."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def merge_task_tables(
    base: pd.DataFrame,
    other: pd.DataFrame,
    name: str,
    *,
    allow_extra: bool = False,
) -> pd.DataFrame:
    """Join task inputs without silently dropping tiles or changing their split.

    A sampled computation may use a complete reference table with
    ``allow_extra=True``; every requested tile must still have a matching key.
    """
    keys = ["tile_id", "split"]
    for label, table in (("base", base), (name, other)):
        if not set(keys) <= set(table):
            raise ValueError(f"{label} must contain tile_id and split")
        if table[keys].isna().any().any() or table["tile_id"].duplicated().any():
            raise ValueError(f"{label} has missing keys or duplicate tile_id values")
    base_index = pd.MultiIndex.from_frame(base[keys])
    other_index = pd.MultiIndex.from_frame(other[keys])
    missing = base_index.difference(other_index)
    extra = other_index.difference(base_index)
    if len(missing) or (len(extra) and not allow_extra):
        raise ValueError(
            f"{name} tile_id/split index mismatch; "
            f"missing={list(missing[:5])} extra={list(extra[:5])}"
        )
    return base.merge(other, on=keys, how="left", validate="one_to_one", suffixes=(False, False))


def read_family(root: Path, family: str, expected_rows: int | None = 10000) -> pd.DataFrame:
    path = root / family / f"{family}_final_table.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    if "tile_id" not in df.columns or "split" not in df.columns:
        raise ValueError(f"{path} must contain tile_id and split")
    if df["tile_id"].duplicated().any():
        dupes = df.loc[df["tile_id"].duplicated(), "tile_id"].head().tolist()
        raise ValueError(f"{path} has duplicate tile_id values: {dupes}")
    if expected_rows is not None and len(df) != expected_rows:
        raise ValueError(f"{path} expected {expected_rows} rows, found {len(df)}")
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
