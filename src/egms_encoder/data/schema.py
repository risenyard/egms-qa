from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

DATE_COLUMN = re.compile(r"^\d{8}$")


def load_metadata(metadata_path: str | Path) -> dict[str, Any]:
    with Path(metadata_path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def parquet_columns(parquet_path: str | Path) -> list[str]:
    return pq.ParquetFile(parquet_path).schema.names


def time_columns(parquet_path: str | Path, metadata_path: str | Path | None = None) -> list[str]:
    if metadata_path is not None and Path(metadata_path).exists():
        metadata = load_metadata(metadata_path)
        configured = metadata.get("time_series", {}).get("time_columns")
        if configured:
            return list(configured)
    return [name for name in parquet_columns(parquet_path) if DATE_COLUMN.match(name)]


def summarize_dataset(parquet_path: str | Path, metadata_path: str | Path | None = None) -> dict[str, Any]:
    parquet_file = pq.ParquetFile(parquet_path)
    columns = parquet_file.schema.names
    dates = time_columns(parquet_path, metadata_path)
    summary: dict[str, Any] = {
        "rows": parquet_file.metadata.num_rows,
        "row_groups": parquet_file.num_row_groups,
        "columns": len(columns),
        "time_columns": len(dates),
        "first_time_column": dates[0] if dates else None,
        "last_time_column": dates[-1] if dates else None,
    }
    if metadata_path is not None and Path(metadata_path).exists():
        metadata = load_metadata(metadata_path)
        summary.update(
            {
                "dataset_name": metadata.get("dataset_name"),
                "dataset_version": metadata.get("dataset_version"),
                "crs": metadata.get("crs"),
            }
        )
    return summary

