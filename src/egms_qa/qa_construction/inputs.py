"""Resolve tile manifests against the configured runtime or published release."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from egms_qa.paths import DATA_DIR, ROOT


def read_tile_manifest(path: str | Path, source_tiles_root: str | Path | None = None) -> pd.DataFrame:
    """Resolve runtime or release-relative paths without changing the manifest."""
    frame = pd.read_parquet(path)
    required = {"tile_id", "split", "path"}
    if not required <= set(frame):
        raise ValueError(f"manifest is missing {sorted(required - set(frame))}")
    if frame.empty or frame["tile_id"].duplicated().any():
        raise ValueError("manifest must contain unique tile IDs and at least one tile")
    if frame[["tile_id", "split", "path"]].isna().any().any():
        raise ValueError("manifest keys and paths must not be missing")
    if not set(frame["split"]) <= {"train", "val", "test"}:
        raise ValueError("manifest splits must be train, val or test")

    def resolve(value: str) -> str:
        tile = Path(value)
        if tile.is_absolute():
            return str(tile)
        prefix = tile.parts[:2]
        if source_tiles_root is not None:
            if prefix not in {("data", "tiles"), ("artifacts", "source_tiles")}:
                raise ValueError(f"unrecognized source tile path: {tile}")
            return str(Path(source_tiles_root).resolve().joinpath(*tile.parts[2:]))
        if prefix == ("data", "tiles"):
            return str((DATA_DIR / "tiles").joinpath(*tile.parts[2:]).resolve())
        if prefix == ("artifacts", "source_tiles"):
            return str(Path(path).resolve().parent.parent / tile)
        return str((ROOT / tile).resolve())

    frame = frame.copy()
    frame["path"] = frame["path"].map(resolve)
    return frame


def load_tile_store(manifest_path: str | Path, data_config_path: str | Path):
    """Build an encoder tile reader with resolved paths and explicit splits."""
    from egms_encoder.data.tile_store import TileStore, TimeWindow

    manifest = read_tile_manifest(manifest_path)
    config = json.loads(Path(data_config_path).read_text(encoding="utf-8"))
    return TileStore(
        manifest=manifest,
        time_window=TimeWindow.from_config(config),
        split_assignments=dict(zip(manifest["tile_id"].astype(str), manifest["split"].astype(str))),
        data_config=config,
    )
