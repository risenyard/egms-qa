from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import pytest

from egms_encoder.data.tile_store import STATIC_KEYS, TileStore


def test_installed_release_contract_constructs_tile_store(
    tmp_path: Path, monkeypatch
) -> None:
    tile_id = "E00N00_x0000000_y0000000"
    tile_path = tmp_path / "data/tiles/E00N00" / f"tile_{tile_id}.npz"
    tile_path.parent.mkdir(parents=True)
    n_points = 3
    np.savez_compressed(
        tile_path,
        coords=np.arange(n_points * 2, dtype=np.float32).reshape(n_points, 2),
        time_series=np.arange(n_points * 294, dtype=np.float32).reshape(n_points, 294),
        **{key: np.arange(n_points, dtype=np.float32) for key in STATIC_KEYS},
    )

    manifest_path = tmp_path / "data/encoder/manifest/split.parquet"
    manifest_path.parent.mkdir(parents=True)
    pd.DataFrame(
        [{
            "tile_id": tile_id,
            "path": f"data/tiles/E00N00/tile_{tile_id}.npz",
            "grid_id": "0_0",
            "split": "train",
            "centroid_x": 1.0,
            "centroid_y": 2.0,
            "n_points": n_points,
        }]
    ).to_parquet(manifest_path, index=False)

    config_path = manifest_path.parent / "data_config.json"
    config = {
        "schema_version": "egms-qa-data-config-1.1",
        "time_window": {
            "stored_steps": 294,
            "t_start": 0,
            "t_end": 294,
            "input_length": 294,
            "end_is_exclusive": True,
            "original_source_steps": 304,
            "original_index_offset": 8,
        },
        "tile_field_layout": {"feature_columns_count": 10},
    }
    config_path.write_text(json.dumps(config), encoding="utf-8")

    monkeypatch.chdir(tmp_path)
    store = TileStore.from_manifest(manifest_path, config_path)
    assert store.num_tiles == 1
    assert store.time_window.input_length == 294
    assert len(store.split_tile_indices("train")) == 1
    assert store.get_tile(0).shape == (n_points, 10 + 294)


def test_tile_store_rejects_shape_that_disagrees_with_config(
    tmp_path: Path, monkeypatch
) -> None:
    tile_id = "E00N00_x0000000_y0000000"
    tile_path = tmp_path / "data/tiles/E00N00" / f"tile_{tile_id}.npz"
    tile_path.parent.mkdir(parents=True)
    np.savez_compressed(
        tile_path,
        coords=np.zeros((2, 2), dtype=np.float32),
        time_series=np.zeros((2, 286), dtype=np.float32),
    )
    manifest_path = tmp_path / "data/encoder/manifest/split.parquet"
    manifest_path.parent.mkdir(parents=True)
    pd.DataFrame(
        [{
            "tile_id": tile_id,
            "path": str(tile_path.relative_to(tmp_path)),
            "split": "train",
            "centroid_x": 0.0,
            "centroid_y": 0.0,
            "n_points": 2,
        }]
    ).to_parquet(manifest_path, index=False)
    config_path = manifest_path.parent / "data_config.json"
    config_path.write_text(
        json.dumps(
            {
                "time_window": {
                    "stored_steps": 294,
                    "t_start": 0,
                    "t_end": 294,
                    "input_length": 294,
                },
                "tile_field_layout": {"feature_columns_count": 10},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    store = TileStore.from_manifest(manifest_path, config_path)
    with pytest.raises(ValueError, match="Refusing implicit or repeated cropping"):
        store.get_tile(0)


def test_tile_store_requires_data_config(tmp_path: Path) -> None:
    manifest_path = tmp_path / "split.parquet"
    pd.DataFrame(
        columns=["tile_id", "path", "n_points", "centroid_x", "centroid_y"]
    ).to_parquet(manifest_path, index=False)
    with pytest.raises(FileNotFoundError, match="data config is required"):
        TileStore.from_manifest(manifest_path, tmp_path / "missing.json")
