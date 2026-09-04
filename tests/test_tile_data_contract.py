from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

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
        time_series=np.arange(n_points * 304, dtype=np.float32).reshape(n_points, 304),
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
        "schema_version": "egms-qa-data-config-1.0",
        "time_window": {
            "source_steps": 304,
            "t_start": 8,
            "t_end": 302,
            "input_length": 294,
            "end_is_exclusive": True,
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
