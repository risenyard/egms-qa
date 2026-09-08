from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from egms_encoder.data.tile_store import STATIC_KEYS, TileStore, TimeWindow


def _static_arrays(n_points: int) -> dict[str, np.ndarray]:
    return {key: np.arange(n_points, dtype=np.float32) for key in STATIC_KEYS}


@pytest.mark.parametrize("stored_path", [
    "tile.npz", "nested/tile.npz", "data/tiles/tile.npz",
    "artifacts/source_tiles/tile.npz",
])
def test_local_manifest_infers_metadata_and_resolves_source_root(tmp_path, stored_path):
    from egms_qa.qa_construction.inputs import read_tile_manifest

    relative = stored_path
    for prefix in ("data/tiles/", "artifacts/source_tiles/"):
        if relative.startswith(prefix):
            relative = relative[len(prefix):]
    source = tmp_path / "inputs" / relative
    source.parent.mkdir(parents=True)
    coords = np.array([[0, 1], [2, 3], [4, 5]], dtype=np.float32)
    np.savez(source, coords=coords, time_series=np.zeros((3, 294), dtype=np.float32),
             **_static_arrays(3))
    manifest = tmp_path / "split.parquet"
    pd.DataFrame([{"tile_id": "custom", "split": "test", "path": stored_path,
                   "centroid_x": 100.0}]).to_parquet(manifest, index=False)
    config = tmp_path / "data_config.json"
    config.write_text(json.dumps({
        "schema_version": "egms-qa-data-config-1.1",
        "time_window": {"stored_steps": 294, "t_start": 0, "t_end": 294,
                        "input_length": 294},
        "tile_field_layout": {"feature_columns_count": 10},
    }))
    store = TileStore.from_manifest(manifest, config, source_tiles_root=tmp_path / "inputs")
    assert store.get_tile(0).shape == (3, 304)
    assert store.manifest.loc[0, "n_points"] == 3
    assert store.manifest.loc[0, "centroid_x"] == 100.0
    assert store.manifest.loc[0, "centroid_y"] == 3.0
    resolved = read_tile_manifest(manifest, tmp_path / "inputs")
    assert Path(resolved.loc[0, "path"]) == source


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
        **_static_arrays(n_points),
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
    assert np.array_equal(
        store.split_tile_indices("validation"), store.split_tile_indices("val")
    )
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
                "schema_version": "egms-qa-data-config-1.1",
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


def test_tile_store_rejects_missing_contract_field(tmp_path: Path) -> None:
    tile_path = tmp_path / "tile.npz"
    arrays = {
        "coords": np.zeros((2, 2), dtype=np.float32),
        "time_series": np.zeros((2, 294), dtype=np.float32),
        **{key: np.zeros(2, dtype=np.float32) for key in STATIC_KEYS if key != "rmse"},
    }
    np.savez_compressed(tile_path, **arrays)
    manifest = pd.DataFrame([{
        "tile_id": "tile",
        "path": "tile.npz",
        "n_points": 2,
        "centroid_x": 0.0,
        "centroid_y": 0.0,
    }])
    store = TileStore(
        manifest,
        TimeWindow(0, 294, stored_steps=294),
        data_root=tmp_path,
    )
    with pytest.raises(ValueError, match="missing fields.*rmse"):
        store.get_tile(0)


def test_tile_store_rejects_incomplete_split_assignment(tmp_path: Path) -> None:
    manifest = pd.DataFrame([
        {"tile_id": "a", "path": "a.npz", "n_points": 2, "centroid_x": 0, "centroid_y": 0},
        {"tile_id": "b", "path": "b.npz", "n_points": 2, "centroid_x": 0, "centroid_y": 0},
    ])
    with pytest.raises(ValueError, match="missing tile_id: b"):
        TileStore(
            manifest,
            TimeWindow(0, 294, stored_steps=294),
            split_assignments={"a": "train"},
            data_root=tmp_path,
        )


@pytest.mark.parametrize("start,end", [(-1, 10), (10, 10), (11, 10)])
def test_time_window_is_validated(start: int, end: int) -> None:
    with pytest.raises(ValueError):
        TimeWindow(start, end)


def test_tile_store_requires_data_config(tmp_path: Path) -> None:
    manifest_path = tmp_path / "split.parquet"
    pd.DataFrame(
        columns=["tile_id", "path", "n_points", "centroid_x", "centroid_y"]
    ).to_parquet(manifest_path, index=False)
    with pytest.raises(FileNotFoundError, match="data config is required"):
        TileStore.from_manifest(manifest_path, tmp_path / "missing.json")


@pytest.mark.parametrize(
    ("schema", "stored_steps", "start", "end", "message"),
    [
        ("egms-qa-data-config-1.0", 304, 8, 302, "unsupported data config schema"),
        (
            "egms-qa-data-config-1.1",
            304,
            8,
            302,
            "must store the model-ready.*directly",
        ),
    ],
)
def test_release_loader_rejects_legacy_304_step_contract(
    tmp_path: Path,
    schema: str,
    stored_steps: int,
    start: int,
    end: int,
    message: str,
) -> None:
    manifest_path = tmp_path / "split.parquet"
    pd.DataFrame(
        [
            {
                "tile_id": "tile",
                "path": "tile.npz",
                "split": "train",
                "n_points": 2,
                "centroid_x": 0.0,
                "centroid_y": 0.0,
            }
        ]
    ).to_parquet(manifest_path, index=False)
    config_path = tmp_path / "data_config.json"
    config_path.write_text(
        json.dumps(
            {
                "schema_version": schema,
                "time_window": {
                    "stored_steps": stored_steps,
                    "t_start": start,
                    "t_end": end,
                    "input_length": end - start,
                    "end_is_exclusive": True,
                },
                "tile_field_layout": {"feature_columns_count": 10},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=message):
        TileStore.from_manifest(manifest_path, config_path)
