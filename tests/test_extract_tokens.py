from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from egms_encoder.data.tile_store import STATIC_KEYS
from egms_encoder.extract_tokens import (
    DEFAULT_DATASET_REPO,
    DEFAULT_ENCODER_REPO,
    _resolve_source_tile_path,
    load_tile_store,
    parse_args,
    resolve_dataset_inputs,
    resolve_encoder_inputs,
)


def test_default_inputs_are_the_published_hf_repositories() -> None:
    args = parse_args([])
    assert args.encoder_repo == DEFAULT_ENCODER_REPO
    assert args.dataset_repo == DEFAULT_DATASET_REPO
    assert args.checkpoint == ""
    assert args.manifest == ""


def test_local_encoder_overrides_are_atomic() -> None:
    args = parse_args(["--checkpoint", "encoder.safetensors"])
    with pytest.raises(ValueError, match="require"):
        resolve_encoder_inputs(args)

    args = parse_args([
        "--checkpoint", "encoder.safetensors",
        "--model-config", "config.json",
        "--normalization", "normalization.json",
    ])
    resolved = resolve_encoder_inputs(args)
    assert resolved.checkpoint == Path("encoder.safetensors")
    assert resolved.repository is None


def test_local_dataset_overrides_are_atomic() -> None:
    args = parse_args(["--manifest", "split.parquet"])
    with pytest.raises(ValueError, match="require"):
        resolve_dataset_inputs(args)


@pytest.mark.parametrize(
    "manifest_path",
    [
        "data/tiles/E00N00/tile_example.npz",
        "artifacts/source_tiles/E00N00/tile_example.npz",
    ],
)
def test_published_tile_paths_resolve_against_hf_source_tree(
    tmp_path: Path,
    manifest_path: str,
) -> None:
    source_root = tmp_path / "artifacts/source_tiles"
    assert _resolve_source_tile_path(manifest_path, source_root) == (
        source_root / "E00N00/tile_example.npz"
    )


def test_hf_snapshot_manifest_reads_the_published_source_tree(tmp_path: Path) -> None:
    source_root = tmp_path / "snapshot/artifacts/source_tiles"
    tile_path = source_root / "E00N00/tile_example.npz"
    tile_path.parent.mkdir(parents=True)
    n_points = 3
    np.savez_compressed(
        tile_path,
        coords=np.zeros((n_points, 2), dtype=np.float32),
        time_series=np.zeros((n_points, 294), dtype=np.float32),
        **{key: np.zeros(n_points, dtype=np.float32) for key in STATIC_KEYS},
    )
    manifest_path = tmp_path / "snapshot/metadata/split_manifest.parquet"
    manifest_path.parent.mkdir(parents=True)
    pd.DataFrame([{
        "tile_id": "example",
        "path": "data/tiles/E00N00/tile_example.npz",
        "n_points": n_points,
        "centroid_x": 0.0,
        "centroid_y": 0.0,
        "split": "test",
    }]).to_parquet(manifest_path, index=False)
    config_path = manifest_path.parent / "data_config.json"
    config_path.write_text(json.dumps({
        "time_window": {
            "stored_steps": 294,
            "t_start": 0,
            "t_end": 294,
            "input_length": 294,
        },
        "tile_field_layout": {"feature_columns_count": 10},
    }), encoding="utf-8")

    store, time_window, manifest = load_tile_store(
        manifest_path,
        config_path,
        source_root,
    )
    assert time_window.input_length == 294
    assert Path(manifest.loc[0, "path"]) == tile_path
    assert store.get_tile(0).shape == (n_points, 304)
