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
    TOKEN_SCHEMA,
    _published_tile_repo_path,
    _resolve_revision,
    load_tile_store,
    parse_args,
    pool_to_spatial_tokens,
    resolve_dataset_inputs,
    resolve_encoder_inputs,
)


def test_default_inputs_are_the_published_hf_repositories() -> None:
    args = parse_args([])
    assert args.encoder_repo == DEFAULT_ENCODER_REPO
    assert args.dataset_repo == DEFAULT_DATASET_REPO
    assert args.encoder_revision == "main"
    assert args.dataset_revision == "main"
    assert args.checkpoint == ""
    assert args.model_config == ""
    assert args.normalization == ""
    assert args.manifest == ""
    assert args.data_config == ""
    assert args.output_name == ""
    assert TOKEN_SCHEMA == "egms-tokens-1.1"


def test_local_encoder_overrides_are_atomic() -> None:
    args = parse_args(["--checkpoint", "encoder.safetensors"])
    with pytest.raises(ValueError, match="require"):
        resolve_encoder_inputs(args)

    args = parse_args(
        [
            "--checkpoint",
            "encoder.safetensors",
            "--model-config",
            "config.json",
            "--normalization",
            "normalization.json",
        ]
    )
    resolved = resolve_encoder_inputs(args)
    assert resolved.checkpoint == Path("encoder.safetensors")
    assert resolved.repository is None
    assert resolved.revision is None


def test_local_dataset_overrides_are_atomic() -> None:
    args = parse_args(["--manifest", "split.parquet"])
    with pytest.raises(ValueError, match="require"):
        resolve_dataset_inputs(args)

    args = parse_args(
        [
            "--manifest",
            "split.parquet",
            "--data-config",
            "data_config.json",
            "--source-tiles-root",
            "tiles",
        ]
    )
    resolved = resolve_dataset_inputs(args)
    assert resolved.manifest == Path("split.parquet")
    assert resolved.source_tiles_root == Path("tiles")
    assert resolved.repository is None


@pytest.mark.parametrize(
    ("manifest_path", "expected"),
    [
        (
            "data/tiles/E00N00/tile_example.npz",
            "artifacts/source_tiles/E00N00/tile_example.npz",
        ),
        (
            "artifacts/source_tiles/E00N00/tile_example.npz",
            "artifacts/source_tiles/E00N00/tile_example.npz",
        ),
    ],
)
def test_published_tile_repo_paths_are_canonical(
    manifest_path: str,
    expected: str,
) -> None:
    assert _published_tile_repo_path(manifest_path) == Path(expected)


def test_published_tile_repo_path_rejects_other_roots() -> None:
    with pytest.raises(ValueError, match="outside"):
        _published_tile_repo_path("other/location/tile.npz")


def test_hf_snapshot_manifest_uses_strict_294_step_store(tmp_path: Path) -> None:
    source_root = tmp_path / "snapshot/artifacts/source_tiles"
    tile_path = source_root / "E00N00/tile_example.npz"
    tile_path.parent.mkdir(parents=True)
    point_count = 3
    np.savez_compressed(
        tile_path,
        coords=np.zeros((point_count, 2), dtype=np.float32),
        time_series=np.zeros((point_count, 294), dtype=np.float32),
        **{key: np.zeros(point_count, dtype=np.float32) for key in STATIC_KEYS},
    )
    manifest_path = tmp_path / "snapshot/metadata/split_manifest.parquet"
    manifest_path.parent.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "tile_id": "example",
                "path": "data/tiles/E00N00/tile_example.npz",
                "n_points": point_count,
                "centroid_x": 0.0,
                "centroid_y": 0.0,
                "split": "test",
            }
        ]
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
                    "end_is_exclusive": True,
                },
                "tile_field_layout": {"feature_columns_count": 10},
            }
        ),
        encoding="utf-8",
    )

    store, manifest = load_tile_store(manifest_path, config_path, source_root)
    assert store.time_window.input_length == 294
    assert Path(manifest.loc[0, "path"]) == tile_path
    assert store.get_tile(0).shape == (point_count, 304)


def test_full_commit_revision_does_not_require_api_resolution() -> None:
    revision = "A" * 40
    assert _resolve_revision("owner/repo", revision, repo_type="model") == revision.lower()


def test_spatial_pooling_uses_summary_token_and_row_major_cells() -> None:
    embedding = np.asarray([[1.0, 3.0], [5.0, 7.0]], dtype=np.float32)
    coords = np.asarray([[-2.0, -2.0], [2.0, 2.0]], dtype=np.float32)
    tokens, mask, counts = pool_to_spatial_tokens(
        embedding, coords, grid_size=2, tile_size=8.0
    )
    np.testing.assert_array_equal(tokens[0], embedding.mean(axis=0))
    np.testing.assert_array_equal(tokens[1], embedding[0])
    np.testing.assert_array_equal(tokens[4], embedding[1])
    np.testing.assert_array_equal(mask, [True, True, False, False, True])
    np.testing.assert_array_equal(counts, [1, 0, 0, 1])


def test_spatial_pooling_validates_shapes() -> None:
    with pytest.raises(ValueError, match="centered_coords"):
        pool_to_spatial_tokens(
            np.zeros((2, 4), dtype=np.float32),
            np.zeros((1, 2), dtype=np.float32),
            grid_size=8,
            tile_size=7000.0,
        )
