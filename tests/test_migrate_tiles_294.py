from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from egms_qa.migrate_tiles_294 import crop_one
from egms_qa.qa_construction.tasks.d2.d2_compute import (
    D2TimeAxis,
    _configure_time_axis,
    _read_one,
)


def _write_tile(path: Path, steps: int) -> dict[str, np.ndarray]:
    arrays = {
        "coords": np.arange(8, dtype=np.float32).reshape(4, 2),
        "time_series": np.arange(4 * steps, dtype=np.float32).reshape(4, steps),
        "bbox": np.array([1, 2, 3, 4], dtype=np.float32),
        "pid": np.array(["p3", "p1", "p4", "p2"]),
        "height": np.array([np.nan, 2, 3, 4], dtype=np.float32),
        "rmse": np.arange(4, dtype=np.float32),
        "mean_velocity": np.arange(4, dtype=np.float32) + 10,
        "mean_velocity_std": np.arange(4, dtype=np.float32) + 20,
        "acceleration": np.arange(4, dtype=np.float32) + 30,
        "acceleration_std": np.arange(4, dtype=np.float32) + 40,
        "seasonality": np.arange(4, dtype=np.float32) + 50,
        "seasonality_std": np.arange(4, dtype=np.float32) + 60,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    return arrays


def test_crop_is_exact_and_preserves_non_time_arrays(tmp_path: Path) -> None:
    source = tmp_path / "old.npz"
    target = tmp_path / "new.npz"
    arrays = _write_tile(source, 304)
    audit = crop_one(source, target, "artifacts/source_tiles/old.npz")

    assert audit.source_steps == 304
    assert audit.stored_steps == 294
    assert audit.keys_unchanged
    assert audit.non_time_arrays_bitwise_equal
    assert audit.time_series_matches_source_slice
    with np.load(target, allow_pickle=False) as migrated:
        assert migrated.files == list(arrays)
        assert np.array_equal(migrated["time_series"], arrays["time_series"][:, 8:302])
        for key, value in arrays.items():
            if key != "time_series":
                assert migrated[key].dtype == value.dtype
                assert migrated[key].shape == value.shape
                assert migrated[key].tobytes() == value.tobytes()


@pytest.mark.parametrize("steps", [294, 286, 305])
def test_crop_rejects_non_304_source(tmp_path: Path, steps: int) -> None:
    source = tmp_path / f"source-{steps}.npz"
    _write_tile(source, steps)
    with pytest.raises(ValueError, match="refusing repeated or ambiguous cropping"):
        crop_one(source, tmp_path / "target.npz", "tile.npz")


def test_d2_requires_original_offset(tmp_path: Path) -> None:
    config = {
        "time_window": {
            "stored_steps": 294,
            "t_start": 0,
            "t_end": 294,
            "input_length": 294,
            "original_source_steps": 304,
            "original_epoch_year": 2019,
            "cadence_days": 6,
        }
    }
    path = tmp_path / "data_config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="original_index_offset"):
        D2TimeAxis.from_file(path)


def test_d2_preserves_original_physical_time_origin(tmp_path: Path) -> None:
    config = {
        "time_window": {
            "stored_steps": 294,
            "t_start": 0,
            "t_end": 294,
            "input_length": 294,
            "original_source_steps": 304,
            "original_index_offset": 8,
            "original_epoch_year": 2019,
            "cadence_days": 6,
        }
    }
    path = tmp_path / "data_config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    axis = D2TimeAxis.from_file(path)
    assert axis.start_year == 2019 + 8 * 6 / 365.25


def test_d2_values_match_before_and_after_rebasing(tmp_path: Path) -> None:
    source = tmp_path / "old.npz"
    target = tmp_path / "new.npz"
    _write_tile(source, 304)
    crop_one(source, target, "tile.npz")
    old_axis = D2TimeAxis(304, 8, 302, 304, 8, 2019.0, 6.0)
    new_axis = D2TimeAxis(294, 0, 294, 304, 8, 2019.0, 6.0)
    _configure_time_axis(old_axis)
    old = _read_one(("tile", "train", str(source)))
    _configure_time_axis(new_axis)
    new = _read_one(("tile", "train", str(target)))
    assert old.keys() == new.keys()
    for key in old:
        if isinstance(old[key], float):
            assert old[key] == pytest.approx(new[key], abs=1e-12, nan_ok=True)
        else:
            assert old[key] == new[key]
