"""Whole-system execution plans and frozen-reference inference contracts."""
from pathlib import Path
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from egms_qa.qa_construction import run_tasks as runner


def options(tmp_path, mode="reproduce"):
    args = runner.parser().parse_args(["--out-dir", str(tmp_path / "run"), "--mode", mode])
    return args


def test_plan_covers_every_group_and_resolves_rebuilt_dependencies(tmp_path):
    args = options(tmp_path)
    plan = runner.build_plan(args, 20001)
    seen = set()
    for task in plan["tasks"]:
        assert set(task["dependencies"]) <= seen
        seen.add(task["group"])
        arguments = sum(task["commands"], [])
        if task["group"] == "s3":
            assert arguments[arguments.index("--tasks-root") + 1] == str(args.out_dir)
        else:
            for group in task["dependencies"]:
                assert str(runner.table_path(args.out_dir, group)) in arguments
        if task["group"] in {"a1", "a2"}:
            assert len(task["commands"]) == 2
            assert arguments[arguments.index("--sample-tiles") + 1] == "20001"
    assert seen == set(runner.GROUPS)
    assert len(seen) == 27
    c4 = next(t for t in plan["tasks"] if t["group"] == "c4")
    assert c4["commands"][0][3] == "final"


def test_new_tiles_use_generated_tokens_and_frozen_reference(tmp_path):
    args = options(tmp_path, "new-tiles")
    plan = runner.build_plan(args, 1)
    assert plan["extract_tokens"] is not None
    for task in plan["tasks"]:
        flat = sum(task["commands"], [])
        assert ("--reference-state" in flat) == (task["group"] in runner.CALIBRATED)
    args.token_cache = tmp_path / "provided.pt"
    assert runner.build_plan(args, 1)["extract_tokens"] is None


def test_dry_run_does_not_write_output_or_require_gpu(tmp_path):
    manifest = tmp_path / "split.parquet"
    pd.DataFrame({"tile_id": ["new"], "split": ["test"], "path": [str(tmp_path / "not-downloaded.npz")]}).to_parquet(manifest)
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src"))
    output = tmp_path / "out"
    result = subprocess.run([sys.executable, "-m", "egms_qa.qa_construction.run_tasks", "--mode", "new-tiles",
                             "--manifest", str(manifest), "--out-dir", str(output), "--dry-run"],
                            env=env, check=True, capture_output=True, text=True)
    assert len(json.loads(result.stdout)["tasks"]) == 27
    assert not output.exists()


def test_output_root_protection(tmp_path):
    source = tmp_path / "data" / "file"
    source.parent.mkdir()
    source.write_text("input")
    with pytest.raises(ValueError):
        runner.check_output_root(source.parent, [source], False)
    with pytest.raises(FileExistsError):
        runner.check_output_root(tmp_path, [], False)
    link = tmp_path / "linked"
    link.symlink_to(source.parent, target_is_directory=True)
    with pytest.raises(ValueError):
        runner.check_output_root(link, [], False)


def test_failed_command_stops_following_commands(tmp_path):
    marker = tmp_path / "downstream"
    with pytest.raises(subprocess.CalledProcessError):
        runner.execute([[sys.executable, "-c", "raise SystemExit(7)"],
                        [sys.executable, "-c", f"from pathlib import Path; Path({str(marker)!r}).touch()"]], tmp_path / "run.log")
    assert not marker.exists()


@pytest.mark.parametrize("module,function,column", [
    ("c1", "_add_moving_extent_class", "C11_noise_aware_moving_fraction"),
    ("c2", "_add_concentration_class", "C21_spatial_concentration_score"),
    ("c3", "_add_front_strength_class", "C31_deformation_front_strength_mm_yr"),
])
def test_spatial_reference_does_not_fit_target_split(module, function, column):
    import importlib
    code = importlib.import_module(f"egms_qa.qa_construction.tasks.{module}.{module}_compute")
    classify = getattr(code, function)
    reference = pd.DataFrame({"split": ["train"] * 101, column: np.linspace(0, 1, 101)})
    target = pd.DataFrame({"tile_id": ["x"], "split": ["test"], column: [0.6]})
    a, thresholds = classify(target, reference)
    b, other = classify(target.assign(split="train"), reference)
    pd.testing.assert_frame_equal(a.drop(columns="split"), b.drop(columns="split"))
    assert thresholds == other


def test_generic_table_reader_preserves_release_default(tmp_path):
    from egms_qa.qa_construction.tables import read_family
    group = tmp_path / "c1"
    group.mkdir()
    pd.DataFrame({"tile_id": ["new"], "split": ["test"]}).to_csv(group / "c1_final_table.csv", index=False)
    with pytest.raises(ValueError, match="10000"):
        read_family(tmp_path, "c1")
    assert len(read_family(tmp_path, "c1", expected_rows=None)) == 1


def test_token_provenance_and_split_validation(tmp_path):
    import torch
    frame = pd.DataFrame({"tile_id": ["a", "b"], "split": ["test", "test"]})
    expected = {"encoder_weights": "weights", "encoder_config": "config", "normalization": "norm"}
    cache = {"spatial_tokens": torch.zeros(2, 65, 256), "token_mask": torch.ones(2, 65, dtype=torch.bool),
             "point_count_per_bin": torch.ones(2, 64), "tile_ids": ["a", "b"], "splits": ["test", "test"],
             "metadata": {"reproduction_input_sha256": expected.copy()}}
    path = tmp_path / "cache.pt"
    torch.save(cache, path)
    runner.validate_tokens(path, frame, expected)
    cache["tile_ids"].append("b")
    torch.save(cache, path)
    with pytest.raises(ValueError, match="inconsistent lengths"):
        runner.validate_tokens(path, frame, expected)
    cache["tile_ids"].pop()
    cache["metadata"]["reproduction_input_sha256"]["encoder_weights"] = "another-model"
    torch.save(cache, path)
    with pytest.raises(ValueError, match="encoder and normalization"):
        runner.validate_tokens(path, frame, expected)
    cache["tile_ids"].append("b")
    torch.save(cache, path)
    with pytest.raises(ValueError, match="inconsistent lengths"):
        runner.validate_tokens(path, frame, expected)
    cache["tile_ids"].pop()
    cache["metadata"]["reproduction_input_sha256"] = expected
    cache["splits"] = ["train", "test"]
    torch.save(cache, path)
    with pytest.raises(ValueError, match="IDs or splits"):
        runner.validate_tokens(path, frame, expected)


def test_manifest_metadata_is_derived_from_tiles(tmp_path):
    from egms_encoder.data.tile_store import STATIC_KEYS
    from test_temporal_reproduction import config_file
    tile = tmp_path / "tile.npz"
    coords = np.asarray([[100., 200.], [102., 204.]])
    np.savez(tile, coords=coords, time_series=np.zeros((2, 294)), **{k: np.ones(2) for k in STATIC_KEYS})
    frame = pd.DataFrame({"tile_id": ["a"], "split": ["test"], "path": [str(tile)]})
    result = runner.prepare_manifest(frame, config_file(tmp_path), 1)
    assert result['n_points'].tolist() == [2]
    assert result['centroid_x'].tolist() == [101.]
    assert result['centroid_y'].tolist() == [202.]
    assert list(frame.columns) == ['tile_id', 'split', 'path']
