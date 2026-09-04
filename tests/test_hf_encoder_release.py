from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from numpy.lib import format as npy_format
from safetensors.torch import load_file

from egms_encoder.checkpoint import load_encoder_checkpoint, load_normalization
from egms_encoder.data.tile_store import FEATURE_COLUMNS_COUNT, TileStore
from egms_encoder.extract_tokens import pool_to_spatial_tokens
from egms_encoder.install_data import install_encoder_data
from egms_encoder.pretrain import apply_release_config, parse_args


pytestmark = pytest.mark.hf_integration


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


@pytest.fixture(scope="module")
def release_dirs() -> tuple[Path, Path]:
    encoder_value = os.environ.get("EGMS_QA_HF_ENCODER_DIR")
    dataset_value = os.environ.get("EGMS_QA_HF_DATASET_DIR")
    required = os.environ.get("EGMS_QA_REQUIRE_HF_INTEGRATION") == "1"
    if not encoder_value or not dataset_value:
        message = (
            "set EGMS_QA_HF_ENCODER_DIR and EGMS_QA_HF_DATASET_DIR to run "
            "the Hugging Face release integration tests"
        )
        if required:
            pytest.fail(message)
        pytest.skip(message)
    encoder = Path(encoder_value).resolve()
    dataset = Path(dataset_value).resolve()
    if not encoder.is_dir() or not dataset.is_dir():
        pytest.fail(f"release directories do not exist: {encoder}, {dataset}")
    return encoder, dataset


def _load_release_jsons(
    release_dirs: tuple[Path, Path],
) -> tuple[dict, dict, dict, dict, dict]:
    encoder, dataset = release_dirs
    return (
        json.loads((encoder / "config.json").read_text(encoding="utf-8")),
        json.loads((encoder / "training_args.json").read_text(encoding="utf-8")),
        json.loads((encoder / "eval_results.json").read_text(encoding="utf-8")),
        json.loads((dataset / "metadata/data_config.json").read_text(encoding="utf-8")),
        json.loads(
            (
                dataset
                / "artifacts/representations/egms_tokens_10k_metadata.json"
            ).read_text(encoding="utf-8")
        ),
    )


def _tile_path(dataset: Path, manifest_value: object) -> Path:
    relative = Path(str(manifest_value))
    try:
        relative = relative.relative_to("data/tiles")
    except ValueError as exc:
        raise AssertionError(f"unexpected manifest tile path: {manifest_value}") from exc
    return dataset / "artifacts/source_tiles" / relative


def _npz_array_shape(path: Path, array_name: str) -> tuple[int, ...]:
    with zipfile.ZipFile(path) as archive:
        with archive.open(f"{array_name}.npy") as handle:
            version = npy_format.read_magic(handle)
            if version == (1, 0):
                shape, _, _ = npy_format.read_array_header_1_0(handle)
            else:
                shape, _, _ = npy_format.read_array_header_2_0(handle)
    return tuple(int(value) for value in shape)


def test_hf_encoder_file_and_config_contract(
    release_dirs: tuple[Path, Path],
) -> None:
    encoder, dataset = release_dirs
    required = {
        "README.md",
        "config.json",
        "encoder.safetensors",
        "eval_results.json",
        "normalization.json",
        "training_args.json",
    }
    assert required <= {path.name for path in encoder.iterdir() if path.is_file()}
    assert not (encoder / "encoder.pt").exists()
    assert not (encoder / "args.json").exists()

    config, training, evaluation, data_config, _ = _load_release_jsons(release_dirs)
    assert config["schema_version"] == "egms-qa-encoder-config-1.0"
    assert config["model_type"] == "egms_encoder"
    assert config["architectures"] == ["TileEncoder"]
    assert config["input_length"] == 294
    assert config["patch_size"] == 8
    assert config["d_model"] == 256
    assert training["schema_version"] == "egms-qa-encoder-training-1.0"
    assert training["data"]["model_input_steps"] == config["input_length"]
    assert evaluation["schema_version"] == "egms-qa-encoder-evaluation-1.0"
    assert evaluation["sample"]["input_steps"] == config["input_length"]
    assert _sha256(encoder / "normalization.json") == _sha256(
        dataset / "metadata/normalization.json"
    )
    assert data_config["normalization"]["sha256"] == _sha256(
        encoder / "normalization.json"
    )

    args = apply_release_config(parse_args([]), config, training)
    assert args.input_length == 294
    assert args.model_config.endswith("config.json")
    assert args.training_args.endswith("training_args.json")


def test_hf_dataset_stores_only_direct_294_step_tiles(
    release_dirs: tuple[Path, Path],
) -> None:
    _, dataset = release_dirs
    _, _, _, data_config, _ = _load_release_jsons(release_dirs)
    window = data_config["time_window"]
    assert data_config["schema_version"] == "egms-qa-data-config-1.1"
    assert (
        window["stored_steps"],
        window["t_start"],
        window["t_end"],
        window["input_length"],
    ) == (294, 0, 294, 294)
    assert (window["original_t_start"], window["original_t_end"]) == (8, 302)
    assert window["original_index_offset"] == 8

    tiles = sorted((dataset / "artifacts/source_tiles").rglob("*.npz"))
    require_full = os.environ.get("EGMS_QA_REQUIRE_FULL_HF_DATASET") == "1"
    if require_full:
        assert len(tiles) == 10_000
    else:
        assert tiles
    invalid = [path for path in tiles if _npz_array_shape(path, "time_series")[1] != 294]
    assert not invalid, f"non-294 tile found: {invalid[0]}"


def test_hf_token_metadata_matches_current_294_contract(
    release_dirs: tuple[Path, Path],
) -> None:
    encoder, dataset = release_dirs
    _, _, _, _, metadata = _load_release_jsons(release_dirs)
    assert metadata["schema_version"] == "egms-tokens-1.1"
    assert metadata["release_name"] == "EGMS-QA token cache"
    contract = metadata["input_contract"]
    assert (
        contract["stored_steps"],
        contract["stored_window"],
        contract["source_axis_steps"],
        contract["source_window"],
        contract["source_index_offset"],
        contract["token_count"],
        contract["token_width"],
    ) == (294, "[0,294)", 304, "[8,302)", 8, 65, 256)
    assert metadata["statistics"]["tiles"] == 10_000
    assert not (
        dataset / "artifacts/representations/encoder_tokens_10k.pt"
    ).exists()
    assert not (
        dataset / "artifacts/representations/encoder_tokens_10k_metadata.json"
    ).exists()

    provenance = metadata["reproduction_input_sha256"]
    assert provenance["encoder_weights"] == _sha256(
        encoder / "encoder.safetensors"
    )
    assert provenance["encoder_config"] == _sha256(encoder / "config.json")
    assert provenance["normalization"] == _sha256(encoder / "normalization.json")
    assert provenance["split_manifest"] == _sha256(
        dataset / "metadata/split_manifest.parquet"
    )
    assert provenance["data_config"] == _sha256(
        dataset / "metadata/data_config.json"
    )

    if os.environ.get("EGMS_QA_REQUIRE_FULL_HF_DATASET") == "1":
        payload = torch.load(
            dataset / "artifacts/representations/egms_tokens_10k.pt",
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
        assert payload["spatial_tokens"].shape == (10_000, 65, 256)
        assert payload["token_mask"].shape == (10_000, 65)


def test_real_hf_294_tile_runs_through_released_encoder(
    release_dirs: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    if not torch.cuda.is_available():
        pytest.fail("HF encoder integration test requires CUDA")
    encoder, dataset = release_dirs
    manifest = pd.read_parquet(dataset / "metadata/split_manifest.parquet")
    available = manifest[
        manifest["path"].map(lambda value: _tile_path(dataset, value).is_file())
    ]
    assert not available.empty
    row = available.iloc[[0]].copy()
    tile_path = _tile_path(dataset, row.iloc[0]["path"])
    row.loc[:, "path"] = str(tile_path)
    tile_id = str(row.iloc[0]["tile_id"])
    data_config_path = dataset / "metadata/data_config.json"
    store = TileStore.from_manifest(
        _single_row_manifest(row, tmp_path),
        data_config_path,
    )
    materialized = store.get_tile(0)
    with np.load(tile_path, allow_pickle=False) as archive:
        np.testing.assert_array_equal(
            materialized[:, FEATURE_COLUMNS_COUNT:], archive["time_series"]
        )

    sample = materialized[: min(256, len(materialized))]
    normalization = load_normalization(encoder / "normalization.json")
    series = sample[:, FEATURE_COLUMNS_COUNT:].copy()
    series = (series - float(normalization["mean"])) / float(normalization["std"])
    coords = sample[:, :2].copy()
    coords -= coords.mean(axis=0, keepdims=True)
    device = torch.device("cuda:0")
    model, _ = load_encoder_checkpoint(
        encoder / "encoder.safetensors", encoder / "config.json", device
    )
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        output = model(
            torch.from_numpy(series).unsqueeze(0).to(device),
            coords=torch.from_numpy(coords).unsqueeze(0).to(device),
            point_mask=torch.ones(1, len(sample), dtype=torch.bool, device=device),
        )
    assert output["embedding"].shape == (1, len(sample), 256)
    assert output["reconstruction"].shape == (1, len(sample), 294)
    tokens, mask, counts = pool_to_spatial_tokens(
        output["embedding"].squeeze(0).float().cpu().numpy(),
        coords,
        grid_size=8,
        tile_size=7000.0,
    )
    assert tokens.shape == (65, 256)
    assert mask.shape == (65,)
    assert counts.shape == (64,)
    assert tile_id


def _single_row_manifest(row: pd.DataFrame, directory: Path) -> Path:
    output = directory / "single-row.parquet"
    row.to_parquet(output, index=False)
    return output


def _run_cli(command: list[str], cwd: Path) -> None:
    result = subprocess.run(
        command,
        cwd=cwd,
        text=True,
        capture_output=True,
        check=False,
        timeout=180,
    )
    if result.returncode:
        pytest.fail(
            f"command failed ({result.returncode}): {' '.join(command)}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )


def test_train_exports_inference_bundle_consumed_by_token_cli(
    release_dirs: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    if not torch.cuda.is_available():
        pytest.fail("training-to-inference integration test requires CUDA")
    encoder, dataset = release_dirs
    manifest = pd.read_parquet(dataset / "metadata/split_manifest.parquet")
    selected = []
    for split in ("train", "val", "test"):
        row = manifest.loc[manifest["split"] == split].iloc[[0]].copy()
        row.loc[:, "path"] = str(_tile_path(dataset, row.iloc[0]["path"]))
        selected.append(row)
    small_manifest = tmp_path / "split.parquet"
    pd.concat(selected, ignore_index=True).to_parquet(small_manifest, index=False)
    output_dir = tmp_path / "trained"
    data_config = dataset / "metadata/data_config.json"

    _run_cli(
        [
            sys.executable,
            "-m",
            "egms_encoder.pretrain",
            "--model-config",
            str(encoder / "config.json"),
            "--training-args",
            str(encoder / "training_args.json"),
            "--normalization",
            str(encoder / "normalization.json"),
            "--manifest",
            str(small_manifest),
            "--data-config",
            str(data_config),
            "--init-from-checkpoint",
            str(encoder / "encoder.safetensors"),
            "--output-dir",
            str(output_dir),
            "--device",
            "cuda:0",
            "--precision",
            "bf16",
            "--max-tile-points",
            "32",
            "--tiles-per-batch",
            "1",
            "--max-steps",
            "1",
            "--val-batches",
            "1",
            "--val-every-steps",
            "1",
            "--checkpoint-every-steps",
            "1",
            "--log-every-steps",
            "1",
        ],
        tmp_path,
    )

    expected_files = {
        "best.pt",
        "best.safetensors",
        "latest.pt",
        "config.json",
        "training_args.json",
        "normalization.json",
        "run_args.json",
        "metrics.csv",
    }
    assert expected_files <= {path.name for path in output_dir.iterdir()}
    training_checkpoint = torch.load(
        output_dir / "best.pt", map_location="cpu", weights_only=True
    )
    inference_state = load_file(str(output_dir / "best.safetensors"))
    assert set(inference_state) == set(training_checkpoint["model"])
    for key, value in training_checkpoint["model"].items():
        torch.testing.assert_close(inference_state[key], value, rtol=0, atol=0)

    exported_config = json.loads(
        (output_dir / "config.json").read_text(encoding="utf-8")
    )
    exported_recipe = json.loads(
        (output_dir / "training_args.json").read_text(encoding="utf-8")
    )
    run_args = json.loads((output_dir / "run_args.json").read_text(encoding="utf-8"))
    assert exported_config["input_length"] == 294
    assert exported_recipe["schema_version"] == "egms-qa-encoder-training-1.0"
    assert exported_recipe["optimization"]["maximum_steps"] == 1
    assert run_args["max_steps"] == 1

    _run_cli(
        [
            sys.executable,
            "-m",
            "egms_encoder.pretrain",
            "--model-config",
            str(encoder / "config.json"),
            "--training-args",
            str(encoder / "training_args.json"),
            "--normalization",
            str(encoder / "normalization.json"),
            "--manifest",
            str(small_manifest),
            "--data-config",
            str(data_config),
            "--resume-from",
            str(output_dir / "latest.pt"),
            "--output-dir",
            str(output_dir),
            "--device",
            "cuda:0",
            "--precision",
            "bf16",
            "--max-tile-points",
            "32",
            "--tiles-per-batch",
            "1",
            "--max-steps",
            "2",
            "--val-batches",
            "1",
            "--val-every-steps",
            "1",
            "--checkpoint-every-steps",
            "1",
            "--log-every-steps",
            "1",
        ],
        tmp_path,
    )
    resumed = torch.load(
        output_dir / "latest.pt", map_location="cpu", weights_only=True
    )
    assert resumed["step"] == 2
    resumed_recipe = json.loads(
        (output_dir / "training_args.json").read_text(encoding="utf-8")
    )
    assert resumed_recipe["optimization"]["maximum_steps"] == 2

    token_dir = tmp_path / "tokens"
    _run_cli(
        [
            sys.executable,
            "-m",
            "egms_encoder.extract_tokens",
            "--checkpoint",
            str(output_dir / "best.safetensors"),
            "--model-config",
            str(output_dir / "config.json"),
            "--normalization",
            str(output_dir / "normalization.json"),
            "--manifest",
            str(small_manifest),
            "--data-config",
            str(data_config),
            "--device",
            "cuda:0",
            "--max-tiles",
            "1",
            "--output-dir",
            str(token_dir),
        ],
        tmp_path,
    )
    tokens = torch.load(
        token_dir / "egms_tokens_1.pt", map_location="cpu", weights_only=True
    )
    assert tokens["spatial_tokens"].shape == (1, 65, 256)
    assert tokens["token_mask"].shape == (1, 65)
    assert "source_repositories" not in tokens["metadata"]


def test_standalone_encoder_install_and_default_cli(
    release_dirs: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    if not torch.cuda.is_available():
        pytest.fail("standalone encoder integration test requires CUDA")
    encoder, dataset = release_dirs
    runtime = tmp_path / "runtime"
    install_encoder_data(dataset, runtime)
    checkpoint_dir = runtime / "data/encoder/checkpoint"
    checkpoint_dir.mkdir(parents=True)
    for filename in ("encoder.safetensors", "config.json", "normalization.json"):
        (checkpoint_dir / filename).symlink_to((encoder / filename).resolve())

    _run_cli(
        [
            sys.executable,
            "-m",
            "egms_encoder.extract_tokens",
            "--device",
            "cuda:0",
            "--max-tiles",
            "1",
        ],
        runtime,
    )
    output = torch.load(
        runtime / "outputs/tokens/egms_tokens_1.pt",
        map_location="cpu",
        weights_only=True,
    )
    assert output["spatial_tokens"].shape == (1, 65, 256)
    assert output["metadata"]["input_contract"]["stored_window"] == "[0,294)"
