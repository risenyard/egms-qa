#!/usr/bin/env python3
"""Full regression audit for the 304-step to 294-step release migration."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from egms_encoder.data.tile_store import TileStore, TimeWindow
from egms_encoder.extract_tokens import load_encoder, pool_to_vit_tokens
from egms_qa.migrate_tiles_294 import sha256_file, verify_release_pair
from egms_qa.release import audit_release, build_manifest


NUMERIC_TOLERANCE = 1e-12


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _legacy_config(source_config: Path, output: Path) -> Path:
    config = json.loads(source_config.read_text(encoding="utf-8"))
    config["time_window"] = {
        "stored_steps": 304,
        "t_start": 8,
        "t_end": 302,
        "input_length": 294,
        "end_is_exclusive": True,
        "original_source_steps": 304,
        "original_t_start": 8,
        "original_t_end": 302,
        "original_index_offset": 8,
        "original_epoch_year": 2019.0,
        "cadence_days": 6.0,
    }
    _write_json(output, config)
    return output


def _runtime_manifest(release_dir: Path, output: Path) -> pd.DataFrame:
    frame = pd.read_parquet(release_dir / "metadata/split_manifest.parquet").copy()
    tile_root = release_dir / "artifacts/source_tiles"
    frame["path"] = frame["path"].map(
        lambda value: str(tile_root / Path(str(value)).relative_to("data/tiles"))
    )
    frame.to_parquet(output, index=False)
    return frame


def _run_d2(
    checkout: Path,
    manifest: Path,
    data_config: Path,
    b51_table: Path,
    output: Path,
    workers: int,
) -> None:
    environment = os.environ.copy()
    source_path = str(checkout / "src")
    environment["PYTHONPATH"] = (
        source_path
        if not environment.get("PYTHONPATH")
        else source_path + os.pathsep + environment["PYTHONPATH"]
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "egms_qa.qa_construction.tasks.d2.d2_compute",
            "--manifest",
            str(manifest),
            "--data-config",
            str(data_config),
            "--b51-table",
            str(b51_table),
            "--out-dir",
            str(output),
            "--workers",
            str(workers),
        ],
        cwd=checkout,
        env=environment,
        check=True,
    )


def _compare_frames(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    name: str,
    columns: list[str] | None = None,
) -> dict:
    keys = ["tile_id", "split"]
    left = left.sort_values(keys).reset_index(drop=True)
    right = right.sort_values(keys).reset_index(drop=True)
    if len(left) != len(right):
        raise ValueError(f"{name}: row count changed: {len(left)} != {len(right)}")
    if not left[keys].equals(right[keys]):
        raise ValueError(f"{name}: tile order or split assignment changed")
    compare_columns = columns or [column for column in left if column not in keys]
    maximum_error = 0.0
    numeric_columns = []
    categorical_columns = []
    for column in compare_columns:
        if column not in left or column not in right:
            raise ValueError(f"{name}: missing comparison column {column}")
        if pd.api.types.is_numeric_dtype(left[column]) and pd.api.types.is_numeric_dtype(
            right[column]
        ):
            numeric_columns.append(column)
            left_values = left[column].to_numpy(dtype=np.float64)
            right_values = right[column].to_numpy(dtype=np.float64)
            if not np.array_equal(np.isnan(left_values), np.isnan(right_values)):
                raise ValueError(f"{name}: missing-value state changed for {column}")
            finite = np.isfinite(left_values) & np.isfinite(right_values)
            error = (
                float(np.max(np.abs(left_values[finite] - right_values[finite])))
                if finite.any()
                else 0.0
            )
            maximum_error = max(maximum_error, error)
            if error > NUMERIC_TOLERANCE:
                raise ValueError(
                    f"{name}: {column} maximum error {error:.3g} exceeds "
                    f"{NUMERIC_TOLERANCE:.1e}"
                )
        else:
            categorical_columns.append(column)
            sentinel = "<MISSING>"
            if not left[column].fillna(sentinel).astype(str).equals(
                right[column].fillna(sentinel).astype(str)
            ):
                raise ValueError(f"{name}: categorical values changed for {column}")
    return {
        "rows": int(len(left)),
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "maximum_absolute_error": maximum_error,
        "tolerance": NUMERIC_TOLERANCE,
    }


def _tile_store_pair_smoke(
    source_release: Path,
    target_release: Path,
    source_manifest: Path,
    target_manifest: Path,
    legacy_config: Path,
    target_config: Path,
    encoder_checkpoint: Path,
) -> dict:
    old_manifest = pd.read_parquet(source_manifest)
    new_manifest = pd.read_parquet(target_manifest)
    old_config = json.loads(legacy_config.read_text(encoding="utf-8"))
    new_config = json.loads(target_config.read_text(encoding="utf-8"))
    old_store = TileStore(
        old_manifest,
        TimeWindow.from_config(old_config),
        dict(zip(old_manifest["tile_id"].astype(str), old_manifest["split"].astype(str))),
    )
    new_store = TileStore(
        new_manifest,
        TimeWindow.from_config(new_config),
        dict(zip(new_manifest["tile_id"].astype(str), new_manifest["split"].astype(str))),
    )
    sampled_indices = [
        int(old_store.split_tile_indices(split)[0]) for split in ("train", "val", "test")
    ]
    for index in sampled_indices:
        old_input = old_store.get_tile(index)
        new_input = new_store.get_tile(index)
        if not np.array_equal(old_input, new_input, equal_nan=True):
            raise ValueError(f"encoder input differs for manifest index {index}")

    tile = new_store.get_tile(sampled_indices[0])[:32]
    normalization = json.loads(
        (target_release / "metadata/normalization.json").read_text(encoding="utf-8")
    )
    series = tile[:, 10:].copy()
    series = (series - float(normalization["mean"])) / float(normalization["std"])
    series = np.nan_to_num(series, nan=0.0, posinf=0.0, neginf=0.0)
    coords = tile[:, :2].copy()
    coords -= coords.mean(axis=0, keepdims=True)

    device = torch.device("cpu")
    model, train_args = load_encoder(encoder_checkpoint, device)
    point_mask = torch.ones(1, len(tile), dtype=torch.bool)
    with torch.no_grad():
        output = model(
            torch.from_numpy(series).unsqueeze(0),
            coords=torch.from_numpy(coords).unsqueeze(0),
            point_mask=point_mask,
        )
        masked = series.copy()
        masked[:, 80:100] = 0.0
        pretrain_output = model(
            torch.from_numpy(masked).unsqueeze(0),
            coords=torch.from_numpy(coords).unsqueeze(0),
            point_mask=point_mask,
        )
    if output["embedding"].shape != (1, len(tile), 256):
        raise ValueError(f"unexpected encoder embedding shape {output['embedding'].shape}")
    if pretrain_output["reconstruction"].shape != (1, len(tile), 294):
        raise ValueError(
            f"unexpected pretraining reconstruction shape "
            f"{pretrain_output['reconstruction'].shape}"
        )
    tokens, token_mask = pool_to_vit_tokens(
        output["embedding"].squeeze(0).numpy(), coords, grid=8, tile_size=7000.0
    )
    if tokens.shape != (65, 256) or token_mask.shape != (65,):
        raise ValueError("token extraction smoke returned an invalid shape")
    return {
        "sampled_manifest_indices": sampled_indices,
        "encoder_input_bitwise_equal": True,
        "checkpoint_input_length": int(train_args["input_length"]),
        "encoder_embedding_shape": list(output["embedding"].shape),
        "pretraining_reconstruction_shape": list(pretrain_output["reconstruction"].shape),
        "token_shape": list(tokens.shape),
        "token_mask_shape": list(token_mask.shape),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-release", type=Path, required=True)
    parser.add_argument("--target-release", type=Path, required=True)
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--audit-root", type=Path, required=True)
    parser.add_argument("--encoder-checkpoint", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    source_release = args.source_release.resolve()
    target_release = args.target_release.resolve()
    checkout = args.checkout.resolve()
    audit_root = args.audit_root.resolve()
    audit_root.mkdir(parents=True, exist_ok=False)

    pair_report = verify_release_pair(source_release, target_release, args.workers)
    legacy_config = _legacy_config(
        source_release / "metadata/data_config.json", audit_root / "legacy_data_config.json"
    )
    source_manifest_path = audit_root / "source_manifest.parquet"
    target_manifest_path = audit_root / "target_manifest.parquet"
    _runtime_manifest(source_release, source_manifest_path)
    _runtime_manifest(target_release, target_manifest_path)

    b51_table = target_release / "artifacts/reference_tables/b5/b5_final_table.csv"
    old_output = audit_root / "d2-old"
    new_output = audit_root / "d2-new"
    _run_d2(
        checkout,
        source_manifest_path,
        legacy_config,
        b51_table,
        old_output,
        args.workers,
    )
    _run_d2(
        checkout,
        target_manifest_path,
        target_release / "metadata/data_config.json",
        b51_table,
        new_output,
        args.workers,
    )

    old_detail = pd.read_csv(old_output / "d2_final_diagnostics.csv")
    new_detail = pd.read_csv(new_output / "d2_final_diagnostics.csv")
    detail_report = _compare_frames(old_detail, new_detail, name="D2 diagnostics")
    old_final = pd.read_csv(old_output / "d2_final_table.csv")
    new_final = pd.read_csv(new_output / "d2_final_table.csv")
    final_report = _compare_frames(old_final, new_final, name="D21-D24 final table")
    published = pd.read_csv(
        target_release / "artifacts/reference_tables/d2/d2_final_table.csv"
    )
    published_report = _compare_frames(
        published,
        new_final,
        name="published D21-D24 table",
        columns=[column for column in published.columns if column not in {"tile_id", "split"}],
    )

    smoke_report = _tile_store_pair_smoke(
        source_release,
        target_release,
        source_manifest_path,
        target_manifest_path,
        legacy_config,
        target_release / "metadata/data_config.json",
        args.encoder_checkpoint.resolve(),
    )

    environment = os.environ.copy()
    source_path = str(checkout / "src")
    environment["PYTHONPATH"] = (
        source_path
        if not environment.get("PYTHONPATH")
        else source_path + os.pathsep + environment["PYTHONPATH"]
    )
    subprocess.run(
        [sys.executable, "-m", "pytest", "-q"],
        cwd=checkout,
        env=environment,
        check=True,
    )

    invariant_hashes = {
        "normalization.json": sha256_file(target_release / "metadata/normalization.json"),
        "egms_tokens_10k.pt": sha256_file(
            target_release / "artifacts/representations/egms_tokens_10k.pt"
        ),
        "labels.parquet": sha256_file(
            target_release / "artifacts/labels/labels.parquet"
        ),
        "qa_train.jsonl": sha256_file(target_release / "data/qa/train.jsonl"),
        "qa_validation.jsonl": sha256_file(target_release / "data/qa/validation.jsonl"),
        "qa_test.jsonl": sha256_file(target_release / "data/qa/test.jsonl"),
    }
    report = {
        "schema_version": "egms-qa-crop294-regression-1.0",
        "pair_audit": pair_report,
        "d2_diagnostics": detail_report,
        "d2_final": final_report,
        "d2_published": published_report,
        "encoder_smoke": smoke_report,
        "pytest": "passed",
        "invariant_sha256": invariant_hashes,
    }
    report_path = target_release / "metadata/migration_304_to_294_regression.json"
    _write_json(report_path, report)
    _write_json(audit_root / "regression_report.json", report)

    build_manifest(target_release, workers=args.workers)
    audit_release(target_release, verify_hashes=True, workers=args.workers)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
