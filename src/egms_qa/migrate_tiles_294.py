"""Create and verify the model-ready 294-step EGMS-QA tile release.

The migration is deliberately one-way: every source NPZ must contain exactly
304 time steps, and the output stores ``time_series[:, 8:302]``.  Inputs with
294 or any other number of steps are rejected so a rerun cannot silently crop
an already migrated dataset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


SOURCE_STEPS = 304
SOURCE_START = 8
SOURCE_END = 302
STORED_STEPS = SOURCE_END - SOURCE_START
ROLLBACK_REVISION = "b370ae937baf541067a9a84d424d1130e0d48a7a"

CHANGED_RELEASE_PATHS = {
    "README.md",
    "metadata/SOURCE_PROVENANCE.md",
    "metadata/data_config.json",
    "metadata/files.sha256",
    "metadata/release_manifest.json",
    "metadata/tile_manifest.parquet",
    "metadata/migration_304_to_294.json",
    "metadata/migration_304_to_294_audit.parquet",
    "metadata/migration_304_to_294_regression.json",
}


@dataclass(frozen=True)
class TileAudit:
    release_path: str
    source_sha256: str
    target_sha256: str
    source_bytes: int
    target_bytes: int
    n_points: int
    source_steps: int
    stored_steps: int
    keys_unchanged: bool
    non_time_arrays_bitwise_equal: bool
    time_series_matches_source_slice: bool


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def arrays_bitwise_equal(left: np.ndarray, right: np.ndarray) -> bool:
    return (
        left.dtype == right.dtype
        and left.shape == right.shape
        and left.tobytes(order="C") == right.tobytes(order="C")
    )


def _validate_pair(source: Path, target: Path, release_path: str) -> TileAudit:
    with np.load(source, allow_pickle=False) as old, np.load(
        target, allow_pickle=False
    ) as new:
        if "time_series" not in old.files:
            raise ValueError(f"{source}: missing time_series")
        old_ts = old["time_series"]
        if old_ts.ndim != 2 or old_ts.shape[1] != SOURCE_STEPS:
            raise ValueError(
                f"{source}: migration source must be [N,{SOURCE_STEPS}], "
                f"got {old_ts.shape}; refusing repeated or ambiguous cropping"
            )
        new_ts = new["time_series"]
        expected = old_ts[:, SOURCE_START:SOURCE_END]
        keys_equal = old.files == new.files
        non_time_equal = keys_equal and all(
            arrays_bitwise_equal(old[key], new[key])
            for key in old.files
            if key != "time_series"
        )
        time_equal = arrays_bitwise_equal(expected, new_ts)
        if not (keys_equal and non_time_equal and time_equal):
            raise ValueError(
                f"{target}: migration verification failed "
                f"(keys={keys_equal}, non_time={non_time_equal}, time={time_equal})"
            )
        return TileAudit(
            release_path=release_path,
            source_sha256=sha256_file(source),
            target_sha256=sha256_file(target),
            source_bytes=source.stat().st_size,
            target_bytes=target.stat().st_size,
            n_points=int(old_ts.shape[0]),
            source_steps=int(old_ts.shape[1]),
            stored_steps=int(new_ts.shape[1]),
            keys_unchanged=keys_equal,
            non_time_arrays_bitwise_equal=non_time_equal,
            time_series_matches_source_slice=time_equal,
        )


def crop_one(source: Path, target: Path, release_path: str) -> TileAudit:
    """Atomically create one 294-step NPZ, then verify every stored array."""
    with np.load(source, allow_pickle=False) as old:
        if "time_series" not in old.files:
            raise ValueError(f"{source}: missing time_series")
        old_ts = old["time_series"]
        if old_ts.ndim != 2 or old_ts.shape[1] != SOURCE_STEPS:
            raise ValueError(
                f"{source}: migration source must be [N,{SOURCE_STEPS}], "
                f"got {old_ts.shape}; refusing repeated or ambiguous cropping"
            )
        arrays = {
            key: (
                old[key][:, SOURCE_START:SOURCE_END]
                if key == "time_series"
                else old[key]
            )
            for key in old.files
        }

    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        try:
            return _validate_pair(source, target, release_path)
        except (OSError, KeyError, ValueError):
            pass

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return _validate_pair(source, target, release_path)


def _crop_worker(item: tuple[str, str, str]) -> dict:
    source, target, release_path = item
    return asdict(crop_one(Path(source), Path(target), release_path))


def _copy_release_skeleton(source: Path, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    for item in source.iterdir():
        if item.name == ".cache":
            continue
        if item == source / "artifacts":
            (target / "artifacts").mkdir(exist_ok=True)
            for artifact in item.iterdir():
                if artifact.name == "source_tiles":
                    continue
                shutil.copytree(
                    artifact,
                    target / "artifacts" / artifact.name,
                    dirs_exist_ok=True,
                    symlinks=False,
                )
        elif item.is_dir():
            shutil.copytree(item, target / item.name, dirs_exist_ok=True, symlinks=False)
        else:
            shutil.copy2(item, target / item.name, follow_symlinks=True)


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _update_release_metadata(release_dir: Path) -> None:
    config_path = release_dir / "metadata/data_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    normalization_path = release_dir / "metadata/normalization.json"
    normalization_sha256 = sha256_file(normalization_path)
    expected_normalization_sha256 = config["normalization"]["sha256"]
    if normalization_sha256 != expected_normalization_sha256:
        raise ValueError(
            "normalization.json changed before migration: "
            f"{normalization_sha256} != {expected_normalization_sha256}"
        )

    config["schema_version"] = "egms-qa-data-config-1.1"
    config["time_window"] = {
        "stored_steps": STORED_STEPS,
        "t_start": 0,
        "t_end": STORED_STEPS,
        "input_length": STORED_STEPS,
        "end_is_exclusive": True,
        "original_source_steps": SOURCE_STEPS,
        "original_t_start": SOURCE_START,
        "original_t_end": SOURCE_END,
        "original_index_offset": SOURCE_START,
        "original_epoch_year": 2019.0,
        "cadence_days": 6.0,
        "rationale": (
            "The release stores the verified model window directly. Stored "
            "indices [0,294) are bitwise equal to original prepared indices "
            "[8,302); original_index_offset preserves the physical time origin."
        ),
    }
    config["normalization"].update(
        {
            "time_window": "[0,294)",
            "stored_time_window": "[0,294)",
            "original_time_window": "[8,302)",
            "original_index_offset": SOURCE_START,
            "input_steps": STORED_STEPS,
        }
    )
    config["tile_field_layout"].update(
        {
            "time_series_stored_steps": STORED_STEPS,
            "time_series_model_slice": "[0,294)",
            "time_series_model_steps": STORED_STEPS,
            "time_series_original_source_steps": SOURCE_STEPS,
            "time_series_original_slice": "[8,302)",
            "time_series_original_index_offset": SOURCE_START,
        }
    )
    config["tile_field_layout"].pop("time_series_source_steps", None)
    config["migration"] = {
        "operation": "time_series[:,8:302]",
        "stored_axis_rebased": True,
        "previous_huggingface_revision": ROLLBACK_REVISION,
    }
    _write_json_atomic(config_path, config)

    readme_path = release_dir / "README.md"
    readme = readme_path.read_text(encoding="utf-8")
    readme = readme.replace(
        "10,000 processed EGMS 7 km source tiles in the native NPZ contract",
        "10,000 model-ready EGMS 7 km source tiles in the NPZ contract",
    )
    readme = readme.replace(
        "`time_series [N,304]`, prepared displacement values in millimetres;",
        "`time_series [N,294]`, model-ready displacement values in millimetres;",
    )
    readme = readme.replace(
        "The encoder consumes the verified NaN-free slice `[8,302)`, i.e. 294 time\nsteps.",
        "The encoder consumes the stored slice `[0,294)` directly. These values are\nbitwise equal to original prepared indices `[8,302)`; the metadata retains the\noriginal index offset of 8 for physical-time calculations.",
    )
    readme = readme.replace(
        "training split only, over the `[8,302)` window.",
        "training split only, over the stored `[0,294)` window (original `[8,302)`).",
    )
    readme_path.write_text(readme, encoding="utf-8")

    provenance_path = release_dir / "metadata/SOURCE_PROVENANCE.md"
    provenance = provenance_path.read_text(encoding="utf-8")
    provenance = provenance.replace(
        "The NPZ tile arrays contain 304 prepared time indices. The encoder reads the\n"
        "NaN-free interval `[8,302)`, yielding 294 model input steps. The source NPZ\n"
        "files do not retain the Europe-wide union calendar-date labels; the release\n"
        "therefore documents the exact index contract and does not invent dates.",
        "The model-ready NPZ tile arrays store 294 time indices and the encoder reads\n"
        "`[0,294)` directly. These arrays are bitwise equal to indices `[8,302)` of\n"
        "the 304-step prepared source axis. Metadata retains the original offset 8 and\n"
        "six-day cadence; calendar labels absent from the prepared source are not\n"
        "invented.",
    )
    provenance_path.write_text(provenance, encoding="utf-8")

    migration = {
        "schema_version": "egms-qa-tile-migration-1.0",
        "operation": "new.time_series = old.time_series[:, 8:302]",
        "source_steps": SOURCE_STEPS,
        "stored_steps": STORED_STEPS,
        "stored_window": "[0,294)",
        "original_window": "[8,302)",
        "original_index_offset": SOURCE_START,
        "cadence_days": 6,
        "previous_huggingface_revision": ROLLBACK_REVISION,
        "normalization_sha256": normalization_sha256,
    }
    _write_json_atomic(
        release_dir / "metadata/migration_304_to_294.json", migration
    )


def migrate_release(source: Path, target: Path, workers: int) -> pd.DataFrame:
    source = source.resolve()
    target = target.resolve()
    if source == target:
        raise ValueError("source and target release directories must differ")
    if not (source / "artifacts/source_tiles").is_dir():
        raise FileNotFoundError(f"missing source tile store: {source}")
    _copy_release_skeleton(source, target)

    source_root = source / "artifacts/source_tiles"
    source_tiles = sorted(source_root.rglob("*.npz"))
    if len(source_tiles) != 10_000:
        raise ValueError(f"expected 10,000 source NPZ files, found {len(source_tiles):,}")
    work = []
    for source_tile in source_tiles:
        relative = source_tile.relative_to(source)
        work.append((str(source_tile), str(target / relative), relative.as_posix()))

    with ProcessPoolExecutor(max_workers=max(1, workers)) as executor:
        records = list(executor.map(_crop_worker, work, chunksize=4))
    audit = pd.DataFrame.from_records(records).sort_values("release_path")
    audit_path = target / "metadata/migration_304_to_294_audit.parquet"
    audit.to_parquet(audit_path, index=False)
    _update_release_metadata(target)
    print(
        f"migrated and verified {len(audit):,} tiles into {target}; "
        f"audit={audit_path}",
        flush=True,
    )
    return audit


def _iter_invariant_files(release_dir: Path) -> Iterable[Path]:
    for path in release_dir.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(release_dir).as_posix()
        if ".cache" in path.parts:
            continue
        if relative.startswith("artifacts/source_tiles/"):
            continue
        if relative in CHANGED_RELEASE_PATHS:
            continue
        yield path


def verify_release_pair(source: Path, target: Path, workers: int) -> dict:
    source = source.resolve()
    target = target.resolve()
    audit = pd.read_parquet(target / "metadata/migration_304_to_294_audit.parquet")
    if len(audit) != 10_000:
        raise ValueError(f"migration audit contains {len(audit):,} rows, expected 10,000")
    for column in (
        "keys_unchanged",
        "non_time_arrays_bitwise_equal",
        "time_series_matches_source_slice",
    ):
        if not bool(audit[column].all()):
            raise ValueError(f"migration audit has failed rows in {column}")
    if not bool((audit["source_steps"] == SOURCE_STEPS).all()):
        raise ValueError("migration audit contains non-304 sources")
    if not bool((audit["stored_steps"] == STORED_STEPS).all()):
        raise ValueError("migration audit contains non-294 targets")

    source_files = {
        path.relative_to(source).as_posix(): path for path in _iter_invariant_files(source)
    }
    target_files = {
        path.relative_to(target).as_posix(): path for path in _iter_invariant_files(target)
    }
    if source_files.keys() != target_files.keys():
        raise ValueError("non-tile invariant file inventory changed")

    def compare(relative: str) -> str | None:
        return (
            None
            if sha256_file(source_files[relative]) == sha256_file(target_files[relative])
            else relative
        )

    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        mismatches = [
            value for value in executor.map(compare, sorted(source_files)) if value
        ]
    if mismatches:
        raise ValueError(
            f"{len(mismatches)} invariant release files changed; first={mismatches[0]}"
        )

    split = pd.read_parquet(target / "metadata/split_manifest.parquet")
    counts = split["split"].replace({"val": "validation"}).value_counts().to_dict()
    expected_counts = {"train": 8000, "validation": 1000, "test": 1000}
    if counts != expected_counts:
        raise ValueError(f"split counts changed: {counts}")

    config = json.loads((target / "metadata/data_config.json").read_text())
    time_window = config["time_window"]
    expected_contract = {
        "stored_steps": STORED_STEPS,
        "t_start": 0,
        "t_end": STORED_STEPS,
        "original_source_steps": SOURCE_STEPS,
        "original_index_offset": SOURCE_START,
    }
    for key, expected in expected_contract.items():
        if time_window.get(key) != expected:
            raise ValueError(f"data_config time_window.{key} != {expected}")

    report = {
        "tiles": int(len(audit)),
        "split_counts": counts,
        "invariant_files": len(source_files),
        "normalization_sha256": sha256_file(target / "metadata/normalization.json"),
        "time_contract": expected_contract,
    }
    print(json.dumps(report, indent=2), flush=True)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    migrate = subparsers.add_parser("migrate", help="build a separate 294-step release")
    migrate.add_argument("--source", type=Path, required=True)
    migrate.add_argument("--target", type=Path, required=True)
    migrate.add_argument("--workers", type=int, default=8)
    verify = subparsers.add_parser("verify", help="verify migration and invariant files")
    verify.add_argument("--source", type=Path, required=True)
    verify.add_argument("--target", type=Path, required=True)
    verify.add_argument("--workers", type=int, default=8)
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.command == "migrate":
        migrate_release(args.source, args.target, args.workers)
    else:
        verify_release_pair(args.source, args.target, args.workers)


if __name__ == "__main__":
    main()
