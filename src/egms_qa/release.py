"""Install and audit the structured EGMS-QA Hugging Face data release.

The Hub repository uses a publication-oriented layout.  This module links that
layout into the paths expected by the training and evaluation code without
copying the multi-gigabyte tile store.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path
from typing import Iterable

import pandas as pd

RELEASE_SCHEMA = "egms-qa-release-v1"
RELEASE_MANIFEST = Path("metadata/release_manifest.json")
CHECKSUMS = Path("metadata/files.sha256")


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _release_files(release_dir: Path) -> list[Path]:
    excluded = {RELEASE_MANIFEST.as_posix(), CHECKSUMS.as_posix()}
    files = []
    for path in release_dir.rglob("*"):
        relative = path.relative_to(release_dir).as_posix()
        if ".cache" in path.parts or relative in excluded:
            continue
        if path.is_file() or path.is_symlink():
            files.append(path)
    return sorted(files, key=lambda item: item.relative_to(release_dir).as_posix())


def build_manifest(release_dir: Path, workers: int = 8) -> dict:
    release_dir = release_dir.resolve()

    def inspect(path: Path) -> tuple[str, int, str]:
        relative = path.relative_to(release_dir).as_posix()
        return relative, path.stat().st_size, sha256_file(path)

    files = _release_files(release_dir)
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        records = list(executor.map(inspect, files))
    tile_manifest_relative = "metadata/tile_manifest.parquet"
    # A rerun may see a stale generated tile manifest. It is replaced below and
    # re-hashed after replacement.
    records = [record for record in records if record[0] != tile_manifest_relative]

    metadata_dir = release_dir / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)

    split_path = metadata_dir / "split_manifest.parquet"
    split_summary = None
    if split_path.exists():
        split_frame = pd.read_parquet(split_path)
        split_summary = {
            "tiles": int(len(split_frame)),
            "counts": {
                str(name): int(count)
                for name, count in split_frame["split"].value_counts().to_dict().items()
            },
            "point_rows_with_tile_overlap": int(split_frame["n_points"].sum()),
            "grid_cells": int(split_frame["grid_id"].nunique()),
        }

        file_lookup = {path: (size, digest) for path, size, digest in records}
        tile_manifest = split_frame.copy()
        tile_manifest["release_path"] = tile_manifest["path"].map(
            lambda value: "artifacts/source_tiles/"
            + Path(str(value)).relative_to("data/tiles").as_posix()
        )
        tile_manifest["bytes"] = tile_manifest["release_path"].map(
            lambda value: file_lookup[value][0]
        )
        tile_manifest["sha256"] = tile_manifest["release_path"].map(
            lambda value: file_lookup[value][1]
        )
        tile_manifest.to_parquet(metadata_dir / "tile_manifest.parquet", index=False)

    # The tile manifest is generated from the first pass and must itself be
    # included in the final integrity inventory without re-hashing the full
    # multi-gigabyte tile store.
    tile_manifest_path = metadata_dir / "tile_manifest.parquet"
    if tile_manifest_path.exists():
        records.append(inspect(tile_manifest_path))
    records.sort(key=lambda record: record[0])
    checksum_text = "".join(f"{digest}  {path}\n" for path, _, digest in records)
    (release_dir / CHECKSUMS).write_text(checksum_text, encoding="utf-8")

    component_stats: dict[str, dict[str, int]] = {}
    extension_counts = Counter()
    for relative, size, _ in records:
        top_level = relative.split("/", 1)[0]
        stats = component_stats.setdefault(top_level, {"files": 0, "bytes": 0})
        stats["files"] += 1
        stats["bytes"] += size
        extension_counts[Path(relative).suffix.lower() or "[none]"] += 1

    qa_rows = {}
    for split_name in ("train", "validation", "test"):
        path = release_dir / "data" / "qa" / f"{split_name}.jsonl"
        if path.exists():
            with path.open("rb") as handle:
                qa_rows[split_name] = sum(1 for _ in handle)

    manifest = {
        "schema_version": RELEASE_SCHEMA,
        "release_date": date.today().isoformat(),
        "repository": "risenyard/egms-qa-dataset",
        "code_repository": "https://github.com/risenyard/egms-qa",
        "layout": {
            "data/qa": "Hugging Face-loadable natural-language QA splits",
            "artifacts/source_tiles": "10,000 model-ready 294-step EGMS tiles (NPZ)",
            "artifacts/representations": "frozen encoder token cache",
            "artifacts/labels": "canonical task labels",
            "artifacts/reference_tables": "deterministic per-family task tables",
            "metadata": "current data contract, split, audit, and integrity records",
        },
        "components": component_stats,
        "file_extensions": dict(sorted(extension_counts.items())),
        "qa_rows": qa_rows,
        "tile_split": split_summary,
        "integrity": {
            "algorithm": "sha256",
            "checksums": CHECKSUMS.as_posix(),
            "tile_manifest": "metadata/tile_manifest.parquet",
            "files_hashed": len(records),
            "bytes_hashed": int(sum(size for _, size, _ in records)),
        },
    }
    (release_dir / RELEASE_MANIFEST).write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def read_checksums(release_dir: Path) -> list[tuple[str, str]]:
    records = []
    for line in (release_dir / CHECKSUMS).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, relative = line.split("  ", 1)
        records.append((digest, relative))
    return records


def audit_release(release_dir: Path, verify_hashes: bool = False, workers: int = 8) -> dict:
    release_dir = release_dir.resolve()
    manifest = json.loads((release_dir / RELEASE_MANIFEST).read_text(encoding="utf-8"))
    if manifest.get("schema_version") != RELEASE_SCHEMA:
        raise ValueError(f"unsupported release schema: {manifest.get('schema_version')}")
    checksum_records = read_checksums(release_dir)
    expected_count = int(manifest.get("integrity", {}).get("files_hashed", -1))
    if expected_count != len(checksum_records):
        raise ValueError(
            f"checksum inventory count {len(checksum_records)} != manifest {expected_count}"
        )
    missing = [relative for _, relative in checksum_records if not (release_dir / relative).exists()]
    if missing:
        raise FileNotFoundError(f"release is missing {len(missing)} files; first: {missing[0]}")
    if verify_hashes:
        def verify(record: tuple[str, str]) -> str | None:
            expected, relative = record
            actual = sha256_file(release_dir / relative)
            return None if actual == expected else relative

        with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
            mismatches = [value for value in executor.map(verify, checksum_records) if value]
        if mismatches:
            raise ValueError(f"checksum mismatch for {len(mismatches)} files; first: {mismatches[0]}")
    print(
        f"release audit passed: files={len(checksum_records):,} "
        f"hashes={'verified' if verify_hashes else 'declared'}",
        flush=True,
    )
    return manifest


def _link(source: Path, target: Path) -> None:
    if target.exists() or target.is_symlink():
        if target.is_symlink() and target.resolve() == source.resolve():
            return
        raise FileExistsError(f"installation target already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.symlink_to(source.resolve(), target_is_directory=source.is_dir())


def install_release(release_dir: Path, target_root: Path) -> None:
    release_dir = release_dir.resolve()
    target_root = target_root.resolve()
    audit_release(release_dir, verify_hashes=False)

    directory_links = {
        release_dir / "artifacts/source_tiles": target_root / "data/tiles",
        release_dir / "artifacts/reference_tables": target_root / "outputs/tasks",
    }
    token_file = release_dir / "artifacts/representations/egms_tokens_10k.pt"
    token_metadata = release_dir / "artifacts/representations/egms_tokens_10k_metadata.json"
    file_links = {
        release_dir / "metadata/data_config.json":
            target_root / "data/encoder/manifest/data_config.json",
        release_dir / "metadata/split_manifest.parquet":
            target_root / "data/encoder/manifest/split.parquet",
        token_file:
            target_root / "data/encoder/tokens/egms_tokens_10k.pt",
        token_metadata:
            target_root / "data/encoder/tokens/egms_tokens_10k_metadata.json",
        release_dir / "artifacts/labels/labels.parquet":
            target_root / "outputs/qa/labels.parquet",
        release_dir / "artifacts/labels/metadata.json":
            target_root / "outputs/qa/labels_meta.json",
        release_dir / "metadata/qa_audit.json":
            target_root / "outputs/qa/qa_audit.json",
        release_dir / "data/qa/train.jsonl": target_root / "outputs/qa/v1_train.jsonl",
        release_dir / "data/qa/validation.jsonl": target_root / "outputs/qa/v1_val.jsonl",
        release_dir / "data/qa/test.jsonl": target_root / "outputs/qa/v1_test.jsonl",
    }
    for source, target in {**directory_links, **file_links}.items():
        if not source.exists():
            raise FileNotFoundError(f"release artifact is missing: {source}")
        _link(source, target)
    print(f"installed release links into {target_root}", flush=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    manifest = subparsers.add_parser("manifest", help="Build manifests and SHA256 records")
    manifest.add_argument("--release-dir", type=Path, required=True)
    manifest.add_argument("--workers", type=int, default=8)

    audit = subparsers.add_parser("audit", help="Audit a structured release")
    audit.add_argument("--release-dir", type=Path, required=True)
    audit.add_argument("--verify-hashes", action="store_true")
    audit.add_argument("--workers", type=int, default=8)

    install = subparsers.add_parser("install", help="Link the release into a code checkout")
    install.add_argument("--release-dir", type=Path, required=True)
    install.add_argument("--target-root", type=Path, default=Path("."))
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.command == "manifest":
        manifest = build_manifest(args.release_dir, workers=args.workers)
        print(json.dumps(manifest["integrity"], indent=2), flush=True)
    elif args.command == "audit":
        audit_release(args.release_dir, verify_hashes=args.verify_hashes, workers=args.workers)
    else:
        install_release(args.release_dir, args.target_root)


if __name__ == "__main__":
    main()
