"""Install only the encoder-facing portion of an EGMS-QA dataset release."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable


RELEASE_SCHEMA = "egms-qa-release-v1"
ENCODER_LINKS = {
    "artifacts/source_tiles": "data/tiles",
    "metadata/split_manifest.parquet": "data/encoder/manifest/split.parquet",
    "metadata/data_config.json": "data/encoder/manifest/data_config.json",
    "artifacts/representations/egms_tokens_10k.pt": (
        "data/encoder/tokens/egms_tokens_10k.pt"
    ),
    "artifacts/representations/egms_tokens_10k_metadata.json": (
        "data/encoder/tokens/egms_tokens_10k_metadata.json"
    ),
}


def _link(source: Path, target: Path) -> None:
    if target.exists() or target.is_symlink():
        if target.is_symlink() and target.resolve() == source.resolve():
            return
        raise FileExistsError(f"installation target already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.symlink_to(source.resolve(), target_is_directory=source.is_dir())


def install_encoder_data(release_dir: str | Path, target_root: str | Path) -> None:
    """Link the canonical tile, manifest, config, and token artifacts."""
    release_dir = Path(release_dir).resolve()
    target_root = Path(target_root).resolve()
    manifest_path = release_dir / "metadata/release_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"dataset release manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != RELEASE_SCHEMA:
        raise ValueError(
            f"unsupported dataset release schema: {manifest.get('schema_version')!r}"
        )
    for source_relative, target_relative in ENCODER_LINKS.items():
        source = release_dir / source_relative
        if not source.exists():
            raise FileNotFoundError(f"encoder data artifact is missing: {source}")
        _link(source, target_root / target_relative)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-dir", type=Path, required=True)
    parser.add_argument("--target-root", type=Path, default=Path("."))
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    install_encoder_data(args.release_dir, args.target_root)


if __name__ == "__main__":
    main()
