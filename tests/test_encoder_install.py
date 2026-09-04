from __future__ import annotations

import json
from pathlib import Path

import pytest

from egms_encoder.install_data import ENCODER_LINKS, install_encoder_data


def _write(path: Path, value: bytes = b"x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value)


def _release(tmp_path: Path) -> Path:
    release = tmp_path / "release"
    _write(
        release / "metadata/release_manifest.json",
        json.dumps({"schema_version": "egms-qa-release-v1"}).encode(),
    )
    for relative in ENCODER_LINKS:
        path = release / relative
        if Path(relative).suffix:
            _write(path)
        else:
            path.mkdir(parents=True)
            _write(path / "tile.npz")
    return release


def test_install_encoder_data_creates_only_encoder_runtime_links(
    tmp_path: Path,
) -> None:
    release = _release(tmp_path)
    target = tmp_path / "target"
    install_encoder_data(release, target)
    for source_relative, target_relative in ENCODER_LINKS.items():
        installed = target / target_relative
        assert installed.is_symlink()
        assert installed.resolve() == (release / source_relative).resolve()
    assert not (target / "outputs").exists()


def test_install_encoder_data_rejects_existing_target(tmp_path: Path) -> None:
    release = _release(tmp_path)
    target = tmp_path / "target"
    install_encoder_data(release, target)
    manifest = target / "data/encoder/manifest/split.parquet"
    manifest.unlink()
    manifest.write_text("different", encoding="utf-8")
    with pytest.raises(FileExistsError):
        install_encoder_data(release, target)
