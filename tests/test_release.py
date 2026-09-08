from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from egms_qa.release import audit_release, build_manifest, install_release


def _write(path: Path, value: bytes = b"x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value)


def test_release_manifest_audit_and_install(tmp_path: Path) -> None:
    release = tmp_path / "release"
    tile_path = release / "artifacts/source_tiles/E00N00/tile_0.npz"
    _write(tile_path, b"tile")
    _write(release / "artifacts/representations/egms_tokens_10k.pt", b"tokens")
    _write(release / "artifacts/representations/egms_tokens_10k_metadata.json", b"{}")
    _write(release / "artifacts/labels/labels.parquet", b"labels")
    _write(release / "artifacts/reference_tables/a1/a1_final_table.csv", b"tile_id\n")
    _write(release / "artifacts/labels/metadata.json", b"{}")
    _write(release / "metadata/qa_audit.json", b"{}")
    _write(release / "metadata/data_config.json", b"{}")
    for split in ("train", "validation", "test"):
        _write(release / f"data/qa/{split}.jsonl", b"{}\n")
    pd.DataFrame(
        [{
            "tile_id": "tile_0", "path": "data/tiles/E00N00/tile_0.npz",
            "grid_id": "0_0", "split": "train", "centroid_x": 1.0,
            "centroid_y": 2.0, "n_points": 3,
        }]
    ).to_parquet(release / "metadata/split_manifest.parquet", index=False)

    manifest = build_manifest(release, workers=2)
    assert manifest["qa_rows"] == {"train": 1, "validation": 1, "test": 1}
    assert manifest["tile_split"]["tiles"] == 1
    audit_release(release, verify_hashes=True, workers=2)

    target = tmp_path / "checkout"
    install_release(release, target)
    assert (target / "data/tiles/E00N00/tile_0.npz").read_bytes() == b"tile"
    assert (target / "data/encoder/manifest/data_config.json").read_bytes() == b"{}"
    assert (target / "data/encoder/manifest/split.parquet").is_symlink()
    assert (target / "outputs/qa/v1_val.jsonl").read_bytes() == b"{}\n"
    assert (target / "outputs/tasks/a1/a1_final_table.csv").exists()


def test_manifest_preserves_public_inventory_exclusions(tmp_path: Path) -> None:
    _write(tmp_path / "README.md", b"Dataset")
    _write(tmp_path / ".gitattributes", b"*.npz filter=lfs\n")
    first = build_manifest(tmp_path, workers=1)
    inventory = (tmp_path / "metadata/files.sha256").read_text()
    second = build_manifest(tmp_path, workers=1)
    assert first["integrity"]["files_hashed"] == second["integrity"]["files_hashed"] == 1
    assert (tmp_path / "metadata/files.sha256").read_text() == inventory
    assert inventory.endswith("  README.md\n")


def test_release_installer_rejects_legacy_token_names(tmp_path: Path) -> None:
    release = tmp_path / "release"
    _write(release / "artifacts/source_tiles/E00N00/tile_0.npz", b"tile")
    _write(release / "artifacts/representations/encoder_tokens_10k.pt", b"legacy")
    _write(
        release / "artifacts/representations/encoder_tokens_10k_metadata.json",
        b"{}",
    )
    _write(release / "artifacts/labels/labels.parquet", b"labels")
    _write(release / "artifacts/reference_tables/a1/a1_final_table.csv", b"tile_id\n")
    _write(release / "artifacts/labels/metadata.json", b"{}")
    _write(release / "metadata/qa_audit.json", b"{}")
    _write(release / "metadata/data_config.json", b"{}")
    for split in ("train", "validation", "test"):
        _write(release / f"data/qa/{split}.jsonl", b"{}\n")
    pd.DataFrame(
        [
            {
                "tile_id": "tile_0",
                "path": "data/tiles/E00N00/tile_0.npz",
                "grid_id": "0_0",
                "split": "train",
                "centroid_x": 1.0,
                "centroid_y": 2.0,
                "n_points": 3,
            }
        ]
    ).to_parquet(release / "metadata/split_manifest.parquet", index=False)
    build_manifest(release, workers=1)
    with pytest.raises(FileNotFoundError, match="egms_tokens_10k"):
        install_release(release, tmp_path / "target")


GROUP_FILES = {
    'qa': [
        'data/qa/train.jsonl', 'data/qa/validation.jsonl', 'data/qa/test.jsonl',
        'artifacts/labels/labels.parquet', 'artifacts/labels/metadata.json',
        'artifacts/reference_tables/a1/a1_final_table.csv',
    ],
    'tokens': [
        'artifacts/representations/egms_tokens_10k.pt',
        'artifacts/representations/egms_tokens_10k_metadata.json',
    ],
    'tiles': ['artifacts/source_tiles/E00N00/tile_0.npz'],
}


@pytest.fixture
def component_release(tmp_path):
    release = tmp_path / 'complete'
    for files in GROUP_FILES.values():
        for name in files:
            _write(release / name, b'{}\n')
    _write(release / 'README.md', b'Dataset')
    _write(release / 'metadata/data_config.json', b'{}')
    _write(release / 'metadata/qa_audit.json', b'{}')
    pd.DataFrame([{
        'tile_id': 'tile_0', 'path': 'data/tiles/E00N00/tile_0.npz',
        'grid_id': '0_0', 'split': 'train', 'n_points': 3,
    }]).to_parquet(release / 'metadata/split_manifest.parquet', index=False)
    build_manifest(release, workers=1)
    return release


def copy_group(release, partial, group):
    import shutil
    for path in [*release.glob('*.md'), *release.glob('metadata/*'),
                 *(release / name for name in GROUP_FILES[group])]:
        target = partial / path.relative_to(release)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)


@pytest.mark.parametrize('group', ['qa', 'tokens', 'tiles'])
def test_each_group_installs_without_other_downloads(component_release, tmp_path, group):
    partial = tmp_path / 'partial'
    copy_group(component_release, partial, group)
    target = tmp_path / 'runtime'
    install_release(partial, target, components=[group])
    assert (target / 'outputs/qa/labels.parquet').exists() == (group == 'qa')
    assert (target / 'outputs/tasks/a1/a1_final_table.csv').exists() == (group == 'qa')
    assert (target / 'data/encoder/tokens/egms_tokens_10k.pt').exists() == (group == 'tokens')
    assert (target / 'data/tiles/E00N00/tile_0.npz').exists() == (group == 'tiles')
    assert (target / 'data/encoder/manifest/split.parquet').exists()
    # A partial download must never pass the full-release audit.
    with pytest.raises(FileNotFoundError):
        audit_release(partial)


@pytest.mark.parametrize('group', ['qa', 'tokens', 'tiles'])
def test_installation_checks_integrity_automatically(component_release, tmp_path, group):
    partial = tmp_path / 'partial'
    copy_group(component_release, partial, group)
    (partial / GROUP_FILES[group][0]).write_bytes(b'corrupt')
    target = tmp_path / 'runtime'
    with pytest.raises(ValueError, match='checksum mismatch'):
        install_release(partial, target, components=[group])
    assert not target.exists()


def test_groups_can_be_added_and_installed_again(component_release, tmp_path):
    partial = tmp_path / 'partial'
    target = tmp_path / 'runtime'
    for group in ['qa', 'tokens', 'tiles']:
        copy_group(component_release, partial, group)
        install_release(partial, target, components=[group])
    install_release(partial, target, components=['qa', 'tokens', 'tiles'])
    assert (target / 'data/tiles/E00N00/tile_0.npz').exists()
    assert (target / 'outputs/qa/v1_test.jsonl').exists()


def test_destination_conflict_does_not_create_partial_install(component_release, tmp_path):
    target = tmp_path / 'runtime'
    occupied = target / 'outputs/qa/v1_test.jsonl'
    _write(occupied, b'my data')
    with pytest.raises(FileExistsError):
        install_release(component_release, target, components=['qa'])
    assert occupied.read_bytes() == b'my data'
    assert not (target / 'outputs/tasks').exists()
    assert not (target / 'data').exists()


@pytest.mark.parametrize('groups', [['qa'], ['tokens'], ['tiles'], ['qa', 'tokens']])
def test_download_fetches_only_selected_groups(monkeypatch, tmp_path, groups):
    import fnmatch
    import huggingface_hub
    from egms_qa.release import download_release
    captured = {}
    def snapshot(**kwargs):
        captured.update(kwargs)
        return str(tmp_path)
    monkeypatch.setattr(huggingface_hub, 'snapshot_download', snapshot)
    assert download_release(groups, revision='fixed-revision') == tmp_path
    assert captured['revision'] == 'fixed-revision'
    assert captured['repo_id'] == 'risenyard/egms-qa-dataset'
    for group, files in GROUP_FILES.items():
        for name in files:
            included = any(fnmatch.fnmatch(name, pattern) for pattern in captured['allow_patterns'])
            assert included == (group in groups), name
    assert any(fnmatch.fnmatch('metadata/files.sha256', p) for p in captured['allow_patterns'])
