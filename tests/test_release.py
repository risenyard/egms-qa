from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from egms_qa.release import audit_release, build_manifest, install_release


def _write(path: Path, value: bytes = b"x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value)


def test_release_manifest_audit_and_install(tmp_path: Path) -> None:
    release = tmp_path / "release"
    tile_path = release / "artifacts/source_tiles/E00N00/tile_0.npz"
    _write(tile_path, b"tile")
    _write(release / "artifacts/representations/encoder_tokens_10k.pt", b"tokens")
    _write(release / "artifacts/representations/encoder_tokens_10k_metadata.json", b"{}")
    _write(release / "artifacts/labels/labels.parquet", b"labels")
    _write(release / "artifacts/reference_tables/a1/a1_final_table.csv", b"tile_id\n")
    _write(release / "metadata/labels_meta.json", b"{}")
    _write(release / "metadata/qa_audit.json", b"{}")
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
    assert (target / "outputs/qa/v1_val.jsonl").read_bytes() == b"{}\n"
    assert (target / "outputs/tasks/a1/a1_final_table.csv").exists()
