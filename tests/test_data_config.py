from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from egms_encoder.data.tile_store import TileStore


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "data/encoder/manifest/data_config.json"
SPLIT_PATH = ROOT / "data/encoder/manifest/split.parquet"


def test_release_data_config_is_self_contained() -> None:
    text = CONFIG_PATH.read_text(encoding="utf-8")
    lowered = text.lower()
    for forbidden in ("/home/", "archive/", "v3.3", "v4", "lazytile"):
        assert forbidden not in lowered

    config = json.loads(text)
    assert config["schema_version"] == "egms-qa-data-config-1.0"
    assert config["time_window"] == {
        "source_steps": 304,
        "t_start": 8,
        "t_end": 302,
        "input_length": 294,
        "end_is_exclusive": True,
        "rationale": config["time_window"]["rationale"],
    }
    assert config["split"]["counts"] == {
        "train": 8000,
        "validation": 1000,
        "test": 1000,
    }


def test_release_manifest_constructs_tile_store_without_private_files() -> None:
    store = TileStore.from_manifest(SPLIT_PATH, CONFIG_PATH)
    assert store.num_tiles == 10_000
    assert store.time_window.input_length == 294
    assert len(store.split_tile_indices("train")) == 8_000
    assert len(store.split_tile_indices("val")) == 1_000
    assert len(store.split_tile_indices("test")) == 1_000

    manifest = pd.read_parquet(SPLIT_PATH)
    assert manifest["path"].str.startswith("data/tiles/").all()
