from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARGS_PATH = ROOT / "data/encoder/checkpoint/args.json"


def test_encoder_training_config_uses_public_release_names() -> None:
    text = ARGS_PATH.read_text(encoding="utf-8")
    lowered = text.lower()
    for forbidden in ("/home/", "v3_3", "v3.3", "v4_config", "v4_normalization"):
        assert forbidden not in lowered

    config = json.loads(text)
    assert config["schema_version"] == "egms-qa-encoder-training-config-1.0"
    assert config["architecture"]["class"].endswith(".TileEncoder")
    assert config["architecture"]["input_length"] == 294
    assert config["architecture"]["coord_scale_m"] == 3500.0
    assert config["data"]["manifest"] == "data/encoder/manifest/split.parquet"
    assert config["optimization"]["maximum_steps"] == 150_000
