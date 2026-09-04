from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from egms_encoder.checkpoint import (
    build_encoder,
    load_encoder_checkpoint,
    load_encoder_config,
    load_normalization,
)


def encoder_config() -> dict:
    return {
        "schema_version": "egms-qa-encoder-training-config-1.0",
        "encoder_version": "4.3",
        "architecture": {
            "class": "egms_encoder.models.tile_encoder.TileEncoder",
            "input_length": 24,
            "patch_size": 8,
            "d_model": 32,
            "temporal_layers": 1,
            "temporal_heads": 4,
            "spatial_layers": 1,
            "spatial_heads": 4,
            "dropout": 0.0,
            "coord_scale_m": 3500.0,
            "residual_head_mode": "additive",
        },
    }


def test_checkpoint_load_is_strict_and_safe(tmp_path: Path) -> None:
    config_path = tmp_path / "args.json"
    config_path.write_text(json.dumps(encoder_config()), encoding="utf-8")
    model = build_encoder(encoder_config())
    checkpoint_path = tmp_path / "encoder.pt"
    torch.save({"model": model.state_dict(), "args": {"coord_scale": 3500.0}}, checkpoint_path)

    loaded, config = load_encoder_checkpoint(checkpoint_path, config_path, torch.device("cpu"))
    assert config["encoder_version"] == "4.3"
    assert loaded.training is False

    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state["model"].pop(next(iter(state["model"])))
    torch.save(state, checkpoint_path)
    with pytest.raises(RuntimeError):
        load_encoder_checkpoint(checkpoint_path, config_path, torch.device("cpu"))


def test_checkpoint_metadata_must_match_public_config(tmp_path: Path) -> None:
    config_path = tmp_path / "args.json"
    config_path.write_text(json.dumps(encoder_config()), encoding="utf-8")
    model = build_encoder(encoder_config())
    checkpoint_path = tmp_path / "encoder.pt"
    torch.save(
        {"model": model.state_dict(), "args": {"coord_scale": 1.0}},
        checkpoint_path,
    )
    with pytest.raises(ValueError, match="does not match"):
        load_encoder_checkpoint(checkpoint_path, config_path, torch.device("cpu"))


def test_config_and_normalization_are_validated(tmp_path: Path) -> None:
    config_path = tmp_path / "args.json"
    bad_config = encoder_config()
    bad_config["encoder_version"] = "unknown"
    config_path.write_text(json.dumps(bad_config), encoding="utf-8")
    with pytest.raises(ValueError, match="4.3"):
        load_encoder_config(config_path)

    normalization_path = tmp_path / "normalization.json"
    normalization_path.write_text(
        json.dumps({"mean": 0.0, "std": 0.0, "residual_std": 1.0}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="must be positive"):
        load_normalization(normalization_path)
