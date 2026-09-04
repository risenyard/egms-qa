"""Strict loading utilities for the released EGMS Encoder checkpoint."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from egms_encoder.models.tile_encoder import TileEncoder


def load_encoder_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != "egms-qa-encoder-training-config-1.0":
        raise ValueError(f"unsupported encoder config schema in {config_path}")
    if str(config.get("encoder_version")) != "4.3":
        raise ValueError(f"expected EGMS Encoder 4.3 config in {config_path}")
    return config


def build_encoder(config: dict[str, Any]) -> TileEncoder:
    architecture = config["architecture"]
    return TileEncoder(
        input_length=int(architecture["input_length"]),
        d_model=int(architecture["d_model"]),
        patch_size=int(architecture["patch_size"]),
        temporal_layers=int(architecture["temporal_layers"]),
        temporal_heads=int(architecture["temporal_heads"]),
        spatial_layers=int(architecture["spatial_layers"]),
        spatial_heads=int(architecture["spatial_heads"]),
        dropout=float(architecture["dropout"]),
        residual_head_mode=str(architecture["residual_head_mode"]),
        coord_scale=float(architecture["coord_scale_m"]),
    )


def _validate_checkpoint_config(checkpoint: dict[str, Any], config: dict[str, Any]) -> None:
    embedded = checkpoint.get("args")
    if not isinstance(embedded, dict):
        return
    architecture = config["architecture"]
    expected = {
        "input_length": architecture["input_length"],
        "d_model": architecture["d_model"],
        "patch_size": architecture["patch_size"],
        "temporal_layers": architecture["temporal_layers"],
        "temporal_heads": architecture["temporal_heads"],
        "num_layers": architecture["spatial_layers"],
        "num_heads": architecture["spatial_heads"],
        "dropout": architecture["dropout"],
        "residual_head_mode": architecture["residual_head_mode"],
        "coord_scale": architecture["coord_scale_m"],
    }
    mismatches = {
        key: (embedded.get(key), value)
        for key, value in expected.items()
        if key in embedded and embedded.get(key) != value
    }
    if mismatches:
        raise ValueError(f"encoder config does not match checkpoint metadata: {mismatches}")


def load_encoder_checkpoint(
    checkpoint_path: str | Path,
    config_path: str | Path,
    device: torch.device,
) -> tuple[TileEncoder, dict[str, Any]]:
    checkpoint_path = Path(checkpoint_path)
    config = load_encoder_config(config_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if "model" not in checkpoint:
        raise ValueError(f"checkpoint has no model state: {checkpoint_path}")
    _validate_checkpoint_config(checkpoint, config)
    model = build_encoder(config)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval().to(device)
    return model, config


def load_normalization(path: str | Path) -> dict[str, float | int]:
    normalization_path = Path(path)
    values = json.loads(normalization_path.read_text(encoding="utf-8"))
    for key in ("mean", "std", "residual_std"):
        if key not in values:
            raise ValueError(f"normalization is missing {key}: {normalization_path}")
    if float(values["std"]) <= 0 or float(values["residual_std"]) <= 0:
        raise ValueError(f"normalization scales must be positive: {normalization_path}")
    return values
