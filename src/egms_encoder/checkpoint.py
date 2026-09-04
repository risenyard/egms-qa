"""Load the standalone EGMS-QA Encoder artifact."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file

from egms_encoder.models.tile_encoder import TileEncoder

CONFIG_SCHEMA = "egms-qa-encoder-config-1.0"
MODEL_TYPE = "egms_encoder"
ARCHITECTURE_KEYS = (
    "input_length",
    "patch_size",
    "d_model",
    "temporal_layers",
    "temporal_heads",
    "spatial_layers",
    "spatial_heads",
    "dropout",
    "coord_scale_m",
    "residual_head_mode",
)


def load_encoder_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != CONFIG_SCHEMA:
        raise ValueError(f"unsupported encoder config schema in {config_path}")
    if config.get("model_type") != MODEL_TYPE:
        raise ValueError(f"unsupported model_type in {config_path}")
    missing = [key for key in ARCHITECTURE_KEYS if key not in config]
    if missing:
        raise ValueError(f"encoder config is missing {missing}: {config_path}")
    return config


def build_encoder(config: dict[str, Any]) -> TileEncoder:
    return TileEncoder(
        input_length=int(config["input_length"]),
        d_model=int(config["d_model"]),
        patch_size=int(config["patch_size"]),
        temporal_layers=int(config["temporal_layers"]),
        temporal_heads=int(config["temporal_heads"]),
        spatial_layers=int(config["spatial_layers"]),
        spatial_heads=int(config["spatial_heads"]),
        dropout=float(config["dropout"]),
        residual_head_mode=str(config["residual_head_mode"]),
        coord_scale=float(config["coord_scale_m"]),
    )


def load_encoder_checkpoint(
    weights_path: str | Path,
    config_path: str | Path,
    device: torch.device | str = "cpu",
) -> tuple[TileEncoder, dict[str, Any]]:
    config = load_encoder_config(config_path)
    state = load_file(str(weights_path), device="cpu")
    model = build_encoder(config)
    model.load_state_dict(state, strict=True)
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
