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
    """Read and validate the public model configuration."""
    config_path = Path(path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"encoder config must be a JSON object: {config_path}")
    if config.get("schema_version") != CONFIG_SCHEMA:
        raise ValueError(f"unsupported encoder config schema in {config_path}")
    if config.get("model_type") != MODEL_TYPE:
        raise ValueError(f"unsupported model_type in {config_path}")
    missing = [key for key in ARCHITECTURE_KEYS if key not in config]
    if missing:
        raise ValueError(f"encoder config is missing {missing}: {config_path}")

    positive_ints = (
        "input_length",
        "patch_size",
        "d_model",
        "temporal_layers",
        "temporal_heads",
        "spatial_layers",
        "spatial_heads",
    )
    for key in positive_ints:
        if not isinstance(config[key], int) or isinstance(config[key], bool) or config[key] <= 0:
            raise ValueError(f"{key} must be a positive integer in {config_path}")
    if not 0.0 <= float(config["dropout"]) < 1.0:
        raise ValueError(f"dropout must be in [0,1) in {config_path}")
    if float(config["coord_scale_m"]) <= 0:
        raise ValueError(f"coord_scale_m must be positive in {config_path}")
    if config["residual_head_mode"] not in {"additive", "aux_only"}:
        raise ValueError(f"unsupported residual_head_mode in {config_path}")
    architectures = config.get("architectures")
    if architectures is not None and architectures != ["TileEncoder"]:
        raise ValueError(f"unsupported architectures in {config_path}")
    return config


def build_encoder(config: dict[str, Any]) -> TileEncoder:
    """Construct the encoder architecture described by ``config.json``."""
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
    """Load public Safetensors weights strictly into the configured model."""
    weights_path = Path(weights_path)
    if weights_path.suffix != ".safetensors":
        raise ValueError(f"public encoder weights must use .safetensors: {weights_path}")
    config = load_encoder_config(config_path)
    state = load_file(str(weights_path), device="cpu")
    model = build_encoder(config)
    model.load_state_dict(state, strict=True)
    model.eval().to(device)
    return model, config


def load_normalization(path: str | Path) -> dict[str, Any]:
    """Read the released displacement normalization constants."""
    normalization_path = Path(path)
    values = json.loads(normalization_path.read_text(encoding="utf-8"))
    if not isinstance(values, dict):
        raise ValueError(f"normalization must be a JSON object: {normalization_path}")
    required = {"mean", "std", "residual_std"}
    missing = required - set(values)
    if missing:
        raise ValueError(f"normalization is missing {sorted(missing)}: {normalization_path}")
    if float(values["std"]) <= 0 or float(values["residual_std"]) <= 0:
        raise ValueError(f"normalization scales must be positive: {normalization_path}")
    return values
