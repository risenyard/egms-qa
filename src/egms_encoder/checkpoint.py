"""Load and export the standalone EGMS-QA Encoder artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import torch
from safetensors.torch import load_file, save_file

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


def export_legacy_checkpoint(checkpoint_path: str | Path, output_path: str | Path) -> None:
    """Export only model tensors from a trusted legacy training checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state = checkpoint.get("model") if isinstance(checkpoint, dict) else None
    if not isinstance(state, dict) or not state:
        raise ValueError(f"training checkpoint has no model state: {checkpoint_path}")
    if not all(isinstance(value, torch.Tensor) for value in state.values()):
        raise ValueError("model state contains non-tensor values")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    contiguous = {key: value.detach().cpu().contiguous() for key, value in state.items()}
    save_file(contiguous, str(output_path), metadata={"model_type": MODEL_TYPE})


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    export = subparsers.add_parser("export", help="Export weights from a legacy training checkpoint")
    export.add_argument("--checkpoint", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    export_legacy_checkpoint(args.checkpoint, args.output)


if __name__ == "__main__":
    main()
