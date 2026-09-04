from __future__ import annotations

import json

import torch
from safetensors.torch import load_file, save_file

from egms_encoder.checkpoint import (
    CONFIG_SCHEMA,
    MODEL_TYPE,
    build_encoder,
    export_legacy_checkpoint,
    load_encoder_checkpoint,
    load_encoder_config,
    load_normalization,
)


def model_config() -> dict:
    return {
        "schema_version": CONFIG_SCHEMA,
        "model_type": MODEL_TYPE,
        "architectures": ["TileEncoder"],
        "input_length": 16,
        "patch_size": 8,
        "d_model": 32,
        "temporal_layers": 1,
        "temporal_heads": 4,
        "spatial_layers": 1,
        "spatial_heads": 4,
        "dropout": 0.0,
        "coord_scale_m": 3500.0,
        "residual_head_mode": "additive",
    }


def test_safetensors_checkpoint_round_trip(tmp_path) -> None:
    config = model_config()
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    original = build_encoder(config).eval()
    weights_path = tmp_path / "encoder.safetensors"
    save_file(
        {key: value.detach().contiguous() for key, value in original.state_dict().items()},
        str(weights_path),
    )

    loaded, loaded_config = load_encoder_checkpoint(weights_path, config_path)
    assert loaded_config == config
    for key, value in original.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[key], value, rtol=0, atol=0)

    series = torch.randn(1, 5, 16)
    coords = torch.randn(1, 5, 2)
    mask = torch.ones(1, 5, dtype=torch.bool)
    with torch.no_grad():
        expected = original(series, coords=coords, point_mask=mask)["embedding"]
        actual = loaded(series, coords=coords, point_mask=mask)["embedding"]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_export_legacy_checkpoint_keeps_only_model_tensors(tmp_path) -> None:
    model = build_encoder(model_config())
    legacy_path = tmp_path / "training.pt"
    torch.save({"model": model.state_dict(), "optimizer": {"state": {}}, "step": 10}, legacy_path)
    output_path = tmp_path / "encoder.safetensors"
    export_legacy_checkpoint(legacy_path, output_path)
    exported = load_file(str(output_path))
    assert set(exported) == set(model.state_dict())


def test_config_and_normalization_validation(tmp_path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(model_config()), encoding="utf-8")
    assert load_encoder_config(config_path)["model_type"] == MODEL_TYPE

    normalization_path = tmp_path / "normalization.json"
    normalization_path.write_text(
        json.dumps({"mean": 0.0, "std": 2.0, "residual_std": 0.5}), encoding="utf-8"
    )
    assert load_normalization(normalization_path)["std"] == 2.0
