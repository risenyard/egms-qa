from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from egms_qa.translator.checkpoint import (
    CONFIG_SCHEMA,
    MODEL_TYPE,
    build_projector,
    load_projector,
    load_translator_config,
)
from egms_qa.translator.evaluate import resolve_host_model


def translator_config() -> dict:
    return {
        "schema_version": CONFIG_SCHEMA,
        "model_type": MODEL_TYPE,
        "variant": "test",
        "base_model": {"name_or_path": "vendor/host-model", "revision": "abc123"},
        "input": {
            "token_count": 65,
            "token_width": 8,
            "mask_length": 65,
            "layout": "test",
        },
        "projector": {
            "architecture": "EGMSProjector",
            "input_dim": 8,
            "hidden_dim": 16,
            "output_dim": 16,
            "activation": "gelu",
            "dropout": 0.05,
            "torch_dtype": "bfloat16",
        },
        "adapter": {"format": "peft", "path": "adapter"},
        "prompt": {
            "question_template": "Question: {question}{response_instruction}\nAnswer:",
            "response_instruction_template": "\nResponse format: {instruction}",
            "answer_protocol": "natural_language",
        },
        "generation": {"decoding": "greedy"},
    }


def test_projector_safetensors_round_trip(tmp_path: Path) -> None:
    config = translator_config()
    config_path = tmp_path / "translator_config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    original = build_projector(config).to(torch.bfloat16).eval()
    weights_path = tmp_path / "projector.safetensors"
    save_file(
        {key: value.detach().contiguous() for key, value in original.state_dict().items()},
        str(weights_path),
    )
    loaded, loaded_config = load_projector(weights_path, config_path)
    assert loaded_config == config
    assert loaded.training is False
    for key, value in original.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[key], value, rtol=0, atol=0)


def test_translator_config_rejects_incompatible_dimensions(tmp_path: Path) -> None:
    config = translator_config()
    config["projector"]["input_dim"] = 7
    path = tmp_path / "translator_config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="input_dim"):
        load_translator_config(path)


def test_host_model_is_resolved_from_release_config(tmp_path: Path) -> None:
    variant_dir = tmp_path / "qwen"
    adapter_config = variant_dir / "adapter/adapter_config.json"
    adapter_config.parent.mkdir(parents=True)
    adapter_config.write_text(
        json.dumps({"base_model_name_or_path": "vendor/host-model"}),
        encoding="utf-8",
    )
    assert resolve_host_model(variant_dir, translator_config()) == "vendor/host-model"


def test_adapter_and_release_base_models_must_match(tmp_path: Path) -> None:
    adapter_config = tmp_path / "adapter/adapter_config.json"
    adapter_config.parent.mkdir(parents=True)
    adapter_config.write_text(
        json.dumps({"base_model_name_or_path": "another/model"}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="does not match"):
        resolve_host_model(tmp_path, translator_config())


def test_explicit_host_model_override_has_priority(tmp_path: Path) -> None:
    assert resolve_host_model(tmp_path, translator_config(), "local/model") == "local/model"
