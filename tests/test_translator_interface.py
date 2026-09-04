from __future__ import annotations

import json
from pathlib import Path

from egms_qa.translator.evaluate import resolve_host_model


def test_host_model_is_resolved_from_adapter_config(tmp_path: Path) -> None:
    adapter_dir = tmp_path / "checkpoint"
    config_path = adapter_dir / "lora_adapter/adapter_config.json"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        json.dumps({"base_model_name_or_path": "vendor/host-model"}), encoding="utf-8"
    )
    assert resolve_host_model(adapter_dir, {"args": {}}) == "vendor/host-model"


def test_explicit_host_model_override_has_priority(tmp_path: Path) -> None:
    assert resolve_host_model(tmp_path, {"args": {}}, "local/model") == "local/model"
