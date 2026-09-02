from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, get_args, get_origin, get_type_hints

import yaml


@dataclass
class PathConfig:
    data_path: str = "data/processed/egms_merged_U_no_tail.parquet"
    metadata_path: str = "data/processed/metadata_no_tail.json"
    output_dir: str = "outputs/egms_autoencoder"


@dataclass
class DataConfig:
    component: str = "U"
    crs: str = "EPSG:3035"
    time_series_length: int = 302
    static_columns: list[str] = field(
        default_factory=lambda: [
            "height",
            "rmse",
            "mean_velocity",
            "mean_velocity_std",
            "acceleration",
            "acceleration_std",
            "seasonality",
            "seasonality_std",
        ]
    )


@dataclass
class ExperimentConfig:
    project_name: str = "egms_encoder"
    paths: PathConfig = field(default_factory=PathConfig)
    data: DataConfig = field(default_factory=DataConfig)


def load_config(path: str | Path, overrides: list[str] | None = None) -> ExperimentConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    for override in overrides or []:
        apply_override(raw, override)
    return _coerce_dataclass(ExperimentConfig, raw)


def save_resolved_config(config: ExperimentConfig, path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(as_plain_dict(config), handle, sort_keys=False)


def as_plain_dict(config: Any) -> dict[str, Any]:
    if is_dataclass(config):
        return asdict(config)
    if isinstance(config, dict):
        return {key: as_plain_dict(value) for key, value in config.items()}
    if isinstance(config, list):
        return [as_plain_dict(value) for value in config]
    return config


def apply_override(config: dict[str, Any], override: str) -> None:
    if "=" not in override:
        raise ValueError(f"Override must use key=value syntax: {override}")
    dotted_key, raw_value = override.split("=", 1)
    keys = dotted_key.split(".")
    cursor = config
    for key in keys[:-1]:
        if key not in cursor or cursor[key] is None:
            cursor[key] = {}
        if not isinstance(cursor[key], dict):
            raise ValueError(f"Cannot override nested key under non-dict value: {dotted_key}")
        cursor = cursor[key]
    cursor[keys[-1]] = parse_override_value(raw_value)


def parse_override_value(value: str) -> Any:
    lowered = value.lower()
    if lowered == "null":
        return None
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    try:
        return yaml.safe_load(value)
    except yaml.YAMLError:
        return value


def _coerce_dataclass(cls: type[Any], value: dict[str, Any]) -> Any:
    if not is_dataclass(cls):
        return value
    field_values: dict[str, Any] = {}
    type_hints = get_type_hints(cls)
    for item in fields(cls):
        if item.name not in value:
            continue
        field_values[item.name] = _coerce_value(type_hints.get(item.name, item.type), value[item.name])
    return cls(**field_values)


def _coerce_value(annotation: Any, value: Any) -> Any:
    origin = get_origin(annotation)
    args = get_args(annotation)
    if value is None:
        return None
    if is_dataclass(annotation):
        return _coerce_dataclass(annotation, value)
    if origin is list and isinstance(value, list):
        inner = args[0] if args else Any
        return [_coerce_value(inner, item) for item in value]
    if origin is dict and isinstance(value, dict):
        return value
    return value
