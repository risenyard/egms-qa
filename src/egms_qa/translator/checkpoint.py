"""Strict loading utilities for released EGMS-QA translator projectors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import torch
from safetensors.torch import load_file, save_file

from egms_qa.translator.modeling import EGMSProjector

CONFIG_SCHEMA = "egms-qa-translator-config-1.0"
MODEL_TYPE = "egms_qa_translator"
QUESTION_TEMPLATE = "Question: {question}{response_instruction}\nAnswer:"
RESPONSE_INSTRUCTION_TEMPLATE = "\nResponse format: {instruction}"


def load_translator_config(path: str | Path) -> dict[str, Any]:
    """Read and validate one released translator variant configuration."""
    config_path = Path(path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"translator config must be a JSON object: {config_path}")
    if config.get("schema_version") != CONFIG_SCHEMA:
        raise ValueError(f"unsupported translator config schema in {config_path}")
    if config.get("model_type") != MODEL_TYPE:
        raise ValueError(f"unsupported model_type in {config_path}")

    try:
        base_model = config["base_model"]
        name_or_path = str(base_model["name_or_path"]).strip()
        input_config = config["input"]
        token_count = int(input_config["token_count"])
        token_width = int(input_config["token_width"])
        mask_length = int(input_config["mask_length"])
        projector = config["projector"]
        input_dim = int(projector["input_dim"])
        hidden_dim = int(projector["hidden_dim"])
        output_dim = int(projector["output_dim"])
        dropout = float(projector["dropout"])
        prompt = config["prompt"]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"incomplete translator config: {config_path}") from exc

    if not name_or_path:
        raise ValueError(f"base_model.name_or_path is empty in {config_path}")
    if min(token_count, token_width, mask_length, input_dim, hidden_dim, output_dim) <= 0:
        raise ValueError(f"translator dimensions must be positive in {config_path}")
    if mask_length != token_count:
        raise ValueError(f"mask_length must equal token_count in {config_path}")
    if input_dim != token_width:
        raise ValueError(f"projector input_dim must equal token_width in {config_path}")
    if hidden_dim != max(input_dim, output_dim):
        raise ValueError(f"projector hidden_dim is incompatible with EGMSProjector in {config_path}")
    if projector.get("architecture") != "EGMSProjector":
        raise ValueError(f"unsupported projector architecture in {config_path}")
    if str(projector.get("activation", "")).lower() != "gelu":
        raise ValueError(f"unsupported projector activation in {config_path}")
    if not 0.0 <= dropout < 1.0:
        raise ValueError(f"projector dropout must be in [0,1) in {config_path}")
    if prompt.get("question_template") != QUESTION_TEMPLATE:
        raise ValueError(f"unsupported question template in {config_path}")
    if prompt.get("response_instruction_template") != RESPONSE_INSTRUCTION_TEMPLATE:
        raise ValueError(f"unsupported response-instruction template in {config_path}")
    return config


def build_projector(config: dict[str, Any]) -> EGMSProjector:
    """Construct the projector described by ``translator_config.json``."""
    projector = config["projector"]
    return EGMSProjector(
        int(projector["input_dim"]),
        int(projector["output_dim"]),
        float(projector["dropout"]),
    )


def load_projector(
    weights_path: str | Path,
    config_path: str | Path,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[EGMSProjector, dict[str, Any]]:
    """Strictly load pure Safetensors projector weights."""
    weights_path = Path(weights_path)
    if weights_path.suffix != ".safetensors":
        raise ValueError(f"projector weights must use .safetensors: {weights_path}")
    config = load_translator_config(config_path)
    state = load_file(str(weights_path), device="cpu")
    projector = build_projector(config).to(device=device, dtype=dtype)
    projector.load_state_dict(state, strict=True)
    projector.eval()
    return projector, config


def export_legacy_projector(checkpoint_path: str | Path, output_path: str | Path) -> None:
    """Export projector tensors from a trusted legacy training checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state = checkpoint.get("projector_state") if isinstance(checkpoint, dict) else None
    if not isinstance(state, dict) or not state:
        raise ValueError(f"checkpoint has no projector state: {checkpoint_path}")
    if not all(isinstance(value, torch.Tensor) for value in state.values()):
        raise ValueError("projector state contains non-tensor values")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tensors = {key: value.detach().cpu().contiguous() for key, value in state.items()}
    save_file(tensors, str(output_path), metadata={"model_type": "egms_qa_translator_projector"})


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    export = subparsers.add_parser("export", help="Export a legacy projector checkpoint")
    export.add_argument("--checkpoint", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    export_legacy_projector(args.checkpoint, args.output)


if __name__ == "__main__":
    main()
