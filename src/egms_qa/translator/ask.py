"""Answer one question about one tile using a released translator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from egms_qa.translator.evaluate import greedy_decode_one, load_model


def select_tile(token_cache: Path, tile_id: str | None) -> str:
    """Resolve the requested tile before loading the host model."""
    cache = torch.load(token_cache, map_location="cpu", weights_only=True)
    tile_ids = [str(value) for value in cache["tile_ids"]]
    if not tile_ids:
        raise ValueError("the token cache contains no tiles")
    if tile_id is None:
        return tile_ids[0]
    if tile_id not in tile_ids:
        raise ValueError(f"tile_id {tile_id!r} is absent from the token cache")
    return tile_id


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant-dir", type=Path, required=True,
                        help="Directory containing the translator config, projector, and adapter.")
    parser.add_argument("--token-cache", type=Path, required=True,
                        help="Token cache produced by the released encoder.")
    parser.add_argument("--question", required=True)
    parser.add_argument("--tile-id", help="Tile to use; defaults to the first tile in the cache.")
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--output", type=Path, help="Also save the answer as a JSON file.")
    args = parser.parse_args(argv)
    question = args.question.strip()
    if not question:
        parser.error("--question must not be empty")
    if args.max_new_tokens <= 0:
        parser.error("--max-new-tokens must be positive")
    for path in (args.token_cache, args.variant_dir / "translator_config.json",
                 args.variant_dir / "projector.safetensors",
                 args.variant_dir / "adapter/adapter_config.json",
                 args.variant_dir / "adapter/adapter_model.safetensors"):
        if not path.is_file():
            parser.error(f"required file is missing: {path}")
    try:
        tile_id = select_tile(args.token_cache, args.tile_id)
    except ValueError as exc:
        parser.error(str(exc))
    if not torch.cuda.is_available():
        parser.error("answer generation requires CUDA; use a machine with a supported NVIDIA GPU")

    tokenizer, model, projector, spatial, mask, tile_index, device = load_model(
        adapter_dir=str(args.variant_dir), token_mode="normal", seed=0,
        token_cache=str(args.token_cache),
    )
    row = {"tile_id": tile_id, "question": question, "response_instruction": ""}
    answer = greedy_decode_one(
        row, spatial, mask, tile_index, tokenizer, projector, model, device,
        max_new=args.max_new_tokens,
    )
    result = {"tile_id": tile_id, "question": question, "answer": answer}
    text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
