from __future__ import annotations

import json

import pytest
import torch

from egms_qa.translator import ask


def test_tile_selection_supports_encoder_output_and_rejects_unknown_ids(tmp_path):
    path = tmp_path / "tokens.pt"
    torch.save({"tile_ids": ["tile-a", "tile-b"]}, path)
    assert ask.select_tile(path, None) == "tile-a"
    assert ask.select_tile(path, "tile-b") == "tile-b"
    with pytest.raises(ValueError, match="absent"):
        ask.select_tile(path, "unknown")
    torch.save({"tile_ids": []}, path)
    with pytest.raises(ValueError, match="no tiles"):
        ask.select_tile(path, None)


def test_single_question_uses_shared_decoder_without_reference_answers(tmp_path, monkeypatch):
    variant = tmp_path / "variant"
    for name in ("translator_config.json", "projector.safetensors",
                 "adapter/adapter_config.json", "adapter/adapter_model.safetensors"):
        path = variant / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    cache = tmp_path / "tokens.pt"
    torch.save({"tile_ids": ["tile-a", "tile-b"]}, cache)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(ask, "load_model", lambda **kwargs: (None,) * 7)
    calls = []

    def decode(row, *args, **kwargs):
        calls.append((row, kwargs))
        return "A generated answer."

    monkeypatch.setattr(ask, "greedy_decode_one", decode)
    output = tmp_path / "answers/one.json"
    ask.main(["--variant-dir", str(variant), "--token-cache", str(cache),
              "--tile-id", "tile-b", "--question", "  What is the mean velocity?  ",
              "--max-new-tokens", "32", "--output", str(output)])
    assert calls == [({"tile_id": "tile-b", "question": "What is the mean velocity?",
                       "response_instruction": ""}, {"max_new": 32})]
    assert json.loads(output.read_text()) == {
        "tile_id": "tile-b", "question": "What is the mean velocity?",
        "answer": "A generated answer.",
    }
