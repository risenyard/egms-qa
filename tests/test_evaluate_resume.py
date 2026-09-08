from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import pandas as pd
import pytest

from egms_qa.translator import evaluate


@pytest.mark.parametrize("same_destination", [True, False])
def test_resumed_outputs_include_old_and_new_answers(tmp_path, monkeypatch, same_destination):
    task = SimpleNamespace(task_id="X11", probe_applicable=False, label_type="categorical")
    rows = [{"task": "X11", "tile_id": f"tile-{i}", "phrase_id": 0,
             "question": "What caused the motion?", "answer": "Cannot determine the cause.",
             "answer_value": "refusal", "answer_type": "refusal"} for i in range(2)]
    prior_record = {**rows[0], "generation": "Cannot determine the cause."}
    prior = tmp_path / "partial.jsonl"
    prior.write_text(json.dumps(prior_record) + "\n")
    destination = prior if same_destination else tmp_path / "complete.jsonl"
    export = tmp_path / "secondary.jsonl"
    monkeypatch.setattr(sys, "argv", ["evaluate", "--adapter-dir", str(tmp_path / "model"),
                        "--resume-dump", str(prior), "--dump", str(destination),
                        "--xverify-export", str(export), "--output", str(tmp_path / "metrics.json")])
    monkeypatch.setattr(evaluate.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(evaluate, "load_task_records", lambda *a: [task])
    monkeypatch.setattr(evaluate, "build_eval_rows", lambda *a: (rows, pd.DataFrame()))
    monkeypatch.setattr(evaluate, "load_model", lambda *a: (None,) * 7)
    generated = []

    def decode(row, *args):
        generated.append(row["tile_id"])
        return "Cannot determine the cause."

    monkeypatch.setattr(evaluate, "greedy_decode_one", decode)
    monkeypatch.setattr(evaluate, "extract_answer", lambda *a: SimpleNamespace(
        value="refusal", status="parsed", method="fixture", evidence=[], reason=""))
    monkeypatch.setattr(evaluate, "extraction_is_correct", lambda *a: True)
    monkeypatch.setattr(evaluate, "summarize_task", lambda *a: {
        "metric_type": "classification", "balanced_acc": 1.0, "acc": 1.0})
    monkeypatch.setattr(evaluate, "summarize", lambda *a: {
        "numeric_r2_summary": {}, "classification_balanced_acc_mean": 1.0,
        "classification_acc_micro": 1.0, "family_numeric_r2_summary": {},
        "family_classification_balanced_acc_mean": {}})
    evaluate.main()
    complete = [json.loads(x) for x in destination.read_text().splitlines()]
    secondary = [json.loads(x) for x in export.read_text().splitlines()]
    assert generated == ["tile-1"]
    assert len(complete) == len(secondary) == 2
    assert complete[0] == prior_record
    assert {x["tile_id"] for x in complete} == {"tile-0", "tile-1"}
    assert all(x == {"question": rows[0]["question"], "correct_answer": rows[0]["answer"],
                     "llm_output": "Cannot determine the cause."} for x in secondary)
