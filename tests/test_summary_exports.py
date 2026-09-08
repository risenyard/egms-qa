from __future__ import annotations

import csv
import json
import sys

from egms_qa.translator import summarize_results


def test_per_task_csv_preserves_raw_r2_below_minus_one(tmp_path, monkeypatch):
    per_task = {"A11": {"metric_type": "numeric", "r2": -2.0}}
    per_task.update({t: {"metric_type": "classification", "balanced_acc": 0.5}
                     for t in ["B12", "C12", "D11", "S11", "X11"]})
    root = tmp_path / "evaluation"
    for variant in ["qwen", "gemma", "llama", "mistral"]:
        directory = root / variant
        directory.mkdir(parents=True)
        (directory / "metrics.json").write_text(json.dumps({
            "task_ids": list(per_task), "per_task": per_task, "extraction_coverage": 1.0,
        }))
    output = tmp_path / "report"
    monkeypatch.setattr(summarize_results, "OUTPUT_ROOT", output)
    monkeypatch.setattr(sys, "argv", ["summarize", "--evaluation-root", str(root)])
    summarize_results.main()
    with (output / "four_model_per_task.csv").open() as handle:
        row = next(r for r in csv.DictReader(handle) if r["task"] == "A11")
    assert row["metric"] == "R2"
    assert float(row["Qwen"]) == -2.0
    # Bounded composite scores retain their existing definition.
    report = json.loads((output / "four_model_test.json").read_text())
    assert report["models"]["Qwen"]["family_scores"]["A"] == -1.0
