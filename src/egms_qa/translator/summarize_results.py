#!/usr/bin/env python3
"""Aggregate the four host-model test evaluations into one EGMS-QA report.

Reads each host model's test summary produced by ``evaluate.py`` and writes a
combined JSON, a per-task CSV, and a short Markdown table with the headline
numbers (token-task score, numeric R2, categorical balanced accuracy, X-family
refusal accuracy, answer-extraction coverage).

Expected layout (override the root with EGMS_QA_OUTPUTS):

    outputs/runs/<model>/generation_eval/test_normal_summary.json

where <model> is one of qwen, gemma, llama, mistral.
"""
from __future__ import annotations

import csv
import json
from statistics import mean

from egms_qa.paths import OUTPUTS_DIR, HOST_MODELS

RUNS_ROOT = OUTPUTS_DIR / "runs"
OUTPUT_ROOT = OUTPUTS_DIR / "report"

# Display name -> run directory / host-model key.
MODELS = {"Llama": "llama", "Mistral": "mistral", "Qwen": "qwen", "Gemma": "gemma"}
BASE_MODELS = {name: HOST_MODELS[key] for name, key in MODELS.items()}


def task_score(metrics: dict) -> float:
    if metrics["metric_type"] == "numeric":
        return max(-1.0, min(1.0, float(metrics["r2"])))
    return float(metrics["balanced_acc"])


def summarize(data: dict) -> dict:
    per_task = data["per_task"]
    token_tasks = {t: m for t, m in per_task.items() if not t.startswith("X")}
    numeric = [m for m in token_tasks.values() if m["metric_type"] == "numeric"]
    categorical = [m for m in token_tasks.values() if m["metric_type"] == "classification"]
    x_tasks = [m for t, m in per_task.items() if t.startswith("X")]
    families = {
        family: mean(
            task_score(m) for t, m in token_tasks.items() if t.startswith(family)
        )
        for family in "ABCDS"
    }
    return {
        "token_score": mean(task_score(m) for m in token_tasks.values()),
        "numeric_mean_r2": mean(float(m["r2"]) for m in numeric),
        "classification_mean_bacc": mean(float(m["balanced_acc"]) for m in categorical),
        "x_mean_bacc": mean(float(m["balanced_acc"]) for m in x_tasks),
        "extraction_coverage": float(data["extraction_coverage"]),
        "family_scores": families,
    }


def load_run(key: str) -> dict:
    path = RUNS_ROOT / key / "generation_eval" / "test_normal_summary.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    data = {}
    for name, key in MODELS.items():
        raw = load_run(key)
        data[name] = {"raw": raw, "summary": summarize(raw)}

    task_ids = next(iter(data.values()))["raw"]["task_ids"]
    if any(set(item["raw"]["task_ids"]) != set(task_ids) for item in data.values()):
        raise ValueError("host models were evaluated on different task sets")

    report = {
        "qa_system": "EGMS-QA",
        "split": "test",
        "models": {
            name: {"base_model": BASE_MODELS[name], **item["summary"]}
            for name, item in data.items()
        },
    }

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    json_path = OUTPUT_ROOT / "four_model_test.json"
    csv_path = OUTPUT_ROOT / "four_model_per_task.csv"
    md_path = OUTPUT_ROOT / "four_model_test.md"
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["task", "metric", *MODELS])
        for task in task_ids:
            metric_type = data["Qwen"]["raw"]["per_task"][task]["metric_type"]
            metric = "R2" if metric_type == "numeric" else "balanced_accuracy"
            writer.writerow([
                task, metric,
                *(task_score(data[name]["raw"]["per_task"][task]) for name in MODELS),
            ])

    lines = [
        "# EGMS-QA four-model test results",
        "",
        "| Model | Base LLM | Token score | Numeric R2 | Categorical bal. acc | X refusal acc | Extraction coverage |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for name, item in data.items():
        s = item["summary"]
        lines.append(
            f"| {name} | {BASE_MODELS[name]} | {s['token_score']:.4f} | "
            f"{s['numeric_mean_r2']:.4f} | {s['classification_mean_bacc']:.4f} | "
            f"{s['x_mean_bacc']:.4f} | {s['extraction_coverage']:.4f} |"
        )
    lines += ["", "## Task-family macro scores", "",
              "| Model | A | B | C | D | S |", "|---|---:|---:|---:|---:|---:|"]
    for name, item in data.items():
        f = item["summary"]["family_scores"]
        lines.append(
            f"| {name} | {f['A']:.4f} | {f['B']:.4f} | {f['C']:.4f} | "
            f"{f['D']:.4f} | {f['S']:.4f} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {json_path}, {csv_path}, and {md_path}")


if __name__ == "__main__":
    main()
