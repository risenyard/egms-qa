"""Generate EGMS-QA VQA JSONL files from delivered EGMS-QA task labels.

This intentionally does NOT expand all 20 phrasings into the training set.
The 20 variants are a phrasing pool. For each train cycle, each token-dependent
tile-task pair receives one deterministic phrasing; later cycles rotate to a
different phrasing. Static X refusal tasks are sampled, because they are not
token-dependent and should not dominate the loss.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path

import pandas as pd

from egms_qa.qa.qa_lib import (
    ANSWER_PROTOCOL,
    DEFAULT_LABELS,
    DEFAULT_META,
    DEFAULT_QA_AUDIT,
    DEFAULT_TASKS_DIR,
    N_PHRASES,
    QA_SYSTEM_VERSION,
    TARGET_FORMATS,
    TaskRecord,
    load_labels,
    load_qa_audit,
    load_task_records,
    logical_row_count,
    render_row,
    response_format_instruction,
    supervision_target,
)


def stable_seed(*parts: object) -> int:
    h = 2166136261
    for part in parts:
        for b in str(part).encode("utf-8"):
            h ^= b
            h = (h * 16777619) & 0xFFFFFFFF
    return h


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--labels", default=str(DEFAULT_LABELS))
    p.add_argument("--meta", default=str(DEFAULT_META))
    p.add_argument("--tasks-root", default=str(DEFAULT_TASKS_DIR))
    p.add_argument("--qa-audit-manifest", default=str(DEFAULT_QA_AUDIT))
    p.add_argument("--out-dir", default="outputs/qa")
    p.add_argument("--phrases", type=int, default=N_PHRASES)
    p.add_argument("--target-format", choices=TARGET_FORMATS, default="natural")
    p.add_argument("--train-cycles", type=int, default=2,
                   help="Number of train files with rotated phrasing: v1_train_eXX.jsonl.")
    p.add_argument("--x-train-per-task", type=int, default=3000)
    p.add_argument("--x-val-per-task", type=int, default=300)
    p.add_argument("--x-test-per-task", type=int, default=300)
    p.add_argument("--seed", type=int, default=7012)
    p.add_argument("--max-tiles", type=int, default=0, help="Smoke-test cap per split; 0 = all tiles.")
    return p.parse_args()


def select_split_rows(labels: pd.DataFrame, split: str, max_tiles: int) -> pd.DataFrame:
    sub = labels[labels["split"] == split].copy()
    if max_tiles:
        sub = sub.head(max_tiles)
    return sub.reset_index(drop=True)


def x_sample_indices(rows: pd.DataFrame, task: TaskRecord, n: int, seed: int) -> list[int]:
    if n <= 0:
        return []
    idx = list(range(len(rows)))
    random.Random(stable_seed(seed, task.task_id, "x-sample")).shuffle(idx)
    return idx[: min(n, len(idx))]


def phrase_for(tile_id: str, task_id: str, cycle: int, n_phrases: int) -> int:
    return (stable_seed(tile_id, task_id) + cycle) % n_phrases


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def build_split_rows(
    split_rows: pd.DataFrame,
    tasks: list[TaskRecord],
    split: str,
    cycle: int,
    n_phrases: int,
    x_cap: int,
    seed: int,
    phrase_ids_by_task: dict[str, list[int]],
    target_format: str,
) -> list[dict]:
    token_tasks = [t for t in tasks if t.probe_applicable]
    x_tasks = [t for t in tasks if not t.probe_applicable]
    out: list[dict] = []
    for _, row in split_rows.iterrows():
        tile_id = str(row["tile_id"])
        for task in token_tasks:
            slot = phrase_for(tile_id, task.task_id, cycle, n_phrases)
            pidx = phrase_ids_by_task[task.task_id][slot]
            item = render_row(row, task, pidx)
            item["training_target"] = supervision_target(item, target_format)
            item["target_format"] = target_format
            item["response_instruction"] = response_format_instruction(target_format)
            out.append(item)
    for task in x_tasks:
        for idx in x_sample_indices(split_rows, task, x_cap, seed + cycle):
            row = split_rows.iloc[idx]
            tile_id = str(row["tile_id"])
            slot = phrase_for(tile_id, task.task_id, cycle, n_phrases)
            pidx = phrase_ids_by_task[task.task_id][slot]
            item = render_row(row, task, pidx)
            item["training_target"] = supervision_target(item, target_format)
            item["target_format"] = target_format
            item["response_instruction"] = response_format_instruction(target_format)
            out.append(item)
    random.Random(stable_seed(seed, split, cycle, "shuffle")).shuffle(out)
    return out


def write_counts(path: Path, files: dict[str, Counter]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file", "task", "rows"])
        for fname, counts in sorted(files.items()):
            for task, n in sorted(counts.items()):
                w.writerow([fname, task, n])


def main() -> None:
    args = parse_args()
    if args.phrases < 1 or args.phrases > N_PHRASES:
        raise ValueError(f"--phrases must be between 1 and {N_PHRASES}")

    out_dir = Path(args.out_dir)
    qa_dir = out_dir / "qa"
    qa_dir.mkdir(parents=True, exist_ok=True)

    labels = load_labels(Path(args.labels))
    tasks = load_task_records(Path(args.meta), Path(args.tasks_root))
    qa_audit = load_qa_audit(args.qa_audit_manifest, args.phrases)
    phrase_ids_by_task = qa_audit["approved_phrase_ids"]
    token_tasks = [t for t in tasks if t.probe_applicable]
    x_tasks = [t for t in tasks if not t.probe_applicable]
    split_rows = {sp: select_split_rows(labels, sp, args.max_tiles) for sp in ("train", "val", "test")}

    counts_by_file: dict[str, Counter] = {}
    row_counts: dict[str, int] = {}

    for cycle in range(args.train_cycles):
        rows = build_split_rows(
            split_rows["train"],
            tasks,
            "train",
            cycle,
            args.phrases,
            args.x_train_per_task,
            args.seed,
            phrase_ids_by_task,
            args.target_format,
        )
        fname = f"v1_train_e{cycle:02d}.jsonl"
        write_jsonl(qa_dir / fname, rows)
        counts_by_file[fname] = Counter(r["task"] for r in rows)
        row_counts[fname] = len(rows)
        print(f"wrote {qa_dir / fname} rows={len(rows)}", flush=True)

    for split, x_cap in (("val", args.x_val_per_task), ("test", args.x_test_per_task)):
        rows = build_split_rows(
            split_rows[split],
            tasks,
            split,
            0,
            args.phrases,
            x_cap,
            args.seed,
            phrase_ids_by_task,
            args.target_format,
        )
        fname = f"v1_{split}.jsonl"
        write_jsonl(qa_dir / fname, rows)
        counts_by_file[fname] = Counter(r["task"] for r in rows)
        row_counts[fname] = len(rows)
        print(f"wrote {qa_dir / fname} rows={len(rows)}", flush=True)

    write_counts(out_dir / "task_counts.csv", counts_by_file)
    meta = {
        "version": QA_SYSTEM_VERSION,
        "answer_protocol": ANSWER_PROTOCOL,
        "n_tiles": {sp: int(len(df)) for sp, df in split_rows.items()},
        "n_tasks": len(tasks),
        "token_dependent_tasks": len(token_tasks),
        "x_refusal_tasks": len(x_tasks),
        "phrasing_pool_size": args.phrases,
        "target_format": args.target_format,
        "train_cycles": args.train_cycles,
        "x_caps": {
            "train_per_task": args.x_train_per_task,
            "val_per_task": args.x_val_per_task,
            "test_per_task": args.x_test_per_task,
        },
        "row_counts": row_counts,
        "full_expansion_rows_not_used": logical_row_count(len(labels), len(tasks), args.phrases),
        "policy": (
            "Token-dependent tasks are expanded once per tile-task per cycle. "
            "X refusal tasks are sampled per task to avoid over-refusal bias."
        ),
        "qa_audit_manifest": args.qa_audit_manifest,
        "qa_audit_version": qa_audit.get("version"),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    main()
