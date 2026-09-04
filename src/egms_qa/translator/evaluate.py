"""Free-generation evaluation for EGMS-QA sampled VQA checkpoints.

This complements the trainer's teacher-forced held-out loss. It greedily decodes
answers on dynamically sampled test rows, scores generated answers against the
canonical EGMS-QA task labels, and can run a shuffled-token control.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import torch

from egms_qa.translator.modeling import EGMSProjector
from egms_qa.translator.generation import build_prompt_q
from egms_qa.translator.answer_extractor import (
    AMBIGUOUS,
    PARSED,
    UNPARSED,
    extract_answer,
    extraction_is_correct,
)
from egms_qa.qa_construction.qa_lib import (
    ANSWER_PROTOCOL,
    DEFAULT_LABELS,
    DEFAULT_META,
    DEFAULT_QA_AUDIT,
    DEFAULT_TASKS_DIR,
    MISSING_VALUE,
    N_PHRASES,
    QA_SYSTEM_VERSION,
    TaskRecord,
    format_number,
    load_labels,
    load_qa_audit,
    load_task_config,
    load_task_records,
    render_row,
)


from egms_qa.paths import ENCODER_TOKENS, DEFAULT_HOST_MODEL

TOK_DEFAULT = str(ENCODER_TOKENS)
QWEN_DEFAULT = DEFAULT_HOST_MODEL


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--adapter-dir", required=True, help="Checkpoint dir containing projector.pt and lora_adapter/.")
    p.add_argument("--token-cache", default=TOK_DEFAULT)
    p.add_argument("--labels", default=str(DEFAULT_LABELS))
    p.add_argument("--meta", default=str(DEFAULT_META))
    p.add_argument("--tasks-root", default=str(DEFAULT_TASKS_DIR))
    p.add_argument("--qa-audit-manifest", default=str(DEFAULT_QA_AUDIT))
    p.add_argument("--task-config", default="",
                   help="JSON manifest with maintain and focus task lists.")
    p.add_argument("--split", default="test", choices=["val", "test"])
    p.add_argument(
        "--token-mode",
        default="normal",
        choices=["normal", "shuffled", "summary_only"],
        help=(
            "Visual-prefix control: matched full tokens, mismatched full tokens, "
            "or the matched summary token with all 64 cell tokens withheld."
        ),
    )
    p.add_argument("--cap-per-task", type=int, default=40)
    p.add_argument("--x-cap-per-task", type=int, default=20)
    p.add_argument("--phrases", type=int, default=N_PHRASES)
    p.add_argument("--seed", type=int, default=17012)
    p.add_argument("--max-new", type=int, default=96)
    p.add_argument("--output", default="")
    p.add_argument("--dump", default="")
    p.add_argument(
        "--resume-dump",
        default="",
        help="Optional compatible partial JSONL dump. Existing task/tile/phrase records are retained and skipped.",
    )
    p.add_argument(
        "--xverify-export",
        default="",
        help="Optional JSONL export with question/correct_answer/llm_output for secondary xVerify auditing.",
    )
    return p.parse_args()


def stable_seed(*parts: object) -> int:
    h = 2166136261
    for part in parts:
        for b in str(part).encode("utf-8"):
            h ^= b
            h = (h * 16777619) & 0xFFFFFFFF
    return h


def numeric_target(row: dict[str, Any], task: TaskRecord) -> float:
    """Numeric target as rendered in the supervised answer text."""
    try:
        return float(format_number(row.get("answer_value"), task))
    except Exception:
        return float(row.get("answer_value"))


def score_row(row: dict[str, Any], gen: str, task: TaskRecord, task_labels: list[Any]) -> tuple[bool, Any | None]:
    result = extract_answer(gen, task, task_labels)
    return extraction_is_correct(row, task, result), result.value


def balanced_rows(
    label_rows,
    task: TaskRecord,
    split: str,
    seed: int,
    cap: int,
    phrase_ids: list[int],
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    idxs = list(range(len(label_rows)))
    rng.shuffle(idxs)
    if task.probe_applicable and task.label_type != "numeric":
        buckets: dict[str, list[int]] = collections.defaultdict(list)
        for i in idxs:
            value = label_rows.iloc[i].get(task.task_id)
            buckets[str(value) if not _is_missing_for_eval(value) else MISSING_VALUE].append(i)
        out_idx: list[int] = []
        keys = list(buckets)
        while keys and len(out_idx) < cap:
            next_keys = []
            for key in keys:
                if buckets[key] and len(out_idx) < cap:
                    out_idx.append(buckets[key].pop())
                if buckets[key]:
                    next_keys.append(key)
            keys = next_keys
    else:
        out_idx = idxs[:cap]
    out = []
    for i in out_idx:
        pidx = rng.choice(phrase_ids)
        r = render_row(label_rows.iloc[i], task, pidx)
        r["split"] = split
        out.append(r)
    return out


def _is_missing_for_eval(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(math.isnan(float(value))) if isinstance(value, float) else False
    except Exception:
        return str(value) in ("", "nan", "None", "<NA>", "NaN")


def eval_record_key(record: dict[str, Any]) -> tuple[str, str, int]:
    return str(record["task"]), str(record["tile_id"]), int(record["phrase_id"])


def load_resume_records(path: str) -> dict[tuple[str, str, int], dict[str, Any]]:
    if not path:
        return {}
    resume_path = Path(path)
    if not resume_path.is_file():
        raise FileNotFoundError(f"resume dump does not exist: {resume_path}")
    records: dict[tuple[str, str, int], dict[str, Any]] = {}
    for lineno, line in enumerate(resume_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        record = json.loads(line)
        key = eval_record_key(record)
        if key in records:
            raise ValueError(f"duplicate resume record at {resume_path}:{lineno}: {key}")
        records[key] = record
    return records


def build_eval_rows(args: argparse.Namespace, tasks: list[TaskRecord]):
    labels = load_labels(Path(args.labels))
    qa_audit = load_qa_audit(args.qa_audit_manifest, args.phrases)
    approved = qa_audit["approved_phrase_ids"]
    split_rows = labels[labels["split"] == args.split].reset_index(drop=True)
    all_rows: list[dict[str, Any]] = []
    for t in tasks:
        cap = args.cap_per_task if t.probe_applicable else args.x_cap_per_task
        task_rows = balanced_rows(
            split_rows,
            t,
            args.split,
            stable_seed(args.seed, t.task_id, args.split),
            cap,
            approved[t.task_id],
        )
        for row in task_rows:
            row["response_instruction"] = ""
        all_rows.extend(task_rows)
    random.Random(args.seed).shuffle(all_rows)
    return all_rows, labels


def load_model(adapter_dir: str, token_mode: str, seed: int, token_cache: str):
    import transformers
    from peft import PeftModel
    from transformers import AutoTokenizer

    device = torch.device("cuda:0")
    ck = torch.load(Path(adapter_dir) / "projector.pt", map_location="cpu", weights_only=False)
    base = ck.get("args", {}).get("qwen_path", QWEN_DEFAULT)
    tokenizer = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_type = ""
    try:
        model_type = json.load(open(Path(base) / "config.json")).get("model_type", "").lower()
    except Exception:
        pass
    if "gemma" in model_type:
        order = ["Gemma3ForConditionalGeneration", "AutoModelForImageTextToText", "AutoModelForCausalLM"]
    elif "qwen" in model_type:
        order = ["AutoModelForImageTextToText", "AutoModelForCausalLM"]
    else:
        order = ["AutoModelForCausalLM", "AutoModelForImageTextToText"]
    kw = dict(trust_remote_code=True, torch_dtype=torch.bfloat16, device_map={"": 0})
    model = None
    used = None
    last = None
    for cls_name in order:
        if not hasattr(transformers, cls_name):
            continue
        try:
            model = getattr(transformers, cls_name).from_pretrained(base, **kw)
            used = cls_name
            break
        except Exception as exc:
            last = exc
    if model is None:
        raise RuntimeError(f"could not load base={base} model_type={model_type}: {last}")
    print(f"[eval] loaded base model_type={model_type or '?'} via {used}", flush=True)

    model = PeftModel.from_pretrained(model, str(Path(adapter_dir) / "lora_adapter"))
    model.eval()
    model.config.use_cache = True
    model._needs_tti = any("vision" in n for n, _ in model.named_modules())

    projector = EGMSProjector(ck["egms_dim"], ck["llm_hidden"]).to(device, torch.bfloat16)
    projector.load_state_dict(ck["projector_state"])
    projector.eval()

    cache = torch.load(token_cache, map_location="cpu", weights_only=False)
    spatial = cache["spatial_tokens"].float()
    token_mask = cache["token_mask"]
    if ck.get("args", {}).get("cls_only"):
        spatial = spatial[:, :1]
        token_mask = token_mask[:, :1]
    if token_mode == "summary_only":
        spatial = spatial[:, :1]
        token_mask = token_mask[:, :1]
        print(
            "[eval] summary-only inference control: withholding all 64 cell tokens",
            flush=True,
        )
    if token_mode == "shuffled":
        perm = list(range(spatial.shape[0]))
        random.Random(seed + 99).shuffle(perm)
        for i in range(len(perm)):
            if perm[i] == i:
                j = (i + 1) % len(perm)
                perm[i], perm[j] = perm[j], perm[i]
        spatial = spatial[perm]
        token_mask = token_mask[perm]
    tid2idx = {str(t): i for i, t in enumerate(cache["tile_ids"])}
    return tokenizer, model, projector, spatial, token_mask, tid2idx, device


@torch.no_grad()
def greedy_decode_one(row, spatial, tok_mask, tid2idx, tokenizer, projector, model, device, max_new: int) -> str:
    idx = tid2idx[str(row["tile_id"])]
    prefix = spatial[idx:idx + 1].to(device)
    prefix_mask = tok_mask[idx:idx + 1].to(device).long()
    prompt_ids = tokenizer(build_prompt_q(row), add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)
    prefix_proj = projector(prefix.to(torch.bfloat16))
    prompt_emb = model.get_input_embeddings()(prompt_ids).to(torch.bfloat16)
    inputs_embeds = torch.cat([prefix_proj, prompt_emb], dim=1)
    attn = torch.cat([prefix_mask, torch.ones_like(prompt_ids)], dim=1)
    fkw = {}
    if getattr(model, "_needs_tti", False):
        fkw["token_type_ids"] = torch.zeros(inputs_embeds.shape[:2], dtype=torch.long, device=device)
    eos = tokenizer.eos_token_id
    gen: list[int] = []
    out = model(inputs_embeds=inputs_embeds, attention_mask=attn, use_cache=True, **fkw)
    past = out.past_key_values
    nxt = out.logits[:, -1, :].argmax(dim=-1)
    gen.append(int(nxt.item()))
    for _ in range(max_new - 1):
        if eos is not None and gen[-1] == eos:
            gen.pop()
            break
        emb = model.get_input_embeddings()(nxt.unsqueeze(0)).to(torch.bfloat16)
        attn = torch.cat([attn, torch.ones(1, 1, dtype=torch.long, device=device)], dim=1)
        step_kw = {}
        if getattr(model, "_needs_tti", False):
            step_kw["token_type_ids"] = torch.zeros((1, 1), dtype=torch.long, device=device)
        out = model(inputs_embeds=emb, attention_mask=attn, past_key_values=past, use_cache=True, **step_kw)
        past = out.past_key_values
        nxt = out.logits[:, -1, :].argmax(dim=-1)
        gen.append(int(nxt.item()))
    return tokenizer.decode(gen, skip_special_tokens=True).strip()


def _round_or_none(value: float | None, ndigits: int = 6) -> float | None:
    if value is None or not math.isfinite(value):
        return None
    return round(float(value), ndigits)


def extraction_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = collections.Counter(str(row.get("extraction_status", UNPARSED)) for row in rows)
    parsed = counts[PARSED]
    return {
        "extraction_coverage": round(parsed / max(len(rows), 1), 4),
        "n_parsed": parsed,
        "n_unparsed": counts[UNPARSED],
        "n_ambiguous": counts[AMBIGUOUS],
    }


def summarize_task(task: TaskRecord, rows: list[dict[str, Any]]) -> dict[str, Any]:
    diagnostics = extraction_summary(rows)
    if task.label_type == "numeric":
        y_true: list[float] = []
        y_pred: list[float] = []
        missing_rows = [row for row in rows if row.get("answer_type") == "missing"]
        for row in rows:
            if row.get("answer_type") != "numeric":
                continue
            pred = row.get("pred")
            if row.get("extraction_status") != PARSED or pred is None:
                continue
            try:
                target_value = float(row.get("rendered_target_value", row["answer_value"]))
                predicted_value = float(pred)
            except Exception:
                continue
            y_true.append(target_value)
            y_pred.append(predicted_value)
        n = len(rows)
        n_numeric = sum(1 for row in rows if row.get("answer_type") == "numeric")
        n_missing = len(missing_rows)
        n_pred = len(y_pred)
        missing_acc = None
        if n_missing:
            missing_acc = sum(bool(row["ok"]) for row in missing_rows) / n_missing
        if n_pred == 0:
            return {
                "metric_type": "numeric",
                "n": n,
                "n_numeric": n_numeric,
                "n_missing": n_missing,
                "n_pred": 0,
                "numeric_extraction_coverage": 0.0,
                "value_accuracy": 0.0,
                "missing_acc": _round_or_none(missing_acc, 4),
                "mae": None,
                "rmse": None,
                "r2": None,
                **diagnostics,
            }
        errors = [p - y for p, y in zip(y_pred, y_true)]
        mae = sum(abs(e) for e in errors) / n_pred
        rmse = math.sqrt(sum(e * e for e in errors) / n_pred)
        y_mean = sum(y_true) / n_pred
        ss_tot = sum((y - y_mean) ** 2 for y in y_true)
        ss_res = sum(e * e for e in errors)
        r2 = None if ss_tot <= 0 else 1.0 - ss_res / ss_tot
        return {
            "metric_type": "numeric",
            "n": n,
            "n_numeric": n_numeric,
            "n_missing": n_missing,
            "n_pred": n_pred,
            "numeric_extraction_coverage": round(n_pred / max(n_numeric, 1), 4),
            "value_accuracy": round(
                sum(bool(row["ok"]) for row in rows if row.get("answer_type") == "numeric")
                / max(n_numeric, 1),
                4,
            ),
            "missing_acc": _round_or_none(missing_acc, 4),
            "mae": _round_or_none(mae),
            "rmse": _round_or_none(rmse),
            "r2": _round_or_none(r2),
            **diagnostics,
        }

    by_gold: dict[str, list[bool]] = collections.defaultdict(list)
    for row in rows:
        by_gold[str(row["answer_value"])].append(bool(row["ok"]))
    recalls = [sum(v) / len(v) for v in by_gold.values() if v]
    return {
        "metric_type": "classification",
        "n": len(rows),
        "n_gold": len(by_gold),
        "acc": round(sum(bool(row["ok"]) for row in rows) / max(len(rows), 1), 4),
        "balanced_acc": round(sum(recalls) / max(len(recalls), 1), 4),
        **diagnostics,
    }


def _mean_metric(items: list[dict[str, Any]], key: str) -> float | None:
    vals = [float(item[key]) for item in items if item.get(key) is not None]
    if not vals:
        return None
    return round(sum(vals) / len(vals), 6)


def _median_metric(items: list[dict[str, Any]], key: str) -> float | None:
    vals = sorted(float(item[key]) for item in items if item.get(key) is not None)
    if not vals:
        return None
    mid = len(vals) // 2
    if len(vals) % 2:
        return round(vals[mid], 6)
    return round((vals[mid - 1] + vals[mid]) / 2.0, 6)


def numeric_collection_summary(items: list[dict[str, Any]]) -> dict[str, Any]:
    # MAE/RMSE have task-specific units/scales, so they stay in per-task rows.
    # R2 is scale-free and is the only numeric metric aggregated across tasks.
    return {
        "mean_r2": _mean_metric(items, "r2"),
        "median_r2": _median_metric(items, "r2"),
        "mean_extraction_coverage": _mean_metric(items, "numeric_extraction_coverage"),
        "mean_value_accuracy": _mean_metric(items, "value_accuracy"),
        "n_tasks": len(items),
    }


def summarize(results: list[dict[str, Any]], task_by_id: dict[str, TaskRecord]) -> dict[str, Any]:
    by_task: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for r in results:
        by_task[r["task"]].append(r)
    per_task: dict[str, dict[str, Any]] = {}
    for task, rows in sorted(by_task.items()):
        per_task[task] = summarize_task(task_by_id[task], rows)

    numeric_items = [v for v in per_task.values() if v["metric_type"] == "numeric"]
    class_items = [v for v in per_task.values() if v["metric_type"] == "classification"]
    family_numeric: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    family_class: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for task, rec in per_task.items():
        if rec["metric_type"] == "numeric":
            family_numeric[task[0]].append(rec)
        else:
            family_class[task[0]].append(rec)

    return {
        **extraction_summary(results),
        "numeric_r2_summary": numeric_collection_summary(numeric_items),
        "classification_balanced_acc_mean": _mean_metric(class_items, "balanced_acc"),
        "classification_acc_micro": (
            round(
                sum(bool(r["ok"]) for r in results if task_by_id[r["task"]].label_type != "numeric")
                / max(sum(1 for r in results if task_by_id[r["task"]].label_type != "numeric"), 1),
                4,
            )
            if class_items else None
        ),
        "family_numeric_r2_summary": {
            family: numeric_collection_summary(items)
            for family, items in sorted(family_numeric.items())
        },
        "family_classification_balanced_acc_mean": {
            family: _mean_metric(items, "balanced_acc")
            for family, items in sorted(family_class.items())
        },
        "per_task": per_task,
    }


def main() -> None:
    args = parse_args()
    if args.phrases < 1 or args.phrases > N_PHRASES:
        raise ValueError(f"--phrases must be 1..{N_PHRASES}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    tasks = load_task_records(Path(args.meta), Path(args.tasks_root))
    task_config = load_task_config(args.task_config) if args.task_config else None
    if task_config:
        include = set(task_config["task_ids"])
        known = {task.task_id for task in tasks}
        unknown = sorted(include - known)
        if unknown:
            raise ValueError(f"unknown task IDs: {unknown}")
        tasks = [task for task in tasks if task.task_id in include]
        print(
            f"task config={args.task_config} maintain={len(task_config['maintain'])} "
            f"focus={len(task_config['focus'])}",
            flush=True,
        )
    task_by_id = {t.task_id: t for t in tasks}
    rows, labels_df = build_eval_rows(args, tasks)
    task_labels: dict[str, list[Any]] = {}
    for task in tasks:
        if task.probe_applicable and task.label_type != "numeric":
            vals = labels_df[task.task_id].dropna().astype(str).unique().tolist()
            task_labels[task.task_id] = vals

    resume_records = load_resume_records(args.resume_dump)
    expected_keys = {eval_record_key(row) for row in rows}
    unknown_resume = set(resume_records) - expected_keys
    if unknown_resume:
        raise ValueError(f"resume dump has {len(unknown_resume)} rows incompatible with this evaluation configuration")
    if resume_records:
        print(f"[eval] resuming {len(resume_records)}/{len(rows)} compatible records from {args.resume_dump}", flush=True)

    tok, model, proj, spatial, tmask, tid2idx, device = load_model(args.adapter_dir, args.token_mode, args.seed, args.token_cache)
    for path_string in (args.dump, args.xverify_export):
        if path_string:
            Path(path_string).parent.mkdir(parents=True, exist_ok=True)
    append_dump = bool(args.dump and args.resume_dump and Path(args.dump).resolve() == Path(args.resume_dump).resolve())
    dumpf = open(args.dump, "a" if append_dump else "w", encoding="utf-8", buffering=1) if args.dump else None
    xverifyf = open(args.xverify_export, "w", encoding="utf-8", buffering=1) if args.xverify_export else None
    results = list(resume_records.values())
    t0 = time.monotonic()
    by_task = collections.defaultdict(list)
    for r in rows:
        by_task[r["task"]].append(r)
    done = len(results)
    for task_id, task_rows in sorted(by_task.items()):
        task = task_by_id[task_id]
        for r in task_rows:
            key = eval_record_key(r)
            if key in resume_records:
                continue
            gen = greedy_decode_one(r, spatial, tmask, tid2idx, tok, proj, model, device, args.max_new)
            extraction = extract_answer(gen, task, task_labels.get(task_id, []))
            ok = extraction_is_correct(r, task, extraction)
            rec = {
                "task": task_id,
                "tile_id": r["tile_id"],
                "phrase_id": r["phrase_id"],
                "answer_value": r["answer_value"],
                "answer_type": r["answer_type"],
                "pred": extraction.value,
                "ok": bool(ok),
                "extraction_status": extraction.status,
                "extraction_method": extraction.method,
                "extraction_evidence": list(extraction.evidence),
                "extraction_reason": extraction.reason,
                "natural_answer": gen,
                "question": r["question"],
                "answer": r["answer"],
                "generation": gen,
            }
            if r["answer_type"] == "numeric":
                rec["rendered_target_value"] = numeric_target(r, task)
            results.append(rec)
            done += 1
            if dumpf:
                dumpf.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if xverifyf:
                xverifyf.write(json.dumps({
                    "question": r["question"],
                    "correct_answer": r["answer"],
                    "llm_output": gen,
                }, ensure_ascii=False) + "\n")
        task_summary = summarize_task(task, [r for r in results if r["task"] == task_id])
        if task_summary["metric_type"] == "numeric":
            print(
                f"[eval] {task_id} mae={task_summary['mae']} rmse={task_summary['rmse']} "
                f"r2={task_summary['r2']} n={len(task_rows)} done={done}/{len(rows)}",
                flush=True,
            )
        else:
            print(
                f"[eval] {task_id} bacc={task_summary['balanced_acc']:.3f} "
                f"acc={task_summary['acc']:.3f} n={len(task_rows)} done={done}/{len(rows)}",
                flush=True,
            )
    if dumpf:
        dumpf.close()
    if xverifyf:
        xverifyf.close()
    summary = summarize(results, task_by_id)
    summary.update({
        "qa_system_version": QA_SYSTEM_VERSION,
        "answer_protocol": ANSWER_PROTOCOL,
        "adapter": args.adapter_dir,
        "token_mode": args.token_mode,
        "evaluator": "task-aware visible-answer extraction",
        "xverify_export": args.xverify_export or None,
        "split": args.split,
        "n_rows": len(results),
        "cap_per_task": args.cap_per_task,
        "x_cap_per_task": args.x_cap_per_task,
        "seed": args.seed,
        "task_config": args.task_config or None,
        "task_ids": [task.task_id for task in tasks],
        "wall_s": round(time.monotonic() - t0, 1),
    })
    out = args.output or str(Path(args.adapter_dir) / f"generation_{args.split}_{args.token_mode}.json")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        k: summary[k]
        for k in (
            "adapter",
            "token_mode",
            "split",
            "n_rows",
            "numeric_r2_summary",
            "classification_balanced_acc_mean",
            "classification_acc_micro",
            "family_numeric_r2_summary",
            "family_classification_balanced_acc_mean",
            "wall_s",
        )
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
