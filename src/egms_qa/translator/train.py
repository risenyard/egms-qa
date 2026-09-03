"""Sampled EGMS-QA instruction tuning over canonical EGMS-QA task values.

Unlike the older JSONL trainer, this does not materialize every train row.
It samples from the logical pool on the fly:

  tile x delivered task x 20 user-facing phrasings

Token-dependent tasks are weighted as one full tile sweep per logical cycle.
Static X refusal tasks get a smaller per-task weight so they teach boundaries
without dominating the loss.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from egms_qa.translator.train_core import (
    EGMSProjector,
    build_batch,
    evaluate,
    forward_loss,
    per_task_eval_loss,
)
from egms_qa.qa_construction.qa_lib import (
    ANSWER_PROTOCOL,
    DEFAULT_LABELS,
    DEFAULT_META,
    DEFAULT_QA_AUDIT,
    DEFAULT_TASKS_DIR,
    N_PHRASES,
    NUMERIC_ANSWER_STYLES,
    QA_SYSTEM_VERSION,
    TARGET_FORMATS,
    TaskRecord,
    categorical_label_description,
    load_labels,
    load_qa_audit,
    load_task_config,
    load_task_records,
    render_row,
    response_format_instruction,
    supervision_target,
)


from egms_qa.paths import ENCODER_TOKENS, DEFAULT_HOST_MODEL

QWEN_DEFAULT = DEFAULT_HOST_MODEL
TOK_DEFAULT = str(ENCODER_TOKENS)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--qwen-path", default=QWEN_DEFAULT)
    p.add_argument("--token-cache", default=TOK_DEFAULT)
    p.add_argument("--labels", default=str(DEFAULT_LABELS))
    p.add_argument("--meta", default=str(DEFAULT_META))
    p.add_argument("--tasks-root", default=str(DEFAULT_TASKS_DIR))
    p.add_argument("--qa-audit-manifest", default=str(DEFAULT_QA_AUDIT),
                   help="Passed QA audit that supplies approved per-task phrase IDs.")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--resume-adapter", default="")
    p.add_argument("--warm-start-projector", default="none")
    p.add_argument("--task-config", default="",
                   help="JSON manifest with maintain and focus task lists.")
    p.add_argument("--focus-weight", type=float, default=3.0,
                   help="Relative sampled-mode weight for focus tasks.")
    p.add_argument("--semantic-token-weight", type=float, default=1.0,
                   help="Loss multiplier for the rendered value/class phrase of focus tasks.")
    p.add_argument("--balance-focus-classes", action="store_true",
                   help="Sample labels uniformly within focused categorical tasks.")
    p.add_argument("--focus-numeric-quantile-bins", type=int, default=0,
                   help="If positive, sample focused numeric tasks uniformly across this many "
                        "value-quantile bins, with missing values as one additional bin.")
    p.add_argument("--focus-numeric-answer-style", choices=NUMERIC_ANSWER_STYLES, default="standard",
                   help="Numeric answer style used only for focus tasks; maintenance tasks stay standard.")
    p.add_argument("--numeric-aux-weight", type=float, default=0.0,
                   help="Training-only Huber-loss weight for standardized focused numeric targets.")
    p.add_argument("--numeric-rank-aux-weight", type=float, default=0.0,
                   help="Training-only pairwise rank-loss weight for focused numeric targets.")
    p.add_argument("--availability-aux-weight", type=float, default=0.0,
                   help="Training-only balanced BCE weight for availability in focused numeric tasks.")
    p.add_argument("--auxiliary-lr", type=float, default=1e-3,
                   help="Learning rate for training-only numeric auxiliary heads.")

    p.add_argument("--train-steps", type=int, default=34625,
                   help="Micro-steps. 34625 is roughly two 554k-row logical cycles at batch 32.")
    p.add_argument("--one-epoch", action="store_true",
                   help="Run exactly one strict epoch; requires the logical row count to divide the batch size.")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--grad-accum-steps", type=int, default=4)
    p.add_argument("--max-prompt-tokens", type=int, default=192)
    p.add_argument("--max-val-examples", type=int, default=1560)
    p.add_argument("--max-test-examples", type=int, default=1560)
    p.add_argument("--skip-test-eval", action="store_true",
                   help="Do not touch the test split during iterative validation tuning.")
    p.add_argument("--x-train-per-task", type=int, default=3000,
                   help="Relative X-task weight; token tasks are weighted by n_train_tiles.")
    p.add_argument("--phrases", type=int, default=N_PHRASES)
    p.add_argument("--token-phrases", type=int, default=0,
                   help="Training phrase-pool size for token-dependent tasks; 0 uses --phrases.")
    p.add_argument("--x-phrases", type=int, default=0,
                   help="Training phrase-pool size for X tasks; 0 uses --phrases.")
    p.add_argument("--eval-phrases", type=int, default=0,
                   help="Internal validation phrase-pool size; 0 uses --phrases.")
    p.add_argument("--max-train-tiles", type=int, default=0,
                   help="Deterministic train-tile subset for sweeps; 0 uses the full train split.")
    p.add_argument("--train-subset-seed", type=int, default=7013,
                   help="Subset seed kept separate from training order so sweep stages use identical tiles.")
    p.add_argument("--target-format", choices=TARGET_FORMATS, default="natural",
                   help="EGMS-QA trains on the visible natural answer only.")
    p.add_argument("--sampler-mode", choices=("sampled", "epoch"), default="sampled",
                   help="sampled draws tile-task rows with replacement; epoch covers every "
                        "train tile for every token-dependent task once per logical epoch, "
                        "with limited X refusal rows.")

    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--projector-lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--warmup-steps", type=int, default=200)
    p.add_argument("--warmup-ratio", type=float, default=0.0,
                   help="If positive, override --warmup-steps with this fraction of training steps.")
    p.add_argument("--lr-scheduler", choices=("constant", "linear", "cosine"), default="constant")
    p.add_argument("--min-lr-ratio", type=float, default=0.0,
                   help="Final/base learning-rate ratio for linear or cosine decay.")
    p.add_argument("--projector-only-ratio", type=float, default=0.0,
                   help="Initial training fraction that updates the projector while holding LoRA fixed.")
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--projector-dropout", type=float, default=0.05)
    p.add_argument("--eval-every-steps", type=int, default=1500)
    p.add_argument("--save-every-steps", type=int, default=3000)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--seed", type=int, default=7012)
    p.add_argument("--no-4bit", action="store_true")
    p.add_argument("--require-cuda", action="store_true")
    return p.parse_args()


def learning_rate_scale(
    step: int,
    total_steps: int,
    warmup_steps: int,
    scheduler: str,
    min_lr_ratio: float,
) -> float:
    """Return the warmup-plus-decay multiplier for one training micro-step."""
    if total_steps < 1:
        raise ValueError("total_steps must be positive")
    if warmup_steps < 0:
        raise ValueError("warmup_steps must be non-negative")
    if not 0.0 <= min_lr_ratio <= 1.0:
        raise ValueError("min_lr_ratio must be between zero and one")
    if scheduler not in {"constant", "linear", "cosine"}:
        raise ValueError(f"unknown learning-rate scheduler: {scheduler}")
    if warmup_steps and step < warmup_steps:
        return step / warmup_steps
    if scheduler == "constant":
        return 1.0
    decay_steps = max(total_steps - warmup_steps, 1)
    progress = min(1.0, max(0.0, (step - warmup_steps) / decay_steps))
    if scheduler == "linear":
        decay = 1.0 - progress
    else:
        decay = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr_ratio + (1.0 - min_lr_ratio) * decay


def optimizer_group_lr_scales(
    step: int,
    total_steps: int,
    projector_warmup_steps: int,
    lora_warmup_steps: int,
    scheduler: str,
    min_lr_ratio: float,
    projector_only_steps: int,
) -> tuple[float, float]:
    """Return projector and LoRA scales, including an optional projector-only prefix."""
    if not 0 <= projector_only_steps < total_steps:
        raise ValueError("projector_only_steps must be in [0, total_steps)")
    projector_scale = learning_rate_scale(
        step,
        total_steps,
        projector_warmup_steps,
        scheduler,
        min_lr_ratio,
    )
    if step <= projector_only_steps:
        return projector_scale, 0.0
    lora_scale = learning_rate_scale(
        step - projector_only_steps,
        total_steps - projector_only_steps,
        lora_warmup_steps,
        scheduler,
        min_lr_ratio,
    )
    return projector_scale, lora_scale


class QASampler:
    def __init__(
        self,
        labels_path: str,
        meta_path: str,
        tasks_root: str,
        split: str,
        seed: int,
        phrases: int,
        x_per_task: int,
        mode: str = "sampled",
        include_task_ids: set[str] | None = None,
        task_weight_multipliers: dict[str, float] | None = None,
        phrase_ids_by_task: dict[str, list[int]] | None = None,
        target_format: str = "natural",
        token_phrases: int | None = None,
        x_phrases: int | None = None,
        max_rows: int = 0,
        subset_seed: int | None = None,
        focus_task_ids: set[str] | None = None,
        balance_focus_classes: bool = False,
        focus_numeric_quantile_bins: int = 0,
        focus_numeric_answer_style: str = "standard",
        numeric_balance_task_ids: set[str] | None = None,
        availability_balance_task_ids: set[str] | None = None,
    ) -> None:
        self.rng = random.Random(seed)
        self.token_phrases = token_phrases or phrases
        self.x_phrases = x_phrases or phrases
        self.phrases = max(self.token_phrases, self.x_phrases)
        self.mode = mode
        self.target_format = target_format
        self.x_per_task = x_per_task
        self.focus_task_ids = focus_task_ids or set()
        self.balance_focus_classes = balance_focus_classes
        self.focus_numeric_quantile_bins = focus_numeric_quantile_bins
        self.numeric_balance_task_ids = (
            self.focus_task_ids
            if numeric_balance_task_ids is None
            else numeric_balance_task_ids
        )
        self.availability_balance_task_ids = availability_balance_task_ids or set()
        if focus_numeric_answer_style not in NUMERIC_ANSWER_STYLES:
            raise ValueError(f"unknown focus numeric answer style: {focus_numeric_answer_style}")
        self.focus_numeric_answer_style = focus_numeric_answer_style
        if focus_numeric_quantile_bins < 0:
            raise ValueError("focus numeric quantile bins must be non-negative")
        if (
            balance_focus_classes
            or (focus_numeric_quantile_bins and self.numeric_balance_task_ids)
            or self.availability_balance_task_ids
        ) \
                and mode != "sampled":
            raise ValueError("focus balancing requires --sampler-mode sampled")
        labels = load_labels(Path(labels_path))
        self.rows = labels[labels["split"] == split].reset_index(drop=True)
        if self.rows.empty:
            raise RuntimeError(f"no labels for split={split}")
        if max_rows:
            if max_rows < 1:
                raise ValueError(f"max_rows must be positive: {max_rows}")
            if max_rows > len(self.rows):
                raise ValueError(f"max_rows={max_rows} exceeds split size {len(self.rows)}")
            subset_rng = random.Random((seed if subset_seed is None else subset_seed) + 70_130_019)
            chosen = sorted(subset_rng.sample(range(len(self.rows)), max_rows))
            self.rows = self.rows.iloc[chosen].reset_index(drop=True)
        all_tasks = load_task_records(Path(meta_path), Path(tasks_root))
        known_ids = {task.task_id for task in all_tasks}
        include_task_ids = include_task_ids or set()
        unknown = sorted(include_task_ids - known_ids)
        if unknown:
            raise ValueError(f"unknown task IDs: {unknown}")
        self.tasks = [task for task in all_tasks if not include_task_ids or task.task_id in include_task_ids]
        if not self.tasks:
            raise RuntimeError("task selection is empty")
        selected_ids = {task.task_id for task in self.tasks}
        unknown_focus = sorted(self.focus_task_ids - selected_ids)
        if unknown_focus:
            raise ValueError(f"focus task IDs are not selected: {unknown_focus}")
        unknown_numeric_balance = sorted(
            self.numeric_balance_task_ids - self.focus_task_ids
        )
        if unknown_numeric_balance:
            raise ValueError(
                "numeric-balanced task IDs are not focused: "
                f"{unknown_numeric_balance}"
            )
        unknown_availability = sorted(self.availability_balance_task_ids - self.focus_task_ids)
        if unknown_availability:
            raise ValueError(
                "availability-balanced task IDs are not focused: "
                f"{unknown_availability}"
            )
        phrase_ids_by_task = phrase_ids_by_task or {}
        self.phrase_ids: dict[str, list[int]] = {}
        for task in self.tasks:
            ids = list(phrase_ids_by_task.get(task.task_id, []))
            required = self.token_phrases if task.probe_applicable else self.x_phrases
            if len(ids) < required:
                raise ValueError(
                    f"approved phrase pool for {task.task_id} has {len(ids)} entries; "
                    f"{required} required"
                )
            self.phrase_ids[task.task_id] = ids[:required]
        task_weight_multipliers = task_weight_multipliers or {}
        if mode == "epoch" and any(float(v) != 1.0 for v in task_weight_multipliers.values()):
            raise ValueError("task weight multipliers require --sampler-mode sampled")
        self.token_task_idxs = [i for i, task in enumerate(self.tasks) if task.probe_applicable]
        self.x_task_idxs = [i for i, task in enumerate(self.tasks) if not task.probe_applicable]
        weights = []
        for task in self.tasks:
            base_weight = len(self.rows) if task.probe_applicable else x_per_task
            multiplier = float(task_weight_multipliers.get(task.task_id, 1.0))
            if multiplier <= 0:
                raise ValueError(f"task weight must be positive for {task.task_id}: {multiplier}")
            weights.append(base_weight * multiplier)
        self.weights = np.array(weights, dtype=np.float64)
        self.cum = np.cumsum(self.weights)
        self.total = float(self.cum[-1])
        self.balanced_class_rows: dict[str, list[list[int]]] = {}
        if self.balance_focus_classes:
            for task in self.tasks:
                if task.task_id not in self.focus_task_ids or task.label_type != "categorical":
                    continue
                groups: dict[str, list[int]] = {}
                for ridx, value in self.rows[task.task_id].items():
                    value_text = str(value)
                    if value_text in ("", "nan", "None", "<NA>", "NaN"):
                        continue
                    groups.setdefault(value_text, []).append(int(ridx))
                if not groups:
                    raise ValueError(f"focused categorical task has no valid classes: {task.task_id}")
                self.balanced_class_rows[task.task_id] = [groups[key] for key in sorted(groups)]
        self.balanced_numeric_rows: dict[str, list[list[int]]] = {}
        self.balanced_numeric_value_rows: dict[str, list[list[int]]] = {}
        if self.focus_numeric_quantile_bins:
            for task in self.tasks:
                if (
                    task.task_id not in self.numeric_balance_task_ids
                    or task.label_type != "numeric"
                ):
                    continue
                valid_rows: list[int] = []
                valid_values: list[float] = []
                missing_rows: list[int] = []
                for ridx, value in self.rows[task.task_id].items():
                    try:
                        number = float(value)
                    except (TypeError, ValueError):
                        number = float("nan")
                    if math.isfinite(number):
                        valid_rows.append(int(ridx))
                        valid_values.append(number)
                    else:
                        missing_rows.append(int(ridx))
                if not valid_rows:
                    raise ValueError(f"focused numeric task has no valid values: {task.task_id}")
                values = np.asarray(valid_values, dtype=np.float64)
                edges = np.unique(np.quantile(
                    values,
                    np.linspace(0.0, 1.0, self.focus_numeric_quantile_bins + 1),
                ))
                internal_edges = edges[1:-1]
                groups: dict[int, list[int]] = {}
                for ridx, value in zip(valid_rows, values):
                    bin_idx = int(np.searchsorted(internal_edges, value, side="right"))
                    groups.setdefault(bin_idx, []).append(ridx)
                buckets = [groups[key] for key in sorted(groups)]
                self.balanced_numeric_value_rows[task.task_id] = list(buckets)
                if missing_rows:
                    buckets.append(missing_rows)
                self.balanced_numeric_rows[task.task_id] = buckets
        self.balanced_availability_rows: dict[str, list[list[int]]] = {}
        for task in self.tasks:
            if task.task_id not in self.availability_balance_task_ids:
                continue
            if task.label_type != "numeric":
                raise ValueError(
                    f"availability balancing requires a numeric task: {task.task_id}"
                )
            valid_rows: list[int] = []
            missing_rows: list[int] = []
            for ridx, value in self.rows[task.task_id].items():
                try:
                    number = float(value)
                except (TypeError, ValueError):
                    number = float("nan")
                (valid_rows if math.isfinite(number) else missing_rows).append(int(ridx))
            if not valid_rows or not missing_rows:
                raise ValueError(
                    f"availability balancing requires valid and missing rows: {task.task_id}"
                )
            self.balanced_availability_rows[task.task_id] = [valid_rows, missing_rows]
        self._epoch_order: list[tuple[int, int, int]] = []
        self._epoch_pos = 0
        self._epoch_idx = 0

    def _sample_phrase(self, task: TaskRecord) -> int:
        return self.rng.choice(self.phrase_ids[task.task_id])

    def _render_training_row(self, ridx: int, task: TaskRecord, phrase_id: int) -> dict:
        numeric_answer_style = (
            self.focus_numeric_answer_style
            if task.task_id in self.focus_task_ids
            else "standard"
        )
        row = render_row(
            self.rows.iloc[ridx],
            task,
            phrase_id,
            numeric_answer_style=numeric_answer_style,
        )
        row["natural_answer"] = row["answer"]
        row["answer"] = supervision_target(row, self.target_format)
        row["target_format"] = self.target_format
        row["response_instruction"] = response_format_instruction(self.target_format)
        if task.task_id in self.focus_task_ids:
            if row["answer_type"] == "numeric" and row["rendered_target_value"] is not None:
                row["loss_focus_text"] = str(row["rendered_target_value"])
            elif row["answer_type"] == "categorical":
                row["loss_focus_text"] = categorical_label_description(
                    task.task_id,
                    row["answer_value"],
                )
        return row

    def _sample_task(self) -> TaskRecord:
        x = self.rng.random() * self.total
        return self.tasks[int(np.searchsorted(self.cum, x, side="right"))]

    def _build_epoch_order(self) -> None:
        """Create one strict-coverage epoch for token-dependent tasks.

        The token task block covers every split tile for every probe-applicable task
        exactly once. X refusal rows are capped per task because they are not
        token-dependent and should teach scope without dominating the token tasks.
        """
        self._epoch_idx += 1
        order: list[tuple[int, int, int]] = []
        for tidx in self.token_task_idxs:
            task = self.tasks[tidx]
            row_idxs = list(range(len(self.rows)))
            self.rng.shuffle(row_idxs)
            phrase_ids = list(self.phrase_ids[task.task_id])
            self.rng.shuffle(phrase_ids)
            for pos, ridx in enumerate(row_idxs):
                order.append((tidx, ridx, phrase_ids[pos % len(phrase_ids)]))

        for tidx in self.x_task_idxs:
            task = self.tasks[tidx]
            if self.x_per_task > len(self.rows):
                raise ValueError(
                    f"epoch mode requires unique X tiles, but x_per_task={self.x_per_task} "
                    f"exceeds {len(self.rows)} rows"
                )
            row_idxs = self.rng.sample(range(len(self.rows)), self.x_per_task)
            phrase_ids = list(self.phrase_ids[task.task_id])
            self.rng.shuffle(phrase_ids)
            for pos, ridx in enumerate(row_idxs):
                order.append((tidx, ridx, phrase_ids[pos % len(phrase_ids)]))

        self.rng.shuffle(order)

        self._epoch_order = order
        self._epoch_pos = 0

    def _sample_epoch_row(self) -> dict:
        if self._epoch_pos >= len(self._epoch_order):
            self._build_epoch_order()
        tidx, ridx, pidx = self._epoch_order[self._epoch_pos]
        self._epoch_pos += 1
        return self._render_training_row(ridx, self.tasks[tidx], pidx)

    def sample_row(self) -> dict:
        if self.mode == "epoch":
            return self._sample_epoch_row()
        task = self._sample_task()
        availability_rows = self.balanced_availability_rows.get(task.task_id)
        if availability_rows:
            availability_idx = self.rng.randrange(len(availability_rows))
            numeric_value_rows = self.balanced_numeric_value_rows.get(task.task_id)
            if availability_idx == 0 and numeric_value_rows:
                ridx = self.rng.choice(self.rng.choice(numeric_value_rows))
            else:
                ridx = self.rng.choice(availability_rows[availability_idx])
        else:
            class_rows = self.balanced_class_rows.get(task.task_id)
            if class_rows:
                ridx = self.rng.choice(self.rng.choice(class_rows))
            else:
                numeric_rows = self.balanced_numeric_rows.get(task.task_id)
                if numeric_rows:
                    ridx = self.rng.choice(self.rng.choice(numeric_rows))
                else:
                    ridx = self.rng.randrange(len(self.rows))
        pidx = self._sample_phrase(task)
        return self._render_training_row(ridx, task, pidx)

    def sample_batch(self, n: int) -> list[dict]:
        return [self.sample_row() for _ in range(n)]

    def balanced_eval_rows(self, n_total: int, seed: int) -> list[dict]:
        rng = random.Random(seed)
        per_task = max(1, math.ceil(n_total / len(self.tasks)))
        out = []
        for task in self.tasks:
            idxs = list(range(len(self.rows)))
            rng.shuffle(idxs)
            if per_task <= len(idxs):
                chosen = idxs[:per_task]
            else:
                chosen = [rng.choice(idxs) for _ in range(per_task)]
            for ridx in chosen:
                out.append(self._render_training_row(
                    ridx,
                    task,
                    rng.choice(self.phrase_ids[task.task_id]),
                ))
        rng.shuffle(out)
        return out[:n_total]

    def expected_task_fraction(self) -> dict[str, float]:
        if self.mode == "epoch":
            epoch_total = len(self.token_task_idxs) * len(self.rows) + len(self.x_task_idxs) * self.x_per_task
            return {
                t.task_id: float((len(self.rows) if t.probe_applicable else self.x_per_task) / epoch_total)
                for t in self.tasks
            }
        return {t.task_id: float(w / self.total) for t, w in zip(self.tasks, self.weights)}

    @property
    def logical_epoch_examples(self) -> int:
        return len(self.token_task_idxs) * len(self.rows) + len(self.x_task_idxs) * self.x_per_task




def numeric_auxiliary_stats(rows, tasks: list[TaskRecord], focus_task_ids: set[str]) -> dict[str, dict]:
    """Compute train-only normalization and availability statistics per numeric task."""
    stats: dict[str, dict] = {}
    for task in tasks:
        if task.task_id not in focus_task_ids or task.label_type != "numeric":
            continue
        values = []
        n_missing = 0
        for raw in rows[task.task_id].tolist():
            try:
                value = float(raw)
            except (TypeError, ValueError):
                value = float("nan")
            if math.isfinite(value):
                values.append(value)
            else:
                n_missing += 1
        if not values:
            raise ValueError(f"focused numeric task has no valid training targets: {task.task_id}")
        values_array = np.asarray(values, dtype=np.float64)
        scale = float(values_array.std())
        if not math.isfinite(scale) or scale < 1e-8:
            scale = 1.0
        stats[task.task_id] = {
            "mean": float(values_array.mean()),
            "scale": scale,
            "n_valid": len(values),
            "n_missing": n_missing,
            "has_availability_target": bool(n_missing),
        }
    return stats


class NumericAuxiliaryHeads(nn.Module):
    """Small task-routed heads used only to shape decoder hidden states."""

    def __init__(self, hidden_size: int, stats: dict[str, dict]):
        super().__init__()
        self.regression = nn.ModuleDict({task_id: nn.Linear(hidden_size, 1) for task_id in stats})
        self.availability = nn.ModuleDict({
            task_id: nn.Linear(hidden_size, 1)
            for task_id, task_stats in stats.items()
            if task_stats["has_availability_target"]
        })


def numeric_auxiliary_losses(
    rows: list[dict],
    prompt_hidden: torch.Tensor,
    focus_hidden: torch.Tensor,
    heads: NumericAuxiliaryHeads,
    stats: dict[str, dict],
) -> dict[str, torch.Tensor | int]:
    """Return task-balanced value, rank, and availability losses for one batch."""
    zero = prompt_hidden.sum() * 0.0
    regression_losses = []
    rank_losses = []
    availability_losses = []
    n_regression = 0
    n_rank_pairs = 0
    n_availability = 0

    for task_id, task_stats in stats.items():
        task_indices = [i for i, row in enumerate(rows) if row["task"] == task_id]
        if not task_indices:
            continue
        available_indices = [i for i in task_indices if rows[i]["answer_type"] == "numeric"]
        if available_indices:
            index = torch.tensor(available_indices, dtype=torch.long, device=focus_hidden.device)
            hidden = focus_hidden.index_select(0, index)
            predictions = heads.regression[task_id](hidden).squeeze(-1).float()
            targets = torch.tensor(
                [
                    (float(rows[i]["answer_value"]) - task_stats["mean"]) / task_stats["scale"]
                    for i in available_indices
                ],
                dtype=torch.float32,
                device=focus_hidden.device,
            )
            regression_losses.append(F.smooth_l1_loss(predictions, targets, beta=1.0))
            n_regression += len(available_indices)
            if len(available_indices) > 1:
                pair_i, pair_j = torch.triu_indices(
                    len(available_indices), len(available_indices), offset=1,
                    device=focus_hidden.device,
                )
                target_delta = targets[pair_i] - targets[pair_j]
                unequal = target_delta.abs() > 1e-6
                if unequal.any():
                    prediction_delta = predictions[pair_i] - predictions[pair_j]
                    rank_losses.append(F.softplus(
                        -target_delta[unequal].sign() * prediction_delta[unequal]
                    ).mean())
                    n_rank_pairs += int(unequal.sum().item())

        if task_id in heads.availability:
            index = torch.tensor(task_indices, dtype=torch.long, device=prompt_hidden.device)
            hidden = prompt_hidden.index_select(0, index)
            logits = heads.availability[task_id](hidden).squeeze(-1).float()
            targets = torch.tensor(
                [1.0 if rows[i]["answer_type"] == "numeric" else 0.0 for i in task_indices],
                dtype=torch.float32,
                device=prompt_hidden.device,
            )
            availability_losses.append(F.binary_cross_entropy_with_logits(
                logits,
                targets,
            ))
            n_availability += len(task_indices)

    return {
        "regression": torch.stack(regression_losses).mean() if regression_losses else zero,
        "rank": torch.stack(rank_losses).mean() if rank_losses else zero,
        "availability": torch.stack(availability_losses).mean() if availability_losses else zero,
        "n_regression": n_regression,
        "n_rank_pairs": n_rank_pairs,
        "n_availability": n_availability,
    }


def load_base_model(args: argparse.Namespace, device: torch.device):
    import transformers
    from transformers import AutoConfig, AutoTokenizer

    config = AutoConfig.from_pretrained(args.qwen_path, trust_remote_code=True)
    model_type = str(getattr(config, "model_type", "")).lower()
    tokenizer = AutoTokenizer.from_pretrained(args.qwen_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    kw = dict(trust_remote_code=True, torch_dtype=torch.bfloat16, device_map={"": 0})
    if not args.no_4bit:
        from transformers import BitsAndBytesConfig

        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    print(f"loading base 4bit={not args.no_4bit} model_type={model_type} from {args.qwen_path}", flush=True)
    model = None
    used = None
    if model_type == "gemma3":
        load_order = ["Gemma3ForConditionalGeneration", "AutoModelForImageTextToText", "AutoModelForCausalLM"]
    elif "qwen" in model_type:
        load_order = ["AutoModelForImageTextToText", "AutoModelForCausalLM"]
    else:
        load_order = ["AutoModelForCausalLM", "AutoModelForImageTextToText"]
    for cls_name in load_order:
        if not hasattr(transformers, cls_name):
            continue
        try:
            model = getattr(transformers, cls_name).from_pretrained(args.qwen_path, **kw)
            used = cls_name
            break
        except Exception as exc:
            print(f"{cls_name} failed: {repr(exc)[:160]}", flush=True)
    if model is None:
        raise RuntimeError("could not load base model")
    print(f"loaded via {used}", flush=True)

    from peft import LoraConfig, PeftModel, get_peft_model

    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    has_vision = any("vision" in n for n, _ in model.named_modules())
    tgt = (
        r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"
        if has_vision
        else "all-linear"
    )
    if args.resume_adapter and args.resume_adapter.lower() != "none":
        model = PeftModel.from_pretrained(model, args.resume_adapter, is_trainable=True)
        print(f"resumed LoRA adapter from {args.resume_adapter}", flush=True)
    else:
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            target_modules=tgt,
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
    model._needs_tti = has_vision
    model.config.use_cache = False
    model.train()
    model.print_trainable_parameters()
    hidden = model.get_input_embeddings().embedding_dim
    return tokenizer, model, hidden


def main() -> None:
    args = parse_args()
    if args.semantic_token_weight < 1.0:
        raise ValueError("--semantic-token-weight must be >= 1")
    if args.focus_numeric_quantile_bins < 0:
        raise ValueError("--focus-numeric-quantile-bins must be non-negative")
    if (
        args.numeric_aux_weight < 0
        or args.numeric_rank_aux_weight < 0
        or args.availability_aux_weight < 0
    ):
        raise ValueError("auxiliary loss weights must be non-negative")
    if args.auxiliary_lr <= 0:
        raise ValueError("--auxiliary-lr must be positive")
    if not 0.0 <= args.min_lr_ratio <= 1.0:
        raise ValueError("--min-lr-ratio must be between zero and one")
    if not 0.0 <= args.warmup_ratio < 1.0:
        raise ValueError("--warmup-ratio must be in [0, 1)")
    if not 0.0 <= args.projector_only_ratio < 1.0:
        raise ValueError("--projector-only-ratio must be in [0, 1)")
    if args.require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    token_phrases = args.token_phrases or args.phrases
    x_phrases = args.x_phrases or args.phrases
    eval_phrases = args.eval_phrases or args.phrases
    phrase_counts = {
        "phrases": args.phrases,
        "token_phrases": token_phrases,
        "x_phrases": x_phrases,
        "eval_phrases": eval_phrases,
    }
    invalid_phrase_counts = {key: value for key, value in phrase_counts.items() if not 1 <= value <= N_PHRASES}
    if invalid_phrase_counts:
        raise ValueError(f"phrase counts must be 1..{N_PHRASES}: {invalid_phrase_counts}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    task_config = load_task_config(args.task_config) if args.task_config else None
    qa_audit = load_qa_audit(args.qa_audit_manifest, max(phrase_counts.values()))
    phrase_ids_by_task = qa_audit["approved_phrase_ids"]
    print(
        f"QA gate={args.qa_audit_manifest} status=pass "
        f"token_phrases={token_phrases} x_phrases={x_phrases} eval_phrases={eval_phrases}",
        flush=True,
    )
    include_task_ids = set(task_config["task_ids"]) if task_config else set()
    task_weight_multipliers = (
        {task_id: args.focus_weight for task_id in task_config["focus"]}
        if task_config else {}
    )
    focus_task_ids = set(task_config["focus"]) if task_config else set()
    numeric_auxiliary_task_ids = (
        set(task_config["numeric_auxiliary"]) if task_config else set()
    )
    numeric_balance_task_ids = (
        set(task_config["numeric_balance"]) if task_config else set()
    )
    availability_balance_task_ids = (
        set(task_config["availability_balance"]) if task_config else set()
    )
    if (
        args.semantic_token_weight > 1.0
        or args.balance_focus_classes
        or args.focus_numeric_quantile_bins
        or args.focus_numeric_answer_style != "standard"
        or args.numeric_aux_weight > 0
        or args.numeric_rank_aux_weight > 0
        or args.availability_aux_weight > 0
        or availability_balance_task_ids
    ) and not focus_task_ids:
        raise ValueError("focus losses and balancing require --task-config focus tasks")
    if task_config:
        print(
            f"task config={args.task_config} maintain={len(task_config['maintain'])} "
            f"focus={len(task_config['focus'])} focus_weight={args.focus_weight} "
            f"semantic_token_weight={args.semantic_token_weight} "
            f"balance_focus_classes={args.balance_focus_classes} "
            f"focus_numeric_quantile_bins={args.focus_numeric_quantile_bins} "
            f"focus_numeric_answer_style={args.focus_numeric_answer_style} "
            f"numeric_aux_weight={args.numeric_aux_weight} "
            f"numeric_rank_aux_weight={args.numeric_rank_aux_weight} "
            f"numeric_auxiliary_tasks={sorted(numeric_auxiliary_task_ids)} "
            f"numeric_balance={sorted(numeric_balance_task_ids)} "
            f"availability_balance={sorted(availability_balance_task_ids)} "
            f"availability_aux_weight={args.availability_aux_weight}",
            flush=True,
        )

    train_sampler = QASampler(
        args.labels,
        args.meta,
        args.tasks_root,
        "train",
        args.seed,
        args.phrases,
        args.x_train_per_task,
        args.sampler_mode,
        include_task_ids,
        task_weight_multipliers,
        phrase_ids_by_task,
        args.target_format,
        token_phrases=token_phrases,
        x_phrases=x_phrases,
        max_rows=args.max_train_tiles,
        subset_seed=args.train_subset_seed,
        focus_task_ids=focus_task_ids,
        balance_focus_classes=args.balance_focus_classes,
        focus_numeric_quantile_bins=args.focus_numeric_quantile_bins,
        focus_numeric_answer_style=args.focus_numeric_answer_style,
        numeric_balance_task_ids=numeric_balance_task_ids,
        availability_balance_task_ids=availability_balance_task_ids,
    )
    val_sampler = QASampler(
        args.labels,
        args.meta,
        args.tasks_root,
        "val",
        args.seed + 1,
        eval_phrases,
        max(1, args.x_train_per_task // 10),
        "sampled",
        include_task_ids,
        phrase_ids_by_task=phrase_ids_by_task,
        target_format=args.target_format,
        token_phrases=eval_phrases,
        x_phrases=eval_phrases,
    )
    test_sampler = None
    if not args.skip_test_eval:
        test_sampler = QASampler(
            args.labels,
            args.meta,
            args.tasks_root,
            "test",
            args.seed + 2,
            eval_phrases,
            max(1, args.x_train_per_task // 10),
            "sampled",
            include_task_ids,
            phrase_ids_by_task=phrase_ids_by_task,
            target_format=args.target_format,
            token_phrases=eval_phrases,
            x_phrases=eval_phrases,
        )
    if args.one_epoch:
        if args.sampler_mode != "epoch":
            raise ValueError("--one-epoch requires --sampler-mode epoch")
        logical_examples = train_sampler.logical_epoch_examples
        if logical_examples % args.batch_size:
            raise ValueError(
                f"logical epoch has {logical_examples} examples, not divisible by batch size {args.batch_size}"
            )
        args.train_steps = logical_examples // args.batch_size
    if args.warmup_ratio:
        args.warmup_steps = round(args.train_steps * args.warmup_ratio)
    projector_only_steps = round(args.train_steps * args.projector_only_ratio)
    if projector_only_steps >= args.train_steps:
        raise ValueError("projector-only prefix leaves no LoRA training steps")
    lora_train_steps = args.train_steps - projector_only_steps
    lora_warmup_steps = (
        round(lora_train_steps * args.warmup_ratio)
        if args.warmup_ratio
        else min(args.warmup_steps, lora_train_steps)
    )
    print(
        f"sampled train split={len(train_sampler.rows)} tiles tasks={len(train_sampler.tasks)} "
        f"max_val_examples={args.max_val_examples} max_test_examples={args.max_test_examples} "
        f"token_phrases={token_phrases} x_phrases={x_phrases} eval_phrases={eval_phrases} "
        f"sampler_mode={args.sampler_mode} logical_epoch_examples={train_sampler.logical_epoch_examples} "
        f"target_format={args.target_format}",
        flush=True,
    )
    print(
        f"learning-rate schedule={args.lr_scheduler} min_lr_ratio={args.min_lr_ratio} "
        f"projector_lr={args.projector_lr} lora_lr={args.lr} "
        f"projector_warmup_steps={args.warmup_steps} "
        f"projector_only_steps={projector_only_steps} lora_warmup_steps={lora_warmup_steps}",
        flush=True,
    )

    auxiliary_enabled = (
        args.numeric_aux_weight > 0
        or args.numeric_rank_aux_weight > 0
        or args.availability_aux_weight > 0
    )
    auxiliary_stats = (
        numeric_auxiliary_stats(
            train_sampler.rows,
            train_sampler.tasks,
            numeric_auxiliary_task_ids,
        )
        if auxiliary_enabled else {}
    )
    if auxiliary_enabled and not auxiliary_stats:
        raise ValueError("auxiliary supervision requested but no focused numeric tasks were selected")
    if auxiliary_enabled:
        print(f"numeric auxiliary stats={json.dumps(auxiliary_stats, sort_keys=True)}", flush=True)

    tokenizer, model, llm_hidden = load_base_model(args, device)
    lora_params = [p for p in model.parameters() if p.requires_grad]
    auxiliary_heads = None
    if auxiliary_enabled:
        auxiliary_heads = NumericAuxiliaryHeads(llm_hidden, auxiliary_stats).to(device, torch.bfloat16)

    cache = torch.load(args.token_cache, map_location="cpu", weights_only=False)
    spatial = cache["spatial_tokens"].to(torch.float32)
    tok_mask = cache["token_mask"]
    tid2idx = {str(t): i for i, t in enumerate(cache["tile_ids"])}
    egms_dim = spatial.shape[-1]
    print(f"token cache {tuple(spatial.shape)} egms_dim={egms_dim}", flush=True)

    projector = EGMSProjector(egms_dim, llm_hidden, args.projector_dropout).to(device, torch.bfloat16)
    if args.warm_start_projector and args.warm_start_projector.lower() != "none":
        ck = torch.load(args.warm_start_projector, map_location="cpu", weights_only=False)
        projector.load_state_dict(ck["projector_state"])
        print(f"warm-started projector from {args.warm_start_projector}", flush=True)
    else:
        print("projector trained from scratch", flush=True)
    projector.train()

    optimizer_groups = [
        {"params": list(projector.parameters()), "lr": args.projector_lr, "weight_decay": args.weight_decay},
        {"params": lora_params, "lr": args.lr, "weight_decay": args.weight_decay},
    ]
    if auxiliary_heads is not None:
        optimizer_groups.append({
            "params": list(auxiliary_heads.parameters()),
            "lr": args.auxiliary_lr,
            "weight_decay": args.weight_decay,
        })
    opt = torch.optim.AdamW(optimizer_groups)
    base_lrs = [g["lr"] for g in opt.param_groups]

    best_ppl = float("inf")
    best_metrics = None
    gstep = 0
    t0 = time.monotonic()
    train_log = open(out / "train_log.jsonl", "w", encoding="utf-8")
    eval_log = open(out / "eval_log.jsonl", "w", encoding="utf-8")

    def save_ckpt(tag: str, metrics: dict) -> None:
        d = out / tag
        d.mkdir(parents=True, exist_ok=True)
        checkpoint = {
                "projector_state": projector.state_dict(),
                "qa_system_version": QA_SYSTEM_VERSION,
                "answer_protocol": ANSWER_PROTOCOL,
                "args": vars(args),
                "metrics": metrics,
                "egms_dim": egms_dim,
                "llm_hidden": llm_hidden,
        }
        if auxiliary_heads is not None:
            checkpoint["numeric_auxiliary_state"] = auxiliary_heads.state_dict()
            checkpoint["numeric_auxiliary_stats"] = auxiliary_stats
        torch.save(checkpoint, d / "projector.pt")
        model.save_pretrained(d / "qwen_lora_adapter")

    def run_eval() -> None:
        nonlocal best_ppl, best_metrics
        sample_seed = args.seed + 100_000 + gstep
        val_rows = val_sampler.balanced_eval_rows(args.max_val_examples, sample_seed)
        m = evaluate(val_rows, spatial, tok_mask, tid2idx, tokenizer, projector, model, device, args)
        m["step"] = gstep
        m["sample_seed"] = sample_seed
        m["n_eval_examples"] = len(val_rows)
        m["per_task_loss"] = per_task_eval_loss(
            val_rows, spatial, tok_mask, tid2idx, tokenizer, projector, model, device, args, cap=40
        )
        eval_log.write(json.dumps(m) + "\n")
        eval_log.flush()
        print(f"  EVAL step {gstep}: loss={m['loss']:.4f} ppl={m['ppl']:.3f}", flush=True)
        if m["ppl"] < best_ppl:
            best_ppl = m["ppl"]
            best_metrics = m
            save_ckpt("best", m)
            print(f"  saved best ppl={best_ppl:.3f}", flush=True)

    opt.zero_grad(set_to_none=True)
    print(f"total training micro-steps {args.train_steps}", flush=True)
    for gstep in range(1, args.train_steps + 1):
        rows = train_sampler.sample_batch(args.batch_size)
        batch = build_batch(
            rows,
            spatial,
            tok_mask,
            tid2idx,
            tokenizer,
            device,
            args.max_prompt_tokens,
            semantic_token_weight=args.semantic_token_weight,
            track_focus_positions=auxiliary_heads is not None,
        )
        loss, info = forward_loss(
            batch,
            projector,
            model,
            return_prompt_hidden=auxiliary_heads is not None,
            return_focus_hidden=auxiliary_heads is not None,
        )
        if auxiliary_heads is not None:
            prompt_hidden = info.pop("prompt_hidden")
            focus_hidden = info.pop("focus_hidden")
            auxiliary = numeric_auxiliary_losses(
                rows, prompt_hidden, focus_hidden, auxiliary_heads, auxiliary_stats
            )
            lm_loss = loss
            loss = (
                lm_loss
                + args.numeric_aux_weight * auxiliary["regression"]
                + args.numeric_rank_aux_weight * auxiliary["rank"]
                + args.availability_aux_weight * auxiliary["availability"]
            )
            info.update({
                "lm_loss": info["loss"],
                "numeric_aux_loss": float(auxiliary["regression"].item()),
                "numeric_rank_aux_loss": float(auxiliary["rank"].item()),
                "availability_aux_loss": float(auxiliary["availability"].item()),
                "n_numeric_aux": auxiliary["n_regression"],
                "n_numeric_rank_pairs": auxiliary["n_rank_pairs"],
                "n_availability_aux": auxiliary["n_availability"],
                "loss": float(loss.item()),
            })
        (loss / args.grad_accum_steps).backward()
        if gstep % args.grad_accum_steps == 0:
            projector_scale, lora_scale = optimizer_group_lr_scales(
                gstep,
                args.train_steps,
                args.warmup_steps,
                lora_warmup_steps,
                args.lr_scheduler,
                args.min_lr_ratio,
                projector_only_steps,
            )
            opt.param_groups[0]["lr"] = base_lrs[0] * projector_scale
            opt.param_groups[1]["lr"] = base_lrs[1] * lora_scale
            if lora_scale == 0.0:
                for param in lora_params:
                    param.grad = None
            for group, base_lr in zip(opt.param_groups[2:], base_lrs[2:]):
                group["lr"] = base_lr * projector_scale
            trainable = list(projector.parameters()) + lora_params
            if auxiliary_heads is not None:
                trainable += list(auxiliary_heads.parameters())
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
        if gstep % args.log_every == 0:
            elapsed = time.monotonic() - t0
            rec = {
                "step": gstep,
                **info,
                "lr": opt.param_groups[1]["lr"],
                "projector_lr": opt.param_groups[0]["lr"],
                "elapsed": elapsed,
            }
            train_log.write(json.dumps(rec) + "\n")
            train_log.flush()
            print(
                f"  step {gstep}/{args.train_steps} loss={info['loss']:.4f} "
                f"lr={opt.param_groups[1]['lr']:.2e} "
                f"projector_lr={opt.param_groups[0]['lr']:.2e} {elapsed:.0f}s",
                flush=True,
            )
        if args.eval_every_steps and gstep % args.eval_every_steps == 0:
            run_eval()
        if args.save_every_steps and gstep % args.save_every_steps == 0:
            save_ckpt("last", {"step": gstep})

    final_seed = args.seed + 900_000 + gstep
    final_val_rows = val_sampler.balanced_eval_rows(args.max_val_examples, final_seed)
    final = evaluate(final_val_rows, spatial, tok_mask, tid2idx, tokenizer, projector, model, device, args)
    final["step"] = gstep
    final["sample_seed"] = final_seed
    final["n_eval_examples"] = len(final_val_rows)
    final["per_task_loss"] = per_task_eval_loss(
        final_val_rows, spatial, tok_mask, tid2idx, tokenizer, projector, model, device, args, cap=40
    )
    eval_log.write(json.dumps({"final": True, "split": "val", **final}) + "\n")
    test_final = None
    if test_sampler is not None:
        test_seed = args.seed + 950_000 + gstep
        test_rows = test_sampler.balanced_eval_rows(args.max_test_examples, test_seed)
        test_final = evaluate(test_rows, spatial, tok_mask, tid2idx, tokenizer, projector, model, device, args)
        test_final["sample_seed"] = test_seed
        test_final["n_eval_examples"] = len(test_rows)
        eval_log.write(json.dumps({"final": True, "split": "test", **test_final}) + "\n")
    eval_log.close()
    train_log.close()
    save_ckpt("last", final)
    if final["ppl"] < best_ppl:
        best_ppl = final["ppl"]
        best_metrics = final
        save_ckpt("best", final)

    summary = {
        "qa_system_version": QA_SYSTEM_VERSION,
        "answer_protocol": ANSWER_PROTOCOL,
        "best_metrics": best_metrics,
        "final_metrics": final,
        "args": vars(args),
        "n_train_tiles": len(train_sampler.rows),
        "n_val_examples": args.max_val_examples,
        "n_test_examples": 0 if args.skip_test_eval else args.max_test_examples,
        "sampled_train_examples": args.train_steps * args.batch_size,
        "logical_epoch_examples": train_sampler.logical_epoch_examples,
        "training_phrase_counts": phrase_counts,
        "test_metrics": test_final,
        "x_train_per_task_weight": args.x_train_per_task,
        "task_weight_fraction": train_sampler.expected_task_fraction(),
        "task_config": task_config,
        "numeric_auxiliary_stats": auxiliary_stats,
        "qa_audit_manifest": args.qa_audit_manifest,
        "qa_audit_version": qa_audit.get("version"),
        "global_step": gstep,
        "resolved_projector_warmup_steps": args.warmup_steps,
        "resolved_lora_warmup_steps": lora_warmup_steps,
        "resolved_projector_only_steps": projector_only_steps,
        "wall_time_seconds": round(time.monotonic() - t0, 2),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {out}/{{best,last}} best ppl={best_ppl:.3f}", flush=True)


if __name__ == "__main__":
    main()
