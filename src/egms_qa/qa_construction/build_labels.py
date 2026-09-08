"""Assemble task result tables into EGMS-QA labels and metadata.

Tile-dependent targets are aligned by tile ID and split. X refusal tasks stay
in metadata. The fixed release size is checked only with --validate-release.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from egms_qa.paths import (
    OUTPUTS_DIR,
    TASKS_DIR as DEFAULT_TASKS_DIR,
)

from egms_qa.qa_construction.task_specs import TASK_SPECS
from egms_qa.qa_construction.tables import align_family_to_base, read_family

DEFAULT_OUT = OUTPUTS_DIR / "labels-generated"


def check_output_paths(out: Path) -> None:
    """Reject existing outputs, including dangling links to release files."""
    for name in ("labels.parquet", "labels_meta.json"):
        path = out / name
        if path.exists() or path.is_symlink():
            raise FileExistsError(
                f"Refusing to overwrite {path}; choose a different --out-dir."
            )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--tasks-root", default=str(DEFAULT_TASKS_DIR))
    p.add_argument(
        "--out-dir", default=str(DEFAULT_OUT),
        help="Output directory (default: %(default)s); existing output files are never overwritten.",
    )
    p.add_argument("--encoder-cache", help="Optionally validate and order labels against this token cache.")
    p.add_argument("--skip-cache-validation", action="store_true", help="Omit token-cache validation, even if a cache is supplied.")
    p.add_argument("--validate-release", action="store_true", help="Require the released 8,000/1,000/1,000 tile split.")
    return p.parse_args()


def normalize_series(s: pd.Series, label_type: str) -> pd.Series:
    if label_type == "numeric":
        return pd.to_numeric(s, errors="coerce")
    out = s.astype("string").str.strip()
    out = out.mask(out.isin(["", "nan", "None", "<NA>"]))
    return out


def load_cache_ids(path: Path) -> tuple[list[str], list[str]]:
    import torch

    cache = torch.load(path, map_location="cpu", weights_only=True)
    ids = [str(t) for t in cache["tile_ids"]]
    splits = [str(s) for s in cache.get("splits", [""] * len(ids))]
    if len(ids) != len(set(ids)):
        raise ValueError("encoder_cache contains duplicate tile IDs")
    if len(splits) != len(ids):
        raise ValueError("encoder_cache tile_id/split length mismatch")
    return ids, splits


def collect_x_tasks(root: Path) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for family in ("x1", "x2", "x3"):
        path = root / family / f"{family}_final_table.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        df = pd.read_csv(path)
        required = {"task_id", "target_column"}
        if not required.issubset(df.columns):
            raise ValueError(f"{path} missing {sorted(required - set(df.columns))}")
        for row in df.itertuples(index=False):
            task_id = str(getattr(row, "task_id"))
            target_column = str(getattr(row, "target_column"))
            name = target_column
            if target_column.startswith(task_id + "_"):
                name = target_column[len(task_id) + 1 :].replace("_", " ")
            tasks.append({
                "id": task_id,
                "family": "X",
                "source_family": family,
                "source_path": str(path.resolve()),
                "target_column": target_column,
                "label_type": "refusal",
                "name": name,
                "construct": False,
                "probe_applicable": False,
                "note": "Static boundary/refusal catalog; no token-dependent target.",
            })
    return tasks


def split_counts(df: pd.DataFrame) -> dict[str, int]:
    return {str(k): int(v) for k, v in df["split"].value_counts().sort_index().items()}


def label_stats(df: pd.DataFrame, task_id: str, label_type: str) -> dict[str, Any]:
    s = df[task_id]
    if label_type == "numeric":
        v = pd.to_numeric(s, errors="coerce")
        return {
            "label_n": int(v.notna().sum()),
            "label_std": float(np.std(v.dropna().to_numpy(dtype=float))) if v.notna().any() else None,
            "class_count": None,
            "min_class_frac": None,
            "majority_class_frac": None,
        }
    v = s.astype("string").dropna()
    vc = Counter(v.tolist())
    total = sum(vc.values())
    return {
        "label_n": int(total),
        "label_std": None,
        "class_count": int(len(vc)),
        "min_class_frac": float(min(vc.values()) / total) if total else None,
        "majority_class_frac": float(max(vc.values()) / total) if total else None,
    }


def main() -> None:
    args = parse_args()
    root = Path(args.tasks_root)
    out = Path(args.out_dir)
    check_output_paths(out)

    families = sorted({spec["source_family"] for spec in TASK_SPECS})
    tables = {family: read_family(root, family, expected_rows=10000 if args.validate_release else None)
              for family in families}

    first = tables[families[0]][["tile_id", "split"]].copy()
    first["tile_id"] = first["tile_id"].astype(str)
    first["split"] = first["split"].astype(str)
    labels = first.copy()
    base_index = pd.MultiIndex.from_frame(labels[["tile_id", "split"]])
    if base_index.has_duplicates:
        raise ValueError("base tile_id/split index has duplicates")

    table_checks: dict[str, Any] = {}
    for family, table in tables.items():
        aligned = align_family_to_base(table, base_index, family)
        tables[family] = aligned
        table_checks[family] = {
            "rows": int(len(table)),
            "split_counts": split_counts(table),
            "duplicate_tile_id": int(table["tile_id"].duplicated().sum()),
            "order": "aligned_to_base_tile_id_split_index",
        }

    tasks: list[dict[str, Any]] = []
    for spec in TASK_SPECS:
        family = spec["source_family"]
        table = tables[family]
        target = spec["target_column"]
        if target not in table.columns:
            raise ValueError(f"{family} is missing target column {target} for {spec['id']}")
        col = normalize_series(table[target], spec["label_type"])
        labels[spec["id"]] = col
        task = {
            "id": spec["id"],
            "family": spec["id"][0],
            "source_family": family,
            "source_path": str((root / family / f"{family}_final_table.csv").resolve()),
            "target_column": target,
            "label_type": spec["label_type"],
            "name": spec["name"],
            "construct": bool(spec.get("construct", False)),
            "probe_applicable": True,
        }
        task.update(label_stats(labels, spec["id"], spec["label_type"]))
        tasks.append(task)

    x_tasks = collect_x_tasks(root)
    tasks.extend(x_tasks)

    cache_checks: dict[str, Any] = {}
    if args.encoder_cache is not None and not args.skip_cache_validation:
        encoder_path = Path(args.encoder_cache)
        encoder_ids, encoder_splits = load_cache_ids(encoder_path)
        label_ids = labels["tile_id"].astype(str).tolist()
        label_split_by_id = dict(zip(labels["tile_id"].astype(str), labels["split"].astype(str)))
        if set(encoder_ids) != set(label_ids):
            missing = sorted(set(encoder_ids) - set(label_ids))[:5]
            extra = sorted(set(label_ids) - set(encoder_ids))[:5]
            raise ValueError(f"encoder_cache tile_id set mismatch; missing={missing} extra={extra}")
        if any(encoder_splits):
            bad = [
                (tid, sp, label_split_by_id.get(tid))
                for tid, sp in zip(encoder_ids, encoder_splits)
                if label_split_by_id.get(tid) != sp
            ]
            if bad:
                raise ValueError(f"encoder_cache split mismatch examples: {bad[:5]}")

        # Canonicalize label row order to the encoder cache. The probe engine
        # indexes by tile_id, but keeping the parquet aligned makes downstream
        # audits and joins unambiguous.
        labels = labels.set_index("tile_id").loc[encoder_ids].reset_index()

        cache_checks["encoder_cache"] = {
            "path": str(encoder_path.resolve()),
            "rows": len(encoder_ids),
            "split_counts": dict(sorted(Counter(encoder_splits).items())),
            "order": "labels reordered to encoder_cache tile_ids",
        }


    if args.validate_release and split_counts(labels) != {"test": 1000, "train": 8000, "val": 1000}:
        raise ValueError(f"unexpected split counts: {split_counts(labels)}")

    parquet_path = out / "labels.parquet"
    meta = {
        "version": "EGMS-QA",
        "description": "EGMS-QA task labels assembled from task result tables",
        "labels": str(parquet_path.resolve()),
        "tasks": tasks,
        "probe_policy": {
            "fit_splits": ["train", "val"],
            "test_split": "test",
            "numeric_metric": "R2",
            "categorical_metric": "balanced_accuracy",
            "x_tasks": "metadata_only_not_token_probed",
        },
        "sources": {
            "tasks_root": str(root.resolve()),
            "encoder_cache": str(Path(args.encoder_cache).resolve()) if cache_checks else None,
        },
        "verification": {
            "rows": int(len(labels)),
            "release_contract_checked": args.validate_release,
            "columns": list(labels.columns),
            "split_counts": split_counts(labels),
            "duplicate_tile_id": int(labels["tile_id"].duplicated().sum()),
            "table_checks": table_checks,
            "cache_checks": cache_checks,
            "probe_applicable_tasks": int(sum(bool(t.get("probe_applicable")) for t in tasks)),
            "metadata_only_tasks": int(sum(not bool(t.get("probe_applicable")) for t in tasks)),
        },
    }
    meta_path = out / "labels_meta.json"
    out.mkdir(parents=True, exist_ok=True)
    check_output_paths(out)
    with parquet_path.open("xb") as label_file, meta_path.open("x", encoding="utf-8") as meta_file:
        labels.to_parquet(label_file, index=False)
        meta_file.write(json.dumps(meta, indent=2))

    print(f"wrote {parquet_path}")
    print(f"wrote {meta_path}")
    print(f"probe-applicable tasks: {meta['verification']['probe_applicable_tasks']}")
    print(f"metadata-only tasks: {meta['verification']['metadata_only_tasks']}")


if __name__ == "__main__":
    main()
