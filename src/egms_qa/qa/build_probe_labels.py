"""Build EGMS-QA delivered-task labels for token probing.

This builder treats the delivered family final tables as the source of truth.
It materializes one canonical probe target per leaf task and keeps X refusal
tasks in metadata only because they are static boundary policies, not
token-dependent targets.
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
    QA_DIR as DEFAULT_OUT,
    TASKS_DIR as DEFAULT_TASKS_DIR,
    ENCODER_TOKENS as DEFAULT_ENCODER_CACHE,
    OUTPUTS_DIR,
)

DEFAULT_PERBIN_CACHE = OUTPUTS_DIR / "perbin_scalar_tokens/perbin_scalar_tokens.pt"


TASK_SPECS: list[dict[str, Any]] = [
    # A. Observation gate
    dict(id="A11", source_family="a1", target_column="A11_global_angular_drift", label_type="numeric", name="global representation drift"),
    dict(id="A12", source_family="a1", target_column="A12_representation_stability_class", label_type="categorical", name="representation stability class"),
    dict(id="A21", source_family="a2", target_column="A21_masked_global_mse_z", label_type="numeric", name="masked reconstruction loss"),
    dict(id="A22", source_family="a2", target_column="A22_reconstruction_reliability_class", label_type="categorical", name="reconstruction reliability class"),
    dict(id="A31", source_family="a3", target_column="A31_valid_bin_fraction_8x8", label_type="numeric", name="spatial observation coverage"),
    dict(id="A32", source_family="a3", target_column="A32_spatial_coverage_class", label_type="categorical", name="spatial coverage class"),
    dict(id="A41", source_family="a4", target_column="A41_median_rmse_mm", label_type="numeric", name="median measurement noise"),
    dict(id="A42", source_family="a4", target_column="A42_noise_level_class", label_type="categorical", name="measurement noise class"),
    dict(id="A51", source_family="a5", target_column="A51_monitoring_usability_class", label_type="categorical", name="monitoring usability gate"),
    dict(id="A52", source_family="a5", target_column="A52_monitoring_usability_reason", label_type="categorical", name="monitoring usability reason"),
    # B. Motion vital signs
    dict(id="B11", source_family="b1", target_column="B11_subsidence_snr", label_type="numeric", name="average subsidence SNR"),
    dict(id="B12", source_family="b1", target_column="B12_clear_subsidence_class", label_type="categorical", name="clear average subsidence signal"),
    dict(id="B21", source_family="b2", target_column="B21_mean_velocity_mm_yr", label_type="numeric", name="mean velocity"),
    dict(id="B22", source_family="b2", target_column="B22_mean_subsidence_intensity_band", label_type="categorical", name="mean subsidence intensity band"),
    dict(id="B31", source_family="b3", target_column="B31_velocity_p10_mm_yr", label_type="numeric", name="sinking tail velocity"),
    dict(id="B32", source_family="b3", target_column="B32_velocity_p90_mm_yr", label_type="numeric", name="upper-tail velocity"),
    dict(id="B33", source_family="b3", target_column="B33_vel_abs_p90_mm_yr", label_type="numeric", name="absolute tail velocity"),
    dict(id="B34", source_family="b3", target_column="B34_uplift_protected_direction", label_type="categorical", name="uplift-protected direction"),
    dict(id="B35", source_family="b3", target_column="B35_worst_point_significance", label_type="categorical", name="worst-point significance"),
    dict(id="B36", source_family="b3", target_column="B36_european_velocity_typicality", label_type="categorical", name="European velocity typicality"),
    dict(id="B41", source_family="b4", target_column="B41_acc_abs_p90", label_type="numeric", name="acceleration strength"),
    dict(id="B42", source_family="b4", target_column="B42_european_acceleration_typicality", label_type="categorical", name="European acceleration typicality"),
    dict(id="B51", source_family="b5", target_column="B51_seasonality_p90", label_type="numeric", name="seasonality strength"),
    dict(id="B61", source_family="b6", target_column="B61_monitoring_trigger", label_type="categorical", name="monitoring trigger"),
    # C. Spatial organization
    dict(id="C11", source_family="c1", target_column="C11_noise_aware_moving_fraction", label_type="numeric", name="noise-aware moving point fraction"),
    dict(id="C12", source_family="c1", target_column="C12_motion_extent_class", label_type="categorical", name="motion extent class"),
    dict(id="C13", source_family="c1", target_column="C13_moving_bin_location", label_type="categorical", name="strongest motion bin location"),
    dict(id="C21", source_family="c2", target_column="C21_spatial_concentration_score", label_type="numeric", name="spatial concentration"),
    dict(id="C22", source_family="c2", target_column="C22_spatial_concentration_class", label_type="categorical", name="spatial concentration class"),
    dict(id="C31", source_family="c3", target_column="C31_deformation_front_strength_mm_yr", label_type="numeric", name="deformation front strength"),
    dict(id="C32", source_family="c3", target_column="C32_front_location", label_type="categorical", name="deformation front location"),
    dict(id="C33", source_family="c3", target_column="C33_deformation_front_strength_class", label_type="categorical", name="deformation front strength class"),
    dict(id="C41", source_family="c4", target_column="C41_fast_tail_bin_fraction", label_type="numeric", name="fast-tail bin fraction"),
    dict(id="C42", source_family="c4", target_column="C42_fast_tail_extent_class", label_type="categorical", name="fast-tail extent class"),
    dict(id="C51", source_family="c5", target_column="C51_monitoring_priority", label_type="categorical", name="monitoring priority"),
    dict(id="C52", source_family="c5", target_column="C52_hidden_local_risk", label_type="categorical", name="hidden local risk"),
    # D. Temporal dynamics
    dict(id="D11", source_family="d1", target_column="D11_long_term_trend_shape", label_type="categorical", name="D12/D13 p85 primitive trend shape"),
    dict(id="D12", source_family="d1", target_column="D12_curvature_strength", label_type="numeric", name="geometry curvature strength"),
    dict(id="D13", source_family="d1", target_column="D13_changepoint_strength", label_type="numeric", name="geometry changepoint strength"),
    dict(id="D14", source_family="d1", target_column="D14_dominant_changepoint_time_year", label_type="numeric", name="D13 strong changepoint time"),
    dict(id="D21", source_family="d2", target_column="D21_dominant_seasonal_peak", label_type="categorical", name="coherent dominant seasonal phase"),
    dict(id="D22", source_family="d2", target_column="D22_phase_coherence", label_type="numeric", name="seasonal phase coherence"),
    dict(id="D23", source_family="d2", target_column="D23_phase_dispersion_days", label_type="numeric", name="seasonal phase dispersion"),
    dict(id="D24", source_family="d2", target_column="D24_seasonal_amplitude_change_mm", label_type="numeric", name="seasonal amplitude change"),
    dict(id="D31", source_family="d3", target_column="D31_motion_intensification_mm_yr2", label_type="numeric", name="motion intensification"),
    dict(id="D32", source_family="d3", target_column="D32_acceleration_support_fraction", label_type="numeric", name="acceleration spatial support"),
    dict(id="D33", source_family="d3", target_column="D33_intensification_spread_mm_yr2", label_type="numeric", name="intensification spread"),
    dict(id="D34", source_family="d3", target_column="D34_intensification_hotspot_strength_mm_yr2", label_type="numeric", name="intensification hotspot strength"),
    dict(id="D35", source_family="d3", target_column="D35_intensification_hotspot_location", label_type="categorical", name="intensification hotspot location"),
    dict(id="D41", source_family="d4", target_column="D41_temporal_dominant_process", label_type="categorical", name="temporal dominant process"),
    dict(id="D42", source_family="d4", target_column="D42_temporal_evolution_archetype", label_type="categorical", name="temporal evolution archetype"),
    # S. Encoder representation understanding
    dict(id="S11", source_family="s1", target_column="S11_reference_anchor_profile", label_type="categorical", name="reference anchor profile", construct=True),
    dict(id="S12", source_family="s1", target_column="S12_reference_anchor_distance", label_type="numeric", name="nearest reference anchor distance", construct=True),
    dict(id="S13", source_family="s1", target_column="S13_reference_anchor_margin", label_type="numeric", name="reference anchor margin", construct=True),
    dict(id="S14", source_family="s1", target_column="S14_reference_assignment_status", label_type="categorical", name="reference assignment status", construct=True),
    dict(id="S15", source_family="s1", target_column="S15_reference_anchor_profile_description", label_type="categorical", name="reference anchor profile description", construct=True),
    dict(id="S21", source_family="s2", target_column="S21_local_isolation_score", label_type="numeric", name="local isolation score", construct=True),
    dict(id="S22", source_family="s2", target_column="S22_representation_rarity_class", label_type="categorical", name="representation rarity class", construct=True),
    dict(id="S31", source_family="s3", target_column="S31_representation_monitoring_rarity_gap_p", label_type="numeric", name="representation-monitoring rarity gap", construct=True),
    dict(id="S32", source_family="s3", target_column="S32_representation_monitoring_rarity_relation", label_type="categorical", name="representation-monitoring rarity relation", construct=True),
    dict(id="S33", source_family="s3", target_column="S33_monitoring_distinctive_dimension", label_type="categorical", name="monitoring distinctive dimension", construct=True),
    dict(id="S41", source_family="s4", target_column="S41_encoder_perceived_local_structure_strength", label_type="numeric", name="encoder-perceived local structure strength", construct=True),
    dict(id="S42", source_family="s4", target_column="S42_encoder_perceived_local_structure_class", label_type="categorical", name="encoder-perceived local structure class", construct=True),
    dict(id="S43", source_family="s4", target_column="S43_encoder_perceived_local_structure_concentration", label_type="numeric", name="encoder-perceived local structure concentration", construct=True),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--tasks-root", default=str(DEFAULT_TASKS_DIR))
    p.add_argument("--out-dir", default=str(DEFAULT_OUT))
    p.add_argument("--encoder-cache", default=str(DEFAULT_ENCODER_CACHE))
    p.add_argument("--perbin-cache", default=str(DEFAULT_PERBIN_CACHE))
    p.add_argument("--skip-cache-validation", action="store_true")
    return p.parse_args()


def read_family(root: Path, family: str) -> pd.DataFrame:
    path = root / family / f"{family}_final_table.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    if "tile_id" not in df.columns or "split" not in df.columns:
        raise ValueError(f"{path} must contain tile_id and split")
    if df["tile_id"].duplicated().any():
        dupes = df.loc[df["tile_id"].duplicated(), "tile_id"].head().tolist()
        raise ValueError(f"{path} has duplicate tile_id values: {dupes}")
    if len(df) != 10000:
        raise ValueError(f"{path} expected 10000 rows, found {len(df)}")
    return df


def align_family_to_base(table: pd.DataFrame, base_index: pd.MultiIndex, family: str) -> pd.DataFrame:
    keyed = table.copy()
    keyed["tile_id"] = keyed["tile_id"].astype(str)
    keyed["split"] = keyed["split"].astype(str)
    keyed = keyed.set_index(["tile_id", "split"], verify_integrity=True)
    missing = base_index.difference(keyed.index)
    extra = keyed.index.difference(base_index)
    if len(missing) or len(extra):
        raise ValueError(
            f"{family} tile_id/split index mismatch; "
            f"missing={list(missing[:5])} extra={list(extra[:5])}"
        )
    return keyed.loc[base_index].reset_index()


def normalize_series(s: pd.Series, label_type: str) -> pd.Series:
    if label_type == "numeric":
        return pd.to_numeric(s, errors="coerce")
    out = s.astype("string").str.strip()
    out = out.mask(out.isin(["", "nan", "None", "<NA>"]))
    return out


def load_cache_ids(path: Path) -> tuple[list[str], list[str]]:
    import torch

    cache = torch.load(path, map_location="cpu", weights_only=False)
    ids = [str(t) for t in cache["tile_ids"]]
    splits = [str(s) for s in cache.get("splits", [""] * len(ids))]
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
            "label_std": float(np.nanstd(v.to_numpy(dtype=float))),
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
    out.mkdir(parents=True, exist_ok=True)

    families = sorted({spec["source_family"] for spec in TASK_SPECS})
    tables = {family: read_family(root, family) for family in families}

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
    if not args.skip_cache_validation:
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

        perbin_path = Path(args.perbin_cache)
        perbin_ids, perbin_splits = load_cache_ids(perbin_path)
        if perbin_ids != labels["tile_id"].astype(str).tolist():
            raise ValueError("perbin_cache tile_ids do not match encoder-ordered EGMS-QA labels")
        if any(perbin_splits) and perbin_splits != labels["split"].astype(str).tolist():
            raise ValueError("perbin_cache splits do not match encoder-ordered EGMS-QA labels")
        cache_checks["perbin_cache"] = {
            "path": str(perbin_path.resolve()),
            "rows": len(perbin_ids),
            "split_counts": dict(sorted(Counter(perbin_splits).items())),
            "order": "matches encoder-ordered labels",
        }

    if split_counts(labels) != {"test": 1000, "train": 8000, "val": 1000}:
        raise ValueError(f"unexpected split counts: {split_counts(labels)}")

    parquet_path = out / "labels.parquet"
    labels.to_parquet(parquet_path, index=False)
    meta = {
        "version": "EGMS-QA",
        "description": "Canonical delivered EGMS-QA A-X task labels for encoder-token probing",
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
            "encoder_cache": str(Path(args.encoder_cache).resolve()),
            "perbin_cache": str(Path(args.perbin_cache).resolve()),
        },
        "verification": {
            "rows": int(len(labels)),
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
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"wrote {parquet_path}")
    print(f"wrote {meta_path}")
    print(f"probe-applicable tasks: {meta['verification']['probe_applicable_tasks']}")
    print(f"metadata-only tasks: {meta['verification']['metadata_only_tasks']}")


if __name__ == "__main__":
    main()
