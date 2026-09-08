"""Frozen task reference state, fitted from the released training population."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, normalize

SCHEMA = "egms-qa-task-reference-1"


def fit_space(values, train_mask):
    scaler = StandardScaler()
    train = scaler.fit_transform(values[train_mask])
    pca = PCA(n_components=25, random_state=0).fit(train)
    return {"scaler": scaler, "pca": pca}


def transform_space(values, state):
    return normalize(state["pca"].transform(state["scaler"].transform(values)), norm="l2").astype(np.float32)


@lru_cache(maxsize=2)
def load_reference(path: str | Path):
    """Read a locally generated reference state produced by the task runner."""
    state = joblib.load(path)
    if state.get("schema") != SCHEMA:
        raise ValueError("unsupported task reference state")
    return state


def reference_table(path, group):
    return None if path is None else load_reference(path)["tables"][group]


def build_reference(tables_root: Path, token_cache: Path, output: Path):
    from .tables import read_family
    from .tasks.s1 import s1_compute as s1
    from .tasks.s2 import s2_compute as s2

    groups = [f"{family}{i}" for family, count in [("a", 5), ("b", 6), ("c", 5), ("d", 4), ("s", 4)] for i in range(1, count + 1)]
    tables = {g: read_family(tables_root, g) for g in groups}
    values, meta = s1.load_cls(token_cache)
    expected_keys = set(zip(meta["tile_id"].astype(str), meta["split"].astype(str)))
    for group, table in tables.items():
        if set(zip(table["tile_id"].astype(str), table["split"].astype(str))) != expected_keys:
            raise ValueError(f"reference {group} IDs or splits differ from reference tokens")
    train = meta["split"].eq("train").to_numpy()
    if meta["split"].value_counts().to_dict() != {"train": 8000, "val": 1000, "test": 1000}:
        raise ValueError("reference tokens must contain the released 8,000/1,000/1,000 tiles")
    space = fit_space(values, train)
    features = transform_space(values, space)
    anchors, details, _ = s1.fit_reference_anchors(features, train, meta["tile_id"])
    assignments = s1.assign_all(features, anchors, details)
    status = s1.fit_status_reference(assignments, train)
    scores = s2.compute_knn_isolation(features, meta, train, k=20, n_jobs=1)
    state = {"schema": SCHEMA, "tables": tables, "space": space,
             "s1": {"anchors": anchors, "details": details, "status": status},
             "s2": {"features": features[train], "tile_ids": meta.loc[train, "tile_id"].tolist(),
                    "scores": scores.loc[train].copy()},
             "reference_tiles": len(meta), "reference_train_tiles": int(train.sum())}
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(state, output)
