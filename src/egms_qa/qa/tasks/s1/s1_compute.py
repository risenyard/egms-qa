"""Compute EGMS-QA S1 reference-anchor attributes.

S1 uses train-defined CLS representation anchors:

1. Fit StandardScaler and PCA25 on train CLS embeddings only.
2. L2-normalize PCA features.
3. Fit HDBSCAN(min_cluster_size=50, min_samples=80) on train features.
4. Use each train dense-core cluster medoid as a reference anchor.
5. Assign every tile to its nearest train anchor and compute distance/margin.
6. Fit a train-only 2D Gaussian mixture over [distance, margin]. BIC selects the
   component count. Components are merged into the S14 status labels.

The output table contains S11-S15 plus an auxiliary reference_anchor_id.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.cluster import HDBSCAN
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler, normalize


ROOT = Path(".")
DEFAULT_TOKEN_CACHE = ROOT / "data/encoder/tokens/encoder_tokens_10k.pt"
DEFAULT_OUT_DIR = ROOT / "outputs/tasks/s1"

PROFILE_MAP = {
    0: {
        "profile": "mixed_acceleration_complex_trend_reference",
        "description": "large mixed dynamic reference with elevated acceleration and complex trend behavior",
    },
    1: {
        "profile": "spring_trend_acceleration_reference",
        "description": "spring-associated trend and acceleration mixed reference",
    },
    2: {
        "profile": "coherent_autumn_seasonal_reference",
        "description": "compact autumn-seasonal reference with high phase coherence",
    },
    3: {
        "profile": "extreme_localized_deformation_front_reference",
        "description": "small extreme reference with strong localized deformation, front strength, and fast-tail extent",
    },
    4: {
        "profile": "stable_low_activity_background_reference",
        "description": "large low-activity stable background reference with low velocity and acceleration",
    },
    5: {
        "profile": "summer_trend_seasonal_mixed_reference",
        "description": "summer-associated trend-seasonal mixed reference with relatively diffuse spatial structure",
    },
}


def load_cls(token_cache: Path) -> tuple[np.ndarray, pd.DataFrame]:
    obj = torch.load(token_cache, map_location="cpu")
    cls = obj["spatial_tokens"][:, 0, :].float().numpy().astype(np.float32)
    meta = pd.DataFrame({"tile_id": obj["tile_ids"], "split": obj["splits"]})
    return cls, meta


def pca25_features(cls: np.ndarray, train_mask: np.ndarray) -> np.ndarray:
    scaler = StandardScaler()
    z_train = scaler.fit_transform(cls[train_mask])
    z_all = scaler.transform(cls)
    pca = PCA(n_components=25, random_state=0)
    pca.fit(z_train)
    return normalize(pca.transform(z_all), norm="l2").astype(np.float32)


def cosine_distance_to_anchors(x: np.ndarray, anchors: np.ndarray) -> np.ndarray:
    sim = np.clip(normalize(x, norm="l2") @ normalize(anchors, norm="l2").T, -1.0, 1.0)
    return 1.0 - sim


def medoid_index(x: np.ndarray, indices: np.ndarray) -> int:
    if len(indices) == 1:
        return int(indices[0])
    sub = normalize(x[indices], norm="l2")
    dist_sum = (1.0 - np.clip(sub @ sub.T, -1.0, 1.0)).sum(axis=1)
    return int(indices[int(np.argmin(dist_sum))])


def fit_reference_anchors(
    x_all: np.ndarray,
    train_mask: np.ndarray,
    tile_ids: pd.Series,
) -> tuple[np.ndarray, pd.DataFrame, np.ndarray]:
    x_train = x_all[train_mask]
    global_train_indices = np.flatnonzero(train_mask)
    clusterer = HDBSCAN(
        min_cluster_size=50,
        min_samples=80,
        metric="euclidean",
        cluster_selection_method="eom",
    )
    train_labels = clusterer.fit_predict(x_train).astype(np.int32)
    anchor_rows = []
    anchors = []

    for label in sorted(int(v) for v in np.unique(train_labels) if v >= 0):
        members = np.flatnonzero(train_labels == label)
        medoid = medoid_index(x_train, members)
        anchor = x_train[medoid]
        member_dist = cosine_distance_to_anchors(x_train[members], anchor.reshape(1, -1)).ravel()
        global_medoid = int(global_train_indices[medoid])
        anchor_rows.append(
            {
                "reference_anchor_id": label,
                "train_core_size": int(len(members)),
                "train_core_fraction": float(len(members) / len(x_train)),
                "medoid_global_index": global_medoid,
                "medoid_tile_id": str(tile_ids.iloc[global_medoid]),
                "within_distance_median": float(np.median(member_dist)),
                "within_distance_q75": float(np.quantile(member_dist, 0.75)),
                "within_distance_q90": float(np.quantile(member_dist, 0.90)),
                "within_distance_q95": float(np.quantile(member_dist, 0.95)),
            }
        )
        anchors.append(anchor)

    details = pd.DataFrame(anchor_rows)
    if len(anchors) != 6:
        raise RuntimeError(f"Expected 6 S1 reference anchors, got {len(anchors)}")

    anchors_arr = np.vstack(anchors).astype(np.float32)
    aa = cosine_distance_to_anchors(anchors_arr, anchors_arr)
    np.fill_diagonal(aa, np.nan)
    details["nearest_anchor_distance"] = np.nanmin(aa, axis=1)
    return anchors_arr, details, train_labels


def assign_all(x_all: np.ndarray, anchors: np.ndarray, details: pd.DataFrame) -> pd.DataFrame:
    distances = cosine_distance_to_anchors(x_all, anchors)
    order = np.argsort(distances, axis=1)
    nearest_col = order[:, 0]
    second_col = order[:, 1]
    nearest_distance = distances[np.arange(len(x_all)), nearest_col]
    second_distance = distances[np.arange(len(x_all)), second_col]
    margin = second_distance - nearest_distance
    anchor_ids = details["reference_anchor_id"].to_numpy(dtype=np.int32)
    return pd.DataFrame(
        {
            "reference_anchor_id": anchor_ids[nearest_col],
            "S12_reference_anchor_distance": nearest_distance.astype(float),
            "S13_reference_anchor_margin": margin.astype(float),
        }
    )


def fit_gmm_bic(x: np.ndarray, max_components: int = 6) -> tuple[GaussianMixture, pd.DataFrame]:
    rows = []
    models = []
    for k in range(1, max_components + 1):
        model = GaussianMixture(
            n_components=k,
            covariance_type="full",
            random_state=0,
            n_init=20,
            reg_covar=1e-6,
        )
        model.fit(x)
        rows.append({"n_components": k, "bic": float(model.bic(x)), "aic": float(model.aic(x))})
        models.append(model)
    metrics = pd.DataFrame(rows)
    return models[int(metrics["bic"].idxmin())], metrics


def gmm2d_status(assignments: pd.DataFrame, train_mask: np.ndarray) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    raw = assignments[["S12_reference_anchor_distance", "S13_reference_anchor_margin"]].to_numpy(dtype=float)
    scaler = StandardScaler().fit(raw[train_mask])
    train_z = scaler.transform(raw[train_mask])
    all_z = scaler.transform(raw)
    model, bic = fit_gmm_bic(train_z, max_components=6)
    train_comp = model.predict(train_z)
    all_comp = model.predict(all_z)

    rows = []
    train_raw = raw[train_mask]
    for comp in sorted(np.unique(train_comp)):
        vals = train_raw[train_comp == comp]
        rows.append(
            {
                "component": int(comp),
                "train_count": int(len(vals)),
                "train_fraction": float(len(vals) / len(train_raw)),
                "distance_median": float(np.median(vals[:, 0])),
                "distance_q10": float(np.quantile(vals[:, 0], 0.10)),
                "distance_q90": float(np.quantile(vals[:, 0], 0.90)),
                "margin_median": float(np.median(vals[:, 1])),
                "margin_q10": float(np.quantile(vals[:, 1], 0.10)),
                "margin_q90": float(np.quantile(vals[:, 1], 0.90)),
            }
        )
    comp_df = pd.DataFrame(rows)

    labels = {int(comp): "transition_or_weakly_anchored" for comp in comp_df["component"]}
    strong_idx = ((-comp_df["margin_median"]) + comp_df["distance_median"]).idxmin()
    strong_comp = int(comp_df.loc[strong_idx, "component"])
    far_comp = int(comp_df.sort_values("distance_median", ascending=False).iloc[0]["component"])
    low_margin_comp = int(comp_df.sort_values("margin_median", ascending=True).iloc[0]["component"])
    labels[strong_comp] = "strongly_anchored"
    labels[far_comp] = "far_or_ambiguous_from_reference_anchors"
    labels[low_margin_comp] = "far_or_ambiguous_from_reference_anchors"

    comp_df["S14_reference_assignment_status"] = comp_df["component"].astype(int).map(labels)
    status = np.asarray([labels[int(comp)] for comp in all_comp], dtype=object)
    return status, comp_df, bic


def build_outputs(token_cache: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    cls, meta = load_cls(token_cache)
    train_mask = meta["split"].astype(str).to_numpy() == "train"
    x_all = pca25_features(cls, train_mask)
    anchors, anchor_details, _ = fit_reference_anchors(x_all, train_mask, meta["tile_id"])
    assignments = assign_all(x_all, anchors, anchor_details)
    status, gmm_components, gmm_bic = gmm2d_status(assignments, train_mask)

    profile = assignments["reference_anchor_id"].map(lambda x: PROFILE_MAP[int(x)]["profile"])
    description = assignments["reference_anchor_id"].map(lambda x: PROFILE_MAP[int(x)]["description"])
    final = pd.DataFrame(
        {
            "tile_id": meta["tile_id"].astype(str),
            "split": meta["split"].astype(str),
            "reference_anchor_id": assignments["reference_anchor_id"].astype(int),
            "S11_reference_anchor_profile": profile,
            "S12_reference_anchor_distance": assignments["S12_reference_anchor_distance"],
            "S13_reference_anchor_margin": assignments["S13_reference_anchor_margin"],
            "S14_reference_assignment_status": status,
            "S15_reference_anchor_profile_description": description,
        }
    )
    final.to_csv(out_dir / "s1_final_table.csv", index=False)

    profile_map = anchor_details.copy()
    profile_map["S11_reference_anchor_profile"] = profile_map["reference_anchor_id"].map(
        lambda x: PROFILE_MAP[int(x)]["profile"]
    )
    profile_map["S15_reference_anchor_profile_description"] = profile_map["reference_anchor_id"].map(
        lambda x: PROFILE_MAP[int(x)]["description"]
    )
    profile_map.to_csv(out_dir / "s1_anchor_profile_map.csv", index=False)
    gmm_components.to_csv(out_dir / "s1_s14_gmm2d_components.csv", index=False)
    gmm_bic.to_csv(out_dir / "s1_s14_gmm2d_bic.csv", index=False)

    status_counts = (
        final["S14_reference_assignment_status"].value_counts().rename_axis("label").reset_index(name="count")
    )
    status_counts["fraction"] = status_counts["count"] / len(final)
    anchor_counts = (
        final.groupby(["reference_anchor_id", "S11_reference_anchor_profile"])
        .size()
        .rename("count")
        .reset_index()
        .sort_values("reference_anchor_id")
    )
    anchor_counts["fraction"] = anchor_counts["count"] / len(final)
    status_counts.to_csv(out_dir / "s1_s14_counts.csv", index=False)
    anchor_counts.to_csv(out_dir / "s1_anchor_counts.csv", index=False)

    summary = {
        "token_cache": str(token_cache),
        "n_tiles": int(len(final)),
        "train_tiles": int(train_mask.sum()),
        "selected_algorithm": "StandardScaler(train) + PCA25(train) + L2 + HDBSCAN(min_cluster_size=50,min_samples=80) on train CLS",
        "s14_algorithm": "train-only 2D GaussianMixture over [S12 distance, S13 margin], BIC-selected k=6, merged into three states",
        "status_counts": status_counts.to_dict(orient="records"),
        "anchor_counts": anchor_counts.to_dict(orient="records"),
    }
    (out_dir / "s1_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    plot(final, out_dir)


def plot(final: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    for label, group in final.groupby("S14_reference_assignment_status"):
        ax.scatter(
            group["S12_reference_anchor_distance"],
            group["S13_reference_anchor_margin"],
            s=7,
            alpha=0.35,
            label=label,
        )
    ax.set_xlabel("S12 nearest-anchor distance")
    ax.set_ylabel("S13 anchor margin")
    ax.set_title("S14 distribution-driven assignment status")
    ax.legend(fontsize=7)

    ax = axes[1]
    counts = final["reference_anchor_id"].value_counts().sort_index()
    ax.bar([str(v) for v in counts.index], counts.values)
    ax.set_xlabel("reference anchor id")
    ax.set_ylabel("tile count")
    ax.set_title("S11 nearest reference-anchor distribution")
    fig.tight_layout()
    fig.savefig(out_dir / "s1_distribution.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--token-cache", type=Path, default=DEFAULT_TOKEN_CACHE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    build_outputs(args.token_cache, args.out_dir)


if __name__ == "__main__":
    main()
