"""Compute EGMS-QA S2 local representation support attributes.

S21 is a continuous local isolation score in the same representation coordinate
system as S1:

1. Extract EGMS encoder CLS embeddings.
2. Fit StandardScaler and PCA25 on train CLS embeddings only.
3. L2-normalize PCA features.
4. Use train tiles as the reference library.
5. For every tile, compute the mean cosine distance to its nearest k train
   neighbors. For train queries, the tile itself is excluded.

Higher S21 means the tile is more isolated from the train reference manifold.
S22 turns S21 into a corpus-relative rarity class using train p75/p95/p99.
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
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler, normalize


ROOT = Path(".")
DEFAULT_TOKEN_CACHE = ROOT / "data/encoder/tokens/egms_tokens_10k.pt"
DEFAULT_OUT_DIR = ROOT / "outputs/tasks/s2"
S22_LABELS = ("common", "unusual", "rare", "extreme")


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


def compute_knn_isolation(
    x_all: np.ndarray,
    meta: pd.DataFrame,
    train_mask: np.ndarray,
    k: int,
    n_jobs: int,
) -> pd.DataFrame:
    if k < 1:
        raise ValueError("k must be >= 1")

    train_indices = np.flatnonzero(train_mask)
    x_train = x_all[train_indices]
    if len(train_indices) <= k:
        raise ValueError(f"Need more than k train samples, got {len(train_indices)} train samples for k={k}")

    train_local_by_global = {int(global_idx): int(local_idx) for local_idx, global_idx in enumerate(train_indices)}
    nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine", algorithm="brute", n_jobs=n_jobs)
    nn.fit(x_train)
    distances, indices = nn.kneighbors(x_all, return_distance=True)

    rows = []
    tile_ids = meta["tile_id"].astype(str).to_numpy()
    splits = meta["split"].astype(str).to_numpy()
    for global_idx, (dist_row, idx_row) in enumerate(zip(distances, indices)):
        local_self = train_local_by_global.get(global_idx)
        if local_self is not None:
            keep = idx_row != local_self
            dist_kept = dist_row[keep][:k]
        else:
            dist_kept = dist_row[:k]

        if len(dist_kept) != k:
            raise RuntimeError(f"Could not find {k} non-self train neighbors for row {global_idx}")

        rows.append(
            {
                "tile_id": tile_ids[global_idx],
                "split": splits[global_idx],
                "S21_local_isolation_score": float(np.mean(dist_kept)),
            }
        )

    return pd.DataFrame(rows)


def add_s22_rarity(final: pd.DataFrame, train_mask: np.ndarray) -> tuple[pd.DataFrame, dict[str, float]]:
    train_values = final.loc[train_mask, "S21_local_isolation_score"].to_numpy(dtype=float)
    p75, p95, p99 = np.quantile(train_values, [0.75, 0.95, 0.99])
    values = final["S21_local_isolation_score"].to_numpy(dtype=float)
    labels = np.full(len(final), "common", dtype=object)
    labels[values > p75] = "unusual"
    labels[values > p95] = "rare"
    labels[values > p99] = "extreme"
    out = final.copy()
    out["S22_representation_rarity_class"] = labels
    thresholds = {
        "common_unusual": float(p75),
        "unusual_rare": float(p95),
        "rare_extreme": float(p99),
    }
    return out, thresholds


def summarize(final: pd.DataFrame, train_mask: np.ndarray, out_dir: Path, k: int) -> None:
    values = final["S21_local_isolation_score"].to_numpy(dtype=float)
    train_values = values[train_mask]
    quantiles = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    _, s22_thresholds = add_s22_rarity(final[["tile_id", "split", "S21_local_isolation_score"]], train_mask)
    s22_counts = (
        final["S22_representation_rarity_class"]
        .value_counts()
        .reindex(S22_LABELS, fill_value=0)
        .rename_axis("label")
        .reset_index(name="count")
    )
    s22_counts["fraction"] = s22_counts["count"] / len(final)

    summary = {
        "family": "S2",
        "computed_tasks": ["S21_local_isolation_score", "S22_representation_rarity_class"],
        "n_tiles": int(len(final)),
        "train_tiles": int(train_mask.sum()),
        "algorithm": "train-only StandardScaler + PCA25 + L2 CLS features; S21 = mean cosine distance to nearest k train neighbors, excluding self for train queries",
        "k": int(k),
        "s22_algorithm": "train-only p75/p95/p99 thresholds on S21; corpus-relative rarity tail labels",
        "s22_thresholds": s22_thresholds,
        "s22_counts": s22_counts.to_dict(orient="records"),
        "all_quantiles": {str(q): float(np.quantile(values, q)) for q in quantiles},
        "train_quantiles": {str(q): float(np.quantile(train_values, q)) for q in quantiles},
        "split_quantiles": {},
    }
    for split, group in final.groupby("split"):
        split_values = group["S21_local_isolation_score"].to_numpy(dtype=float)
        summary["split_quantiles"][str(split)] = {str(q): float(np.quantile(split_values, q)) for q in quantiles}
    (out_dir / "s2_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].hist(train_values, bins=80, alpha=0.85, color="#4c78a8", label="train")
    axes[0].hist(values, bins=80, alpha=0.35, color="#f58518", label="all")
    for threshold, label in zip(s22_thresholds.values(), ["p75", "p95", "p99"]):
        axes[0].axvline(threshold, linestyle="--", linewidth=1.2, label=label)
    axes[0].set_title("S21 local isolation distribution")
    axes[0].set_xlabel("mean cosine distance to k train neighbors")
    axes[0].set_ylabel("tile count")
    axes[0].legend()

    axes[1].bar(s22_counts["label"], s22_counts["count"], color=["#4c78a8", "#f58518", "#54a24b", "#e45756"])
    axes[1].set_title("S22 representation rarity class")
    axes[1].set_ylabel("tile count")
    axes[1].tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(out_dir / "s2_distribution.png", dpi=180)
    plt.close(fig)


def build_outputs(token_cache: Path, out_dir: Path, k: int, n_jobs: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    cls, meta = load_cls(token_cache)
    train_mask = meta["split"].astype(str).to_numpy() == "train"
    x_all = pca25_features(cls, train_mask)
    final = compute_knn_isolation(x_all, meta, train_mask, k=k, n_jobs=n_jobs)
    final, _ = add_s22_rarity(final, train_mask)
    final.to_csv(out_dir / "s2_final_table.csv", index=False)
    summarize(final, train_mask, out_dir, k=k)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--token-cache", type=Path, default=DEFAULT_TOKEN_CACHE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--n-jobs", type=int, default=8)
    args = parser.parse_args()
    build_outputs(args.token_cache, args.out_dir, k=args.k, n_jobs=args.n_jobs)


if __name__ == "__main__":
    main()
