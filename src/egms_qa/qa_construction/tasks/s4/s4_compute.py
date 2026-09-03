"""Compute S4 encoder-perceived local spatial structure tasks.

S41 is a single-tile encoder construct. It uses only the current tile's valid
patch tokens and measures how much local patch representations deviate from
their tile-level patch centroid. It is interpreted as the strength of
encoder-perceived local spatial structure, not as a direct physical label.

S42 is the train-distribution class derived from S41.

S43 is a continuous concentration construct derived from the same valid patch
tokens. It asks whether the local structure measured by S41 is spread across
many patches or concentrated in fewer patches.
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


ROOT = Path(".")
DEFAULT_TOKEN_CACHE = ROOT / "data/encoder/tokens/encoder_tokens_10k.pt"
DEFAULT_OUT_DIR = ROOT / "outputs/tasks/s4"
S41_COL = "S41_encoder_perceived_local_structure_strength"
S42_COL = "S42_encoder_perceived_local_structure_class"
S43_COL = "S43_encoder_perceived_local_structure_concentration"
S42_LABELS = [
    "spatially_coherent",
    "weak_local_structure",
    "clear_local_structure",
    "strong_local_structure",
]


def load_tokens(token_cache: Path) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    obj = torch.load(token_cache, map_location="cpu")
    tokens = obj["spatial_tokens"].float().numpy().astype(np.float32)
    token_mask = obj["token_mask"].numpy().astype(bool)
    meta = pd.DataFrame({"tile_id": obj["tile_ids"], "split": obj["splits"]})
    return tokens, token_mask, meta


def gini_nonnegative(values: np.ndarray) -> float:
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if len(x) == 0 or float(x.sum()) <= 0:
        return np.nan
    x = np.sort(x)
    n = len(x)
    weights = 2 * np.arange(1, n + 1) - n - 1
    return float(np.dot(weights, x) / (n * x.sum()))


def compute_s41_s43(
    tokens: np.ndarray,
    token_mask: np.ndarray,
    meta: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    patch_tokens = tokens[:, 1:, :]
    patch_mask = token_mask[:, 1:]

    rows = []
    diagnostics = []
    for i in range(len(meta)):
        valid = patch_mask[i]
        x = patch_tokens[i, valid]
        n = int(len(x))
        if n < 2:
            dispersion = np.nan
            concentration = np.nan
        else:
            centroid = x.mean(axis=0, keepdims=True)
            residual = x - centroid
            residual_norm = np.linalg.norm(residual, axis=1)
            residual_rms = float(np.sqrt(np.mean(np.sum(residual * residual, axis=1))))
            token_rms = float(np.sqrt(np.mean(np.sum(x * x, axis=1))))
            dispersion = residual_rms / max(token_rms, 1e-12)
            concentration = gini_nonnegative(residual_norm)

        rows.append(
            {
                "tile_id": meta.loc[i, "tile_id"],
                "split": meta.loc[i, "split"],
                S41_COL: dispersion,
                S43_COL: concentration,
            }
        )
        diagnostics.append(
            {
                "tile_id": meta.loc[i, "tile_id"],
                "split": meta.loc[i, "split"],
                "valid_patch_count": n,
                "valid_patch_fraction": n / 64.0,
                S41_COL: dispersion,
                S43_COL: concentration,
            }
        )

    return pd.DataFrame(rows), pd.DataFrame(diagnostics)


def add_s42(final: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    train_values = final.loc[final["split"].astype(str) == "train", S41_COL].to_numpy(dtype=np.float64)
    p50, p90, p95 = np.quantile(train_values, [0.50, 0.90, 0.95])
    values = final[S41_COL].to_numpy(dtype=np.float64)
    labels = np.full(len(final), "spatially_coherent", dtype=object)
    labels[values > p50] = "weak_local_structure"
    labels[values > p90] = "clear_local_structure"
    labels[values > p95] = "strong_local_structure"

    out = final.copy()
    out[S42_COL] = labels
    thresholds = {
        "spatially_coherent_max": float(p50),
        "weak_local_structure_max": float(p90),
        "clear_local_structure_max": float(p95),
    }
    return out, thresholds


def quantiles(values: pd.Series) -> dict[str, float]:
    arr = values.dropna().to_numpy(dtype=np.float64)
    qs = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    return {f"p{int(q * 100):02d}": float(np.quantile(arr, q)) for q in qs}


def summarize(
    final: pd.DataFrame,
    diagnostics: pd.DataFrame,
    out_dir: Path,
    token_cache: Path,
    s42_thresholds: dict[str, float],
) -> None:
    values = final[S41_COL]
    s42_counts = final[S42_COL].value_counts().reindex(S42_LABELS, fill_value=0)
    split_s42_counts = (
        final.groupby("split")[S42_COL]
        .value_counts()
        .unstack(fill_value=0)
        .reindex(columns=S42_LABELS, fill_value=0)
    )
    summary = {
        "family": "S4",
        "computed_tasks": [S41_COL, S42_COL, S43_COL],
        "n_tiles": int(len(final)),
        "n_missing": final[[S41_COL, S42_COL, S43_COL]].isna().sum().astype(int).to_dict(),
        "single_tile_inputs": "current tile valid patch tokens only; no CLS, no geographic neighbors, no A/B/C/D labels",
        "token_cache": str(token_cache),
        "s41_formula": "RMS(patch_token - patch_centroid) / RMS(patch_token)",
        "s41_interpretation": "Strength of encoder-perceived local spatial structure inside the tile.",
        "s42_algorithm": "train-only S41 p50/p90/p95 thresholds; corpus-relative right-tail local-structure labels",
        "s43_formula": "Gini(||patch_token_i - patch_centroid|| over valid patch tokens)",
        "s43_interpretation": "Higher values mean encoder-perceived local structure is concentrated in fewer patches.",
        "s42_thresholds": s42_thresholds,
        "s42_class_counts": s42_counts.astype(int).to_dict(),
        "s42_class_fractions": (s42_counts / len(final)).astype(float).to_dict(),
        "s42_class_counts_by_split": {
            str(idx): row.astype(int).to_dict() for idx, row in split_s42_counts.iterrows()
        },
        "valid_patch_count": quantiles(diagnostics["valid_patch_count"]),
        "valid_patch_fraction": quantiles(diagnostics["valid_patch_fraction"]),
        "s41_quantiles": quantiles(values),
        "s41_mean": float(values.mean()),
        "s41_std": float(values.std(ddof=1)),
        "s43_quantiles": quantiles(final[S43_COL]),
        "s43_mean": float(final[S43_COL].mean()),
        "s43_std": float(final[S43_COL].std(ddof=1)),
        "s43_spearman": {
            "vs_s41": float(final[[S43_COL, S41_COL]].corr(method="spearman").iloc[0, 1]),
            "vs_valid_patch_count": float(
                diagnostics[[S43_COL, "valid_patch_count"]].corr(method="spearman").iloc[0, 1]
            ),
        },
        "split_quantiles": {},
    }
    for split, group in final.groupby("split"):
        summary["split_quantiles"][str(split)] = {
            "s41": quantiles(group[S41_COL]),
            "s43": quantiles(group[S43_COL]),
        }
    (out_dir / "s4_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def plot_distribution(final: pd.DataFrame, diagnostics: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.ravel()

    axes[0].hist(final[S41_COL], bins=80, color="#4c78a8", alpha=0.85)
    axes[0].set_title("S41 local structure strength")
    axes[0].set_xlabel("residual RMS ratio")
    axes[0].set_ylabel("tile count")

    axes[1].scatter(
        diagnostics["valid_patch_fraction"],
        diagnostics[S41_COL],
        s=6,
        alpha=0.25,
        color="#54a24b",
    )
    axes[1].set_title("S41 vs valid patch fraction")
    axes[1].set_xlabel("valid patch fraction")
    axes[1].set_ylabel("S41")

    counts = final[S42_COL].value_counts().reindex(S42_LABELS, fill_value=0)
    axes[2].bar(counts.index, counts.values, color="#f58518")
    axes[2].set_title("S42 local structure class")
    axes[2].set_ylabel("tile count")
    axes[2].tick_params(axis="x", rotation=25)

    axes[3].hist(final[S43_COL], bins=80, color="#b279a2", alpha=0.85)
    axes[3].set_title("S43 local structure concentration")
    axes[3].set_xlabel("Gini of patch residual norms")
    axes[3].set_ylabel("tile count")

    fig.tight_layout()
    fig.savefig(out_dir / "s4_distribution.png", dpi=180)
    plt.close(fig)


def build_outputs(token_cache: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    tokens, token_mask, meta = load_tokens(token_cache)
    final, diagnostics = compute_s41_s43(tokens, token_mask, meta)
    final, s42_thresholds = add_s42(final)
    final = final[["tile_id", "split", S41_COL, S42_COL, S43_COL]]
    final.to_csv(out_dir / "s4_final_table.csv", index=False)
    summarize(final, diagnostics, out_dir, token_cache, s42_thresholds)
    plot_distribution(final, diagnostics, out_dir)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--token-cache", type=Path, default=DEFAULT_TOKEN_CACHE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    build_outputs(args.token_cache, args.out_dir)


if __name__ == "__main__":
    main()
