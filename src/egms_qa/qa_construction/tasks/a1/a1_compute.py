"""A11/A12 severe global representation instability outputs for EGMS-QA.

A11 asks one narrow question:
  Does the encoder's global tile representation drift under severe point loss?

For each tile:
  1. Use the cached full-tile CLS token as the reference.
  2. Keep a fixed severe fraction of points, by default 20%.
  3. Re-run the frozen encoder for several random seeds.
  4. Compute angular drift between full CLS and subsampled CLS:

       drift = arccos(cosine(CLS_full, CLS_subsample)) / pi

The tile-level target is the mean drift across seeds. Lower is more stable.
No coverage, patch-token, scalar-probe, cluster, or manual threshold enters A11.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from egms_encoder.checkpoint import load_encoder_checkpoint, load_normalization
from egms_qa.paths import ENCODER_CKPT, ENCODER_CONFIG, ENCODER_NORMALIZATION


ROOT = Path(".")
ENCODER_DATA = ROOT / "data/encoder"

CKPT = ENCODER_CKPT
MODEL_CONFIG = ENCODER_CONFIG
NORMALIZATION = ENCODER_NORMALIZATION
MANIFEST = ENCODER_DATA / "manifest/split.parquet"
DATA_CONFIG = ENCODER_DATA / "manifest/data_config.json"
TOKEN_CACHE = ROOT / "data/encoder/tokens/egms_tokens_10k.pt"

FC = 10


def parse_csv_ints(text: str) -> list[int]:
    return [int(x) for x in text.split(",") if x.strip()]


def stable_seed(tile_id: str, frac: float, seed: int) -> int:
    payload = f"{tile_id}|{frac:.4f}|{seed}".encode("utf-8")
    return int(hashlib.sha1(payload).hexdigest()[:8], 16)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / den) if den > 0 else float("nan")


def angular_drift(cosine_value: float) -> float:
    if not np.isfinite(cosine_value):
        return float("nan")
    return float(np.arccos(np.clip(cosine_value, -1.0, 1.0)) / np.pi)


def load_encoder(
    checkpoint_path: Path,
    config_path: Path,
    normalization_path: Path,
    device: torch.device,
):
    model, config = load_encoder_checkpoint(checkpoint_path, config_path, device)
    norm = load_normalization(normalization_path)
    return model, config, float(norm["mean"]), float(norm["std"])


def load_store(manifest_path: Path, data_config_path: Path):
    from egms_encoder.data.tile_store import TileStore, TimeWindow

    cfg = json.load(open(data_config_path))
    tw = TimeWindow.from_config(cfg)
    manifest = pd.read_parquet(manifest_path)
    store = TileStore(
        manifest=manifest,
        time_window=tw,
        split_assignments=dict(zip(manifest["tile_id"].astype(str), manifest["split"].astype(str))),
    )
    return store, manifest, tw


def choose_tiles(token_cache: dict, n: int, seed: int) -> list[str]:
    ids = np.asarray([str(t) for t in token_cache["tile_ids"]])
    rng = np.random.default_rng(seed)
    chosen = ids.copy()
    rng.shuffle(chosen)
    return chosen[:n].tolist()


def shard_items(items: list[str], shard_index: int, num_shards: int) -> list[str]:
    if num_shards <= 1:
        return items
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError(f"shard_index must be in [0, {num_shards}), got {shard_index}")
    idx = np.array_split(np.arange(len(items)), num_shards)[shard_index]
    return [items[int(i)] for i in idx]


def metric_summary(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    return {
        "mean": float(np.nanmean(values)),
        "std": float(np.nanstd(values)),
        "min": float(np.nanmin(values)),
        "p01": float(np.nanpercentile(values, 1)),
        "p05": float(np.nanpercentile(values, 5)),
        "p10": float(np.nanpercentile(values, 10)),
        "p25": float(np.nanpercentile(values, 25)),
        "p50": float(np.nanpercentile(values, 50)),
        "p75": float(np.nanpercentile(values, 75)),
        "p90": float(np.nanpercentile(values, 90)),
        "p95": float(np.nanpercentile(values, 95)),
        "p99": float(np.nanpercentile(values, 99)),
        "max": float(np.nanmax(values)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(CKPT))
    ap.add_argument("--model-config", default=str(MODEL_CONFIG))
    ap.add_argument("--normalization", default=str(NORMALIZATION))
    ap.add_argument("--manifest", default=str(MANIFEST))
    ap.add_argument("--data-config", default=str(DATA_CONFIG))
    ap.add_argument("--token-cache", default=str(TOKEN_CACHE))
    ap.add_argument("--out-dir", default="outputs/tasks/a1/work")
    ap.add_argument("--sample-tiles", type=int, default=10000)
    ap.add_argument("--subsample-frac", type=float, default=0.2)
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--sample-seed", type=int, default=7012)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda:0"])
    ap.add_argument("--log-every", type=int, default=25)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = parse_csv_ints(args.seeds)
    if args.device == "auto":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device if args.device != "cuda:0" or torch.cuda.is_available() else "cpu")
    print(f"[device] {device}", flush=True)

    token_cache = torch.load(args.token_cache, map_location="cpu", weights_only=False)
    token_metadata = token_cache.get("metadata", {})
    full_tokens = token_cache["spatial_tokens"].numpy().astype(np.float64)
    tile_ids = [str(t) for t in token_cache["tile_ids"]]
    splits = [str(s) for s in token_cache["splits"]]
    id_to_cache = {t: i for i, t in enumerate(tile_ids)}

    store, manifest, tw = load_store(Path(args.manifest), Path(args.data_config))
    all_chosen = choose_tiles(token_cache, args.sample_tiles, args.sample_seed)
    chosen = shard_items(all_chosen, args.shard_index, args.num_shards)
    manifest_idx = {str(t): i for i, t in enumerate(manifest["tile_id"].astype(str))}

    model, model_config, norm_mean, norm_std = load_encoder(
        Path(args.checkpoint), Path(args.model_config), Path(args.normalization), device
    )
    input_length = int(model_config["input_length"])
    if tw.input_length != input_length:
        raise ValueError(f"time window length {tw.input_length} != checkpoint input_length {input_length}")

    rows = []
    t0 = time.monotonic()
    autocast_device = "cuda" if device.type == "cuda" else "cpu"
    autocast_enabled = device.type == "cuda"
    with torch.no_grad(), torch.amp.autocast(autocast_device, dtype=torch.bfloat16, enabled=autocast_enabled):
        for ti, tile_id in enumerate(chosen):
            si = manifest_idx[tile_id]
            ci = id_to_cache[tile_id]
            td = store.get_tile(si)
            coords = td[:, :2].copy()
            series = td[:, FC:FC + input_length].copy()
            n_pts = int(series.shape[0])
            full_center = coords.mean(0, keepdims=True)
            ref_cls = full_tokens[ci, 0]

            k = max(8, int(round(n_pts * args.subsample_frac)))
            for seed in seeds:
                rng = np.random.default_rng(stable_seed(tile_id, args.subsample_frac, seed))
                idx = np.sort(rng.choice(n_pts, size=min(k, n_pts), replace=False))
                coords_sub = coords[idx]
                series_sub = (series[idx] - norm_mean) / norm_std
                series_sub = np.nan_to_num(series_sub, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
                cc_sub = (coords_sub - full_center).astype(np.float32)

                series_t = torch.from_numpy(series_sub).unsqueeze(0).to(device)
                coords_t = torch.from_numpy(cc_sub).unsqueeze(0).to(device)
                pmask = torch.ones(1, series_sub.shape[0], dtype=torch.bool, device=device)
                out = model(series_t, coords=coords_t, point_mask=pmask)
                emb = out["embedding"].squeeze(0).float().cpu().numpy()
                sub_cls = emb.mean(axis=0)
                cls_cos = cosine(ref_cls, sub_cls)
                drift = angular_drift(cls_cos)
                rows.append({
                    "tile_id": tile_id,
                    "split": splits[ci],
                    "n_points": n_pts,
                    "subsample_frac": float(args.subsample_frac),
                    "seed": int(seed),
                    "n_subsample_points": int(len(idx)),
                    "cls_cosine": cls_cos,
                    "A11_global_angular_drift": drift,
                    "A11_global_stability": float(1.0 - drift) if np.isfinite(drift) else float("nan"),
                })

            if (ti + 1) % args.log_every == 0 or ti == len(chosen) - 1:
                dt = time.monotonic() - t0
                print(f"  {ti+1}/{len(chosen)} tiles  {dt:.1f}s  {((ti+1)*len(seeds))/dt:.2f} forwards/s", flush=True)

    obs = pd.DataFrame(rows)
    obs_path = out_dir / "a1_global_instability_observations.parquet"
    obs.to_parquet(obs_path, index=False)
    tile = (
        obs.groupby(["tile_id", "split", "n_points"], as_index=False)
        .agg(
            cls_cosine_mean=("cls_cosine", "mean"),
            cls_cosine_min=("cls_cosine", "min"),
            A11_global_angular_drift=("A11_global_angular_drift", "mean"),
            A11_global_angular_drift_p50=("A11_global_angular_drift", "median"),
            A11_global_angular_drift_max=("A11_global_angular_drift", "max"),
            A11_global_stability=("A11_global_stability", "mean"),
        )
    )
    tile_path = out_dir / "a1_global_instability_by_tile.csv"
    tile.to_csv(tile_path, index=False)

    summary = {
        "task": "A11",
        "method": "severe global representation instability under point subsampling",
        "target": "A11_global_angular_drift",
        "target_type": "continuous_regression",
        "target_definition": "mean over seeds of arccos(cosine(CLS_full, CLS_20pct_subsample)) / pi; lower is more stable.",
        "sample_tiles_total": int(len(all_chosen)),
        "sample_tiles_this_shard": int(len(chosen)),
        "num_shards": int(args.num_shards),
        "shard_index": int(args.shard_index),
        "subsample_frac": float(args.subsample_frac),
        "seeds": seeds,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "token_cache": str(Path(args.token_cache).resolve()),
        "token_cache_checkpoint": token_metadata.get("encoder_checkpoint"),
        "coord_scale": train_args.get("coord_scale"),
        "metric_diagnostics": {
            "A11_global_angular_drift": metric_summary(tile["A11_global_angular_drift"].to_numpy(dtype=float)),
            "A11_global_stability": metric_summary(tile["A11_global_stability"].to_numpy(dtype=float)),
            "cls_cosine_mean": metric_summary(tile["cls_cosine_mean"].to_numpy(dtype=float)),
        },
        "outputs": {
            "observations": str(obs_path),
            "tile_summary": str(tile_path),
        },
    }
    summary_path = out_dir / "a1_global_instability_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print("\n[A11 global instability]")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
