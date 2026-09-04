"""A21/A22 masked temporal reconstruction outputs for EGMS-QA.

A21 measures whether the EGMS encoder can reconstruct a synchronized masked
temporal block for each tile. It reuses the EGMS encoder reconstruction setup:
30% fixed centered time block, max 4096 points per tile, frozen encoder.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch


ROOT = Path(".")
ENCODER_DATA = ROOT / "data/encoder"

CKPT = ENCODER_DATA / "checkpoint/encoder.pt"
MANIFEST = ENCODER_DATA / "manifest/split.parquet"
DATA_CONFIG = ENCODER_DATA / "manifest/data_config.json"

from egms_encoder.data.tile_store import TileStore  # noqa: E402
from egms_encoder.pretrain import FEATURE_COLUMNS, build_model  # noqa: E402


def stable_seed(text: str) -> int:
    return int(hashlib.sha1(text.encode("utf-8")).hexdigest()[:8], 16)


def shard_items(items: np.ndarray, shard_index: int, num_shards: int) -> np.ndarray:
    if num_shards <= 1:
        return items
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError(f"shard_index must be in [0, {num_shards}), got {shard_index}")
    return np.array_split(items, num_shards)[shard_index]


def linear_detrend_np(x: np.ndarray) -> np.ndarray:
    t = np.linspace(-1.0, 1.0, x.shape[-1], dtype=np.float32)
    t = t - t.mean()
    denom = max(float((t * t).sum()), np.finfo(np.float32).eps)
    intercept = x.mean(axis=-1, keepdims=True)
    slope = (x * t).sum(axis=-1, keepdims=True) / denom
    return x - (intercept + slope * t)


def summarize(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    return {
        "mean": float(np.nanmean(values)),
        "std": float(np.nanstd(values)),
        "min": float(np.nanmin(values)),
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
    ap.add_argument("--manifest", default=str(MANIFEST))
    ap.add_argument("--data-config", default=str(DATA_CONFIG))
    ap.add_argument("--out-dir", default="outputs/tasks/a2/work")
    ap.add_argument("--sample-tiles", type=int, default=10000)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda:0"])
    ap.add_argument("--log-every", type=int, default=20)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.device == "auto":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device if args.device != "cuda:0" or torch.cuda.is_available() else "cpu")
    print(f"[device] {device}", flush=True)

    ckpt_path = Path(args.checkpoint)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    train_args = SimpleNamespace(**ckpt["args"])
    norm = json.load(open(ckpt_path.parent / "normalization.json"))
    norm_mean = float(norm["mean"])
    norm_std = float(norm["std"])
    residual_std = float(norm.get("residual_std", 1.0))

    model = build_model(train_args, norm)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()

    store = TileStore.from_manifest(args.manifest, args.data_config)
    input_length = int(store.time_window.input_length)
    max_points = int(getattr(train_args, "max_tile_points", 4096))
    mask_ratio = float(getattr(train_args, "eval_mask_ratio", getattr(train_args, "mask_ratio", 0.3)))
    block_len = max(1, int(round(input_length * mask_ratio)))
    mask_start = (input_length - block_len) // 2
    mask_end = mask_start + block_len
    fc = len(FEATURE_COLUMNS)

    all_indices = np.arange(store.num_tiles, dtype=np.int64)[: args.sample_tiles]
    indices = shard_items(all_indices, args.shard_index, args.num_shards)

    split_by_tile = {}
    for split in ("train", "val", "test"):
        for idx in store.split_tile_indices(split):
            split_by_tile[int(idx)] = split

    rows = []
    t0 = time.monotonic()
    autocast_enabled = device.type == "cuda"
    autocast_device = "cuda" if device.type == "cuda" else "cpu"
    with torch.no_grad(), torch.amp.autocast(autocast_device, dtype=torch.bfloat16, enabled=autocast_enabled):
        for pos, tile_idx in enumerate(indices):
            tile_idx = int(tile_idx)
            meta = store.tile_metadata[tile_idx]
            tile_id = str(meta["tile_id"])
            tile = store.get_tile(tile_idx)
            n_points = int(len(tile))
            if n_points > max_points:
                rng = np.random.default_rng(stable_seed(f"A2|{tile_id}|{max_points}"))
                chosen = np.sort(rng.choice(n_points, size=max_points, replace=False))
                sub = tile[chosen]
            else:
                sub = tile

            series = sub[:, fc:fc + input_length].copy()
            finite = np.isfinite(series)
            norm_series = np.nan_to_num(
                (series - norm_mean) / norm_std,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            ).astype(np.float32)
            masked = norm_series.copy()
            masked[:, mask_start:mask_end] = 0.0
            coords = (sub[:, :2] - sub[:, :2].mean(axis=0, keepdims=True)).astype(np.float32)

            out = model(
                torch.from_numpy(masked).unsqueeze(0).to(device),
                coords=torch.from_numpy(coords).unsqueeze(0).to(device),
                point_mask=torch.ones(1, len(sub), dtype=torch.bool, device=device),
            )
            recon = out["reconstruction"].squeeze(0).float().cpu().numpy()
            eval_mask = finite[:, mask_start:mask_end]
            if int(eval_mask.sum()) == 0:
                global_mse = global_mae = residual_mse = base_mse = residual_head_mse = float("nan")
            else:
                pred_block = recon[:, mask_start:mask_end]
                target_block = norm_series[:, mask_start:mask_end]
                diff = pred_block[eval_mask] - target_block[eval_mask]
                global_mse = float(np.mean(np.square(diff)))
                global_mae = float(np.mean(np.abs(diff)))

                recon_resid = linear_detrend_np(recon)
                target_resid = linear_detrend_np(norm_series)
                resid_diff = recon_resid[:, mask_start:mask_end][eval_mask] - target_resid[:, mask_start:mask_end][eval_mask]
                residual_mse = float(np.mean(np.square(resid_diff)))

                if "base_reconstruction" in out:
                    base = out["base_reconstruction"].squeeze(0).float().cpu().numpy()
                    base_diff = base[:, mask_start:mask_end][eval_mask] - target_block[eval_mask]
                    base_mse = float(np.mean(np.square(base_diff)))
                else:
                    base_mse = global_mse

                if "residual_prediction" in out:
                    resid_pred = out["residual_prediction"].squeeze(0).float().cpu().numpy()
                    target_resid_z = target_resid / max(residual_std, 1e-6)
                    rh_diff = resid_pred[:, mask_start:mask_end][eval_mask] - target_resid_z[:, mask_start:mask_end][eval_mask]
                    residual_head_mse = float(np.mean(np.square(rh_diff)))
                else:
                    residual_head_mse = residual_mse

            total_loss = (
                global_mse
                + float(getattr(train_args, "residual_loss_weight", 1.0)) * residual_head_mse
                + float(getattr(train_args, "residual_consistency_weight", 0.1)) * residual_mse
            )
            rows.append({
                "tile_idx": tile_idx,
                "tile_id": tile_id,
                "split": split_by_tile.get(tile_idx, ""),
                "n_points": n_points,
                "n_eval_points": int(len(sub)),
                "max_tile_points": max_points,
                "mask_start": int(mask_start),
                "mask_end": int(mask_end),
                "mask_ratio": float(mask_ratio),
                "masked_count": int(eval_mask.sum()),
                "A21_masked_global_mse_z": global_mse,
                "A21_masked_global_rmse_z": float(np.sqrt(global_mse)) if np.isfinite(global_mse) else float("nan"),
                "A21_masked_global_mae_z": global_mae,
                "A21_masked_global_mse_mm2": global_mse * norm_std * norm_std if np.isfinite(global_mse) else float("nan"),
                "A21_masked_global_rmse_mm": float(np.sqrt(global_mse)) * norm_std if np.isfinite(global_mse) else float("nan"),
                "A21_masked_global_mae_mm": global_mae * norm_std if np.isfinite(global_mae) else float("nan"),
                "A21_masked_residual_mse_z": residual_mse,
                "A21_base_global_mse_z": base_mse,
                "A21_residual_head_mse_z": residual_head_mse,
                "A21_training_weighted_loss": total_loss,
            })

            if (pos + 1) % args.log_every == 0 or pos == len(indices) - 1:
                dt = time.monotonic() - t0
                print(f"  {pos+1}/{len(indices)} tiles  {dt:.1f}s  {(pos+1)/max(dt,1e-6):.2f} tiles/s", flush=True)

    table = pd.DataFrame(rows)
    table_path = out_dir / "a2_by_tile.csv"
    summary_path = out_dir / "a2_summary.json"
    table.to_csv(table_path, index=False)
    summary = {
        "task": "A21",
        "method": "EGMS encoder fixed synchronized 30% temporal block masked reconstruction",
        "checkpoint": str(ckpt_path),
        "data_config": str(args.data_config),
        "sample_tiles_total": int(len(all_indices)),
        "sample_tiles_this_shard": int(len(indices)),
        "num_shards": int(args.num_shards),
        "shard_index": int(args.shard_index),
        "target": "A21_masked_global_mse_z",
        "target_definition": "per-tile mean squared error on masked finite normalized time-series positions",
        "mask_start": int(mask_start),
        "mask_end": int(mask_end),
        "mask_ratio": float(mask_ratio),
        "max_tile_points": int(max_points),
        "metric_diagnostics": {
            "A21_masked_global_mse_z": summarize(table["A21_masked_global_mse_z"].to_numpy(dtype=float)),
            "A21_masked_global_rmse_mm": summarize(table["A21_masked_global_rmse_mm"].to_numpy(dtype=float)),
        },
        "outputs": {"tile_summary": str(table_path)},
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
