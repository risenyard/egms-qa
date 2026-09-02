"""EGMS-QA token extraction with the EGMS encoder on Europe-wide 10k tiles.

Applies the frozen encoder and ViT-style 65-token pooling (CLS + 8x8 spatial
bins) over the per-tile point histories in the LazyTileStore.

Output
------
outputs/tokens/encoder_tokens_10k.pt
  - spatial_tokens [T, 65, 256] float32
  - token_mask [T, 65] bool
  - tile_indices [T] int32 (index into manifest)
  - tile_ids [T] (str)
  - point_count_per_bin [T, 64] int32
  - n_points_per_tile [T] int32
  - splits [T] (str: train/val/test)
  - metadata dict
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch


# The encoder package (egms_encoder) is installed; the raw per-tile point store
# lives in the separate EGMS encoder project, located via EGMS_ENCODER_HOME.
from egms_qa.paths import ENCODER_HOME, ENCODER_CKPT, SPLIT_MANIFEST

TILE_SIZE = 7000.0
GRID = 8


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=str(ENCODER_CKPT))
    p.add_argument("--manifest", default=str(SPLIT_MANIFEST))
    p.add_argument("--data-config",
                   default=str(ENCODER_HOME / "data/processed/v4/v4_data_config.json"))
    p.add_argument("--output-dir", default="outputs/tokens")
    p.add_argument("--grid", type=int, default=GRID)
    p.add_argument("--tile-size", type=float, default=TILE_SIZE)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--max-tiles", type=int, default=None)
    p.add_argument("--log-every", type=int, default=500)
    return p.parse_args()


def load_encoder(checkpoint_path: Path, device: torch.device):
    print(f"[load] {checkpoint_path}", flush=True)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    train_args = ckpt["args"]
    if train_args.get("model_version") != "v3_3":
        raise ValueError(f"unexpected model_version={train_args.get('model_version')}")

    from egms_encoder.models.tile_encoder_v31 import TileEncoderV31

    # coord_scale is applied INSIDE the model's forward (coords / coord_scale
    # before the coord embedding). It MUST be reconstructed from the checkpoint
    # args or EGMS encoder (coord_scale=3500) would see raw-scale coords and produce
    # garbage tokens. v4.2 has coord_scale=None, so this is a no-op there
    # (backward compatible).
    coord_scale = train_args.get("coord_scale")
    print(f"[coord_scale] {coord_scale}", flush=True)
    model = TileEncoderV31(
        input_length=train_args["input_length"],
        d_model=train_args["d_model"],
        patch_size=train_args.get("patch_size", 8),
        temporal_layers=train_args.get("temporal_layers", 2),
        temporal_heads=train_args.get("temporal_heads", 4),
        spatial_layers=train_args["num_layers"],
        spatial_heads=train_args["num_heads"],
        residual_head_mode=train_args.get("residual_head_mode", "additive"),
        coord_scale=coord_scale,
    )
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    if unexpected:
        print(f"  unexpected: {unexpected[:5]}", flush=True)
    if missing:
        print(f"  missing: {missing[:5]}", flush=True)
    model.eval().to(device)
    return model, train_args


def load_lazy_store(manifest_path: Path, data_config_path: Path):
    from egms_encoder.data.lazy_tile_store import LazyTileStore, V4TimeWindow

    with open(data_config_path) as f:
        cfg = json.load(f)
    tw = V4TimeWindow(t_start=cfg["time_window"]["t_start"],
                      t_end=cfg["time_window"]["t_end"])
    print(f"[manifest] reading {manifest_path}", flush=True)
    manifest = pd.read_parquet(manifest_path)
    split_assignments = dict(zip(manifest["tile_id"].astype(str),
                                 manifest["split"].astype(str)))
    store = LazyTileStore(
        manifest=manifest,
        time_window=tw,
        split_assignments=split_assignments,
    )
    print(f"[manifest] {store.num_tiles} tiles, input_length={tw.input_length}", flush=True)
    return store, tw, manifest


def pool_to_vit_tokens(embedding, coords_centered, grid, tile_size):
    d = embedding.shape[1]
    n_tok = grid * grid + 1
    tokens = np.zeros((n_tok, d), dtype=np.float32)
    mask = np.zeros(n_tok, dtype=bool)
    tokens[0] = embedding.mean(axis=0)
    mask[0] = True
    half = tile_size * 0.5
    bx = np.clip(np.floor((coords_centered[:, 0] + half) / tile_size * grid).astype(np.int64), 0, grid - 1)
    by = np.clip(np.floor((coords_centered[:, 1] + half) / tile_size * grid).astype(np.int64), 0, grid - 1)
    bidx = by * grid + bx
    for b in range(grid * grid):
        sel = bidx == b
        if sel.any():
            tokens[1 + b] = embedding[sel].mean(axis=0)
            mask[1 + b] = True
    return tokens, mask


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if str(device) == "cpu":
        print("CUDA unavailable, running CPU (slow).", flush=True)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = Path(args.checkpoint)
    norm_path = ckpt_path.parent / "normalization.json"
    with open(norm_path) as f:
        norm = json.load(f)
    norm_mean = float(norm["mean"]); norm_std = float(norm["std"])
    print(f"[norm] mean={norm_mean:.4f} std={norm_std:.4f}", flush=True)

    model, train_args = load_encoder(ckpt_path, device)
    print(f"[encoder] v4 encoder, d_model={train_args['d_model']}, "
          f"layers={train_args['num_layers']}, heads={train_args['num_heads']}", flush=True)
    input_length = train_args["input_length"]
    fc = 10  # FEATURE_COLUMNS_COUNT in LazyTileStore

    store, tw, manifest = load_lazy_store(Path(args.manifest), Path(args.data_config))
    if tw.input_length != input_length:
        raise ValueError(f"time_window len {tw.input_length} != checkpoint input_length {input_length}")

    n_total = store.num_tiles
    n_tiles = n_total if args.max_tiles is None else min(n_total, args.max_tiles)
    n_patch = args.grid * args.grid
    n_tok = n_patch + 1
    d_model = int(train_args["d_model"])

    spatial_tokens = np.zeros((n_tiles, n_tok, d_model), dtype=np.float32)
    token_mask = np.zeros((n_tiles, n_tok), dtype=bool)
    tile_indices = np.arange(n_tiles, dtype=np.int32)
    tile_ids = manifest["tile_id"].astype(str).values[:n_tiles]
    splits = manifest["split"].astype(str).values[:n_tiles]
    point_count_per_bin = np.zeros((n_tiles, n_patch), dtype=np.int32)
    n_points_per_tile = np.zeros(n_tiles, dtype=np.int32)

    autocast_device = "cuda" if device.type == "cuda" else "cpu"
    autocast_enabled = device.type == "cuda"
    t0 = time.monotonic()
    with torch.no_grad(), torch.amp.autocast(autocast_device, dtype=torch.bfloat16,
                                              enabled=autocast_enabled):
        for ti in range(n_tiles):
            td = store.get_tile(ti)
            n_pts = td.shape[0]
            n_points_per_tile[ti] = n_pts

            coords_np = td[:, :2].copy()
            series_np = td[:, fc:fc + input_length].copy()

            series_np = (series_np - norm_mean) / norm_std
            series_np = np.nan_to_num(series_np, nan=0.0, posinf=0.0, neginf=0.0)
            center = coords_np.mean(axis=0, keepdims=True)
            cc = (coords_np - center).astype(np.float32)

            series_t = torch.from_numpy(series_np).unsqueeze(0).to(device)
            coords_t = torch.from_numpy(cc).unsqueeze(0).to(device)
            pmask_t = torch.ones(1, n_pts, dtype=torch.bool, device=device)

            out = model(series_t, coords=coords_t, point_mask=pmask_t)
            emb = out["embedding"].squeeze(0).float().cpu().numpy()

            tokens, mask = pool_to_vit_tokens(emb, cc, args.grid, args.tile_size)
            spatial_tokens[ti] = tokens
            token_mask[ti] = mask

            half = args.tile_size * 0.5
            bx = np.clip(np.floor((cc[:, 0] + half) / args.tile_size * args.grid).astype(np.int64), 0, args.grid - 1)
            by = np.clip(np.floor((cc[:, 1] + half) / args.tile_size * args.grid).astype(np.int64), 0, args.grid - 1)
            np.add.at(point_count_per_bin[ti], by * args.grid + bx, 1)

            if (ti + 1) % args.log_every == 0 or ti == n_tiles - 1:
                elapsed = time.monotonic() - t0
                rate = (ti + 1) / elapsed
                eta = (n_tiles - ti - 1) / max(rate, 1e-6)
                print(f"  {ti+1}/{n_tiles}  elapsed={elapsed:.1f}s  "
                      f"rate={rate:.2f} tiles/s  eta={eta:.0f}s", flush=True)

    occ = (point_count_per_bin > 0).mean(axis=1)
    print(f"\noccupancy mean={occ.mean():.3f}  median={np.median(occ):.3f}", flush=True)
    print(f"n_points mean={n_points_per_tile.mean():.1f}  "
          f"min={n_points_per_tile.min()}  max={n_points_per_tile.max()}", flush=True)

    out_pt = out_dir / "encoder_tokens.pt"
    metadata = {
        "encoder_checkpoint": str(ckpt_path.resolve()),
        "coord_scale": train_args.get("coord_scale"),
        "encoder_args": dict(train_args),
        "manifest_path": str(Path(args.manifest).resolve()),
        "data_config_path": str(Path(args.data_config).resolve()),
        "normalizer_mean": norm_mean,
        "normalizer_std": norm_std,
        "tile_size": float(args.tile_size),
        "grid_size": int(args.grid),
        "n_tokens": int(n_tok),
        "n_tiles": int(n_tiles),
        "d_model": int(d_model),
        "input_length": int(input_length),
        "time_window_start": int(tw.t_start),
        "time_window_end": int(tw.t_end),
        "token_layout": f"index 0 = CLS, 1..{n_patch} = {args.grid}x{args.grid} bins row-major",
        "extraction_script": str(Path(__file__).resolve()),
        "extraction_date_utc": dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "bin_occupancy_mean": float(occ.mean()),
        "bin_occupancy_median": float(np.median(occ)),
        "n_points_per_tile_mean": float(n_points_per_tile.mean()),
        "n_points_per_tile_min": int(n_points_per_tile.min()),
        "n_points_per_tile_max": int(n_points_per_tile.max()),
    }
    torch.save({
        "spatial_tokens": torch.from_numpy(spatial_tokens),
        "token_mask": torch.from_numpy(token_mask),
        "tile_indices": torch.from_numpy(tile_indices),
        "tile_ids": list(tile_ids),
        "splits": list(splits),
        "point_count_per_bin": torch.from_numpy(point_count_per_bin),
        "n_points_per_tile": torch.from_numpy(n_points_per_tile),
        "metadata": metadata,
    }, out_pt)
    with open(out_dir / "extraction_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    print(f"\nwrote {out_pt}", flush=True)


if __name__ == "__main__":
    main()
