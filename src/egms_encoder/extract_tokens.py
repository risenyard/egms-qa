"""Extract EGMS tokens from the released 10,000-tile dataset.

Applies the frozen encoder and 65-token pooling (one tile summary plus an 8x8
spatial grid) over the per-tile point histories in the TileStore.

Output
------
outputs/tokens/egms_tokens_10k.pt
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
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from egms_encoder import __version__
from egms_encoder.checkpoint import load_encoder_checkpoint, load_normalization
from egms_encoder.data.tile_store import FEATURE_COLUMNS_COUNT, TileStore

ENCODER_CKPT = Path("data/encoder/checkpoint/encoder.safetensors")
ENCODER_CONFIG = Path("data/encoder/checkpoint/config.json")
ENCODER_NORMALIZATION = Path("data/encoder/checkpoint/normalization.json")
SPLIT_MANIFEST = Path("data/encoder/manifest/split.parquet")
DEFAULT_TILE_SIZE = 7000.0
DEFAULT_GRID_SIZE = 8
TOKEN_SCHEMA = "egms-tokens-1.1"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run the released EGMS-QA Encoder and export pooled EGMS tokens."
    )
    p.add_argument("--checkpoint", default=str(ENCODER_CKPT))
    p.add_argument("--model-config", default=str(ENCODER_CONFIG))
    p.add_argument("--normalization", default=str(ENCODER_NORMALIZATION))
    p.add_argument("--manifest", default=str(SPLIT_MANIFEST))
    p.add_argument("--data-config",
                   default=str(SPLIT_MANIFEST.parent / "data_config.json"))
    p.add_argument("--output-dir", default="outputs/tokens")
    p.add_argument("--output-name", default="")
    p.add_argument("--grid-size", type=int, default=DEFAULT_GRID_SIZE)
    p.add_argument("--tile-size", type=float, default=None)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--max-tiles", type=int, default=None)
    p.add_argument("--log-every", type=int, default=500)
    p.add_argument(
        "--encoder-repository",
        default="",
        help="Optional public repository recorded as encoder provenance.",
    )
    p.add_argument(
        "--dataset-repository",
        default="",
        help="Optional public repository recorded as dataset provenance.",
    )
    return p.parse_args(argv)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_tile_store(manifest_path: Path, data_config_path: Path) -> tuple[TileStore, Any]:
    print(f"[manifest] reading {manifest_path}", flush=True)
    store = TileStore.from_manifest(manifest_path, data_config_path)
    print(
        f"[manifest] {store.num_tiles} tiles, input_length={store.time_window.input_length}",
        flush=True,
    )
    return store, store.manifest


def pool_to_spatial_tokens(
    embedding: np.ndarray,
    centered_coords: np.ndarray,
    grid_size: int,
    tile_size: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if embedding.ndim != 2 or embedding.shape[0] == 0:
        raise ValueError(f"embedding must have shape [N,D] with N>0, got {embedding.shape}")
    if centered_coords.shape != (embedding.shape[0], 2):
        raise ValueError(
            f"centered_coords must have shape [{embedding.shape[0]},2], "
            f"got {centered_coords.shape}"
        )
    if grid_size <= 0 or tile_size <= 0:
        raise ValueError("grid_size and tile_size must be positive")
    d = embedding.shape[1]
    n_tok = grid_size * grid_size + 1
    tokens = np.zeros((n_tok, d), dtype=np.float32)
    mask = np.zeros(n_tok, dtype=bool)
    tokens[0] = embedding.mean(axis=0)
    mask[0] = True
    half = tile_size * 0.5
    bx = np.clip(
        np.floor((centered_coords[:, 0] + half) / tile_size * grid_size).astype(np.int64),
        0,
        grid_size - 1,
    )
    by = np.clip(
        np.floor((centered_coords[:, 1] + half) / tile_size * grid_size).astype(np.int64),
        0,
        grid_size - 1,
    )
    bidx = by * grid_size + bx
    counts = np.bincount(bidx, minlength=grid_size * grid_size).astype(np.int32)
    for b in range(grid_size * grid_size):
        sel = bidx == b
        if sel.any():
            tokens[1 + b] = embedding[sel].mean(axis=0)
            mask[1 + b] = True
    return tokens, mask, counts


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if str(device) == "cpu":
        print("CUDA unavailable, running CPU (slow).", flush=True)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = Path(args.checkpoint)
    norm_path = Path(args.normalization)
    norm = load_normalization(norm_path)
    norm_mean = float(norm["mean"])
    norm_std = float(norm["std"])
    print(f"[norm] mean={norm_mean:.4f} std={norm_std:.4f}", flush=True)

    model_config_path = Path(args.model_config)
    print(f"[load] {ckpt_path}", flush=True)
    model, model_config = load_encoder_checkpoint(
        ckpt_path, model_config_path, device
    )
    print(
        f"[encoder] d_model={model_config['d_model']}, "
        f"layers={model_config['spatial_layers']}, heads={model_config['spatial_heads']}",
        flush=True,
    )
    input_length = int(model_config["input_length"])
    tile_size = float(args.tile_size or DEFAULT_TILE_SIZE)
    if tile_size <= 0:
        raise ValueError("--tile-size must be positive")

    manifest_path = Path(args.manifest)
    data_config_path = Path(args.data_config)
    store, manifest = load_tile_store(manifest_path, data_config_path)
    time_window = store.time_window
    if time_window.input_length != input_length:
        raise ValueError(
            f"time_window length {time_window.input_length} != encoder input_length {input_length}"
        )

    n_total = store.num_tiles
    n_tiles = n_total if args.max_tiles is None else min(n_total, args.max_tiles)
    if n_tiles <= 0:
        raise ValueError("--max-tiles must select at least one tile")
    if args.grid_size <= 0:
        raise ValueError("--grid-size must be positive")
    if args.log_every <= 0:
        raise ValueError("--log-every must be positive")
    n_patch = args.grid_size * args.grid_size
    n_tok = n_patch + 1
    d_model = int(model_config["d_model"])

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
        for tile_index in range(n_tiles):
            tile_data = store.get_tile(tile_index)
            n_pts = tile_data.shape[0]
            n_points_per_tile[tile_index] = n_pts

            coords_np = tile_data[:, :2].copy()
            series_np = tile_data[
                :, FEATURE_COLUMNS_COUNT : FEATURE_COLUMNS_COUNT + input_length
            ].copy()

            series_np = (series_np - norm_mean) / norm_std
            series_np = np.nan_to_num(series_np, nan=0.0, posinf=0.0, neginf=0.0)
            center = coords_np.mean(axis=0, keepdims=True)
            centered_coords = (coords_np - center).astype(np.float32)

            series_t = torch.from_numpy(series_np).unsqueeze(0).to(device)
            coords_t = torch.from_numpy(centered_coords).unsqueeze(0).to(device)
            pmask_t = torch.ones(1, n_pts, dtype=torch.bool, device=device)

            emb = (
                model.encode(series_t, coords=coords_t, point_mask=pmask_t)
                .squeeze(0)
                .float()
                .cpu()
                .numpy()
            )

            tokens, mask, counts = pool_to_spatial_tokens(
                emb, centered_coords, args.grid_size, tile_size
            )
            spatial_tokens[tile_index] = tokens
            token_mask[tile_index] = mask
            point_count_per_bin[tile_index] = counts

            if (tile_index + 1) % args.log_every == 0 or tile_index == n_tiles - 1:
                elapsed = time.monotonic() - t0
                rate = (tile_index + 1) / elapsed
                eta = (n_tiles - tile_index - 1) / max(rate, 1e-6)
                print(f"  {tile_index+1}/{n_tiles}  elapsed={elapsed:.1f}s  "
                      f"rate={rate:.2f} tiles/s  eta={eta:.0f}s", flush=True)

    occ = (point_count_per_bin > 0).mean(axis=1)
    print(f"\noccupancy mean={occ.mean():.3f}  median={np.median(occ):.3f}", flush=True)
    print(f"n_points mean={n_points_per_tile.mean():.1f}  "
          f"min={n_points_per_tile.min()}  max={n_points_per_tile.max()}", flush=True)

    output_name = args.output_name or (
        "egms_tokens_10k.pt" if n_tiles == 10_000 else f"egms_tokens_{n_tiles}.pt"
    )
    out_pt = out_dir / output_name
    source_window = store.data_config.get("time_window", {})
    metadata = {
        "schema_version": TOKEN_SCHEMA,
        "release_name": "EGMS-QA token cache",
        "code_version": __version__,
        "input_contract": {
            "stored_steps": int(source_window.get("stored_steps", input_length)),
            "stored_window": f"[{time_window.t_start},{time_window.t_end})",
            "source_axis_steps": int(
                source_window.get("original_source_steps", input_length)
            ),
            "source_window": (
                f"[{source_window['original_t_start']},"
                f"{source_window['original_t_end']})"
                if "original_t_start" in source_window
                and "original_t_end" in source_window
                else f"[{time_window.t_start},{time_window.t_end})"
            ),
            "source_index_offset": int(
                source_window.get("original_index_offset", time_window.t_start)
            ),
            "cadence_days": float(source_window.get("cadence_days", 0.0)),
            "tile_size_m": tile_size,
            "token_count": int(n_tok),
            "token_width": int(d_model),
            "token_layout": (
                f"index 0 = tile summary; 1..{n_patch} = "
                f"{args.grid_size}x{args.grid_size} spatial cells in row-major order"
            ),
        },
        "reproduction_input_sha256": {
            "encoder_weights": _sha256(ckpt_path),
            "encoder_config": _sha256(model_config_path),
            "normalization": _sha256(norm_path),
            "split_manifest": _sha256(manifest_path),
            "data_config": _sha256(data_config_path),
        },
        "extraction": {
            "module": "egms_encoder.extract_tokens",
            "date_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        },
        "statistics": {
            "tiles": int(n_tiles),
            "bin_occupancy_mean": float(occ.mean()),
            "bin_occupancy_median": float(np.median(occ)),
            "points_per_tile_mean": float(n_points_per_tile.mean()),
            "points_per_tile_min": int(n_points_per_tile.min()),
            "points_per_tile_max": int(n_points_per_tile.max()),
        },
        "output_file": output_name,
    }
    source_repositories = {}
    if args.encoder_repository:
        source_repositories["encoder"] = args.encoder_repository
    if args.dataset_repository:
        source_repositories.update(
            {
                "dataset": args.dataset_repository,
                "tiles": "artifacts/source_tiles",
            }
        )
    if source_repositories:
        metadata["source_repositories"] = source_repositories
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
    metadata_name = f"{Path(output_name).stem}_metadata.json"
    with open(out_dir / metadata_name, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=str)
    print(f"\nwrote {out_pt}", flush=True)


if __name__ == "__main__":
    main()
