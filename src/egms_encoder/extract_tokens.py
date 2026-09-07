"""EGMS-QA token extraction with the EGMS encoder on Europe-wide 10k tiles.

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
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch


from egms_encoder.checkpoint import load_encoder_checkpoint, load_normalization

TILE_SIZE = 7000.0
GRID = 8
DEFAULT_ENCODER_REPO = "risenyard/egms-qa-encoder"
DEFAULT_DATASET_REPO = "risenyard/egms-qa-dataset"


@dataclass(frozen=True)
class EncoderInputs:
    checkpoint: Path
    model_config: Path
    normalization: Path
    repository: str | None
    revision: str | None


@dataclass(frozen=True)
class DatasetInputs:
    manifest: Path
    data_config: Path
    source_tiles_root: Path | None
    repository: str | None
    revision: str | None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download the published EGMS-QA artifacts and extract pooled tokens."
    )
    p.add_argument("--encoder-repo", default=DEFAULT_ENCODER_REPO)
    p.add_argument("--encoder-revision", default="main")
    p.add_argument("--dataset-repo", default=DEFAULT_DATASET_REPO)
    p.add_argument("--dataset-revision", default="main")
    p.add_argument("--cache-dir", default=None)
    p.add_argument(
        "--checkpoint",
        default="",
        help="Local override for encoder.safetensors; requires --model-config and --normalization.",
    )
    p.add_argument("--model-config", default="")
    p.add_argument("--normalization", default="")
    p.add_argument(
        "--manifest",
        default="",
        help="Local override for split.parquet; requires --data-config.",
    )
    p.add_argument("--data-config", default="")
    p.add_argument(
        "--source-tiles-root",
        default="",
        help="Optional local root containing the manifest's source tile paths.",
    )
    p.add_argument("--output-dir", default="outputs/tokens")
    p.add_argument("--output-name", default="")
    p.add_argument("--grid", type=int, default=GRID)
    p.add_argument("--tile-size", type=float, default=TILE_SIZE)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--max-tiles", type=int, default=None)
    p.add_argument("--log-every", type=int, default=500)
    return p.parse_args(argv)


def resolve_encoder_inputs(args: argparse.Namespace) -> EncoderInputs:
    local_values = (args.checkpoint, args.model_config, args.normalization)
    if any(local_values):
        if not all(local_values):
            raise ValueError(
                "Local encoder overrides require --checkpoint, --model-config, "
                "and --normalization together."
            )
        return EncoderInputs(
            checkpoint=Path(args.checkpoint),
            model_config=Path(args.model_config),
            normalization=Path(args.normalization),
            repository=None,
            revision=None,
        )

    from huggingface_hub import HfApi, hf_hub_download

    revision = HfApi().model_info(
        args.encoder_repo,
        revision=args.encoder_revision,
    ).sha
    def download(filename: str) -> Path:
        return Path(hf_hub_download(
            repo_id=args.encoder_repo,
            filename=filename,
            revision=revision,
            cache_dir=args.cache_dir,
        ))
    return EncoderInputs(
        checkpoint=download("encoder.safetensors"),
        model_config=download("config.json"),
        normalization=download("normalization.json"),
        repository=args.encoder_repo,
        revision=revision,
    )


def resolve_dataset_inputs(args: argparse.Namespace) -> DatasetInputs:
    local_values = (args.manifest, args.data_config)
    if any(local_values) or args.source_tiles_root:
        if not all(local_values):
            raise ValueError(
                "Local dataset overrides require --manifest and --data-config together."
            )
        return DatasetInputs(
            manifest=Path(args.manifest),
            data_config=Path(args.data_config),
            source_tiles_root=(Path(args.source_tiles_root) if args.source_tiles_root else None),
            repository=None,
            revision=None,
        )

    from huggingface_hub import HfApi, hf_hub_download, snapshot_download

    revision = HfApi().dataset_info(
        args.dataset_repo,
        revision=args.dataset_revision,
    ).sha
    manifest = Path(hf_hub_download(
        repo_id=args.dataset_repo,
        repo_type="dataset",
        filename="metadata/split_manifest.parquet",
        revision=revision,
        cache_dir=args.cache_dir,
    ))
    data_config = Path(hf_hub_download(
        repo_id=args.dataset_repo,
        repo_type="dataset",
        filename="metadata/data_config.json",
        revision=revision,
        cache_dir=args.cache_dir,
    ))
    if args.max_tiles is None:
        tile_patterns = ["artifacts/source_tiles/**/*.npz"]
    else:
        if args.max_tiles <= 0:
            raise ValueError("--max-tiles must be positive")
        manifest_rows = pd.read_parquet(manifest, columns=["path"])
        tile_patterns = [
            _published_tile_repo_path(value).as_posix()
            for value in manifest_rows["path"].iloc[: args.max_tiles]
        ]
    snapshot = Path(snapshot_download(
        repo_id=args.dataset_repo,
        repo_type="dataset",
        revision=revision,
        cache_dir=args.cache_dir,
        allow_patterns=tile_patterns,
    ))
    return DatasetInputs(
        manifest=manifest,
        data_config=data_config,
        source_tiles_root=snapshot / "artifacts/source_tiles",
        repository=args.dataset_repo,
        revision=revision,
    )


def load_encoder(checkpoint_path: Path, config_path: Path, device: torch.device):
    print(f"[load] {checkpoint_path}", flush=True)
    model, config = load_encoder_checkpoint(checkpoint_path, config_path, device)
    print(f"[coord_scale] {config['coord_scale_m']}", flush=True)
    return model, config


def _published_tile_repo_path(value: object) -> Path:
    path = Path(str(value))
    parts = path.parts
    if parts[:2] == ("data", "tiles"):
        return Path("artifacts/source_tiles").joinpath(*parts[2:])
    if parts[:2] == ("artifacts", "source_tiles"):
        return path
    raise ValueError(f"manifest path is outside the published source-tile tree: {path}")


def _resolve_source_tile_path(value: object, source_tiles_root: Path) -> Path:
    path = Path(str(value))
    if path.is_absolute():
        return path
    repository_path = _published_tile_repo_path(path)
    return source_tiles_root.joinpath(*repository_path.parts[2:])


def load_tile_store(
    manifest_path: Path,
    data_config_path: Path,
    source_tiles_root: Path | None = None,
):
    from egms_encoder.data.tile_store import TileStore, TimeWindow

    print(f"[manifest] reading {manifest_path}", flush=True)
    manifest = pd.read_parquet(manifest_path)
    with data_config_path.open(encoding="utf-8") as handle:
        data_config = json.load(handle)
    if source_tiles_root is not None:
        manifest = manifest.copy()
        manifest["path"] = [
            str(_resolve_source_tile_path(value, source_tiles_root))
            for value in manifest["path"]
        ]
    split_assignments = None
    if "split" in manifest.columns:
        split_assignments = dict(
            zip(manifest["tile_id"].astype(str), manifest["split"].astype(str))
        )
    store = TileStore(
        manifest=manifest,
        time_window=TimeWindow.from_config(data_config),
        split_assignments=split_assignments,
        feature_columns_count=int(
            data_config["tile_field_layout"]["feature_columns_count"]
        ),
    )
    tw = store.time_window
    print(f"[manifest] {store.num_tiles} tiles, input_length={tw.input_length}", flush=True)
    return store, tw, store.manifest


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

    encoder_inputs = resolve_encoder_inputs(args)
    dataset_inputs = resolve_dataset_inputs(args)
    ckpt_path = encoder_inputs.checkpoint
    norm = load_normalization(encoder_inputs.normalization)
    norm_mean = float(norm["mean"]); norm_std = float(norm["std"])
    print(f"[norm] mean={norm_mean:.4f} std={norm_std:.4f}", flush=True)

    model, model_config = load_encoder(
        ckpt_path,
        encoder_inputs.model_config,
        device,
    )
    print(f"[encoder] d_model={model_config['d_model']}, "
          f"layers={model_config['spatial_layers']}, "
          f"heads={model_config['spatial_heads']}", flush=True)
    input_length = int(model_config["input_length"])
    fc = 10  # FEATURE_COLUMNS_COUNT in TileStore

    store, tw, manifest = load_tile_store(
        dataset_inputs.manifest,
        dataset_inputs.data_config,
        dataset_inputs.source_tiles_root,
    )
    if tw.input_length != input_length:
        raise ValueError(f"time_window len {tw.input_length} != checkpoint input_length {input_length}")

    n_total = store.num_tiles
    n_tiles = n_total if args.max_tiles is None else min(n_total, args.max_tiles)
    n_patch = args.grid * args.grid
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

    output_name = args.output_name or (
        "egms_tokens_10k.pt" if n_tiles == 10_000 else f"egms_tokens_{n_tiles}.pt"
    )
    out_pt = out_dir / output_name
    with dataset_inputs.data_config.open(encoding="utf-8") as handle:
        data_config = json.load(handle)
    configured_axis = data_config["time_window"]
    encoder_source = (
        {
            "type": "huggingface",
            "repository": encoder_inputs.repository,
            "revision": encoder_inputs.revision,
        }
        if encoder_inputs.repository
        else {
            "type": "local",
            "checkpoint": encoder_inputs.checkpoint.name,
            "config": encoder_inputs.model_config.name,
            "normalization": encoder_inputs.normalization.name,
        }
    )
    dataset_source = (
        {
            "type": "huggingface",
            "repository": dataset_inputs.repository,
            "revision": dataset_inputs.revision,
        }
        if dataset_inputs.repository
        else {
            "type": "local",
            "manifest": dataset_inputs.manifest.name,
            "data_config": dataset_inputs.data_config.name,
        }
    )
    metadata = {
        "schema_version": "egms-tokens-1.1",
        "encoder_source": encoder_source,
        "dataset_source": dataset_source,
        "coord_scale": float(model_config["coord_scale_m"]),
        "encoder_config": dict(model_config),
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
        "input_contract": {
            "stored_steps": int(tw.stored_steps),
            "stored_window": f"[{tw.t_start},{tw.t_end})",
            "original_source_steps": int(
                configured_axis.get("original_source_steps", tw.stored_steps)
            ),
            "original_window": (
                f"[{configured_axis.get('original_t_start', tw.t_start)},"
                f"{configured_axis.get('original_t_end', tw.t_end)})"
            ),
            "original_index_offset": int(
                configured_axis.get("original_index_offset", tw.t_start)
            ),
            "cadence_days": configured_axis.get("cadence_days"),
        },
        "token_layout": (
            f"index 0 = tile summary; 1..{n_patch} = "
            f"{args.grid}x{args.grid} cells in row-major order"
        ),
        "extraction_module": "egms_encoder.extract_tokens",
        "extraction_date_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
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
    metadata_path = out_dir / f"{Path(output_name).stem}_metadata.json"
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=str)
    print(f"\nwrote {out_pt}", flush=True)
    print(f"wrote {metadata_path}", flush=True)


if __name__ == "__main__":
    main()
