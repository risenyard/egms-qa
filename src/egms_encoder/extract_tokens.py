"""Extract fixed EGMS tokens from published HF or compatible local tiles."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from egms_encoder import __version__
from egms_encoder.checkpoint import load_encoder_checkpoint, load_normalization
from egms_encoder.data.tile_store import FEATURE_COLUMNS_COUNT, TileStore

DEFAULT_ENCODER_REPO = "risenyard/egms-qa-encoder"
DEFAULT_DATASET_REPO = "risenyard/egms-qa-dataset"
DEFAULT_TILE_SIZE = 7000.0
DEFAULT_GRID_SIZE = 8
TOKEN_SCHEMA = "egms-tokens-1.1"
COMMIT_SHA = re.compile(r"^[0-9a-fA-F]{40}$")


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
    parser = argparse.ArgumentParser(
        description=(
            "Run EGMS-QA Encoder on the published HF dataset or compatible "
            "local 294-step tiles."
        )
    )
    parser.add_argument("--encoder-repo", default=DEFAULT_ENCODER_REPO)
    parser.add_argument("--encoder-revision", default="main")
    parser.add_argument("--dataset-repo", default=DEFAULT_DATASET_REPO)
    parser.add_argument("--dataset-revision", default="main")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument(
        "--checkpoint",
        default="",
        help=(
            "Local encoder.safetensors override; requires --model-config and "
            "--normalization."
        ),
    )
    parser.add_argument("--model-config", default="")
    parser.add_argument("--normalization", default="")
    parser.add_argument(
        "--manifest",
        default="",
        help="Local manifest override; requires --data-config.",
    )
    parser.add_argument("--data-config", default="")
    parser.add_argument(
        "--source-tiles-root",
        default="",
        help=(
            "Optional local directory corresponding to artifacts/source_tiles; "
            "use when manifest paths are not relative to the working directory."
        ),
    )
    parser.add_argument("--output-dir", default="outputs/tokens")
    parser.add_argument("--output-name", default="")
    parser.add_argument("--grid-size", type=int, default=DEFAULT_GRID_SIZE)
    parser.add_argument("--tile-size", type=float, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-tiles", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=500)
    return parser.parse_args(argv)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _published_tile_repo_path(value: object) -> Path:
    path = Path(str(value))
    if path.parts[:2] == ("data", "tiles"):
        return Path("artifacts/source_tiles").joinpath(*path.parts[2:])
    if path.parts[:2] == ("artifacts", "source_tiles"):
        return path
    raise ValueError(f"manifest path is outside the published source-tile tree: {path}")


def _resolve_revision(repository: str, revision: str, *, repo_type: str) -> str:
    if COMMIT_SHA.fullmatch(revision):
        return revision.lower()
    from huggingface_hub import HfApi

    api = HfApi()
    if repo_type == "dataset":
        return str(api.dataset_info(repository, revision=revision).sha)
    return str(api.model_info(repository, revision=revision).sha)


def resolve_encoder_inputs(args: argparse.Namespace) -> EncoderInputs:
    local_values = (args.checkpoint, args.model_config, args.normalization)
    if any(local_values):
        if not all(local_values):
            raise ValueError(
                "local encoder overrides require --checkpoint, --model-config, "
                "and --normalization together"
            )
        return EncoderInputs(
            checkpoint=Path(args.checkpoint),
            model_config=Path(args.model_config),
            normalization=Path(args.normalization),
            repository=None,
            revision=None,
        )

    from huggingface_hub import hf_hub_download

    revision = _resolve_revision(
        args.encoder_repo, args.encoder_revision, repo_type="model"
    )

    def download(filename: str) -> Path:
        return Path(
            hf_hub_download(
                repo_id=args.encoder_repo,
                filename=filename,
                revision=revision,
                cache_dir=args.cache_dir,
            )
        )

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
                "local dataset overrides require --manifest and --data-config together"
            )
        return DatasetInputs(
            manifest=Path(args.manifest),
            data_config=Path(args.data_config),
            source_tiles_root=(
                Path(args.source_tiles_root) if args.source_tiles_root else None
            ),
            repository=None,
            revision=None,
        )

    from huggingface_hub import hf_hub_download, snapshot_download

    revision = _resolve_revision(
        args.dataset_repo, args.dataset_revision, repo_type="dataset"
    )
    manifest = Path(
        hf_hub_download(
            repo_id=args.dataset_repo,
            repo_type="dataset",
            filename="metadata/split_manifest.parquet",
            revision=revision,
            cache_dir=args.cache_dir,
        )
    )
    data_config = Path(
        hf_hub_download(
            repo_id=args.dataset_repo,
            repo_type="dataset",
            filename="metadata/data_config.json",
            revision=revision,
            cache_dir=args.cache_dir,
        )
    )
    if args.max_tiles is not None and args.max_tiles <= 0:
        raise ValueError("--max-tiles must be positive")
    if args.max_tiles is None:
        tile_patterns = ["artifacts/source_tiles/**/*.npz"]
    else:
        manifest_rows = pd.read_parquet(manifest, columns=["path"])
        tile_patterns = [
            _published_tile_repo_path(value).as_posix()
            for value in manifest_rows["path"].iloc[: args.max_tiles]
        ]
    snapshot = Path(
        snapshot_download(
            repo_id=args.dataset_repo,
            repo_type="dataset",
            revision=revision,
            cache_dir=args.cache_dir,
            allow_patterns=tile_patterns,
        )
    )
    return DatasetInputs(
        manifest=manifest,
        data_config=data_config,
        source_tiles_root=snapshot / "artifacts/source_tiles",
        repository=args.dataset_repo,
        revision=revision,
    )


def load_tile_store(
    manifest_path: Path,
    data_config_path: Path,
    source_tiles_root: Path | None = None,
) -> tuple[TileStore, Any]:
    print(f"[manifest] reading {manifest_path}", flush=True)
    store = TileStore.from_manifest(
        manifest_path,
        data_config_path,
        source_tiles_root=source_tiles_root,
        require_static_fields=False,
    )
    print(
        f"[manifest] {store.num_tiles} tiles, "
        f"input_length={store.time_window.input_length}",
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
    width = embedding.shape[1]
    cell_count = grid_size * grid_size
    tokens = np.zeros((cell_count + 1, width), dtype=np.float32)
    mask = np.zeros(cell_count + 1, dtype=bool)
    tokens[0] = embedding.mean(axis=0)
    mask[0] = True
    half = tile_size * 0.5
    bx = np.clip(
        np.floor((centered_coords[:, 0] + half) / tile_size * grid_size).astype(
            np.int64
        ),
        0,
        grid_size - 1,
    )
    by = np.clip(
        np.floor((centered_coords[:, 1] + half) / tile_size * grid_size).astype(
            np.int64
        ),
        0,
        grid_size - 1,
    )
    bin_indices = by * grid_size + bx
    counts = np.bincount(bin_indices, minlength=cell_count).astype(np.int32)
    for cell in range(cell_count):
        selected = bin_indices == cell
        if selected.any():
            tokens[1 + cell] = embedding[selected].mean(axis=0)
            mask[1 + cell] = True
    return tokens, mask, counts


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("CUDA unavailable, running CPU (slow).", flush=True)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    encoder_inputs = resolve_encoder_inputs(args)
    dataset_inputs = resolve_dataset_inputs(args)
    normalizer = load_normalization(encoder_inputs.normalization)
    norm_mean = float(normalizer["mean"])
    norm_std = float(normalizer["std"])
    print(f"[norm] mean={norm_mean:.4f} std={norm_std:.4f}", flush=True)

    print(f"[load] {encoder_inputs.checkpoint}", flush=True)
    model, model_config = load_encoder_checkpoint(
        encoder_inputs.checkpoint,
        encoder_inputs.model_config,
        device,
    )
    print(
        f"[encoder] d_model={model_config['d_model']}, "
        f"layers={model_config['spatial_layers']}, "
        f"heads={model_config['spatial_heads']}",
        flush=True,
    )
    input_length = int(model_config["input_length"])
    tile_size = float(args.tile_size or DEFAULT_TILE_SIZE)
    if tile_size <= 0:
        raise ValueError("--tile-size must be positive")

    store, manifest = load_tile_store(
        dataset_inputs.manifest,
        dataset_inputs.data_config,
        dataset_inputs.source_tiles_root,
    )
    time_window = store.time_window
    if time_window.input_length != input_length:
        raise ValueError(
            f"time_window length {time_window.input_length} != "
            f"encoder input_length {input_length}"
        )

    total_tiles = store.num_tiles
    n_tiles = total_tiles if args.max_tiles is None else min(total_tiles, args.max_tiles)
    if n_tiles <= 0:
        raise ValueError("--max-tiles must select at least one tile")
    if args.grid_size <= 0:
        raise ValueError("--grid-size must be positive")
    if args.log_every <= 0:
        raise ValueError("--log-every must be positive")
    cell_count = args.grid_size * args.grid_size
    token_count = cell_count + 1
    model_width = int(model_config["d_model"])

    spatial_tokens = np.zeros(
        (n_tiles, token_count, model_width), dtype=np.float32
    )
    token_mask = np.zeros((n_tiles, token_count), dtype=bool)
    tile_indices = np.arange(n_tiles, dtype=np.int32)
    tile_ids = manifest["tile_id"].astype(str).values[:n_tiles]
    splits = manifest["split"].astype(str).values[:n_tiles]
    point_count_per_bin = np.zeros((n_tiles, cell_count), dtype=np.int32)
    n_points_per_tile = np.zeros(n_tiles, dtype=np.int32)

    autocast_device = "cuda" if device.type == "cuda" else "cpu"
    start_time = time.monotonic()
    with torch.no_grad(), torch.amp.autocast(
        autocast_device,
        dtype=torch.bfloat16,
        enabled=device.type == "cuda",
    ):
        for tile_index in range(n_tiles):
            tile_data = store.get_tile(tile_index)
            point_count = tile_data.shape[0]
            n_points_per_tile[tile_index] = point_count
            coords = tile_data[:, :2].copy()
            series = tile_data[
                :,
                FEATURE_COLUMNS_COUNT : FEATURE_COLUMNS_COUNT + input_length,
            ].copy()
            series = (series - norm_mean) / norm_std
            series = np.nan_to_num(series, nan=0.0, posinf=0.0, neginf=0.0)
            centered_coords = (coords - coords.mean(axis=0, keepdims=True)).astype(
                np.float32
            )

            series_tensor = torch.from_numpy(series).unsqueeze(0).to(device)
            coords_tensor = torch.from_numpy(centered_coords).unsqueeze(0).to(device)
            point_mask = torch.ones(
                1, point_count, dtype=torch.bool, device=device
            )
            embedding = (
                model.encode(
                    series_tensor,
                    coords=coords_tensor,
                    point_mask=point_mask,
                )
                .squeeze(0)
                .float()
                .cpu()
                .numpy()
            )
            tokens, mask, counts = pool_to_spatial_tokens(
                embedding,
                centered_coords,
                args.grid_size,
                tile_size,
            )
            spatial_tokens[tile_index] = tokens
            token_mask[tile_index] = mask
            point_count_per_bin[tile_index] = counts

            if (tile_index + 1) % args.log_every == 0 or tile_index == n_tiles - 1:
                elapsed = time.monotonic() - start_time
                rate = (tile_index + 1) / elapsed
                eta = (n_tiles - tile_index - 1) / max(rate, 1e-6)
                print(
                    f"  {tile_index + 1}/{n_tiles} elapsed={elapsed:.1f}s "
                    f"rate={rate:.2f} tiles/s eta={eta:.0f}s",
                    flush=True,
                )

    occupancy = (point_count_per_bin > 0).mean(axis=1)
    print(
        f"occupancy mean={occupancy.mean():.3f} "
        f"median={np.median(occupancy):.3f}",
        flush=True,
    )
    print(
        f"n_points mean={n_points_per_tile.mean():.1f} "
        f"min={n_points_per_tile.min()} max={n_points_per_tile.max()}",
        flush=True,
    )

    output_name = args.output_name or (
        "egms_tokens_10k.pt" if n_tiles == 10_000 else f"egms_tokens_{n_tiles}.pt"
    )
    output_path = output_dir / output_name
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
            "token_count": token_count,
            "token_width": model_width,
            "token_layout": (
                f"index 0 = tile summary; 1..{cell_count} = "
                f"{args.grid_size}x{args.grid_size} spatial cells in row-major order"
            ),
        },
        "reproduction_input_sha256": {
            "encoder_weights": _sha256(encoder_inputs.checkpoint),
            "encoder_config": _sha256(encoder_inputs.model_config),
            "normalization": _sha256(encoder_inputs.normalization),
            "split_manifest": _sha256(dataset_inputs.manifest),
            "data_config": _sha256(dataset_inputs.data_config),
        },
        "extraction": {
            "module": "egms_encoder.extract_tokens",
            "date_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        },
        "statistics": {
            "tiles": n_tiles,
            "bin_occupancy_mean": float(occupancy.mean()),
            "bin_occupancy_median": float(np.median(occupancy)),
            "points_per_tile_mean": float(n_points_per_tile.mean()),
            "points_per_tile_min": int(n_points_per_tile.min()),
            "points_per_tile_max": int(n_points_per_tile.max()),
        },
        "output_file": output_name,
    }
    source_repositories = {}
    source_revisions = {}
    if encoder_inputs.repository:
        source_repositories["encoder"] = encoder_inputs.repository
        source_revisions["encoder"] = encoder_inputs.revision
    if dataset_inputs.repository:
        source_repositories.update(
            {
                "dataset": dataset_inputs.repository,
                "tiles": "artifacts/source_tiles",
            }
        )
        source_revisions["dataset"] = dataset_inputs.revision
    if source_repositories:
        metadata["source_repositories"] = source_repositories
        metadata["source_revisions"] = source_revisions

    torch.save(
        {
            "spatial_tokens": torch.from_numpy(spatial_tokens),
            "token_mask": torch.from_numpy(token_mask),
            "tile_indices": torch.from_numpy(tile_indices),
            "tile_ids": list(tile_ids),
            "splits": list(splits),
            "point_count_per_bin": torch.from_numpy(point_count_per_bin),
            "n_points_per_tile": torch.from_numpy(n_points_per_tile),
            "metadata": metadata,
        },
        output_path,
    )
    metadata_path = output_dir / f"{Path(output_name).stem}_metadata.json"
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
        handle.write("\n")
    print(f"wrote {output_path}", flush=True)
    print(f"wrote {metadata_path}", flush=True)


if __name__ == "__main__":
    main()
