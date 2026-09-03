"""V4 training entrypoint: tile-aware encoder on the all-Europe V4 pool.

Forked from `train_tile_aware.py` (V3.3 entrypoint). V3.3 file is not
modified. Differences vs V3.3:
  * Data loading uses LazyTileStore.from_config (per-tile npz, no
    eager parquet load).
  * Train/val/test split is precomputed in data/processed/v4/v4_split.json;
    legacy --val-fraction / --split-strategy / --stratify-bins are
    accepted but ignored.
  * Normalization is loaded from data/processed/v4/v4_normalization.json;
    `fit_tile_normalizer` is not called.
  * `--input-length` is locked to the V4 time window (default 294) and
    cannot be overridden above the configured window.
All other training logic (train_step, prepare_batch, build_model,
build_optimizer, evaluate, loss, scheduler, checkpointing) is reused
unchanged from V3.3.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import deque
from contextlib import nullcontext
from itertools import islice
from pathlib import Path

import numpy as np
import torch
from torch import nn

from egms_encoder.data.lazy_tile_store import LazyTileStore
from egms_encoder.data.tile_store import iter_tile_batches
from egms_encoder.models.tile_encoder import TileEncoder

STATIC_COLUMNS = [
    "height", "rmse", "mean_velocity", "mean_velocity_std",
    "acceleration", "acceleration_std", "seasonality", "seasonality_std",
]
FEATURE_COLUMNS = ["easting", "northing", *STATIC_COLUMNS]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="V4 tile-aware masked reconstruction training (all-Europe pool).")
    # V4 data — replaces V3.3's --data-path / --metadata-path
    p.add_argument("--v4-config", default="data/processed/v4/v4_data_config.json",
                   help="V4 data config JSON (LazyTileStore + split + window).")
    p.add_argument("--v4-normalization", default="data/processed/v4/v4_normalization.json",
                   help="Precomputed normalization JSON (skip fit step).")
    p.add_argument("--output-dir", default="outputs/encoder_pretrain")
    # Tile parameters
    p.add_argument("--tile-size", type=float, default=7000.0, help="Tile side length in metres")
    p.add_argument("--min-tile-points", type=int, default=200, help="Discard tiles with fewer points")
    p.add_argument("--max-tile-points", type=int, default=2048, help="Truncate tiles larger than this (dense attention is O(N^2))")
    p.add_argument("--tiles-per-batch", type=int, default=16, help="Number of tiles per GPU batch")
    # Model parameters
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--num-layers", type=int, default=6)
    p.add_argument("--num-heads", type=int, default=8)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--input-length", type=int, default=None,
                   help="Number of time steps to use; defaults to all metadata time columns")
    # Masking
    p.add_argument("--mask-ratio", type=float, default=0.3)
    p.add_argument("--mask-strategy", default="block", choices=["random", "block"])
    p.add_argument("--sync-mask", dest="sync_mask", action=argparse.BooleanOptionalAction, default=True,
                   help="Synchronized masking: all points in a tile share the same time mask. "
                        "V4 default ON. Disabling lets points 'borrow' masked-position values from "
                        "neighbors, which drops reconstruction loss but corrupts embedding quality "
                        "(V3.x experiments confirmed ACC probe R^2 collapsed from 0.84 to 0.45).")
    p.add_argument("--mask-schedule", default="fixed", choices=["fixed", "short_mix"],
                   help="V3.1: training mask schedule; validation uses eval_mask_ratio")
    p.add_argument("--eval-mask-ratio", type=float, default=0.30,
                   help="V3.1: fixed validation mask ratio")
    # Model version
    p.add_argument("--model-version", default="v1", choices=["v1", "v2", "v3", "v3_1", "v3_3"],
                   help="v1=Linear temporal, v2=PatchTST, v3=v2+residual loss, v3_1/v3_3=v2+residual head")
    p.add_argument("--patch-size", type=int, default=16, help="V2: time patch size")
    p.add_argument("--temporal-layers", type=int, default=2, help="V2: temporal Transformer layers")
    p.add_argument("--temporal-heads", type=int, default=4, help="V2: temporal attention heads")
    p.add_argument("--residual-loss-weight", type=float, default=0.0,
                   help="V3/V3.1: residual auxiliary loss weight")
    p.add_argument("--residual-consistency-weight", type=float, default=0.1,
                   help="V3.1: weight for final reconstruction residual consistency")
    p.add_argument("--residual-head-mode", default="additive", choices=["additive", "aux_only"],
                   help="V3.1: add residual correction to reconstruction or train it only as an auxiliary head")
    p.add_argument("--coord-scale", type=float, default=None,
                   help="Divide centered coords by this before coord_embedding (e.g. 3500 = tile half-width). "
                        "None keeps legacy raw-metre coords (~+-3700).")
    p.add_argument("--residual-head-lr", type=float, default=None,
                   help="V3.1: optional learning rate for residual head parameters")
    p.add_argument("--init-from-checkpoint", default=None,
                   help="Warm-start compatible model weights without loading optimizer/scaler state")
    p.add_argument("--point-sampling", default="uniform", choices=["uniform", "residual_weighted"],
                   help="V3.1: point sampling strategy for oversized training tiles")
    p.add_argument("--residual-sampling-alpha", type=float, default=0.5,
                   help="V3.1: fraction of oversized tile points sampled by residual RMS")
    # Training
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--duration-hours", type=float, default=None)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--min-lr", type=float, default=3e-5)
    p.add_argument("--lr-scheduler", default="cosine", choices=["none", "cosine"])
    p.add_argument("--scheduler-total-steps", type=int, default=20000)
    p.add_argument("--warmup-steps", type=int, default=1000)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--precision", default="bf16", choices=["fp32", "bf16", "fp16"])
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=42)
    # Validation
    p.add_argument("--split-strategy", default="random", choices=["random", "stratified"],
                   help="Tile split strategy for train/val/test assignment")
    p.add_argument("--val-fraction", type=float, default=0.05)
    p.add_argument("--test-fraction", type=float, default=0.0,
                   help="Held-out test tile fraction reserved from training and validation")
    p.add_argument("--stratify-bins", type=int, default=3,
                   help="Number of quantile bins per descriptor for stratified tile splitting")
    p.add_argument("--val-batches", type=int, default=16)
    p.add_argument(
        "--resample-val-batches",
        action="store_true",
        help="Draw a new, reproducible set of non-overlapping validation batches at each validation step.",
    )
    p.add_argument("--val-every-steps", type=int, default=500)
    p.add_argument("--val-seed", type=int, default=1729)
    # Logging / checkpointing
    p.add_argument("--checkpoint-every-steps", type=int, default=1000)
    p.add_argument("--log-every-steps", type=int, default=20)
    p.add_argument("--train-window-steps", type=int, default=1000)
    p.add_argument("--resume-from", default=None)
    return p.parse_args()


def collect_validation_batches(args, tile_store, rng, *, resampled: bool) -> list[dict]:
    batches = iter_tile_batches(
        tile_store, args.tiles_per_batch,
        split="val", val_fraction=args.val_fraction, split_seed=args.val_seed,
        rng=rng,
        test_fraction=args.test_fraction,
        split_strategy=args.split_strategy,
        stratify_bins=args.stratify_bins,
        max_batches=None if resampled else args.val_batches,
        max_points=args.max_tile_points,
        feature_columns_count=len(FEATURE_COLUMNS), input_length=args.input_length,
        point_sampling="uniform", residual_sampling_alpha=args.residual_sampling_alpha,
    )
    if resampled:
        batches = islice(batches, args.val_batches)
    return list(batches)


def validation_tile_ids(val_batches) -> list[int]:
    return [int(idx) for batch in val_batches for idx in batch["tile_indices"]]


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load V4 data via LazyTileStore (replaces V3.3 eager TileStore.from_parquet)
    tile_store = LazyTileStore.from_config(args.v4_config)
    v4_input_length = tile_store.time_window.input_length
    if args.input_length is None:
        args.input_length = v4_input_length
    elif args.input_length > v4_input_length:
        raise ValueError(
            f"Requested input_length={args.input_length} exceeds V4 window {v4_input_length}"
        )
    elif args.input_length < v4_input_length:
        print(
            f"NOTE: --input-length={args.input_length} < V4 window {v4_input_length} "
            f"(trailing steps will be unused)",
            flush=True,
        )
    with (output_dir / "args.json").open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, sort_keys=True)

    train_indices = tile_store.split_tile_indices("train")
    val_indices = tile_store.split_tile_indices("val")
    test_indices = tile_store.split_tile_indices("test")
    print(
        f"V4 tile split (precomputed from {args.v4_config}): "
        f"train={len(train_indices)} val={len(val_indices)} test={len(test_indices)}",
        flush=True,
    )

    # Load precomputed normalization (skip V3.3's fit_tile_normalizer)
    with open(args.v4_normalization) as f:
        normalizer = json.load(f)
    normalizer.pop("_meta", None)  # strip annotation block before passing into trainer
    with (output_dir / "normalization.json").open("w", encoding="utf-8") as f:
        json.dump(normalizer, f, indent=2)
    print(
        f"V4 normalizer: mean={normalizer['mean']:.6f} std={normalizer['std']:.6f} "
        f"residual_std={normalizer.get('residual_std',1.0):.6f}",
        flush=True,
    )

    # Build model
    model = build_model(args, normalizer).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model: {param_count:,} parameters ({param_count/1e6:.1f}M)", flush=True)

    if args.init_from_checkpoint:
        load_init_checkpoint(model, Path(args.init_from_checkpoint), device, args.model_version)

    optimizer = build_optimizer(args, model)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and args.precision == "fp16")

    step = 0
    epoch = 0
    best_val_loss = float("inf")
    resume_elapsed_hours = 0.0
    resume_train_losses: list[float] = []
    if args.resume_from:
        checkpoint = torch.load(args.resume_from, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])
        step = int(checkpoint.get("step", 0))
        epoch = int(checkpoint.get("epoch", 0))
        best_val_loss = float(checkpoint.get("best_val_loss", float("inf")))
        resume_elapsed_hours, resume_train_losses = read_resume_metrics(
            output_dir / "metrics.csv", args.train_window_steps,
        )
        if resume_elapsed_hours > 0:
            train_step._start = time.monotonic() - resume_elapsed_hours * 3600
        print(
            f"resumed from {args.resume_from}: step={step} epoch={epoch} "
            f"best_val_loss={best_val_loss:.6f}",
            flush=True,
        )

    # Fixed validation batches preserve the legacy protocol. Resampled validation
    # draws a reproducible subset from a fresh permutation at every validation step.
    seen_val_tile_ids: set[int] = set()
    if args.resample_val_batches:
        val_batches = []
        val_tile_count = len(tile_store.split_tile_indices(
            "val", args.val_fraction, args.val_seed,
            test_fraction=args.test_fraction,
            split_strategy=args.split_strategy,
            stratify_bins=args.stratify_bins,
        ))
        print(
            f"validation batches will be resampled: {args.val_batches} batches x "
            f"{args.tiles_per_batch} tiles from {val_tile_count} validation tiles",
            flush=True,
        )
    else:
        val_batches = collect_validation_batches(
            args, tile_store, np.random.default_rng(args.val_seed), resampled=False,
        )
        val_tile_ids = validation_tile_ids(val_batches)
        print(
            f"loaded {len(val_batches)} validation tile-batches: "
            f"presentations={len(val_tile_ids)} unique_tiles={len(set(val_tile_ids))}",
            flush=True,
        )

    # Training loop
    metrics_path = output_dir / "metrics.csv"
    start_time = time.monotonic()
    deadline = start_time + args.duration_hours * 3600 if args.duration_hours else None
    train_rng = np.random.default_rng(args.seed)
    data_rng = np.random.default_rng(args.seed + 1)
    loss_window = MetricWindow(args.train_window_steps)
    for loss_value in resume_train_losses:
        loss_window.add(loss_value)

    fieldnames = [
        "step", "epoch", "loss", "global_loss", "residual_loss",
        "base_global_loss", "residual_head_loss", "residual_std",
        "rmse", "mae", "train_loss_window", "lr",
        "grad_norm", "val_loss", "val_global_loss", "val_residual_loss",
        "val_base_global_loss", "val_residual_head_loss", "val_residual_std",
        "val_rmse", "val_mae", "val_global_rmse", "val_global_mae", "val_gap",
        "elapsed_hours", "peak_memory_mb", "tiles_in_batch", "max_points_in_batch",
    ]

    append_metrics = bool(args.resume_from and metrics_path.exists() and step > 0)
    with metrics_path.open("a" if append_metrics else "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not append_metrics:
            writer.writeheader()
        should_stop = False

        while not should_stop:
            epoch += 1
            batches = iter_tile_batches(
                tile_store, args.tiles_per_batch,
                split="train", val_fraction=args.val_fraction, split_seed=args.val_seed,
                rng=data_rng,
                test_fraction=args.test_fraction,
                split_strategy=args.split_strategy,
                stratify_bins=args.stratify_bins,
                max_points=args.max_tile_points,
                feature_columns_count=len(FEATURE_COLUMNS), input_length=args.input_length,
                point_sampling=args.point_sampling, residual_sampling_alpha=args.residual_sampling_alpha,
            )

            for tile_batch in batches:
                step += 1
                current_lr = get_lr(args, step)
                set_optimizer_lrs(args, optimizer, current_lr)

                row = train_step(args, model, optimizer, scaler, tile_batch, device, train_rng, normalizer)
                row["step"] = step
                row["epoch"] = epoch
                row["lr"] = current_lr
                loss_window.add(row["loss"])
                row["train_loss_window"] = loss_window.mean

                # Validation
                if args.val_every_steps > 0 and step % args.val_every_steps == 0:
                    if args.resample_val_batches:
                        val_rng = np.random.default_rng(np.random.SeedSequence([args.val_seed, step]))
                        val_batches = collect_validation_batches(args, tile_store, val_rng, resampled=True)
                        sampled_tile_ids = validation_tile_ids(val_batches)
                        seen_val_tile_ids.update(sampled_tile_ids)
                        print(
                            f"validation sample step={step}: presentations={len(sampled_tile_ids)} "
                            f"unique={len(set(sampled_tile_ids))} "
                            f"cumulative_unique={len(seen_val_tile_ids)}/{val_tile_count}",
                            flush=True,
                        )

                    val_metrics = evaluate(args, model, val_batches, device, normalizer)
                    row["val_loss"] = val_metrics["loss"]
                    row["val_global_loss"] = val_metrics["global_loss"]
                    row["val_residual_loss"] = val_metrics["residual_loss"]
                    row["val_base_global_loss"] = val_metrics["base_global_loss"]
                    row["val_residual_head_loss"] = val_metrics["residual_head_loss"]
                    row["val_residual_std"] = val_metrics["residual_std"]
                    row["val_rmse"] = val_metrics["rmse"]
                    row["val_mae"] = val_metrics["mae"]
                    row["val_global_rmse"] = val_metrics["global_rmse"]
                    row["val_global_mae"] = val_metrics["global_mae"]
                    row["val_gap"] = val_metrics["loss"] - loss_window.mean
                    if val_metrics["loss"] < best_val_loss:
                        best_val_loss = val_metrics["loss"]
                        save_checkpoint(output_dir / "best.pt", model, optimizer, scaler, args, step, epoch, best_val_loss)

                writer.writerow(row)
                if step % args.log_every_steps == 0:
                    val_msg = ""
                    if "val_loss" in row and row["val_loss"] is not None:
                        val_msg = (
                            f" val={row['val_loss']:.6f}"
                            f" val_g={row['val_global_loss']:.6f}"
                            f" val_r={row['val_residual_loss']:.6f}"
                            f" val_hr={row['val_residual_head_loss']:.6f}"
                            f" gap={row['val_gap']:.4f}"
                        )
                    print(
                        f"step={step} ep={epoch} loss={row['loss']:.6f} "
                        f"g={row['global_loss']:.6f} r={row['residual_loss']:.6f} "
                        f"hr={row['residual_head_loss']:.6f} "
                        f"rmse={row['rmse']:.6f} "
                        f"mae={row['mae']:.6f} win={row['train_loss_window']:.6f} lr={current_lr:.2e}"
                        f"{val_msg} h={row['elapsed_hours']:.3f} mem={row['peak_memory_mb']:.0f}MB "
                        f"tiles={row['tiles_in_batch']} pts={row['max_points_in_batch']}",
                        flush=True,
                    )
                    handle.flush()

                if step % args.checkpoint_every_steps == 0:
                    save_checkpoint(output_dir / "latest.pt", model, optimizer, scaler, args, step, epoch, best_val_loss)

                if args.max_steps and step >= args.max_steps:
                    should_stop = True
                    break
                if deadline and time.monotonic() >= deadline:
                    should_stop = True
                    break

    save_checkpoint(output_dir / "latest.pt", model, optimizer, scaler, args, step, epoch, best_val_loss)
    print(f"Training complete: {step} steps, best_val_loss={best_val_loss:.6f}")
    print(f"wrote {metrics_path}")
    print(f"wrote {output_dir / 'latest.pt'}")


def build_model(args, normalizer):
    """Construct the EGMS tile encoder."""
    residual_scale = float((normalizer or {}).get("residual_std", 1.0))
    return TileEncoder(
        input_length=args.input_length,
        d_model=args.d_model,
        patch_size=args.patch_size,
        temporal_layers=args.temporal_layers,
        temporal_heads=args.temporal_heads,
        spatial_layers=args.num_layers,
        spatial_heads=args.num_heads,
        dropout=args.dropout,
        residual_scale=residual_scale,
        residual_head_mode=args.residual_head_mode,
        coord_scale=getattr(args, "coord_scale", None),
    )


def build_optimizer(args, model):
    """Use an optional dedicated LR for the V3.1 residual head."""
    if args.model_version not in ("v3_1", "v3_3") or args.residual_head_lr is None:
        return torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    residual_ids = {id(p) for p in model.residual_head.parameters()}
    base_params = [p for p in model.parameters() if id(p) not in residual_ids]
    head_params = list(model.residual_head.parameters())
    return torch.optim.AdamW(
        [
            {"params": base_params, "lr": args.lr},
            {"params": head_params, "lr": args.residual_head_lr},
        ],
        weight_decay=args.weight_decay,
    )


def set_optimizer_lrs(args, optimizer, current_lr: float) -> None:
    """Update scheduler-controlled base LR while preserving residual head LR."""
    if args.model_version in ("v3_1", "v3_3") and args.residual_head_lr is not None and len(optimizer.param_groups) > 1:
        optimizer.param_groups[0]["lr"] = current_lr
        optimizer.param_groups[1]["lr"] = args.residual_head_lr
        return
    for group in optimizer.param_groups:
        group["lr"] = current_lr


def load_init_checkpoint(model, checkpoint_path: Path, device, model_version: str) -> None:
    """Warm-start weights without loading optimizer/scaler state."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = checkpoint["model"]
    if model_version in ("v3_1", "v3_3"):
        incompatible = model.load_state_dict(state, strict=False)
        allowed_missing = {
            "residual_scale",
            "residual_head.0.weight",
            "residual_head.0.bias",
            "residual_head.1.weight",
            "residual_head.1.bias",
            "residual_head.4.weight",
            "residual_head.4.bias",
        }
        missing = set(incompatible.missing_keys)
        unexpected = set(incompatible.unexpected_keys)
        bad_missing = missing - allowed_missing
        if bad_missing or unexpected:
            raise RuntimeError(
                "Unexpected init checkpoint mismatch: "
                f"missing={sorted(bad_missing)} unexpected={sorted(unexpected)}"
            )
    else:
        model.load_state_dict(state)
    print(f"initialized weights from {checkpoint_path}", flush=True)


def train_step(args, model, optimizer, scaler, tile_batch, device, rng, normalizer):
    model.train()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    optimizer.zero_grad(set_to_none=True)

    series, coords, point_mask, loss_mask, target = prepare_batch(
        tile_batch, args, rng, normalizer, device, is_eval=False,
    )

    with autocast_context(device, args.precision):
        out = model(series, coords=coords, point_mask=point_mask)
        losses = reconstruction_losses(out, target.float(), loss_mask, args, normalizer)
        loss = losses["loss"]
        global_loss = losses["global_loss"]
        residual_loss = losses["residual_loss"]
        base_global_loss = losses["base_global_loss"]
        residual_head_loss = losses["residual_head_loss"]
        mae = losses["mae"]
        rmse = losses["rmse"]

    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    if torch.isfinite(grad_norm):
        scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

    peak_mb = torch.cuda.max_memory_allocated(device) / 1024**2 if device.type == "cuda" else 0.0
    start_time_ref = getattr(train_step, "_start", None)
    if start_time_ref is None:
        train_step._start = time.monotonic()
        start_time_ref = train_step._start

    return {
        "loss": float(loss.detach().cpu()),
        "global_loss": float(global_loss.detach().cpu()),
        "residual_loss": float(residual_loss.detach().cpu()),
        "base_global_loss": float(base_global_loss.detach().cpu()),
        "residual_head_loss": float(residual_head_loss.detach().cpu()),
        "residual_std": float(losses["residual_std"]),
        "rmse": float(rmse.detach().cpu()),
        "mae": float(mae.detach().cpu()),
        "grad_norm": float(grad_norm.detach().cpu()),
        "elapsed_hours": (time.monotonic() - start_time_ref) / 3600,
        "peak_memory_mb": peak_mb,
        "tiles_in_batch": int(tile_batch["series"].shape[0]),
        "max_points_in_batch": int(tile_batch["series"].shape[1]),
    }


def prepare_batch(tile_batch, args, rng, normalizer, device, is_eval=False):
    """Prepare a tile batch for training: normalize, mask, convert to tensors."""
    series_np = tile_batch["series"].copy()  # [B, N_max, T]
    coords_np = tile_batch["coords"].copy()  # [B, N_max, 2]
    pmask_np = tile_batch["point_mask"]       # [B, N_max]

    B, N_max, T = series_np.shape

    # Normalize series
    finite = np.isfinite(series_np) & pmask_np[:, :, None]
    if normalizer is not None:
        series_np = (series_np - normalizer["mean"]) / normalizer["std"]
    series_np = np.nan_to_num(series_np, nan=0.0, posinf=0.0, neginf=0.0)
    target = series_np.copy()

    mask_ratio = effective_mask_ratio(args, rng, is_eval=is_eval)

    # Create time mask for reconstruction objective
    time_mask = np.zeros_like(series_np, dtype=bool)
    for b in range(B):
        n_real = int(pmask_np[b].sum())
        if n_real == 0:
            continue
        if args.mask_strategy == "block":
            block_len = max(1, int(round(T * mask_ratio)))
            if getattr(args, 'sync_mask', False):
                # Synchronized: one start per tile, all points share the same mask
                start = rng.integers(0, T - block_len + 1)
                time_mask[b, :n_real, start:start + block_len] = True
            else:
                # Independent: each point gets its own random start
                starts = rng.integers(0, T - block_len + 1, size=n_real)
                positions = np.arange(T)[None, :]
                time_mask[b, :n_real, :] = (positions >= starts[:, None]) & (positions < starts[:, None] + block_len)
        else:
            time_mask[b, :n_real, :] = rng.random((n_real, T)) < mask_ratio

    loss_mask = time_mask & finite
    # Ensure at least one masked position per batch
    for b in range(B):
        if not loss_mask[b].any():
            real_finite = finite[b]
            if real_finite.any():
                idx = rng.integers(int(real_finite.sum()))
                pos = np.flatnonzero(real_finite.ravel())[idx]
                loss_mask[b].ravel()[pos] = True

    # Zero out masked positions in input
    series_masked = series_np.copy()
    series_masked[time_mask | ~finite] = 0.0

    # Center coordinates per tile
    for b in range(B):
        n_real = int(pmask_np[b].sum())
        if n_real > 0:
            center = coords_np[b, :n_real].mean(axis=0, keepdims=True)
            coords_np[b, :n_real] -= center

    # Convert to tensors
    series_t = torch.from_numpy(series_masked).to(device, non_blocking=True)
    target_t = torch.from_numpy(target).to(device, non_blocking=True)
    coords_t = torch.from_numpy(coords_np).to(device, non_blocking=True)
    pmask_t = torch.from_numpy(pmask_np).to(device, non_blocking=True)
    loss_mask_t = torch.from_numpy(loss_mask).to(device, non_blocking=True)

    return series_t, coords_t, pmask_t, loss_mask_t, target_t


def effective_mask_ratio(args, rng, is_eval: bool) -> float:
    if is_eval:
        return float(getattr(args, "eval_mask_ratio", args.mask_ratio))
    if getattr(args, "mask_schedule", "fixed") == "short_mix":
        return float(rng.choice([0.10, 0.20, 0.30], p=[0.40, 0.35, 0.25]))
    return float(args.mask_ratio)


def reconstruction_losses(out, target, loss_mask, args, normalizer):
    """Compute masked global, residual, and V3.1 residual-head losses."""
    pred = out["reconstruction"].float()
    diff = pred - target
    masked_diff = diff[loss_mask]
    global_loss = masked_diff.square().mean()
    mae = masked_diff.abs().mean()
    rmse = global_loss.sqrt()

    pred_residual = linear_detrend(pred)
    target_residual = linear_detrend(target)
    residual_diff = pred_residual - target_residual
    residual_loss = residual_diff[loss_mask].square().mean()

    base_pred = out.get("base_reconstruction", pred).float()
    base_diff = base_pred - target
    base_global_loss = base_diff[loss_mask].square().mean()

    residual_std = float((normalizer or {}).get("residual_std", 1.0))
    residual_std = max(residual_std, 1e-6)
    if "residual_prediction" in out:
        residual_prediction_z = out["residual_prediction"].float()
        target_residual_z = target_residual / residual_std
        residual_head_loss = (residual_prediction_z - target_residual_z)[loss_mask].square().mean()
    else:
        residual_head_loss = residual_loss

    if args.model_version in ("v3_1", "v3_3"):
        loss = (
            global_loss
            + float(args.residual_loss_weight) * residual_head_loss
            + float(args.residual_consistency_weight) * residual_loss
        )
    else:
        loss = global_loss + float(args.residual_loss_weight) * residual_loss
    return {
        "loss": loss,
        "global_loss": global_loss,
        "residual_loss": residual_loss,
        "base_global_loss": base_global_loss,
        "residual_head_loss": residual_head_loss,
        "residual_std": residual_std,
        "rmse": rmse,
        "mae": mae,
    }


def linear_detrend(x):
    """Remove the best fixed linear fit along the time axis for each point series."""
    T = x.shape[-1]
    t = torch.linspace(-1.0, 1.0, T, device=x.device, dtype=x.dtype)
    t = t - t.mean()
    denom = (t * t).sum().clamp_min(torch.finfo(x.dtype).eps)
    intercept = x.mean(dim=-1, keepdim=True)
    slope = (x * t).sum(dim=-1, keepdim=True) / denom
    return x - (intercept + slope * t)


@torch.no_grad()
def evaluate(args, model, val_batches, device, normalizer):
    model.eval()
    total_se, total_ae, total_residual_se, total_base_se = 0.0, 0.0, 0.0, 0.0
    total_residual_head_se, total_weighted_loss, count = 0.0, 0.0, 0
    for tile_batch in val_batches:
        rng = np.random.default_rng(args.val_seed)
        series, coords, pmask, loss_mask, target = prepare_batch(
            tile_batch, args, rng, normalizer, device, is_eval=True,
        )
        with autocast_context(device, args.precision):
            out = model(series, coords=coords, point_mask=pmask)
            losses = reconstruction_losses(out, target.float(), loss_mask, args, normalizer)
        masked_diff = (out["reconstruction"].float() - target.float())[loss_mask]
        n = int(loss_mask.sum())
        total_se += float(losses["global_loss"].detach().cpu()) * n
        total_ae += float(masked_diff.abs().sum().cpu())
        total_residual_se += float(losses["residual_loss"].detach().cpu()) * n
        total_base_se += float(losses["base_global_loss"].detach().cpu()) * n
        total_residual_head_se += float(losses["residual_head_loss"].detach().cpu()) * n
        total_weighted_loss += float(losses["loss"].detach().cpu()) * n
        count += n
    model.train()
    mse = total_se / max(count, 1)
    residual_mse = total_residual_se / max(count, 1)
    base_mse = total_base_se / max(count, 1)
    residual_head_mse = total_residual_head_se / max(count, 1)
    total_loss = total_weighted_loss / max(count, 1)
    rmse = float(np.sqrt(mse))
    mae = total_ae / max(count, 1)
    return {
        "loss": total_loss,
        "global_loss": mse,
        "residual_loss": residual_mse,
        "base_global_loss": base_mse,
        "residual_head_loss": residual_head_mse,
        "residual_std": float((normalizer or {}).get("residual_std", 1.0)),
        "rmse": rmse,
        "mae": mae,
        "global_rmse": rmse,
        "global_mae": mae,
    }


def fit_tile_normalizer(tile_store, args):
    """Compute global mean/std from training tiles."""
    train_indices = tile_store.split_tile_indices(
        "train", args.val_fraction, args.val_seed,
        test_fraction=args.test_fraction,
        split_strategy=args.split_strategy,
        stratify_bins=args.stratify_bins,
    )
    total, sq_total, residual_sq_total, count, residual_count = 0.0, 0.0, 0.0, 0, 0
    fc = len(FEATURE_COLUMNS)
    for idx in train_indices:
        tile = tile_store.get_tile(int(idx))
        series = tile[:, fc : fc + args.input_length]
        finite = np.isfinite(series)
        vals = np.where(finite, series, 0.0).astype(np.float64)
        n = int(finite.sum())
        total += float(vals.sum())
        sq_total += float((vals ** 2).sum())
        count += n
        if args.model_version in ("v3_1", "v3_3"):
            res_sq, res_count = residual_sum_squares_np(series)
            residual_sq_total += res_sq
            residual_count += res_count
    if count == 0:
        return None
    mean = total / count
    std = max(math.sqrt(sq_total / count - mean * mean), 1e-6)
    normalizer = {"mean": float(mean), "std": float(std), "count": count}
    if args.model_version in ("v3_1", "v3_3") and residual_count > 0:
        residual_raw_std = max(math.sqrt(residual_sq_total / residual_count), 1e-6)
        normalizer["residual_std"] = float(residual_raw_std / std)
        normalizer["residual_raw_std"] = float(residual_raw_std)
        normalizer["residual_count"] = int(residual_count)
    print(
        f"fitted tile normalizer: mean={mean:.6f} std={std:.6f} "
        f"residual_std={normalizer.get('residual_std', 1.0):.6f} from {count:,} values",
        flush=True,
    )
    return normalizer


def residual_sum_squares_np(series: np.ndarray) -> tuple[float, int]:
    """Sum squared linear-detrended residuals for raw series values."""
    values = series.astype(np.float64, copy=False)
    finite = np.isfinite(values)
    counts = finite.sum(axis=1)
    valid = counts > 1
    if not valid.any():
        return 0.0, 0

    t = np.linspace(-1.0, 1.0, values.shape[1], dtype=np.float64)
    sub = values[valid]
    sub_finite = finite[valid]
    sub_counts = counts[valid].astype(np.float64)
    y_sum = np.where(sub_finite, sub, 0.0).sum(axis=1, keepdims=True)
    t_sum = np.where(sub_finite, t, 0.0).sum(axis=1, keepdims=True)
    y_mean = y_sum / sub_counts[:, None]
    t_mean = t_sum / sub_counts[:, None]
    centered_t = np.where(sub_finite, t - t_mean, 0.0)
    centered_y = np.where(sub_finite, sub - y_mean, 0.0)
    denom = np.square(centered_t).sum(axis=1, keepdims=True)
    slope = np.divide(
        (centered_y * centered_t).sum(axis=1, keepdims=True),
        denom,
        out=np.zeros_like(denom),
        where=denom > 1e-12,
    )
    residual = np.where(sub_finite, sub - (y_mean + slope * (t - t_mean)), 0.0)
    return float(np.square(residual).sum()), int(sub_counts.sum())


def get_lr(args, step):
    if args.lr_scheduler == "none":
        return args.lr
    if args.warmup_steps > 0 and step <= args.warmup_steps:
        return args.min_lr + (args.lr - args.min_lr) * (step / args.warmup_steps)
    total = args.scheduler_total_steps
    cosine_steps = max(1, total - args.warmup_steps)
    elapsed = min(max(0, step - args.warmup_steps), cosine_steps)
    progress = elapsed / cosine_steps
    return args.min_lr + 0.5 * (args.lr - args.min_lr) * (1.0 + math.cos(math.pi * progress))


def autocast_context(device, precision):
    if device.type != "cuda" or precision == "fp32":
        return nullcontext()
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def save_checkpoint(path, model, optimizer, scaler, args, step, epoch, best_val_loss):
    torch.save({
        "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(), "args": vars(args),
        "step": step, "epoch": epoch, "best_val_loss": best_val_loss,
    }, path)


def read_resume_metrics(metrics_path: Path, train_window_steps: int) -> tuple[float, list[float]]:
    """Read elapsed time and recent train losses from an existing metrics file."""
    if not metrics_path.exists():
        return 0.0, []

    elapsed_hours = 0.0
    losses: deque[float] = deque(maxlen=max(1, train_window_steps))
    with metrics_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                elapsed_hours = max(elapsed_hours, float(row.get("elapsed_hours") or 0.0))
                losses.append(float(row["loss"]))
            except (KeyError, TypeError, ValueError):
                continue
    return elapsed_hours, list(losses)


class MetricWindow:
    def __init__(self, maxlen):
        self.values = deque(maxlen=max(1, maxlen))
        self.total = 0.0

    def add(self, value):
        if len(self.values) == self.values.maxlen:
            self.total -= self.values[0]
        self.values.append(value)
        self.total += value

    @property
    def mean(self):
        return self.total / len(self.values) if self.values else float("nan")


if __name__ == "__main__":
    main()
