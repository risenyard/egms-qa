"""Encoder pretraining entrypoint: self-supervised masked reconstruction of the released
10k tile set (train split).

Data loading uses TileStore.from_manifest over per-tile npz files, with a
precomputed train/val/test split and normalization. The input length is locked
to the configured time window (default 294). See `TileEncoder` for the model.
"""

from __future__ import annotations

import argparse
import copy
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
from safetensors.torch import save_file

from egms_encoder.checkpoint import load_encoder_config, load_normalization
from egms_encoder.data.tile_batching import iter_tile_batches
from egms_encoder.data.tile_store import FEATURE_COLUMNS_COUNT, TileStore
from egms_encoder.models.tile_encoder import TileEncoder

DEFAULT_MODEL_CONFIG = "data/encoder/checkpoint/config.json"
DEFAULT_TRAINING_ARGS = "data/encoder/checkpoint/training_args.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="EGMS-QA Encoder masked-reconstruction pretraining."
    )
    # Released tile data: split manifest + data config + normalization
    p.add_argument(
        "--model-config",
        default=DEFAULT_MODEL_CONFIG,
        help="Public encoder architecture config.json.",
    )
    p.add_argument(
        "--training-args",
        default=DEFAULT_TRAINING_ARGS,
        help="Public training recipe training_args.json.",
    )
    p.add_argument("--manifest", default="data/encoder/manifest/split.parquet",
                   help="Split manifest parquet (tile_id, path, split, ...).")
    p.add_argument("--data-config", default="data/encoder/manifest/data_config.json",
                   help="Data config JSON (time window + tile-row layout).")
    p.add_argument("--normalization", default="data/encoder/checkpoint/normalization.json",
                   help="Precomputed normalization JSON (mean/std/residual_std).")
    p.add_argument("--output-dir", default="outputs/encoder_pretrain")
    # Tile parameters
    p.add_argument(
        "--max-tile-points",
        type=int,
        default=None,
        help="Maximum points sampled from a tile; dense attention is O(N^2).",
    )
    p.add_argument(
        "--tiles-per-batch",
        type=int,
        default=None,
        help="Number of tiles per GPU batch.",
    )
    # Model parameters
    p.add_argument("--d-model", type=int, default=None)
    p.add_argument("--num-layers", type=int, default=None)
    p.add_argument("--num-heads", type=int, default=None)
    p.add_argument("--dropout", type=float, default=None)
    p.add_argument("--input-length", type=int, default=None,
                   help="Number of time steps; defaults to config.json.")
    # Masking
    p.add_argument("--mask-ratio", type=float, default=None)
    p.add_argument("--mask-strategy", default=None, choices=["random", "block"])
    p.add_argument("--sync-mask", dest="sync_mask", action=argparse.BooleanOptionalAction, default=None,
                   help="Use the same temporal mask for all points in a tile so that "
                        "masked observations are also hidden from neighboring points. "
                        "Defaults to the published training recipe.")
    p.add_argument("--mask-schedule", default=None, choices=["fixed", "short_mix"],
                   help="training mask schedule; validation uses eval_mask_ratio")
    p.add_argument("--eval-mask-ratio", type=float, default=None,
                   help="fixed validation mask ratio")
    p.add_argument("--patch-size", type=int, default=None, help="Time patch size.")
    p.add_argument("--temporal-layers", type=int, default=None)
    p.add_argument("--temporal-heads", type=int, default=None)
    p.add_argument("--residual-loss-weight", type=float, default=None,
                   help="residual auxiliary loss weight")
    p.add_argument("--residual-consistency-weight", type=float, default=None,
                   help="weight for final reconstruction residual consistency")
    p.add_argument("--residual-head-mode", default=None, choices=["additive", "aux_only"],
                   help="add residual correction to reconstruction or train it only as an auxiliary head")
    p.add_argument("--coord-scale", type=float, default=None,
                   help="Positive coordinate divisor; defaults to config.json.")
    p.add_argument("--residual-head-lr", type=float, default=None,
                   help="optional learning rate for residual head parameters")
    p.add_argument("--init-from-checkpoint", default=None,
                   help="Warm-start compatible model weights without loading optimizer/scaler state")
    p.add_argument("--point-sampling", default=None, choices=["uniform", "residual_weighted"],
                   help="point sampling strategy for oversized training tiles")
    p.add_argument("--residual-sampling-alpha", type=float, default=None,
                   help="fraction of oversized tile points sampled by residual RMS")
    # Training
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--duration-hours", type=float, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--min-lr", type=float, default=None)
    p.add_argument("--lr-scheduler", default=None, choices=["none", "cosine"])
    p.add_argument("--scheduler-total-steps", type=int, default=None)
    p.add_argument("--warmup-steps", type=int, default=None)
    p.add_argument("--weight-decay", type=float, default=None)
    p.add_argument("--precision", default=None, choices=["fp32", "bf16", "fp16"])
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=None)
    # Validation (train/val/test split is precomputed in the manifest)
    p.add_argument("--val-batches", type=int, default=None)
    p.add_argument(
        "--resample-val-batches",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Draw a new, reproducible set of non-overlapping validation batches at each validation step.",
    )
    p.add_argument("--val-every-steps", type=int, default=None)
    p.add_argument("--val-seed", type=int, default=None)
    # Logging / checkpointing
    p.add_argument("--checkpoint-every-steps", type=int, default=None)
    p.add_argument("--log-every-steps", type=int, default=None)
    p.add_argument("--train-window-steps", type=int, default=None)
    p.add_argument("--resume-from", default=None)
    return p.parse_args(argv)


def apply_release_config(
    args: argparse.Namespace,
    model_config: dict,
    training_args: dict,
) -> argparse.Namespace:
    """Fill unspecified CLI options from the public model and training files."""
    if training_args.get("schema_version") != "egms-qa-encoder-training-1.0":
        raise ValueError("unsupported encoder training_args schema")
    data = training_args["data"]
    masking = training_args["masking"]
    point_sampling = training_args["point_sampling"]
    loss = training_args["loss"]
    optimization = training_args["optimization"]
    validation = training_args["validation"]
    checkpointing = training_args.get("checkpointing", {})
    if int(data["model_input_steps"]) != int(model_config["input_length"]):
        raise ValueError("training_args model_input_steps does not match config.json")
    strategy = str(masking["strategy"])
    defaults = {
        "max_tile_points": data["maximum_points_per_tile"],
        "tiles_per_batch": optimization["tiles_per_batch"],
        "d_model": model_config["d_model"],
        "num_layers": model_config["spatial_layers"],
        "num_heads": model_config["spatial_heads"],
        "dropout": model_config["dropout"],
        "input_length": model_config["input_length"],
        "mask_ratio": masking["train_ratio"],
        "mask_strategy": "block" if strategy == "synchronized_block" else strategy,
        "sync_mask": strategy == "synchronized_block",
        "mask_schedule": masking["schedule"],
        "eval_mask_ratio": masking["evaluation_ratio"],
        "patch_size": model_config["patch_size"],
        "temporal_layers": model_config["temporal_layers"],
        "temporal_heads": model_config["temporal_heads"],
        "residual_loss_weight": loss["residual_loss_weight"],
        "residual_consistency_weight": loss["residual_consistency_weight"],
        "residual_head_mode": model_config["residual_head_mode"],
        "coord_scale": model_config["coord_scale_m"],
        "residual_head_lr": optimization["residual_head_learning_rate"],
        "point_sampling": point_sampling["method"],
        "residual_sampling_alpha": point_sampling["residual_sampling_alpha"],
        "max_steps": optimization["maximum_steps"],
        "lr": optimization["learning_rate"],
        "min_lr": optimization["minimum_learning_rate"],
        "lr_scheduler": optimization["scheduler"],
        "scheduler_total_steps": optimization["scheduler_total_steps"],
        "warmup_steps": optimization["warmup_steps"],
        "weight_decay": optimization["weight_decay"],
        "precision": optimization["precision"],
        "seed": optimization["seed"],
        "val_batches": validation["batches"],
        "resample_val_batches": validation.get("resample_each_validation", False),
        "val_every_steps": validation["interval_steps"],
        "val_seed": validation["seed"],
        "checkpoint_every_steps": checkpointing.get("interval_steps", 5_000),
        "log_every_steps": checkpointing.get("log_interval_steps", 200),
        "train_window_steps": checkpointing.get("rolling_window_steps", 1_000),
    }
    for name, value in defaults.items():
        if getattr(args, name) is None:
            setattr(args, name, value)
    return args


def validate_training_args(args: argparse.Namespace) -> None:
    positive = (
        "max_tile_points", "tiles_per_batch", "d_model", "num_layers", "num_heads",
        "patch_size", "temporal_layers", "temporal_heads", "max_steps",
        "scheduler_total_steps", "val_batches", "checkpoint_every_steps",
        "log_every_steps", "train_window_steps",
    )
    for name in positive:
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    for name in ("mask_ratio", "eval_mask_ratio", "residual_sampling_alpha"):
        value = float(getattr(args, name))
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"--{name.replace('_', '-')} must be in [0,1]")
    if float(args.coord_scale) <= 0:
        raise ValueError("--coord-scale must be positive")
    if float(args.lr) <= 0 or float(args.min_lr) < 0:
        raise ValueError("learning rates must be non-negative and --lr must be positive")


def resolved_model_config(model_config: dict, args: argparse.Namespace) -> dict:
    """Return the inference config that exactly matches the effective model."""
    resolved = copy.deepcopy(model_config)
    resolved.update(
        {
            "input_length": int(args.input_length),
            "patch_size": int(args.patch_size),
            "d_model": int(args.d_model),
            "temporal_layers": int(args.temporal_layers),
            "temporal_heads": int(args.temporal_heads),
            "spatial_layers": int(args.num_layers),
            "spatial_heads": int(args.num_heads),
            "dropout": float(args.dropout),
            "coord_scale_m": float(args.coord_scale),
            "residual_head_mode": str(args.residual_head_mode),
        }
    )
    return resolved


def resolved_training_recipe(training_args: dict, args: argparse.Namespace) -> dict:
    """Return the public training schema with all CLI overrides applied."""
    resolved = copy.deepcopy(training_args)
    resolved.pop("checkpoint_selection", None)
    resolved["data"].update(
        {
            "model_input_steps": int(args.input_length),
            "maximum_points_per_tile": int(args.max_tile_points),
        }
    )
    strategy = (
        "synchronized_block"
        if args.mask_strategy == "block" and args.sync_mask
        else str(args.mask_strategy)
    )
    resolved["masking"].update(
        {
            "strategy": strategy,
            "train_ratio": float(args.mask_ratio),
            "evaluation_ratio": float(args.eval_mask_ratio),
            "schedule": str(args.mask_schedule),
        }
    )
    resolved["point_sampling"].update(
        {
            "method": str(args.point_sampling),
            "residual_sampling_alpha": float(args.residual_sampling_alpha),
        }
    )
    resolved["loss"].update(
        {
            "residual_loss_weight": float(args.residual_loss_weight),
            "residual_consistency_weight": float(args.residual_consistency_weight),
        }
    )
    resolved["optimization"].update(
        {
            "maximum_steps": int(args.max_steps),
            "tiles_per_batch": int(args.tiles_per_batch),
            "learning_rate": float(args.lr),
            "minimum_learning_rate": float(args.min_lr),
            "residual_head_learning_rate": float(args.residual_head_lr),
            "scheduler": str(args.lr_scheduler),
            "scheduler_total_steps": int(args.scheduler_total_steps),
            "warmup_steps": int(args.warmup_steps),
            "weight_decay": float(args.weight_decay),
            "precision": str(args.precision),
            "seed": int(args.seed),
        }
    )
    resolved["validation"].update(
        {
            "batches": int(args.val_batches),
            "resample_each_validation": bool(args.resample_val_batches),
            "interval_steps": int(args.val_every_steps),
            "seed": int(args.val_seed),
        }
    )
    resolved.setdefault("checkpointing", {}).update(
        {
            "interval_steps": int(args.checkpoint_every_steps),
            "log_interval_steps": int(args.log_every_steps),
            "rolling_window_steps": int(args.train_window_steps),
        }
    )
    return resolved


def write_inference_bundle_metadata(
    output_dir: Path,
    model_config: dict,
    training_args: dict,
    normalizer: dict,
    args: argparse.Namespace,
) -> None:
    """Write configs that can be reused directly by token extraction."""
    payloads = {
        "config.json": resolved_model_config(model_config, args),
        "training_args.json": resolved_training_recipe(training_args, args),
        "normalization.json": normalizer,
        "run_args.json": vars(args),
    }
    for filename, payload in payloads.items():
        with (output_dir / filename).open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")


def save_inference_state(path: Path, state: dict[str, torch.Tensor]) -> None:
    """Save a pure model state in the public Safetensors format."""
    tensors = {
        key: value.detach().cpu().contiguous()
        for key, value in state.items()
    }
    save_file(tensors, str(path), metadata={"model_type": "egms_encoder"})


def save_inference_weights(path: Path, model: TileEncoder) -> None:
    """Save the current model for strict inference loading."""
    save_inference_state(path, model.state_dict())


def collect_validation_batches(args, tile_store, rng, *, resampled: bool) -> list[dict]:
    batches = iter_tile_batches(
        tile_store, args.tiles_per_batch,
        split="val",
        rng=rng,
        max_batches=None if resampled else args.val_batches,
        max_points=args.max_tile_points,
        feature_columns_count=FEATURE_COLUMNS_COUNT, input_length=args.input_length,
        point_sampling="uniform", residual_sampling_alpha=args.residual_sampling_alpha,
    )
    if resampled:
        batches = islice(batches, args.val_batches)
    return list(batches)


def validation_tile_ids(val_batches) -> list[int]:
    return [int(idx) for batch in val_batches for idx in batch["tile_indices"]]


def main() -> None:
    args = parse_args()
    model_config_path = Path(args.model_config)
    if not model_config_path.is_file():
        raise FileNotFoundError(
            f"Encoder config not found: {model_config_path}. "
            "Download risenyard/egms-qa-encoder into data/encoder/checkpoint first."
        )
    training_args_path = Path(args.training_args)
    if not training_args_path.is_file():
        raise FileNotFoundError(f"Training recipe not found: {training_args_path}")
    model_config = load_encoder_config(model_config_path)
    training_args = json.loads(training_args_path.read_text(encoding="utf-8"))
    args = apply_release_config(args, model_config, training_args)
    validate_training_args(args)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data via TileStore
    tile_store = TileStore.from_manifest(
        args.manifest, args.data_config, require_static_fields=False,
    )
    configured_input_length = tile_store.time_window.input_length
    if args.input_length != configured_input_length:
        raise ValueError(
            f"Model input_length={args.input_length} does not match the data contract "
            f"({configured_input_length})"
        )
    train_indices = tile_store.split_tile_indices("train")
    val_indices = tile_store.split_tile_indices("val")
    test_indices = tile_store.split_tile_indices("test")
    print(
        f"tile split (precomputed from {args.manifest}): "
        f"train={len(train_indices)} val={len(val_indices)} test={len(test_indices)}",
        flush=True,
    )

    # Load precomputed normalization (skip the fit step)
    normalizer = load_normalization(args.normalization)
    normalizer.pop("_meta", None)  # strip annotation block before passing into trainer
    write_inference_bundle_metadata(
        output_dir, model_config, training_args, normalizer, args
    )
    print(
        f"normalizer: mean={normalizer['mean']:.6f} std={normalizer['std']:.6f} "
        f"residual_std={normalizer.get('residual_std',1.0):.6f}",
        flush=True,
    )

    # Build model
    model = build_model(args, normalizer).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model: {param_count:,} parameters ({param_count/1e6:.1f}M)", flush=True)

    if args.init_from_checkpoint:
        load_init_checkpoint(model, Path(args.init_from_checkpoint), device)

    optimizer = build_optimizer(args, model)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and args.precision == "fp16")

    step = 0
    epoch = 0
    best_val_loss = float("inf")
    best_weights_path = output_dir / "best.safetensors"
    best_weights_ready = bool(args.resume_from and best_weights_path.is_file())
    resume_elapsed_hours = 0.0
    resume_train_losses: list[float] = []
    if args.resume_from:
        checkpoint = torch.load(args.resume_from, map_location=device, weights_only=True)
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

    # Fixed validation batches preserve the release protocol. Resampled validation
    # draws a reproducible subset from a fresh permutation at every validation step.
    seen_val_tile_ids: set[int] = set()
    if args.resample_val_batches:
        val_batches = []
        val_tile_count = len(tile_store.split_tile_indices("val"))
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
                split="train",
                rng=data_rng,
                max_points=args.max_tile_points,
                feature_columns_count=FEATURE_COLUMNS_COUNT, input_length=args.input_length,
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
                        save_inference_weights(best_weights_path, model)
                        best_weights_ready = True

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
    best_checkpoint_path = output_dir / "best.pt"
    if not best_checkpoint_path.is_file():
        save_checkpoint(
            best_checkpoint_path,
            model,
            optimizer,
            scaler,
            args,
            step,
            epoch,
            best_val_loss,
        )
    if not best_weights_ready:
        best_checkpoint = torch.load(
            best_checkpoint_path, map_location="cpu", weights_only=True
        )
        save_inference_state(best_weights_path, best_checkpoint["model"])
    print(f"Training complete: {step} steps, best_val_loss={best_val_loss:.6f}")
    print(f"wrote {metrics_path}")
    print(f"wrote {output_dir / 'latest.pt'}")
    print(f"wrote {best_weights_path}")


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
    """Use an optional dedicated LR for the residual head."""
    if args.residual_head_lr is None:
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
    if args.residual_head_lr is not None and len(optimizer.param_groups) > 1:
        optimizer.param_groups[0]["lr"] = current_lr
        optimizer.param_groups[1]["lr"] = args.residual_head_lr
        return
    for group in optimizer.param_groups:
        group["lr"] = current_lr


def load_init_checkpoint(model, checkpoint_path: Path, device) -> None:
    """Warm-start weights without loading optimizer/scaler state."""
    if checkpoint_path.suffix == ".safetensors":
        from safetensors.torch import load_file

        state = load_file(str(checkpoint_path), device=str(device))
    else:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        state = checkpoint["model"]
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
    """Compute masked global, residual, and residual-head losses."""
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

    loss = (
        global_loss
        + float(args.residual_loss_weight) * residual_head_loss
        + float(args.residual_consistency_weight) * residual_loss
    )
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
    rng = np.random.default_rng(args.val_seed)
    for tile_batch in val_batches:
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
