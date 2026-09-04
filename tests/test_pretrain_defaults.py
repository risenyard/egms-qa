from __future__ import annotations

from egms_encoder.pretrain import apply_encoder_43_config, parse_args, validate_training_args


EXPECTED_ENCODER_43_DEFAULTS = {
    "max_tile_points": 4096,
    "tiles_per_batch": 8,
    "d_model": 256,
    "num_layers": 6,
    "num_heads": 8,
    "dropout": 0.1,
    "mask_ratio": 0.3,
    "eval_mask_ratio": 0.3,
    "patch_size": 8,
    "temporal_layers": 2,
    "temporal_heads": 4,
    "residual_loss_weight": 1.0,
    "residual_consistency_weight": 0.1,
    "residual_head_mode": "additive",
    "coord_scale": 3500.0,
    "residual_head_lr": 1e-4,
    "point_sampling": "residual_weighted",
    "residual_sampling_alpha": 0.75,
    "max_steps": 150_000,
    "lr": 4e-4,
    "min_lr": 3e-5,
    "lr_scheduler": "cosine",
    "scheduler_total_steps": 150_000,
    "warmup_steps": 1_000,
    "weight_decay": 1e-2,
    "precision": "bf16",
    "seed": 42,
    "val_batches": 16,
    "val_every_steps": 1_000,
    "val_seed": 1729,
    "checkpoint_every_steps": 5_000,
    "log_every_steps": 200,
    "train_window_steps": 1_000,
}


def test_pretrain_defaults_match_public_encoder_recipe() -> None:
    config = {
        "schema_version": "egms-qa-encoder-training-config-1.0",
        "encoder_version": "4.3",
        "architecture": {
            "d_model": 256, "spatial_layers": 6, "spatial_heads": 8,
            "dropout": 0.1, "patch_size": 8, "temporal_layers": 2,
            "temporal_heads": 4, "residual_head_mode": "additive",
            "coord_scale_m": 3500.0,
        },
        "data": {"maximum_points_per_tile": 4096},
        "masking": {
            "strategy": "synchronized_block", "train_ratio": 0.3,
            "evaluation_ratio": 0.3, "schedule": "fixed",
        },
        "point_sampling": {"method": "residual_weighted", "residual_sampling_alpha": 0.75},
        "loss": {"residual_loss_weight": 1.0, "residual_consistency_weight": 0.1},
        "optimization": {
            "tiles_per_batch": 8, "maximum_steps": 150_000,
            "learning_rate": 4e-4, "minimum_learning_rate": 3e-5,
            "residual_head_learning_rate": 1e-4, "scheduler": "cosine",
            "scheduler_total_steps": 150_000, "warmup_steps": 1_000,
            "weight_decay": 1e-2, "precision": "bf16", "seed": 42,
        },
        "validation": {"batches": 16, "interval_steps": 1_000, "seed": 1729},
        "checkpointing": {
            "interval_steps": 5_000, "log_interval_steps": 200,
            "rolling_window_steps": 1_000,
        },
    }
    args = parse_args([])
    args = apply_encoder_43_config(args, config)
    assert {
        key: getattr(args, key) for key in EXPECTED_ENCODER_43_DEFAULTS
    } == EXPECTED_ENCODER_43_DEFAULTS
    assert args.manifest == "data/encoder/manifest/split.parquet"
    assert args.data_config == "data/encoder/manifest/data_config.json"
    assert args.normalization == "data/encoder/checkpoint/normalization.json"
    validate_training_args(args)
