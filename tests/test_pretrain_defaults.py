from __future__ import annotations

from egms_encoder.pretrain import (
    apply_release_config,
    parse_args,
    resolved_model_config,
    resolved_training_recipe,
    validate_training_args,
)


EXPECTED_RELEASE_DEFAULTS = {
    "max_tile_points": 4096,
    "tiles_per_batch": 8,
    "d_model": 256,
    "num_layers": 6,
    "num_heads": 8,
    "dropout": 0.1,
    "input_length": 294,
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
    model_config = {
        "schema_version": "egms-qa-encoder-config-1.0",
        "model_type": "egms_encoder",
        "input_length": 294,
        "d_model": 256, "spatial_layers": 6, "spatial_heads": 8,
        "dropout": 0.1, "patch_size": 8, "temporal_layers": 2,
        "temporal_heads": 4, "residual_head_mode": "additive",
        "coord_scale_m": 3500.0,
    }
    training_args = {
        "schema_version": "egms-qa-encoder-training-1.0",
        "data": {"model_input_steps": 294, "maximum_points_per_tile": 4096},
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
    args = apply_release_config(args, model_config, training_args)
    assert {
        key: getattr(args, key) for key in EXPECTED_RELEASE_DEFAULTS
    } == EXPECTED_RELEASE_DEFAULTS
    assert args.model_config == "data/encoder/checkpoint/config.json"
    assert args.training_args == "data/encoder/checkpoint/training_args.json"
    assert args.manifest == "data/encoder/manifest/split.parquet"
    assert args.data_config == "data/encoder/manifest/data_config.json"
    assert args.normalization == "data/encoder/checkpoint/normalization.json"
    validate_training_args(args)

    output_config = resolved_model_config(model_config, args)
    assert output_config["input_length"] == args.input_length
    assert output_config["d_model"] == args.d_model
    assert output_config["spatial_layers"] == args.num_layers
    assert output_config["coord_scale_m"] == args.coord_scale

    output_recipe = resolved_training_recipe(training_args, args)
    assert output_recipe["schema_version"] == "egms-qa-encoder-training-1.0"
    assert output_recipe["optimization"]["maximum_steps"] == args.max_steps
    assert output_recipe["data"]["maximum_points_per_tile"] == args.max_tile_points
    assert output_recipe["masking"]["strategy"] == "synchronized_block"
