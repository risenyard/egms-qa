from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch


def pad_tile_batch(
    samples: Sequence[Mapping[str, torch.Tensor]],
    max_points: int | None = None,
    overflow_policy: str = "error",
    pad_value: float = 0.0,
) -> dict[str, torch.Tensor]:
    """Pad variable-point EGMS tiles into one batch.

    Expected sample keys:
    - series: [N, T]
    - coords: [N, 2]
    - static: [N, F] (optional, but if present it must be present for all samples)

    By default this function keeps all points. It pads to the largest tile in
    the batch, or to max_points when max_points is provided. If a tile exceeds
    max_points, the default behavior is to raise instead of silently dropping
    points.
    """

    if not samples:
        raise ValueError("samples must not be empty")
    if overflow_policy not in {"error", "truncate"}:
        raise ValueError("overflow_policy must be 'error' or 'truncate'")

    point_counts = [_point_count(sample) for sample in samples]
    target_points = max_points or max(point_counts)
    if target_points < 1:
        raise ValueError("target point count must be positive")

    has_static = "static" in samples[0]
    if any(("static" in sample) != has_static for sample in samples):
        raise ValueError("Either all samples or no samples must include static features")

    prepared = [_prepare_sample(sample, target_points, overflow_policy) for sample in samples]
    series = _pad_stack([sample["series"] for sample in prepared], target_points, pad_value)
    coords = _pad_stack([sample["coords"] for sample in prepared], target_points, pad_value)
    num_points = torch.tensor([_point_count(sample) for sample in prepared], dtype=torch.long)
    point_mask = torch.arange(target_points).unsqueeze(0) < num_points.unsqueeze(1)

    batch: dict[str, torch.Tensor] = {
        "series": series,
        "coords": coords,
        "point_mask": point_mask,
        "num_points": num_points,
    }
    if has_static:
        batch["static"] = _pad_stack([sample["static"] for sample in prepared], target_points, pad_value)
    return batch


def _point_count(sample: Mapping[str, torch.Tensor]) -> int:
    if "series" not in sample or "coords" not in sample:
        raise ValueError("Each sample must include series and coords")
    series = sample["series"]
    coords = sample["coords"]
    if series.ndim != 2:
        raise ValueError(f"Expected series shape [N, T], got {tuple(series.shape)}")
    if coords.ndim != 2 or coords.shape[-1] != 2:
        raise ValueError(f"Expected coords shape [N, 2], got {tuple(coords.shape)}")
    if coords.shape[0] != series.shape[0]:
        raise ValueError("series and coords must have the same point count")
    if "static" in sample:
        static = sample["static"]
        if static.ndim != 2:
            raise ValueError(f"Expected static shape [N, F], got {tuple(static.shape)}")
        if static.shape[0] != series.shape[0]:
            raise ValueError("series and static must have the same point count")
    return int(series.shape[0])


def _prepare_sample(
    sample: Mapping[str, torch.Tensor],
    target_points: int,
    overflow_policy: str,
) -> Mapping[str, torch.Tensor]:
    count = _point_count(sample)
    if count <= target_points:
        return sample
    if overflow_policy == "error":
        raise ValueError(
            f"Tile has {count} points, which exceeds max_points={target_points}. "
            "Increase max_points or split the tile before batching."
        )
    return {key: value[:target_points] for key, value in sample.items()}


def _pad_stack(tensors: Sequence[torch.Tensor], target_points: int, pad_value: float) -> torch.Tensor:
    trailing_shape = tensors[0].shape[1:]
    dtype = tensors[0].dtype
    device = tensors[0].device
    output = torch.full(
        (len(tensors), target_points, *trailing_shape),
        pad_value,
        dtype=dtype,
        device=device,
    )
    for index, tensor in enumerate(tensors):
        if tensor.shape[1:] != trailing_shape:
            raise ValueError("All tensors must share trailing dimensions")
        if tensor.dtype != dtype:
            raise ValueError("All tensors must share dtype")
        if tensor.device != device:
            raise ValueError("All tensors must share device")
        output[index, : tensor.shape[0]] = tensor
    return output
