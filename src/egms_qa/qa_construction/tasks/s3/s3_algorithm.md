# S3 Algorithm: Representation-Monitoring Consistency

## Delivered Tasks

`S31_representation_monitoring_rarity_gap_p`

S31 asks whether the encoder representation considers a tile more or less unusual
than the A/B/C/D monitoring scalar system does.

```text
S31 = encoder representation rarity p - A/B/C/D monitoring rarity p
```

The unit is percentile points on a train-defined `p0-p99` scale.

`S32_representation_monitoring_rarity_relation`

S32 is the five-class, question-friendly relation derived from S31. Because the
S31 gap is approximately symmetric and bell-shaped, S32 uses train z-score
thresholds instead of percentile quotas.

`S33_monitoring_distinctive_dimension`

S33 explains which A/B/C/D monitoring axis contributes the most distinctive
scalar-side signal: `quality`, `motion`, `spatial`, or `temporal`.

## Inputs

Encoder-side input:

- `S21_local_isolation_score`

A/B/C/D monitoring sentinel scalars:

- quality axis: `A41_median_rmse_mm`
- motion axis: `B33_vel_abs_p90_mm_yr`, `B41_acc_abs_p90`, `B51_seasonality_p90`
- spatial axis: `C11_noise_aware_moving_fraction`, `C21_spatial_concentration_score`,
  `C31_deformation_front_strength_mm_yr`, `C41_fast_tail_bin_fraction`
- temporal axis: `D12_trend_order_mean`, `D13_top_changepoint_probability`,
  `D22_phase_coherence`, `D31_motion_intensification_mm_yr2`

These sentinels are oriented so larger values mean stronger monitoring signal,
stronger structure, or worse observation quality.

## Formula

1. Use train tiles only as the reference population.
2. Convert `S21_local_isolation_score` to `embedding_rarity_p`.
3. Convert every sentinel scalar to its train empirical percentile `p0-p99`.
4. For each axis, take the maximum sentinel percentile inside that axis:
   `quality`, `motion`, `spatial`, and `temporal`.
5. Average the four axis scores to form a raw monitoring-system score.
6. Convert that raw monitoring-system score again to train empirical percentile
   `monitoring_rarity_p`.
7. Output:

```text
S31_representation_monitoring_rarity_gap_p
    = embedding_rarity_p - monitoring_rarity_p
```

Positive values mean the encoder representation is rarer than expected from the
A/B/C/D monitoring scalar system. Negative values mean the scalar monitoring
system is rarer than the encoder representation.

## S32 Class Rule

Use only train tiles to estimate the S31 gap mean and standard deviation:

```text
train mean = 0.0000000
train std  = 35.2308617

-1.96 sigma = -69.0524888
-1.00 sigma = -35.2308617
+1.00 sigma =  35.2308617
+1.96 sigma =  69.0524888
```

Then assign:

```text
S31 <= -1.96 sigma                 -> strong_monitoring_excess
-1.96 sigma < S31 <= -1 sigma      -> moderate_monitoring_excess
-1 sigma < S31 < +1 sigma          -> aligned
+1 sigma <= S31 < +1.96 sigma      -> moderate_encoder_excess
S31 >= +1.96 sigma                 -> strong_encoder_excess
```

The thresholds are corpus-relative train z-score thresholds, not physical
thresholds. The strong classes correspond to an approximate two-sided 95%
normal-style deviation.

## S33 Dimension Rule

S33 reuses the same monitoring axes used by S31:

```text
quality_axis  = p(A41)
motion_axis   = max(p(B33), p(B41), p(B51))
spatial_axis  = max(p(C11), p(C21), p(C31), p(C41))
temporal_axis = max(p(D12), p(D13), p(D22), p(D31))

S33_monitoring_distinctive_dimension
    = argmax(quality_axis, motion_axis, spatial_axis, temporal_axis)
```

S33 is a monitoring-side explanation task. It explains which scalar monitoring
dimension is most distinctive, not why the encoder embedding itself is rare.
Exact ties are resolved deterministically by axis order:
`quality -> motion -> spatial -> temporal`.

## Current All10k Result

`S31_representation_monitoring_rarity_gap_p`:

```text
mean = 0.103
std  = 35.081
p05  = -58.324
p25  = -23.407
p50  = -0.130
p75  =  23.825
p95  =  59.834
```

The distribution is approximately symmetric and bell-shaped, but it is not used
as a normal-distribution assumption. S31 is delivered as a continuous construct
scalar. S32 is the derived five-class relation.

`S32_representation_monitoring_rarity_relation` all10k counts:

```text
strong_monitoring_excess       255
moderate_monitoring_excess    1348
aligned                       6706
moderate_encoder_excess       1435
strong_encoder_excess          256
```

`S33_monitoring_distinctive_dimension` all10k counts:

```text
quality       956
motion       1545
spatial      3262
temporal     4237
```

## File Inventory

- `s3_final_table.csv`: final task table with `tile_id`, `split`, S31, S32, and S33.
- `s3_compute.py`: reproducible computation script.
- `s3_summary.json`: distribution and reproducibility summary.
- `s3_distribution.png`: final distribution/scatter diagnostic.
