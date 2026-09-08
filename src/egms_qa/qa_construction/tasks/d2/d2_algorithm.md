# D2: Seasonal phase

## Task overview

D2 describes when annual displacement cycles peak, how closely the point peak times agree, and whether their amplitude changes. Phase means position within the annual cycle; coherence measures agreement of those phases across the tile.

| Task | Type | Output and relationship |
|---|---|---|
| D21 | Classification | Names the season of the shared annual peak when seasonal strength and D22 coherence pass their gates; otherwise reports no clear peak. |
| D22 | Numeric score | Measures agreement among point annual phases, from 0 to 1. Also supplies the coherence gate for D21. |
| D23 | Numeric value (days) | Measures the spread of annual peak timing using the same point phases as D21 and D22, with low-amplitude points filtered. |
| D24 | Numeric value (mm) | Median change in annual amplitude from the early half to the late half of the observation window; positive means stronger seasonality. |

[Task index](../README.md) · [Published table](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/artifacts/reference_tables/d2/d2_final_table.csv) · [Implementation](d2_compute.py) · [Run and files](../README.md#run-d2)

For each tile, read the stored model-ready displacement
`time_series[:, 0:294]`. The data config records that this is identical to
`[8,302)` on the original 304-step prepared axis. D2 uses the stored window but
adds the original index offset when constructing the physical time axis.

### Validity

- valid point: at least 50 valid epochs
- valid D24 point: at least 50 valid epochs in both early and late windows
- valid tile: at least 30 valid points

The seasonal-strength gate uses [B51](../b5/b5_algorithm.md), the 90th percentile of the source point seasonality field. The formulas use `S_i(t)` for point displacement, `R_i(t)` for its detrended residual, `t_year` for time in years, and a small positive `eps` to avoid division by zero.

## Algorithm steps

### D22 Formula

For each point, fit and remove a linear trend from its displacement time series:

```text
R_i(t) = S_i(t) - linear_fit_i(t)
```

Project the residual onto annual sine/cosine components. The complex vector
(phasor) `z_i` stores the cosine coefficient in its real part and the sine
coefficient in its imaginary part; `j` is the imaginary unit. Its angle gives
the annual phase, and its magnitude gives the strength of that annual component:

```text
a_i = mean(R_i(t) * cos(2pi t_year))
b_i = mean(R_i(t) * sin(2pi t_year))
z_i = a_i + j b_i
```

Then:

```text
D22_phase_coherence = |sum_i z_i| / (sum_i |z_i| + eps)
```

Interpretation:

- near 1: point seasonal phases are aligned
- near 0: point seasonal phases cancel or are scattered

### D21 Formula

D21 uses the annual vectors `z_i` defined for D22.

Compute the raw tile peak phase from the vector sum:

```text
raw_phase = mod(angle(sum_i z_i) / (2pi), 1)
raw_peak_day_of_year = raw_phase * 365.25
```

Map raw peak day to season:

| raw peak day | raw season |
|---|---|
| `[0, 59)` or `[334, 365.25)` | `winter_peak` |
| `[59, 151)` | `spring_peak` |
| `[151, 243)` | `summer_peak` |
| `[243, 334)` | `autumn_peak` |

The final D21 label uses the selected no-clear gate:

```text
if invalid raw phase:
    D21_dominant_seasonal_peak = no_clear_seasonal_peak
elif B51_seasonality_p90 < 1.0:
    D21_dominant_seasonal_peak = no_clear_seasonal_peak
elif D22_phase_coherence < 0.20:
    D21_dominant_seasonal_peak = no_clear_seasonal_peak
else:
    D21_dominant_seasonal_peak = raw season
```

These gates are corpus-relative choices for the released 10,000 tiles. D21 reports a seasonal peak only when both seasonal strength and phase agreement are sufficient.

### D23 Formula

D23 converts the same point-level annual phasors into circular phase dispersion
in calendar days.

For valid points:

```text
A_i = |z_i|
u_i = z_i / (|z_i| + eps)
```

Exclude the lowest-amplitude 10% of valid points within the tile. If fewer than
10 points remain, use all valid points. Then:

```text
Rbar = |sum_i A_i u_i| / (sum_i A_i + eps)
sigma = sqrt(-2 ln(clip(Rbar, eps, 1)))
D23_phase_dispersion_days = sigma / (2pi) * 365.25
```

Interpretation:

- smaller values: seasonal peaks are concentrated within fewer calendar days
- larger values: seasonal peaks are spread across a wider part of the year

### D24 Formula

D24 compares annual seasonal amplitude between the early and late halves of the
same time window.

Window split:

| window | input epochs |
|---|---|
| early | stored `time_series[:, 0:147]` |
| late | stored `time_series[:, 147:294]` |

For each point and each window, fit and remove a linear trend. Then fit annual
cosine/sine coefficients by least squares:

```text
R_i(t) = S_i(t) - linear_fit_i(t)
R_i(t) ~= beta_cos_i cos(2pi t_year) + beta_sin_i sin(2pi t_year)
annual_amplitude_i = sqrt(beta_cos_i^2 + beta_sin_i^2)
```

The tile scalar is the median point-level late-minus-early change:

```text
D24_seasonal_amplitude_change_mm =
    median_i(late_annual_amplitude_i - early_annual_amplitude_i)
```

Interpretation:

- positive values: seasonal amplitude became stronger
- negative values: seasonal amplitude became weaker
- values near zero: seasonal amplitude was approximately stable

## Results

The following summaries use all 10,000 rows of the
[published reference table](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/artifacts/reference_tables/d2/d2_final_table.csv). Numeric summaries use finite
values. Missing targets are reported separately. In numeric tables, p05 and p95
are the 5th and 95th percentiles.

### Numeric targets

| task | defined | missing | p05 | median | p95 |
|---|---:|---:|---:|---:|---:|
| D22 | 10,000 | 0 | 0.0746656 | 0.298174 | 0.784234 |
| D23 | 10,000 | 0 | 39.4858 | 89.8984 | 132.048 |
| D24 | 10,000 | 0 | -0.104846 | 0.0159081 | 0.113439 |

### D21 label distribution

| label | count | share |
|---|---:|---:|
| `no_clear_seasonal_peak` | 3,472 | 34.72% |
| `summer_peak` | 2,405 | 24.05% |
| `winter_peak` | 1,783 | 17.83% |
| `spring_peak` | 1,698 | 16.98% |
| `autumn_peak` | 642 | 6.42% |
