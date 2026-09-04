# D2 Seasonal Phase Algorithm

## Current Scope

This folder is the D2 seasonal-phase family. The current computed target is:

- `D21_dominant_seasonal_peak`
- `D22_phase_coherence`
- `D23_phase_dispersion_days`
- `D24_seasonal_amplitude_change_mm`

D21 is a derived class. It asks whether the tile has a coherent dominant
seasonal peak, and if so which season it peaks in.

D22 is delivered as a continuous scalar. No standalone D22 low/high class is
part of the formal target, but D22 is used as the coherence gate for D21.

D23 is also delivered as a continuous scalar only. D21 should reuse the same
per-point annual phasors.

D24 is delivered as a continuous scalar only. It asks whether the annual
seasonal amplitude became stronger or weaker between the first and second half
of the observation window.

## Input

For each tile, read the stored model-ready displacement
`time_series[:, 0:294]`. The data config records that this is identical to
`[8,302)` on the original 304-step prepared axis. D2 uses the stored window but
adds the original index offset when constructing the physical time axis.

## D21 Formula

D21 uses the same detrended annual point phasors as D22/D23:

```text
z_i = a_i + j b_i
```

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

The threshold is corpus-relative for the current EGMS encoder all-10k delivery. It
formalizes D21 as a tile-wide coherent seasonal peak, not merely an average
seasonal phase.

## D22 Formula

For each point, fit and remove a linear trend from its displacement time series:

```text
R_i(t) = S_i(t) - linear_fit_i(t)
```

Project the residual onto annual sine/cosine components:

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

## D23 Formula

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

## D24 Formula

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

D24 is scalar-only. It has no `shrinking / stable / growing` class in the formal
target because the all-10k distribution is strongly concentrated near zero.

## Validity

- valid point: at least 50 valid epochs
- valid D24 point: at least 50 valid epochs in both early and late windows
- valid tile: at least 30 valid points

## Files

- `d2_final_table.csv`: formal D21/D22/D23/D24 target table.
- `d2_final_diagnostics.csv`: support diagnostics for seasonal-phase threshold analysis.
- `d2_compute.py`: recomputes D21, D22, D23, and D24.

## Current All-10k Result

`D21_dominant_seasonal_peak`:

| class | count |
|---|---:|
| `no_clear_seasonal_peak` | 3472 |
| `summer_peak` | 2405 |
| `winter_peak` | 1783 |
| `spring_peak` | 1698 |
| `autumn_peak` | 642 |

D21 gate reasons:

| reason | count |
|---|---:|
| `clear` | 6528 |
| `low_phase_coherence` | 2683 |
| `weak_seasonality` | 564 |
| `weak_seasonality_and_low_coherence` | 225 |

Raw peak-season counts before D21 gating:

| class | count |
|---|---:|
| `summer_peak` | 3324 |
| `spring_peak` | 2860 |
| `winter_peak` | 2666 |
| `autumn_peak` | 1150 |

`D22_phase_coherence`:

| statistic | value |
|---|---:|
| mean | 0.345591 |
| p10 | 0.107155 |
| p25 | 0.183483 |
| p50 | 0.298174 |
| p75 | 0.468239 |
| p90 | 0.666640 |
| p95 | 0.784234 |

`D23_phase_dispersion_days`:

| statistic | value |
|---|---:|
| mean | 88.466987 |
| p10 | 51.545815 |
| p25 | 71.035680 |
| p50 | 89.898365 |
| p75 | 106.584310 |
| p90 | 122.568380 |
| p95 | 132.047655 |

`D24_seasonal_amplitude_change_mm`:

| statistic | value |
|---|---:|
| mean | 0.007545 |
| p01 | -0.233549 |
| p05 | -0.104846 |
| p10 | -0.064743 |
| p25 | -0.020479 |
| p50 | 0.015908 |
| p75 | 0.048531 |
| p90 | 0.083171 |
| p95 | 0.113439 |
| p99 | 0.207153 |

Candidate low-coherence rates, used to select the D21 coherence gate:

| threshold | fraction below |
|---|---:|
| 0.10 | 8.97% |
| 0.15 | 18.13% |
| 0.20 | 29.08% |
| 0.25 | 39.92% |
| 0.30 | 50.39% |

D22 and D23 remain continuous scalars. The D22=0.20 threshold is used only as
D21's coherence gate.

D24 remains a continuous scalar. No D24 class threshold is used.
