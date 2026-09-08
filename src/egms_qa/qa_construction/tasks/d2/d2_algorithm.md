# D2: Seasonal phase

## Task overview

| task | description |
|---|---|
| **D2 group** | Describe annual phase, phase agreement, and changes in seasonal amplitude. |
| D21 | Dominant seasonal peak category when amplitude and phase-support conditions permit interpretation. |
| D22 | Concentration of the annual phase estimates, describing seasonal coherence. |
| D23 | Circular phase dispersion expressed in days. |
| D24 | Change in annual amplitude between the specified early and late windows. |

[Task index](../README.md) · [Published table](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/artifacts/reference_tables/d2/d2_final_table.csv) · [Implementation](d2_compute.py)

## Key concepts

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

### Input

For each tile, read the stored model-ready displacement
`time_series[:, 0:294]`. The data config records that this is identical to
`[8,302)` on the original 304-step prepared axis. D2 uses the stored window but
adds the original index offset when constructing the physical time axis.

### Validity

- valid point: at least 50 valid epochs
- valid D24 point: at least 50 valid epochs in both early and late windows
- valid tile: at least 30 valid points

## Algorithm steps

### D21 Formula

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

### D22 Formula

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

D24 is scalar-only. It has no `shrinking / stable / growing` class in the formal
target because the all-10k distribution is strongly concentrated near zero.

## Run and files

Complete the [task setup](../README.md#setup) first. Run these commands from
the repository root:

```bash
python -m egms_qa.qa_construction.tasks.d2.d2_compute \
    --out-dir outputs/tasks-rebuilt/d2
```

The new table is written to `outputs/tasks-rebuilt/d2/d2_final_table.csv`.
The installed reference remains at `outputs/tasks/d2/d2_final_table.csv`.
See the [path conventions](../README.md#paths) for the relationship to Hugging Face.

| required input | published source | installed path |
|---|---|---|
| NPZ source tiles | [HF file](https://huggingface.co/datasets/risenyard/egms-qa-dataset/tree/main/artifacts/source_tiles) | `data/tiles/` |
| Split manifest | [HF file](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/metadata/split_manifest.parquet) | `data/encoder/manifest/split.parquet` |
| Data configuration | [HF file](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/metadata/data_config.json) | `data/encoder/manifest/data_config.json` |
| B5 reference table | [HF file](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/artifacts/reference_tables/b5/b5_final_table.csv) | `outputs/tasks/b5/b5_final_table.csv` |

The computation may also write local summaries or diagnostics next to its
new table. Their filenames and options are defined in the linked script;
they are not part of the published reference-table inventory unless linked
explicitly above.

## Results

The following summaries use all 10,000 rows of the
[published reference table](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/artifacts/reference_tables/d2/d2_final_table.csv). Numeric summaries use finite
values. Missing targets are reported separately.

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
