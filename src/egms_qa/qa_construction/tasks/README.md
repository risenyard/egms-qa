# EGMS-QA Task System

The task system computes 78 targets in 27 groups across six families. Each
linked method defines its targets, algorithm, and reference results. The
[QA construction guide](../README.md#workflows) explains how task tables become
labels and question–answer records.

## Task system

| Family | Subject |
|---|---|
| A | Observation and representation quality |
| B | Motion magnitude, direction, and monitoring signals |
| C | Spatial extent, concentration, and deformation fronts |
| D | Temporal trend shape, seasonality, and intensification |
| S | Representation similarity, rarity, and local structure |
| X | Unsupported questions and refusal boundaries |

A group such as A1 contains related targets A11 and A12. A/B/C/D/S produce
per-tile results; X1–X3 produce task-level refusal catalogs.

## Task groups

| Group and method | Leaf tasks | Subject | Implementation | HF reference table | Main inputs / steps |
|---|---|---|---|---|---|
| [A1](a1/a1_algorithm.md) | A11–A12 | Representation drift and stability | [script](a1/a1_compute.py) | [table](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/artifacts/reference_tables/a1/a1_final_table.csv) | NPZ tiles, encoder and token cache; combine shards |
| [A2](a2/a2_algorithm.md) | A21–A22 | Masked reconstruction and reliability | [script](a2/a2_compute.py) | [table](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/artifacts/reference_tables/a2/a2_final_table.csv) | NPZ tiles and encoder; combine shards |
| [A3](a3/a3_algorithm.md) | A31–A32 | Spatial observation coverage | [script](a3/a3_compute.py) | [table](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/artifacts/reference_tables/a3/a3_final_table.csv) | Cached spatial-cell point counts |
| [A4](a4/a4_algorithm.md) | A41–A42 | Measurement noise | [script](a4/a4_compute.py) | [table](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/artifacts/reference_tables/a4/a4_final_table.csv) | NPZ tiles |
| [A5](a5/a5_algorithm.md) | A51–A52 | Monitoring usability | [script](a5/a5_compute.py) | [table](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/artifacts/reference_tables/a5/a5_final_table.csv) | A1, A2, A3, A4 tables |
| [B1](b1/b1_algorithm.md) | B11–B12 | Subsidence signal-to-noise ratio | [script](b1/b1_compute.py) | [table](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/artifacts/reference_tables/b1/b1_final_table.csv) | NPZ tiles |
| [B2](b2/b2_algorithm.md) | B21–B22 | Mean velocity | [script](b2/b2_compute.py) | [table](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/artifacts/reference_tables/b2/b2_final_table.csv) | NPZ tiles |
| [B3](b3/b3_algorithm.md) | B31–B36 | Velocity tails and direction | [script](b3/b3_compute.py) | [table](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/artifacts/reference_tables/b3/b3_final_table.csv) | NPZ tiles |
| [B4](b4/b4_algorithm.md) | B41–B42 | Acceleration strength | [script](b4/b4_compute.py) | [table](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/artifacts/reference_tables/b4/b4_final_table.csv) | NPZ tiles |
| [B5](b5/b5_algorithm.md) | B51 | Seasonality strength | [script](b5/b5_compute.py) | [table](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/artifacts/reference_tables/b5/b5_final_table.csv) | NPZ tiles |
| [B6](b6/b6_algorithm.md) | B61 | Monitoring trigger | [script](b6/b6_compute.py) | [table](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/artifacts/reference_tables/b6/b6_final_table.csv) | B3, B4 tables |
| [C1](c1/c1_algorithm.md) | C11–C13 | Moving fraction and location | [script](c1/c1_compute.py) | [table](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/artifacts/reference_tables/c1/c1_final_table.csv) | NPZ tiles |
| [C2](c2/c2_algorithm.md) | C21–C22 | Spatial concentration | [script](c2/c2_compute.py) | [table](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/artifacts/reference_tables/c2/c2_final_table.csv) | NPZ tiles |
| [C3](c3/c3_algorithm.md) | C31–C33 | Deformation fronts | [script](c3/c3_compute.py) | [table](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/artifacts/reference_tables/c3/c3_final_table.csv) | NPZ tiles |
| [C4](c4/c4_algorithm.md) | C41–C42 | Fast-tail spatial extent | [script](c4/c4_compute.py) | [table](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/artifacts/reference_tables/c4/c4_final_table.csv) | NPZ tiles |
| [C5](c5/c5_algorithm.md) | C51–C52 | Monitoring priority | [script](c5/c5_compute.py) | [table](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/artifacts/reference_tables/c5/c5_final_table.csv) | B2, B3, B6, C3 tables |
| [D1](d1/d1_algorithm.md) | D11–D14 | Trend geometry and changepoints | [script](d1/d1_compute.py) | [table](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/artifacts/reference_tables/d1/d1_final_table.csv) | NPZ tiles, split manifest and data config |
| [D2](d2/d2_algorithm.md) | D21–D24 | Seasonal phase and amplitude | [script](d2/d2_compute.py) | [table](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/artifacts/reference_tables/d2/d2_final_table.csv) | NPZ tiles, data config, B5 table |
| [D3](d3/d3_algorithm.md) | D31–D35 | Motion intensification | [script](d3/d3_compute.py) | [table](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/artifacts/reference_tables/d3/d3_final_table.csv) | NPZ tiles |
| [D4](d4/d4_algorithm.md) | D41–D42 | Dominant temporal process | [script](d4/d4_compute.py) | [table](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/artifacts/reference_tables/d4/d4_final_table.csv) | B3, B4, B5, D1, D2, D3 tables |
| [S1](s1/s1_algorithm.md) | S11–S15 | Representation anchors | [script](s1/s1_compute.py) | [table](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/artifacts/reference_tables/s1/s1_final_table.csv) | Token cache |
| [S2](s2/s2_algorithm.md) | S21–S22 | Representation isolation | [script](s2/s2_compute.py) | [table](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/artifacts/reference_tables/s2/s2_final_table.csv) | Token cache |
| [S3](s3/s3_algorithm.md) | S31–S33 | Representation–monitoring relation | [script](s3/s3_compute.py) | [table](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/artifacts/reference_tables/s3/s3_final_table.csv) | A4, B3–B5, C1–C4, D1–D3, S2 reference tables |
| [S4](s4/s4_algorithm.md) | S41–S43 | Local representation structure | [script](s4/s4_compute.py) | [table](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/artifacts/reference_tables/s4/s4_final_table.csv) | Token cache |
| [X1](x1/x1_algorithm.md) | X11–X15 | Unsupported inference | [script](x1/x1_compute.py) | [table](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/artifacts/reference_tables/x1/x1_final_table.csv) | Static refusal catalog |
| [X2](x2/x2_algorithm.md) | X21–X26 | Unavailable data or scale | [script](x2/x2_compute.py) | [table](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/artifacts/reference_tables/x2/x2_final_table.csv) | Static refusal catalog |
| [X3](x3/x3_algorithm.md) | X31–X33 | Representation boundary | [script](x3/x3_compute.py) | [table](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/artifacts/reference_tables/x3/x3_final_table.csv) | Static refusal catalog |

## Setup

Complete the [project installation](../../../../README.md#installation), then
install task dependencies and the published inputs from the repository root:

```bash
pip install -e '.[tasks]'
python -m egms_qa.release install --download --components qa tokens tiles
hf download risenyard/egms-qa-encoder --local-dir data/encoder/checkpoint
```

The [Dataset](https://huggingface.co/datasets/risenyard/egms-qa-dataset#data-files)
provides tiles, tokens, reference tables, and metadata. Encoder files are used
by A1/A2 and by token extraction for new tiles. GPU execution is recommended
for these steps; the remaining calculations run on CPU.

## Run the task system

Run all 27 groups in dependency order:

```bash
python -m egms_qa.qa_construction.run_tasks \
    --out-dir outputs/tasks-rebuilt
```

The runner handles A1/A2 computation and aggregation and passes every new
upstream table to its dependent tasks. It uses C4's released cutoff and keeps
each method's existing parameters. It writes
`outputs/tasks-rebuilt/<group>/<group>_final_table.csv`, plus per-group diagnostics,
`logs/`, and `run.json`. Labels and QA are separate steps in the
[QA construction guide](../README.md#reproduce-task-computation).

Add `--dry-run` to inspect the plan without writing files. `--device cpu` or
`--device cuda:0` selects the execution device; `--workers` controls CPU workers.
Use a new output directory, or add `--resume` to continue an interrupted run.
Resume verifies inputs, parameters, code, dependency versions, and completed artifacts before
skipping any group. A failed group stops downstream execution.

The Dataset tables are fixed reference results. Re-execution can differ in
encoder-dependent values, labels near cutoffs, and percentile ranks when
scores are tied. For exact released targets, use the
[published reference tables](https://huggingface.co/datasets/risenyard/egms-qa-dataset/tree/main/artifacts/reference_tables)
with the [QA generation workflow](../README.md#reproduce-qa-generation).

### New tiles

Provide a manifest with `tile_id`, `split`, and `path`, and NPZ files matching
the [tile contract](https://huggingface.co/datasets/risenyard/egms-qa-dataset#source-tile-contract).
The complete task system requires displacement histories, coordinates, and the
eight static fields listed in the [tile reader](../../../egms_encoder/data/tile_store.py).
The runner derives point counts and centroids if the manifest omits them.

```bash
python -m egms_qa.qa_construction.run_tasks \
    --mode new-tiles --manifest my_data/split.parquet \
    --out-dir outputs/tasks-new
```

Tokens are extracted with the released encoder when `--token-cache` is omitted.
A supplied cache must match the tile IDs, splits, encoder, normalization, and
data configuration. Use `--source-tiles-root` or `--data-config` to specify
local inputs; the stored 294-step vertical-displacement contract still applies.

New tiles use the published training population for thresholds, percentile
ranks, representation transforms, and S1 profiles. No new-tile rows enter the
fit, even when their split is `train`. The resulting reference state is saved
locally as `reference_state.joblib`; its source files and hashes are recorded
in `run.json`. New collections need not contain 10,000 tiles or a train split.

### Task reference

Individual implementations remain callable through their linked modules;
use `--help` for task-specific options. Corpus-relative labels retain the
reference population specified by each method. Re-estimating European
candidate-pool cutoffs requires the separate reference data described by that
method and is outside this runner.
