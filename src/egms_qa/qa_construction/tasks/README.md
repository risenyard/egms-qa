# Task implementation index

The 78 leaf tasks are organized into 27 task groups under six families
(A/B/C/D/S/X). A directory such as `a1/` implements A11 and A12; it is not a
separate top-level family. Each linked task page describes all of its leaf tasks. See the
[QA construction guide](../README.md#workflows) for reference-table reproduction,
task recomputation, and construction from your own data.

## Read the task system

A tile is a spatial sample of EGMS point displacement observations. Each task
returns a numeric value, class, location, description, or refusal rule.
A task group collects related outputs; for example, A11 measures a score and
A12 classifies that same score.

| Family | What it describes |
|---|---|
| A | Observation and representation quality, combined into monitoring usability |
| B | Motion magnitude, direction, seasonality, and a monitoring trigger |
| C | Spatial extent, concentration, fronts, and monitoring context |
| D | Temporal trend shape, seasonal timing, intensification, and their combination |
| S | Similarity, rarity, and local structure of encoder vectors |
| X | Questions that require unsupported inference or unavailable information |

Start with the [group index](#task-groups) for definitions and results, or
[Run and files](#run-and-files) to compute a table. Each method introduces its
own inputs and explains any upstream task it uses. “Training percentile” means
a cutoff fitted on training tiles and then applied to all splits;
“corpus-relative” means that the reference population determines the scale.

## Setup

Install the code using the [project installation](../../../../README.md#installation).
For tasks that read measurements, representations, or upstream tables, also
[install the required data groups](../README.md#installation-and-data-setup):
`qa` supplies reference tables, `tokens` supplies encoder representations, and
`tiles` supplies source measurements. Select groups using the input columns
in the task table below. Run commands from the cloned repository root.

```bash
pip install -e '.[tasks]'
```

A1 and A2 additionally use the released encoder files:

```bash
hf download risenyard/egms-qa-encoder --local-dir data/encoder/checkpoint
```

The X1–X3 catalogs are static Python definitions. They require the code
installation but no Dataset or encoder download.

## Paths

The [Run and files](#run-and-files) section below collects commands for every
group. Shared inputs and their installed paths are listed once here. Each
method page links directly to its command.

Task scripts also honor `EGMS_QA_ROOT`, `EGMS_QA_DATA`, and `EGMS_QA_OUTPUTS`
from the [shared path configuration](../../paths.py). Set these to absolute
paths when running from another directory. Explicit command-line paths take
precedence over defaults; the manifest's `data/tiles/` entries resolve under
`EGMS_QA_DATA`.

| location | role |
|---|---|
| HF `artifacts/reference_tables/<group>/` | published reference tables and explicitly listed scientific inputs |
| local `outputs/tasks/<group>/` | installed links to those HF files |
| local `outputs/tasks-rebuilt/<group>/` | new outputs from the documented computation command |
| GitHub `tasks/<group>/` | Python implementation and method documentation |

A referenced CSV is not tracked beside the algorithm merely because the
method mentions its filename. **HF file** links identify published inputs;
other summaries, plots, and intermediate files are generated locally when
supported by the script. Computation outputs default to
`outputs/tasks-rebuilt/<group>/`, keeping installed reference files intact.
A1/A2 write numbered work shards there; their combiners default to one shard.
When using multiple shards, pass the same shard count to the computation and
combination commands. To use rebuilt tables as downstream inputs, pass the
corresponding input flags listed under its command below.

### Shared input files

These paths are created by the downloads in [Setup](#setup). The per-group
commands below state which inputs are needed. A manifest maps tile IDs and
splits to NPZ files; measurements live in the NPZ files themselves.

| Required input | Published source | Installed path |
|---|---|---|
| NPZ source tiles | [HF file](https://huggingface.co/datasets/risenyard/egms-qa-dataset/tree/main/artifacts/source_tiles) | `data/tiles/` |
| Split manifest | [HF file](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/metadata/split_manifest.parquet) | `data/encoder/manifest/split.parquet` |
| Data configuration | [HF file](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/metadata/data_config.json) | `data/encoder/manifest/data_config.json` |
| Encoder token cache | [HF file](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/artifacts/representations/egms_tokens_10k.pt) | `data/encoder/tokens/egms_tokens_10k.pt` |
| Encoder weights | [HF file](https://huggingface.co/risenyard/egms-qa-encoder/blob/main/encoder.safetensors) | `data/encoder/checkpoint/encoder.safetensors` |
| Encoder configuration | [HF file](https://huggingface.co/risenyard/egms-qa-encoder/blob/main/config.json) | `data/encoder/checkpoint/config.json` |
| Encoder normalization | [HF file](https://huggingface.co/risenyard/egms-qa-encoder/blob/main/normalization.json) | `data/encoder/checkpoint/normalization.json` |
| Encoder training recipe | [HF file](https://huggingface.co/risenyard/egms-qa-encoder/blob/main/training_args.json) | `data/encoder/checkpoint/training_args.json` |

## Shared modules

| Module | Responsibility |
|---|---|
| [task_specs.py](../task_specs.py) | Canonical columns, types and names for the 64 token-dependent targets |
| [inputs.py](../inputs.py) | Resolve source-tile paths and load manifests with validated tile/split keys |
| [tables.py](../tables.py) | Read task tables and reject missing, duplicate, or inconsistent join keys |
| [build_labels.py](../build_labels.py) | Aggregate published tables into labels and collect the 14 X refusal tasks |
| [qa_lib.py](../qa_lib.py) | Question phrasing, answer rendering and validation |
| [generate_qa.py](../generate_qa.py) | Generate the three QA splits |

Label aggregation enforces the released 10,000-tile contract. Task-specific
fitting, thresholds, and label rules remain in their task groups.

For tasks with train-fitted thresholds or representations, a small sample is
only a run check. Reproducing the published labels requires the full training
reference pool described in the method, including when classifying validation
or test tiles.

## Task groups

The script links identify the implementation entry points. They are not a
claim that every group can be rebuilt from the released inputs alone; see the
reconstruction scope below. Each released table lives at
`artifacts/reference_tables/<group>/<group>_final_table.csv` in
[the Dataset](https://huggingface.co/datasets/risenyard/egms-qa-dataset/tree/main/artifacts/reference_tables).
The release installer exposes these under `outputs/tasks/<group>/`.

| Group and method | Leaf tasks | Subject | Implementation | HF reference table | Main inputs / steps |
|---|---|---|---|---|---|
| [A1](a1/a1_algorithm.md) | A11–A12 | Representation drift and stability | [script](a1/a1_compute.py) · [run](#run-a1) | [table](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/artifacts/reference_tables/a1/a1_final_table.csv) | NPZ tiles, encoder and token cache; combine shards |
| [A2](a2/a2_algorithm.md) | A21–A22 | Masked reconstruction and reliability | [script](a2/a2_compute.py) · [run](#run-a2) | [table](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/artifacts/reference_tables/a2/a2_final_table.csv) | NPZ tiles and encoder; combine shards |
| [A3](a3/a3_algorithm.md) | A31–A32 | Spatial observation coverage | [script](a3/a3_compute.py) · [run](#run-a3) | [table](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/artifacts/reference_tables/a3/a3_final_table.csv) | Cached spatial-cell point counts |
| [A4](a4/a4_algorithm.md) | A41–A42 | Measurement noise | [script](a4/a4_compute.py) · [run](#run-a4) | [table](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/artifacts/reference_tables/a4/a4_final_table.csv) | NPZ tiles |
| [A5](a5/a5_algorithm.md) | A51–A52 | Monitoring usability | [script](a5/a5_compute.py) · [run](#run-a5) | [table](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/artifacts/reference_tables/a5/a5_final_table.csv) | A1, A2, A3, A4 tables |
| [B1](b1/b1_algorithm.md) | B11–B12 | Subsidence signal-to-noise ratio | [script](b1/b1_compute.py) · [run](#run-b1) | [table](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/artifacts/reference_tables/b1/b1_final_table.csv) | NPZ tiles |
| [B2](b2/b2_algorithm.md) | B21–B22 | Mean velocity | [script](b2/b2_compute.py) · [run](#run-b2) | [table](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/artifacts/reference_tables/b2/b2_final_table.csv) | NPZ tiles |
| [B3](b3/b3_algorithm.md) | B31–B36 | Velocity tails and direction | [script](b3/b3_compute.py) · [run](#run-b3) | [table](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/artifacts/reference_tables/b3/b3_final_table.csv) | NPZ tiles |
| [B4](b4/b4_algorithm.md) | B41–B42 | Acceleration strength | [script](b4/b4_compute.py) · [run](#run-b4) | [table](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/artifacts/reference_tables/b4/b4_final_table.csv) | NPZ tiles |
| [B5](b5/b5_algorithm.md) | B51 | Seasonality strength | [script](b5/b5_compute.py) · [run](#run-b5) | [table](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/artifacts/reference_tables/b5/b5_final_table.csv) | NPZ tiles |
| [B6](b6/b6_algorithm.md) | B61 | Monitoring trigger | [script](b6/b6_compute.py) · [run](#run-b6) | [table](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/artifacts/reference_tables/b6/b6_final_table.csv) | B3, B4 tables |
| [C1](c1/c1_algorithm.md) | C11–C13 | Moving fraction and location | [script](c1/c1_compute.py) · [run](#run-c1) | [table](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/artifacts/reference_tables/c1/c1_final_table.csv) | NPZ tiles |
| [C2](c2/c2_algorithm.md) | C21–C22 | Spatial concentration | [script](c2/c2_compute.py) · [run](#run-c2) | [table](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/artifacts/reference_tables/c2/c2_final_table.csv) | NPZ tiles |
| [C3](c3/c3_algorithm.md) | C31–C33 | Deformation fronts | [script](c3/c3_compute.py) · [run](#run-c3) | [table](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/artifacts/reference_tables/c3/c3_final_table.csv) | NPZ tiles |
| [C4](c4/c4_algorithm.md) | C41–C42 | Fast-tail spatial extent | [script](c4/c4_compute.py) · [run](#run-c4) | [table](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/artifacts/reference_tables/c4/c4_final_table.csv) | NPZ tiles |
| [C5](c5/c5_algorithm.md) | C51–C52 | Monitoring priority | [script](c5/c5_compute.py) · [run](#run-c5) | [table](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/artifacts/reference_tables/c5/c5_final_table.csv) | B2, B3, B6, C3 tables |
| [D1](d1/d1_algorithm.md) | D11–D14 | Trend geometry and changepoints | [script](d1/d1_compute.py) · [run](#run-d1) | [table](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/artifacts/reference_tables/d1/d1_final_table.csv) | NPZ tiles, split manifest and data config |
| [D2](d2/d2_algorithm.md) | D21–D24 | Seasonal phase and amplitude | [script](d2/d2_compute.py) · [run](#run-d2) | [table](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/artifacts/reference_tables/d2/d2_final_table.csv) | NPZ tiles, data config, B5 table |
| [D3](d3/d3_algorithm.md) | D31–D35 | Motion intensification | [script](d3/d3_compute.py) · [run](#run-d3) | [table](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/artifacts/reference_tables/d3/d3_final_table.csv) | NPZ tiles |
| [D4](d4/d4_algorithm.md) | D41–D42 | Dominant temporal process | [script](d4/d4_compute.py) · [run](#run-d4) | [table](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/artifacts/reference_tables/d4/d4_final_table.csv) | B3, B4, B5, D1, D2, D3 tables |
| [S1](s1/s1_algorithm.md) | S11–S15 | Representation anchors | [script](s1/s1_compute.py) · [run](#run-s1) | [table](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/artifacts/reference_tables/s1/s1_final_table.csv) | Token cache |
| [S2](s2/s2_algorithm.md) | S21–S22 | Representation isolation | [script](s2/s2_compute.py) · [run](#run-s2) | [table](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/artifacts/reference_tables/s2/s2_final_table.csv) | Token cache |
| [S3](s3/s3_algorithm.md) | S31–S33 | Representation–monitoring relation | [script](s3/s3_compute.py) · [run](#run-s3) | [table](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/artifacts/reference_tables/s3/s3_final_table.csv) | A4, B3–B5, C1–C4, D1–D3, S2 reference tables |
| [S4](s4/s4_algorithm.md) | S41–S43 | Local representation structure | [script](s4/s4_compute.py) · [run](#run-s4) | [table](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/artifacts/reference_tables/s4/s4_final_table.csv) | Token cache |
| [X1](x1/x1_algorithm.md) | X11–X15 | Unsupported inference | [script](x1/x1_compute.py) · [run](#run-x1) | [table](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/artifacts/reference_tables/x1/x1_final_table.csv) | Static refusal catalog |
| [X2](x2/x2_algorithm.md) | X21–X26 | Unavailable data or scale | [script](x2/x2_compute.py) · [run](#run-x2) | [table](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/artifacts/reference_tables/x2/x2_final_table.csv) | Static refusal catalog |
| [X3](x3/x3_algorithm.md) | X31–X33 | Representation boundary | [script](x3/x3_compute.py) · [run](#run-x3) | [table](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/artifacts/reference_tables/x3/x3_final_table.csv) | Static refusal catalog |

A1 and A2 also provide [A1 shard aggregation](a1/a1_combine_shards.py) and
[A2 shard aggregation](a2/a2_combine_shards.py). Use their `--help` output for
shard locations and final-table destinations. X1–X3 write task-level refusal
catalogs, rather than 10,000 tile-level records.

## Reconstruction scope

The supported complete QA workflow starts from the released reference tables:
rebuild labels, then render questions and answers. The Dataset supplies all 27
tables needed for that workflow. The [reference-table workflow](../README.md#reproduce-qa-construction)
provides the label and QA commands. The
[task recomputation workflow](../README.md#recompute-task-labels-and-regenerate-qa)
shows how to replace B2 and its dependent C5 table in a working set, then pass
that set to label aggregation. Both write outputs separately from installed
reference files.

D1 computes the published geometry scores directly from the model-ready NPZ
tiles and source time axis. D4 reads its trend-shape label; S3 reads its
curvature and changepoint strengths alongside the other reference tables.
See the [D4](d4/d4_algorithm.md) and [S3](s3/s3_algorithm.md) methods for the
input columns and the `--d1-table` override for a newly computed D1 table.

For other groups, consult the linked method and script for the full input,
parameter and execution requirements. The index records dependencies; it does
not assert numerical equivalence of independently recomputed tables.

## Run and files

Complete [Setup](#setup), then run the chosen command from the repository root.
The [shared input files](#shared-input-files) table gives download sources and
installed paths. Upstream reference tables are linked in the [group index](#task-groups).

Each command writes `outputs/tasks-rebuilt/<group>/<group>_final_table.csv`.
Downloaded reference tables remain at `outputs/tasks/<group>/<group>_final_table.csv`.
Scripts may also generate local summaries and diagnostics beside their new
table; these are not additional downloadable reference files.

Commands use the installed reference tables for upstream inputs by default.
Rebuilding a group does not automatically change the inputs of later groups.
To use new tables, supply the input flags listed below with their rebuilt paths.
Thus A5 follows A1–A4; B6 follows B3/B4; C5 follows B2/B3/B6/C3; D2 follows B5;
D4 follows B3/B4/B5/D1/D2/D3; and S3 follows its listed measurement groups and S2.
X1–X3 are independent static catalogs.

### Run A1

[Method and results](a1/a1_algorithm.md) · [Implementation](a1/a1_compute.py)

Inputs: NPZ source tiles, Split manifest, Data configuration, Encoder token cache, Encoder weights, Encoder configuration, Encoder normalization.

```bash
python -m egms_qa.qa_construction.tasks.a1.a1_compute \
    --out-dir outputs/tasks-rebuilt/a1/work/shards/shard_0 \
    --num-shards 1 --shard-index 0 --device cuda:0
python -m egms_qa.qa_construction.tasks.a1.a1_combine_shards \
    --base-dir outputs/tasks-rebuilt/a1/work --num-shards 1 \
    --out-path outputs/tasks-rebuilt/a1/a1_final_table.csv
```

This example uses CUDA and one work shard. For multiple shards, run every shard index with the same `--num-shards` value, then combine using that count. The combiner removes the group’s `work/` directory by default; add `--keep-work` to retain intermediate files.

### Run A2

[Method and results](a2/a2_algorithm.md) · [Implementation](a2/a2_compute.py)

Inputs: NPZ source tiles, Split manifest, Data configuration, Encoder weights, Encoder configuration, Encoder normalization, Encoder training recipe.

```bash
python -m egms_qa.qa_construction.tasks.a2.a2_compute \
    --out-dir outputs/tasks-rebuilt/a2/work/shards/shard_0 \
    --num-shards 1 --shard-index 0 --device cuda:0
python -m egms_qa.qa_construction.tasks.a2.a2_combine_shards \
    --base-dir outputs/tasks-rebuilt/a2/work --num-shards 1 \
    --out-path outputs/tasks-rebuilt/a2/a2_final_table.csv
```

This example uses CUDA and one work shard. For multiple shards, run every shard index with the same `--num-shards` value, then combine using that count. The combiner removes the group’s `work/` directory by default; add `--keep-work` to retain intermediate files.

### Run A3

[Method and results](a3/a3_algorithm.md) · [Implementation](a3/a3_compute.py)

Inputs: Encoder token cache.

```bash
python -m egms_qa.qa_construction.tasks.a3.a3_compute \
    --out-path outputs/tasks-rebuilt/a3/a3_final_table.csv
```

### Run A4

[Method and results](a4/a4_algorithm.md) · [Implementation](a4/a4_compute.py)

Inputs: NPZ source tiles, Split manifest, Data configuration.

```bash
python -m egms_qa.qa_construction.tasks.a4.a4_compute \
    --out-path outputs/tasks-rebuilt/a4/a4_final_table.csv
```

### Run A5

[Method and results](a5/a5_algorithm.md) · [Implementation](a5/a5_compute.py)

Inputs: [A1 table](a1/a1_algorithm.md), [A2 table](a2/a2_algorithm.md), [A3 table](a3/a3_algorithm.md), [A4 table](a4/a4_algorithm.md).

```bash
python -m egms_qa.qa_construction.tasks.a5.a5_compute \
    --out-path outputs/tasks-rebuilt/a5/a5_final_table.csv
```

Use `--a1-path`, `--a2-path`, `--a3-path`, and `--a4-path` to supply rebuilt quality tables.

### Run B1

[Method and results](b1/b1_algorithm.md) · [Implementation](b1/b1_compute.py)

Inputs: NPZ source tiles, Split manifest.

```bash
python -m egms_qa.qa_construction.tasks.b1.b1_compute \
    --out-dir outputs/tasks-rebuilt/b1
```

### Run B2

[Method and results](b2/b2_algorithm.md) · [Implementation](b2/b2_compute.py)

Inputs: NPZ source tiles, Split manifest.

```bash
python -m egms_qa.qa_construction.tasks.b2.b2_compute \
    --out-dir outputs/tasks-rebuilt/b2
```

### Run B3

[Method and results](b3/b3_algorithm.md) · [Implementation](b3/b3_compute.py)

Inputs: NPZ source tiles, Split manifest.

```bash
python -m egms_qa.qa_construction.tasks.b3.b3_compute \
    --out-dir outputs/tasks-rebuilt/b3
```

### Run B4

[Method and results](b4/b4_algorithm.md) · [Implementation](b4/b4_compute.py)

Inputs: NPZ source tiles, Split manifest.

```bash
python -m egms_qa.qa_construction.tasks.b4.b4_compute \
    --out-dir outputs/tasks-rebuilt/b4
```

### Run B5

[Method and results](b5/b5_algorithm.md) · [Implementation](b5/b5_compute.py)

Inputs: NPZ source tiles, Split manifest.

```bash
python -m egms_qa.qa_construction.tasks.b5.b5_compute \
    --out-dir outputs/tasks-rebuilt/b5
```

### Run B6

[Method and results](b6/b6_algorithm.md) · [Implementation](b6/b6_compute.py)

Inputs: [B3 table](b3/b3_algorithm.md), [B4 table](b4/b4_algorithm.md).

```bash
python -m egms_qa.qa_construction.tasks.b6.b6_compute \
    --out-dir outputs/tasks-rebuilt/b6
```

Use `--b3-table` and `--b4-table` to supply rebuilt motion tables.

### Run C1

[Method and results](c1/c1_algorithm.md) · [Implementation](c1/c1_compute.py)

Inputs: NPZ source tiles, Split manifest.

```bash
python -m egms_qa.qa_construction.tasks.c1.c1_compute \
    --out-dir outputs/tasks-rebuilt/c1
```

### Run C2

[Method and results](c2/c2_algorithm.md) · [Implementation](c2/c2_compute.py)

Inputs: NPZ source tiles, Split manifest.

```bash
python -m egms_qa.qa_construction.tasks.c2.c2_compute \
    --out-dir outputs/tasks-rebuilt/c2
```

### Run C3

[Method and results](c3/c3_algorithm.md) · [Implementation](c3/c3_compute.py)

Inputs: NPZ source tiles, Split manifest.

```bash
python -m egms_qa.qa_construction.tasks.c3.c3_compute \
    --out-dir outputs/tasks-rebuilt/c3
```

### Run C4

[Method and results](c4/c4_algorithm.md) · [Implementation](c4/c4_compute.py)

Inputs: NPZ source tiles, Split manifest.

```bash
python -m egms_qa.qa_construction.tasks.c4.c4_compute final \
    --out-dir outputs/tasks-rebuilt/c4
```

`final` uses the fixed 4.8 mm/yr cutoff. Re-estimating it with `reference` requires the full European candidate-pool manifest, which is not distributed with this Dataset. That command creates a local reference JSON; pass it to `final` with `--reference-json`.

### Run C5

[Method and results](c5/c5_algorithm.md) · [Implementation](c5/c5_compute.py)

Inputs: [B2 table](b2/b2_algorithm.md), [B3 table](b3/b3_algorithm.md), [B6 table](b6/b6_algorithm.md), [C3 table](c3/c3_algorithm.md).

```bash
python -m egms_qa.qa_construction.tasks.c5.c5_compute \
    --out-dir outputs/tasks-rebuilt/c5
```

Use `--b2`, `--b3`, `--b6`, and `--c3` to supply rebuilt input tables.

### Run D1

[Method and results](d1/d1_algorithm.md) · [Implementation](d1/d1_compute.py)

Inputs: NPZ source tiles, Split manifest, Data configuration.

```bash
python -m egms_qa.qa_construction.tasks.d1.d1_compute \
    --out-dir outputs/tasks-rebuilt/d1
```

### Run D2

[Method and results](d2/d2_algorithm.md) · [Implementation](d2/d2_compute.py)

Inputs: NPZ source tiles, Split manifest, Data configuration, [B5 table](b5/b5_algorithm.md).

```bash
python -m egms_qa.qa_construction.tasks.d2.d2_compute \
    --out-dir outputs/tasks-rebuilt/d2
```

Use `--b51-table` to supply a rebuilt B5 table.

### Run D3

[Method and results](d3/d3_algorithm.md) · [Implementation](d3/d3_compute.py)

Inputs: NPZ source tiles, Split manifest.

```bash
python -m egms_qa.qa_construction.tasks.d3.d3_compute \
    --out-dir outputs/tasks-rebuilt/d3
```

### Run D4

[Method and results](d4/d4_algorithm.md) · [Implementation](d4/d4_compute.py)

Inputs: [B3 table](b3/b3_algorithm.md), [B4 table](b4/b4_algorithm.md), [B5 table](b5/b5_algorithm.md), [D1 table](d1/d1_algorithm.md), [D2 table](d2/d2_algorithm.md), [D3 table](d3/d3_algorithm.md).

```bash
python -m egms_qa.qa_construction.tasks.d4.d4_compute \
    --out-dir outputs/tasks-rebuilt/d4
```

Use `--b3-table`, `--b4-table`, `--b5-table`, `--d1-table`, `--d2-table`, and `--d3-table` to supply rebuilt input tables. For example, `--d1-table outputs/tasks-rebuilt/d1/d1_final_table.csv` supplies both the copied D11 value and the trend detail used for D42.

### Run S1

[Method and results](s1/s1_algorithm.md) · [Implementation](s1/s1_compute.py)

Inputs: Encoder token cache.

```bash
python -m egms_qa.qa_construction.tasks.s1.s1_compute \
    --out-dir outputs/tasks-rebuilt/s1
```

### Run S2

[Method and results](s2/s2_algorithm.md) · [Implementation](s2/s2_compute.py)

Inputs: Encoder token cache.

```bash
python -m egms_qa.qa_construction.tasks.s2.s2_compute \
    --out-dir outputs/tasks-rebuilt/s2
```

### Run S3

[Method and results](s3/s3_algorithm.md) · [Implementation](s3/s3_compute.py)

Inputs: [A4 table](a4/a4_algorithm.md), [B3 table](b3/b3_algorithm.md), [B4 table](b4/b4_algorithm.md), [B5 table](b5/b5_algorithm.md), [C1 table](c1/c1_algorithm.md), [C2 table](c2/c2_algorithm.md), [C3 table](c3/c3_algorithm.md), [C4 table](c4/c4_algorithm.md), [D1 table](d1/d1_algorithm.md), [D2 table](d2/d2_algorithm.md), [D3 table](d3/d3_algorithm.md), [S2 table](s2/s2_algorithm.md).

```bash
python -m egms_qa.qa_construction.tasks.s3.s3_compute \
    --out-dir outputs/tasks-rebuilt/s3
```

Use `--tasks-root outputs/tasks-rebuilt` when all required tables have been rebuilt, or `--d1-table outputs/tasks-rebuilt/d1/d1_final_table.csv` to replace only D1 while keeping the other installed reference tables.

### Run S4

[Method and results](s4/s4_algorithm.md) · [Implementation](s4/s4_compute.py)

Inputs: Encoder token cache.

```bash
python -m egms_qa.qa_construction.tasks.s4.s4_compute \
    --out-dir outputs/tasks-rebuilt/s4
```

### Run X1

[Method and results](x1/x1_algorithm.md) · [Implementation](x1/x1_compute.py)

Inputs: static definitions in the Python script; no data download.

```bash
python -m egms_qa.qa_construction.tasks.x1.x1_compute \
    --out-dir outputs/tasks-rebuilt/x1
```

### Run X2

[Method and results](x2/x2_algorithm.md) · [Implementation](x2/x2_compute.py)

Inputs: static definitions in the Python script; no data download.

```bash
python -m egms_qa.qa_construction.tasks.x2.x2_compute \
    --out-dir outputs/tasks-rebuilt/x2
```

### Run X3

[Method and results](x3/x3_algorithm.md) · [Implementation](x3/x3_compute.py)

Inputs: static definitions in the Python script; no data download.

```bash
python -m egms_qa.qa_construction.tasks.x3.x3_compute \
    --out-dir outputs/tasks-rebuilt/x3
```
