# Task implementation index

The 78 leaf tasks are organized into 27 task groups under six families
(A/B/C/D/S/X). A directory such as `a1/` implements A11 and A12; it is not a
separate top-level family. See the [QA guide](../README.md) for all leaf-task
meanings and the Dataset installation commands.

## Setup

Install the code using the [project installation](../../../../README.md#installation).
For tasks that read measurements, representations, or upstream tables, also
[install the Dataset](../README.md#use-the-released-records). Run all task
commands from the cloned repository root.

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

Each method's **Run and files** section links its inputs to the actual HF files
and shows their installed paths. Use the paths in the commands from the
repository root.

| location | role |
|---|---|
| HF `artifacts/reference_tables/<group>/` | published reference tables and explicitly listed scientific inputs |
| local `outputs/tasks/<group>/` | installed links to those HF files |
| local `outputs/tasks-rebuilt/<group>/` | new outputs from the documented computation command |
| GitHub `tasks/<group>/` | Python implementation and method documentation |

A referenced CSV is not tracked beside the algorithm merely because the
method mentions its filename. **HF file** links identify published inputs;
other summaries, plots, and intermediate files are generated locally when
supported by the script. Keep the documented output override when recomputing
so that installed reference files stay intact.

## Shared modules

| Module | Responsibility |
|---|---|
| [task_specs.py](../task_specs.py) | Canonical columns, types and names for the 64 token-dependent targets |
| [tables.py](../tables.py) | Read complete task tables and align unique tile/split keys |
| [build_labels.py](../build_labels.py) | Aggregate published tables into labels and collect the 14 X refusal tasks |
| [qa_lib.py](../qa_lib.py) | Question phrasing, answer rendering and validation |
| [generate_qa.py](../generate_qa.py) | Generate the three QA splits |
| [summarize_temporal.py](../summarize_temporal.py) | Optional D1–D4 summary; see [usage](../temporal_summary.md) |

The summary tool is separate from task algorithms and writes to
`outputs/summaries/temporal/`. It does not define additional tasks or feed label
aggregation. Shared table utilities enforce the released 10,000-tile contract;
task-specific fitting, thresholds and label rules remain in their task groups.

## Task groups

The script links identify the implementation entry points. They are not a
claim that every group can be rebuilt from the released inputs alone; see the
reconstruction scope below. Each released table lives at
`artifacts/reference_tables/<group>/<group>_final_table.csv` in
[the Dataset](https://huggingface.co/datasets/risenyard/egms-qa-dataset/tree/main/artifacts/reference_tables).
The release installer exposes these under `outputs/tasks/<group>/`.

| Group and method | Leaf tasks | Subject | Implementation | HF reference table | Main inputs / steps |
|---|---|---|---|---|---|
| [A1](a1/a1_algorithm.md) | A11–A12 | Representation drift and stability | [script](a1/a1_compute.py) | [table](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/artifacts/reference_tables/a1/a1_final_table.csv) | NPZ tiles, encoder and token cache; combine shards |
| [A2](a2/a2_algorithm.md) | A21–A22 | Masked reconstruction and reliability | [script](a2/a2_compute.py) | [table](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/artifacts/reference_tables/a2/a2_final_table.csv) | NPZ tiles and encoder; combine shards |
| [A3](a3/a3_algorithm.md) | A31–A32 | Spatial observation coverage | [script](a3/a3_compute.py) | [table](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/artifacts/reference_tables/a3/a3_final_table.csv) | Token validity masks |
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
| [S3](s3/s3_algorithm.md) | S31–S33 | Representation–monitoring relation | [script](s3/s3_compute.py) | [table](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/artifacts/reference_tables/s3/s3_final_table.csv) | A4, B3–B5, C1–C4, D2–D3, S2 tables and frozen S3 temporal inputs |
| [S4](s4/s4_algorithm.md) | S41–S43 | Local representation structure | [script](s4/s4_compute.py) | [table](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/artifacts/reference_tables/s4/s4_final_table.csv) | Token cache |
| [X1](x1/x1_algorithm.md) | X11–X15 | Unsupported inference | [script](x1/x1_compute.py) | [table](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/artifacts/reference_tables/x1/x1_final_table.csv) | Static refusal catalog |
| [X2](x2/x2_algorithm.md) | X21–X26 | Unavailable data or scale | [script](x2/x2_compute.py) | [table](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/artifacts/reference_tables/x2/x2_final_table.csv) | Static refusal catalog |
| [X3](x3/x3_algorithm.md) | X31–X33 | Representation boundary | [script](x3/x3_compute.py) | [table](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/artifacts/reference_tables/x3/x3_final_table.csv) | Static refusal catalog |

A1 and A2 also provide [A1 shard aggregation](a1/a1_combine_shards.py) and
[A2 shard aggregation](a2/a2_combine_shards.py). Use their `--help` output for
shard locations and final-table destinations. X1–X3 write task-level refusal
catalogs, rather than 10,000 tile-level records.

## Reconstruction scope

The supported complete QA workflow starts from the released reference tables:
rebuild labels, then render questions and answers. The Dataset supplies all 27
tables needed for that workflow. The [QA guide](../README.md) gives commands
that write new outputs separately from the downloaded artifacts.

D1 computes the published geometry scores directly from the model-ready NPZ
tiles and source time axis. S3 reads the frozen posterior estimates in
`s3/s3_temporal_inputs.csv`, alongside the other released reference tables.
These inputs define the exact published S3 targets.

The [S3 temporal estimator](s3/s3_temporal_compute.py) provides an optional
BEAST refit from NPZ tiles. Its Monte Carlo estimates can vary across builds
and hardware even with fixed tile seeds. A fresh posterior realization is not
a byte-for-byte replacement for the frozen S3 inputs. See the
[S3 method](s3/s3_algorithm.md) for the commands and provenance.

For other groups, consult the linked method and script for the full input,
parameter and execution requirements. The index records dependencies; it does
not assert numerical equivalence of independently recomputed tables.

Install `pip install -e '.[tasks]'` before using task computation scripts.
Run them from the checkout root, and direct generated outputs to a separate
working directory instead of the installed reference-table links.
