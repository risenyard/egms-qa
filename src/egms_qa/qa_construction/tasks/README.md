# Task implementation index

The 78 leaf tasks are organized into 27 task groups under six families
(A/B/C/D/S/X). A directory such as `a1/` implements A11 and A12; it is not a
separate top-level family. See the [QA guide](../README.md) for all leaf-task
meanings and the Dataset installation commands.

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

| Group | Leaf tasks | Subject | Implementation | Algorithm | Main inputs / steps |
|---|---|---|---|---|---|
| A1 | A11–A12 | Representation drift and stability | [script](a1/a1_compute.py) | [method](a1/a1_algorithm.md) | NPZ tiles, encoder and token cache; combine shards |
| A2 | A21–A22 | Masked reconstruction and reliability | [script](a2/a2_compute.py) | [method](a2/a2_algorithm.md) | NPZ tiles and encoder; combine shards |
| A3 | A31–A32 | Spatial observation coverage | [script](a3/a3_compute.py) | [method](a3/a3_algorithm.md) | Token validity masks |
| A4 | A41–A42 | Measurement noise | [script](a4/a4_compute.py) | [method](a4/a4_algorithm.md) | NPZ tiles |
| A5 | A51–A52 | Monitoring usability | [script](a5/a5_compute.py) | [method](a5/a5_algorithm.md) | A1, A2, A3, A4 tables |
| B1 | B11–B12 | Subsidence signal-to-noise ratio | [script](b1/b1_compute.py) | [method](b1/b1_algorithm.md) | NPZ tiles |
| B2 | B21–B22 | Mean velocity | [script](b2/b2_compute.py) | [method](b2/b2_algorithm.md) | NPZ tiles |
| B3 | B31–B36 | Velocity tails and direction | [script](b3/b3_compute.py) | [method](b3/b3_algorithm.md) | NPZ tiles |
| B4 | B41–B42 | Acceleration strength | [script](b4/b4_compute.py) | [method](b4/b4_algorithm.md) | NPZ tiles |
| B5 | B51 | Seasonality strength | [script](b5/b5_compute.py) | [method](b5/b5_algorithm.md) | NPZ tiles |
| B6 | B61 | Monitoring trigger | [script](b6/b6_compute.py) | [method](b6/b6_algorithm.md) | B3, B4 tables |
| C1 | C11–C13 | Moving fraction and location | [script](c1/c1_compute.py) | [method](c1/c1_algorithm.md) | NPZ tiles |
| C2 | C21–C22 | Spatial concentration | [script](c2/c2_compute.py) | [method](c2/c2_algorithm.md) | NPZ tiles |
| C3 | C31–C33 | Deformation fronts | [script](c3/c3_compute.py) | [method](c3/c3_algorithm.md) | NPZ tiles |
| C4 | C41–C42 | Fast-tail spatial extent | [script](c4/c4_compute.py) | [method](c4/c4_algorithm.md) | NPZ tiles |
| C5 | C51–C52 | Monitoring priority | [script](c5/c5_compute.py) | [method](c5/c5_algorithm.md) | B2, B3, B6, C3 tables |
| D1 | D11–D14 | Trend geometry and changepoints | [script](d1/d1_compute.py) | [method](d1/d1_algorithm.md) | NPZ tiles, split manifest and data config |
| D2 | D21–D24 | Seasonal phase and amplitude | [script](d2/d2_compute.py) | [method](d2/d2_algorithm.md) | NPZ tiles, data config, B5 table |
| D3 | D31–D35 | Motion intensification | [script](d3/d3_compute.py) | [method](d3/d3_algorithm.md) | NPZ tiles |
| D4 | D41–D42 | Dominant temporal process | [script](d4/d4_compute.py) | [method](d4/d4_algorithm.md) | B3, B4, B5, D1, D2, D3 tables |
| S1 | S11–S15 | Representation anchors | [script](s1/s1_compute.py) | [method](s1/s1_algorithm.md) | Token cache |
| S2 | S21–S22 | Representation isolation | [script](s2/s2_compute.py) | [method](s2/s2_algorithm.md) | Token cache |
| S3 | S31–S33 | Representation–monitoring relation | [script](s3/s3_compute.py) | [method](s3/s3_algorithm.md) | A4, B3–B5, C1–C4, D2–D3, S2 tables and frozen S3 temporal inputs |
| S4 | S41–S43 | Local representation structure | [script](s4/s4_compute.py) | [method](s4/s4_algorithm.md) | Token cache |
| X1 | X11–X15 | Unsupported inference | [script](x1/x1_compute.py) | [method](x1/x1_algorithm.md) | Static refusal catalog |
| X2 | X21–X26 | Unavailable data or scale | [script](x2/x2_compute.py) | [method](x2/x2_algorithm.md) | Static refusal catalog |
| X3 | X31–X33 | Representation boundary | [script](x3/x3_compute.py) | [method](x3/x3_algorithm.md) | Static refusal catalog |

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
