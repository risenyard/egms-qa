# EGMS-QA Construction

This module builds task reference values, and renders
natural-language question–answer records. Each task group computes reference values from the required data or encoder outputs. 
These values provide the answers to predefined questions. See the [task index](tasks/README.md) for task definitions, inputs, and runnable scripts. The released tables and labels can
also be used as starting points for new QA corpora. 

## Workflows

This guide covers construction commands and the task catalog for
[EGMS-QA](../../../README.md). The
[Hugging Face Dataset card](https://huggingface.co/datasets/risenyard/egms-qa-dataset)
documents the released records, fields, file layout, and scientific provenance.

| goal | where to start |
|---|---|
| Read the published QA records | [HF QA use](https://huggingface.co/datasets/risenyard/egms-qa-dataset#qa-use) |
| Use released artifacts in this codebase | [Install the code](#installation), then [install the Dataset](#use-the-released-records) |
| Render a new QA corpus | [Generate question–answer records](#generate-questionanswer-records) |
| Rebuild labels or reference values | [Construction workflow](#construction-workflow) and [task index](tasks/README.md) |
| Find a task definition | [Task catalog](#task-catalog) |

## Installation

Python 3.10 or later is required. Clone and install the code, then run all
commands below from the `egms-qa` repository root.

```bash
git clone https://github.com/risenyard/egms-qa
cd egms-qa
pip install -e .
```

Label aggregation and QA rendering can run on CPU. Task-specific computation
may require additional dependencies or GPU execution, as described in the
linked methods.

## Use the released records

After [installing the code](#installation), download only the data you need.
Each command downloads and installs one group; groups can be added later.

| group | includes | used for |
|---|---|---|
| `qa` | QA pairs, labels, task reference tables, and QA metadata | reading and rendering QA |
| `tokens` | precomputed EGMS tokens and token metadata | translator workflows and token-based checks |
| `tiles` | source NPZ tiles and the data configuration | encoder workflows and measurement-based task computation |

QA pairs and construction files:

```bash
python -m egms_qa.release install --download --components qa
```

Precomputed EGMS tokens:

```bash
python -m egms_qa.release install --download --components tokens
```

Source tiles:

```bash
python -m egms_qa.release install --download --components tiles
```

Shared metadata is downloaded automatically. Installation checks the required
files and their integrity, then links them into the working directory. It
preserves existing files. Downloads are stored in the Hugging Face cache;
`--target-root` selects a different working directory.

Combine groups in one command when needed. For example, translator workflows
and label rebuilding use QA data and tokens:

```bash
python -m egms_qa.release install --download --components qa tokens
```

Omit `--components` to install all three groups. For an existing download, use
`--release-dir release/egms-qa-dataset` in place of `--download`, with the same
component selection.

| data | installed local path |
|---|---|
| QA pairs | `outputs/qa/v1_{train,val,test}.jsonl` |
| Labels and QA metadata | `outputs/qa/` |
| Task reference tables | `outputs/tasks/` |
| EGMS tokens | `data/encoder/tokens/` |
| Source tiles | `data/tiles/` |
| Split manifest and data configuration | `data/encoder/manifest/` |

For the Dataset's schemas, file layout, and direct streaming example, see the
[HF Dataset card](https://huggingface.co/datasets/risenyard/egms-qa-dataset).

## Generate question–answer records

Install the `qa` group, then start with a small rendering run from its labels
and approved phrasings:

```bash
python -m egms_qa.qa_construction.generate_qa \
    --out-dir outputs/qa-generated \
    --max-tiles 2 --train-cycles 1
```

The command writes `v1_train_e00.jsonl`, `v1_val.jsonl`, and `v1_test.jsonl`
under `outputs/qa-generated/qa/`. Record counts and rendering metadata appear
in the parent directory.

Remove `--max-tiles 2 --train-cycles 1` to render the full installed corpus
with the default two training phrasing cycles. This also writes
`v1_train_e01.jsonl`. Each token-dependent tile–task pair receives one phrasing
per cycle from a pool of 20. Refusal tasks use a capped sample of tiles per task.

These commands create new corpora in a separate output directory. Use the
Dataset's fixed `data/qa/{train,validation,test}.jsonl` files when reproducing
the published model results. Matching record counts alone does not establish
byte-for-byte reproduction of those files.

## Construction workflow

| step | implementation | output |
|---|---|---|
| Task reference values | group scripts and algorithm notes in `tasks/` | one reference table per task group |
| Label aggregation | `build_labels.py`, `task_specs.py`, and `tables.py` | `labels.parquet` and `labels_meta.json` |
| Question and answer rendering | `generate_qa.py` and `qa_lib.py` | split JSONL files and rendering metadata |

The [task implementation index](tasks/README.md) links all 27 task groups and
their dependencies. Install `qa tokens` to rebuild labels from the reference tables and
render them into a new directory:

```bash
python -m egms_qa.qa_construction.build_labels \
    --out-dir outputs/labels-generated
python -m egms_qa.qa_construction.generate_qa \
    --labels outputs/labels-generated/labels.parquet \
    --meta outputs/labels-generated/labels_meta.json \
    --out-dir outputs/qa-generated
```

The label builder checks table identities and splits against the published
encoder token cache, then aligns label rows to its tile order. The released
label file supplies the targets for the published QA records. The
optional [temporal summary](temporal_summary.md) combines D1–D4 tables for analysis.

Label building defaults to `outputs/labels-generated/`. It refuses to replace
existing output files or symbolic links; choose a different `--out-dir` for a
new build.

## Recompute D1, D4, and S3

Install `qa tiles` for these commands. D1 fits temporal geometry from the NPZ tiles and
data configuration. Its curvature and changepoint thresholds are fitted on
the training split. S3 uses D1 curvature and changepoint strength alongside
the other A/B/C/D monitoring indicators. D4 uses D1 trend shape to distinguish
the trend-dominated evolution archetypes.

```bash
pip install -e '.[tasks]'
python -m egms_qa.qa_construction.tasks.d1.d1_compute \
    --out-dir outputs/tasks-rebuilt/d1 --workers 8
python -m egms_qa.qa_construction.tasks.d4.d4_compute \
    --d1-table outputs/tasks-rebuilt/d1/d1_final_table.csv \
    --out-dir outputs/tasks-rebuilt/d4
python -m egms_qa.qa_construction.tasks.s3.s3_compute \
    --d1-table outputs/tasks-rebuilt/d1/d1_final_table.csv \
    --out-dir outputs/tasks-rebuilt/s3
```

These commands write outputs separately from the installed tables. The
[D1 method](tasks/d1/d1_algorithm.md), [D4 method](tasks/d4/d4_algorithm.md),
and [S3 method](tasks/s3/s3_algorithm.md) describe their inputs and formulas.
D4 and S3 both use the D1 file passed in the commands above.

The label-generation examples above use the canonical released tables.
Consult the task index for other groups' dependencies and reconstruction scope.

## Dataset contract

The [Dataset card](https://huggingface.co/datasets/risenyard/egms-qa-dataset)
is the reference for QA fields, split sizes, source-tile arrays, normalization,
and supporting scientific inputs. Its
[source provenance](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/SOURCE_PROVENANCE.md)
documents the EGMS product and the relationship between stored and source time indices.

## Task catalog

The catalog contains 78 tasks in 27 task groups under six families
(A/B/C/D/S/X). Families A–D and S contain 64 token-dependent tasks with numeric
or categorical targets. Family X contains
14 refusal tasks for questions outside the supported scope. Numeric targets
describe quantities, categorical targets assign classes, and refusal targets
state the evidence boundary.

| family | task groups | leaf tasks | focus |
|---|---:|---:|---|
| A | 5 | 10 | Observation quality and usability |
| B | 6 | 14 | Motion magnitude, direction, and typicality |
| C | 5 | 12 | Spatial organization and monitoring context |
| D | 4 | 15 | Temporal trend, seasonality, and intensification |
| S | 4 | 13 | Properties of the encoder representation |
| X | 3 | 14 | Boundaries of supported questions |

The tables use `num` for numeric targets and `cat` for categorical targets.
The translator's reported evaluation uses a defined 71-task subset of this
catalog. The [evaluation protocol](../translator/README.md#evaluation-protocol)
describes its three answer types and links to the exact configuration.

### A: observation gate (10)
Record quality and representation reliability.

| Task | Type | Description | Task | Type | Description |
|---|---|---|---|---|---|
| A11 | num | global representation drift | A12 | cat | representation stability class |
| A21 | num | masked reconstruction loss | A22 | cat | reconstruction reliability class |
| A31 | num | spatial observation coverage | A32 | cat | spatial coverage class |
| A41 | num | median measurement noise | A42 | cat | measurement noise class |
| A51 | cat | monitoring usability gate | A52 | cat | monitoring usability reason |

### B: motion vital signs (14)
Strength and character of the observed motion.

| Task | Type | Description | Task | Type | Description |
|---|---|---|---|---|---|
| B11 | num | average subsidence SNR | B12 | cat | clear subsidence signal |
| B21 | num | mean velocity | B22 | cat | mean subsidence intensity band |
| B31 | num | sinking-tail velocity | B32 | num | upper-tail velocity |
| B33 | num | absolute tail velocity | B34 | cat | uplift-protected direction |
| B35 | cat | worst-point significance | B36 | cat | European velocity typicality |
| B41 | num | acceleration strength | B42 | cat | European acceleration typicality |
| B51 | num | seasonality strength | B61 | cat | monitoring trigger |

### C: spatial organization (12)
The spatial distribution of motion within the tile.

| Task | Type | Description | Task | Type | Description |
|---|---|---|---|---|---|
| C11 | num | moving-point fraction | C12 | cat | motion extent class |
| C13 | cat | strongest-motion bin | C21 | num | spatial concentration |
| C22 | cat | concentration class | C31 | num | deformation-front strength |
| C32 | cat | front location | C33 | cat | front strength class |
| C41 | num | fast-tail bin fraction | C42 | cat | fast-tail extent class |
| C51 | cat | monitoring priority | C52 | cat | hidden local risk |

### D: temporal dynamics (15)
Changes in motion over the observation period.

| Task | Type | Description | Task | Type | Description |
|---|---|---|---|---|---|
| D11 | cat | trend shape | D12 | num | curvature strength |
| D13 | num | changepoint strength | D14 | num | strong-changepoint time |
| D21 | cat | dominant seasonal phase | D22 | num | seasonal phase coherence |
| D23 | num | seasonal phase dispersion | D24 | num | seasonal amplitude change |
| D31 | num | motion intensification | D32 | num | acceleration spatial support |
| D33 | num | intensification spread | D34 | num | hotspot strength |
| D35 | cat | hotspot location | D41 | cat | dominant process |
| D42 | cat | evolution archetype | | | |

### S: representation constructs (13)
Properties of the frozen encoder representation.

| Task | Type | Description | Task | Type | Description |
|---|---|---|---|---|---|
| S11 | cat | reference anchor profile | S12 | num | nearest anchor distance |
| S13 | num | anchor margin | S14 | cat | assignment status |
| S15 | cat | anchor profile description | S21 | num | local isolation score |
| S22 | cat | representation rarity class | S31 | num | representation–monitoring rarity gap |
| S32 | cat | rarity relation | S33 | cat | distinctive dimension |
| S41 | num | local structure strength | S42 | cat | local structure class |
| S43 | num | local structure concentration | | | |

### X: refusal boundary (14)
Questions that require a designated refusal.

| Tasks | Category |
|---|---|
| X11–X15 | unsupported inference (cause, forecast, safety, …) |
| X21–X26 | unavailable data or scale (exact assets, sub-cell points, other components, external context, live status, open rankings) |
| X31–X33 | representation boundary |

Each task-group directory contains its algorithm note, such as
[`tasks/d2/d2_algorithm.md`](tasks/d2/d2_algorithm.md). Refer to these notes for
task-specific inputs, reference-value definitions, and validity conditions.

## License and scope

Task targets describe observed displacement and representation properties.
Refusal tasks define the boundary of supported questions. Data provenance,
licensing, and application limits are documented in the
[Dataset card](https://huggingface.co/datasets/risenyard/egms-qa-dataset#provenance-terms-and-limitations)
and [DATA_LICENSE](../../../DATA_LICENSE).
