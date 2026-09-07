# EGMS-QA Construction

This module converts task reference values into labels and natural-language
question–answer records. The Dataset supplies the released reference tables,
labels, QA splits, and encoder tokens.

[Released Dataset](https://huggingface.co/datasets/risenyard/egms-qa-dataset) ·
[Project guide](../../../README.md) ·
[Translator guide](../translator/README.md)

## Use the released records

Run these commands from the repository root to download and install the Dataset:

```bash
pip install -e .
hf download risenyard/egms-qa-dataset --repo-type dataset \
    --local-dir release/egms-qa-dataset
python -m egms_qa.release install \
    --release-dir release/egms-qa-dataset --target-root .
```

The release provides QA under `data/qa/`, labels under `artifacts/labels/`,
and task-family tables under `artifacts/reference_tables/`. The installer links
these files to `outputs/qa/` and `outputs/tasks/` for use by the code.

## Generate question–answer records

With the Dataset installed, render QA from its labels and approved phrasings:

```bash
python -m egms_qa.qa_construction.generate_qa \
    --out-dir outputs/qa-generated
```

Generated JSONL files appear in `outputs/qa-generated/qa/`, with record counts
and rendering metadata in the parent directory. Each token-dependent
tile–task pair receives one phrasing per training cycle from a pool of 20.
Training cycles rotate the phrasing. Refusal tasks use a capped sample of
tiles per task.

Add `--max-tiles 2 --train-cycles 1` for a small rendering check. These options
reduce the generated records and do not reproduce the full release counts.

## Construction workflow

| step | implementation | output |
|---|---|---|
| Task reference values | family scripts and algorithm notes in `tasks/` | one reference table per task family |
| Label aggregation | `build_labels.py` | `labels.parquet` and `labels_meta.json` |
| Question and answer rendering | `generate_qa.py` and `qa_lib.py` | split JSONL files and rendering metadata |

The task-family notes define the inputs, formulas, thresholds, and target
columns used by the reference tables. Label aggregation joins the tables by
tile ID and split. Rendering converts the resulting targets into visible
natural-language answers using the approved question phrasings.

To rebuild labels from the released tables and render them into a new directory:

```bash
python -m egms_qa.qa_construction.build_labels \
    --out-dir outputs/labels-generated --skip-cache-validation
python -m egms_qa.qa_construction.generate_qa \
    --labels outputs/labels-generated/labels.parquet \
    --meta outputs/labels-generated/labels_meta.json \
    --out-dir outputs/qa-generated
```

This aggregation command checks table identities, joins, and split counts.
`--skip-cache-validation` omits the additional representation-cache order
checks, so label row order follows the reference tables. The released label
file remains the canonical input for reproducing model results.

## Dataset contract

| property | released value |
|---|---|
| source | EGMS Level-3 Ortho Vertical, 2019–2023 |
| spatial unit | overlapping 7 km tiles |
| tile count | 10,000 |
| tile split | 8,000 train, 1,000 validation, 1,000 test |
| stored displacement | `time_series [N,294]` in NPZ |
| tile representation | 65 tokens of width 256 and a validity mask |
| released QA records | 554,000 train, 68,200 validation, 68,200 test |

The split is fixed at tile level. Overlapping tiles can share measurement
points. Stored indices `[0,294)` correspond to source-preparation indices
`[8,302)` on a 304-step axis. The
[source provenance](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/SOURCE_PROVENANCE.md)
describes the source product and preparation contract.

## Task catalog

The full catalog contains 78 tasks in six groups. Groups A–D and S contain
64 token-dependent tasks with numeric or categorical targets. Group X contains
14 refusal tasks for questions outside the supported scope. Numeric targets
describe quantities, categorical targets assign classes, and refusal targets
state the evidence boundary.

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

Each task-family directory contains its algorithm note, such as
[`tasks/d2/d2_algorithm.md`](tasks/d2/d2_algorithm.md). Refer to these notes for
task-specific inputs, reference-value definitions, and validity conditions.

## License and scope

The QA records and derived reference tables are released under CC-BY-4.0.
Copernicus-derived measurements retain the source and modification requirements
in [DATA_LICENSE](../../../DATA_LICENSE). Task targets describe observed
displacement and representation properties. Refusal tasks define the limits on
causal, predictive, safety-related, and otherwise unsupported answers.
