# EGMS-QA Construction

EGMS-QA constructs question–answer records by computing task reference values
from ground-motion measurements and encoder representations, assembling those
values into labels, and rendering questions and reference answers. This guide
explains the construction process and provides three workflows for reproducing
the released QA, recomputing tasks, and applying the task catalog to new data.

For ready-to-use records, use the
[HF Dataset loading example](https://huggingface.co/datasets/risenyard/egms-qa-dataset#qa-use).
The Dataset card documents the released files, formats, splits, and data terms.

## Construction process

The [task catalog](tasks/README.md#task-groups) defines 78 tasks in 27 groups.
A/B/C/D/S contribute 64 tile-dependent targets covering observation quality,
motion, spatial organization, temporal dynamics, and representation properties.
The 14 X tasks define refusal boundaries and are stored as task catalogs rather
than tile-dependent label columns.

For example, [B21](tasks/b2/b2_algorithm.md) takes the mean of a tile's point
velocities. Its reference-table column, `B21_mean_velocity_mm_yr`, becomes the
label column `B21`. One question phrasing is “What is the average vertical
ground velocity?” An illustrative value of `-1.6` yields the reference answer
“The mean vertical ground velocity is -1.60 mm/yr.” The question phrasing can
vary while the target remains fixed. B22 separately assigns a mean-motion
class using its documented direction rule and corpus-relative cutoffs.

Answers are rendered from task values and predefined templates. Missing
targets produce a missing-data answer; X tasks produce a task-specific refusal.
The [QA record schema](https://huggingface.co/datasets/risenyard/egms-qa-dataset#qa-records)
describes how each record retains its tile, task, target value, and phrasing.

## Workflows

| Workflow | Starting point | Result |
|---|---|---|
| [Reproduce QA construction](#reproduce-qa-construction) | Released task reference tables | Rebuilt labels and newly generated QA splits |
| [Recompute task labels and regenerate QA](#recompute-task-labels-and-regenerate-qa) | Prepared measurements or representations and task dependencies | Recomputed task tables, updated labels, and QA |
| [Construct QA from your own data](#construct-qa-from-your-own-data) | Your prepared data and targets following the existing task definitions | QA records for your tile IDs and splits |

## Installation and data setup

Use Python 3.10 or later. Run all commands from the cloned repository root:

```bash
git clone https://github.com/risenyard/egms-qa
cd egms-qa
pip install -e .
python -m egms_qa.release install --download --components qa
```

The `qa` group installs the published QA, labels, reference tables, and shared
metadata. The installer creates `outputs/tasks/` for reference tables and
`outputs/qa/` for released labels, label metadata, the phrasing audit, and QA.
See the [Dataset file layout](https://huggingface.co/datasets/risenyard/egms-qa-dataset#release-layers)
for the corresponding published files.

Label aggregation and QA generation run on CPU. Task computation may require
additional dependencies and data; the second workflow specifies these for B2.
Other tasks list their requirements in the [task index](tasks/README.md).
Use a new output directory for each run. The examples below keep generated
results separate from the installed reference files.

## Reproduce QA construction

Use this workflow to reconstruct labels from all 27 released reference tables
and generate a QA corpus with the existing task definitions and approved
question phrasings. Complete [installation and data setup](#installation-and-data-setup)
first. Source tiles and encoder inference are not required for these steps.

### 1. Assemble task labels

```bash
python -m egms_qa.qa_construction.build_labels \
    --tasks-root outputs/tasks \
    --skip-cache-validation \
    --out-dir outputs/labels-reproduced
```

This writes `labels.parquet` and `labels_meta.json`. Aggregation aligns the
tables by tile and split, checks their required columns, and enforces the
released 8,000/1,000/1,000 tile split. The metadata maps each target to its task
and source column and includes the X task definitions.

`--skip-cache-validation` omits the optional check against encoder token IDs
and ordering; reference-table checks still run. To include that check, install
`--components tokens` with the release installer and omit the flag.

### 2. Generate questions and reference answers

```bash
python -m egms_qa.qa_construction.generate_qa \
    --labels outputs/labels-reproduced/labels.parquet \
    --meta outputs/labels-reproduced/labels_meta.json \
    --tasks-root outputs/tasks \
    --qa-audit-manifest outputs/qa/qa_audit.json \
    --out-dir outputs/qa-reproduced
```

The generator writes the following under `outputs/qa-reproduced/`:

| Output | Contents |
|---|---|
| `qa/v1_train_e00.jsonl`, `qa/v1_train_e01.jsonl` | Two training cycles with rotated question phrasings |
| `qa/v1_val.jsonl`, `qa/v1_test.jsonl` | Validation and test QA |
| `task_counts.csv` | Record counts by output file and task |
| `meta.json` | Tile counts, generation settings, and output record counts |

Each cycle renders one approved phrasing per tile-dependent task and tile.
The default is two training cycles; `--train-cycles` changes this number.
X tasks are sampled separately, with default caps of 3,000 training and 300
validation/test tiles per refusal task. With the complete released tables,
each training file contains 554,000 records, and validation and test each
contain 68,200 records.

For an initial run check, add `--max-tiles 2 --train-cycles 1` and use a
different output directory. This caps each split at two tiles and produces
156 records per file; it does not reproduce the full corpus.

### 3. Check the result

Follow [output verification](#output-verification) below. Compare rebuilt
labels with the released label table by `tile_id` and `split`, rather than row
position. Check target values and missingness before interpreting QA differences.

The HF Dataset provides fixed published QA splits. These commands render new
files; additional cycles and generation settings can change phrasing, sampling,
and record order. Numerical agreement of labels alone does not establish
byte-for-byte equality with the published QA.

## Recompute task labels and regenerate QA

Use this workflow to compute task reference values again and propagate them
into QA. The example recomputes B2 from the released source tiles, then C5,
which reads the B22 class. Other reference tables remain inputs to this run.

### 1. Prepare inputs and a working set of reference tables

Complete [installation and data setup](#installation-and-data-setup), then:

```bash
pip install -e '.[tasks]'
python -m egms_qa.release install --download --components tiles
python - <<'PY'
from pathlib import Path
import shutil

source = Path("outputs/tasks")
target = Path("outputs/tasks-working")
target.mkdir(parents=True, exist_ok=False)
for table in sorted(source.glob("*/*_final_table.csv")):
    destination = target / table.relative_to(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(table, destination)
PY
```

The working set contains ordinary copies of the final tables. B2 reads the
installed split manifest and NPZ `mean_velocity` arrays. Its method and input
requirements are documented in the [B2 task page](tasks/b2/b2_algorithm.md).

### 2. Recompute B2 and its dependent C5 table

```bash
python -m egms_qa.qa_construction.tasks.b2.b2_compute \
    --manifest data/encoder/manifest/split.parquet \
    --workers 4 \
    --out-dir outputs/tasks-working/b2
python -m egms_qa.qa_construction.tasks.c5.c5_compute \
    --b2 outputs/tasks-working/b2/b2_final_table.csv \
    --b3 outputs/tasks-working/b3/b3_final_table.csv \
    --b6 outputs/tasks-working/b6/b6_final_table.csv \
    --c3 outputs/tasks-working/c3/c3_final_table.csv \
    --out-dir outputs/tasks-working/c5
```

This replaces B2 and C5 in the working set. The C5 command explicitly consumes
the new B2 result. For another task, follow the dependencies in the
[task index](tasks/README.md#task-groups) and recompute affected downstream
tables in dependency order. Pass each new input using the flags documented by
that task; default paths otherwise continue to read the installed references.

Retain each method's reference population and fitted parameters. A small sample
can check execution, but cannot reproduce thresholds fitted on the complete
training pool. B22 uses fixed European corpus-relative cutoffs, which are not
physical severity thresholds. See the
[reconstruction scope](tasks/README.md#reconstruction-scope) for task-specific
limits on reproducing reference values.

### 3. Rebuild labels and QA from the working tables

```bash
python -m egms_qa.qa_construction.build_labels \
    --tasks-root outputs/tasks-working \
    --skip-cache-validation \
    --out-dir outputs/labels-recomputed
python -m egms_qa.qa_construction.generate_qa \
    --labels outputs/labels-recomputed/labels.parquet \
    --meta outputs/labels-recomputed/labels_meta.json \
    --tasks-root outputs/tasks-working \
    --qa-audit-manifest outputs/qa/qa_audit.json \
    --out-dir outputs/qa-recomputed
```

Check the recomputed B2 and C5 tables against their installed references by
tile and split, including numeric differences, class counts, and missing
targets. Confirm that the new label columns match the working tables, then
apply [output verification](#output-verification) to `outputs/qa-recomputed/`.
This workflow uses the released tile population; its label aggregation step
retains the fixed release split requirement.

## Construct QA from your own data

Use this workflow to render the existing task catalog for a new tile collection.
Complete [installation and data setup](#installation-and-data-setup) to obtain
the task metadata, X catalogs, and approved phrasings.

### 1. Compute targets under the task definitions

For each applicable task, use the [task method](tasks/README.md#task-groups) to
identify the required measurements, representations, upstream tables, units,
and label rules. Prepared tiles must satisfy the
[source-tile contract](https://huggingface.co/datasets/risenyard/egms-qa-dataset#source-tile-contract);
representation tasks also require the
[encoder input and extraction workflow](../../egms_encoder/README.md#input-requirements).
Official-product acquisition and conversion into prepared tiles are separate
preparation steps.

Preserve the meanings of existing targets. Corpus-relative classes retain the
reference population described by the method when reusing the released rules.
Fitting new cutoffs or adding tasks requires a separate method and template
adaptation; passing a different label file does not perform that adaptation.

### 2. Prepare a label table

Provide one row per unique `tile_id`, a `split` value of `train`, `val`, or
`test`, and the 64 tile-dependent task columns listed in the
[label metadata](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/artifacts/labels/metadata.json).
Store numeric targets in their documented units and categorical targets using
the exact method-defined labels. Represent unavailable targets as null values.
The generator renders missing-data answers for these targets; it does not
omit their QA records. X tasks use the installed refusal catalogs.

The following runnable example illustrates the label interface with three
synthetic tiles and a numeric B21 target. Replace the illustrative rows with
your computed targets for actual use:

```bash
python - <<'PY'
import json
from pathlib import Path
import pandas as pd

meta = json.loads(Path("outputs/qa/labels_meta.json").read_text())
task_ids = [t["id"] for t in meta["tasks"] if t["probe_applicable"]]
rows = []
for tile_id, split, velocity in [
    ("example-train", "train", -1.6),
    ("example-val", "val", 0.2),
    ("example-test", "test", None),
]:
    row = dict.fromkeys(task_ids)
    row.update(tile_id=tile_id, split=split, B21=velocity)
    rows.append(row)
labels = pd.DataFrame(rows, columns=["tile_id", "split", *task_ids])
assert labels["tile_id"].is_unique
assert set(labels["split"]) <= {"train", "val", "test"}
Path("my_data").mkdir(exist_ok=True)
labels.to_parquet("my_data/labels.parquet", index=False)
PY
```

The release label builder enforces 10,000 tiles with the fixed release split.
For another collection size, assemble the label table as above and pass it
directly to the generator.

### 3. Generate and verify QA

```bash
python -m egms_qa.qa_construction.generate_qa \
    --labels my_data/labels.parquet \
    --meta outputs/qa/labels_meta.json \
    --tasks-root outputs/tasks \
    --qa-audit-manifest outputs/qa/qa_audit.json \
    --train-cycles 1 \
    --out-dir outputs/qa-custom
```

The example produces 78 records in each split: 64 tile-dependent tasks and
14 refusal tasks. B21 has numeric answers in train and validation and a
missing-data answer in test. The other tile-dependent targets are missing in
this interface example. Apply [output verification](#output-verification)
with `root = Path("outputs/qa-custom")`, and check that every record retains
your intended tile ID and split.

## Output verification

Each generation run writes `meta.json` and `task_counts.csv` beside its `qa/`
directory. This check compares the recorded counts with the actual JSONL files
and shows one record from each split. Set `root` to the output directory of
the workflow you ran:

```bash
python - <<'PY'
import json
from pathlib import Path
import pandas as pd

root = Path("outputs/qa-reproduced")
meta = json.loads((root / "meta.json").read_text())
counts = pd.read_csv(root / "task_counts.csv")
print("Tiles:", meta["n_tiles"])
for name, expected in meta["row_counts"].items():
    with (root / "qa" / name).open() as stream:
        first = json.loads(next(stream)) if expected else None
        actual = (1 if first is not None else 0) + sum(1 for _ in stream)
    assert actual == expected
    assert int(counts.loc[counts["file"] == name, "rows"].sum()) == actual
    print(name, actual, first)
PY
```

Counts confirm output completeness, not target correctness. Inspect numeric,
categorical, missing, and refusal records against the input labels and task
definitions. Check that tile IDs remain in their assigned split and that
updated task results reach the corresponding `answer_value` fields.

For model training, pair QA tile IDs with the corresponding encoder tokens and
follow the [Translator training guide](../translator/README.md#reproduce-training).
Data provenance, licensing, and application limits are documented in the
[HF Dataset card](https://huggingface.co/datasets/risenyard/egms-qa-dataset#provenance-terms-and-limitations).
