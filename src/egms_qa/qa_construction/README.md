# EGMS-QA Construction

QA construction derives task reference values from ground-motion measurements
and encoder representations, assembles them into tile-level labels, and renders
natural-language questions and reference answers. The task catalog covers
observation quality, motion, spatial and temporal properties, representation
properties, and refusal boundaries.

## Workflows

This guide covers QA construction and reproduction for
[EGMS-QA](../../../README.md). The
[Hugging Face Dataset card](https://huggingface.co/datasets/risenyard/egms-qa-dataset)
documents the released files, data formats, and splits. The
[task index](tasks/README.md) provides task definitions, methods, and computation
commands.

The middle columns show which released artifacts each workflow reuses.
“Partial” means that task groups not being recomputed retain their released
results.

| Goal | Released tiles | Released task results | Released QA records | Where to start |
|---|---|---|---|---|
| Use published QA | — | — | ✓ | [Read QA records](https://huggingface.co/datasets/risenyard/egms-qa-dataset#qa-use) |
| Reproduce QA | — | ✓ | — | [Generate QA from released task results](#generate-qa-from-released-task-results) |
| Reproduce task results and QA | As needed | Partial | — | [Recompute task results and generate QA](#recompute-task-results-and-generate-qa) |
| Generate QA for a new collection | — | — | — | [Generate QA from your own labels](#generate-qa-from-your-own-labels) |

The last workflow takes your prepared labels as input. Task-specific token
and metadata requirements are listed in the [task index](tasks/README.md#setup).

## Installation and data setup

Python 3.10 or later is required. Label aggregation and QA generation run on
CPU. Clone and install the code, then download the QA artifacts:

```bash
git clone https://github.com/risenyard/egms-qa
cd egms-qa
pip install -e .
python -m egms_qa.release install --download --components qa
```

Run all commands below from the repository root. Setup installs reference
tables under `outputs/tasks/` and published labels, metadata, and QA under
`outputs/qa/`. Use separate output directories for new results.

To read the published QA directly, use the
[Dataset loading example](https://huggingface.co/datasets/risenyard/egms-qa-dataset#qa-use).

## Generate QA from released task results

After completing [setup](#installation-and-data-setup), rebuild labels from
the released reference tables and pass them to the QA generator:

```bash
python -m egms_qa.qa_construction.build_labels \
    --skip-cache-validation --out-dir outputs/labels-reproduced
python -m egms_qa.qa_construction.generate_qa \
    --labels outputs/labels-reproduced/labels.parquet \
    --meta outputs/labels-reproduced/labels_meta.json \
    --out-dir outputs/qa-reproduced
```

The label builder writes `labels.parquet` and `labels_meta.json`.
`--skip-cache-validation` omits the optional alignment check against encoder
tokens; reference-table validation still runs.

QA files appear under `outputs/qa-reproduced/qa/`: two training phrasing cycles
(`v1_train_e00.jsonl` and `v1_train_e01.jsonl`), plus `v1_val.jsonl` and
`v1_test.jsonl`. `meta.json` and `task_counts.csv` record generation settings
and output counts.

Set `--train-cycles` to change the number of training cycles. For a small run,
add `--max-tiles 2 --train-cycles 1` to the generation command. The Dataset
contains fixed published splits; these commands generate a new corpus from
the released targets and approved phrasings.

## Recompute task results and generate QA

Complete [setup](#installation-and-data-setup), then follow the
[task setup and input requirements](tasks/README.md#setup) to install task
dependencies and any required tiles, tokens, or encoder files.

Start with a new working copy of the reference tables:

```bash
cp -rL outputs/tasks outputs/tasks-working
```

Run the selected [task commands](tasks/README.md#run-and-files), directing each
output to `outputs/tasks-working/<group>/`. Recompute affected downstream
tasks as well, passing the new tables through their documented input flags.
For example, updating B2 requires updating C5 with the new B2 table. Keep the
complete set of 27 final tables in the working directory.

Rebuild labels and QA from that working set:

```bash
python -m egms_qa.qa_construction.build_labels \
    --tasks-root outputs/tasks-working \
    --skip-cache-validation --out-dir outputs/labels-recomputed
python -m egms_qa.qa_construction.generate_qa \
    --labels outputs/labels-recomputed/labels.parquet \
    --meta outputs/labels-recomputed/labels_meta.json \
    --tasks-root outputs/tasks-working \
    --out-dir outputs/qa-recomputed
```

The output layout matches the QA generation workflow above. Compare recomputed
values and class counts with the installed references before using the new QA.
Task-specific fitting requirements and reproduction limits are documented in
the [task methods and reconstruction scope](tasks/README.md#reconstruction-scope).

## Generate QA from your own labels

With [setup](#installation-and-data-setup) complete, provide a Parquet label
table following the [Dataset label contract](https://huggingface.co/datasets/risenyard/egms-qa-dataset#labels-and-task-metadata).
Use the existing task definitions, units, and categorical labels. Unavailable
targets receive missing-data answers. The
[task methods](tasks/README.md#task-groups) describe how to compute each target.

```bash
python -m egms_qa.qa_construction.generate_qa \
    --labels my_data/labels.parquet \
    --out-dir outputs/qa-custom
```

The generator reuses the installed task metadata and approved phrasings and
writes QA under `outputs/qa-custom/qa/`. Supply `--meta` and `--tasks-root`
when using a separate task-metadata file or refusal catalogs.

The release label builder requires the fixed 10,000-tile split. For another
collection, prepare the label table directly and use the generation command
above. Adding tasks or changing label rules also requires adapting their
question and answer definitions.

## Code reference

| module | purpose |
|---|---|
| [tasks/](tasks/README.md) | task definitions, algorithms, and computation commands |
| [build_labels.py](build_labels.py) | aggregate task reference tables into labels and metadata |
| [qa_lib.py](qa_lib.py) | question phrasings, answer rendering, and validation |
| [generate_qa.py](generate_qa.py) | generate QA splits and record output counts |

## Scope

The workflows use prepared task inputs and the EGMS-QA task catalog. Official
EGMS acquisition and preparation of a new tile collection require a separate
workflow. The [Dataset card](https://huggingface.co/datasets/risenyard/egms-qa-dataset#provenance-terms-and-limitations)
documents data provenance, licensing, and application limits.

For model training with the resulting QA and corresponding tile tokens, follow
the [Translator guide](../translator/README.md#reproduce-training).
