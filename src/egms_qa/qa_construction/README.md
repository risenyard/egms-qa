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

Each workflow starts at a different stage. The checks indicate the released
data stages taken as already prepared; all three workflows generate new QA.

| Goal | Released tiles | Released task results | Released QA records | Where to start |
|---|---|---|---|---|
| Reproduce QA generation | ✓ | ✓ | — | [From released task results](#reproduce-qa-generation) |
| Reproduce task computation | ✓ | — | — | [From released tiles](#reproduce-task-computation) |
| Construct QA for new tiles | — | — | — | [From new tiles](#construct-qa-for-new-tiles) |

The new-tile workflow uses your own source tiles. To use the published QA
without rebuilding it, follow the [Dataset loading example](https://huggingface.co/datasets/risenyard/egms-qa-dataset#qa-use).

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


## Reproduce QA generation

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

QA files appear under `outputs/qa-reproduced/qa/`: two different question phrasings for training dataset
(`v1_train_e00.jsonl` and `v1_train_e01.jsonl`), plus `v1_val.jsonl` and
`v1_test.jsonl`. `meta.json` and `task_counts.csv` record generation settings
and output counts.

Set `--train-cycles` to change the number of question phrasings for training dataset. For a small run,
add `--max-tiles 2 --train-cycles 1` to the generation command. The Dataset
contains fixed published splits; these commands generate a new corpus from
the released targets and approved phrasings.

## Reproduce task computation

Complete [setup](#installation-and-data-setup), then follow the
[task setup and input requirements](tasks/README.md#setup) to install task
dependencies and any required tiles, tokens, or encoder files.

Compute task results from the released tiles and representations using the
[task commands](tasks/README.md#run-and-files). Write each group's results to
`outputs/tasks-working/<group>/`, and assemble all 27 final tables there,
including the X refusal catalogs.

Follow task dependency order and pass newly computed upstream tables through
the documented input flags. For example, C5 must read the new B2 table.
Default input paths otherwise read the installed reference tables.

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

## Construct QA for new tiles

Start with your prepared NPZ tiles and a manifest containing `tile_id`, `split`,
and `path`. Match the [source-tile format](https://huggingface.co/datasets/risenyard/egms-qa-dataset#source-tile-contract).
For tasks that use representations, extract tokens for these tiles using the
[Encoder local-input workflow](../../egms_encoder/README.md#use-local-inputs).

Compute targets from the new tiles using the [task methods and commands](tasks/README.md#run-and-files).
Supply your manifest, tokens, and computed upstream tables through each task's
input options. Preserve the documented units, categorical labels, and
reference populations when applying the existing task definitions.

Assemble the resulting targets into `my_data/labels.parquet` following the
[Dataset label contract](https://huggingface.co/datasets/risenyard/egms-qa-dataset#labels-and-task-metadata).
The release label builder requires the fixed 10,000-tile split; another
collection needs its own label-table assembly. With
[setup](#installation-and-data-setup) complete, generate QA from that table:

```bash
python -m egms_qa.qa_construction.generate_qa \
    --labels my_data/labels.parquet \
    --out-dir outputs/qa-custom
```

The generator reuses the installed task definitions and approved phrasings and
writes QA under `outputs/qa-custom/qa/`. Unavailable targets receive missing-data
answers. Adapting task rules or reference populations requires corresponding
changes to the task and answer definitions.

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
