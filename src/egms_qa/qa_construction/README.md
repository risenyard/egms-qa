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

Follow the shared [installation instructions](../../../README.md#installation)
to clone the repository, create a Python environment, and install the core
package. Label aggregation and QA generation run on CPU. For task reference
computation, also install the `tasks` extra as shown there. The
[environment guide](../../../docs/environment.md) lists verified versions and
CPU/CUDA checks. With the environment activated, download the QA artifacts:

```bash
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
    --validate-release --out-dir outputs/labels-reproduced
python -m egms_qa.qa_construction.generate_qa \
    --labels outputs/labels-reproduced/labels.parquet \
    --meta outputs/labels-reproduced/labels_meta.json \
    --out-dir outputs/qa-reproduced
```

The label builder writes `labels.parquet` and `labels_meta.json`.
`--validate-release` checks the fixed 8,000/1,000/1,000 tile split. To also
check token alignment, supply `--encoder-cache` with a token-cache path.

QA files appear under `outputs/qa-reproduced/qa/`: two different question phrasings for training dataset
(`v1_train_e00.jsonl` and `v1_train_e01.jsonl`), plus `v1_val.jsonl` and
`v1_test.jsonl`. `meta.json` and `task_counts.csv` record generation settings
and output counts.

Set `--train-cycles` to change the number of question phrasings for training dataset. For a small run,
add `--max-tiles 2 --train-cycles 1` to the generation command. The Dataset
contains fixed published splits; these commands generate a new corpus from
the released targets and approved phrasings.

## Reproduce task computation

Complete the [task-system setup](tasks/README.md#setup), then compute all task
groups in dependency order:

```bash
python -m egms_qa.qa_construction.run_tasks \
    --out-dir outputs/tasks-rebuilt
```

The runner writes a complete set of task tables and automatically connects
upstream results to downstream computations. See the
[task-system guide](tasks/README.md#run-the-task-system) for execution options
and output checks. To turn these results into labels and QA:

```bash
python -m egms_qa.qa_construction.build_labels \
    --tasks-root outputs/tasks-rebuilt \
    --validate-release --out-dir outputs/labels-recomputed
python -m egms_qa.qa_construction.generate_qa \
    --labels outputs/labels-recomputed/labels.parquet \
    --meta outputs/labels-recomputed/labels_meta.json \
    --tasks-root outputs/tasks-rebuilt \
    --out-dir outputs/qa-recomputed
```

## Construct QA for new tiles

Complete the [task-system setup](tasks/README.md#setup).
Prepare NPZ tiles following the [tile contract](https://huggingface.co/datasets/risenyard/egms-qa-dataset#source-tile-contract)
and a manifest with `tile_id`, `split`, and `path`. Each tile needs displacement
histories, coordinates, and the static fields required by the
[tile reader](../../egms_encoder/data/tile_store.py). Missing point counts
and centroids are computed automatically.

```bash
python -m egms_qa.qa_construction.run_tasks \
    --mode new-tiles --manifest my_data/split.parquet \
    --out-dir outputs/tasks-new
```

The task tables are written to `outputs/tasks-new/<group>/<group>_final_table.csv`.

Tokens are extracted automatically with the released encoder. To reuse an
existing cache, pass `--token-cache`. The cache must match the manifest,
encoder, normalization, and data configuration.

Use `--source-tiles-root` to change the tile directory or `--data-config` to
supply a data configuration. Tiles must still contain 294 steps of vertical
displacement.

Manifest paths may be absolute or relative to `--source-tiles-root`. For
example, `tile_01.npz` with `--source-tiles-root my_tiles` reads
`my_tiles/tile_01.npz`.

New tiles use the release's training reference for classification and
representation scores. These settings stay fixed. A train split is not
required, and the collection can have a different number of tiles.
Reference settings are saved in `reference_state.joblib`, with input hashes
and run details in `run.json`.

Merge the task tables into labels, then generate QA:

```bash
python -m egms_qa.qa_construction.build_labels \
    --tasks-root outputs/tasks-new \
    --out-dir outputs/labels-new
python -m egms_qa.qa_construction.generate_qa \
    --labels outputs/labels-new/labels.parquet \
    --meta outputs/labels-new/labels_meta.json \
    --tasks-root outputs/tasks-new \
    --out-dir outputs/qa-custom
```

The builder aligns all task tables by `tile_id` and `split` and supports any
number of tiles. It writes `labels.parquet` and `labels_meta.json`; the
[Dataset label contract](https://huggingface.co/datasets/risenyard/egms-qa-dataset#labels-and-task-metadata)
describes their fields.

QA appears under `outputs/qa-custom/qa/`. The generator uses the new label
metadata and the published approved phrasings. Missing targets receive missing-data answers.

## Code reference

| module | purpose |
|---|---|
| [run_tasks.py](run_tasks.py) | run the complete task system with aligned inputs |
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
