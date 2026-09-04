# QA construction (task definitions and question–answer records)

> 🤗 Released dataset (train/val/test + labels + tables): [`risenyard/egms-qa-dataset`](https://huggingface.co/datasets/risenyard/egms-qa-dataset)

This block turns the frozen tile representation and the raw tile records into
the EGMS-QA question–answer dataset. It has three stages:

1. **Task reference values** — `tasks/<family>/<task>_compute.py` computes the
   deterministic reference value for each task family over the 10,000 tiles and
   writes a per-family table under `outputs/tasks/`. Each family carries its
   exact algorithm in `tasks/<family>/<task>_algorithm.md`.
2. **Label aggregation** — `build_labels.py` reads the per-family tables
   and materializes one canonical target per leaf task into
   `outputs/qa/labels.parquet` (+ `labels_meta.json`).
3. **QA rendering** — `generate_qa.py` (with `qa_lib.py`, `verifier` logic in
   `qa_lib`) renders question–answer JSONL for train/val/test. For each
   tile–task pair one phrasing is drawn from a 20-phrase pool per cycle;
   static X refusal tasks are sampled so they do not dominate the loss.

```bash
python -m egms_qa.qa_construction.build_labels          # -> outputs/qa/labels.parquet
python -m egms_qa.qa_construction.generate_qa --out-dir outputs/qa
```

## Task system (78 tasks, six families)

Tasks are grouped into six families forming a monitoring workflow. Families A–D
and S are token-dependent (64 tasks) and scored against reference values; family
X (14 tasks) fixes the boundary of the supported scope with designated refusal
answers. Each task is numeric or categorical; a renderer converts the reference
value into the natural-language answer used for supervision.

### A — observation gate (10)
*Does the record and its representation support monitoring at all?*

| Task | Type | Description | Task | Type | Description |
|---|---|---|---|---|---|
| A11 | num | global representation drift | A12 | cat | representation stability class |
| A21 | num | masked reconstruction loss | A22 | cat | reconstruction reliability class |
| A31 | num | spatial observation coverage | A32 | cat | spatial coverage class |
| A41 | num | median measurement noise | A42 | cat | measurement noise class |
| A51 | cat | monitoring usability gate | A52 | cat | monitoring usability reason |

### B — motion vital signs (14)
*Strength and character of the observed motion.*

| Task | Type | Description | Task | Type | Description |
|---|---|---|---|---|---|
| B11 | num | average subsidence SNR | B12 | cat | clear subsidence signal |
| B21 | num | mean velocity | B22 | cat | mean subsidence intensity band |
| B31 | num | sinking-tail velocity | B32 | num | upper-tail velocity |
| B33 | num | absolute tail velocity | B34 | cat | uplift-protected direction |
| B35 | cat | worst-point significance | B36 | cat | European velocity typicality |
| B41 | num | acceleration strength | B42 | cat | European acceleration typicality |
| B51 | num | seasonality strength | B61 | cat | monitoring trigger |

### C — spatial organization (12)
*Where motion concentrates within the tile.*

| Task | Type | Description | Task | Type | Description |
|---|---|---|---|---|---|
| C11 | num | moving-point fraction | C12 | cat | motion extent class |
| C13 | cat | strongest-motion bin | C21 | num | spatial concentration |
| C22 | cat | concentration class | C31 | num | deformation-front strength |
| C32 | cat | front location | C33 | cat | front strength class |
| C41 | num | fast-tail bin fraction | C42 | cat | fast-tail extent class |
| C51 | cat | monitoring priority | C52 | cat | hidden local risk |

### D — temporal dynamics (15)
*How motion evolves over the observation period.*

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

### S — representation constructs (13)
*Properties of the frozen encoder representation itself.*

| Task | Type | Description | Task | Type | Description |
|---|---|---|---|---|---|
| S11 | cat | reference anchor profile | S12 | num | nearest anchor distance |
| S13 | num | anchor margin | S14 | cat | assignment status |
| S15 | cat | anchor profile description | S21 | num | local isolation score |
| S22 | cat | representation rarity class | S31 | num | representation–monitoring rarity gap |
| S32 | cat | rarity relation | S33 | cat | distinctive dimension |
| S41 | num | local structure strength | S42 | cat | local structure class |
| S43 | num | local structure concentration | | | |

### X — refusal boundary (14)
*Out-of-scope questions; the correct answer is a designated refusal.*

| Tasks | Category |
|---|---|
| X11–X15 | unsupported inference (cause, forecast, safety, …) |
| X21–X26 | unavailable data or scale (exact assets, sub-cell points, other components, external context, live status, open rankings) |
| X31–X33 | representation boundary |

For the exact reference-value algorithm of any task, see the matching
`tasks/<family>/<task>_algorithm.md`.

## Dataset (datasheet)

- **Unit**: a 7 km tile; 10,000 tiles across Europe, split 8,000 / 1,000 / 1,000
  (train / validation / test) by spatial unit.
- **Source**: EGMS Level-3 Ortho Vertical displacement (2019–2023). The 10,000
  model-ready source tiles are released as NPZ under
  `artifacts/source_tiles/` in `risenyard/egms-qa-dataset`. They store 294
  model steps and the encoder reads `[0,294)` directly. This corresponds to
  `[8,302)` on the original 304-step prepared axis. See the
  [HF provenance record](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/metadata/SOURCE_PROVENANCE.md).
- **Representation**: each tile is encoded to 65 tokens (1 summary + 8×8 spatial
  cells) of dimension 256 by the frozen EGMS encoder (see `egms_encoder`).
- **Records**: one question–answer pair per rendered tile–task–phrasing; answers
  are natural language only. The generated JSONL and the reference tables are
  part of the data release (see the top-level README for the download link).
- **Licence**: question–answer records and derived tables are CC-BY-4.0. The
  Copernicus-derived measurements retain the CLMS source and modification
  notices in `../../../DATA_LICENSE`.
