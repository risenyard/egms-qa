# EGMS-QA coordinated release contract

## Release surfaces

| Surface | Canonical repository | Required role |
|---|---|---|
| Code | `https://github.com/risenyard/egms-qa` | Python source, tests, documentation, packaging |
| Dataset | `risenyard/egms-qa-dataset` (`dataset`) | QA records, 10k NPZ tiles, tokens, tables, manifests |
| Encoder | `risenyard/egms-qa-encoder` (`model`) | Encoder 4.3 weights, configuration, normalization |
| Translator | `risenyard/egms-qa-translator` (`model`) | Qwen, Gemma, Llama, and Mistral projectors/adapters |

Hugging Face is the only source of released data, manifests, configuration,
checkpoint metadata, and heavy model artifacts. GitHub must not contain or
trust a tracked `data/` tree.

## Candidate invariants

Require all of the following unless a later release intentionally changes the
public contract and the user approves that change:

- Python distribution: `egms-qa`; public modules: `egms_encoder`, `egms_qa`.
- Encoder release: `4.3`; input length: `294`; embedding width: `256`.
- Dataset: exactly 10,000 tiles with split counts 8,000/1,000/1,000.
- Token cache: `egms_tokens_10k.pt` plus matching metadata; tensor shape
  `[10000, 65, 256]`; token 0 is the tile summary and tokens 1–64 are the 8×8
  spatial cells.
- Dataset metadata schema: `egms-qa-release-v1`; integrity algorithm: SHA256.
- Dataset contains `metadata/release_manifest.json`, `metadata/files.sha256`,
  `metadata/data_config.json`, `metadata/split_manifest.parquet`, and
  `metadata/tile_manifest.parquet`.
- Encoder contains `README.md`, `encoder.safetensors`, `config.json`,
  `normalization.json`, `training_args.json`, and `eval_results.json`.
  Legacy `encoder.pt` and `args.json` files are not part of the release.
- Translator contains `README.md`, `manifest.json`, and complete `qwen`,
  `gemma`, `llama`, and `mistral` directories with `projector.safetensors`,
  `translator_config.json`, training/evaluation metadata, and `adapter/`.
- Cards, package metadata, token metadata, manifests, and release notes use the
  same public names and version. No old private repository names or local
  absolute paths may appear in published material.

If the dataset stores the cropped 294-step series directly, ensure its data
configuration documents both stored `[0,294)` and original `[8,302)` indexing.
Update token provenance when the data configuration or split manifest changes,
even if the resulting token tensor remains bitwise identical.

## Readiness gates

### A. Repository and provenance

- Fetch remote refs, then inspect ahead/behind counts and competing branches.
- Require a clean candidate commit and an unchanged commit throughout testing.
- Require no tracked `data/`, `outputs/`, caches, secrets, broken symlinks, or
  machine-specific paths.
- Check package version and any present `CHANGELOG.md`, `CITATION.cff`, cards,
  badges, and links for consistency.
- Verify license, data terms, Copernicus provenance, modification notice, and
  non-endorsement wording.

### B. Static artifact checks

- Run `scripts/preflight.py --mode publish` against the exact staging trees.
- Pass `--dataset-dir`, `--encoder-dir`, or `--translator-dir` when the release
  candidate is not the canonical `<hf-root>/<component>` directory; never let
  directory naming silently select an older staging tree.
- Verify all staging symlink targets exist before upload.
- Rebuild `metadata/files.sha256` and `metadata/release_manifest.json` only from
  the final staging state; never edit content afterward without rebuilding.
- Confirm token metadata SHA256 values match the exact checkpoint,
  configuration, normalization, manifest, and data configuration used.
- Safe-load PyTorch files with `weights_only=True` where supported and reject
  missing/unexpected model keys.

### C. GPU candidate tests

Submit all tests to `itc-gpu` or `main-gpu` and retain logs:

1. Complete `pytest` suite.
2. Wheel build, clean-environment installation, and import of every public
   module.
3. Full dataset SHA256 audit.
4. One-tile extraction with expected shape `[1,65,256]`.
5. Complete 10k extraction with expected shape `[10000,65,256]` and numerical
   or bitwise regression against the intended cache.
6. Encoder pretraining smoke: train, save, strict reload, resume.
7. Release installation into an empty runtime directory.
8. At least one QA downstream consumer over all 10k tokens.
9. Safe loading of all four translator projector/adapter families.

Do not reuse an older log if its tested Git SHA or any artifact hash differs
from the current candidate.

### D. Remote staging and black-box verification

Remote writes require explicit authorization. Use `hf auth whoami` and
`gh auth status` without printing tokens. Use `hf upload` and record every
returned revision. Avoid unreviewed deletion patterns.

After upload, create a new temporary directory and use only public interfaces:

1. Clone the exact remote Git SHA or candidate ref.
2. Download each HF repository pinned to its captured revision.
3. Audit/install the dataset, install the built package, and read the cached
   tokens.
4. Submit one-tile extraction and downstream smoke to an allowed GPU partition.
5. Compare downloaded file hashes with the release ledger.

An upload that cannot be downloaded and consumed this way is not ready for a
GitHub Release.

## Publication transaction

1. Confirm the proposed semantic version and tag, normally `vX.Y.Z`.
2. Confirm the tested commit equals the remote `main` commit.
3. Confirm GitHub CI succeeded for that exact commit.
4. Tag all three HF repositories at their tested immutable revisions.
5. Create an annotated Git tag; never replace an existing remote tag.
6. Push the tag and create the GitHub Release with `gh release create
   --verify-tag --notes-file <file>`.
7. Attach only artifacts built from the tagged commit. Dataset/model weights
   remain on HF unless the release plan explicitly names a GitHub asset.
8. Record GitHub URL, Git SHA, Git tag, three HF revisions/tags, wheel SHA256,
   dataset manifest SHA256, test job IDs, and public-smoke result.

Treat changing visibility, deleting files/branches/tags/releases, overwriting
assets, force-pushing, or rolling back a public HF revision as a separate
destructive action requiring explicit approval.

## Stop conditions

Stop and report instead of publishing when any of these is true:

- dirty or divergent candidate state is unresolved;
- a requested tag already exists;
- test jobs are incomplete, failed, stale, or ran on a login node/CPU fallback;
- checksums, versions, names, token shapes, or cross-repository hashes disagree;
- staging contains broken links, secrets, machine-specific public metadata, or
  missing artifacts;
- public downloads are unavailable or do not reproduce the tested behavior;
- GitHub/HF credentials lack the required scope;
- the user has not explicitly authorized the remote-writing phase.
