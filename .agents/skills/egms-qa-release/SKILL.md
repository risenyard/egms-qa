---
name: egms-qa-release
description: Prepare, audit, publish, and verify coordinated EGMS-QA releases across the risenyard/egms-qa GitHub repository and the egms-qa-dataset, egms-qa-encoder, and egms-qa-translator Hugging Face repositories. Use when asked to check release readiness, prepare a release candidate, sync GitHub or HF, create a version tag or GitHub Release, verify public downloads, or report EGMS-QA release status. Enforce HF as the data/checkpoint source, GPU-only tests through itc-gpu or main-gpu, reproducible artifact checks, and explicit authorization before remote publication.
---

# EGMS-QA release

Treat the GitHub code repository and three Hugging Face repositories as one
coordinated release. Keep preparation reversible and keep publication gated.

## Start every run

1. Resolve the checkout with `git rev-parse --show-toplevel`. Refuse to operate
   on a different GitHub repository without explicit direction.
2. Read [references/release-contract.md](references/release-contract.md).
3. Run the read-only preflight:

   ```bash
   python .agents/skills/egms-qa-release/scripts/preflight.py \
     --repo-root "$(git rev-parse --show-toplevel)" \
     --hf-root "$(git rev-parse --show-toplevel)/../egms-qa-hf"
   ```

4. Inspect `git status`, the upstream divergence, recent commits, existing
   tags/releases, relevant project release scripts, and current HF revisions.
   Never merge, rebase, reset, or overwrite concurrent work automatically.

## Select the operating mode

- **Status**: perform read-only inspection and report passed, warning, failed,
  and untested gates. Do not modify local or remote state.
- **Prepare**: update local release files or HF staging, submit tests, build
  artifacts, and draft release notes. Do not push, upload, tag, merge, change
  visibility, or create a Release.
- **Publish**: enter only after the user explicitly authorizes the named remote
  writes in the current conversation. Repeat preflight with `--mode publish`
  immediately before writing remotely.
- **Verify**: download exact public GitHub/HF revisions into a new temporary
  directory and test them as an external user. Do not substitute local files.

If the request says “暂时不要 release”, “prepare only”, or equivalent, remain
in Prepare mode even when every gate passes.

## Enforce the test boundary

- Run all unit, integration, model, data, wheel-install, and public-download
  tests in Slurm jobs on `itc-gpu` or `main-gpu`; never run tests on a login
  node. Use the login node only for inspection, editing, submission, monitoring,
  and network operations that are not tests.
- Record job ID, partition, node, GPU model, command, exit status, and log path.
- Treat a queued/running job as untested, not passed. Require terminal success
  and inspect both stdout and stderr.
- Do not silently replace an unavailable GPU test with a CPU or login-node run.

## Prepare the candidate

Follow the ordered gates in the contract. In particular:

1. Reconcile unexpected branch divergence with the user before changing
   history. Preserve unrelated work.
2. Keep GitHub free of tracked `data/`, checkpoints, tokens, generated outputs,
   caches, credentials, and machine-specific paths.
3. Validate the structured HF dataset manifest and full SHA256 inventory.
4. Validate Encoder 4.3 configuration, checkpoint, normalization, token shape
   `[10000, 65, 256]`, and cross-repository hashes.
5. Validate all four translator families and downstream token consumption.
6. Build the wheel and test installation in a clean environment.
7. Produce release notes and a release ledger containing exact Git/HF revisions
   and test evidence.

Use current `hf` commands, not the deprecated `huggingface-cli`. Prefer
`hf upload`; flag legacy `hf upload-large-folder` scripts for modernization
before invoking them.

## Publish with explicit gates

Before each remote-writing phase, state the exact repositories, refs, and files
that will change. Obtain explicit authorization if it has not already been
given for that phase.

Publish in this order:

1. Push the reviewed Git commit/branch and wait for GitHub CI.
2. Upload the reviewed HF staging trees and capture the returned immutable
   revisions for dataset, encoder, and translator.
3. Run the public black-box verification pinned to those exact revisions.
4. Confirm version consistency and that the tested Git SHA is on remote `main`.
5. Create annotated Git/HF version tags without force-updating existing tags.
6. Create the GitHub Release from the verified tag and a notes file.
7. Repeat lightweight public installation and report the final URLs/revisions.

Never publish a partial set as a complete coordinated release. If a phase
fails after a remote write, stop, record the actual remote state, and ask before
rollback, deletion, force-push, tag replacement, or visibility changes.

## Hand off

Lead with one of: `not ready`, `candidate ready`, `published`, or `verification
failed`. List each gate with evidence, distinguish local staging from public
state, identify every remaining action, and never call an unpushed or
unverified candidate “released”.
