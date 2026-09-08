# Changelog

All notable changes to EGMS-QA are documented here. The project follows
[Semantic Versioning](https://semver.org/).

## 1.0.0 - 2026-09-04

### Added

- Self-contained encoder, QA-construction, and translator packages.
- Frozen 10,000-tile split and public data/checkpoint configurations.
- Deterministic definitions and implementations for 78 A/B/C/D/S/X tasks.
- Natural-language QA generation, extraction, training, and evaluation tools.
- Support for Qwen, Gemma, Llama, and Mistral host-model families.
- Structured Hugging Face dataset installer and SHA256 release audit.
- English and Chinese project documentation.

### Release artifacts

- `risenyard/egms-qa-dataset`: QA, processed NPZ source tiles, token cache,
  labels, task tables, provenance, and integrity manifests.
- `risenyard/egms-qa-encoder`: frozen encoder checkpoint and normalization.
- `risenyard/egms-qa-translator`: four projector + LoRA translator checkpoints.

### Data terms

- EGMS-QA-created data and model artifacts: CC-BY-4.0.
- Copernicus-derived measurements: CLMS data policy with source,
  modification, and non-endorsement notices.
