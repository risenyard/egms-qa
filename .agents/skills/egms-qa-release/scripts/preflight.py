#!/usr/bin/env python3
"""Read-only preflight for the coordinated EGMS-QA GitHub/HF release."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


EXPECTED_GITHUB = "https://github.com/risenyard/egms-qa.git"
EXPECTED_HF_REPOS = {
    "dataset": "risenyard/egms-qa-dataset",
    "encoder": "risenyard/egms-qa-encoder",
    "translator": "risenyard/egms-qa-translator",
}
TRANSLATORS = ("qwen", "gemma", "llama", "mistral")


@dataclass(frozen=True)
class Check:
    level: str
    name: str
    detail: str


class Report:
    def __init__(self) -> None:
        self.checks: list[Check] = []

    def add(self, level: str, name: str, detail: str) -> None:
        self.checks.append(Check(level, name, detail))

    def passed(self, name: str, detail: str) -> None:
        self.add("PASS", name, detail)

    def warn(self, name: str, detail: str) -> None:
        self.add("WARN", name, detail)

    def fail(self, name: str, detail: str) -> None:
        self.add("FAIL", name, detail)

    @property
    def ok(self) -> bool:
        return not any(check.level == "FAIL" for check in self.checks)


def run_git(repo: Path, *args: str, check: bool = True) -> str:
    process = subprocess.run(
        ["git", *args], cwd=repo, text=True, capture_output=True, check=False
    )
    if check and process.returncode:
        message = process.stderr.strip() or process.stdout.strip()
        raise RuntimeError(f"git {' '.join(args)} failed: {message}")
    return process.stdout.strip()


def load_json(path: Path, report: Report, name: str) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        report.fail(name, f"cannot read {path}: {error}")
        return None
    if not isinstance(payload, dict):
        report.fail(name, f"expected a JSON object: {path}")
        return None
    return payload


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def require_paths(root: Path, relatives: list[str], report: Report, name: str) -> None:
    missing = [relative for relative in relatives if not (root / relative).exists()]
    if missing:
        report.fail(name, "missing: " + ", ".join(missing))
    else:
        report.passed(name, f"{len(relatives)} required paths present")


def package_version(pyproject: Path) -> str | None:
    text = pyproject.read_text(encoding="utf-8")
    section = re.search(r"(?ms)^\[project\]\s*(.*?)(?=^\[|\Z)", text)
    if not section:
        return None
    match = re.search(r'(?m)^version\s*=\s*["\']([^"\']+)["\']', section.group(1))
    return match.group(1) if match else None


def check_repository(repo: Path, mode: str, expected_version: str | None, report: Report) -> None:
    try:
        top = Path(run_git(repo, "rev-parse", "--show-toplevel")).resolve()
    except RuntimeError as error:
        report.fail("git repository", str(error))
        return
    if top != repo.resolve():
        report.fail("repository root", f"requested {repo.resolve()}, actual {top}")
        return
    report.passed("repository root", str(top))

    origin = run_git(repo, "remote", "get-url", "origin", check=False)
    normalized = origin.removesuffix("/")
    accepted = {
        EXPECTED_GITHUB,
        EXPECTED_GITHUB.removesuffix(".git"),
        "git@github.com:risenyard/egms-qa.git",
    }
    if normalized not in accepted:
        report.fail("GitHub origin", f"expected risenyard/egms-qa, found {origin or '[none]'}")
    else:
        report.passed("GitHub origin", origin)

    branch = run_git(repo, "branch", "--show-current", check=False) or "[detached]"
    report.passed("Git branch", branch)
    dirty = run_git(repo, "status", "--porcelain", check=False)
    if dirty:
        detail = f"working tree has {len(dirty.splitlines())} changed/untracked entries"
        (report.fail if mode == "publish" else report.warn)("working tree", detail)
    else:
        report.passed("working tree", "clean")

    upstream = run_git(repo, "rev-parse", "--abbrev-ref", "@{upstream}", check=False)
    if not upstream:
        (report.fail if mode == "publish" else report.warn)(
            "upstream", "current branch has no upstream"
        )
    else:
        counts = run_git(repo, "rev-list", "--left-right", "--count", f"HEAD...{upstream}")
        try:
            ahead, behind = (int(value) for value in counts.split())
        except (TypeError, ValueError):
            report.fail("upstream divergence", f"cannot parse: {counts}")
        else:
            if ahead or behind:
                detail = f"{branch} vs {upstream}: ahead={ahead}, behind={behind}"
                (report.fail if mode == "publish" else report.warn)(
                    "upstream divergence", detail
                )
            else:
                report.passed("upstream divergence", f"{branch} matches {upstream}")

    required = [
        "README.md",
        "README.zh-CN.md",
        "pyproject.toml",
        "LICENSE",
        "DATA_LICENSE",
        "src/egms_encoder",
        "src/egms_qa",
        "tests",
    ]
    require_paths(repo, required, report, "GitHub source tree")

    tracked_data = run_git(repo, "ls-files", "data", "data/**", check=False)
    if tracked_data:
        report.fail("GitHub data boundary", f"tracked data paths found: {tracked_data.splitlines()[0]}")
    else:
        report.passed("GitHub data boundary", "no tracked data/ paths")

    tracked_outputs = run_git(repo, "ls-files", "outputs", "outputs/**", check=False)
    if tracked_outputs:
        report.fail(
            "generated-output boundary",
            f"tracked outputs found: {tracked_outputs.splitlines()[0]}",
        )
    else:
        report.passed("generated-output boundary", "no tracked outputs/ paths")

    forbidden_pattern = "|".join(
        (
            "/home/" + "lis2",
            "EGMS" + "_VLM",
            "egms-qa-" + "4tu",
            "EGMS-" + "Encoder",
        )
    )
    grep = subprocess.run(
        [
            "git",
            "grep",
            "-nI",
            "-E",
            forbidden_pattern,
            "--",
        ],
        cwd=repo,
        text=True,
        capture_output=True,
        check=False,
    )
    if grep.returncode == 0 and grep.stdout.strip():
        first = grep.stdout.strip().splitlines()[0]
        report.fail("machine-specific tracked text", f"first match: {first}")
    elif grep.returncode == 1:
        report.passed("machine-specific tracked text", "none found")
    else:
        report.fail("machine-specific tracked text", grep.stderr.strip() or "git grep failed")

    symlinks = run_git(repo, "ls-files", "-s", check=False).splitlines()
    absolute_links: list[str] = []
    for entry in symlinks:
        fields = entry.split(maxsplit=3)
        if len(fields) == 4 and fields[0] == "120000":
            link_path = repo / fields[3]
            try:
                target = link_path.read_text(encoding="utf-8").strip()
            except OSError:
                target = ""
            if target.startswith("/"):
                absolute_links.append(fields[3])
    if absolute_links:
        report.fail("tracked symlinks", f"absolute target: {absolute_links[0]}")
    else:
        report.passed("tracked symlinks", "no tracked absolute symlinks")

    try:
        version = package_version(repo / "pyproject.toml")
    except OSError as error:
        report.fail("package version", str(error))
        version = None
    if not version:
        report.fail("package version", "[project].version is missing")
    elif expected_version and version != expected_version:
        report.fail("package version", f"expected {expected_version}, found {version}")
    else:
        report.passed("package version", version)

    citation = repo / "CITATION.cff"
    if citation.exists() and version:
        match = re.search(r"(?m)^version:\s*[\"']?([^\s\"']+)", citation.read_text())
        if not match or match.group(1) != version:
            report.fail("citation version", f"CITATION.cff does not match {version}")
        else:
            report.passed("citation version", version)
    if (repo / "CHANGELOG.md").exists() and version:
        if re.search(rf"(?m)^## \[{re.escape(version)}\](?:\s|$)", (repo / "CHANGELOG.md").read_text()):
            report.passed("changelog version", version)
        else:
            report.fail("changelog version", f"no [{version}] section")

    for command in ("git", "gh", "hf", "sbatch"):
        if shutil.which(command):
            report.passed(f"command: {command}", shutil.which(command) or "available")
        else:
            level = report.fail if mode == "publish" else report.warn
            level(f"command: {command}", "not found on PATH")


def resolve_component(hf_root: Path, supplied: Path | None, name: str) -> Path:
    if supplied is None:
        return (hf_root / name).resolve()
    return supplied.expanduser().resolve()


def check_broken_links(root: Path, report: Report, name: str) -> None:
    broken = [path for path in root.rglob("*") if path.is_symlink() and not path.exists()]
    if broken:
        report.fail(name, f"{len(broken)} broken symlinks; first: {broken[0]}")
    else:
        report.passed(name, "no broken symlinks")


def check_dataset(dataset: Path, report: Report) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    if not dataset.is_dir():
        report.fail("HF dataset staging", f"directory not found: {dataset}")
        return None, None
    report.passed("HF dataset staging", str(dataset))
    required = [
        "README.md",
        "DATA_TERMS.md",
        "SOURCE_PROVENANCE.md",
        "metadata/release_manifest.json",
        "metadata/files.sha256",
        "metadata/data_config.json",
        "metadata/split_manifest.parquet",
        "metadata/tile_manifest.parquet",
        "artifacts/source_tiles",
        "artifacts/representations/egms_tokens_10k.pt",
        "artifacts/representations/egms_tokens_10k_metadata.json",
        "artifacts/labels/labels.parquet",
        "artifacts/labels/metadata.json",
        "artifacts/reference_tables",
        "data/qa/train.jsonl",
        "data/qa/validation.jsonl",
        "data/qa/test.jsonl",
    ]
    require_paths(dataset, required, report, "HF dataset contract")
    check_broken_links(dataset, report, "HF dataset symlinks")

    tiles = list((dataset / "artifacts/source_tiles").rglob("*.npz"))
    if len(tiles) == 10_000:
        report.passed("source tile count", "10,000 NPZ files")
    else:
        report.fail("source tile count", f"expected 10,000, found {len(tiles):,}")

    manifest = load_json(
        dataset / "metadata/release_manifest.json", report, "dataset release manifest"
    )
    if manifest:
        if manifest.get("schema_version") != "egms-qa-release-v1":
            report.fail("dataset manifest schema", str(manifest.get("schema_version")))
        else:
            report.passed("dataset manifest schema", "egms-qa-release-v1")
        if manifest.get("repository") != EXPECTED_HF_REPOS["dataset"]:
            report.fail("dataset repository metadata", str(manifest.get("repository")))
        else:
            report.passed("dataset repository metadata", EXPECTED_HF_REPOS["dataset"])
        split = manifest.get("tile_split") or {}
        counts = split.get("counts") or {}
        expected_counts = {"train": 8000, "val": 1000, "test": 1000}
        if split.get("tiles") == 10_000 and counts == expected_counts:
            report.passed("dataset split", "10,000 = 8,000/1,000/1,000")
        else:
            report.fail("dataset split", f"found tiles={split.get('tiles')}, counts={counts}")

        checksum_path = dataset / str(
            (manifest.get("integrity") or {}).get("checksums", "metadata/files.sha256")
        )
        try:
            lines = [line for line in checksum_path.read_text().splitlines() if line.strip()]
        except OSError as error:
            report.fail("checksum inventory", str(error))
        else:
            expected = (manifest.get("integrity") or {}).get("files_hashed")
            if len(lines) != expected:
                report.fail("checksum inventory", f"manifest={expected}, inventory={len(lines)}")
            else:
                missing = []
                for line in lines:
                    fields = line.split("  ", 1)
                    if len(fields) != 2 or not (dataset / fields[1]).exists():
                        missing.append(fields[-1])
                        break
                if missing:
                    report.fail("checksum inventory", f"missing or malformed entry: {missing[0]}")
                else:
                    report.passed("checksum inventory", f"{len(lines):,} declared files present")

    token_meta = load_json(
        dataset / "artifacts/representations/egms_tokens_10k_metadata.json",
        report,
        "token metadata",
    )
    if token_meta:
        if token_meta.get("schema_version") != "egms-tokens-1.1":
            report.fail("token metadata schema", str(token_meta.get("schema_version")))
        else:
            report.passed("token metadata schema", "egms-tokens-1.1")
        input_contract = token_meta.get("input_contract") or {}
        statistics = token_meta.get("statistics") or {}
        shape = (
            statistics.get("tiles"),
            input_contract.get("token_count"),
            input_contract.get("token_width"),
        )
        if shape == (10_000, 65, 256):
            report.passed("token metadata shape", "[10000, 65, 256]")
        else:
            report.fail("token metadata shape", str(shape))
        if token_meta.get("release_name") == "EGMS-QA token cache":
            report.passed("token release name", "EGMS-QA token cache")
        else:
            report.fail("token release name", str(token_meta.get("release_name")))
        window = (
            input_contract.get("stored_steps"),
            input_contract.get("stored_window"),
            input_contract.get("source_axis_steps"),
            input_contract.get("source_window"),
            input_contract.get("source_index_offset"),
        )
        if window == (294, "[0,294)", 304, "[8,302)", 8):
            report.passed("token time contract", "stored [0,294), original [8,302)")
        else:
            report.fail("token time contract", str(window))
    legacy_tokens = [
        dataset / "artifacts/representations/encoder_tokens_10k.pt",
        dataset / "artifacts/representations/encoder_tokens_10k_metadata.json",
    ]
    present_legacy = [path.name for path in legacy_tokens if path.exists()]
    if present_legacy:
        report.fail("legacy token artifacts", "present: " + ", ".join(present_legacy))
    else:
        report.passed("legacy token artifacts", "none present")
    return manifest, token_meta


def check_encoder(encoder: Path, report: Report) -> dict[str, Any] | None:
    if not encoder.is_dir():
        report.fail("HF encoder staging", f"directory not found: {encoder}")
        return None
    report.passed("HF encoder staging", str(encoder))
    require_paths(
        encoder,
        [
            "README.md",
            "encoder.safetensors",
            "config.json",
            "normalization.json",
            "training_args.json",
            "eval_results.json",
        ],
        report,
        "HF encoder contract",
    )
    check_broken_links(encoder, report, "HF encoder symlinks")
    legacy = [name for name in ("encoder.pt", "args.json") if (encoder / name).exists()]
    if legacy:
        report.fail("legacy encoder artifacts", "present: " + ", ".join(legacy))
    else:
        report.passed("legacy encoder artifacts", "none present")
    config = load_json(encoder / "config.json", report, "encoder config")
    if config:
        actual = (
            config.get("schema_version"),
            config.get("model_type"),
            config.get("input_length"),
            config.get("d_model"),
        )
        if actual == (
            "egms-qa-encoder-config-1.0",
            "egms_encoder",
            294,
            256,
        ):
            report.passed("Encoder 4.3 architecture", "input=294, width=256")
        else:
            report.fail("Encoder 4.3 architecture", str(actual))
    training = load_json(encoder / "training_args.json", report, "encoder training args")
    if training:
        actual = (
            training.get("schema_version"),
            (training.get("data") or {}).get("model_input_steps"),
        )
        if actual == ("egms-qa-encoder-training-1.0", 294):
            report.passed("encoder training contract", "input=294")
        else:
            report.fail("encoder training contract", str(actual))
    evaluation = load_json(encoder / "eval_results.json", report, "encoder evaluation")
    if evaluation:
        actual = (
            evaluation.get("schema_version"),
            (evaluation.get("sample") or {}).get("input_steps"),
        )
        if actual == ("egms-qa-encoder-evaluation-1.0", 294):
            report.passed("encoder evaluation contract", "input=294")
        else:
            report.fail("encoder evaluation contract", str(actual))
    normalization = load_json(encoder / "normalization.json", report, "normalization")
    if normalization:
        mean, std = normalization.get("mean"), normalization.get("std")
        if isinstance(mean, (int, float)) and isinstance(std, (int, float)) and std > 0:
            report.passed("normalization values", f"mean={mean}, std={std}")
        else:
            report.fail("normalization values", f"mean={mean}, std={std}")
    return config


def check_translator(translator: Path, report: Report) -> None:
    if not translator.is_dir():
        report.fail("HF translator staging", f"directory not found: {translator}")
        return
    report.passed("HF translator staging", str(translator))
    required = ["README.md", "manifest.json"]
    for family in TRANSLATORS:
        required.extend(
            [
                f"{family}/projector.safetensors",
                f"{family}/translator_config.json",
                f"{family}/training_args.json",
                f"{family}/eval_results.json",
                f"{family}/adapter/adapter_config.json",
                f"{family}/adapter/adapter_model.safetensors",
            ]
        )
    require_paths(translator, required, report, "HF translator contract")
    check_broken_links(translator, report, "HF translator symlinks")
    manifest = load_json(translator / "manifest.json", report, "translator manifest")
    if manifest:
        variants = set((manifest.get("variants") or {}).keys())
        if (
            manifest.get("schema_version") == "egms-qa-translator-manifest-1.0"
            and variants == set(TRANSLATORS)
        ):
            report.passed("translator manifest contract", ", ".join(TRANSLATORS))
        else:
            report.fail(
                "translator manifest contract",
                f"schema={manifest.get('schema_version')}, variants={sorted(variants)}",
            )


def check_cross_hashes(
    dataset: Path,
    encoder: Path,
    token_meta: dict[str, Any] | None,
    report: Report,
) -> None:
    if not token_meta:
        return
    expected = token_meta.get("reproduction_input_sha256") or {}
    paths = {
        "encoder_weights": encoder / "encoder.safetensors",
        "encoder_config": encoder / "config.json",
        "normalization": encoder / "normalization.json",
        "split_manifest": dataset / "metadata/split_manifest.parquet",
        "data_config": dataset / "metadata/data_config.json",
    }
    for key, path in paths.items():
        if not path.exists() or not expected.get(key):
            report.fail(f"token provenance: {key}", "file or recorded SHA256 is missing")
            continue
        actual = sha256(path)
        if actual == expected[key]:
            report.passed(f"token provenance: {key}", actual)
        else:
            report.fail(
                f"token provenance: {key}",
                f"metadata={expected[key]}, actual={actual}",
            )


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--repo-root", type=Path, required=True)
    result.add_argument("--hf-root", type=Path, required=True)
    result.add_argument("--dataset-dir", type=Path)
    result.add_argument("--encoder-dir", type=Path)
    result.add_argument("--translator-dir", type=Path)
    result.add_argument("--expected-version")
    result.add_argument("--mode", choices=("status", "publish"), default="status")
    result.add_argument("--json", action="store_true", dest="as_json")
    return result


def main() -> int:
    args = parser().parse_args()
    report = Report()
    repo = args.repo_root.expanduser().resolve()
    hf_root = args.hf_root.expanduser().resolve()
    dataset = resolve_component(hf_root, args.dataset_dir, "dataset")
    encoder = resolve_component(hf_root, args.encoder_dir, "encoder")
    translator = resolve_component(hf_root, args.translator_dir, "translator")

    check_repository(repo, args.mode, args.expected_version, report)
    _, token_meta = check_dataset(dataset, report)
    check_encoder(encoder, report)
    check_translator(translator, report)
    check_cross_hashes(dataset, encoder, token_meta, report)

    counts = {
        level: sum(check.level == level for check in report.checks)
        for level in ("PASS", "WARN", "FAIL")
    }
    payload = {
        "schema_version": "egms-qa-release-preflight-1.0",
        "mode": args.mode,
        "ready": report.ok and (args.mode == "publish" or counts["WARN"] == 0),
        "paths": {
            "repo": str(repo),
            "dataset": str(dataset),
            "encoder": str(encoder),
            "translator": str(translator),
        },
        "summary": counts,
        "checks": [asdict(check) for check in report.checks],
    }
    if args.as_json:
        print(json.dumps(payload, indent=2))
    else:
        for check in report.checks:
            print(f"[{check.level}] {check.name}: {check.detail}")
        print(
            f"summary: pass={counts['PASS']} warn={counts['WARN']} "
            f"fail={counts['FAIL']} ready={payload['ready']}"
        )
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
