from __future__ import annotations

import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
ENCODER_ROOT = REPO_ROOT / "src/egms_encoder"


def test_encoder_package_has_no_qa_namespace_imports() -> None:
    violations = []
    for path in ENCODER_ROOT.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [item.name for item in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            if any(name == "egms_qa" or name.startswith("egms_qa.") for name in names):
                violations.append(str(path.relative_to(REPO_ROOT)))
    assert not violations


def test_encoder_readme_contains_no_private_release_operations() -> None:
    text = (ENCODER_ROOT / "README.md").read_text(encoding="utf-8")
    forbidden = (
        "/home/",
        "EGMS_VLM",
        "dataset-staging",
        "migration_",
        "rollback",
        "sbatch",
        "Slurm",
        "family-C4",
    )
    assert not [value for value in forbidden if value in text]
