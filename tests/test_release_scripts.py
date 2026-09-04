from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_encoder_release_suite_requires_gpu_and_full_hf_contract() -> None:
    encoder_test = (
        REPO_ROOT / "scripts/release/test_encoder_release.sbatch"
    ).read_text(encoding="utf-8")
    assert "#SBATCH --partition=itc-gpu,main-gpu" in encoder_test
    assert "#SBATCH --gres=gpu:1" in encoder_test
    assert "EGMS_QA_REQUIRE_HF_INTEGRATION=1" in encoder_test
    assert "EGMS_QA_REQUIRE_FULL_HF_DATASET=1" in encoder_test


def test_release_scripts_are_portable_and_use_current_hf_cli() -> None:
    release_dir = REPO_ROOT / "scripts/release"
    scripts = list(release_dir.glob("*.sh")) + list(release_dir.glob("*.sbatch"))
    assert scripts
    for path in scripts:
        text = path.read_text(encoding="utf-8")
        assert "/home/lis2" not in text, path
        assert "EGMS_VLM" not in text, path
        assert "huggingface-cli" not in text, path
        assert "upload-large-folder" not in text, path
