"""Label construction must preserve downloaded references and previous outputs."""
from pathlib import Path
import sys

import pytest

from egms_qa.paths import OUTPUTS_DIR, QA_DIR
from egms_qa.qa_construction import build_labels


def test_default_output_is_separate_from_installed_labels(monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['build_labels'])
    output = Path(build_labels.parse_args().out_dir)
    assert output == OUTPUTS_DIR / 'labels-generated'
    assert output != QA_DIR


@pytest.mark.parametrize('filename', ['labels.parquet', 'labels_meta.json'])
@pytest.mark.parametrize('kind', ['file', 'symlink', 'dangling_symlink'])
def test_existing_output_is_rejected_before_reading_inputs(tmp_path, monkeypatch, filename, kind):
    output = tmp_path / 'output'
    output.mkdir()
    target = tmp_path / 'downloaded'
    path = output / filename
    if kind == 'file':
        path.write_bytes(b'original output')
    else:
        if kind == 'symlink':
            target.write_bytes(b'downloaded reference')
        path.symlink_to(target)
    monkeypatch.setattr(sys, 'argv', ['build_labels', '--out-dir', str(output)])
    def unexpected_read(*args):
        pytest.fail('Existing outputs should be rejected before loading task inputs')
    monkeypatch.setattr(build_labels, 'read_family', unexpected_read)
    with pytest.raises(FileExistsError, match='choose a different --out-dir'):
        build_labels.main()
    assert list(output.iterdir()) == [path]
    if kind == 'file':
        assert path.read_bytes() == b'original output'
    else:
        assert path.is_symlink()
        if kind == 'symlink':
            assert target.read_bytes() == b'downloaded reference'
        else:
            assert not target.exists()


def test_output_directory_link_cannot_bypass_existing_file_check(tmp_path, monkeypatch):
    download = tmp_path / 'download'
    download.mkdir()
    (download / 'labels.parquet').write_bytes(b'reference')
    alias = tmp_path / 'alias'
    alias.symlink_to(download, target_is_directory=True)
    monkeypatch.setattr(sys, 'argv', ['build_labels', '--out-dir', str(alias)])
    with pytest.raises(FileExistsError):
        build_labels.main()
    assert (download / 'labels.parquet').read_bytes() == b'reference'
    assert not (download / 'labels_meta.json').exists()
