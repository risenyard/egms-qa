"""Task inputs preserve tile identity and defined class boundaries."""
from pathlib import Path

import pandas as pd
import numpy as np
import pytest

from egms_qa.qa_construction.tables import align_family_to_base, merge_task_tables, read_family


@pytest.mark.parametrize('boundary,below,at', [
    (1.5, 'very_low', 'low'),
    (1.9, 'low', 'moderate'),
    (2.24, 'moderate', 'high'),
    (2.9, 'high', 'very_high'),
])
def test_b35_preserves_float32_boundary_membership(boundary, below, at):
    from egms_qa.qa_construction.tasks.b3.b3_compute import _worst_point_significance
    stored = np.float32(boundary)
    assert _worst_point_significance(float(stored)) == at
    assert _worst_point_significance(float(np.nextafter(stored, np.float32(-np.inf)))) == below


def test_alignment_uses_tile_and_split_keys():
    table = pd.DataFrame({'tile_id': ['b', 'a'], 'split': ['test', 'train'], 'value': [2, 1]})
    index = pd.MultiIndex.from_tuples([('a', 'train'), ('b', 'test')], names=['tile_id', 'split'])
    aligned = align_family_to_base(table, index, 'd2')
    assert aligned['value'].tolist() == [1, 2]
    assert table['tile_id'].tolist() == ['b', 'a']


@pytest.mark.parametrize('split', ['val', 'test'])
def test_alignment_rejects_missing_or_different_keys(split):
    table = pd.DataFrame({'tile_id': ['a'], 'split': [split]})
    index = pd.MultiIndex.from_tuples([('a', 'train')], names=['tile_id', 'split'])
    with pytest.raises(ValueError, match='index mismatch'):
        align_family_to_base(table, index, 'd2')


def test_reader_rejects_duplicate_tile_ids(tmp_path: Path):
    directory = tmp_path / 'd1'
    directory.mkdir()
    pd.DataFrame({'tile_id': ['a', 'a'], 'split': ['train', 'test']}).to_csv(
        directory / 'd1_final_table.csv', index=False
    )
    with pytest.raises(ValueError, match='duplicate tile_id'):
        read_family(tmp_path, 'd1')


def test_task_join_preserves_order_and_allows_explicit_reference_superset():
    base = pd.DataFrame({'tile_id': ['b', 'a'], 'split': ['test', 'train'], 'x': [2, 1]})
    other = pd.DataFrame({'tile_id': ['a', 'b', 'c'], 'split': ['train', 'test', 'val'], 'y': [10, 20, 30]})
    with pytest.raises(ValueError, match='index mismatch'):
        merge_task_tables(base, other, 'reference')
    result = merge_task_tables(base, other, 'reference', allow_extra=True)
    assert result['tile_id'].tolist() == ['b', 'a']
    assert result['y'].tolist() == [20, 10]


@pytest.mark.parametrize('problem', ['missing', 'split', 'duplicate', 'null'])
def test_task_join_rejects_inputs_that_would_drop_or_duplicate_tiles(problem):
    base = pd.DataFrame({'tile_id': ['a', 'b'], 'split': ['train', 'test']})
    other = base.assign(value=[1, 2])
    if problem == 'missing':
        other = other.iloc[:1]
    elif problem == 'split':
        other.loc[1, 'split'] = 'val'
    elif problem == 'duplicate':
        other = pd.concat([other, other.iloc[:1]])
    else:
        other.loc[1, 'tile_id'] = None
    with pytest.raises(ValueError, match='index mismatch|missing keys|duplicate'):
        merge_task_tables(base, other, 'reference', allow_extra=True)


def test_a5_rejects_duplicate_ids_instead_of_overwriting_rows(tmp_path):
    from egms_qa.qa_construction.tasks.a5.a5_compute import read_by_tile
    path = tmp_path / 'a1.csv'
    path.write_text('tile_id,split\na,train\na,test\n')
    with pytest.raises(ValueError, match='duplicate tile_id'):
        read_by_tile(path)


def test_c4_missing_tile_is_an_error_not_a_no_motion_label(tmp_path):
    from egms_qa.qa_construction.tasks.c4.c4_compute import _final_one
    with pytest.raises(ValueError, match='cannot compute C4'):
        _final_one(('tile', 'test', str(tmp_path / 'missing.npz')), 4.8)


def test_manifest_resolves_configured_data_root_outside_working_directory(tmp_path, monkeypatch):
    from egms_qa.qa_construction import inputs
    data = tmp_path / 'download'
    monkeypatch.setattr(inputs, 'DATA_DIR', data)
    path = tmp_path / 'split.parquet'
    pd.DataFrame({'tile_id': ['a'], 'split': ['train'], 'path': ['data/tiles/cell/a.npz']}).to_parquet(path)
    assert inputs.read_tile_manifest(path)['path'].tolist() == [str(data / 'tiles/cell/a.npz')]
