"""Table joins must preserve tile identity and the published D target schema."""
from pathlib import Path

import pandas as pd
import pytest

from egms_qa.qa_construction.tables import align_family_to_base, merge_task_tables, read_family
from egms_qa.qa_construction.summarize_temporal import D_COLUMNS, merge_temporal_tables


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


def test_temporal_summary_reads_current_columns_and_preserves_order(tmp_path: Path):
    ids = [f'tile-{i}' for i in range(10000)]
    splits = ['train'] * 8000 + ['val'] * 1000 + ['test'] * 1000
    for family, columns in D_COLUMNS.items():
        directory = tmp_path / family
        directory.mkdir()
        table = pd.DataFrame({'tile_id': ids, 'split': splits})
        for column in columns[2:]:
            table[column] = range(10000)
        if family != 'd1':
            table = table.iloc[::-1]
        table.to_csv(directory / f'{family}_final_table.csv', index=False)
    result = merge_temporal_tables(tmp_path)
    assert result['tile_id'].tolist() == ids
    assert result['D12_curvature_strength'].tolist() == list(range(10000))
    assert result['D13_changepoint_strength'].tolist() == list(range(10000))
    assert result['D22_phase_coherence'].tolist() == list(range(10000))
    # An incomplete family must fail instead of silently dropping tiles in a join.
    path = tmp_path / 'd2/d2_final_table.csv'
    pd.read_csv(path).iloc[:-1].to_csv(path, index=False)
    with pytest.raises(ValueError, match='expected 10000 rows'):
        merge_temporal_tables(tmp_path)
