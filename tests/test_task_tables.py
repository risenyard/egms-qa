"""Table joins must preserve tile identity and the published D target schema."""
from pathlib import Path

import pandas as pd
import pytest

from egms_qa.qa_construction.tables import align_family_to_base, read_family
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
