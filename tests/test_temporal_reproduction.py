"""Scientific and input-contract checks for D1 and S3 reconstruction."""
from pathlib import Path
import json

import numpy as np
import pandas as pd
import pytest

from egms_qa.qa_construction.temporal_inputs import TimeAxis
from egms_qa.qa_construction.inputs import read_tile_manifest
from egms_qa.qa_construction.tasks.d1.d1_compute import classify_tiles, COLUMNS, _configure_axis, _read_one
from egms_qa.qa_construction.tasks.s3 import s3_compute


def config_file(tmp_path: Path, **changes) -> Path:
    window = dict(stored_steps=294, t_start=0, t_end=294, input_length=294,
                  original_index_offset=8, original_epoch_year=2019.0, cadence_days=6.0,
                  original_t_start=8, original_t_end=302, original_source_steps=304)
    window.update(changes)
    path = tmp_path / 'data_config.json'
    path.write_text(json.dumps({'schema_version':'egms-qa-data-config-1.1', 'time_window':window}))
    return path


def test_physical_time_preserves_source_offset(tmp_path):
    axis = TimeAxis.from_file(config_file(tmp_path))
    np.testing.assert_array_equal(axis.years, 2019.0 + 8 * 6.0 / 365.25 + np.arange(294) * (6.0 / 365.25))
    series = np.tile(np.arange(294, dtype=float), (3, 1))
    path = tmp_path / 'tile.npz'
    np.savez(path, time_series=series)
    np.testing.assert_array_equal(axis.tile_median(path), series[0])


@pytest.mark.parametrize('changes', [dict(stored_steps=304), dict(original_index_offset=0), dict(cadence_days=0)])
def test_temporal_input_rejects_inconsistent_axes(tmp_path, changes):
    with pytest.raises(ValueError):
        TimeAxis.from_file(config_file(tmp_path, **changes))


def test_manifest_resolves_release_paths_and_rejects_duplicate_ids(tmp_path):
    frame = pd.DataFrame({'tile_id':['a','b'], 'split':['train','test'],
                          'path':['data/tiles/cell/a.npz','artifacts/source_tiles/cell/b.npz']})
    path = tmp_path/'split.parquet'
    frame.to_parquet(path)
    actual = read_tile_manifest(path, tmp_path/'tiles')
    assert actual.path.tolist() == [str(tmp_path/'tiles/cell/a.npz'), str(tmp_path/'tiles/cell/b.npz')]
    frame['tile_id'] = 'a'
    frame.to_parquet(path)
    with pytest.raises(ValueError, match='unique tile'):
        read_tile_manifest(path)


def test_d1_thresholds_use_only_training_tiles_and_gate_time():
    frame = pd.DataFrame({name: np.zeros(102) for name in COLUMNS})
    frame['tile_id'] = [str(i) for i in range(102)]
    frame['split'] = ['train']*100 + ['val','test']
    frame['D1_n_valid_epochs'] = 294
    frame['D12_curvature_strength'] = list(range(100)) + [1e9, 1e9]
    frame['D13_changepoint_strength'] = list(range(100)) + [1e9, 1e9]
    frame['D14_candidate_changepoint_time_year'] = 2021.25
    result, summary = classify_tiles(frame)
    assert summary['thresholds_fit_on_train']['d12_strong_threshold'] == pytest.approx(84.15)
    assert summary['thresholds_fit_on_train']['d13_strong_threshold'] == pytest.approx(84.15)
    assert result.loc[0, 'D11_long_term_trend_shape'] == 'linear_trend'
    assert pd.isna(result.loc[0, 'D14_dominant_changepoint_time_year'])
    assert result.loc[101, 'D11_long_term_trend_shape'] == 'complex_trend'
    assert result.loc[101, 'D14_dominant_changepoint_time_year'] == 2021.25


def test_d1_computes_geometry_from_npz_without_intermediate_table(tmp_path):
    axis = TimeAxis.from_file(config_file(tmp_path))
    _configure_axis(axis)
    x = np.linspace(-1, 1, 294)
    y = 5 * x**2 + 0.01 * np.sin(17 * x)
    tile = tmp_path/'tile.npz'
    np.savez(tile, time_series=np.tile(y, (3,1)))
    result = _read_one(('tile','train',str(tile)))
    assert result['D12_curvature_gain'] > 0.99
    assert result['D12_curvature_strength'] > 0
    assert result['D1_n_valid_epochs'] == 294
    assert np.isfinite(result['D13_changepoint_strength'])


def test_s3_reads_d1_geometry_and_aligns_an_explicit_table(tmp_path, monkeypatch):
    def family_table(root, family, expected_rows=None):
        assert expected_rows is None
        columns = dict(s3_compute.SOURCES)[family]
        return pd.DataFrame({'tile_id':['a','b'], 'split':['train','test'], **{key:[3.,4.] for key in columns}})
    monkeypatch.setattr(s3_compute, 'read_family', family_table)
    assert s3_compute.merge_sources(tmp_path)['D12_curvature_strength'].tolist() == [3.,4.]
    path = tmp_path/'d1.csv'
    inputs = pd.DataFrame({'tile_id':['b','a'],'split':['test','train'],
                           'D12_curvature_strength':[4.,3.], 'D13_changepoint_strength':[5.,np.nan]})
    inputs.to_csv(path,index=False)
    actual = s3_compute.merge_sources(tmp_path,path)
    assert actual['D12_curvature_strength'].tolist() == [3.,4.]
    assert pd.isna(actual.loc[0,'D13_changepoint_strength'])
    inputs.loc[0,'split']='val'
    inputs.to_csv(path,index=False)
    with pytest.raises(ValueError, match='index mismatch'):
        s3_compute.merge_sources(tmp_path,path)
