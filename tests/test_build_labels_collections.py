"""Label assembly follows tile keys rather than a fixed corpus size."""
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
import pytest

from egms_qa.qa_construction import build_labels
from egms_qa.qa_construction.task_specs import TASK_SPECS


def write_tables(root, n=3, split="test"):
    ids = [f"tile-{i}" for i in range(n)]
    for family in sorted({s['source_family'] for s in TASK_SPECS}):
        rows = {'tile_id': ids, 'split': [split] * n}
        for spec in [s for s in TASK_SPECS if s['source_family'] == family]:
            rows[spec['target_column']] = np.arange(n, dtype=float) if spec['label_type'] == 'numeric' else [f'class-{i}' for i in range(n)]
        frame = pd.DataFrame(rows)
        if family != 'a1':
            frame = frame.iloc[::-1]
        path = root / family / f'{family}_final_table.csv'
        path.parent.mkdir(parents=True)
        frame.to_csv(path, index=False)
    for family, count in [('x1',5),('x2',6),('x3',3)]:
        path = root / family / f'{family}_final_table.csv'
        path.parent.mkdir()
        pd.DataFrame({'task_id':[f'{family.upper()}{i}' for i in range(1,count+1)],
                      'target_column':[f'{family.upper()}{i}_refusal' for i in range(1,count+1)]}).to_csv(path,index=False)
    return ids


def run_builder(monkeypatch, root, out, *options):
    monkeypatch.setattr(sys, 'argv', ['build_labels', '--tasks-root', str(root), '--out-dir', str(out), *options])
    build_labels.main()
    return pd.read_parquet(out/'labels.parquet'), json.loads((out/'labels_meta.json').read_text())


@pytest.mark.parametrize('n', [1, 3])
def test_new_collection_without_training_split_or_tokens(tmp_path, monkeypatch, n):
    root=tmp_path/'tables';ids=write_tables(root,n)
    labels,meta=run_builder(monkeypatch,root,tmp_path/'labels')
    assert labels['tile_id'].tolist()==ids
    assert labels['B21'].tolist()==list(range(n))
    assert labels['split'].tolist()==['test']*n
    assert len(labels.columns)==66
    assert len(meta['tasks'])==78
    assert meta['verification']['release_contract_checked'] is False
    assert meta['verification']['split_counts']=={'test':n}
    assert meta['sources']['encoder_cache'] is None


def test_release_size_check_is_explicit(tmp_path, monkeypatch):
    root=tmp_path/'tables';write_tables(root)
    with pytest.raises(ValueError,match='expected 10000'):
        run_builder(monkeypatch,root,tmp_path/'labels','--validate-release')
    assert not (tmp_path/'labels/labels.parquet').exists()


def test_release_split_check_is_preserved(tmp_path, monkeypatch):
    root=tmp_path/'tables';write_tables(root,10000)
    with pytest.raises(ValueError,match='unexpected split counts'):
        run_builder(monkeypatch,root,tmp_path/'labels','--validate-release')


@pytest.mark.parametrize('problem,match', [('missing_row','index mismatch'),('duplicate','duplicate'),
    ('different_split','index mismatch'),('invalid_split','invalid split'),('missing_id','missing'),('empty','empty')])
def test_bad_table_keys_fail_before_output(tmp_path, monkeypatch, problem, match):
    root=tmp_path/'tables';write_tables(root)
    path=root/'b2/b2_final_table.csv';frame=pd.read_csv(path)
    if problem=='missing_row':frame=frame.iloc[:-1]
    elif problem=='duplicate':frame.loc[0,'tile_id']=frame.loc[1,'tile_id']
    elif problem=='different_split':frame.loc[0,'split']='train'
    elif problem=='invalid_split':frame.loc[0,'split']='dev'
    elif problem=='missing_id':frame.loc[0,'tile_id']=None
    elif problem=='empty':frame=frame.iloc[:0]
    frame.to_csv(path,index=False)
    with pytest.raises(ValueError,match=match):
        run_builder(monkeypatch,root,tmp_path/'labels')
    assert not (tmp_path/'labels/labels.parquet').exists()


def test_missing_targets_are_retained(tmp_path, monkeypatch):
    root=tmp_path/'tables';write_tables(root)
    path=root/'b2/b2_final_table.csv';frame=pd.read_csv(path)
    frame['B21_mean_velocity_mm_yr']=np.nan;frame.to_csv(path,index=False)
    labels,meta=run_builder(monkeypatch,root,tmp_path/'labels')
    assert labels['B21'].isna().all()
    stats=next(t for t in meta['tasks'] if t['id']=='B21')
    assert stats['label_n']==0 and stats['label_std'] is None


def test_optional_cache_alignment_and_duplicate_rejection(tmp_path, monkeypatch):
    import torch
    root=tmp_path/'tables';ids=write_tables(root)
    cache=tmp_path/'cache.pt'
    torch.save({'tile_ids':ids[::-1],'splits':['test']*3},cache)
    labels,meta=run_builder(monkeypatch,root,tmp_path/'labels','--encoder-cache',str(cache))
    assert labels['tile_id'].tolist()==ids[::-1]
    assert labels['B21'].tolist()==[2.,1.,0.]
    assert meta['sources']['encoder_cache']==str(cache.resolve())
    torch.save({'tile_ids':[ids[0],ids[0],ids[1]],'splits':['test']*3},cache)
    with pytest.raises(ValueError,match='duplicate tile IDs'):
        run_builder(monkeypatch,root,tmp_path/'other','--encoder-cache',str(cache))


def test_cache_length_and_split_errors(tmp_path, monkeypatch):
    import torch
    root=tmp_path/'tables';ids=write_tables(root)
    cache=tmp_path/'cache.pt'
    for splits,message in [(['test'],'length mismatch'),(['train','test','test'],'split mismatch')]:
        torch.save({'tile_ids':ids,'splits':splits},cache)
        with pytest.raises(ValueError,match=message):
            run_builder(monkeypatch,root,tmp_path/'labels','--encoder-cache',str(cache))


def test_explicit_skip_keeps_existing_cli_behavior(tmp_path, monkeypatch):
    root=tmp_path/'tables';write_tables(root)
    _,meta=run_builder(monkeypatch,root,tmp_path/'labels','--encoder-cache',str(tmp_path/'absent.pt'),'--skip-cache-validation')
    assert meta['verification']['cache_checks']=={}
    assert meta['sources']['encoder_cache'] is None
