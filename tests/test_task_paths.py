"""Task entry points must honor configured data and output roots."""
import os
from pathlib import Path
import subprocess
import sys


def test_all_task_cli_defaults_honor_independent_roots(tmp_path):
    code = r'''
import argparse, importlib
from pathlib import Path
import egms_qa.qa_construction
class Captured(Exception):
    pass
parsers = []
def capture(self, *args, **kwargs):
    parsers.append(self)
    raise Captured
argparse.ArgumentParser.parse_args = capture
root = Path(egms_qa.qa_construction.__file__).parent / 'tasks'
for path in sorted(root.glob('*/*.py')):
    module = importlib.import_module('egms_qa.qa_construction.tasks.' + path.parent.name + '.' + path.stem)
    try:
        module.main()
    except Captured:
        pass
    else:
        raise AssertionError(path)
def actions(parser):
    for action in parser._actions:
        yield action
        if isinstance(action, argparse._SubParsersAction):
            for child in action.choices.values():
                yield from actions(child)
import os
for parser in parsers:
    for action in actions(parser):
        default = action.default
        if isinstance(default, (str, Path)):
            value = str(default)
            if '/encoder/' in value:
                assert value.startswith(os.environ['EGMS_QA_DATA'] + '/'), (action.dest, value)
            if '/tasks/' in value or '/tasks-rebuilt/' in value or '/s3-temporal-refit' in value:
                assert value.startswith(os.environ['EGMS_QA_OUTPUTS'] + '/'), (action.dest, value)
            assert not value.startswith(('data/', 'outputs/', './data/', './outputs/')), (action.dest, value)
assert len(parsers) == 30, len(parsers)
for parser in parsers:
    for action in actions(parser):
        if action.dest in {'out_dir', 'out_path', 'base_dir'} and action.default is not None:
            assert '/tasks/' not in str(action.default), (action.dest, action.default)
'''
    env = dict(os.environ, EGMS_QA_ROOT=str(tmp_path / 'work'),
               EGMS_QA_DATA=str(tmp_path / 'download'),
               EGMS_QA_OUTPUTS=str(tmp_path / 'results'),
               PYTHONPATH=str(Path(__file__).resolve().parents[1] / 'src'))
    subprocess.run([sys.executable, '-c', code], cwd=tmp_path, env=env, check=True)
