import json
from pathlib import Path

import pytest

from egms_qa.reproduce import translator_plan, evaluation_plan


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def test_training_stages_use_outputs_created_by_the_preceding_stage(tmp_path):
    variant = tmp_path/'variant'
    write_json(variant/'translator_config.json', {
        'variant': 'qwen', 'base_model': {'name_or_path': 'vendor/base', 'revision': 'fixed-revision'},
    })
    recipe = {'schema_version': 'egms-qa-translator-training-1.1',
              'common_cli_args': {'batch_size': 32, 'no_4bit': True}, 'stages': [
                  {'id': 'alignment', 'initialize_from': 'base_model',
                   'cli_args': {'train_steps': 100}, 'tasks': {'maintain': ['A11'], 'focus': []}},
                  {'id': 'instruction', 'initialize_from': 'alignment',
                   'cli_args': {'train_steps': 200}, 'tasks': {'maintain': ['A11','X11'], 'focus': []}},
              ]}
    write_json(variant/'training_args.json', recipe)
    output = tmp_path/'run'
    plan = translator_plan(variant, output)
    first, second = plan['commands']
    assert '--resume-adapter' not in first
    assert second[second.index('--resume-adapter')+1] == str(output/'alignment/best/adapter')
    assert second[second.index('--warm-start-projector')+1] == str(output/'alignment/best/projector.safetensors')
    assert first[first.index('--host-model-revision')+1] == 'fixed-revision'
    assert not output.exists()  # planning must not start training or write runtime files
    recipe['stages'][1]['initialize_from'] = 'unpublished'
    write_json(variant/'training_args.json', recipe)
    with pytest.raises(ValueError, match='complete sequential recipe'):
        translator_plan(variant, output)


def test_evaluation_uses_variant_seed_and_full_caps(tmp_path):
    write_json(tmp_path/'translator_config.json', {'variant': 'llama'})
    protocol = {'schema_version': 'egms-qa-evaluation-protocol-1.0',
                'tasks': [f'T{i}' for i in range(71)], 'split': 'test',
                'tiles_per_task': 1000, 'phrases': 20, 'max_new_tokens': 96,
                'seeds': {'llama': 96313}}
    path = tmp_path/'protocol.json'
    write_json(path, protocol)
    command = evaluation_plan(tmp_path, path, tmp_path/'out')['commands'][0]
    for flag, value in [('--seed','96313'), ('--cap-per-task','1000'), ('--x-cap-per-task','1000')]:
        assert command[command.index(flag)+1] == value
    protocol['tasks'][-1] = protocol['tasks'][0]
    write_json(path, protocol)
    with pytest.raises(ValueError, match='71 distinct tasks'):
        evaluation_plan(tmp_path, path, tmp_path/'out')
