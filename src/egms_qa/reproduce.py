"""Run the training recipes and evaluation protocol distributed on the Hub.

Use --dry-run to inspect all commands and generated task manifests before a run.
Run from a checkout with the dataset installed by egms_qa.release.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8'))


def cli_args(values: dict) -> list[str]:
    result = []
    for key, value in values.items():
        flag = '--' + key.replace('_', '-')
        if isinstance(value, bool):
            if value:
                result.append(flag)
        elif value is not None:
            result.extend([flag, str(value)])
    return result


def encoder_plan(model_dir: Path, output_dir: Path) -> dict:
    model = read_json(model_dir / 'config.json')
    recipe = read_json(model_dir / 'training_args.json')
    data, mask = recipe['data'], recipe['masking']
    loss, opt = recipe['loss'], recipe['optimization']
    val, ckpt = recipe['validation'], recipe['checkpointing']
    if model['input_length'] != data['model_input_steps']:
        raise ValueError('Model and training recipe disagree on input length')
    if mask['strategy'] != 'synchronized_block' or opt['optimizer'] != 'AdamW':
        raise ValueError('Unsupported encoder training recipe')
    args = dict(
        output_dir=output_dir, normalization=model_dir/'normalization.json',
        input_length=model['input_length'], patch_size=model['patch_size'],
        d_model=model['d_model'], temporal_layers=model['temporal_layers'],
        temporal_heads=model['temporal_heads'], num_layers=model['spatial_layers'],
        num_heads=model['spatial_heads'], dropout=model['dropout'],
        coord_scale=model['coord_scale_m'], residual_head_mode=model['residual_head_mode'],
        tile_size=data['tile_size_m'], min_tile_points=data['minimum_points_per_tile'],
        max_tile_points=data['maximum_points_per_tile'],
        mask_strategy='block', sync_mask=True, mask_ratio=mask['train_ratio'],
        eval_mask_ratio=mask['evaluation_ratio'], mask_schedule=mask['schedule'],
        point_sampling=recipe['point_sampling']['method'],
        residual_sampling_alpha=recipe['point_sampling']['residual_sampling_alpha'],
        residual_loss_weight=loss['residual_loss_weight'],
        residual_consistency_weight=loss['residual_consistency_weight'],
        max_steps=opt['maximum_steps'], tiles_per_batch=opt['tiles_per_batch'],
        lr=opt['learning_rate'], min_lr=opt['minimum_learning_rate'],
        residual_head_lr=opt['residual_head_learning_rate'], lr_scheduler=opt['scheduler'],
        scheduler_total_steps=opt['scheduler_total_steps'], warmup_steps=opt['warmup_steps'],
        weight_decay=opt['weight_decay'], precision=opt['precision'], seed=opt['seed'],
        val_batches=val['batches'], val_every_steps=val['interval_steps'], val_seed=val['seed'],
        checkpoint_every_steps=ckpt['interval_steps'], log_every_steps=ckpt['log_interval_steps'],
        train_window_steps=ckpt['rolling_window_steps'],
    )
    return {'commands': [[sys.executable, '-m', 'egms_encoder.pretrain', *cli_args(args)]], 'task_manifests': {}}


def translator_plan(variant_dir: Path, output_dir: Path) -> dict:
    config = read_json(variant_dir / 'translator_config.json')
    recipe = read_json(variant_dir / 'training_args.json')
    if recipe.get('schema_version') != 'egms-qa-translator-training-1.1':
        raise ValueError('A complete translator training recipe (schema 1.1) is required')
    commands, manifests, stages = [], {}, recipe['stages']
    if not stages or stages[0]['initialize_from'] != 'base_model':
        raise ValueError('The first stage must start from the base model')
    stage_ids = [stage['id'] for stage in stages]
    if len(set(stage_ids)) != len(stage_ids):
        raise ValueError('Duplicate training stage IDs')
    for index, stage in enumerate(stages):
        if stage['id'] != Path(stage['id']).name or stage['id'] in {'.', '..'}:
            raise ValueError('Stage ID must be a directory name')
        args = {**recipe['common_cli_args'], **stage['cli_args']}
        args.update(host_model=config['base_model']['name_or_path'],
                    host_model_revision=config['base_model']['revision'],
                    variant=config['variant'], output_dir=output_dir/stage['id'])
        if index:
            if stage['initialize_from'] != stages[index-1]['id']:
                raise ValueError('Training stages must form a complete sequential recipe')
            previous = output_dir/stages[index-1]['id']/'best'
            args.update(resume_adapter=previous/'adapter', warm_start_projector=previous/'projector.safetensors')
        task_file = output_dir/'task_manifests'/f"{stage['id']}.json"
        tasks = stage['tasks']
        ids = tasks['maintain'] + tasks['focus']
        if not ids or len(set(ids)) != len(ids):
            raise ValueError('Each training stage must define unique tasks')
        manifests[str(task_file)] = tasks
        args['task_config'] = task_file
        commands.append([sys.executable, '-m', 'egms_qa.translator.train', *cli_args(args)])
    return {'commands': commands, 'task_manifests': manifests}


def evaluation_plan(variant_dir: Path, protocol_path: Path, output_dir: Path) -> dict:
    variant = read_json(variant_dir/'translator_config.json')['variant']
    protocol = read_json(protocol_path)
    if protocol.get('schema_version') != 'egms-qa-evaluation-protocol-1.0':
        raise ValueError('Unsupported evaluation protocol')
    tasks = protocol['tasks']
    if len(tasks) != 71 or len(set(tasks)) != 71:
        raise ValueError('The reported evaluation requires 71 distinct tasks')
    task_file = output_dir/'tasks.json'
    args = dict(adapter_dir=variant_dir, task_config=task_file,
                split=protocol['split'], cap_per_task=protocol['tiles_per_task'],
                x_cap_per_task=protocol['tiles_per_task'], phrases=protocol['phrases'],
                seed=protocol['seeds'][variant], max_new=protocol['max_new_tokens'],
                token_mode='normal', output=output_dir/'metrics.json', dump=output_dir/'answers.jsonl')
    return {'commands': [[sys.executable, '-m', 'egms_qa.translator.evaluate', *cli_args(args)]],
            'task_manifests': {str(task_file): {'maintain': tasks, 'focus': []}}}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='component', required=True)
    for name in ('encoder', 'translator', 'evaluate'):
        item = sub.add_parser(name)
        item.add_argument('--model-dir' if name == 'encoder' else '--variant-dir', type=Path, required=True)
        item.add_argument('--output-dir', type=Path, required=True)
        item.add_argument('--dry-run', action='store_true')
        if name == 'evaluate':
            item.add_argument('--evaluation-config', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.component == 'encoder':
        plan = encoder_plan(args.model_dir.resolve(), args.output_dir.resolve())
    elif args.component == 'translator':
        plan = translator_plan(args.variant_dir.resolve(), args.output_dir.resolve())
    else:
        plan = evaluation_plan(args.variant_dir.resolve(), args.evaluation_config.resolve(), args.output_dir.resolve())
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for path, content in plan['task_manifests'].items():
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(content, indent=2) + '\n', encoding='utf-8')
    for command in plan['commands']:
        subprocess.run(command, check=True)


if __name__ == '__main__':
    main()
