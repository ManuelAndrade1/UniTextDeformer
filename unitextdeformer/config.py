"""Resolve one common protocol and three explicit ablations, without GPU imports."""
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = {
    'uni3d-vc': {'view_consistency': True, 'consistency_loss_weight': 0.5, 'delta_clip_weight': 1.0},
    'uni3d-no-vc': {'view_consistency': False, 'consistency_loss_weight': 0.0, 'delta_clip_weight': 1.0},
    'uni3d-no-vc-no-delta': {'view_consistency': False, 'consistency_loss_weight': 0.0, 'delta_clip_weight': 0.0},
}


def repository_path(value):
    """Normalize an input/output path and keep it inside the repository."""
    candidate = Path(value)
    resolved = (ROOT / candidate).resolve()
    try:
        relative = resolved.relative_to(ROOT)
    except ValueError:
        raise ValueError('Inputs, models and outputs must be inside the repository') from None
    return relative.as_posix()


def load_cases(path):
    cases = json.loads(Path(path).read_text())
    if not isinstance(cases, list) or not cases:
        raise ValueError('Cases must be a nonempty JSON list')
    seen = set()
    for case in cases:
        for key in ('id', 'mesh', 'base_text_prompt', 'text_prompt'):
            if not isinstance(case.get(key), str) or not case[key].strip():
                raise ValueError(f'Missing case field: {key}')
        if case['id'] in seen or Path(case['id']).name != case['id'] or case['id'] in {'.', '..'}:
            raise ValueError(f'Invalid or duplicate case ID: {case["id"]}')
        seen.add(case['id'])
    return cases


def resolve_config(base, case, variant, output, steps=None):
    cfg = dict(base)
    cfg.update(VARIANTS[variant])
    cfg.update(case_id=case['id'], variant=variant, mesh=repository_path(case['mesh']),
               base_text_prompt=case['base_text_prompt'], text_prompt=case['text_prompt'],
               mesh_sha256=case.get('mesh_sha256'), output_path=repository_path(output))
    for key in ('point_checkpoint', 'text_checkpoint', 'text_config', 'model_cache'):
        cfg[key] = repository_path(cfg[key])
    if steps is not None:
        cfg['steps'] = steps
    for key in ('steps', 'seed', 'batch_size', 'train_res', 'point_count'):
        if type(cfg[key]) is not int or (key != 'seed' and cfg[key] < 1):
            raise ValueError(f'{key} must be an integer in its valid range')
    if cfg['batch_size'] < 2 and cfg['view_consistency']:
        raise ValueError('View consistency needs at least two cameras')
    if not 0 <= cfg['seed'] < 2**32:
        raise ValueError('seed must be between 0 and 2**32 - 1')
    if cfg['point_count'] < 512:
        raise ValueError('Uni3D-B requires at least 512 points')
    if type(cfg['camera_rng_interval']) is not int or cfg['camera_rng_interval'] < 1:
        raise ValueError('camera_rng_interval must be a positive integer')
    if not 0 <= cfg['consistency_vit_layer'] < 12:
        raise ValueError('consistency_vit_layer must be between 0 and 11')
    if cfg['lr'] <= 0 or cfg['clip_weight'] < 0 or cfg['regularize_jacobians_weight'] < 0:
        raise ValueError('Invalid learning rate or loss weight')
    cfg['training_camera_count'] = cfg['batch_size'] if cfg['view_consistency'] else 0
    return cfg


def load_base(path):
    cfg = yaml.safe_load(Path(path).read_text())
    if not isinstance(cfg, dict):
        raise ValueError('Configuration must be a YAML mapping')
    return cfg
