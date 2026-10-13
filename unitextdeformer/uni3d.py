"""Frozen Uni3D-B and aligned EVA02-E-14-plus text tower; live input gradients."""
import ast
import gc
import json
import logging
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]

def definitions(path, names, namespace):
    tree = ast.parse(path.read_text(), filename=str(path))
    selected = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
    if {node.name for node in selected} != set(names):
        raise RuntimeError(f'Pinned source architecture changed: {path}')
    namespace = dict(namespace)
    namespace.update(__name__='guidance.upstream', __file__=str(path))
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace

class FrozenGuidance(nn.Module):
    def __init__(self, point_model, text_embeddings, device, metadata):
        super().__init__()
        self.point_model = point_model.to(device).eval()
        self.point_model.requires_grad_(False)
        self.register_buffer('target_text', text_embeddings[0:1].detach().to(device).float())
        self.register_buffer('base_text', text_embeddings[1:2].detach().to(device).float())
        self.metadata = metadata

    def forward(self, points):
        return self.point_model(points).float()

def load_uni3d(cfg, device):
    import timm
    import open_clip
    from open_clip.model import _build_text_tower
    from pointnet2_ops import pointnet2_utils
    from safetensors import safe_open
    root = ROOT / 'external/Uni3D'
    common = dict(torch=torch, nn=nn, np=np, logging=logging, pointnet2_utils=pointnet2_utils)
    point = definitions(root / 'models/point_encoder.py',
                        ['fps', 'knn_point', 'square_distance', 'PatchDropout', 'Group', 'Encoder', 'PointcloudEncoder'], common)
    args = SimpleNamespace(pc_feat_dim=768, embed_dim=1024, group_size=64, num_group=512,
                           pc_encoder_dim=512, patch_dropout=0.0)
    transformer = timm.create_model('eva02_base_patch14_448', pretrained=False, drop_path_rate=0.0)
    encoder = point['PointcloudEncoder'](transformer, args)
    architecture = definitions(root / 'models/uni3d.py', ['Uni3D'], dict(common, PointcloudEncoder=point['PointcloudEncoder']))
    model = architecture['Uni3D'](encoder)
    path = ROOT / cfg.get('point_checkpoint', 'models/uni3d/base.pt')
    checkpoint = torch.load(path, map_location='cpu')
    raw = checkpoint.get('module', checkpoint.get('state_dict', checkpoint))
    state = {key.removeprefix('module.'): value for key, value in raw.items()}
    model.load_state_dict(state, strict=True)
    text_config_path = ROOT / cfg.get('text_config', 'models/uni3d/open_clip_config.json')
    config = json.loads(text_config_path.read_text())['model_cfg']
    official = open_clip.get_model_config('EVA02-E-14-plus')
    if config['text_cfg'] != official['text_cfg'] or config['embed_dim'] != official['embed_dim']:
        raise RuntimeError('Uni3D textual configuration differs from EVA02-E-14-plus')
    text_model = _build_text_tower(config['embed_dim'], config['text_cfg'])
    text_path = ROOT / cfg.get('text_checkpoint', 'models/uni3d/eva_clip.safetensors')
    with safe_open(str(text_path), framework='pt', device='cpu') as source:
        text_state = {key[len('text.'):]: source.get_tensor(key) for key in source.keys() if key.startswith('text.')}
    text_model.load_state_dict(text_state, strict=True)
    text_model.eval().requires_grad_(False)
    with torch.no_grad():
        embeddings = text_model(open_clip.get_tokenizer('EVA02-E-14-plus')([cfg['text_prompt'], cfg['base_text_prompt']]))
    del text_state, text_model, checkpoint, raw, state
    gc.collect()
    class PointsOnly(nn.Module):
        def __init__(self, model, rgb):
            super().__init__()
            self.model, self.rgb = model, rgb
        def forward(self, points):
            return self.model.encode_pc(torch.cat((points, torch.full_like(points, self.rgb)), dim=-1))
    return FrozenGuidance(PointsOnly(model, cfg.get('point_rgb', 0.4)), embeddings, device,
                          dict(backend='uni3d', architecture='Uni3D-B', checkpoint=path.relative_to(ROOT).as_posix(),
                               point_count=cfg.get('point_count', 10000), text_encoder='EVA02-E-14-plus',
                               text_checkpoint=text_path.relative_to(ROOT).as_posix(), point_rgb=cfg.get('point_rgb', 0.4),
                               normalization='sample mean / maximum radius', text_device='cpu'))
