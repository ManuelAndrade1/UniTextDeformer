"""Record environment versions and file hashes for reproducible runs."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def cpu_tree(value):
    import torch
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(cpu_tree(item) for item in value)
    return value


def provenance(root, cfg, torch):
    packages = {}
    for name in ('torch', 'torchvision', 'numpy', 'scipy', 'libigl', 'cholespy', 'torch-sparse',
                 'pymeshlab', 'timm', 'open-clip-torch', 'safetensors', 'nvdiffrast', 'pointnet2-ops', 'clip'):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    paths = [*root.glob('unitextdeformer/*.py'), *root.glob('configs/*'),
             *root.glob('patches/*'), root / 'run.py', root / 'requirements.txt']
    runtime = root / '.runtime/TextDeformer'
    paths += [p for p in runtime.rglob('*') if p.suffix in {'.py', '.cu', '.cpp', '.h'}]
    paths += [root / 'external/Uni3D/models/point_encoder.py', root / 'external/Uni3D/models/uni3d.py']
    revisions = {}
    for name in ('TextDeformer', 'Uni3D', 'Pointnet2_PyTorch', 'nvdiffrast_official'):
        directory = root / 'external' / name
        revisions[name] = subprocess.check_output(['git', '-C', str(directory), 'rev-parse', 'HEAD'], text=True).strip()
    return dict(configuration=cfg, packages=packages, python=platform.python_version(),
                gpu=torch.cuda.get_device_name(cfg['gpu']), cuda_runtime=torch.version.cuda,
                repository_revisions=revisions, source_mesh_sha256=sha256(cfg['mesh']),
                source_sha256={str(p.relative_to(root)): sha256(p) for p in paths},
                asset_paths={key: cfg.get(key) for key in ('point_checkpoint', 'text_checkpoint', 'text_config')},
                final_mesh_selection='vertices before the final Adam update',
                checkpoint_selection='Jacobian/Adam state after the final attempted update')
