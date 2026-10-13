#!/usr/bin/env python3
"""Extract pinned upstream runtime/meshes; prepare PointNet2; optionally fetch weights."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def git(directory, *args):
    return subprocess.check_output(['git', '-C', str(directory), *args])


def prepare():
    manifest = json.loads((ROOT / 'configs/assets.json').read_text())
    repositories = {r['id']: r for r in manifest['repositories']}
    source = ROOT / 'external/TextDeformer'
    pointnet = ROOT / 'external/Pointnet2_PyTorch'
    for name, directory in [('textdeformer_code', source), ('pointnet2_code', pointnet)]:
        expected = repositories[name]['revision']
        if git(directory, 'rev-parse', 'HEAD').decode().strip() != expected:
            raise ValueError(f'{directory} must be checked out at {expected}')
    runtime = ROOT / '.runtime/TextDeformer'
    runtime.mkdir(parents=True, exist_ok=True)
    names = git(source, 'ls-tree', '-r', '--name-only', 'HEAD').decode().splitlines()
    utilities = {'utilities/camera.py', 'utilities/clip_spatial.py', 'utilities/resize_right.py', 'utilities/helpers.py'}
    for name in names:
        required = (name.startswith('NeuralJacobianFields/') or name.startswith('nvdiffmodeling/') or name in utilities or name == 'LICENSE')
        if not required or '/tests/' in name or name.endswith('/regularizer.py'):
            continue
        destination = runtime / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(git(source, 'show', 'HEAD:' + name))
    subprocess.run(['git', 'apply', '--check', str(ROOT / 'patches/textdeformer.patch')], cwd=runtime, check=True)
    subprocess.run(['git', 'apply', str(ROOT / 'patches/textdeformer.patch')], cwd=runtime, check=True)
    # Retrieve exact official input bytes, not potentially edited checkout copies.
    meshes = ROOT / 'meshes'
    meshes.mkdir(exist_ok=True)
    cases = json.loads((ROOT / 'configs/cases.json').read_text())
    for case in cases:
        destination = ROOT / case['mesh']
        data = git(source, 'show', 'HEAD:' + case['mesh'])
        if hashlib.sha256(data).hexdigest() != case['mesh_sha256']:
            raise ValueError(f'Mesh manifest differs from pinned upstream: {case["mesh"]}')
        if destination.exists() and destination.read_bytes() != data:
            raise FileExistsError(f'Preserve custom input: move it before preparing {destination}')
        destination.write_bytes(data)
    (meshes / 'spot.mtl').write_bytes(git(source, 'show', 'HEAD:meshes/spot.mtl'))
    build = ROOT / '.build/pointnet2_ops_lib'
    for name in git(pointnet, 'ls-tree', '-r', '--name-only', 'HEAD', 'pointnet2_ops_lib').decode().splitlines():
        destination = ROOT / '.build' / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(git(pointnet, 'show', 'HEAD:' + name))
    setup = build / 'setup.py'
    assignment = 'os.environ["TORCH_CUDA_ARCH_LIST"] = "3.7+PTX;5.0;6.0;6.1;6.2;7.0;7.5"'
    value = setup.read_text()
    if assignment not in value:
        raise ValueError('Pinned PointNet2 setup differs from the expected architecture assignment')
    setup.write_text(value.replace(assignment, '# Honor the installer TORCH_CUDA_ARCH_LIST.'))
    print('Prepared patched runtime, 7 input meshes and PointNet2 build source.')


def download_models():
    assets = json.loads((ROOT / 'configs/assets.json').read_text())['assets']
    for asset in assets:
        destination = ROOT / asset['path']
        if destination.exists():
            if digest(destination) != asset['sha256']:
                raise ValueError(f'Existing asset has a different checksum: {destination}')
            print(f'Verified: {asset["path"]}')
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + '.part')
        print(f'Downloading {asset["path"]} ({asset["size"] / 1024**3:.2f} GiB)', flush=True)
        urllib.request.urlretrieve(asset['url'], temporary)
        if digest(temporary) != asset['sha256']:
            raise ValueError(f'Checksum failed: {temporary}')
        temporary.replace(destination)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download-models', action='store_true', help='Download approximately 10 GiB of official weights')
    args = parser.parse_args()
    prepare()
    if args.download_models:
        download_models()
