#!/usr/bin/env python3
"""Run complete experiments in order, with one independent process at a time."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from unitextdeformer.config import ROOT, VARIANTS, load_base, load_cases, resolve_config, repository_path
from unitextdeformer.provenance import sha256


def completed(output, cfg):
    try:
        value = json.loads((output / 'run.json').read_text())
        return (value['status'] == 'complete' and value['configuration'] == cfg
                and value['completed_iterations'] == cfg['steps']
                and value['final_mesh_sha256'] == sha256(output / 'mesh_final/mesh.obj')
                and value['checkpoint_sha256'] == sha256(output / 'checkpoint.pt')
                and all(sha256(ROOT / name) == digest for name, digest in value['source_sha256'].items()))
    except (OSError, ValueError, KeyError):
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/experiment.yml')
    parser.add_argument('--manifest', type=Path, default=ROOT / 'configs/cases.json')
    parser.add_argument('--cases', nargs='+', help='Case IDs; omit for all 20 cases')
    parser.add_argument('--variants', nargs='+', choices=tuple(VARIANTS), default=list(VARIANTS),
                        help='Select experiments; each completes all cases before the next starts')
    parser.add_argument('--steps', type=int, help='Override 2500 for a smoke test or a new experiment')
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/experiment')
    parser.add_argument('--dry-run', action='store_true', help='Print resolved configurations without importing PyTorch')
    parser.add_argument('--skip-complete', action='store_true', help='Skip verified complete runs with identical configuration')
    parser.add_argument('--worker-config', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker_config:
        from unitextdeformer.pipeline import run
        import os
        os.chdir(ROOT)
        run(json.loads((ROOT / repository_path(args.worker_config)).read_text()))
        return
    base = load_base(ROOT / repository_path(args.config))
    cases = load_cases(ROOT / repository_path(args.manifest))
    output_root = ROOT / repository_path(args.output)
    if args.cases:
        unknown = set(args.cases) - {c['id'] for c in cases}
        if unknown:
            parser.error(f'Unknown cases: {sorted(unknown)}')
        cases = [c for c in cases if c['id'] in args.cases]
    plans = []
    for variant in VARIANTS:
        if variant not in args.variants:
            continue
        for case in cases:
            output = output_root / case['id'] / variant
            cfg = resolve_config(base, case, variant, output, args.steps)
            plans.append((output, cfg))
    if args.dry_run:
        print(json.dumps([cfg for _, cfg in plans], indent=2))
        return
    # Check the whole selection before starting; never overwrite prior attempts.
    jobs = []
    for output, cfg in plans:
        if output.exists():
            if args.skip_complete and completed(output, cfg):
                print(f'Skip complete: {cfg["case_id"]} {cfg["variant"]}', flush=True)
                continue
            parser.error(f'Output already exists: {output}. Choose a new --output or use --skip-complete for complete runs.')
        jobs.append((output, cfg))
    configs = output_root / 'configs'
    configs.mkdir(parents=True, exist_ok=True)
    for output, cfg in jobs:
        path = configs / f'{cfg["case_id"]}_{cfg["variant"]}.json'
        path.write_text(json.dumps(cfg, indent=2) + '\n')
        subprocess.run([sys.executable, str(ROOT / 'run.py'), '--worker-config', str(path)], cwd=ROOT, check=True)
    print(f'Finished {len(jobs)} runs.', flush=True)


if __name__ == '__main__':
    main()
