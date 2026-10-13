#!/usr/bin/env python3
"""Run the case manifest sequentially with the original TextDeformer main.py."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=ROOT / 'external/TextDeformer',
                        help='Checkout of the original TextDeformer repository')
    parser.add_argument('--python', default=sys.executable, help='Python from the original TextDeformer environment')
    parser.add_argument('--manifest', type=Path, default=ROOT / 'configs/cases.json')
    parser.add_argument('--cases', nargs='+', help='Case IDs; omit for all 20 cases')
    parser.add_argument('--steps', type=int, default=2500)
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/textdeformer')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if args.steps < 1:
        parser.error('--steps must be positive')
    repo = args.repo.resolve()
    if not (repo / 'main.py').is_file():
        parser.error('--repo must contain the original TextDeformer main.py')
    upstream_config = repo / 'example_config.yml'
    upstream = yaml.safe_load(upstream_config.read_text())
    protocol = yaml.safe_load((ROOT / 'configs/experiment.yml').read_text())
    cases = json.loads(args.manifest.read_text())
    if not cases or len({case['id'] for case in cases}) != len(cases):
        parser.error('The manifest must contain nonempty, unique case IDs')
    if args.cases:
        unknown = set(args.cases) - {case['id'] for case in cases}
        if unknown:
            parser.error(f'Unknown cases: {sorted(unknown)}')
        cases = [case for case in cases if case['id'] in args.cases]
    output = args.output.resolve()
    plans = []
    for case in cases:
        if Path(case['id']).name != case['id'] or case['id'] in {'.', '..'}:
            parser.error(f'Invalid case ID: {case["id"]}')
        destination = output / case['id']
        parameters = {key: value for key, value in protocol.items() if key in upstream}
        parameters.update(mesh=str((ROOT / case['mesh']).resolve()),
                          text_prompt=case['text_prompt'], base_text_prompt=case['base_text_prompt'],
                          epochs=args.steps, output_path=str(destination), log_interval_im=protocol['camera_rng_interval'])
        command = [args.python, str(repo / 'main.py'), '--config', str(upstream_config)]
        for key, value in parameters.items():
            command.extend([f'--{key}', str(value)])
        plans.append(dict(case_id=case['id'], output=str(destination), command=command))
    if args.dry_run:
        print(json.dumps(plans, indent=2))
        return
    for plan in plans:
        if Path(plan['output']).exists():
            parser.error(f'Output already exists: {plan["output"]}. Choose a new --output.')
    for plan in plans:
        print(f'TextDeformer {plan["case_id"]}', flush=True)
        subprocess.run(plan['command'], cwd=repo, check=True)
    print(f'Finished {len(plans)} TextDeformer runs.', flush=True)


if __name__ == '__main__':
    main()
