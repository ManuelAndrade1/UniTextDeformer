"""Check experiment ordering, isolated selections and original baseline invocation."""
from contextlib import redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import yaml

from unitextdeformer.config import ROOT, VARIANTS
import run as experiments

spec = importlib.util.spec_from_file_location('run_textdeformer', ROOT / 'scripts/run_textdeformer.py')
baseline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(baseline)


class LauncherTests(unittest.TestCase):
    def plan(self, *arguments):
        output = io.StringIO()
        with patch.object(sys, 'argv', ['run.py', '--dry-run', *arguments]), redirect_stdout(output):
            experiments.main()
        return json.loads(output.getvalue())

    def test_default_completes_each_experiment_before_the_next(self):
        plan = self.plan()
        expected = [(variant, f'C{case:02d}') for variant in VARIANTS for case in range(2, 22)]
        self.assertEqual([(cfg['variant'], cfg['case_id']) for cfg in plan], expected)

    def test_each_experiment_can_run_independently(self):
        for variant in VARIANTS:
            plan = self.plan('--variants', variant)
            self.assertEqual(len(plan), 20)
            self.assertEqual({cfg['variant'] for cfg in plan}, {variant})
            self.assertEqual(len({cfg['output_path'] for cfg in plan}), 20)

    def test_selection_preserves_experiment_order_without_duplicates(self):
        plan = self.plan('--variants', 'uni3d-no-vc-no-delta', 'uni3d-vc', 'uni3d-vc', '--cases', 'C04', 'C02')
        self.assertEqual([(cfg['variant'], cfg['case_id']) for cfg in plan],
                         [('uni3d-vc', 'C02'), ('uni3d-vc', 'C04'),
                          ('uni3d-no-vc-no-delta', 'C02'), ('uni3d-no-vc-no-delta', 'C04')])

    def test_workers_are_waited_for_and_failure_stops_the_launch(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            events = []

            def worker(command, cwd, check):
                self.assertEqual(cwd, ROOT)
                self.assertTrue(check)
                cfg = json.loads(Path(command[-1]).read_text())
                label = (cfg['variant'], cfg['case_id'])
                events.append(('start', label))
                if len(events) == 3:
                    raise subprocess.CalledProcessError(1, command)
                events.append(('finish', label))

            with patch.object(sys, 'argv', ['run.py', '--cases', 'C02', 'C04', '--output', directory]), \
                    patch.object(experiments.subprocess, 'run', side_effect=worker), redirect_stdout(io.StringIO()):
                with self.assertRaises(subprocess.CalledProcessError):
                    experiments.main()
            self.assertEqual(events, [('start', ('uni3d-vc', 'C02')),
                                      ('finish', ('uni3d-vc', 'C02')),
                                      ('start', ('uni3d-vc', 'C04'))])

    def make_upstream(self, directory):
        repo = Path(directory) / 'original'
        repo.mkdir()
        (repo / 'main.py').write_text('# Original entry point fixture\n')
        (repo / 'example_config.yml').write_text(yaml.safe_dump(dict(seed=99, batch_size=25, train_res=512,
                                                                    consistency_vit_stride=8, clip_model='ViT-B/32')))
        return repo

    def test_baseline_only_invokes_original_entry_point_with_common_parameters(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = self.make_upstream(directory)
            original_bytes = {path.name: path.read_bytes() for path in repo.iterdir()}
            output = io.StringIO()
            with patch.object(sys, 'argv', ['run_textdeformer.py', '--repo', str(repo), '--python', '/original/bin/python',
                                           '--dry-run']), redirect_stdout(output):
                baseline.main()
            plans = json.loads(output.getvalue())
            self.assertEqual(len(plans), 20)
            self.assertEqual([plan['case_id'] for plan in plans], [f'C{i:02d}' for i in range(2, 22)])
            self.assertEqual(len({plan['output'] for plan in plans}), 20)
            for plan in plans:
                command = plan['command']
                self.assertEqual(command[:2], ['/original/bin/python', str(repo / 'main.py')])
                parameters = dict(zip(command[2::2], command[3::2]))
                self.assertEqual(parameters['--config'], str(repo / 'example_config.yml'))
                self.assertEqual(parameters['--seed'], '42')
                self.assertEqual(parameters['--epochs'], '2500')
                self.assertEqual(parameters['--batch_size'], '10')
                self.assertEqual(parameters['--consistency_vit_stride'], '16')
                self.assertNotIn('--point_checkpoint', parameters)
            self.assertEqual(original_bytes, {path.name: path.read_bytes() for path in repo.iterdir()})

    def test_baseline_waits_for_each_case_and_stops_on_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = self.make_upstream(directory)
            commands = []

            def worker(command, cwd, check):
                self.assertEqual(cwd, repo)
                self.assertTrue(check)
                commands.append(command)
                if len(commands) == 2:
                    raise subprocess.CalledProcessError(1, command)

            with patch.object(sys, 'argv', ['run_textdeformer.py', '--repo', str(repo), '--output', str(Path(directory) / 'outputs')]), \
                    patch.object(baseline.subprocess, 'run', side_effect=worker), redirect_stdout(io.StringIO()):
                with self.assertRaises(subprocess.CalledProcessError):
                    baseline.main()
            self.assertEqual(len(commands), 2)

    def test_baseline_checks_all_outputs_before_starting(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = self.make_upstream(directory)
            output = Path(directory) / 'outputs'
            (output / 'C03').mkdir(parents=True)
            with patch.object(sys, 'argv', ['run_textdeformer.py', '--repo', str(repo), '--output', str(output)]), \
                    patch.object(baseline.subprocess, 'run') as worker, redirect_stdout(io.StringIO()), \
                    patch.object(sys, 'stderr', io.StringIO()):
                with self.assertRaises(SystemExit):
                    baseline.main()
            worker.assert_not_called()


if __name__ == '__main__':
    unittest.main()
