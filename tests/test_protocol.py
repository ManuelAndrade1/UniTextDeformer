"""CPU checks for campaign composition, input identity and live gradients."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from unitextdeformer.config import ROOT, VARIANTS, load_base, load_cases, resolve_config, repository_path
from unitextdeformer.sampling import SurfaceSampler
from unitextdeformer.semantics import PointSemantics
from unitextdeformer.numerics import NumericalGuard
from unitextdeformer.provenance import sha256


class ProtocolTests(unittest.TestCase):
    def test_sixty_runs_twenty_cases_seven_meshes_and_three_objectives(self):
        cases = load_cases(ROOT / 'configs/cases.json')
        self.assertEqual([c['id'] for c in cases], [f'C{i:02d}' for i in range(2, 22)])
        self.assertEqual(len({c['mesh'] for c in cases}), 7)
        cfg = load_base(ROOT / 'configs/experiment.yml')
        jobs = [resolve_config(cfg, case, variant, 'outputs/experiment') for case in cases for variant in VARIANTS]
        self.assertEqual(len(jobs), 60)
        for job in jobs:
            self.assertEqual((job['steps'], job['seed'], job['point_count'], job['train_res']), (2500, 42, 10000, 512))
            self.assertEqual(job['training_camera_count'], 10 if job['variant'] == 'uni3d-vc' else 0)
            self.assertEqual(job['delta_clip_weight'], 0 if job['variant'].endswith('no-delta') else 1)

    def test_prepared_meshes_match_manifest(self):
        if not (ROOT / 'meshes').exists():
            self.skipTest('Run scripts/prepare.py to check the downloaded inputs')
        for case in load_cases(ROOT / 'configs/cases.json'):
            self.assertEqual(sha256(ROOT / case['mesh']), case['mesh_sha256'])

    def test_configuration_paths_are_relative_and_contained(self):
        case = load_cases(ROOT / 'configs/cases.json')[0]
        cfg = resolve_config(load_base(ROOT / 'configs/experiment.yml'), case, 'uni3d-vc', ROOT / 'outputs/test')
        for key in ('mesh', 'output_path', 'point_checkpoint', 'text_checkpoint', 'text_config', 'model_cache'):
            self.assertFalse(Path(cfg[key]).is_absolute())
            self.assertNotIn('..', Path(cfg[key]).parts)
        with self.assertRaises(ValueError):
            repository_path(ROOT.parent / 'outside')

    def test_optional_qualitative_case_uses_an_included_mesh(self):
        cases = load_cases(ROOT / 'configs/qualitative.json')
        self.assertEqual([(c['base_text_prompt'], c['text_prompt']) for c in cases], [('a cow', 'a giraffe')])
        self.assertEqual(sha256(ROOT / cases[0]['mesh']), cases[0]['mesh_sha256'])

    def test_unknown_or_unsafe_case_ids_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'cases.json'
            path.write_text(json.dumps([dict(id='../escape', mesh='mesh.obj', text_prompt='a target', base_text_prompt='a base')]))
            with self.assertRaises(ValueError):
                load_cases(path)

    def test_sampling_replays_seed_and_backpropagates_to_live_vertices(self):
        v = torch.tensor([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]], requires_grad=True)
        f = torch.tensor([[0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 3, 2]])
        a, b = SurfaceSampler(v, f, 64, seed=42), SurfaceSampler(v, f, 64, seed=42)
        self.assertEqual(a.fingerprint(), b.fingerprint())
        points = a(v)
        self.assertTrue(torch.allclose(points.mean(1), torch.zeros(1, 3), atol=1e-6))
        self.assertAlmostEqual(float(points.norm(dim=-1).amax()), 1.0, places=6)
        points.square().mean().backward()
        self.assertTrue(torch.isfinite(v.grad).all())
        self.assertGreater(float(v.grad.norm()), 0)

    def test_point_reference_policy_and_rng_isolation(self):
        from easydict import EasyDict
        class Guide(torch.nn.Module):
            target_text = torch.tensor([[.2, .8, .4]])
            base_text = torch.tensor([[.6, .1, .1]])
            metadata = {}
            def forward(self, points):
                return points.square().mean(1)
        vertices = torch.tensor([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]], requires_grad=True)
        faces = torch.tensor([[0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 3, 2]])
        with tempfile.TemporaryDirectory() as directory:
            guard = NumericalGuard(directory, {})
            state = torch.get_rng_state()
            with patch('unitextdeformer.semantics.load_uni3d', return_value=Guide()):
                point = PointSemantics(EasyDict(seed=42, point_count=64), torch.device('cpu'), vertices, faces, guard)
            self.assertTrue(torch.equal(state, torch.get_rng_state()))
            self.assertNotEqual(point.sample_a.fingerprint(), point.sample_b.fingerprint())
            lp, delta0 = point.losses(vertices, 0)
            _, delta1 = point.losses(vertices, 1)
            self.assertNotEqual(float(delta0), 0)
            self.assertEqual(float(delta1), 0)
            lp.backward()
            self.assertGreater(float(vertices.grad.norm()), 0)


if __name__ == '__main__':
    unittest.main()
