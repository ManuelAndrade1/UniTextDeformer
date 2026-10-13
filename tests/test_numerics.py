"""Regression/fault-injection tests for the exact guarded math and Adam path."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from unitextdeformer.numerics import NumericalGuard, finite_tree


class StableNumerics(unittest.TestCase):
    def make_fixture(self, folder, optimizer_class=torch.optim.Adam):
        j = torch.eye(3).reshape(1, 1, 3, 3).clone().requires_grad_()
        vertices = torch.tensor([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
        source = SimpleNamespace(vertices_from_jacobians=lambda value: (vertices @ value[0, 0].T).unsqueeze(0))
        optimizer = optimizer_class([j], lr=.0025)
        guard = NumericalGuard(folder, {'regularize_jacobians_weight': .5})
        guard.begin(1)
        return j, optimizer, source, guard

    def test_reported_empty_consistency_nan_is_reproduced_and_fixed(self):
        with tempfile.TemporaryDirectory() as folder:
            guard = NumericalGuard(folder, {})
            cosines = torch.zeros(4, 10, 10, requires_grad=True)
            self.assertTrue(torch.isnan(cosines[cosines != 0].mean()))
            with contextlib.redirect_stdout(io.StringIO()):
                result = guard.consistency(cosines)
            self.assertEqual(result.item(), 0)
            self.assertFalse(result.requires_grad)
            self.assertEqual(guard.counts['empty_consistency'], 1)

    def test_identical_shape_has_finite_zero_direction_and_live_absolute_gradient(self):
        with tempfile.TemporaryDirectory() as folder:
            guard = NumericalGuard(folder, {})
            for dtype in [torch.float16, torch.float32]:
                current = torch.tensor([[.5, .3, .1]], dtype=dtype, requires_grad=True)
                reference = current.detach().clone()
                self.assertTrue(torch.isnan((current - reference) / (current - reference).norm()).all())
                with contextlib.redirect_stdout(io.StringIO()):
                    lp, ld = guard.semantic_losses(current, reference, torch.tensor([[0., 1., 0.]]), torch.tensor([[1., 0., 0.]]))
                self.assertEqual(ld.item(), 0)
                self.assertTrue(torch.equal(torch.autograd.grad(ld, current, retain_graph=True)[0], torch.zeros_like(current)))
                grad = torch.autograd.grad(lp, current)[0]
                self.assertTrue(torch.isfinite(grad).all())
                self.assertGreater(grad.norm(), 0)

    def test_finite_nonzero_semantics_agree_with_original_formula(self):
        with tempfile.TemporaryDirectory() as folder:
            guard = NumericalGuard(folder, {})
            torch.manual_seed(42)
            current = torch.randn(10, 8, requires_grad=True)
            reference, target, text_direction = [torch.randn(10, 8) for _ in range(3)]
            lp, ld = guard.semantic_losses(current, reference, target, text_direction)
            cosine = torch.nn.CosineSimilarity()
            expected_lp = -cosine(current / current.norm(dim=-1, keepdim=True), target / target.norm(dim=-1, keepdim=True)).mean()
            delta = current - reference
            expected_ld = -cosine(delta / delta.norm(dim=-1, keepdim=True), text_direction / text_direction.norm(dim=-1, keepdim=True)).mean()
            self.assertTrue(torch.allclose(lp, expected_lp, atol=1e-6))
            self.assertTrue(torch.allclose(ld, expected_ld, atol=1e-6))
            actual = torch.autograd.grad(lp + ld, current, retain_graph=True)[0]
            expected = torch.autograd.grad(expected_lp + expected_ld, current)[0]
            self.assertTrue(torch.allclose(actual, expected, atol=1e-6))

    def test_huge_finite_embeddings_do_not_overflow_norm_or_json(self):
        with tempfile.TemporaryDirectory() as folder:
            guard = NumericalGuard(folder, {})
            current = torch.tensor([[1e38, 1e38, 1e38]], requires_grad=True)
            lp, ld = guard.semantic_losses(current, -current.detach(), torch.ones(1, 3), torch.tensor([[.1, .2, .9]]))
            grad = torch.autograd.grad(lp + ld, current)[0]
            self.assertTrue(torch.isfinite(grad).all())
            self.assertTrue(torch.isfinite(lp + ld))
            json.dumps(guard.current, allow_nan=False)

    def test_nonfinite_encoder_or_consistency_is_dropped_and_recorded(self):
        with tempfile.TemporaryDirectory() as folder:
            guard = NumericalGuard(folder, {})
            with contextlib.redirect_stdout(io.StringIO()):
                lp, ld = guard.semantic_losses(torch.full((1, 3), float('nan')), *[torch.ones(1, 3)] * 3)
                consistency = guard.consistency(torch.tensor([float('nan'), .5]))
            self.assertEqual(float(lp + ld + consistency), 0)
            self.assertTrue(guard.current['semantic_dropped'])
            self.assertTrue(guard.current['consistency_dropped'])
            events = [json.loads(line)['event'] for line in (Path(folder) / 'numerical_events.jsonl').read_text().splitlines()]
            self.assertEqual(events, ['nonfinite_semantic_dropped', 'nonfinite_consistency_dropped'])

    def test_nonfinite_backward_uses_identity_and_preserves_finite_adam(self):
        with tempfile.TemporaryDirectory() as folder:
            j, optimizer, source, guard = self.make_fixture(folder)
            with torch.no_grad():
                j.add_(.1)
            bad_loss = (j - j.detach()).sum().sqrt()
            self.assertEqual(float(bad_loss.detach()), 0)
            with contextlib.redirect_stdout(io.StringIO()):
                guard.optimize(bad_loss, j, optimizer, source)
            self.assertTrue(guard.current['backward_fallback'])
            self.assertTrue(guard.current['update_accepted'])
            self.assertTrue(finite_tree(optimizer.state))
            self.assertTrue(torch.isfinite(j).all())
            self.assertEqual(optimizer.param_groups[0]['lr'], .00125)

    def test_huge_gradient_is_clipped_before_adam(self):
        with tempfile.TemporaryDirectory() as folder:
            j, optimizer, source, guard = self.make_fixture(folder)
            guard.optimize((j * 1e30).sum(), j, optimizer, source)
            self.assertLessEqual(float(j.grad.double().norm()), 1.000001)
            self.assertTrue(finite_tree(optimizer.state))
            self.assertEqual(guard.counts['gradient_clipped_updates'], 1)

    def test_corrupt_adam_update_restores_both_parameters_and_moments(self):
        class CorruptOnce(torch.optim.Adam):
            def step(self, closure=None):
                result = super().step(closure)
                if not getattr(self, 'corrupted', False):
                    self.corrupted = True
                    with torch.no_grad():
                        self.param_groups[0]['params'][0].fill_(float('nan'))
                        next(iter(self.state.values()))['exp_avg'].fill_(float('inf'))
                return result
        with tempfile.TemporaryDirectory() as folder:
            j, optimizer, source, guard = self.make_fixture(folder, CorruptOnce)
            before = j.detach().clone()
            with contextlib.redirect_stdout(io.StringIO()):
                guard.optimize(j.square().mean(), j, optimizer, source)
            self.assertTrue(torch.equal(j, before))
            self.assertFalse(guard.current['update_accepted'])
            self.assertFalse(optimizer.state)
            optimizer.zero_grad()
            guard.begin(2)
            guard.optimize(j.square().mean(), j, optimizer, source)
            self.assertTrue(guard.current['update_accepted'])
            self.assertTrue(finite_tree(optimizer.state))

    def test_finite_collapsed_geometry_is_rejected(self):
        class Collapse(torch.optim.Adam):
            def step(self, closure=None):
                super().step(closure)
                with torch.no_grad():
                    self.param_groups[0]['params'][0].zero_()
        with tempfile.TemporaryDirectory() as folder:
            j, optimizer, source, guard = self.make_fixture(folder, Collapse)
            before = j.detach().clone()
            with contextlib.redirect_stdout(io.StringIO()):
                guard.optimize(j.square().mean(), j, optimizer, source)
            self.assertTrue(torch.equal(j, before))
            self.assertEqual(guard.counts['invalid_update_rolled_back'], 1)

    def test_candidate_poisson_nan_assertion_rolls_back_instead_of_stopping(self):
        with tempfile.TemporaryDirectory() as folder:
            j, optimizer, source, guard = self.make_fixture(folder)
            before = j.detach().clone()
            def invalid_solve(value):
                raise AssertionError('Nan in the forward pass of the POISSON SOLVE')
            source.vertices_from_jacobians = invalid_solve
            with contextlib.redirect_stdout(io.StringIO()):
                guard.optimize(j.square().mean(), j, optimizer, source)
            self.assertTrue(torch.equal(j, before))
            self.assertEqual(guard.counts['invalid_update_rolled_back'], 1)
            self.assertFalse(optimizer.state)


if __name__ == "__main__":
    unittest.main()
