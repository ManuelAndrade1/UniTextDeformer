"""Check RGB, visibility and gradient parity with a deterministic CPU fixture."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]


def load_render(path, calls):
    tree = ast.parse(path.read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'render_mesh')
    raster = torch.arange(16, dtype=torch.float32).reshape(1, 2, 2, 4)
    raster[..., -1] = 1

    class Peeler:
        def __init__(self, *args):
            calls['peelers'] = calls.get('peelers', 0) + 1

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def rasterize_next_layer(self):
            return raster, torch.zeros_like(raster)

    def shade(rast, db, mesh, *args):
        calls['shading'] = calls.get('shading', 0) + 1
        return torch.cat((mesh.v_pos.sum().expand(1, 2, 2, 3), torch.ones(1, 2, 2, 1)), dim=-1)

    namespace = dict(torch=torch, render_layer=shade,
        ru=SimpleNamespace(xfm_points=lambda vertices, transform: vertices),
        dr=SimpleNamespace(DepthPeeler=Peeler, antialias=lambda color, *args: color))
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['render_mesh']


class VisibilityCandidate(unittest.TestCase):
    def setUp(self):
        if not self.original.is_file():
            self.skipTest('Run scripts/prepare.py for upstream renderer parity checks')

    def run_path(self, path, visibility=False, layers=1):
        calls = {}
        function = load_render(path, calls)
        vertices = torch.ones(3, 3, requires_grad=True)
        mesh = SimpleNamespace(v_pos=vertices, t_pos_idx=torch.tensor([[0, 1, 2]]))
        output = function(None, mesh, torch.eye(4).unsqueeze(0), torch.zeros(1, 3),
            torch.zeros(1, 3), torch.ones(1, 3), 2, background=torch.zeros(1, 2, 2, 3),
            return_rast_map=visibility, num_layers=layers)
        return output, vertices, calls

    @property
    def original(self):
        return ROOT / '.runtime/TextDeformer/nvdiffmodeling/src/render.py'

    @property
    def candidate(self):
        return ROOT / 'unitextdeformer/render.py'

    def test_visibility_is_identical_and_skips_shading(self):
        before, _, old_calls = self.run_path(self.original, visibility=True)
        after, _, new_calls = self.run_path(self.candidate, visibility=True)
        self.assertTrue(torch.equal(before, after))
        self.assertFalse(after.requires_grad)
        self.assertEqual(old_calls['shading'], 1)
        self.assertEqual(new_calls.get('shading', 0), 0)
        self.assertEqual(new_calls['peelers'], 1)

    def test_rgb_and_vertex_gradients_are_identical(self):
        before, old_vertices, old_calls = self.run_path(self.original)
        after, new_vertices, new_calls = self.run_path(self.candidate)
        self.assertTrue(torch.equal(before, after))
        before.sum().backward()
        after.sum().backward()
        self.assertTrue(torch.equal(old_vertices.grad, new_vertices.grad))
        self.assertEqual(old_calls, new_calls)

    def test_combined_rgb_visibility_uses_one_pass_and_preserves_gradient(self):
        calls = {}
        function = load_render(self.candidate, calls)
        vertices = torch.ones(3, 3, requires_grad=True)
        mesh = SimpleNamespace(v_pos=vertices, t_pos_idx=torch.tensor([[0, 1, 2]]))
        rgb, raster = function(None, mesh, torch.eye(4).unsqueeze(0), torch.zeros(1, 3),
            torch.zeros(1, 3), torch.ones(1, 3), 2,
            background=torch.zeros(1, 2, 2, 3), return_rgb_and_rast=True)
        expected, original_vertices, _ = self.run_path(self.original)
        self.assertTrue(torch.equal(rgb, expected))
        rgb.sum().backward()
        expected.sum().backward()
        self.assertTrue(torch.equal(vertices.grad, original_vertices.grad))
        self.assertFalse(raster.requires_grad)
        self.assertEqual(calls, dict(peelers=1, shading=1))

    def test_visibility_still_rejects_multiple_layers(self):
        with self.assertRaises(ValueError):
            self.run_path(self.candidate, visibility=True, layers=2)


if __name__ == '__main__':
    unittest.main()
