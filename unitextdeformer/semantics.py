"""Paired, differentiable surface observations and frozen Uni3D guidance."""
from contextlib import contextmanager
import random

import numpy as np
import torch

from .sampling import SurfaceSampler
from .uni3d import load_uni3d


@contextmanager
def isolated_rng(device):
    python_state, numpy_state = random.getstate(), np.random.get_state()
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []
    try:
        with torch.random.fork_rng(devices=devices):
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


class PointSemantics:
    def __init__(self, cfg, device, vertices, faces, guard):
        self.guard = guard
        with isolated_rng(device):
            generator = torch.Generator(device='cpu').manual_seed(cfg.seed)
            self.sample_a = SurfaceSampler(vertices, faces, cfg.point_count, generator=generator)
            self.sample_b = SurfaceSampler(vertices, faces, cfg.point_count, generator=generator)
            self.guide = load_uni3d(cfg, device)
            with torch.no_grad():
                self.reference_a = self.guide(self.sample_a(vertices)).detach()
                self.reference_b = self.guide(self.sample_b(vertices)).detach()
                self.target = guard.normalize(self.guide.target_text, 'point_target_text')
                self.text_direction = guard.normalize(self.guide.target_text - self.guide.base_text, 'point_text_direction')

    def losses(self, vertices, iteration):
        current = self.guide(self.sample_a(vertices))
        reference = self.reference_b if iteration == 0 else self.reference_a
        # Record DeltaP even when its contribution to the objective is zero.
        return self.guard.semantic_losses(current, reference, self.target, self.text_direction)

    def state_dict(self):
        return dict(sample_a=self.sample_a.state_dict(), sample_b=self.sample_b.state_dict(),
                    reference_a=self.reference_a, reference_b=self.reference_b,
                    target=self.target, text_direction=self.text_direction,
                    metadata=self.guide.metadata)
