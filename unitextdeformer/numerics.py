"""Fixed, recorded numerical policy shared by the three Uni3D variants.

Undefined directions/empty view sets have zero contribution. Invalid encoder
outputs are dropped and recorded. Invalid backward uses an analytic identity
gradient; invalid Adam/geometry updates are rolled back and recorded. No NaN is
accepted into Jacobians, Adam moments, or a saved checkpoint.
"""
from collections import Counter
import json
from pathlib import Path

import torch

NUMERICAL_POLICY = dict(
    numerical_policy='guarded_v1', normalization_epsilon=1e-6,
    gradient_clip_norm=1.0, recovery_lr_factor=0.5, minimum_lr=1e-5,
    maximum_geometry_magnitude=10000.0, minimum_geometry_extent=1e-8,
    semantic_precision='float32', empty_consistency_policy='zero',
    norm_accumulation='float64',
    backward_recovery='analytic identity gradient',
    invalid_update_policy='restore Jacobians and Adam; reduce learning rate',
)


def clone_tree(value):
    if torch.is_tensor(value):
        return value.detach().clone()
    if isinstance(value, dict):
        return {key: clone_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clone_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(clone_tree(item) for item in value)
    return value


def finite_tree(value):
    if torch.is_tensor(value):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(finite_tree(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return all(finite_tree(item) for item in value)
    return True


class NumericalGuard:
    def __init__(self, output, cfg):
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        (self.output / 'numerical_events.jsonl').touch(exist_ok=True)
        self.policy = dict(NUMERICAL_POLICY)
        self.cfg = cfg
        self.epsilon = self.policy['normalization_epsilon']
        self.counts = Counter()
        self.iteration = 0
        self.current = {}

    def begin(self, update):
        self.iteration = update
        self.current = dict(semantic_dropped=False, consistency_dropped=False,
                            backward_fallback=False, update_accepted=False)
        self.counts['attempted_iterations'] += 1

    def event(self, name, **details):
        self.counts[name] += 1
        self.output.mkdir(parents=True, exist_ok=True)
        with (self.output / 'numerical_events.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(update=self.iteration, event=name, **details), allow_nan=False) + '\n')
        if self.counts[name] <= 3 or self.counts[name] % 100 == 0:
            print(f'NUMERICAL {name} update={self.iteration} count={self.counts[name]}', flush=True)

    def normalize(self, value, name):
        value = value.float()
        if not torch.isfinite(value).all():
            raise FloatingPointError(f'Nonfinite initialization embedding: {name}')
        norm = torch.linalg.vector_norm(value.double(), dim=-1, keepdim=True)
        return (value.double() / norm.clamp_min(self.epsilon)).float()

    def semantic_losses(self, current, reference, target, text_direction):
        values = [value.float() for value in (current, reference, target, text_direction)]
        if not all(bool(torch.isfinite(value).all()) for value in values):
            self.current['semantic_dropped'] = True
            self.event('nonfinite_semantic_dropped')
            zero = current.new_zeros((), dtype=torch.float32)
            return zero, zero
        current, reference, target, text_direction = values
        absolute = -(self.normalize(current, 'current')
                     * self.normalize(target, 'target')).sum(-1).clamp(-1, 1).mean()
        delta = current.double() - reference.double()
        norm = torch.linalg.vector_norm(delta, dim=-1, keepdim=True)
        active = (norm.detach() > self.epsilon)
        direction = (delta / norm.clamp_min(self.epsilon)).float()
        directional = -(direction * self.normalize(text_direction, 'text_direction')).sum(-1)
        directional = (directional.clamp(-1, 1) * active.squeeze(-1)).mean()
        self.current['direction_inactive_rows'] = int((~active).sum())
        self.current['direction_delta_min_norm'] = float(norm.detach().min())
        if not active.all():
            self.event('near_zero_direction', rows=int((~active).sum()))
        return absolute, directional

    def consistency(self, cosines):
        values = cosines.float()[cosines != 0]
        self.current['consistency_valid_values'] = values.numel()
        if values.numel() == 0:
            self.event('empty_consistency')
            return cosines.new_zeros((), dtype=torch.float32)
        if not torch.isfinite(values).all():
            self.current['consistency_dropped'] = True
            self.event('nonfinite_consistency_dropped')
            return cosines.new_zeros((), dtype=torch.float32)
        return values.clamp(-1, 1).mean()

    def backoff(self, optimizer):
        for group in optimizer.param_groups:
            group['lr'] = max(self.policy['minimum_lr'], group['lr'] * self.policy['recovery_lr_factor'])

    def valid_geometry(self, source, jacobians):
        with torch.no_grad():
            if (not torch.isfinite(jacobians).all()
                    or jacobians.abs().amax() >= self.policy['maximum_geometry_magnitude']):
                return False
            try:
                vertices = source.vertices_from_jacobians(jacobians).reshape(-1, 3)
            except AssertionError as error:
                if 'Nan in the forward pass of the POISSON SOLVE' not in str(error):
                    raise
                return False
            if not torch.isfinite(vertices).all():
                return False
            extent = (vertices.amax(0) - vertices.amin(0)).amax()
            return bool(extent > self.policy['minimum_geometry_extent']) and bool(
                vertices.abs().amax() < self.policy['maximum_geometry_magnitude'])

    def optimize(self, loss, jacobians, optimizer, source):
        if not torch.isfinite(loss).all() or not torch.isfinite(jacobians).all():
            raise FloatingPointError('Invalid state before checked Adam update')
        try:
            loss.backward()
            gradient = jacobians.grad
        except AssertionError as error:
            # The existing Poisson autograd function raises before its NaN
            # reaches J.grad. Recover only this known numerical assertion.
            if 'Nan in the backward pass of the POISSON SOLVE' not in str(error):
                raise
            self.event('poisson_backward_nonfinite')
            gradient = None
        if gradient is None or not torch.isfinite(gradient).all():
            self.current['backward_fallback'] = True
            self.event('nonfinite_backward_identity_fallback')
            self.backoff(optimizer)
            identity = torch.eye(3, dtype=jacobians.dtype, device=jacobians.device)
            jacobians.grad = (2 * self.cfg['regularize_jacobians_weight']
                              * (jacobians.detach() - identity) / jacobians.numel())
        elif self.current.get('semantic_dropped'):
            self.backoff(optimizer)
        # FP64 norm avoids overflow while measuring a finite but very large gradient.
        norm = float(torch.linalg.vector_norm(jacobians.grad.double()))
        self.current['gradient_norm_before_clip'] = norm
        if norm > self.policy['gradient_clip_norm']:
            jacobians.grad.mul_(self.policy['gradient_clip_norm'] / norm)
            self.counts['gradient_clipped_updates'] += 1
        previous_j = jacobians.detach().clone()
        previous_adam = clone_tree(optimizer.state_dict())
        optimizer.step()
        valid = (finite_tree(jacobians) and finite_tree(optimizer.state)
                 and self.valid_geometry(source, jacobians))
        if not valid:
            with torch.no_grad():
                jacobians.copy_(previous_j)
            optimizer.load_state_dict(previous_adam)
            self.backoff(optimizer)
            optimizer.zero_grad(set_to_none=True)
            self.event('invalid_update_rolled_back')
        else:
            self.current['update_accepted'] = True
            self.counts['accepted_optimizer_updates'] += 1
        self.current['learning_rate'] = optimizer.param_groups[0]['lr']

    def summary(self):
        return dict(policy=self.policy, counts=dict(self.counts),
                    effective_optimizer_updates=self.counts['accepted_optimizer_updates'],
                    last_step=dict(self.current), event_log='numerical_events.jsonl')

    def quality_status(self):
        recovery = ['nonfinite_semantic_dropped', 'nonfinite_consistency_dropped',
                    'nonfinite_backward_identity_fallback', 'invalid_update_rolled_back']
        return 'requires_numerical_review' if any(self.counts[name] for name in recovery) else 'finite'
