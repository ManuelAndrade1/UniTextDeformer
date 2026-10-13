import hashlib
import torch
from torch import nn


class SurfaceSampler(nn.Module):
    """Fixed area-weighted face selection; live barycentric vertex interpolation."""
    def __init__(self, vertices, faces, count, seed=42, generator=None):
        super().__init__()
        if count <= 0:
            raise ValueError('point_count must be positive')
        with torch.no_grad():
            triangles = vertices.detach().cpu()[faces.detach().cpu().long()]
            areas = torch.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0], dim=-1).norm(dim=-1)
            if not torch.isfinite(areas).all() or areas.sum() <= 0:
                raise ValueError('Source mesh has no finite positive surface area')
            rng = generator if generator is not None else torch.Generator(device='cpu').manual_seed(seed)
            selected = torch.multinomial(areas, count, replacement=True, generator=rng)
            uv = torch.rand(count, 2, generator=rng)
            root = uv[:, :1].sqrt()
            bary = torch.cat((1 - root, root * (1 - uv[:, 1:]), root * uv[:, 1:]), dim=-1)
            indices = faces.detach().cpu().long()[selected]
        self.register_buffer('indices', indices.to(vertices.device))
        self.register_buffer('barycentric', bary.to(vertices.device, vertices.dtype))

    def forward(self, vertices):
        points = (vertices[self.indices] * self.barycentric[..., None]).sum(dim=1)
        centered = points - points.mean(dim=0, keepdim=True)
        radius = centered.norm(dim=-1).amax().clamp_min(1e-6)
        return (centered / radius).unsqueeze(0)

    def fingerprint(self):
        h = hashlib.sha256()
        for value in (self.indices, self.barycentric):
            h.update(value.detach().cpu().numpy().tobytes())
        return h.hexdigest()

