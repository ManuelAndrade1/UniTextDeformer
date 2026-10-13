"""One Poisson/Jacobian optimization loop for the three Uni3D variants."""
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
import yaml
from easydict import EasyDict

from .config import ROOT
from .numerics import NumericalGuard, NUMERICAL_POLICY
from .provenance import atomic_json, cpu_tree, provenance, sha256
from .semantics import PointSemantics


def prepare_mesh(cfg, output, device):
    import kornia
    import pymeshlab
    from NeuralJacobianFields import SourceMesh
    from nvdiffmodeling.src import mesh, obj, texture

    temporary = output / 'tmp'
    temporary.mkdir()
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(cfg.mesh)
    if cfg.retriangulate:
        ms.meshing_isotropic_explicit_remeshing()
    if not ms.current_mesh().has_wedge_tex_coord():
        ms.compute_texcoord_parametrization_triangle_trivial_per_wedge(textdim=10000)
    path = str(temporary / 'mesh.obj')
    ms.save_current_mesh(path)
    loaded = mesh.unit_size(obj.load_obj(path))
    ms.add_mesh(pymeshlab.Mesh(vertex_matrix=loaded.v_pos.cpu().numpy(), face_matrix=loaded.t_pos_idx.cpu().numpy()))
    ms.save_current_mesh(path, save_vertex_color=False)
    # Retain material initialization, including its NumPy RNG consumption.
    kd = texture.create_trainable(np.random.uniform(size=[512, 512, 3], low=0.0, high=1.0), [512, 512], True)
    normal = texture.create_trainable(np.array([0, 0, 1]), [512, 512], True)
    ks = texture.create_trainable(np.array([0, 0, 0]), [512, 512], True)
    loaded = mesh.Mesh(material=dict(bsdf=cfg.bsdf, kd=kd, ks=ks, normal=normal), base=loaded)
    source = SourceMesh.SourceMesh(0, path, {}, 1, ttype=torch.float)
    source.load()
    source.to(device)
    with torch.no_grad():
        jacobians = source.jacobians_from_vertices(loaded.v_pos.unsqueeze(0))
    jacobians.requires_grad_(True)

    def blur(value):
        return texture.Texture2D(kornia.filters.gaussian_blur2d(
            value.data.permute(0, 3, 1, 2), kernel_size=(7, 7), sigma=(3, 3)
        ).permute(0, 2, 3, 1).contiguous())

    with torch.no_grad():
        material = dict(bsdf=cfg.bsdf, kd=texture.Texture2D(torch.full_like(kd.data, 0.5)),
                        ks=blur(ks), normal=blur(normal))
    return source, loaded, jacobians, material


def create_cameras(cfg):
    from utilities.camera import CameraBatch
    data = CameraBatch(cfg.train_res, [cfg.dist_min, cfg.dist_max],
        [cfg.azim_min, cfg.azim_max], [cfg.elev_alpha, cfg.elev_beta, cfg.elev_max],
        [cfg.fov_min, cfg.fov_max], cfg.aug_loc, cfg.aug_light, cfg.aug_bkg,
        cfg.batch_size, rand_solid=True)
    return torch.utils.data.DataLoader(data, cfg.batch_size, num_workers=0, pin_memory=True)


def run(cfg):
    runtime = ROOT / '.runtime/TextDeformer'
    if not (runtime / 'NeuralJacobianFields/SourceMesh.py').is_file():
        raise FileNotFoundError('Run python scripts/prepare.py before training')
    sys.path.insert(0, str(runtime))
    if not torch.cuda.is_available():
        raise RuntimeError('Training requires a working NVIDIA GPU and CUDA PyTorch')
    import nvdiffrast
    if nvdiffrast.__version__ != '0.2.8':
        raise RuntimeError('The replication protocol requires nvdiffrast 0.2.8')
    cfg = EasyDict(cfg)
    device = torch.device(f'cuda:{cfg.gpu}')
    torch.cuda.set_device(device)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)
    torch.backends.cudnn.deterministic = True
    if cfg.mesh_sha256 and sha256(cfg.mesh) != cfg.mesh_sha256:
        raise ValueError(f'Input mesh differs from the manifest: {cfg.mesh}')
    output = Path(cfg.output_path)
    output.mkdir(parents=True, exist_ok=False)
    (output / 'config.yml').write_text(yaml.safe_dump(dict(cfg), sort_keys=False))
    record = provenance(ROOT, dict(cfg), torch)
    record.update(status='running', completed_iterations=0, numerical_policy=NUMERICAL_POLICY)
    atomic_json(output / 'run.json', record)
    guard = NumericalGuard(output, cfg)
    try:
        _optimize(cfg, output, device, guard, record)
    except BaseException as error:
        record.update(status='failed', error=f'{type(error).__name__}: {error}', numerical_guard=guard.summary())
        atomic_json(output / 'run.json', record)
        raise


def _optimize(cfg, output, device, guard, record):
    from nvdiffmodeling.src import mesh, obj

    encoder = context = cameras = None
    if cfg.view_consistency:
        import clip
        import nvdiffrast.torch as dr
        from utilities.clip_spatial import CLIPVisualEncoder
        from utilities.resize_right import resize, cubic, linear, lanczos2, lanczos3
        from utilities.helpers import create_scene
        from .render import render_mesh
        from .consistency import view_consistency
        # Preserve the original model-initialization RNG sequence in the VC branch.
        unused_global, _ = clip.load(cfg.consistency_clip_model, device=device, download_root=cfg.model_cache)
        unused_global.float().eval().requires_grad_(False)
        encoder = CLIPVisualEncoder(cfg.consistency_clip_model, cfg.consistency_vit_stride, device, download_root=cfg.model_cache)
        encoder.model.float().eval().requires_grad_(False)
        context = dr.RasterizeGLContext()
        mean = torch.tensor([0.48154660, 0.45782750, 0.40821073], device=device)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device)
        interpolation = {'cubic': cubic, 'linear': linear, 'lanczos2': lanczos2, 'lanczos3': lanczos3}[cfg.resize_method]
    source, loaded, jacobians, material = prepare_mesh(cfg, output, device)
    optimizer = torch.optim.Adam([jacobians], lr=cfg.lr)
    if cfg.view_consistency:
        cameras = create_cameras(cfg)
    with torch.no_grad():
        original_vertices = source.vertices_from_jacobians(jacobians).squeeze().detach()
    point = PointSemantics(cfg, device, original_vertices, loaded.t_pos_idx, guard)
    if cfg.view_consistency:
        del unused_global
    record['guidance'] = point.guide.metadata
    torch.save(cpu_tree(point.state_dict()), output / 'references.pt')
    record['optimization_s'] = 0.0
    torch.cuda.reset_peak_memory_stats(device)
    with (output / 'steps.jsonl').open('w', buffering=1) as stream:
        for iteration in range(cfg.steps):
            guard.begin(iteration + 1)
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            vertices = source.vertices_from_jacobians(jacobians).squeeze()
            current_mesh = mesh.Mesh(vertices, loaded.t_pos_idx, material=material, base=loaded)
            consistency = torch.zeros((), device=device)
            if cfg.view_consistency:
                scene = mesh.compute_tangents(mesh.auto_normals(create_scene([current_mesh.eval()], sz=512)))
                if cfg.adapt_dist and iteration > 0:
                    with torch.no_grad():
                        centered = vertices.detach().clone()
                        centered -= (centered.amin(0) + centered.amax(0)) / 2
                        mult = torch.cat([centered.amin(0), centered.amax(0)]).abs().amax().cpu()
                        cameras.dataset.dist_min = cfg.dist_min * mult
                        cameras.dataset.dist_max = cfg.dist_max * mult
                params = {key: value.to(device) for key, value in next(iter(cameras)).items()}
                shown = scene.eval(params)
                rgb, raster = render_mesh(context, shown, params['mvp'], params['campos'],
                    params['lightpos'], cfg.light_power, cfg.train_res, spp=1, num_layers=1,
                    msaa=False, background=params['bkgs'], return_rgb_and_rast=True)
                image = resize(rgb.permute(0, 3, 1, 2), out_shape=(224, 224), interp_method=interpolation)
                # Define the camera RNG sequence: one extra initial batch and
                # one permutation at the configured interval.
                if iteration == 0:
                    next(iter(cameras))
                if iteration % cfg.camera_rng_interval == 0:
                    torch.randperm(cfg.batch_size)
            optimizer.zero_grad()
            lp, delta = point.losses(vertices, iteration)
            identity = (jacobians - torch.eye(3, device=device)).square().mean()
            if cfg.view_consistency:
                normalized = (image - mean[None, :, None, None]) / std[None, :, None, None]
                consistency = view_consistency(shown, params, raster, normalized, encoder, cfg, guard)
            total = cfg.clip_weight * lp + cfg.delta_clip_weight * delta + cfg.regularize_jacobians_weight * identity - cfg.consistency_loss_weight * consistency
            if not torch.isfinite(total):
                raise FloatingPointError('Nonfinite total loss; see numerical_events.jsonl')
            guard.optimize(total, jacobians, optimizer, source)
            torch.cuda.synchronize(device)
            record['optimization_s'] += time.perf_counter() - started
            row = dict(iteration=iteration + 1, lp=float(lp.detach()), delta_p=float(delta.detach()),
                       identity=float(identity.detach()), consistency=float(consistency.detach()),
                       total=float(total.detach()), numerical=dict(guard.current))
            stream.write(json.dumps(row, allow_nan=False) + '\n')
            record['completed_iterations'] = iteration + 1
            if (iteration + 1) % 100 == 0 or iteration + 1 == cfg.steps:
                record['numerical_guard'] = guard.summary()
                atomic_json(output / 'run.json', record)
                print(f'{cfg.case_id} {cfg.variant}: {iteration + 1}/{cfg.steps} loss={row["total"]:.6f}', flush=True)
    # Export vertices evaluated before the last Adam step.
    (output / 'mesh_final').mkdir()
    obj.write_obj(str(output / 'mesh_final'), current_mesh.eval())
    torch.save(cpu_tree(dict(jacobians=jacobians, optimizer=optimizer.state_dict(),
        completed_iterations=cfg.steps, configuration=dict(cfg), references=point.state_dict(),
        original_vertices=original_vertices, faces=loaded.t_pos_idx,
        rng=dict(python=random.getstate(), numpy=np.random.get_state(), cpu=torch.get_rng_state(),
                 cuda=torch.cuda.get_rng_state_all()))), output / 'checkpoint.pt')
    record.update(status='complete', numerical_guard=guard.summary(), quality_status=guard.quality_status(),
                  peak_allocated_gib=torch.cuda.max_memory_allocated(device) / 1024**3,
                  final_mesh_sha256=sha256(output / 'mesh_final/mesh.obj'),
                  checkpoint_sha256=sha256(output / 'checkpoint.pt'))
    atomic_json(output / 'run.json', record)
