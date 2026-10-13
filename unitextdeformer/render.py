# Copyright (c) 2020-2021, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

# See licenses/NVIDIA.txt for the full license.
"""Original shading/compositing with single-pass RGB plus visibility output."""
import torch
from nvdiffmodeling.src.render import dr, ru, util, render_layer


def render_mesh(ctx, mesh, mtx_in, view_pos, light_pos, light_power, resolution,
                spp=1, num_layers=1, msaa=False, background=None, antialias=True,
                min_roughness=0.08, return_rast_map=False, return_rgb_and_rast=False):
    if (return_rast_map or return_rgb_and_rast) and num_layers != 1:
        raise ValueError('Visibility output requires one rasterized layer')
    device = mesh.v_pos.device

    def prepare(x):
        x = torch.tensor(x, dtype=torch.float32, device=device) if not torch.is_tensor(x) else x
        return x[:, None, None, :] if x.ndim == 2 else x

    mtx_in = torch.tensor(mtx_in, dtype=torch.float32, device=device) if not torch.is_tensor(mtx_in) else mtx_in
    light_pos, light_power, view_pos = prepare(light_pos), prepare(light_power), prepare(view_pos)
    full_res = resolution * spp
    v_pos_clip = ru.xfm_points(mesh.v_pos[None, ...], mtx_in)
    layers = []
    with dr.DepthPeeler(ctx, v_pos_clip, mesh.t_pos_idx.int(), [full_res, full_res]) as peeler:
        for _ in range(num_layers):
            rast, db = peeler.rasterize_next_layer()
            if not return_rast_map:
                layers.append((render_layer(rast, db, mesh, view_pos, light_pos, light_power,
                    resolution, min_roughness, spp, msaa), rast))
    if return_rast_map:
        return rast.detach()
    if background is not None:
        assert background.shape[1] == resolution and background.shape[2] == resolution
        if spp > 1:
            background = util.scale_img_nhwc(background, [full_res, full_res], mag='nearest', min='nearest')
        accumulated = background
    else:
        accumulated = torch.zeros((1, full_res, full_res, 3), dtype=torch.float32, device=device)
    for color, raster in reversed(layers):
        alpha = (raster[..., -1:] > 0) * color[..., 3:4]
        accumulated = torch.lerp(accumulated, color[..., :3], alpha)
        if antialias:
            accumulated = dr.antialias(accumulated.contiguous(), raster, v_pos_clip, mesh.t_pos_idx.int())
    rgb = util.avg_pool_nhwc(accumulated, spp) if spp > 1 else accumulated
    return (rgb, rast.detach()) if return_rgb_and_rast else rgb
