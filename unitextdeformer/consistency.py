"""TextDeformer's patch CLIP view consistency with an empty-set guard."""
import torch


def view_consistency(final_mesh, cameras, raster, normalized_render, encoder, cfg, guard):
    from utilities.helpers import get_vp_map

    vp_map = get_vp_map(final_mesh.v_pos, cameras['mvp'], 224)
    for index, raster_faces in enumerate(raster[:, :, :, 3].view(cfg.batch_size, -1)):
        # Preserve the original visibility convention (first unique ID is background).
        visible_faces = raster_faces.unique().long()[1:] - 1
        vertices = torch.arange(len(final_mesh.v_pos), device=final_mesh.v_pos.device)
        unique, counts = torch.cat([vertices, final_mesh.t_pos_idx[visible_faces].flatten()]).unique(return_counts=True)
        vp_map[index][unique[counts < 2]] = torch.tensor([224, 224], device=vertices.device)

    median = (encoder.old_stride - 1) / 2
    vp_map[vp_map < median] = median
    vp_map[(vp_map > 224 - encoder.old_stride) & (vp_map < 224)] = 223 - median
    patches = ((vp_map - median) / encoder.new_stride).round()
    flat = patches[..., 0] * (((224 - encoder.old_stride) / encoder.new_stride) + 1) + patches[..., 1]
    features = encoder(normalized_render)
    flat[flat > features[0].shape[-1] - 1] = features[0].shape[-1]
    flat = flat.long()[:, None, :].repeat(1, features[0].shape[1], 1)
    deep = torch.nn.functional.pad(features[cfg.consistency_vit_layer], (0, 1))
    deep = torch.gather(deep, dim=2, index=flat)
    deep = torch.nn.functional.normalize(deep, dim=1, eps=1e-6)
    elevation = torch.cdist(cameras['elev'].unsqueeze(1), cameras['elev'].unsqueeze(1)).abs() < torch.deg2rad(torch.tensor(cfg.consistency_elev_filter))
    azimuth = torch.cdist(cameras['azim'].unsqueeze(1), cameras['azim'].unsqueeze(1)).abs() < torch.deg2rad(torch.tensor(cfg.consistency_azim_filter))
    cosines = torch.einsum('ijk, lkj -> ilk', deep, deep.permute(0, 2, 1))
    cosines = (cosines * azimuth.unsqueeze(-1) * elevation.unsqueeze(-1)).permute(2, 0, 1).triu(1)
    result = guard.consistency(cosines)
    encoder.featuremaps.clear()
    return result
