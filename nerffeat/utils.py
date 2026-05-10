import json
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import torch
from torch.nn import functional as F


def visualize_point_cloud(points):
    """Visualize a point cloud with Open3D (interactive window)."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.visualization.draw_geometries([pcd])


def rotation_matrix_from_euler(euler_angles):
    x, y, z = euler_angles
    rot_x = np.array([[1.0, 0.0, 0.0], [0.0, np.cos(x), -np.sin(x)], [0.0, np.sin(x), np.cos(x)]])
    rot_y = np.array([[np.cos(y), 0.0, np.sin(y)], [0.0, 1.0, 0.0], [-np.sin(y), 0.0, np.cos(y)]])
    rot_z = np.array([[np.cos(z), -np.sin(z), 0.0], [np.sin(z), np.cos(z), 0.0], [0.0, 0.0, 1.0]])
    return rot_z @ rot_y @ rot_x


def load_bop_pose(path, occurrence_id):
    p = Path(path)
    scene_dir = p.parent.parent
    image_id = str(int(p.stem))
    with open(scene_dir / "scene_gt.json", encoding="utf-8") as f:
        scene_gt = json.load(f)
    rotation = np.asarray(scene_gt[image_id][occurrence_id]["cam_R_m2c"]).reshape(3, 3)
    translation = np.asarray(scene_gt[image_id][occurrence_id]["cam_t_m2c"])
    return rotation, translation


def huber(predictions, targets, scaling=0.1):
    diff_sq = (predictions - targets) ** 2
    return ((1 + diff_sq / (scaling**2)).clamp(1e-4).sqrt() - 1) * float(scaling)


def sample_images_at_mc_locs(target_images, screen_xy):
    batch_size = target_images.shape[0]
    channels = target_images.shape[-1]
    spatial_size = screen_xy.shape[1:-1]
    sampled = F.grid_sample(
        target_images.permute(0, 3, 1, 2),
        -screen_xy.view(batch_size, -1, 1, 2),
        align_corners=True,
        mode="nearest",
    )
    return sampled.permute(0, 2, 3, 1).view(batch_size, *spatial_size, channels)


def embedding_to_rgb(feature_map: torch.Tensor, mask: torch.Tensor = None, demean: bool = False):
    last_dim = feature_map.shape[-1]
    if demean:
        center = (
            feature_map[mask].view(-1, last_dim).mean(dim=0)
            if mask is not None
            else feature_map.view(-1, last_dim).mean(dim=0)
        )
        feature_map = feature_map - center
    shape = feature_map.shape[:-1]
    feature_map = feature_map.view(*shape, 3, -1).mean(dim=-1)
    if mask is not None:
        feature_map[~mask] = 0.0
    feature_map = feature_map / (torch.abs(feature_map).max() + 1e-9)
    return feature_map.mul(0.5).add(0.5)


def normalize_image_for_preview(image):
    image = image / (torch.abs(image).max() + 1e-9)
    return image.mul(0.5).add(0.5)


def save_render_preview(
    field,
    camera,
    output_dir,
    renderer_grid,
    normalize_features=False,
    preview_id="0",
    filename_suffix="_nerf",
):
    out = Path(output_dir)
    with torch.no_grad():
        rendered_image_silhouette, _, _ = renderer_grid(
            cameras=camera, volumetric_function=field.batched_forward
        )
        last_dim = rendered_image_silhouette.shape[-1]
        rendered_image, _ = rendered_image_silhouette[0].split([last_dim - 1, 1], dim=-1)

    if last_dim >= 13:
        if normalize_features:
            features = (rendered_image[:, :, 0:12] - 0.5) * 2
            rendered_image[:, :, 0:12] = features / (
                0.01 + torch.norm(features, dim=2).unsqueeze(2)
            )
        cv2.imwrite(
            str(out / f"{preview_id}_rendered_features.jpg"),
            embedding_to_rgb(rendered_image[:, :, 0:12].cpu().detach(), demean=True).cpu().numpy() * 255,
        )
        for idx, start in enumerate(range(0, min(12, rendered_image.shape[-1]), 3), start=1):
            cv2.imwrite(
                str(out / f"{preview_id}_feat_channel{idx}.jpg"),
                rendered_image[:, :, start : start + 3].detach().cpu().numpy() * 255,
            )
    elif last_dim > 4:
        cv2.imwrite(str(out / f"{preview_id}.jpg"), rendered_image[:, :, 0:3].cpu().numpy() * 255)
        cv2.imwrite(str(out / f"{preview_id}_unray.jpg"), rendered_image[:, :, 3:6].cpu().numpy() * 255)
    else:
        cv2.imwrite(str(out / f"{preview_id}{filename_suffix}.jpg"), rendered_image.cpu().numpy() * 255)


def contrastive_loss_with_negatives(queries, pos_keys, neg_keys):
    """InfoNCE-style contrastive loss with one positive key and a bank of negatives."""
    sim_pos = (pos_keys * queries).sum(dim=-1, keepdim=True)
    sim_neg = queries @ neg_keys.permute(0, 2, 1)
    logits = torch.cat((sim_pos, sim_neg), dim=-1).permute(0, 2, 1)
    target = torch.zeros(sim_pos.shape[0], sim_pos.shape[1], dtype=torch.long, device=queries.device)
    return F.cross_entropy(logits, target)
