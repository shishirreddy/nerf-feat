
import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import open3d as o3d
import torch
from pytorch3d.ops import sample_farthest_points as fps
from sklearn.neighbors import KDTree

from nerffeat.artifacts import save_numpy
from nerffeat.data.bop import make_batch_cameras

LOGGER = logging.getLogger(__name__)


def configure_radiance_field(field: torch.nn.Module, mode: str) -> torch.nn.Module:
    if mode == "color":
        branch_trainability = {
            "mlp": True,
            "harmonic_embedding": True,
            "density_layer": True,
            "color_layer": True,
            "feature_layer": False,
        }
    elif mode == "feature":
        branch_trainability = {
            "mlp": False,
            "harmonic_embedding": False,
            "density_layer": False,
            "color_layer": False,
            "feature_layer": True,
        }
    else:
        raise ValueError(f"Unsupported radiance-field mode: {mode}")

    for branch_name, trainable in branch_trainability.items():
        getattr(field, branch_name).requires_grad_(trainable)
    field.mode = mode
    return field


def load_model_checkpoint(model: torch.nn.Module, path) -> dict:
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["model_state_dict"])
    LOGGER.info("loaded checkpoint step=%d from %s", checkpoint.get("step", -1), path)
    return checkpoint


def load_or_create_negative_surface_points(
    *,
    cache_path: Path,
    field: torch.nn.Module,
    renderers,
    cameras,
    silhouettes: torch.Tensor,
    image_size: int,
    device: torch.device,
    renderer_config: dict,
    cache_enabled: bool,
    mask_rays: bool,
) -> torch.Tensor:
    if cache_path.exists():
        return torch.from_numpy(np.load(cache_path).astype("float32")).to(device).unsqueeze(0)

    camera_indices = torch.randperm(len(cameras))
    camera_batch = make_batch_cameras(
        cameras,
        camera_indices,
        image_size,
        device=device,
    )
    sampling_passes = renderer_config["negative_sampling_passes"] if cache_enabled else 1

    negative_points = torch.Tensor([])
    for _pass_idx in range(sampling_passes):
        with torch.no_grad():
            _, sampled_rays, ray_weights = renderers.monte_carlo(
                cameras=camera_batch,
                volumetric_function=field.batched_forward,
                mask_rays=mask_rays,
                mask_image=silhouettes[camera_indices, ..., None],
            )

        surface_points = sampled_rays.origins + sampled_rays.directions * torch.max(
            sampled_rays.lengths * ray_weights, dim=-1
        )[0].unsqueeze(-1)
        valid_indices = torch.where(torch.norm(surface_points - sampled_rays.origins, dim=-1)[0])[0]
        negative_points = torch.cat(
            [negative_points, surface_points[:, valid_indices].cpu()], dim=1
        )

    farthest_indices = fps(negative_points, K=renderer_config["negative_farthest_points"])[1][
        0
    ].cpu()
    negative_points = negative_points[:, farthest_indices]
    negative_points = negative_points[
        0,
        torch.where(
            torch.max(torch.abs(negative_points[0, :, :]), dim=-1)[0]
            < renderer_config["max_negative_point_abs"]
        )[0],
    ].unsqueeze(0)

    with torch.no_grad():
        mesh_vertices = field.extract_mesh(threshold=renderer_config["negative_mesh_threshold"])[0]

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(mesh_vertices)
    _filtered_cloud, inlier_indices = point_cloud.remove_radius_outlier(
        nb_points=renderer_config["radius_outlier_points"],
        radius=renderer_config["radius_outlier_distance"],
    )
    mesh_vertices = mesh_vertices[inlier_indices]

    surface_tree = KDTree(np.asarray(mesh_vertices), leaf_size=2)
    nearest_distances, _nearest_indices = surface_tree.query(negative_points[0].cpu().numpy(), k=1)
    negative_points = negative_points[
        :,
        np.where(nearest_distances[:, 0] < renderer_config["negative_surface_distance"])[0],
    ]
    save_numpy(cache_path, negative_points[0].cpu().numpy())
    return negative_points.to(device)


def set_group_lrs(
    optimizer: torch.optim.Optimizer,
    base_lrs: Sequence[float],
    step: int,
    warmup_steps: int,
) -> None:
    factor = min(1.0, (step + 1) / warmup_steps) if warmup_steps > 0 else 1.0
    for group, base_lr in zip(optimizer.param_groups, base_lrs):
        group["lr"] = base_lr * factor
