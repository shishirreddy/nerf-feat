import argparse
import os
from pathlib import Path

import numpy as np
import open3d as o3d
import torch
from pytorch3d.ops import sample_farthest_points as fps
from pytorch3d.renderer import NDCMultinomialRaysampler, RayBundle
from sklearn.neighbors import KDTree

from nerffeat import cli
from nerffeat.artifacts import correspondence_cache_dirs, save_numpy, stage_artifacts
from nerffeat.config import add_common_dataset_args, add_object_args, add_split_arg, configure_logger
from nerffeat.data.bop import load_bop_scene, make_batch_cameras
from nerffeat.radiance_field import NeuralRadianceFieldFeat
from nerffeat.rendering.stratified import (
    EmissionAbsorptionRaymarcherStratified,
    ImplicitRendererStratified,
)
from nerffeat.training import configure_radiance_field, load_model_checkpoint
from nerffeat.utils import visualize_point_cloud

EXPERIMENT_CONFIG = cli.load_experiment_config("correspondences")
CORRESPONDENCE_CONFIG = EXPERIMENT_CONFIG.section

arg_parser = argparse.ArgumentParser(
    description="Generate NeRF-Feat correspondence caches",
    parents=[EXPERIMENT_CONFIG.bootstrap_parser],
)
add_object_args(arg_parser)
add_split_arg(arg_parser)
arg_parser.add_argument("--visualize", default=False, action="store_true")
arg_parser.add_argument("--background-rays", dest="background_rays", default=False, action="store_true",
                        help="Also generate and cache background ray correspondences (disabled by default).")
add_common_dataset_args(arg_parser)

args = arg_parser.parse_args()
LOGGER = configure_logger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

object_id = str(args.object_id)
dataset_root = cli.dataset_root(args, EXPERIMENT_CONFIG.paths)
output_root = cli.output_root(args, EXPERIMENT_CONFIG.paths)
correspondence_artifacts = stage_artifacts(output_root, object_id, "correspondences").create()
radiance_artifacts = stage_artifacts(output_root, object_id, "radiance_field").create()

use_siren = CORRESPONDENCE_CONFIG["use_siren"]

LOGGER.info("object_id=%s", object_id)

render_size = CORRESPONDENCE_CONFIG["image_size"]
train_image_ids = np.load(cli.split_file(args, object_id, EXPERIMENT_CONFIG.paths))
scene = load_bop_scene(
    dataset_root=dataset_root,
    object_id=object_id,
    image_size=CORRESPONDENCE_CONFIG["image_size"],
    device=device,
    image_ids=train_image_ids,
    split_name=CORRESPONDENCE_CONFIG.get("split_name", "train"),
    dataset_name=CORRESPONDENCE_CONFIG.get("dataset_name", "tless"),
)
target_silhouettes = scene.silhouettes
target_cameras = scene.cameras

export_normals = CORRESPONDENCE_CONFIG["export_normals"]
mask_rays = CORRESPONDENCE_CONFIG["mask_rays"]

LOGGER.info("Loaded %d images, silhouettes, and cameras", len(scene.images))


samples_per_ray = CORRESPONDENCE_CONFIG["samples_per_ray"]
raysampler_grid = NDCMultinomialRaysampler(
    image_height=render_size,
    image_width=render_size,
    n_pts_per_ray=samples_per_ray,
    min_depth=scene.min_depth,
    max_depth=scene.max_depth,
)

raymarcher = EmissionAbsorptionRaymarcherStratified()

renderer_grid = ImplicitRendererStratified(
    raysampler=raysampler_grid, raymarcher=raymarcher, device=device
)

renderer_grid = renderer_grid.to(device)

torch.manual_seed(1)
neural_radiance_field = NeuralRadianceFieldFeat(use_siren=use_siren)
configure_radiance_field(neural_radiance_field, mode="feature")
raymarcher.threshold_mode = True
density_threshold = CORRESPONDENCE_CONFIG["density_threshold"]
raymarcher.threshold = density_threshold

if args.background_rays:
    from nerffeat.rendering.background import EmissionAbsorptionRaymarcherStratified as BackgroundRaymarcher
    background_raymarcher = BackgroundRaymarcher()
    background_raymarcher.threshold_mode = True
    background_raymarcher.threshold = density_threshold


load_model_checkpoint(
    neural_radiance_field, radiance_artifacts.checkpoints / "fine_field_latest.pt"
)


neural_radiance_field = neural_radiance_field.to(device)


if export_normals:
    mesh_vertices_path = correspondence_artifacts.meshes / "vertices.npy"
    sampled_vertices_path = correspondence_artifacts.meshes / "sampled_vertices.npy"
    sampled_normals_path = correspondence_artifacts.meshes / "sampled_normals.npy"
    if not mesh_vertices_path.exists():
        with torch.no_grad():
            save_numpy(
                mesh_vertices_path,
                (
                    neural_radiance_field.extract_mesh(
                        threshold=CORRESPONDENCE_CONFIG["mesh_threshold"]
                    )[0]
                ),
            )
    if not sampled_normals_path.exists():
        vertices = np.load(mesh_vertices_path)
        vertices_tensor = torch.from_numpy(vertices.astype("float32")).to(device)
        farthest_indices = fps(
            vertices_tensor.unsqueeze(0), K=CORRESPONDENCE_CONFIG["normal_sample_count"]
        )[1][0].cpu()
        subsampled_vertices = vertices_tensor[farthest_indices]
        import pytorch3d

        normals = -pytorch3d.ops.estimate_pointcloud_normals(
            subsampled_vertices.unsqueeze(0),
            neighborhood_size=CORRESPONDENCE_CONFIG["normal_neighborhood_size"],
        )[0]
        vertices = subsampled_vertices.cpu().numpy()
        normals = normals.cpu().numpy()
        save_numpy(sampled_vertices_path, vertices)
        save_numpy(sampled_normals_path, normals)

cache_dirs = correspondence_cache_dirs(correspondence_artifacts.cache, render_size)
ray_xy_dir = cache_dirs["surface_ray_xys"]
surface_points_dir = cache_dirs["surface_points"]

active_dirs = [cache_dirs["surface_ray_xys"], cache_dirs["surface_points"]]
if args.background_rays:
    active_dirs += [cache_dirs["background_points"], cache_dirs["background_ray_xys"]]
for d in active_dirs:
    d.mkdir(parents=True, exist_ok=True)

with torch.no_grad():
    mesh_vertices = neural_radiance_field.extract_mesh(
        threshold=CORRESPONDENCE_CONFIG["mesh_threshold"]
    )[0]
    surface_tree = KDTree(np.asarray(mesh_vertices), leaf_size=2)

point_cloud = o3d.geometry.PointCloud()
point_cloud.points = o3d.utility.Vector3dVector(mesh_vertices)
filtered_cloud, inlier_indices = point_cloud.remove_radius_outlier(
    nb_points=CORRESPONDENCE_CONFIG["mesh_radius_outlier_points"],
    radius=CORRESPONDENCE_CONFIG["mesh_radius_outlier_distance"],
)
mesh_vertices = mesh_vertices[inlier_indices]
surface_tree = KDTree(np.asarray(mesh_vertices), leaf_size=2)


if args.visualize:
    visualize_point_cloud(mesh_vertices)
save_numpy(correspondence_artifacts.meshes / "filtered_vertices.npy", mesh_vertices)


num_cameras = len(target_cameras)
LOGGER.info("generating correspondences for %d frames to %s", num_cameras, correspondence_artifacts.cache)
for iteration in range(num_cameras):
    batch_indices = torch.tensor([iteration])

    LOGGER.debug("frame %d/%d batch_indices=%s", iteration + 1, num_cameras, batch_indices.tolist())
    if (surface_points_dir / f"{iteration:06d}.pt").exists():
        continue
    if iteration % 100 == 0:
        LOGGER.info("frame %d/%d", iteration, num_cameras)
    batch_cameras = make_batch_cameras(
        target_cameras,
        batch_indices,
        render_size,
        device=device,
    )
    with torch.no_grad():
        _, sampled_rays, weights = renderer_grid(
            cameras=batch_cameras,
            volumetric_function=neural_radiance_field.batched_forward,
            mask_rays=mask_rays,
            mask_image=target_silhouettes[batch_indices, ..., None],
        )

        surface_points = (
            sampled_rays.origins
            + sampled_rays.directions
            * torch.max(sampled_rays.lengths * weights[:, :, 0:samples_per_ray], dim=-1)[
                0
            ].unsqueeze(-1)
        ).cpu()

        nearest_surface_distances, _nearest_surface_indices = surface_tree.query(
            surface_points[0].cpu().numpy(), k=1
        )
        surface_indices = np.where(
            nearest_surface_distances[:, 0] < CORRESPONDENCE_CONFIG["surface_distance"]
        )[0]
        surface_points = surface_points[:, surface_indices, :].cpu()
        sampled_rays = RayBundle(
            origins=sampled_rays.origins[:, surface_indices].cpu(),
            directions=sampled_rays.directions[:, surface_indices].cpu(),
            lengths=sampled_rays.lengths[:, surface_indices].cpu(),
            xys=sampled_rays.xys[:, surface_indices].cpu(),
        )

        if args.background_rays:
            background_ray_lengths = (
                sampled_rays.lengths - sampled_rays.lengths[:, :, 0].unsqueeze(-1)
            ) / 3
            background_rays = RayBundle(
                origins=surface_points.to(device),
                directions=-(
                    sampled_rays.origins / torch.norm(sampled_rays.origins, dim=-1).unsqueeze(-1)
                ).to(device),
                lengths=background_ray_lengths.to(device),
                xys=sampled_rays.xys.to(device),
            )
            background_densities = neural_radiance_field.batched_forward_density(
                ray_bundle=background_rays
            )
            _, background_weights = background_raymarcher(
                rays_densities=background_densities,
                rays_features=torch.zeros_like(background_densities),
            )
            background_points = (
                background_rays.origins
                + background_rays.directions
                * torch.max(
                    background_rays.lengths * background_weights[:, :, samples_per_ray:], dim=-1
                )[0].unsqueeze(-1)
            ).cpu()

            nearest_background_distances, _ = surface_tree.query(
                background_points[0].cpu().numpy(), k=1
            )
            background_indices = np.where(
                nearest_background_distances[:, 0] < CORRESPONDENCE_CONFIG["surface_distance"]
            )[0]
            background_points = background_points[:, background_indices].cpu()
            background_ray_bundle_cpu = RayBundle(
                origins=background_rays.origins.cpu()[:, background_indices],
                directions=background_rays.directions.cpu()[:, background_indices],
                lengths=background_rays.lengths.cpu()[:, background_indices],
                xys=background_rays.xys.cpu()[:, background_indices],
            )
            torch.save(background_points.cpu(), cache_dirs["background_points"] / f"{iteration:06d}.pt")
            torch.save(background_ray_bundle_cpu.xys, cache_dirs["background_ray_xys"] / f"{iteration:06d}.pt")

        ray_bundle_cpu = RayBundle(
            origins=sampled_rays.origins.cpu(),
            directions=sampled_rays.directions.cpu(),
            lengths=sampled_rays.lengths.cpu(),
            xys=sampled_rays.xys.cpu(),
        )

        torch.save(ray_bundle_cpu.xys, ray_xy_dir / f"{iteration:06d}.pt")
        torch.save(surface_points.cpu(), surface_points_dir / f"{iteration:06d}.pt")
        torch.cuda.empty_cache()

LOGGER.info("correspondence cache complete: %d frames written to %s", num_cameras, correspondence_artifacts.cache)
