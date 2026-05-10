import argparse

import numpy as np
import torch
from pytorch3d.ops import sample_farthest_points

from nerffeat import cli
from nerffeat.artifacts import save_numpy, stage_artifacts
from nerffeat.config import add_common_dataset_args, add_object_args, add_split_arg, configure_logger
from nerffeat.data.bop import load_bop_scene, make_batch_cameras
from nerffeat.geometry import estimate_surface_normals, keep_near_mesh_surface
from nerffeat.radiance_field import NeuralRadianceFieldFeat
from nerffeat.rendering.factory import build_single_renderers
from nerffeat.training import configure_radiance_field, load_model_checkpoint

EXPERIMENT_CONFIG = cli.load_experiment_config("export")
EXPORT_CONFIG = EXPERIMENT_CONFIG.section

arg_parser = argparse.ArgumentParser(
    description="Export NeRF-Feat surface points, features, and normals",
    parents=[EXPERIMENT_CONFIG.bootstrap_parser],
)
add_object_args(arg_parser, default_object_id="35")
add_split_arg(arg_parser)
add_common_dataset_args(arg_parser)
args = arg_parser.parse_args()

LOGGER = configure_logger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

object_id = str(args.object_id)
dataset_root = cli.dataset_root(args, EXPERIMENT_CONFIG.paths)
output_root = cli.output_root(args, EXPERIMENT_CONFIG.paths)
pose_artifacts = stage_artifacts(output_root, object_id, "pose_encoder").create()

train_image_ids = np.load(cli.split_file(args, object_id, EXPERIMENT_CONFIG.paths))
scene = load_bop_scene(
    dataset_root=dataset_root,
    object_id=object_id,
    image_size=EXPORT_CONFIG["image_size"],
    device=device,
    image_ids=train_image_ids,
    split_name=EXPORT_CONFIG.get("split_name", "train"),
    dataset_name=EXPORT_CONFIG.get("dataset_name", "tless"),
)
save_numpy(pose_artifacts.metadata / "image_ids.npy", scene.image_ids)

renderers = build_single_renderers(
    config=EXPORT_CONFIG["renderer"],
    render_size=EXPORT_CONFIG["image_size"],
    min_depth=scene.min_depth,
    max_depth=scene.max_depth,
    device=device,
)
renderers.raymarcher.threshold_mode = True
renderers.raymarcher.threshold = EXPORT_CONFIG["renderer"]["density_threshold"]

torch.manual_seed(1)
field = NeuralRadianceFieldFeat(use_siren=EXPORT_CONFIG["use_siren"]).to(device)
configure_radiance_field(field, mode="feature")
load_model_checkpoint(field, pose_artifacts.checkpoints / "feature_field_latest.pt")
field.eval()

LOGGER.info(
    "object_id=%s images=%d candidate_passes=%d farthest_points=%d",
    object_id,
    len(scene.images),
    EXPORT_CONFIG["candidate_passes"],
    EXPORT_CONFIG["farthest_points"],
)

candidate_points = torch.Tensor([])
camera_indices = torch.randperm(len(scene.cameras))
candidate_cameras = make_batch_cameras(
    scene.cameras,
    camera_indices,
    EXPORT_CONFIG["image_size"],
    device=device,
)

num_passes = EXPORT_CONFIG["candidate_passes"]
for pass_idx in range(num_passes):
    with torch.no_grad():
        _, sampled_rays, weights = renderers.monte_carlo(
            cameras=candidate_cameras,
            volumetric_function=field.batched_forward,
            mask_rays=True,
            mask_image=scene.silhouettes[camera_indices, ..., None],
        )
    surface_points = sampled_rays.origins + sampled_rays.directions * torch.max(
        sampled_rays.lengths * weights, dim=-1
    )[0].unsqueeze(-1)
    valid_indices = torch.where(torch.norm(surface_points - sampled_rays.origins, dim=-1)[0])[0]
    candidate_points = torch.cat([candidate_points, surface_points[:, valid_indices].cpu()], dim=1)
    LOGGER.info("pass %d/%d: %d candidate points accumulated", pass_idx + 1, num_passes, candidate_points.shape[1])

candidate_indices = sample_farthest_points(candidate_points, K=EXPORT_CONFIG["farthest_points"])[1][
    0
].cpu()
candidate_points = candidate_points[:, candidate_indices]
candidate_points = candidate_points[
    0,
    torch.where(
        torch.max(torch.abs(candidate_points[0]), dim=-1)[0] < EXPORT_CONFIG["max_point_abs"]
    )[0],
].unsqueeze(0)

with torch.no_grad():
    mesh_vertices, mesh_faces = field.extract_mesh(threshold=EXPORT_CONFIG["mesh_threshold"])
mesh_normals = estimate_surface_normals(mesh_vertices, mesh_faces)
surface_points, surface_normals = keep_near_mesh_surface(
    candidate_points,
    mesh_vertices,
    mesh_normals,
    max_distance=EXPORT_CONFIG["surface_distance"],
)

with torch.no_grad():
    surface_features_with_density = field.batched_forward_features_at_points(
        surface_points.to(device)
    )
feature_channels = surface_features_with_density.shape[-1]
surface_features, _surface_density = surface_features_with_density.split(
    [feature_channels - 1, 1], dim=-1
)

save_numpy(
    pose_artifacts.features / "surface_points.npy",
    surface_points[0].cpu().numpy() * scene.translation_scale,
)
save_numpy(pose_artifacts.features / "surface_features.npy", surface_features[0].cpu().numpy())
save_numpy(pose_artifacts.features / "surface_normals.npy", surface_normals)
LOGGER.info("Exported %d surface features to %s", surface_points.shape[1], pose_artifacts.features)
