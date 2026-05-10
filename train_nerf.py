import argparse
import os

import numpy as np
import torch

from nerffeat import cli
from nerffeat.artifacts import save_numpy, save_ply, save_step_checkpoint, stage_artifacts
from nerffeat.plotting import LossPlotter
from nerffeat.config import (
    add_common_dataset_args,
    add_object_args,
    add_split_arg,
    configure_logger,
)
from nerffeat.data.bop import load_bop_scene, make_batch_cameras
from nerffeat.preview import save_rgb_preview
from nerffeat.radiance_field import NeuralRadianceFieldFeat
from nerffeat.rendering.factory import build_nerf_renderers
from nerffeat.training import configure_radiance_field
from nerffeat.utils import huber, sample_images_at_mc_locs, save_render_preview

EXPERIMENT_CONFIG = cli.load_experiment_config("nerf")
NERF_CONFIG = EXPERIMENT_CONFIG.section

arg_parser = argparse.ArgumentParser(
    description="Train the NeRF-Feat radiance field",
    parents=[EXPERIMENT_CONFIG.bootstrap_parser],
)
add_object_args(arg_parser)
add_split_arg(arg_parser)
arg_parser.add_argument("--epochs", default=NERF_CONFIG["epochs"], type=int)
arg_parser.add_argument(
    "--batch-size", dest="batch_size", default=NERF_CONFIG["batch_size"], type=int
)
arg_parser.add_argument(
    "--learning-rate", dest="learning_rate", default=NERF_CONFIG["learning_rate"], type=float
)
arg_parser.add_argument(
    "--checkpoint-interval", default=NERF_CONFIG["checkpoint_interval"], type=int
)
arg_parser.add_argument("--preview-interval", default=NERF_CONFIG["preview_interval"], type=int)
arg_parser.add_argument(
    "--point-cloud-interval", default=NERF_CONFIG["point_cloud_interval"], type=int
)
arg_parser.add_argument("--log-interval", default=NERF_CONFIG["log_interval"], type=int)
add_common_dataset_args(arg_parser)
args = arg_parser.parse_args()

LOGGER = configure_logger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

object_id = str(args.object_id)
dataset_path = cli.dataset_root(args, EXPERIMENT_CONFIG.paths)
output_path = cli.output_root(args, EXPERIMENT_CONFIG.paths)

os.makedirs(output_path, exist_ok=True)
artifacts = stage_artifacts(output_path, object_id, "radiance_field").create()

train_image_ids = np.load(cli.split_file(args, object_id, EXPERIMENT_CONFIG.paths))

scene = load_bop_scene(
    dataset_root=dataset_path,
    object_id=object_id,
    image_size=NERF_CONFIG["image_size"],
    device=device,
    image_ids=train_image_ids,
    split_name=NERF_CONFIG.get("split_name", "train"),
    dataset_name=NERF_CONFIG.get("dataset_name", "tless"),
)

renderers = build_nerf_renderers(
    config=NERF_CONFIG["renderer"],
    render_size=NERF_CONFIG["image_size"],
    min_depth=scene.min_depth,
    max_depth=scene.max_depth,
    device=device,
)

torch.manual_seed(1)
coarse_field = NeuralRadianceFieldFeat(use_siren=NERF_CONFIG["use_siren"]).to(device)
fine_field = NeuralRadianceFieldFeat(use_siren=NERF_CONFIG["use_siren"]).to(device)
for field in (coarse_field, fine_field):
    configure_radiance_field(field, mode="color")

optimizer = torch.optim.Adam(
    [
        {"params": coarse_field.parameters(), "lr": args.learning_rate},
        {"params": fine_field.parameters(), "lr": args.learning_rate},
    ]
)

LOGGER.info(
    "object_id=%s images=%d image_size=%d min_depth=%.3f max_depth=%.3f",
    object_id,
    len(scene.images),
    NERF_CONFIG["image_size"],
    scene.min_depth,
    scene.max_depth,
)
LOGGER.info(
    "config: epochs=%d batch_size=%d lr=%.1e color_weight=%.0f silhouette_weight=%.0f",
    args.epochs,
    args.batch_size,
    args.learning_rate,
    NERF_CONFIG["losses"]["color_weight"],
    NERF_CONFIG["losses"]["silhouette_weight"],
)

iteration = 0
loss_plotter = LossPlotter()
for _epoch in range(args.epochs):
    epoch_indices = torch.randperm(len(scene.cameras))

    for batch_offset in range(len(epoch_indices) // args.batch_size):
        iteration += 1

        if iteration % args.checkpoint_interval == 0 and iteration > 1:
            save_step_checkpoint(artifacts.checkpoints, "coarse_field", iteration, coarse_field)
            save_step_checkpoint(artifacts.checkpoints, "fine_field", iteration, fine_field)
            LOGGER.info("step=%05d checkpoint saved to %s", iteration, artifacts.checkpoints)

        optimizer.zero_grad()
        batch_indices = epoch_indices[batch_offset : batch_offset + args.batch_size]
        batch_cameras = make_batch_cameras(
            scene.cameras,
            batch_indices,
            NERF_CONFIG["image_size"],
            device=device,
        )

        # Fine renderer uses coarse weights for importance sampling of ray points.
        coarse_render, coarse_rays, coarse_weights = renderers.coarse(
            cameras=batch_cameras,
            volumetric_function=coarse_field,
        )
        color_render, silhouette_render = coarse_render.split(
            [coarse_render.shape[-1] - 1, 1], dim=-1
        )

        renderers.fine.coarse_rays = coarse_rays
        renderers.fine.coarse_weights = coarse_weights.detach()
        fine_render, fine_rays, _fine_weights = renderers.fine(
            cameras=batch_cameras,
            volumetric_function=fine_field,
        )
        fine_color_render, fine_silhouette_render = fine_render.split(
            [fine_render.shape[-1] - 1, 1], dim=-1
        )

        target_silhouettes = sample_images_at_mc_locs(
            scene.silhouettes[batch_indices, ..., None].to(device), fine_rays.xys
        )
        target_colors = sample_images_at_mc_locs(
            scene.images[batch_indices].to(device), fine_rays.xys
        )

        color_loss = (
            huber(color_render, target_colors).abs().mean()
            + huber(fine_color_render, target_colors).abs().mean()
        )
        silhouette_loss = (
            huber(silhouette_render, target_silhouettes).abs().mean()
            + huber(fine_silhouette_render, target_silhouettes).abs().mean()
        )

        weighted_color_loss = NERF_CONFIG["losses"]["color_weight"] * color_loss
        weighted_silhouette_loss = NERF_CONFIG["losses"]["silhouette_weight"] * silhouette_loss
        loss = weighted_color_loss + weighted_silhouette_loss

        if iteration % args.log_interval == 0:
            LOGGER.info(
                "step=%05d color_loss=%.2e silhouette_loss=%.2e",
                iteration,
                float(weighted_color_loss),
                float(weighted_silhouette_loss),
            )
            loss_plotter.update(
                iteration,
                color_loss=float(weighted_color_loss),
                silhouette_loss=float(weighted_silhouette_loss),
            )

        for milestone in (NERF_CONFIG["checkpoint_10k_step"], NERF_CONFIG["checkpoint_50k_step"]):
            if iteration == milestone:
                save_step_checkpoint(artifacts.checkpoints, "coarse_field", iteration, coarse_field)
                save_step_checkpoint(artifacts.checkpoints, "fine_field", iteration, fine_field)
                LOGGER.info("step=%05d milestone checkpoint saved to %s", iteration, artifacts.checkpoints)

        loss.backward()
        optimizer.step()

        if iteration % args.point_cloud_interval == 0:
            step_tag = f"step_{iteration:06d}"
            with torch.no_grad():
                coarse_verts, coarse_faces = coarse_field.extract_mesh(
                    threshold=NERF_CONFIG["mesh_threshold"]
                )
                save_ply(artifacts.meshes / f"{step_tag}_coarse_mesh.ply", coarse_verts, coarse_faces)
                save_ply(artifacts.meshes / "coarse_mesh_latest.ply", coarse_verts, coarse_faces)
                fine_verts, fine_faces = fine_field.extract_mesh(
                    threshold=NERF_CONFIG["mesh_threshold"]
                )
                save_ply(artifacts.meshes / f"{step_tag}_fine_mesh.ply", fine_verts, fine_faces)
                save_ply(artifacts.meshes / "fine_mesh_latest.ply", fine_verts, fine_faces)
            LOGGER.info("step=%05d mesh exported to %s", iteration, artifacts.meshes)

        if iteration % args.preview_interval == 0:
            loss_plotter.save(artifacts.previews / "loss_curves.png")
            preview_id = f"step_{iteration:06d}"
            save_rgb_preview(
                artifacts.previews / f"{preview_id}_target.jpg",
                scene.images[batch_indices][0],
            )
            preview_camera = make_batch_cameras(
                scene.cameras,
                batch_indices[0:1],
                NERF_CONFIG["image_size"],
                device=device,
            )
            save_render_preview(
                coarse_field,
                preview_camera,
                output_dir=str(artifacts.previews),
                renderer_grid=renderers.eval_grid,
                normalize_features=True,
                preview_id=preview_id,
            )
            save_render_preview(
                fine_field,
                preview_camera,
                output_dir=str(artifacts.previews),
                renderer_grid=renderers.eval_grid,
                normalize_features=True,
                preview_id=preview_id,
                filename_suffix="_fine_field",
            )
