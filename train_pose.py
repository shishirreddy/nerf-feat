import argparse
import glob
import os

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as data_utils

from nerffeat import cli
from nerffeat.artifacts import save_numpy, save_step_checkpoint, stage_artifacts
from nerffeat.plotting import LossPlotter
from nerffeat.config import (
    add_common_dataset_args,
    add_object_args,
    add_split_arg,
    configure_logger,
    str2bool,
)
from nerffeat.data.bop import load_bop_scene, make_batch_cameras
from nerffeat.data.dataset import AugmentedSamples
from nerffeat.models.unet import ResNetUNet
from nerffeat.preview import save_pose_encoder_previews
from nerffeat.radiance_field import NeuralRadianceFieldFeat
from nerffeat.rendering.factory import build_single_renderers
from nerffeat.training import (
    configure_radiance_field,
    load_model_checkpoint,
    load_or_create_negative_surface_points,
    set_group_lrs,
)
from nerffeat.utils import (
    contrastive_loss_with_negatives,
    sample_images_at_mc_locs,
    save_render_preview,
)

EXPERIMENT_CONFIG = cli.load_experiment_config("pose")
POSE_CONFIG = EXPERIMENT_CONFIG.section

arg_parser = argparse.ArgumentParser(
    description="Train the NeRF-Feat pose encoder",
    parents=[EXPERIMENT_CONFIG.bootstrap_parser],
)
add_object_args(arg_parser)
add_split_arg(arg_parser)
arg_parser.add_argument("--resume", default=False, type=str2bool)
arg_parser.add_argument(
    "--batch-size", dest="batch_size", default=POSE_CONFIG["batch_size"], type=int
)
arg_parser.add_argument(
    "--num-workers", dest="num_workers", default=POSE_CONFIG["num_workers"], type=int
)
arg_parser.add_argument(
    "--total-updates", dest="total_updates", default=POSE_CONFIG["total_updates"], type=int
)
arg_parser.add_argument(
    "--warmup-steps", dest="warmup_steps", default=POSE_CONFIG["warmup_steps"], type=int
)
arg_parser.add_argument(
    "--checkpoint-interval", default=POSE_CONFIG["checkpoint_interval"], type=int
)
arg_parser.add_argument("--preview-interval", default=POSE_CONFIG["preview_interval"], type=int)
arg_parser.add_argument("--log-interval", default=POSE_CONFIG["log_interval"], type=int)
arg_parser.add_argument("--debug-samples", dest="debug_samples", default=None, type=int)
arg_parser.add_argument(
    "--coco-root",
    default=None,
    help="Directory of COCO background JPEGs. Defaults to $NERFFEAT_COCO_ROOT or data/coco.",
)
add_common_dataset_args(arg_parser)
args = arg_parser.parse_args()
LOGGER = configure_logger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

coco_root = cli.coco_root(args, EXPERIMENT_CONFIG.paths)
coco_images = glob.glob(os.path.join(coco_root, "*.jpg"))
if not coco_images:
    LOGGER.warning(
        "No COCO background images found at '%s'. "
        "Background augmentation will be disabled for this run.\n"
        "  To enable it:\n"
        "    1. Download COCO 2017 train images: "
        "https://cocodataset.org/#download  (train2017.zip)\n"
        "    2. Extract the JPEGs to a directory.\n"
        "    3. Set the path via --coco-root <dir>, "
        "the NERFFEAT_COCO_ROOT environment variable, "
        "or paths.coco_root in your config YAML.",
        coco_root,
    )
object_id = str(args.object_id)
dataset_path = cli.dataset_root(args, EXPERIMENT_CONFIG.paths)
output_path = cli.output_root(args, EXPERIMENT_CONFIG.paths)

radiance_artifacts = stage_artifacts(output_path, object_id, "radiance_field").create()
correspondence_artifacts = stage_artifacts(output_path, object_id, "correspondences").create()
pose_artifacts = stage_artifacts(output_path, object_id, "pose_encoder").create()
use_siren = POSE_CONFIG["use_siren"]
cache_negative_samples = POSE_CONFIG["cache_negative_samples"]
points_per_sample = POSE_CONFIG["points_per_sample"]
image_size = POSE_CONFIG["image_size"]
key_noise = POSE_CONFIG["key_noise"]
batch_size = args.batch_size
cnn_learning_rate = POSE_CONFIG["cnn_learning_rate"]
feature_mlp_learning_rate = POSE_CONFIG["feature_mlp_learning_rate"]

train_image_ids = np.load(cli.split_file(args, object_id, EXPERIMENT_CONFIG.paths))
if args.debug_samples is not None:
    train_image_ids = train_image_ids[: args.debug_samples]

scene = load_bop_scene(
    dataset_root=dataset_path,
    object_id=object_id,
    image_size=image_size,
    device=device,
    image_ids=train_image_ids,
    split_name=POSE_CONFIG.get("split_name", "train"),
    dataset_name=POSE_CONFIG.get("dataset_name", "tless"),
)
target_images = scene.images
target_silhouettes = scene.silhouettes
target_cameras = scene.cameras
image_ids = scene.image_ids

mask_rays = POSE_CONFIG["mask_rays"]

LOGGER.info("Loaded %d images, silhouettes, and cameras", len(target_images))
LOGGER.info(
    "config: total_updates=%d batch_size=%d warmup_steps=%d "
    "cnn_lr=%.1e feature_mlp_lr=%.1e feature_weight=%.4f mask_weight=%.4f",
    args.total_updates,
    batch_size,
    args.warmup_steps,
    cnn_learning_rate,
    feature_mlp_learning_rate,
    POSE_CONFIG["losses"]["feature_weight"],
    POSE_CONFIG["losses"]["mask_weight"],
)

augmented_dataset = AugmentedSamples(
    dataset_path=dataset_path,
    correspondence_dir=str(correspondence_artifacts.cache),
    render_size=image_size,
    sample_size=points_per_sample,
    image_size=image_size,
    target_images=target_images,
    target_silhouettes=target_silhouettes,
    coco_aug=bool(coco_images),
    occluder_split_name=POSE_CONFIG.get("occluder_split_name"),
    object_id=object_id,
    scale_factor=0.8,
    box_occlusion=True,
    total_samples=target_images.shape[0],
    max_scale=1.2,
    translation_scale=0.2,
    normalize_object_scale=True,
    scale_jitter=1.5,
    line_occlusion=True,
    coco_images=coco_images,
    max_augmentation_crop=20,
    min_visible_pixels=10000,
    secondary_translation_scale=0.4,
    add_border=True,
)

train_loader = data_utils.DataLoader(
    augmented_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=args.num_workers,
    drop_last=True,
)

renderers = build_single_renderers(
    config=POSE_CONFIG["renderer"],
    render_size=image_size,
    min_depth=scene.min_depth,
    max_depth=scene.max_depth,
    device=device,
)
raymarcher = renderers.raymarcher

torch.manual_seed(1)
radiance_field = NeuralRadianceFieldFeat(use_siren=use_siren)
# Freeze pretrained density/color branches; only the surface feature MLP trains here.
configure_radiance_field(radiance_field, mode="feature")
raymarcher.threshold_mode = True
raymarcher.threshold = POSE_CONFIG["renderer"]["density_threshold"]

# 12 feature channels + 1 mask logit.
image_encoder = ResNetUNet(num_classes=13, num_decoders=1)

save_numpy(pose_artifacts.metadata / "image_ids.npy", image_ids)

load_model_checkpoint(radiance_field, radiance_artifacts.checkpoints / "fine_field_latest.pt")

if args.resume:
    load_model_checkpoint(radiance_field, pose_artifacts.checkpoints / "feature_field_latest.pt")
    load_model_checkpoint(image_encoder, pose_artifacts.checkpoints / "rgb_encoder_latest.pt")
    LOGGER.info("Resumed feature field and RGB encoder from %s", pose_artifacts.checkpoints)

radiance_field = radiance_field.to(device)
image_encoder = image_encoder.to(device)

optimizer = torch.optim.Adam(
    [
        {"params": radiance_field.parameters(), "lr": feature_mlp_learning_rate},
        {"params": image_encoder.parameters(), "lr": cnn_learning_rate},
    ]
)

num_epochs = int(args.total_updates * batch_size / target_images.shape[0])

iteration = 0
loss_plotter = LossPlotter()
negative_points = torch.Tensor([])
negative_cache_path = pose_artifacts.cache / "negative_surface_points.npy"
for _epoch in range(num_epochs):
    for train_data, batch_indices in train_loader:
        iteration += 1

        set_group_lrs(
            optimizer, [feature_mlp_learning_rate, cnn_learning_rate], iteration, args.warmup_steps
        )

        if iteration % args.checkpoint_interval == 0 and iteration > 1:
            save_step_checkpoint(pose_artifacts.checkpoints, "feature_field", iteration, radiance_field)
            save_step_checkpoint(pose_artifacts.checkpoints, "rgb_encoder", iteration, image_encoder)
            LOGGER.info("step=%05d checkpoint saved to %s", iteration, pose_artifacts.checkpoints)

        optimizer.zero_grad()

        rgb_batch, mask_batch, surface_points, surface_ray_xy = train_data

        image_feature_map = torch.movedim(
            image_encoder(input=torch.movedim(rgb_batch.to(device), 3, 1)), 1, 3
        )
        mask_logits = image_feature_map[..., -1:]
        image_feature_map = image_feature_map[..., 0:12]

        if negative_points.shape[0] == 0:
            negative_points = load_or_create_negative_surface_points(
                cache_path=negative_cache_path,
                field=radiance_field,
                renderers=renderers,
                cameras=target_cameras,
                silhouettes=target_silhouettes,
                image_size=image_size,
                device=device,
                renderer_config=POSE_CONFIG["renderer"],
                cache_enabled=cache_negative_samples,
                mask_rays=mask_rays,
            )

        negative_sample_indices = torch.randperm(negative_points.shape[1], dtype=torch.int64)[
            0 : batch_size * points_per_sample
        ]
        negative_point_batch = negative_points[:, negative_sample_indices].clone()
        negative_point_batch += torch.randn_like(negative_point_batch) * key_noise

        negative_field_output = radiance_field.batched_forward_features_at_points(
            negative_point_batch.to(device)
        )

        last_dim = negative_field_output.shape[-1]
        negative_feature_values, _negative_density_values = negative_field_output.split(
            [last_dim - 1, 1], dim=-1
        )

        negative_features = negative_feature_values.reshape(-1, 12).unsqueeze(0)

        field_output = radiance_field.batched_forward_features_at_points(surface_points.to(device))
        surface_keys = field_output[:, :, 0:12]
        queries = sample_images_at_mc_locs(image_feature_map, surface_ray_xy.to(device))

        feature_loss = contrastive_loss_with_negatives(
            queries, surface_keys, negative_features.view(batch_size, points_per_sample, -1)
        )

        mask_loss = (
            F.binary_cross_entropy(torch.sigmoid(mask_logits[..., 0]), mask_batch.to(device))
            * POSE_CONFIG["losses"]["mask_weight"]
        )

        feature_loss = POSE_CONFIG["losses"]["feature_weight"] * feature_loss
        loss = feature_loss + mask_loss

        if iteration % args.log_interval == 0:
            LOGGER.info(
                "step=%05d feature_loss=%.2e mask_loss=%.2e",
                iteration,
                float(feature_loss),
                float(mask_loss),
            )
            loss_plotter.update(
                iteration,
                feature_loss=float(feature_loss),
                mask_loss=float(mask_loss),
            )

        loss.backward()
        optimizer.step()

        if iteration % args.preview_interval == 0:
            loss_plotter.save(pose_artifacts.previews / "loss_curves.png")
            preview_id = f"step_{iteration:06d}"
            save_pose_encoder_previews(
                output_dir=str(pose_artifacts.previews),
                preview_id=preview_id,
                rgb_batch=rgb_batch,
                mask_logits=mask_logits,
                image_feature_map=image_feature_map,
            )
            preview_camera = make_batch_cameras(
                target_cameras,
                batch_indices[0:1],
                image_size,
                device=device,
            )
            save_render_preview(
                radiance_field,
                preview_camera,
                output_dir=str(pose_artifacts.previews),
                renderer_grid=renderers.eval_grid,
                normalize_features=True,
                preview_id=preview_id,
            )

LOGGER.info(
    "training complete: total_steps=%d checkpoints=%s", iteration, pose_artifacts.checkpoints
)
