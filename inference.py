import argparse
import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from nerffeat import cli
from nerffeat.artifacts import stage_artifacts
from nerffeat.config import add_common_dataset_args, add_object_args, add_split_arg, configure_logger
from nerffeat.data.bop import load_object_diameter
from nerffeat.geometry import (
    add_error,
    crop_bop_instance,
    match_correspondences,
    mesh_vertices,
    normalize_imagenet,
    scale_intrinsics_for_downsampling,
    solve_pnp_ransac,
)
from nerffeat.models.unet import ResNetUNet
from nerffeat.training import load_model_checkpoint
from nerffeat.viz import visualize_correspondences

EXPERIMENT_CONFIG = cli.load_experiment_config("inference")
INFERENCE_CONFIG = EXPERIMENT_CONFIG.section

arg_parser = argparse.ArgumentParser(
    description="Evaluate NeRF-Feat pose estimates (PnP-only, fast)",
    parents=[EXPERIMENT_CONFIG.bootstrap_parser],
)
add_object_args(arg_parser)
add_split_arg(arg_parser)
arg_parser.add_argument("--image-index", default=None, type=int)
arg_parser.add_argument(
    "--split-name",
    dest="split_name",
    default=INFERENCE_CONFIG.get("split_name", "train"),
)
arg_parser.add_argument("--visualize", default=False, action="store_true")
arg_parser.add_argument(
    "--output-csv",
    dest="output_csv",
    default=None,
    help="Write pose estimates in BOP results CSV format to this path.",
)
add_common_dataset_args(arg_parser)
args = arg_parser.parse_args()
LOGGER = configure_logger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
object_id = str(args.object_id)
LOGGER.info("object_id=%s", object_id)
dataset_root = cli.dataset_root(args, EXPERIMENT_CONFIG.paths)
output_root = cli.output_root(args, EXPERIMENT_CONFIG.paths)
predicted_mask_root = cli.mask_root(args, EXPERIMENT_CONFIG.paths)
dataset_name = INFERENCE_CONFIG.get("dataset_name", "tless")
pose_artifacts = stage_artifacts(output_root, object_id, "pose_encoder")

image_size = INFERENCE_CONFIG["image_size"]
diameter = load_object_diameter(dataset_root, object_id)
# NeRF training scales translations by this factor; invert it for BOP mm output.
translation_scale = diameter / 1.8

train_ids_set: set[int] = set()
if args.split_file:
    train_ids_set = set(np.load(args.split_file).tolist())
    LOGGER.info("loaded split_file=%s  excluded_train_frames=%d", args.split_file, len(train_ids_set))

torch.manual_seed(1)

encoder_rgb = ResNetUNet(num_classes=13, num_decoders=1)
load_model_checkpoint(encoder_rgb, pose_artifacts.checkpoints / "rgb_encoder_latest.pt")
encoder_rgb.to(device).eval()

scaled_surface_points = np.load(pose_artifacts.features / "surface_points.npy")
scaled_surface_features = np.load(pose_artifacts.features / "surface_features.npy")
surface_features = torch.from_numpy(scaled_surface_features).to(device)
LOGGER.info(
    "loaded %d surface points from %s",
    scaled_surface_points.shape[0],
    pose_artifacts.features,
)

model_vertices = mesh_vertices(dataset_root, object_id)

object_dir = Path(dataset_root) / args.split_name / str(object_id).zfill(6)
evaluation_files = sorted(object_dir.glob("rgb/*.png"))

if args.image_index is not None:
    evaluation_files = [evaluation_files[int(args.image_index)]]

# Cache JSON files to avoid re-reading them every frame.
with open(object_dir / "scene_gt.json", encoding="utf-8") as f:
    scene_gt_all = json.load(f)
with open(object_dir / "scene_camera.json", encoding="utf-8") as f:
    scene_camera_all = json.load(f)

LOGGER.info(
    "evaluation_dir=%s  total_frames=%d  excluded_train=%d",
    object_dir,
    len(evaluation_files),
    len(train_ids_set),
)

add_success_count = 0
rotation_success_count = 0
evaluated_count = 0

csv_file = None
csv_writer = None
if args.output_csv:
    output_csv_path = Path(args.output_csv)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_file = open(output_csv_path, "w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["scene_id", "im_id", "obj_id", "score", "R", "t", "time"])
    LOGGER.info("writing BOP results to %s", output_csv_path)

scene_id = int(object_id)

for image_path in evaluation_files:
    image_id = int(image_path.stem)

    if image_id in train_ids_set:
        continue

    LOGGER.info(
        "progress evaluated=%d add_success=%d rotation_success=%d",
        evaluated_count,
        add_success_count,
        rotation_success_count,
    )
    evaluated_count += 1

    scene_instances = scene_gt_all[str(image_id)]
    gt_rotation, gt_translation = None, None
    for instance in scene_instances:
        if instance["obj_id"] == int(object_id):
            gt_rotation = np.asarray(instance["cam_R_m2c"]).reshape(3, 3)
            gt_translation = np.asarray(instance["cam_t_m2c"])
            break
    if gt_rotation is None:
        LOGGER.warning("image_id=%d: object %s not found in scene_gt, skipping", image_id, object_id)
        evaluated_count -= 1
        continue

    camera_matrix = np.array(scene_camera_all[str(image_id)]["cam_K"]).reshape(3, 3)
    rgb = cv2.imread(str(image_path))

    # Resolve segmentation mask: predicted layout is
    #   {mask_root}/{dataset_name}/scene{object_id}/{image_id:06d}.png
    # Falls back to the GT mask when the predicted file is absent.
    predicted_mask_path = (
        predicted_mask_root / dataset_name / f"scene{object_id}" / f"{image_id:06d}.png"
    )
    gt_mask_path = image_path.parent.parent / "mask_visib" / f"{image_id:06d}_000000.png"
    mask_path = predicted_mask_path if predicted_mask_path.exists() else gt_mask_path

    if mask_path == gt_mask_path and not gt_mask_path.exists():
        LOGGER.warning("image_id=%d: no mask found (checked %s), skipping", image_id, predicted_mask_path)
        evaluated_count -= 1
        continue

    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        LOGGER.warning("image_id=%d: failed to read mask at %s, skipping", image_id, mask_path)
        evaluated_count -= 1
        continue
    mask = (mask > 0).astype(np.uint8) * 255
    mask = np.stack([mask, mask, mask], axis=-1)

    crop_rgb, crop_mask, crop_intrinsics = crop_bop_instance(
        rgb=rgb,
        mask=mask,
        camera_matrix=camera_matrix,
        image_size=image_size,
        crop_scale=INFERENCE_CONFIG["crop_scale"],
        apply_mask=INFERENCE_CONFIG["apply_mask_to_crop"],
    )

    input_mask = torch.from_numpy(crop_mask[:, :, 0])
    input_image = torch.movedim(
        torch.from_numpy(normalize_imagenet(crop_rgb).astype("float32")).to(device).unsqueeze(0),
        3,
        1,
    )

    with torch.no_grad():
        feature_logits = torch.movedim(encoder_rgb(input_image), 1, 3)
    image_features = feature_logits[..., 0:12]

    down_sample = INFERENCE_CONFIG["downsample"]
    image_features = image_features[:, ::down_sample, ::down_sample]
    input_mask = input_mask[::down_sample, ::down_sample]
    if INFERENCE_CONFIG["scale_camera_for_downsampling"]:
        crop_intrinsics = scale_intrinsics_for_downsampling(crop_intrinsics, down_sample)

    mask_indices = torch.where(input_mask)
    masked_features = image_features[0][mask_indices]

    matched_surface_indices, match_scores = match_correspondences(
        masked_features,
        surface_features,
        top_k=1,
    )
    points_3d = scaled_surface_points[matched_surface_indices.cpu()]

    points_2d = np.zeros((points_3d.shape[0], 2))
    points_2d[:, 0] = mask_indices[1].cpu()
    points_2d[:, 1] = mask_indices[0].cpu()

    if len(match_scores) > 500:
        top_count = int(0.8 * len(match_scores))
        match_threshold = torch.sort(match_scores[:, 0])[0][-top_count + 1]
    else:
        match_threshold = torch.sort(match_scores[:, 0])[0][-len(match_scores) + 1]

    strong_match_indices = torch.where(match_scores[:, 0] > match_threshold)[0].cpu().numpy()
    points_3d = points_3d[strong_match_indices]
    points_2d = points_2d[strong_match_indices]

    if args.visualize:
        surface_sample = scaled_surface_points[
            np.random.choice(scaled_surface_points.shape[0], min(5000, scaled_surface_points.shape[0]), replace=False)
        ]
        query_pts_3d = points_3d.dot(gt_rotation.T) + gt_translation / 3
        surface_pts_3d = surface_sample.dot(gt_rotation.T) + gt_translation / 3
        match_colors = crop_rgb[::down_sample, ::down_sample][
            (mask_indices[0].cpu().numpy()[strong_match_indices],
             mask_indices[1].cpu().numpy()[strong_match_indices])
        ] / 255.0
        visualize_correspondences(
            query_points_3d=query_pts_3d,
            matched_points_3d=points_3d,
            surface_points=surface_pts_3d,
            matched_colors=match_colors,
        )

    t0 = time.perf_counter()
    pose = solve_pnp_ransac(
        points_3d,
        points_2d,
        crop_intrinsics,
        iterations=INFERENCE_CONFIG["pnp_iterations"],
        reprojection_error=INFERENCE_CONFIG["pnp_reprojection_error"],
        method=cv2.SOLVEPNP_P3P,
    )
    elapsed = time.perf_counter() - t0
    if pose is None:
        continue
    estimated_rotation, estimated_translation = pose

    if csv_writer is not None:
        r_str = " ".join(f"{v:.8f}" for v in estimated_rotation.flatten())
        t_mm = estimated_translation * translation_scale
        t_str = " ".join(f"{v:.4f}" for v in t_mm)
        csv_writer.writerow([scene_id, image_id, int(object_id), 1.0, r_str, t_str, f"{elapsed:.4f}"])

    final_add = add_error(
        model_vertices, gt_rotation, gt_translation, estimated_rotation, estimated_translation
    )
    final_rotation_add = add_error(
        model_vertices, gt_rotation, np.zeros(3), estimated_rotation, np.zeros(3)
    )

    LOGGER.info(
        "sample_id=%s add_error=%.4f rotation_add_error=%.4f",
        image_id,
        final_add,
        final_rotation_add,
    )
    if final_add < INFERENCE_CONFIG["add_threshold_fraction"] * diameter:
        add_success_count += 1
        LOGGER.info("sample_id=%s passed ADD threshold", image_id)

    if final_rotation_add < INFERENCE_CONFIG["add_threshold_fraction"] * diameter:
        rotation_success_count += 1
        LOGGER.info("sample_id=%s passed rotation-only ADD threshold", image_id)

if csv_file is not None:
    csv_file.close()

if evaluated_count > 0:
    LOGGER.info(
        "results: evaluated=%d add_success=%d (%.1f%%) rotation_success=%d (%.1f%%)",
        evaluated_count,
        add_success_count,
        100.0 * add_success_count / evaluated_count,
        rotation_success_count,
        100.0 * rotation_success_count / evaluated_count,
    )
