
import logging
from pathlib import Path

import cv2
import numpy as np
import torch
import trimesh
from sklearn.neighbors import KDTree

LOGGER = logging.getLogger(__name__)


def add_error(vertices: np.ndarray, gt_rotation, gt_translation, rotation, translation) -> float:
    """Average Distance of Model Points (ADD)."""

    transformed_gt = vertices.dot(gt_rotation.T) + gt_translation
    transformed_estimate = vertices.dot(rotation.T) + translation
    return float(np.linalg.norm(transformed_gt - transformed_estimate, axis=-1).mean())


def solve_pnp_ransac(
    points_3d,
    points_2d,
    intrinsics,
    *,
    iterations: int = 500,
    reprojection_error: float = 2.0,
    method=cv2.SOLVEPNP_P3P,
):
    status, rvec, tvec, _inliers = cv2.solvePnPRansac(
        points_3d,
        points_2d,
        intrinsics,
        distCoeffs=None,
        iterationsCount=iterations,
        reprojectionError=reprojection_error,
        flags=method,
    )
    if not status:
        LOGGER.warning("Pose could not be estimated from the selected correspondences")
        return None
    return cv2.Rodrigues(rvec)[0], tvec[:, 0]


def normalize_imagenet(image: np.ndarray) -> np.ndarray:
    mean = np.array((0.485, 0.456, 0.406))
    std = np.array((0.229, 0.224, 0.225))
    if image.dtype == np.uint8:
        image = image / 255
    return (image - mean) / std


def match_correspondences(queries: torch.Tensor, surface_features: torch.Tensor, top_k: int = 1):
    scores = torch.log_softmax(queries @ surface_features.T, dim=-1)
    values, indices = torch.topk(scores, k=top_k, dim=-1)
    if top_k == 1:
        return indices[..., 0].cpu(), values
    return indices.cpu(), values


def crop_bop_instance(
    *,
    rgb: np.ndarray,
    mask: np.ndarray,
    camera_matrix: np.ndarray,
    image_size: int,
    crop_scale: float,
    apply_mask: bool,
):
    x, y, width, height = cv2.boundingRect(mask[:, :, 0])
    width -= width % 2
    height -= height % 2
    center_x = x + width / 2
    center_y = y + height / 2
    scale = image_size / max(width, height) / crop_scale
    transform = np.array(((1, 0, -center_x), (0, 1, -center_y)), dtype=np.float32) * scale
    transform[:, 2] += image_size / 2
    crop_intrinsics = np.concatenate((transform, [[0, 0, 1]])) @ camera_matrix
    crop_rgb = cv2.warpAffine(rgb, transform, (image_size, image_size))
    crop_mask = cv2.warpAffine(mask, transform, (image_size, image_size))
    if apply_mask:
        crop_rgb[crop_mask[:, :, 0] == 0] = 0
    return crop_rgb, crop_mask, crop_intrinsics


def scale_intrinsics_for_downsampling(intrinsics: np.ndarray, downsample: int) -> np.ndarray:
    scaled = intrinsics.copy()
    scaled[:2, 2] += 0.5
    scaled[:2] /= downsample
    scaled[:2, 2] -= 0.5
    return scaled


def mesh_vertices(dataset_root: str, object_id: str) -> np.ndarray:
    mesh_path = Path(dataset_root) / "models" / f"obj_{str(object_id).zfill(6)}.ply"
    return np.asarray(trimesh.load_mesh(mesh_path).vertices)


def estimate_surface_normals(mesh_vertices_: np.ndarray, mesh_faces: np.ndarray) -> np.ndarray:
    mesh = trimesh.Trimesh(mesh_vertices_, mesh_faces, process=False)
    return np.asarray(mesh.vertex_normals)


def keep_near_mesh_surface(
    query_points: torch.Tensor,
    mesh_vertices_: np.ndarray,
    mesh_normals: np.ndarray,
    max_distance: float,
):
    surface_tree = KDTree(np.asarray(mesh_vertices_), leaf_size=2)
    distances, indices = surface_tree.query(query_points[0].cpu().numpy(), k=1)
    keep_indices = np.where(distances[:, 0] < max_distance)[0]
    return query_points[:, keep_indices], mesh_normals[indices[:, 0]][keep_indices]
