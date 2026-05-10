import numpy as np
import open3d as o3d


def make_point_cloud(points: np.ndarray, color_channel: int = 0) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    colors = np.zeros((points.shape[0], 3))
    colors[:, color_channel] = 1.0
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def make_line_set(points_a: np.ndarray, points_b: np.ndarray) -> o3d.geometry.LineSet:
    n = points_a.shape[0]
    all_points = np.concatenate([points_a, points_b], axis=0)
    indices = np.stack([np.arange(n), n + np.arange(n)], axis=1)
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(all_points)
    line_set.lines = o3d.utility.Vector2iVector(indices)
    line_set.colors = o3d.utility.Vector3dVector(np.random.rand(n, 3))
    return line_set


def visualize_correspondences(
    query_points_3d: np.ndarray,
    matched_points_3d: np.ndarray,
    surface_points: np.ndarray,
    matched_colors: np.ndarray | None = None,
    indices: np.ndarray | None = None,
) -> None:
    """Open an interactive Open3D window showing 2D-3D correspondences.

    Args:
        query_points_3d:   (N, 3) query points projected into 3D space.
        matched_points_3d: (N, 3) matched surface points.
        surface_points:    (M, 3) full surface point cloud for context.
        matched_colors:    (N, 3) per-point RGB colours for matched points.
        indices:           Subset of correspondence indices to draw lines for.
                           Draws all if None.
    """
    if indices is not None:
        line_set = make_line_set(query_points_3d[indices], matched_points_3d[indices])
    else:
        line_set = make_line_set(query_points_3d, matched_points_3d)

    pcd_query = make_point_cloud(query_points_3d, color_channel=0)
    pcd_matched = make_point_cloud(matched_points_3d, color_channel=1)
    if matched_colors is not None:
        pcd_matched.colors = o3d.utility.Vector3dVector(matched_colors)
    pcd_surface = make_point_cloud(surface_points, color_channel=2)

    o3d.visualization.draw_geometries([line_set, pcd_surface, pcd_query, pcd_matched])
