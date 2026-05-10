from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass(frozen=True)
class StageArtifacts:
    root: Path
    checkpoints: Path
    previews: Path
    meshes: Path
    features: Path
    cache: Path
    metadata: Path

    def create(self) -> "StageArtifacts":
        for d in (self.root, self.checkpoints, self.previews, self.meshes,
                  self.features, self.cache, self.metadata):
            d.mkdir(parents=True, exist_ok=True)
        return self


def stage_artifacts(output_root: str | Path, object_id: str, stage: str) -> StageArtifacts:
    root = Path(output_root) / "objects" / str(object_id) / stage
    return StageArtifacts(
        root=root,
        checkpoints=root / "checkpoints",
        previews=root / "previews",
        meshes=root / "meshes",
        features=root / "features",
        cache=root / "cache",
        metadata=root / "metadata",
    )


def save_checkpoint(path: Path, step: int, model: torch.nn.Module) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"step": step, "model_state_dict": model.state_dict()}, path)


def save_step_checkpoint(directory: Path, prefix: str, step: int, model: torch.nn.Module) -> None:
    save_checkpoint(directory / f"{prefix}_step_{step:06d}.pt", step, model)
    save_checkpoint(directory / f"{prefix}_latest.pt", step, model)


def save_numpy(path: Path, array) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array)


def save_ply(path: Path, vertices: np.ndarray, triangles: np.ndarray | None = None) -> None:
    import open3d as o3d

    path.parent.mkdir(parents=True, exist_ok=True)
    if triangles is not None:
        geometry = o3d.geometry.TriangleMesh()
        geometry.vertices = o3d.utility.Vector3dVector(vertices.astype(np.float64))
        geometry.triangles = o3d.utility.Vector3iVector(triangles.astype(np.int32))
        geometry.compute_vertex_normals()
        o3d.io.write_triangle_mesh(str(path), geometry)
    else:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(vertices.astype(np.float64))
        o3d.io.write_point_cloud(str(path), pcd)


def latest_nerf_artifacts(output_root: str | Path, object_id: str) -> dict[str, Path | None]:
    """Return paths to the most recent NeRF outputs for one object.

    Keys:
      coarse_checkpoint  — coarse_field_latest.pt
      fine_checkpoint    — fine_field_latest.pt
      coarse_mesh        — newest coarse mesh .ply by modification time
      fine_mesh          — newest fine mesh .ply by modification time

    Values are None when the file does not exist yet.
    """
    arts = stage_artifacts(output_root, object_id, "radiance_field")

    def _latest_ply(pattern: str) -> Path | None:
        candidates = sorted(arts.meshes.glob(pattern), key=lambda p: p.stat().st_mtime)
        return candidates[-1] if candidates else None

    return {
        "coarse_checkpoint": arts.checkpoints / "coarse_field_latest.pt",
        "fine_checkpoint": arts.checkpoints / "fine_field_latest.pt",
        "coarse_mesh": _latest_ply("*coarse_mesh*.ply"),
        "fine_mesh": _latest_ply("*fine_mesh*.ply"),
    }


def correspondence_cache_dirs(cache_root: str | Path, render_size: int) -> dict[str, Path]:
    root = Path(cache_root) / f"render_{render_size}"
    return {
        "surface_ray_xys": root / "surface_ray_xys",
        "surface_points": root / "surface_points",
        "background_points": root / "background_points",
        "background_ray_xys": root / "background_ray_xys",
    }
