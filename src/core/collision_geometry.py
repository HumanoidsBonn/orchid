"""Closed target geometry in the input cloud's coordinates (no frame transforms)."""
import math
import os
import tempfile

import numpy as np

import src.core.logger as log


def validate_hull_inflation(factor: float) -> float:
    factor = float(factor)
    if not math.isfinite(factor) or factor < 1.0:
        raise ValueError("collision_hull_inflation_factor must be finite and >= 1")
    return factor


def voxel_collision_mesh(points):
    """Existing regular mesh: cleaned cloud represented by 2 cm voxel cubes."""
    import open3d as o3d

    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Target cloud must have shape (N, 3)")
    points = points[np.isfinite(points).all(axis=1)]
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    cloud, _ = cloud.remove_radius_outlier(nb_points=50, radius=0.05)
    cloud, _ = cloud.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    voxel_size = 0.02
    centres = np.asarray(cloud.voxel_down_sample(voxel_size=voxel_size).points)
    if not len(centres):
        raise ValueError("Point cloud is empty after regular-mesh cleanup")
    mesh = o3d.geometry.TriangleMesh()
    for centre in centres:
        box = o3d.geometry.TriangleMesh.create_box(width=voxel_size, height=voxel_size, depth=voxel_size)
        box.translate(centre - voxel_size / 2.)
        mesh += box
    mesh.merge_close_vertices(0.001)
    mesh.compute_vertex_normals()
    log.info("Regular collision mesh points=%s voxels=%s voxel_size_m=%s bounds=%s..%s",
             len(points), len(centres), voxel_size, mesh.get_min_bound().tolist(), mesh.get_max_bound().tolist())
    return mesh


def inflated_convex_hull(points, inflation_factor: float = 1.1):
    """Drop nonfinite/duplicate points; reject non-volumetric clouds without jitter.

    Inflation is about the mean of hull vertices, leaving coordinates in their
    original frame. Dense clouds retain the existing radius/statistical cleanup;
    small clouds can define a hull without meeting a 50-neighbour density rule.
    """
    import open3d as o3d

    factor = validate_hull_inflation(inflation_factor)
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Target cloud must have shape (N, 3)")
    points = np.unique(points[np.isfinite(points).all(axis=1)], axis=0)
    if len(points) < 4 or np.linalg.matrix_rank(points - points.mean(axis=0)) < 3:
        raise ValueError("Target cloud is empty or degenerate: four noncoplanar points required")
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    if len(points) >= 50:
        cloud, _ = cloud.remove_radius_outlier(nb_points=50, radius=0.05)
        if len(cloud.points) >= 21:
            cloud, _ = cloud.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    cleaned = np.asarray(cloud.points)
    if len(cleaned) < 4 or np.linalg.matrix_rank(cleaned - cleaned.mean(axis=0)) < 3:
        raise ValueError("Target cloud is empty or degenerate after outlier cleanup")
    hull, _ = cloud.compute_convex_hull(joggle_inputs=False)
    oriented = hull.orient_triangles()
    vertices, triangles = np.asarray(hull.vertices), np.asarray(hull.triangles)
    relative = vertices - vertices.mean(axis=0)
    volume = abs(np.einsum("ij,ij->i", relative[triangles[:, 0]],
                          np.cross(relative[triangles[:, 1]], relative[triangles[:, 2]])).sum()) / 6.0
    # Qhull guarantees convexity. Open3D's triangle-intersection check can
    # falsely flag nearly coplanar faces of real depth clouds; check actual
    # boundary topology and volume rather than its is_watertight() shortcut.
    if not oriented or not hull.is_edge_manifold(allow_boundary_edges=False) or not hull.is_vertex_manifold() or not math.isfinite(volume) or volume <= 0:
        raise ValueError("Target hull is not a closed volume")
    original_min, original_max = hull.get_min_bound(), hull.get_max_bound()
    centre = hull.get_center()
    hull.scale(factor, center=centre)
    hull.compute_vertex_normals()
    log.info("Target convex hull points=%s cleaned=%s factor=%s centre=%s original_bounds=%s..%s inflated_bounds=%s..%s",
             len(points), len(cleaned), factor, centre.tolist(), original_min.tolist(), original_max.tolist(),
             hull.get_min_bound().tolist(), hull.get_max_bound().tolist())
    return hull


def write_collision_mesh(mesh) -> str:
    """Give concurrently prepared target/surround meshes distinct backend paths."""
    import open3d as o3d

    fd, path = tempfile.mkstemp(prefix="orchid_collision_", suffix=".stl")
    os.close(fd)
    if not o3d.io.write_triangle_mesh(path, mesh):
        os.unlink(path)
        raise ValueError("Could not write collision mesh")
    return path
