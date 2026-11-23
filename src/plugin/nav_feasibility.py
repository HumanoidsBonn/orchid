"""Batched navigation feasibility checks for manipulation base poses."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.core.geometry import quaternion_to_yaw, transform_points
from src.core.serialization import to_json_safe
from src.core.manipulation_types import BasePose2D


@dataclass
class FootprintSpec:
    """Robot footprint sampled in the candidate base frame."""

    points: np.ndarray
    source: str


@dataclass
class CostmapSnapshot:
    """Occupancy grid and footprint data captured once for a selection pass."""

    source: str
    frame_id: str
    origin_x: float
    origin_y: float
    resolution: float
    width: int
    height: int
    grid: np.ndarray
    occupied_threshold: int = 50
    unknown_is_blocked: bool = False
    footprint: FootprintSpec | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class NavFeasibilityConfig:
    """Configuration for batched candidate checks."""

    local_costmap_sensor: str = "local_costmap"
    map_sensor: str = "map"
    footprint_sensor: str = "local_footprint"
    missing_policy: str = "allow_with_reason"
    robot_radius: float = 0.3
    footprint: list[list[float]] | None = None
    footprint_inflation_m: float = 0.0
    occupied_threshold: int = 50
    near_goal_radius_m: float = 0.0
    near_goal_occupied_threshold: int | None = None
    unknown_is_blocked: bool = False
    timeout_s: float = 0.2


@dataclass
class NavFeasibilityResult:
    """Feasibility result for one candidate base pose."""

    feasible: bool
    reason: str | None = None
    source: str | None = None
    frame_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": bool(self.feasible),
            "feasible": bool(self.feasible),
            "reason": self.reason,
            "source": self.source,
            "frame_id": self.frame_id,
            "metadata": to_json_safe(self.metadata),
        }


class NavigationFeasibilityChecker:
    """Snapshot-backed vectorized feasibility checker for base candidates."""

    def __init__(self, sensor_data_center: Any, config: NavFeasibilityConfig | None = None) -> None:
        self.sdc = sensor_data_center
        self.config = config or NavFeasibilityConfig()
        self._snapshot: CostmapSnapshot | None = None
        self._missing_reason: str | None = None

    def check_candidates(self, candidates: list[BasePose2D]) -> list[dict[str, Any]]:
        """Return one feasibility dict per candidate, using one grid snapshot."""
        if not candidates:
            return []
        snapshot = self.snapshot()
        if snapshot is None:
            feasible = self.config.missing_policy != "reject"
            result = NavFeasibilityResult(
                feasible=feasible,
                reason=self._missing_reason or "missing_costmap_and_map",
                metadata={
                    "missing_policy": self.config.missing_policy,
                    "hard_nav_missing": not feasible,
                },
            ).to_dict()
            return [dict(result) for _ in candidates]
        return [item.to_dict() for item in check_candidates_against_snapshot(candidates, snapshot, self.sdc)]

    def check_candidate(self, candidate: BasePose2D) -> dict[str, Any]:
        """Compatibility wrapper for code paths that still score one pose."""
        return self.check_candidates([candidate])[0]

    def snapshot(self) -> CostmapSnapshot | None:
        """Fetch local costmap/map and footprint at most once."""
        if self._snapshot is not None or self._missing_reason is not None:
            return self._snapshot
        missing_reasons: list[str] = []
        for sensor_name, source_label in (
            (self.config.local_costmap_sensor, "local_costmap"),
            (self.config.map_sensor, "map"),
        ):
            if not sensor_name:
                continue
            msg = self.sdc.get_latest(sensor_name, timeout_s=self.config.timeout_s)
            if msg is None or msg.data is None:
                missing_reasons.append(f"missing_{source_label}")
                continue
            snapshot = build_costmap_snapshot(
                msg.data,
                source_label=source_label,
                occupied_threshold=self.config.occupied_threshold,
                unknown_is_blocked=self.config.unknown_is_blocked,
                footprint=resolve_footprint(self.sdc, self.config),
            )
            if snapshot is None:
                missing_reasons.append(f"invalid_{source_label}")
                continue
            snapshot.metadata["near_goal_radius_m"] = float(self.config.near_goal_radius_m)
            snapshot.metadata["near_goal_occupied_threshold"] = int(
                self.config.near_goal_occupied_threshold
                if self.config.near_goal_occupied_threshold is not None
                else self.config.occupied_threshold
            )
            self._snapshot = snapshot
            return snapshot
        self._missing_reason = ",".join(missing_reasons) or "missing_costmap_and_map"
        return None


def build_costmap_snapshot(
    grid_msg: Any,
    *,
    source_label: str,
    occupied_threshold: int,
    unknown_is_blocked: bool,
    footprint: FootprintSpec,
) -> CostmapSnapshot | None:
    """Convert an OccupancyGrid-like message into a checked snapshot."""
    info = getattr(grid_msg, "info", None)
    if info is None:
        return None
    width = int(getattr(info, "width", 0) or 0)
    height = int(getattr(info, "height", 0) or 0)
    resolution = float(getattr(info, "resolution", 0.0) or 0.0)
    if width <= 0 or height <= 0 or resolution <= 0.0:
        return None
    data = np.asarray(getattr(grid_msg, "data", []), dtype=np.int16)
    if data.size < width * height:
        return None
    origin = getattr(info, "origin", None)
    origin_pos = getattr(origin, "position", None)
    frame_id = _extract_frame_id(grid_msg)
    if not frame_id:
        return None
    return CostmapSnapshot(
        source=source_label,
        frame_id=frame_id,
        origin_x=float(getattr(origin_pos, "x", 0.0)),
        origin_y=float(getattr(origin_pos, "y", 0.0)),
        resolution=resolution,
        width=width,
        height=height,
        grid=data[: width * height].reshape((height, width)),
        occupied_threshold=int(occupied_threshold),
        unknown_is_blocked=bool(unknown_is_blocked),
        footprint=footprint,
    )


def check_candidates_against_snapshot(
    candidates: list[BasePose2D],
    snapshot: CostmapSnapshot,
    sensor_data_center: Any,
) -> list[NavFeasibilityResult]:
    """Vectorized footprint/grid check for all candidates."""
    if not candidates:
        return []
    frame_id = candidates[0].frame_id
    if any(candidate.frame_id != frame_id for candidate in candidates):
        return [_single_frame_mismatch(candidate, snapshot) for candidate in candidates]

    centers = np.asarray([[pose.x, pose.y, 0.0] for pose in candidates], dtype=np.float32)
    yaws = np.asarray([pose.yaw for pose in candidates], dtype=np.float32)
    if frame_id != snapshot.frame_id:
        tf = sensor_data_center.get_pose_at(frame_id, snapshot.frame_id)
        if tf is None:
            return [
                NavFeasibilityResult(
                    False,
                    reason="missing_candidate_to_grid_tf",
                    source=snapshot.source,
                    frame_id=snapshot.frame_id,
                    metadata={"candidate_frame": pose.frame_id, "grid_frame": snapshot.frame_id},
                )
                for pose in candidates
            ]
        centers = transform_points(centers, tf.get("mat", np.eye(4)))
        orientation = tf.get("orientation", {}) or {}
        yaw_delta = quaternion_to_yaw(
            float(orientation.get("x", 0.0)),
            float(orientation.get("y", 0.0)),
            float(orientation.get("z", 0.0)),
            float(orientation.get("w", 1.0)),
        )
        yaws = yaws + float(yaw_delta)

    footprint = snapshot.footprint.points if snapshot.footprint else _circle_footprint(0.3)
    footprint = np.asarray(footprint, dtype=np.float32)
    if footprint.ndim != 2 or footprint.shape[1] != 2 or footprint.shape[0] == 0:
        return [
            NavFeasibilityResult(False, reason="empty_footprint_cells", source=snapshot.source, frame_id=snapshot.frame_id)
            for _ in candidates
        ]
    cos_yaw = np.cos(yaws)[:, None]
    sin_yaw = np.sin(yaws)[:, None]
    local_x = footprint[None, :, 0]
    local_y = footprint[None, :, 1]
    world_x = centers[:, 0:1] + cos_yaw * local_x - sin_yaw * local_y
    world_y = centers[:, 1:2] + sin_yaw * local_x + cos_yaw * local_y
    results: list[NavFeasibilityResult] = []
    for index, pose in enumerate(candidates):
        polygon = np.column_stack((world_x[index], world_y[index])).astype(np.float32)
        footprint_cells = _polygon_grid_cells(polygon, snapshot)
        footprint_blocked = _blocked_cells_for_uv(
            footprint_cells,
            snapshot,
            occupied_threshold=snapshot.occupied_threshold,
            unknown_is_blocked=snapshot.unknown_is_blocked,
            reason_prefix="footprint",
        )
        near_goal_cells = _circle_grid_cells(
            float(centers[index, 0]),
            float(centers[index, 1]),
            float(snapshot.metadata.get("near_goal_radius_m", 0.0) or 0.0),
            snapshot,
        )
        near_goal_blocked = _blocked_cells_for_uv(
            near_goal_cells,
            snapshot,
            occupied_threshold=int(snapshot.metadata.get("near_goal_occupied_threshold", snapshot.occupied_threshold)),
            unknown_is_blocked=snapshot.unknown_is_blocked,
            reason_prefix="near_goal",
        )
        blocked_cells = footprint_blocked + near_goal_blocked
        metadata = {
            "source": snapshot.source,
            "grid_frame": snapshot.frame_id,
            "candidate_frame": pose.frame_id,
            "candidate_in_grid": {"x": float(centers[index, 0]), "y": float(centers[index, 1]), "yaw": float(yaws[index])},
            "checked_cells": int(len(footprint_cells) + len(near_goal_cells)),
            "footprint_checked_cells": int(len(footprint_cells)),
            "near_goal_checked_cells": int(len(near_goal_cells)),
            "blocked_cells": blocked_cells,
            "blocked_count": int(len(blocked_cells)),
            "footprint_blocked_count": int(len(footprint_blocked)),
            "near_goal_blocked_count": int(len(near_goal_blocked)),
            "footprint_source": snapshot.footprint.source if snapshot.footprint else "default_radius",
        }
        if blocked_cells:
            reason = _blocked_reason(snapshot.source, blocked_cells)
            results.append(NavFeasibilityResult(False, reason=reason, source=snapshot.source, frame_id=snapshot.frame_id, metadata=metadata))
        else:
            results.append(NavFeasibilityResult(True, reason=None, source=snapshot.source, frame_id=snapshot.frame_id, metadata=metadata))
    return results


def resolve_footprint(sensor_data_center: Any, config: NavFeasibilityConfig) -> FootprintSpec:
    """Fetch footprint once, then fall back to manifest footprint or radius."""
    msg = sensor_data_center.get_latest(config.footprint_sensor, timeout_s=config.timeout_s) if config.footprint_sensor else None
    if msg is not None and msg.data is not None:
        points = getattr(getattr(msg.data, "polygon", None), "points", None)
        if points:
            footprint = np.asarray([(float(point.x), float(point.y)) for point in points], dtype=np.float32)
            if footprint.shape[0] >= 3:
                footprint = _inflate_footprint(footprint, float(config.footprint_inflation_m))
                return FootprintSpec(points=footprint, source="published_footprint")
    if config.footprint and len(config.footprint) >= 3:
        footprint = _inflate_footprint(np.asarray(config.footprint, dtype=np.float32), float(config.footprint_inflation_m))
        return FootprintSpec(points=footprint, source="manifest_footprint")
    radius = float(config.robot_radius) + max(0.0, float(config.footprint_inflation_m))
    return FootprintSpec(points=_circle_footprint(radius), source="robot_radius")


def _circle_footprint(radius: float, samples: int = 16) -> np.ndarray:
    angles = np.linspace(0.0, 2.0 * np.pi, num=samples, endpoint=False, dtype=np.float32)
    return np.column_stack((float(radius) * np.cos(angles), float(radius) * np.sin(angles))).astype(np.float32)


def _inflate_footprint(points: np.ndarray, inflation_m: float) -> np.ndarray:
    inflation = max(0.0, float(inflation_m))
    if inflation <= 0.0:
        return points.astype(np.float32)
    center = np.mean(points[:, :2], axis=0)
    vectors = points[:, :2] - center[None, :]
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    directions = np.divide(vectors, norms, out=np.zeros_like(vectors), where=norms > 1e-6)
    return (points[:, :2] + directions * inflation).astype(np.float32)


def _polygon_grid_cells(polygon_xy: np.ndarray, snapshot: CostmapSnapshot) -> list[tuple[int, int]]:
    if polygon_xy.shape[0] < 3:
        return []
    min_u = int(np.floor((float(np.min(polygon_xy[:, 0])) - snapshot.origin_x) / snapshot.resolution))
    max_u = int(np.floor((float(np.max(polygon_xy[:, 0])) - snapshot.origin_x) / snapshot.resolution))
    min_v = int(np.floor((float(np.min(polygon_xy[:, 1])) - snapshot.origin_y) / snapshot.resolution))
    max_v = int(np.floor((float(np.max(polygon_xy[:, 1])) - snapshot.origin_y) / snapshot.resolution))
    if max_u < min_u or max_v < min_v:
        return []
    us = np.arange(min_u, max_u + 1, dtype=np.int32)
    vs = np.arange(min_v, max_v + 1, dtype=np.int32)
    if us.size == 0 or vs.size == 0:
        return []
    uu, vv = np.meshgrid(us, vs)
    xs = snapshot.origin_x + (uu.astype(np.float32) + 0.5) * snapshot.resolution
    ys = snapshot.origin_y + (vv.astype(np.float32) + 0.5) * snapshot.resolution
    inside = _points_in_polygon(xs.reshape(-1), ys.reshape(-1), polygon_xy)
    flat_u = uu.reshape(-1)[inside]
    flat_v = vv.reshape(-1)[inside]
    if flat_u.size == 0:
        # A very small footprint can fit between cell centers; check at least the vertices.
        vertex_u = np.floor((polygon_xy[:, 0] - snapshot.origin_x) / snapshot.resolution).astype(np.int32)
        vertex_v = np.floor((polygon_xy[:, 1] - snapshot.origin_y) / snapshot.resolution).astype(np.int32)
        return list(dict.fromkeys((int(u), int(v)) for u, v in zip(vertex_u, vertex_v, strict=False)))
    return list(dict.fromkeys((int(u), int(v)) for u, v in zip(flat_u, flat_v, strict=False)))


def _circle_grid_cells(x: float, y: float, radius_m: float, snapshot: CostmapSnapshot) -> list[tuple[int, int]]:
    radius = max(0.0, float(radius_m))
    if radius <= 0.0:
        return []
    min_u = int(np.floor((x - radius - snapshot.origin_x) / snapshot.resolution))
    max_u = int(np.floor((x + radius - snapshot.origin_x) / snapshot.resolution))
    min_v = int(np.floor((y - radius - snapshot.origin_y) / snapshot.resolution))
    max_v = int(np.floor((y + radius - snapshot.origin_y) / snapshot.resolution))
    us = np.arange(min_u, max_u + 1, dtype=np.int32)
    vs = np.arange(min_v, max_v + 1, dtype=np.int32)
    uu, vv = np.meshgrid(us, vs)
    xs = snapshot.origin_x + (uu.astype(np.float32) + 0.5) * snapshot.resolution
    ys = snapshot.origin_y + (vv.astype(np.float32) + 0.5) * snapshot.resolution
    inside = (xs - float(x)) ** 2 + (ys - float(y)) ** 2 <= radius ** 2
    return list(dict.fromkeys((int(u), int(v)) for u, v in zip(uu.reshape(-1)[inside.reshape(-1)], vv.reshape(-1)[inside.reshape(-1)], strict=False)))


def _points_in_polygon(xs: np.ndarray, ys: np.ndarray, polygon_xy: np.ndarray) -> np.ndarray:
    inside = np.zeros(xs.shape, dtype=bool)
    x_vertices = polygon_xy[:, 0]
    y_vertices = polygon_xy[:, 1]
    j = polygon_xy.shape[0] - 1
    for i in range(polygon_xy.shape[0]):
        yi = y_vertices[i]
        yj = y_vertices[j]
        intersects = ((yi > ys) != (yj > ys)) & (
            xs < (x_vertices[j] - x_vertices[i]) * (ys - yi) / ((yj - yi) if abs(yj - yi) > 1e-9 else 1e-9) + x_vertices[i]
        )
        inside ^= intersects
        j = i
    return inside


def _blocked_cells_for_uv(
    cells: list[tuple[int, int]],
    snapshot: CostmapSnapshot,
    *,
    occupied_threshold: int,
    unknown_is_blocked: bool,
    reason_prefix: str,
) -> list[dict[str, Any]]:
    blocked: list[dict[str, Any]] = []
    for u, v in cells:
        if u < 0 or u >= snapshot.width or v < 0 or v >= snapshot.height:
            blocked.append({"u": int(u), "v": int(v), "value": None, "reason": "out_of_bounds", "check": reason_prefix})
            continue
        value = int(snapshot.grid[v, u])
        if value >= int(occupied_threshold):
            blocked.append({"u": int(u), "v": int(v), "value": value, "reason": "occupied", "check": reason_prefix})
        elif value < 0 and bool(unknown_is_blocked):
            blocked.append({"u": int(u), "v": int(v), "value": value, "reason": "unknown", "check": reason_prefix})
    return blocked


def _extract_frame_id(message: Any) -> str | None:
    header = getattr(message, "header", None)
    frame_id = getattr(header, "frame_id", None)
    return str(frame_id) if frame_id else None


def _blocked_reason(source: str, blocked_cells: list[dict[str, Any]]) -> str:
    reasons = {cell.get("reason") for cell in blocked_cells}
    if "out_of_bounds" in reasons:
        return "candidate_out_of_grid"
    if "unknown" in reasons:
        return "candidate_unknown_grid"
    checks = {cell.get("check") for cell in blocked_cells}
    if "near_goal" in checks:
        return "near_goal_costmap_blocked"
    if "footprint" in checks:
        return "footprint_costmap_blocked"
    return "near_goal_costmap_blocked" if source == "local_costmap" else "map_occupied"


def _single_frame_mismatch(candidate: BasePose2D, snapshot: CostmapSnapshot) -> NavFeasibilityResult:
    return NavFeasibilityResult(
        False,
        reason="mixed_candidate_frames",
        source=snapshot.source,
        frame_id=snapshot.frame_id,
        metadata={"candidate_frame": candidate.frame_id, "grid_frame": snapshot.frame_id},
    )
