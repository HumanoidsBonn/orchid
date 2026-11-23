"""WorldState: shared observation and execution history for the robot session.

Provides:
- Key-value store for observations (detected objects, poses, etc.)
- Action execution history (sliding window of last N actions)
- Access to scene graph for spatial queries
"""

from __future__ import annotations

from collections import deque
import time
from dataclasses import dataclass, field
from typing import Any
from datetime import datetime

import numpy as np

from .action import ActionResult
from .manipulation_types import ReachTarget
from .scene_graph import SceneGraph
from .serialization import to_json_safe


@dataclass
class ActionRecord:
    """Single action execution record."""

    action_name: str
    params: dict[str, Any]
    result: ActionResult
    timestamp: float
    duration_s: float
    bt_node_name: str | None = None
    action_class: str | None = None
    action_type: str | None = None
    planner_visible: bool = True

    def to_dict(self) -> dict:
        """Serialize to dict for logging."""
        return {
            "bt_node_name": self.bt_node_name,
            "action_name": self.action_name,
            "action_class": self.action_class or self.action_name,
            "action_type": self.action_type or self.action_name,
            "planner_visible": self.planner_visible,
            "params": to_json_safe(self.params),
            "result": {
                "status": self.result.status,
                "reason": self.result.reason,
                "payload": to_json_safe(self.result.payload),
                "suggested_replan_context": to_json_safe(self.result.suggested_replan_context),
                "force_replan": bool(self.result.force_replan),
            },
            "timestamp": self.timestamp,
            "duration_s": self.duration_s,
        }


@dataclass
class MovableObjectState:
    """Timestamped state for a movable object observed by perception."""

    object_id: str
    pose_matrix: Any
    frame_id: str
    timestamp: float = field(default_factory=time.time)
    source_action: str | None = None
    confidence: float | None = None
    object_pcd_key: str | None = None
    surround_pcd_key: str | None = None
    source_camera_frame: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def age_s(self, now: float | None = None) -> float:
        return max(0.0, float(now if now is not None else time.time()) - float(self.timestamp))

    def is_fresh(self, ttl_s: float | None, now: float | None = None) -> bool:
        if ttl_s is None:
            return True
        return self.age_s(now) <= float(ttl_s)

    def require_valid_pose(self) -> None:
        if not self.frame_id:
            raise ValueError(f"Movable object '{self.object_id}' is missing frame_id")
        pose = np.asarray(self.pose_matrix, dtype=float)
        if pose.shape != (4, 4):
            raise ValueError(f"Movable object '{self.object_id}' has invalid pose matrix shape {pose.shape}")

    def centroid(self) -> tuple[float, float, float]:
        self.require_valid_pose()
        pose = np.asarray(self.pose_matrix, dtype=float)
        return (float(pose[0, 3]), float(pose[1, 3]), float(pose[2, 3]))

    def reach_target(
        self,
        *,
        object_points: Any | None = None,
        clearance_m: float = 0.06,
        height_guess_m: float = 0.12,
        frame_id: str | None = None,
    ) -> ReachTarget:
        """Derive the position-only reach target for this object."""
        x, y, z = self.centroid()
        target_frame = str(frame_id or self.frame_id)
        if object_points is not None:
            points = np.asarray(object_points, dtype=float)
            if points.ndim == 2 and points.shape[0] > 0 and points.shape[1] >= 3:
                top_z = float(np.nanmax(points[:, 2]))
                if np.isfinite(top_z):
                    return ReachTarget(
                        x=x,
                        y=y,
                        z=top_z + float(clearance_m),
                        frame_id=target_frame,
                        source="object_top_plus_margin",
                    )
        return ReachTarget(
            x=x,
            y=y,
            z=z + 0.5 * float(height_guess_m) + float(clearance_m),
            frame_id=target_frame,
            source="centroid_plus_height_guess",
        )

    def object_height(self, *, object_points: Any | None = None) -> float | None:
        """Return observed object height from point cloud geometry when available."""
        if object_points is None and self.object_pcd_key:
            object_points = self.metadata.get("object_points")
        if object_points is None:
            return None
        try:
            points = np.asarray(object_points, dtype=float)
            if points.ndim != 2 or points.shape[0] == 0 or points.shape[1] < 3:
                return None
            z_values = points[:, 2]
            z_values = z_values[np.isfinite(z_values)]
            if z_values.size == 0:
                return None
            height = float(np.nanmax(z_values) - np.nanmin(z_values))
            return height if np.isfinite(height) and height > 0.0 else None
        except Exception:
            return None

    def pick_z_increment(
        self,
        *,
        object_points: Any | None = None,
        requested_z_inc_m: float = 0.25,
        clearance_m: float = 0.06,
    ) -> tuple[float, dict[str, Any]]:
        """Choose a pick approach/retreat z increment from request and object height."""
        requested = abs(float(requested_z_inc_m))
        height = self.object_height(object_points=object_points)
        if height is None:
            return requested, {
                "source": "requested",
                "requested_z_inc_m": requested,
                "object_height_m": None,
                "clearance_m": float(clearance_m),
            }
        geometry_inc = float(height) + float(clearance_m)
        selected = max(requested, geometry_inc)
        return selected, {
            "source": "object_height_plus_clearance" if selected == geometry_inc else "requested",
            "requested_z_inc_m": requested,
            "object_height_m": height,
            "clearance_m": float(clearance_m),
            "object_height_z_inc_m": geometry_inc,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "object_id": self.object_id,
            "pose_matrix": to_json_safe(self.pose_matrix),
            "frame_id": self.frame_id,
            "timestamp": self.timestamp,
            "source_action": self.source_action,
            "confidence": self.confidence,
            "object_pcd_key": self.object_pcd_key,
            "surround_pcd_key": self.surround_pcd_key,
            "source_camera_frame": self.source_camera_frame,
            "metadata": to_json_safe(self.metadata),
        }


class WorldState:
    """Shared state container for action execution and observations.

    Attributes:
        max_history: Maximum number of action records to keep (sliding window)
    """

    def __init__(self, max_history: int = 10):
        """Initialize WorldState.

        Args:
            max_history: Number of action records to retain (older evicted)
        """
        self._data: dict[str, Any] = {}
        self._scene_graph: SceneGraph = SceneGraph()
        self._action_history: deque[ActionRecord] = deque(maxlen=max_history)

    def get(self, key: str) -> Any:
        """Retrieve observation by key.

        Args:
            key: Observation key (e.g., "detected_objects", "current_pose")

        Returns:
            The value, or None if not found
        """
        return self._data.get(key)
    
    def list(self) -> Any:
        """List all entries in data.

        Returns:
            All keys in data
        """
        return self._data.keys()

    def set(self, key: str, value: Any) -> None:
        """Store observation.

        Args:
            key: Observation key
            value: Value to store
        """
        self._data[key] = value

    def update(self, data: dict[str, Any]) -> None:
        """Batch update observations.

        Args:
            data: Dict of key-value pairs to merge
        """
        self._data.update(data)

    def set_movable_object_state(
        self,
        object_id: str,
        *,
        pose_matrix: Any,
        frame_id: str,
        source_action: str | None = None,
        confidence: float | None = None,
        object_pcd_key: str | None = None,
        surround_pcd_key: str | None = None,
        source_camera_frame: str | None = None,
        metadata: dict[str, Any] | None = None,
        timestamp: float | None = None,
    ) -> MovableObjectState:
        """Store timestamped pose and perception metadata for a movable object."""
        state = MovableObjectState(
            object_id=object_id,
            pose_matrix=pose_matrix,
            frame_id=frame_id,
            timestamp=float(timestamp if timestamp is not None else time.time()),
            source_action=source_action,
            confidence=confidence,
            object_pcd_key=object_pcd_key,
            surround_pcd_key=surround_pcd_key,
            source_camera_frame=source_camera_frame,
            metadata=metadata or {},
        )
        self._data[f"{object_id}_movable_object_state"] = state
        return state

    def get_movable_object_state(
        self,
        object_id: str,
        *,
        ttl_s: float | None = None,
        now: float | None = None,
    ) -> MovableObjectState | None:
        """Return a movable-object state when present and fresh enough."""
        value = self._data.get(f"{object_id}_movable_object_state")
        if isinstance(value, MovableObjectState):
            state = value
        elif isinstance(value, dict):
            try:
                state = MovableObjectState(
                    object_id=str(value.get("object_id", object_id)),
                    pose_matrix=value.get("pose_matrix"),
                    frame_id=str(value.get("frame_id")),
                    timestamp=float(value.get("timestamp", 0.0)),
                    source_action=value.get("source_action"),
                    confidence=value.get("confidence"),
                    object_pcd_key=value.get("object_pcd_key"),
                    surround_pcd_key=value.get("surround_pcd_key"),
                    source_camera_frame=value.get("source_camera_frame"),
                    metadata=value.get("metadata") or {},
                )
            except Exception:
                return None
        else:
            return None
        if not state.frame_id or state.pose_matrix is None:
            return None
        if not state.is_fresh(ttl_s, now=now):
            return None
        return state

    def get_movable_object_reach_target(
        self,
        object_id: str,
        *,
        ttl_s: float | None = None,
        now: float | None = None,
        clearance_m: float = 0.06,
        height_guess_m: float = 0.12,
        frame_id: str | None = None,
    ) -> ReachTarget | None:
        """Return a derived reach target for a fresh movable object."""
        state = self.get_movable_object_state(object_id, ttl_s=ttl_s, now=now)
        if state is None:
            return None
        object_points = self.get(state.object_pcd_key) if state.object_pcd_key else None
        return state.reach_target(
            object_points=object_points,
            clearance_m=clearance_m,
            height_guess_m=height_guess_m,
            frame_id=frame_id,
        )

    def get_movable_object_pick_z_increment(
        self,
        object_id: str,
        *,
        object_points: Any | None = None,
        requested_z_inc_m: float = 0.25,
        clearance_m: float = 0.06,
    ) -> tuple[float, dict[str, Any]]:
        """Return pick z increment using movable-object geometry when possible."""
        state = self.get_movable_object_state(object_id)
        if state is None:
            identity = np.eye(4)
            state = MovableObjectState(
                object_id=object_id,
                pose_matrix=identity,
                frame_id="",
                object_pcd_key=f"{object_id}_object_pcd",
            )
        if object_points is None and state.object_pcd_key:
            object_points = self.get(state.object_pcd_key)
        return state.pick_z_increment(
            object_points=object_points,
            requested_z_inc_m=requested_z_inc_m,
            clearance_m=clearance_m,
        )

    def append_action_record(self, record: ActionRecord) -> None:
        """Add action to history.

        Automatically evicts oldest record if max_history exceeded.

        Args:
            record: ActionRecord to add
        """
        self._action_history.append(record)

    def get_action_history(self) -> list[ActionRecord]:
        """Return recent action execution history.

        Returns:
            List of ActionRecords in execution order (oldest first)
        """
        return list(self._action_history)

    @property
    def scene_graph(self) -> SceneGraph:
        """Access the scene graph.

        Returns:
            The SceneGraph instance
        """
        return self._scene_graph

    def to_dict(self) -> dict:
        """Serialize WorldState to dict (for logging/debugging).

        Returns:
            Dict representation of current state
        """
        return {
            "observations": to_json_safe(self._data),
            "action_history": [rec.to_dict() for rec in self._action_history],
            "scene_graph": self._scene_graph.to_dict(),
        }
