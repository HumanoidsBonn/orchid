"""Pickup geometry and immutable planning contracts, local to the pickup action."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal
import math
import numpy as np

BackoffDirection = Literal["grasp_axis", "horizontal_projection"]
ApproachMode = Literal["auto", "base_only", "arm_only", "coordinated"]
TargetCollisionGeometry = Literal["mesh", "convex_hull"]
BASE_JOINTS = ("odom_x", "odom_y", "odom_t")
ARM_JOINTS = ("arm_lift_joint", "arm_flex_joint", "arm_roll_joint", "wrist_flex_joint", "wrist_roll_joint")
PICKUP_JOINTS = BASE_JOINTS + ARM_JOINTS


@dataclass(frozen=True)
class PickupRequest:
    target: str
    approach_mode: ApproachMode = "auto"
    backoff_direction: BackoffDirection = "grasp_axis"
    pregrasp_distance_m: float = .25
    min_insertion_m: float = .04
    max_insertion_m: float = .60
    clearance_m: float = .03
    post_grasp_lift_m: float = .05
    collision_hull_inflation_factor: float = 2.0
    max_candidates: int = 8
    open_width_m: float = .15
    target_collision_geometry: TargetCollisionGeometry = "mesh"

    def __post_init__(self):
        if self.target_collision_geometry not in {"mesh", "convex_hull"}:
            raise ValueError("target_collision_geometry must be mesh or convex_hull")
        if not math.isfinite(self.open_width_m) or self.open_width_m < 0:
            raise ValueError("open_width_m must be finite and nonnegative")
        if not self.target:
            raise ValueError("target is required")
        if self.approach_mode not in {"auto", "base_only", "arm_only", "coordinated"}:
            raise ValueError("unknown approach_mode")
        if self.backoff_direction not in {"grasp_axis", "horizontal_projection"}:
            raise ValueError("unknown backoff_direction")
        for name in ("pregrasp_distance_m", "min_insertion_m", "max_insertion_m", "clearance_m", "post_grasp_lift_m"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if not 0 < self.min_insertion_m <= self.max_insertion_m:
            raise ValueError("invalid insertion bounds")
        if not math.isfinite(self.collision_hull_inflation_factor) or self.collision_hull_inflation_factor < 1:
            raise ValueError("collision_hull_inflation_factor must be finite and >= 1")
        if isinstance(self.max_candidates, bool) or not isinstance(self.max_candidates, int) or not 1 <= self.max_candidates <= 8:
            raise ValueError("max_candidates must be between 1 and 8")


@dataclass(frozen=True)
class PickupGeometry:
    frame_id: str
    grasp: tuple[tuple[float, ...], ...]
    direction: tuple[float, float, float]
    distances: tuple[float, ...]
    minimum_clearance_backoff_m: float

    def pregrasp(self, distance: float) -> np.ndarray:
        result = np.array(self.grasp)
        result[:3, 3] -= np.array(self.direction) * distance
        return result


def pickup_geometry(request: PickupRequest, grasp: Any, points: Any, frame_id: str,
                    approach_axis: str = "+z") -> PickupGeometry:
    """Back off the hand; the grasp is never adjusted to obtain reachability.

    Require the pregrasp palm to lie behind the observed object's rear extent
    along the insertion direction, with the requested additional clearance.
    """
    pose = np.asarray(grasp, dtype=float)
    cloud = np.asarray(points, dtype=float)
    if pose.shape != (4, 4) or not np.isfinite(pose).all() or not np.allclose(pose[3], [0, 0, 0, 1]):
        raise ValueError("invalid grasp transform")
    if not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-5) or np.linalg.det(pose[:3, :3]) < .99999:
        raise ValueError("invalid grasp rotation")
    if not frame_id or cloud.ndim != 2 or cloud.shape[1] != 3:
        raise ValueError("target geometry and frame are required")
    cloud = cloud[np.isfinite(cloud).all(axis=1)]
    if not len(cloud):
        raise ValueError("empty target geometry")
    if approach_axis not in {"+x", "-x", "+y", "-y", "+z", "-z"}:
        raise ValueError("invalid approach_axis")
    direction = pose[:3, "xyz".index(approach_axis[1])].copy()
    direction *= -1 if approach_axis[0] == "-" else 1
    if request.backoff_direction == "horizontal_projection":
        direction[2] = 0
    length = np.linalg.norm(direction)
    if length < 1e-8:
        raise ValueError("approach direction has no horizontal projection")
    direction /= length
    if request.approach_mode == "base_only" and abs(direction[2]) > 1e-6:
        raise ValueError("base_only requires horizontal insertion; choose geometry explicitly")
    rear_extent = max(0., -float(np.min((cloud - pose[:3, 3]) @ direction)))
    minimum = max(request.min_insertion_m, rear_extent + request.clearance_m)
    if minimum > request.max_insertion_m + 1e-9:
        raise ValueError("required clearance exceeds maximum insertion distance")
    preferred = float(np.clip(request.pregrasp_distance_m, minimum, request.max_insertion_m))
    distances = tuple(dict.fromkeys(round(v, 10) for v in (preferred, minimum, request.max_insertion_m)))
    return PickupGeometry(frame_id, tuple(map(tuple, pose)), tuple(direction), distances, minimum)


@dataclass(frozen=True)
class PickupBackendSettings:
    action_name: str = "/orchid/hsr_pickup/plan"
    timeout_s: float = 180.
    planning_time_s: float = 5.
    cartesian_step_m: float = .005
    position_tolerance_m: float = .005
    orientation_tolerance_rad: float = .02
    base_path_margin_m: float = 0.5 # .15 # .03
    base_yaw_margin_rad: float = 0.5 # 0.15 # .05
    velocity_scale: float = .3
    acceleration_scale: float = .3
    trajectory_time_scale: float = 1.
    jump_revolute_rad: float = .2
    jump_prismatic_m: float = .02
    allow_held_target_octomap_contact: bool = True
    gripper_motor_min_rad: float = -.5
    gripper_motor_max_rad: float = 1.2
    touch_links: tuple[str, ...] = ("hand_palm_link", "hand_l_distal_link", "hand_l_spring_proximal_link", "hand_r_distal_link", "hand_r_spring_proximal_link", "wrist_roll_link")

    @classmethod
    def from_hardware(cls, hardware):
        values = dict(hardware.get("hsr_pickup", {}))
        existing = hardware.get("planners", {}).get("moveit", {}).get("collision_relaxation_links")
        if "touch_links" not in values and existing is not None:
            values["touch_links"] = tuple(existing)
        return cls(**values)

    def __post_init__(self):
        if not math.isfinite(self.trajectory_time_scale) or self.trajectory_time_scale < 1:
            raise ValueError("trajectory_time_scale must be finite and at least 1")
        if not isinstance(self.allow_held_target_octomap_contact, bool):
            raise ValueError("allow_held_target_octomap_contact must be boolean")
        for name in ("timeout_s", "planning_time_s", "cartesian_step_m", "position_tolerance_m", "orientation_tolerance_rad", "base_path_margin_m", "base_yaw_margin_rad", "jump_revolute_rad", "jump_prismatic_m"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not (math.isfinite(self.gripper_motor_min_rad) and math.isfinite(self.gripper_motor_max_rad) and self.gripper_motor_min_rad < self.gripper_motor_max_rad):
            raise ValueError("invalid gripper motor bounds")
        if not 0 < self.velocity_scale <= 1 or not 0 < self.acceleration_scale <= 1:
            raise ValueError("invalid timing scale")


@dataclass(frozen=True)
class PickupObservation:
    grasp: Any
    target_points: Any
    surround_points: Any
    frame_id: str
    approach_axis: str


@dataclass(frozen=True)
class PickupInputs:
    """Planning invocation data, never a runtime scene identity contract."""
    frame_id: str
    start_joints: dict[str, float]
    grasp: Any
    target_points: Any
    surround_points: Any
    robot_description: str
    robot_description_semantic: str
    joint_limits: dict[str, Any]
    scene: Any
    target: Any
    go_joints: dict[str, float]
    approach_axis: str = "+z"


@dataclass(frozen=True)
class PreparedMotion:
    name: str
    trajectory: Any
    start_joints: dict[str, float]
    end_joints: dict[str, float]
    attached: bool = False
    hand_path: tuple = ()
    base_path: tuple = ()


@dataclass(frozen=True)
class PickupPlan:
    target: str
    requested_mode: ApproachMode
    selected_mode: str
    staging_order: str
    inputs: PickupInputs
    motions: tuple[PreparedMotion, ...]
    candidates: tuple[dict[str, Any], ...]
    diagnostics: dict[str, Any] = field(default_factory=dict)
