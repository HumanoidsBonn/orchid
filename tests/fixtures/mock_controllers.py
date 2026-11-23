from src.core.arm_request import ArmPoseRequest
"""Mock controller implementations for testing."""

from typing import Literal, Tuple

from src.core.controllers import (
    BaseArmController,
    BaseGripperController,
    BaseNavController,
    ControllerResult,
    Pose,
    Pose3D,
)


class MockNavController(BaseNavController):
    """Mock navigation controller for testing."""

    def __init__(self, hardware_config: dict | None = None):
        super().__init__(hardware_config or {})
        self.move_count = 0
        self.current_status = "idle"
        self.current_pose = Pose(x=0.0, y=0.0, yaw=0.0)
        self.last_target: Tuple[float, float, float] | None = None
        self.last_frame_id: str | None = None

    def send_goal(self, target: Tuple[float, float, float], frame_id: str | None = None) -> None:
        self.move_count += 1
        self.current_status = "in_progress"
        self.last_target = target
        self.last_frame_id = frame_id

    def update_goal(self, target: Tuple[float, float, float], frame_id: str | None = None) -> None:
        self.last_target = target
        self.last_frame_id = frame_id
        self.current_status = "in_progress"

    def get_status(self) -> Literal["in_progress", "succeeded", "failed", "timeout", "idle"]:
        if self.current_status == "in_progress" and self.move_count > 0:
            return "succeeded"
        if self.move_count == 0:
            return "idle"
        return self.current_status  # type: ignore[return-value]

    def get_error(self) -> str | None:
        return None

    def cancel_goal(self) -> None:
        self.current_status = "idle"

    def get_current_pose(self) -> Pose:
        if self.last_target:
            return Pose(x=self.last_target[0], y=self.last_target[1], yaw=self.last_target[2])
        return self.current_pose


class MockArmController(BaseArmController):
    """Mock arm controller for testing."""

    def __init__(self, hardware_config: dict | None = None):
        super().__init__(hardware_config or {})
        self.move_count = 0
        self.current_status = "idle"
        self.last_pose: Pose3D | None = None
        self.pose_goals: list[Pose3D] = []
        self.pose_goal_options: list[dict] = []
        self.pose_requests = []
        self.last_named_goal: str | None = None
        self.named_goal_history: list[str] = []
        self.last_joint_goal: dict[str, float] | None = None
        self.last_trajectory = None
        self.collision_events: list[tuple[str, str | None]] = []
        self.allowed_collision_pairs: list[tuple[str, str]] = []
        self.last_base_pose_goal = None
        self.base_pose_goal_options: dict | None = None

    def send_pose_goal(self, request: ArmPoseRequest) -> None:
        pose = request.pose
        options = {"segment": request.segment, "planning_scope": request.scope, "planning_motion": request.motion, "looking_hand_constraint": request.looking_hand_constraint, "line_axis": list(request.line.axis) if request.line.axis else None, "line_distance": request.line.distance, "moveit_group": request.moveit.group_name, "planning_strategy": request.moveit.planning_strategy, "moveit_goal_mode": request.moveit.goal_mode, "constrain_base": request.moveit.constrain_base, "base_pose": request.moveit.base_pose.to_dict() if request.moveit.base_pose else None, "base_constraint_tolerance": request.moveit.base_constraint_tolerance, "max_base_path_drift": request.moveit.max_base_path_drift, "collision_policy": request.moveit.collision_policy}
        self.pose_requests.append(request)
        self.move_count += 1
        self.current_status = "in_progress"
        self.last_pose = pose
        self.pose_goals.append(pose)
        self.pose_goal_options.append({key: value for key, value in options.items() if value is not None and (key in {"segment", "looking_hand_constraint"} or (key == "planning_scope" and value != "whole_body") or (key == "planning_motion" and value != "pose") or key not in {"planning_scope", "planning_motion", "collision_policy"} or (key == "collision_policy" and value != "full"))})

    def add_collision_mesh(self, mesh_file_name, name: str = "collision_mesh", *, frame_id: str | None = None) -> bool:
        self.collision_events.append(("add", name))
        return True

    def remove_collision_object(self, name: str) -> None:
        self.collision_events.append(("remove", name))

    def clear_collision_scene(self) -> None:
        self.collision_events.append(("clear", None))

    def attach_collision_object(self, name: str, link_name: str | None = None, touch_links: list[str] | None = None) -> None:
        self.collision_events.append(("attach", name))

    def detach_collision_object(self, name: str) -> None:
        self.collision_events.append(("detach", name))

    def allow_collision_pairs(self, pairs: list[tuple[str, str]]) -> bool:
        self.allowed_collision_pairs.extend((str(a), str(b)) for a, b in pairs)
        return True

    def send_joint_goal(self, joints: dict[str, float]) -> None:
        self.move_count += 1
        self.current_status = "in_progress"
        self.last_joint_goal = joints

    def send_base_pose_goal(self, pose, **options) -> None:
        self.move_count += 1
        self.current_status = "in_progress"
        self.last_base_pose_goal = pose
        self.base_pose_goal_options = dict(options)

    def send_trajectory_goal(self, trajectory) -> None:
        self.move_count += 1
        self.current_status = "in_progress"
        self.last_trajectory = trajectory

    def send_named_goal(self, config_name: str, *, force_execution: bool = False) -> None:
        _ = force_execution
        self.move_count += 1
        self.current_status = "in_progress"
        self.last_named_goal = config_name
        self.named_goal_history.append(config_name)
        try:
            super().send_named_goal(config_name)
        except Exception:
            pass

    def get_status(self) -> Literal["in_progress", "succeeded", "failed", "idle"]:
        if self.current_status == "in_progress" and self.move_count > 0:
            return "succeeded"
        if self.move_count == 0:
            return "idle"
        return self.current_status  # type: ignore[return-value]

    def get_error(self) -> str | None:
        return None

    def cancel_goal(self) -> None:
        self.current_status = "idle"


class MockGripperController(BaseGripperController):
    """Mock gripper controller for testing."""

    def __init__(self, hardware_config: dict | None = None):
        super().__init__(hardware_config or {})
        self.is_open = True
        self.current_status = "idle"
        self.last_open_width: float | None = None
        self.last_close_force: float | None = None

    def open(self, width_m: float | None = None) -> ControllerResult:
        self.is_open = True
        self.current_status = "succeeded"
        self.last_open_width = width_m
        return ControllerResult(success=True)

    def close(self, force_n: float | None = None) -> ControllerResult:
        self.is_open = False
        self.current_status = "succeeded"
        self.last_close_force = force_n
        return ControllerResult(success=True)

    def get_state(self) -> Literal["open", "closed", "moving", "stalled"]:
        return "open" if self.is_open else "closed"

    def get_status(self) -> Literal["in_progress", "succeeded", "failed", "idle"]:
        return self.current_status  # type: ignore[return-value]

    def get_error(self) -> str | None:
        return None

    def cancel_goal(self) -> None:
        self.current_status = "idle"


class FailingNavController(BaseNavController):
    """Navigation controller that fails (for error testing)."""

    def send_goal(self, target: Tuple[float, float, float], frame_id: str | None = None) -> None:
        pass
    
    def get_status(self) -> Literal["in_progress", "succeeded", "failed", "timeout", "idle"]:
        return "failed"
    
    def get_error(self) -> str | None:
        return "Navigation failed"
    
    def cancel_goal(self) -> None:
        pass
    
    def get_current_pose(self) -> Pose:
        return Pose(x=0.0, y=0.0, yaw=0.0)
