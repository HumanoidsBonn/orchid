"""Base controller definitions for the modular mobile manipulation framework.

These abstract base classes define the minimal, non‑blocking interface required by
actions. Concrete implementations (ROS2 Nav2/MoveIt, simulation stubs, or custom
hardware drivers) will inherit from these classes and provide the actual logic.
"""

from __future__ import annotations
from src.core.arm_request import ArmPoseRequest

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal, Tuple
from src.core.failure import FailureInfo

# ---------------------------------------------------------------------------
# Simple data structures
# ---------------------------------------------------------------------------

@dataclass
class Pose:
    """Simple 2-D pose used by navigation controllers."""

    x: float
    y: float
    yaw: float = 0.0


@dataclass
class Pose3D:
    """3-D pose used by arm manipulation controllers."""

    x: float
    y: float
    z: float
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    frame_id: str = "base_link"


@dataclass
class ControllerResult:
    """Standard result returned by synchronous controller calls.

    ``success`` indicates whether the operation completed without error.
    ``error`` holds an optional human‑readable message.
    ``payload`` can carry arbitrary additional data (e.g., the final pose).
    """

    success: bool
    error: str | None = None
    payload: dict = field(default_factory=dict)
    failure: FailureInfo | None = None

    def __post_init__(self):
        if not self.success and self.failure is None:
            self.failure = FailureInfo()



NavStatus = Literal["in_progress", "succeeded", "failed", "timeout", "idle"]
ArmStatus = Literal["in_progress", "succeeded", "failed", "idle"]
HeadStatus = Literal["in_progress", "succeeded", "failed", "idle"]


class BaseNavController(ABC):
    """Interface for navigation controllers.

    Implementations must be non‑blocking – ``send_goal`` should dispatch the goal
    and return immediately. ``get_status`` is polled by the async action loop.
    """

    def __init__(self, hardware_config: dict):
        """Initialize the controller with robot‑specific hardware configuration.

        ``hardware_config`` is a dictionary containing any necessary parameters
        for controller initialization (e.g., ROS2 topic names, action server
        endpoints, or simulation parameters). This allows the same controller
        class to be used across different robots with different hardware setups.
        """
        self.hardware_config = hardware_config


    @abstractmethod
    def send_goal(self, target: Tuple[float, float, float], frame_id: str | None = None) -> None:
        """Dispatch a navigation goal ``(x, y, yaw)``."""

    def update_goal(self, target: Tuple[float, float, float], frame_id: str | None = None) -> None:
        """Update the active goal without cancelling (Nav2-style goal update).

        Default implementation cancels and resends; Nav2 controllers override
        to hot-swap the goal on the action server.
        """
        self.cancel_goal()
        self.send_goal(target, frame_id=frame_id)

    @abstractmethod
    def get_status(self) -> NavStatus:
        """Return the current status of the active navigation goal."""

    @abstractmethod
    def get_error(self) -> str | None:
        """Diagnostic when status is ``failed`` or ``timeout``."""

    @abstractmethod
    def cancel_goal(self) -> None:
        """Cancel any active navigation goal."""

    @abstractmethod
    def get_current_pose(self) -> Pose:
        """Return the robot's estimated pose."""


class BaseArmController(ABC):
    """Interface for arm manipulation controllers.

    Mirrors typical MoveIt behaviour – a goal is sent, then the caller polls the
    status.
    """

    def __init__(self, hardware_config: dict):
        """Initialize the controller with robot‑specific hardware configuration.

        ``hardware_config`` is a dictionary containing any necessary parameters
        for controller initialization (e.g., ROS2 topic names, action server
        endpoints, or simulation parameters). This allows the same controller
        class to be used across different robots with different hardware setups.
        """
        self.hardware_config = hardware_config

    def get_failure(self) -> FailureInfo | None:
        return FailureInfo() if self.get_status() == "failed" else None

    @abstractmethod
    def send_pose_goal(self, request: ArmPoseRequest) -> None:
        """Command the arm to move to the specified pose."""

    @abstractmethod
    def add_collision_mesh(self, mesh_file_name, name: str = "collision_mesh", *, frame_id: str | None = None) -> bool:
        """Add mesh coordinates in frame_id; return whether backend setup succeeded."""

    def remove_collision_object(self, name: str) -> None:
        """Remove one named collision object when the backend supports it."""

    def clear_collision_scene(self) -> None:
        """Clear collision objects when the backend supports it."""

    def attach_collision_object(self, name: str, link_name: str | None = None, touch_links: list[str] | None = None) -> None:
        """Attach a named collision object to the robot when the backend supports it."""

    def detach_collision_object(self, name: str) -> None:
        """Detach a named collision object when the backend supports it."""

    def allow_collision_pairs(self, pairs: list[tuple[str, str]]) -> bool:
        """Allow specific collision pairs when the backend supports planning scene ACM edits."""
        _ = pairs
        return False

    @abstractmethod
    def send_joint_goal(self, joints: dict[str, float]) -> None:
        """Command the arm to move to the specified joint positions."""

    def send_trajectory_goal(self, trajectory: Any) -> None:
        """Command the arm to execute a precomputed trajectory."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support trajectory goals"
        )

    def send_synchronized_trajectory_goal(self, trajectory: Any, *, cancelled=None) -> None:
        """Dispatch one shared-timeline trajectory with strict batch acceptance."""
        raise NotImplementedError(f"{self.__class__.__name__} does not support synchronized trajectories")

    def send_base_pose_goal(self, pose: Any, **options: Any) -> None:
        """Command a whole-body/mobile-base pose through an arm-capable backend."""
        _ = pose, options
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support whole-body base pose goals"
        )

    def resolve_named_target(self, config_name: str) -> dict[str, Any] | None:
        """Resolve a named target from controller hardware configuration.

        Canonical manifest form:
            {"type": "joint_positions", "joints": {"joint": position}}

        Transitional forms are accepted for existing manifests/controllers:
            {"joint_names": [...], "joint_positions": [...]}
            {"x": ..., "y": ..., "z": ...}
        """
        named_targets = self.hardware_config.get("named_targets", {})
        target = named_targets.get(config_name)
        return target if isinstance(target, dict) else None

    def send_named_goal(self, config_name: str, *, force_execution: bool = False) -> None:
        """Command the arm to move to a manifest-defined named target."""
        _ = force_execution
        target = self.resolve_named_target(config_name)
        if target is None:
            raise ValueError(f"Unknown named target '{config_name}'")

        target_type = target.get("type")
        if target_type in (None, "joint_positions"):
            joints = target.get("joints")
            if isinstance(joints, dict):
                self.send_joint_goal({str(k): float(v) for k, v in joints.items()})
                return
            if "joint_names" in target and "joint_positions" in target:
                self.send_joint_goal(
                    {
                        str(name): float(position)
                        for name, position in zip(
                            target["joint_names"],
                            target["joint_positions"],
                            strict=True,
                        )
                    }
                )
                return

        if target_type in (None, "pose") and all(k in target for k in ("x", "y", "z")):
            self.send_pose_goal(ArmPoseRequest(Pose3D(x=float(target['x']), y=float(target['y']), z=float(target['z']), roll=float(target.get('roll', 0.0)), pitch=float(target.get('pitch', 0.0)), yaw=float(target.get('yaw', 0.0)), frame_id=str(target.get('frame_id', 'base_link')))))
            return

        if target_type == "trajectory":
            self.send_trajectory_goal(target.get("trajectory"))
            return

        if target_type == "native":
            raise NotImplementedError(
                f"{self.__class__.__name__} does not support native named target '{config_name}'"
            )

        raise ValueError(f"Invalid named target config for '{config_name}'")

    def update_pose_goal(self, pose: Pose3D) -> None:
        """Update the active arm goal when supported by the backend."""
        self.cancel_goal()
        self.send_pose_goal(ArmPoseRequest(pose))

    @abstractmethod
    def get_status(self) -> ArmStatus:
        """Poll the current execution status."""

    @abstractmethod
    def get_error(self) -> str | None:
        """Diagnostic message if the last action failed."""

    @abstractmethod
    def cancel_goal(self) -> None:
        """Cancel any active arm motion."""


class BaseHeadController(ABC):
    """Interface for pan/tilt head controllers.

    Implementations should dispatch non-blocking pan/tilt or named-state goals
    and expose progress through ``get_status``.
    """

    def __init__(self, hardware_config: dict):
        self.hardware_config = hardware_config

    def resolve_named_target(self, config_name: str) -> dict[str, Any] | None:
        named_targets = self.hardware_config.get("named_targets", {})
        target = named_targets.get(config_name)
        return target if isinstance(target, dict) else None

    def send_named_goal(self, config_name: str) -> None:
        target = self.resolve_named_target(config_name)
        if target is None:
            raise ValueError(f"Unknown head named target '{config_name}'")
        joints = target.get("joints") if isinstance(target, dict) else None
        if isinstance(joints, dict):
            self.send_joint_goal({str(k): float(v) for k, v in joints.items()})
            return
        pan = target.get("pan", target.get("yaw")) if isinstance(target, dict) else None
        tilt = target.get("tilt", target.get("pitch")) if isinstance(target, dict) else None
        if pan is not None or tilt is not None:
            self.send_pan_tilt(float(pan or 0.0), float(tilt or 0.0))
            return
        raise ValueError(f"Invalid head named target config for '{config_name}'")

    @abstractmethod
    def send_pan_tilt(self, pan: float, tilt: float) -> None:
        """Command the head pan/tilt joints."""

    @abstractmethod
    def send_joint_goal(self, joints: dict[str, float]) -> None:
        """Command named head joints directly."""

    @abstractmethod
    def get_status(self) -> HeadStatus:
        """Poll the current head goal status."""

    @abstractmethod
    def get_error(self) -> str | None:
        """Diagnostic message if the last head action failed."""

    @abstractmethod
    def cancel_goal(self) -> None:
        """Cancel any active head motion."""


class BaseGripperController(ABC):
    """Interface for simple gripper hardware.

    Gripper actions are typically fast enough to be synchronous, but the
    interface is kept consistent with the other controllers.
    """

    def __init__(self, hardware_config: dict):
        """Initialize the controller with robot‑specific hardware configuration.

        ``hardware_config`` is a dictionary containing any necessary parameters
        for controller initialization (e.g., ROS2 topic names, action server
        endpoints, or simulation parameters). This allows the same controller
        class to be used across different robots with different hardware setups.
        """
        self.hardware_config = hardware_config

    @abstractmethod
    def open(self, width_m: float | None = None) -> ControllerResult:
        """Open the gripper. ``width_m`` optionally specifies the target width.
        """

    @abstractmethod
    def close(self, force_n: float | None = None, width_m: float | None = None) -> ControllerResult:
        """Close the gripper.

        ``force_n`` can be used by force-control backends. ``width_m`` can be
        used by position-control backends that close to a requested opening.
        """

    @abstractmethod
    def get_state(self) -> Literal["open", "closed", "moving", "stalled"]:
        """Return the current gripper state."""

    @abstractmethod
    def get_status(self) -> ArmStatus:
        """Poll async gripper motion status (MoveIt-backed grippers)."""

    @abstractmethod
    def get_error(self) -> str | None:
        """Diagnostic message if the last gripper command failed."""

    @abstractmethod
    def cancel_goal(self) -> None:
        """Cancel any active gripper motion."""

    def get_distance(self) -> float | None:
        """Current gripper opening distance in meters, if the hardware exposes it.

        Not abstract: most controllers don't need this, only grasp-verification
        callers that want to confirm something is actually between the fingers.
        """
        return None
