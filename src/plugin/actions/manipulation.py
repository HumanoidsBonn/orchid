"""Manipulation plugin actions."""

from __future__ import annotations

import time
from dataclasses import replace
from typing import Any

import py_trees
from src.tree.bt_node_wrapper import ActionScope
from src.core.failure import FailureCode, FailureInfo, FailureOrigin

import numpy as np
import open3d as o3d

from src.core.arm_request import ArmPoseRequest, CartesianLine, MoveItPoseSettings
from src.core.manipulation_types import BasePose2D
from src.core.action import AbstractAction, ActionResult
from src.core.context import Context
from src.core.controllers import Pose3D
from src.core.collision_geometry import validate_hull_inflation
from src.core.geometry import rpy_to_rotation_matrix
from src.core.sensor_data_center import SensorDataCenter
from src.plugin.actions._helpers import (
    ControllerAsyncAction,
    _arm_status_result,
    _gripper_status_result,
    _pose_from_params,
)
import src.core.logger as log

def _create_mesh_for_collision_world(context: Context, pcd=None, *, convex_hull: bool = False, inflation_factor: float = 1.0) -> str | None:
    """Update collision world with sensor data (point clouds, depth images).

    This function creates a mesh from provided pcd or retrieves available sensor data from the robot

    Args:
        context: Execution context with access to sensors and collision world
        pcd: point cloud data to convert to mesh
    """
    if not context.sensor_data_center or not context.world_state: return None
    if convex_hull and pcd is None:
        raise ValueError("Missing target point cloud")
    if pcd is None:
        try:
            pointcloud_msg = context.sensor_data_center.get_latest("head_pointcloud")
            if pointcloud_msg and pointcloud_msg.data:
                pcd = _extract_pcd_data_and_transform(context, pointcloud_msg)
                log.warn('Point cloud extracted for collision world at action time')
        except Exception as e:
            log.warn(f"Failed to add point cloud to collision world: {e}")
            return None
    if pcd is not None:
        if convex_hull:
            return context.sensor_data_center.pcd_to_mesh(pcd, convex_hull=True, inflation_factor=inflation_factor)
        return context.sensor_data_center.pcd_to_mesh(pcd)

def _extract_pcd_data_and_transform(context: Context, pcl_msg: Any) -> None:
    try:
        points = []
        for i in range(0, len(pcl_msg.data.data), 32):  # 4 floats per point
            data = context.sensor_data_center.decode_pcd(pcl_msg.data, offset=i)
            if data is None: continue
            x, y, z, color = data
            points.append([x, y, z])
        if len(points) < 4:
            log.warn("Point cloud too small or invalid for collision mesh")
            return None
        return context.sensor_data_center.transform_data(np.array(points, dtype=np.float32),
                                                        pcl_msg.frame_id, context.hardware.get("odom", 'odom'))
    except Exception as e:
        log.exception(f"Error processing point cloud for collision: {e}")
    return None


def _offset_pose(pose: Pose3D, distance: float, mode: str) -> Pose3D:
    if mode == "world_z":
        dx, dy, dz = 0.0, 0.0, float(distance)
    elif mode == "grasp_minus_x":
        offset = rpy_to_rotation_matrix(pose.roll, pose.pitch, pose.yaw).astype(np.float64) @ np.array(
            [-float(distance), 0.0, 0.0],
            dtype=np.float64,
        )
        dx, dy, dz = offset.tolist()
    else:
        raise ValueError(
            f"Unknown retreat_mode '{mode}'. Expected 'world_z' or 'grasp_minus_x'."
        )

    return Pose3D(
        x=pose.x + dx,
        y=pose.y + dy,
        z=pose.z + dz,
        roll=pose.roll,
        pitch=pose.pitch,
        yaw=pose.yaw,
        frame_id=pose.frame_id,
    )


class _MoveArmToPose(ControllerAsyncAction):
    """Internal action that dispatches an arm pose goal and polls arm status.

    The typed request includes the pose, scope, primitive, hand-view policy,
    and explicit backend overrides. Measured start state belongs to the controller.
    """

    required_controllers = ["arm"]


    def start(self, context: Context) -> None:
        self._dispatch_status = "dispatching"

        def send_goal() -> dict[str, Any]:
            try:
                if self._cancel_requested or context.execution.control is not None or context.execution.cancel_requested:
                    return {"success": False, "error": "dispatch_cancelled"}
                arm = context.require_arm()
                arm.send_pose_goal(self.request)
                if arm.get_status() == "failed":
                    return {
                        "success": False,
                        "error": arm.get_error() or "arm pose goal failed",
                    }
            except Exception as exc:  # noqa: BLE001
                return {"success": False, "error": str(exc)}
            return {"success": True}

        def on_dispatched(result: dict[str, Any]) -> None:
            if result.get("success"):
                self._dispatch_status = "dispatched"
            else:
                self._dispatch_status = "failure"
                self._failure_reason = str(result.get("error") or "arm pose dispatch failed")

        try:
            context.run_in_context_loop(send_goal, callback=on_dispatched)
        except Exception as exc:  # noqa: BLE001
            self._dispatch_status = "failure"
            self._failure_reason = str(exc)

    def check_result(self, context: Context) -> ActionResult:
        if self._cancel_requested:
            return ActionResult(status="failure", reason="Action Cancelled")
        if self._dispatch_status == "failure":
            return ActionResult(status="failure", reason=self._failure_reason, failure=context.require_arm().get_failure() or FailureInfo())
        if self._dispatch_status != "dispatched":
            return ActionResult(status="running")
        return _arm_status_result(context.require_arm())

    def reset(self) -> None:
        super().reset()
        self._dispatch_status = "idle"
        self._failure_reason = None

    def cancel(self, context: Context) -> None:
        self._cancel_requested = True
        context.require_arm().cancel_goal()


    def __init__(self, request: ArmPoseRequest):
        super().__init__()
        self.request = request
        self._dispatch_status = "idle"
        self._failure_reason = None


class _AddCollisionMeshAction(ControllerAsyncAction):
    """Add target geometry strictly; surrounding-cloud setup remains best effort."""

    required_controllers = ["arm"]

    def __init__(self, pcd: Any = None, name: str = "collision_mesh", *,
                 convex_hull: bool = False, inflation_factor: float = 1.0,
                 frame_id: str | None = None, required: bool = False) -> None:
        self.pcd, self.name = pcd, name
        self.convex_hull = convex_hull
        self.inflation_factor = validate_hull_inflation(inflation_factor)
        self.frame_id = frame_id
        self.required = required
        super().__init__()
        self._generation = 0
        self._result = ActionResult(status="running")

    def start(self, context: Context) -> None:
        self._generation += 1
        generation = self._generation
        self._result = ActionResult(status="running")
        pcd = None if self.pcd is None else np.array(self.pcd, copy=True)
        name, frame = self.name, self.frame_id
        convex, factor = self.convex_hull, self.inflation_factor

        def current():
            return generation == self._generation and not self._cancel_requested and context.execution.control is None and not context.execution.cancel_requested

        def failed(exc):
            log.warn("Collision mesh setup failed for %s: %s", name, exc)
            return ActionResult(status="failure", reason=f"collision_mesh_setup_failed:{name}:{exc}",
                                failure=FailureInfo(FailureCode.INVALID_STATE if isinstance(exc, ValueError) else FailureCode.UNKNOWN)) if convex or self.required else ActionResult(status="success")

        def add_mesh():
            try:
                path = _create_mesh_for_collision_world(context, pcd, convex_hull=convex, inflation_factor=factor)
                if not current():
                    return ActionResult(status="failure", reason="Action Cancelled", failure=FailureInfo(FailureCode.CANCELLED))
                if not path:
                    raise ValueError("No collision geometry generated")
                arm = context.require_arm()
                applied = arm.add_collision_mesh(path, name=name, frame_id=frame) if convex or self.required else arm.add_collision_mesh(path, name=name)
                if (convex or self.required) and applied is not True:
                    raise RuntimeError("Target addition or scene readback failed")
                return ActionResult(status="success")
            except Exception as exc:
                return failed(exc)

        def complete(result):
            if current():
                self._result = result

        try:
            context.run_in_context_loop(add_mesh, callback=complete, error_callback=lambda exc: complete(failed(exc)))
        except Exception as exc:
            complete(failed(exc))

    def check_result(self, context: Context) -> ActionResult:
        return ActionResult(status="failure", reason="Action Cancelled", failure=FailureInfo(FailureCode.CANCELLED)) if self._cancel_requested else self._result

    def reset(self) -> None:
        super().reset()
        self._generation += 1
        self._result = ActionResult(status="running")

    def cancel(self, context: Context) -> None:
        self._cancel_requested = True
        self._generation += 1


class _RemoveCollisionObjectAction(ControllerAsyncAction):
    """Internal best-effort collision object removal action."""

    required_controllers = ["arm"]

    def __init__(self, name: str) -> None:
        self.name = name
        super().__init__()
        self._done = False

    def start(self, context: Context) -> None:
        def remove_object() -> None:
            try:
                context.require_arm().remove_collision_object(self.name)
            except Exception as exc:  # noqa: BLE001
                log.warn("Skipping collision object removal due to %s", exc)

        try:
            context.run_in_context_loop(
                remove_object,
                callback=lambda _result: setattr(self, "_done", True),
                error_callback=lambda exc: setattr(self, "_done", True),
            )
        except Exception as exc:  # noqa: BLE001
            log.warn("Skipping collision object removal due to %s", exc)
            self._done = True

    def check_result(self, context: Context) -> ActionResult:
        if self._cancel_requested:
            return ActionResult(status="failure", reason="Action Cancelled")
        return ActionResult(status="success" if self._done else "running")

    def reset(self) -> None:
        super().reset()
        self._done = False

    def cancel(self, context: Context) -> None:
        self._cancel_requested = True


class _AttachCollisionObjectAction(ControllerAsyncAction):
    """Internal best-effort collision object attach action."""

    required_controllers = ["arm"]

    def __init__(self, name: str, link_name: str | None=None, touch_links: list[str] | None=None) -> None:
        self.name = name
        self.link_name = link_name
        self.touch_links = touch_links
        super().__init__()
        self._done = False
        self._failure_reason = None

    def start(self, context: Context) -> None:
        def attach_object() -> None:
            try:
                attached = context.require_arm().attach_collision_object(
                    self.name,
                    link_name=self.link_name,
                    touch_links=self.touch_links,
                )
                if attached is False:
                    self._failure_reason = "collision_object_attachment_failed"
                    log.warn("Collision object attachment rejected: %s", self.name)
            except Exception as exc:  # noqa: BLE001
                log.warn("Skipping collision object attach due to %s", exc)
                self._failure_reason = str(exc)

        try:
            context.run_in_context_loop(
                attach_object,
                callback=lambda _result: setattr(self, "_done", True),
                error_callback=lambda exc: self._attachment_failed(exc),
            )
        except Exception as exc:  # noqa: BLE001
            log.warn("Skipping collision object attach due to %s", exc)
            self._failure_reason = str(exc)
            self._done = True

    def _attachment_failed(self, exc):
        self._failure_reason = str(exc)
        self._done = True

    def check_result(self, context: Context) -> ActionResult:
        if self._cancel_requested:
            return ActionResult(status="failure", reason="Action Cancelled")
        if self._failure_reason:
            return ActionResult(status="failure", reason=self._failure_reason)
        return ActionResult(status="success" if self._done else "running")

    def reset(self) -> None:
        super().reset()
        self._done = False
        self._failure_reason = None

    def cancel(self, context: Context) -> None:
        self._cancel_requested = True


class _AllowCollisionPairsAction(ControllerAsyncAction):
    """Internal best-effort ACM pair relaxation action."""

    required_controllers = ["arm"]

    def __init__(self, pairs: list[tuple[str, str]], required: bool = False) -> None:
        self.pairs = pairs
        self.required = required
        super().__init__()
        self.pairs = [(str(a), str(b)) for a, b in pairs]
        self._done = False
        self._failure_reason = None

    def start(self, context: Context) -> None:
        self._done = False
        self._failure_reason = None
        def allow_pairs() -> None:
            try:
                if self._cancel_requested or context.execution.control is not None:
                    return
                applied = context.require_arm().allow_collision_pairs(self.pairs)
                if self.required and applied is not True:
                    raise RuntimeError("collision policy update was not acknowledged")
            except Exception as exc:  # noqa: BLE001
                self._failure_reason = str(exc) if self.required else None
                log.warn("Collision-pair ACM update failed: %s pairs=%s", exc, self.pairs)

        try:
            context.run_in_context_loop(
                allow_pairs,
                callback=lambda _result: setattr(self, "_done", True),
                error_callback=lambda exc: setattr(self, "_done", True),
            )
        except Exception as exc:  # noqa: BLE001
            log.warn("Skipping collision-pair ACM update due to %s", exc)
            self._done = True

    def check_result(self, context: Context) -> ActionResult:
        if self._cancel_requested:
            return ActionResult(status="failure", reason="Action Cancelled")
        if self._failure_reason:
            return ActionResult(status="failure", reason=self._failure_reason, failure=FailureInfo(FailureCode.INVALID_STATE))
        return ActionResult(status="success" if self._done else "running")

    def reset(self) -> None:
        super().reset()
        self._done = False
        self._failure_reason = None

    def cancel(self, context: Context) -> None:
        self._cancel_requested = True


class _AllowCollisionWithConfiguredLinksAction(ControllerAsyncAction):
    """Internal best-effort ACM relaxation against configured hand links."""

    required_controllers = ["arm"]

    def __init__(self, object_names: list[str]) -> None:
        self.object_names = object_names
        cleaned = [str(name) for name in object_names if str(name)]
        self.object_names = cleaned
        super().__init__()
        self.pairs: list[tuple[str, str]] = []
        self._done = False

    def start(self, context: Context) -> None:
        def allow_configured_links() -> None:
            try:
                arm = context.require_arm()
                cfg = getattr(arm, "hardware_config", {}) or {}
                links = cfg.get("collision_relaxation_links") or []
                log.warn("Allowing collision with configured links: %s", links, 'object_names:', self.object_names)
                if not isinstance(links, (list, tuple)):
                    log.warn("Skipping configured-link ACM update; collision_relaxation_links is not a list")
                    return
                clean_links = [str(link) for link in links if str(link)]
                self.pairs = [
                    (link, object_name)
                    for link in clean_links
                    for object_name in self.object_names
                ]
                if not self.pairs:
                    log.warn("Skipping configured-link ACM update; no links or objects configured")
                    return
                arm.allow_collision_pairs(self.pairs)
            except Exception as exc:  # noqa: BLE001
                log.warn("Skipping configured-link ACM update due to %s", exc)

        try:
            context.run_in_context_loop(
                allow_configured_links,
                callback=lambda _result: setattr(self, "_done", True),
                error_callback=lambda exc: setattr(self, "_done", True),
            )
        except Exception as exc:  # noqa: BLE001
            log.warn("Skipping configured-link ACM update due to %s", exc)
            self._done = True

    def check_result(self, context: Context) -> ActionResult:
        if self._cancel_requested:
            return ActionResult(status="failure", reason="Action Cancelled")
        return ActionResult(status="success" if self._done else "running")

    def reset(self) -> None:
        super().reset()
        self._done = False
        self.pairs = []

    def cancel(self, context: Context) -> None:
        self._cancel_requested = True


class _ClearOctomapAction(ControllerAsyncAction):
    """Internal best-effort MoveIt octomap cleanup action."""

    required_controllers = ["arm"]

    def __init__(self, service_name: str | None=None, timeout_s: float=2.0) -> None:
        self.service_name = service_name
        self.timeout_s = timeout_s
        super().__init__()
        self._done = False

    def start(self, context: Context) -> None:
        def clear_octomap() -> None:
            try:
                modules = getattr(context, "modules", None)
                if modules is None:
                    log.warn("Skipping octomap clear; module proxy unavailable")
                    return
                arm = context.require_arm()
                cfg = getattr(arm, "hardware_config", {}) or {}
                module_name = str(cfg.get("clear_octomap_module") or "moveit_clear_octomap")
                service_name = str(
                    self.service_name
                    or cfg.get("clear_octomap_service")
                    or "/clear_octomap"
                )
                if hasattr(modules, "has") and not modules.has(module_name):
                    log.warn("Skipping octomap clear; module unavailable: %s", module_name)
                    return
                result = modules.call(module_name, {}, timeout=float(self.timeout_s))
                if isinstance(result, dict) and result.get("error"):
                    log.warn("Skipping octomap clear via %s (%s): %s", module_name, service_name, result.get("error"))
                    return
                log.info("Octomap clear requested via module %s service=%s", module_name, service_name)
            except Exception as exc:  # noqa: BLE001
                log.warn("Skipping octomap clear due to %s", exc)

        try:
            context.run_in_context_loop(
                clear_octomap,
                callback=lambda _result: setattr(self, "_done", True),
                error_callback=lambda exc: setattr(self, "_done", True),
            )
        except Exception as exc:  # noqa: BLE001
            log.warn("Skipping octomap clear due to %s", exc)
            self._done = True

    def check_result(self, context: Context) -> ActionResult:
        if self._cancel_requested:
            return ActionResult(status="failure", reason="Action Cancelled")
        return ActionResult(status="success" if self._done else "running")

    def reset(self) -> None:
        super().reset()
        self._done = False

    def cancel(self, context: Context) -> None:
        self._cancel_requested = True


class _MoveJointsRelativeAction(ControllerAsyncAction):
    """Internal action that moves joints by deltas from the latest joint state."""

    required_controllers = ["arm"]

    _DEFAULT_LIMITS = {
        "arm_lift_joint": (0.0, 0.69),
        "head_pan_joint": (-1.75, 1.75),
        "head_tilt_joint": (-1.57, 0.52),
    }

    def __init__(self, joint_deltas: dict[str, float]) -> None:
        self.joint_deltas = joint_deltas
        super().__init__()
        self.joint_deltas = {str(name): float(delta) for name, delta in (joint_deltas or {}).items()}
        self._dispatch_status = 'idle'
        self._failure_reason: str | None = None
        self._move_started = False
        self.goal: dict[str, float] = {}
        self._generation = 0

    def start(self, context: Context) -> None:
        self._generation += 1
        generation = self._generation
        self._cancel_requested = False
        self._move_started = False
        self._failure_reason = None
        self._dispatch_status = "dispatching"
        def invocation_current():
            return generation == self._generation and not self._cancel_requested and context.execution.control is None and not context.execution.cancel_requested
        if not self.joint_deltas:
            self._dispatch_status = "succeeded"
            return

        def send_goal() -> dict[str, Any]:
            try:
                current = _latest_joint_positions(context)
                missing = [name for name in self.joint_deltas if name not in current]
                if missing:
                    return {"success": False, "error": f"missing_joint_state:{','.join(missing)}"}
                limits = _joint_limits_from_context(context)
                goal = {
                    name: _clamp_joint_value(name, current[name] + delta, limits)
                    for name, delta in self.joint_deltas.items()
                }
                if any(self.joint_deltas[n] > 0 and goal[n] < current[n] - 1e-6 for n in goal):
                    raise ValueError("positive relative joint move would reverse direction at configured limit")
                if not invocation_current():
                    return {"success": False, "error": "cancelled"}
                log.info("Relative joint move measured=%s deltas=%s goal=%s", {n:current[n] for n in goal}, self.joint_deltas, goal)
                context.require_arm().send_joint_goal(goal)
                self.goal = goal
                if context.require_arm().get_status() == "failed":
                    return {"success": False, "error": context.require_arm().get_error() or "relative_joint_goal_failed"}
                return {"success": True}
            except Exception as exc:  # noqa: BLE001
                return {"success": False, "error": str(exc)}

        def on_dispatched(result: dict[str, Any]) -> None:
            if not invocation_current():
                return
            if result.get("success"):
                self._move_started = True
                self._dispatch_status = "dispatched"
            else:
                self._dispatch_status = "failure"
                self._failure_reason = str(result.get("error") or "relative_joint_goal_failed")

        try:
            context.run_in_context_loop(send_goal, callback=on_dispatched, error_callback=lambda exc: on_dispatched({"success": False, "error": str(exc)}))
        except Exception as exc:  # noqa: BLE001
            self._dispatch_status = "failure"
            self._failure_reason = str(exc)

    def check_result(self, context: Context) -> ActionResult:
        if self._cancel_requested:
            return ActionResult(status="failure", reason="Action Cancelled")
        if self._dispatch_status == "succeeded":
            return ActionResult(status="success", reason="relative_joint_goal_empty")
        if self._dispatch_status == "failure":
            return ActionResult(status="failure", reason=self._failure_reason or "relative_joint_goal_failed")
        if self._dispatch_status != "dispatched" or not self._move_started:
            return ActionResult(status="running")
        result = _arm_status_result(context.require_arm())
        if result.status == "success":
            return ActionResult(status="success", reason="relative_joint_goal_completed", payload={"goal": self.goal})
        return result

    def reset(self) -> None:
        super().reset()
        self._dispatch_status = "idle"
        self._failure_reason = None
        self._move_started = False
        self.goal = {}

    def cancel(self, context: Context) -> None:
        self._cancel_requested = True
        self._generation += 1
        if context.arm and self._move_started:
            context.arm.cancel_goal()


def _latest_joint_positions(context: Context) -> dict[str, float]:
    msg = context.sensor_data_center.get_latest("joint_states") if context.sensor_data_center else None
    data = msg.data if msg is not None else None
    names = list(getattr(data, "name", []) or [])
    positions = list(getattr(data, "position", []) or [])
    return {
        str(name): float(positions[index])
        for index, name in enumerate(names)
        if index < len(positions)
    }


def _joint_limits_from_context(context: Context) -> dict[str, tuple[float, float]]:
    limits: dict[str, tuple[float, float]] = {}
    for block in (getattr(context, "hardware", None) or {}).values():
        if not isinstance(block, dict):
            continue
        raw = block.get("joint_limits")
        if not isinstance(raw, dict):
            continue
        for name, value in raw.items():
            if isinstance(value, dict) and "min" in value and "max" in value:
                limits[str(name)] = (float(value["min"]), float(value["max"]))
            elif isinstance(value, (list, tuple)) and len(value) >= 2:
                limits[str(name)] = (float(value[0]), float(value[1]))
    limits.update({name: value for name, value in _MoveJointsRelativeAction._DEFAULT_LIMITS.items() if name not in limits})
    return limits


def _clamp_joint_value(name: str, value: float, limits: dict[str, tuple[float, float]]) -> float:
    if name not in limits:
        return float(value)
    lo, hi = limits[name]
    return float(min(max(float(value), lo), hi))


class _ClearCollisionSceneAction(ControllerAsyncAction):
    """Internal best-effort collision scene clearing action."""

    required_controllers = ["arm"]

    def __init__(self) -> None:
        super().__init__()
        self._done = False

    def start(self, context: Context) -> None:
        def clear_scene() -> None:
            try:
                context.require_arm().clear_collision_scene()
            except Exception as exc:  # noqa: BLE001
                log.warn("Skipping collision scene clear due to %s", exc)

        try:
            context.run_in_context_loop(
                clear_scene,
                callback=lambda _result: setattr(self, "_done", True),
                error_callback=lambda exc: setattr(self, "_done", True),
            )
        except Exception as exc:  # noqa: BLE001
            log.warn("Skipping collision scene clear due to %s", exc)
            self._done = True

    def check_result(self, context: Context) -> ActionResult:
        if self._cancel_requested:
            return ActionResult(status="failure", reason="Action Cancelled")
        return ActionResult(status="success" if self._done else "running")

    def reset(self) -> None:
        super().reset()
        self._done = False

    def cancel(self, context: Context) -> None:
        self._cancel_requested = True


class GraspObjectAction(ControllerAsyncAction):
    """Close the gripper to hold an object by a certain force.
    Params:
        force_N: float - N of force to apply to grasp the object.
                        The range force control works well is 0.25 [N] < effort < 0.35 [N]
    """

    required_controllers = ["gripper"]

    def __init__(
        self,
        force_N: float | None = None,
        width_m: float | None = None,
        width: float | None = None,
        min_grasp_gap_m: float = 0.01,
    ) -> None:
        self.force_N = force_N
        self.width_m = width_m
        self.width = width
        self.min_grasp_gap_m = min_grasp_gap_m
        target_width = width_m if width_m is not None else width
        self.width_m = target_width
        super().__init__()
        self._close_status = None

    def start(self, context: Context) -> None:
        self._generation = getattr(self, "_generation", 0) + 1
        generation = self._generation
        self._cancel_requested = False
        self._dispatched = False
        self._close_status = None
        self._failure = None
        def current():
            return generation == self._generation and not self._cancel_requested and not context.execution.cancel_requested and context.execution.control is None
        def work():
            if not current():
                return None
            gripper = context.require_gripper()
            self._dispatched = True
            if self.width_m is not None:
                try:
                    return gripper.close(self.force_N, width_m=float(self.width_m))
                except TypeError:
                    return gripper.close(self.force_N)
            return gripper.close(self.force_N)
        def complete(result):
            if not current() or result is None:
                return
            self._close_status = "success" if result.success else (result.error or "gripper close failed")
            if not result.success:
                failure = getattr(result, "failure", None)
                self._failure = replace(failure, origin=FailureOrigin.CONTROLLER) if isinstance(failure, FailureInfo) else FailureInfo(FailureCode.CONTROLLER_FAILED, retryable=True, origin=FailureOrigin.CONTROLLER)
                log.warn("Gripper close failed: %s; failure=%s", self._close_status, self._failure)
        def failed(exc):
            if current():
                invalid = isinstance(exc, (ValueError, TypeError))
                self._close_status = str(exc)
                self._failure = FailureInfo(FailureCode.INVALID_REQUEST if invalid else FailureCode.CONTROLLER_FAILED,
                    retryable=not invalid, origin=FailureOrigin.INPUT if invalid else FailureOrigin.CONTROLLER)
                log.warn("Gripper close dispatch failed: %s", exc)
        context.run_in_context_loop(work, callback=complete, error_callback=failed)

    def check_result(self, context: Context) -> ActionResult:
        if self._cancel_requested:
            return ActionResult(status="failure", reason="Action Cancelled", failure=FailureInfo(FailureCode.CANCELLED))
        if self._close_status is None:
            return ActionResult(status="running")
        if self._close_status != "success":
            return ActionResult(status="failure", reason=self._close_status, failure=self._failure)
        result = _gripper_status_result(context.require_gripper())
        if result.status == "success":
            distance = context.require_gripper().get_distance()
            log.info("GraspObjectAction: gripper closed distance=%s", distance)
        return result

    def reset(self) -> None:
        super().reset()
        self._generation = getattr(self, "_generation", 0) + 1
        self._close_status = None

    def cancel(self, context):
        if self._cancel_requested:
            return
        self._cancel_requested = True
        self._generation = getattr(self, "_generation", 0) + 1
        if getattr(self, "_dispatched", False):
            context.require_gripper().cancel_goal()


class ReleaseObjectAction(ControllerAsyncAction):
    """Open the gripper to release a held object by a certain distance.
    Params:
        distance: float - width in meters for gap between the gripper fingers
    """

    required_controllers = ["gripper"]

    def __init__(self, distance: float | None=None) -> None:
        self.distance = distance
        super().__init__()
        self._open_status = None

    def start(self, context: Context) -> None:
        self._generation = getattr(self, "_generation", 0) + 1
        generation = self._generation
        self._cancel_requested = False
        self._dispatched = False
        self._open_status = None
        self._failure = None
        def current():
            return generation == self._generation and not self._cancel_requested and not context.execution.cancel_requested and context.execution.control is None
        def work():
            if not current():
                return None
            gripper = context.require_gripper()
            self._dispatched = True
            return gripper.open(self.distance)
        def complete(result):
            if not current() or result is None:
                return
            self._open_status = "success" if result.success else (result.error or "gripper open failed")
            if not result.success:
                failure = getattr(result, "failure", None)
                self._failure = replace(failure, origin=FailureOrigin.CONTROLLER) if isinstance(failure, FailureInfo) else FailureInfo(FailureCode.CONTROLLER_FAILED, retryable=True, origin=FailureOrigin.CONTROLLER)
                log.warn("Gripper open failed: %s; failure=%s", self._open_status, self._failure)
        def failed(exc):
            if current():
                invalid = isinstance(exc, (ValueError, TypeError))
                self._open_status = str(exc)
                self._failure = FailureInfo(FailureCode.INVALID_REQUEST if invalid else FailureCode.CONTROLLER_FAILED,
                    retryable=not invalid, origin=FailureOrigin.INPUT if invalid else FailureOrigin.CONTROLLER)
                log.warn("Gripper open dispatch failed: %s", exc)
        context.run_in_context_loop(work, callback=complete, error_callback=failed)

    def check_result(self, context: Context) -> ActionResult:
        if self._cancel_requested:
            return ActionResult(status="failure", reason="Action Cancelled", failure=FailureInfo(FailureCode.CANCELLED))
        if self._open_status is None:
            return ActionResult(status="running")
        if self._open_status != "success":
            return ActionResult(status="failure", reason=self._open_status, failure=self._failure)
        result = _gripper_status_result(context.require_gripper())
        return result

    def reset(self) -> None:
        super().reset()
        self._generation = getattr(self, "_generation", 0) + 1
        self._open_status = None

    def cancel(self, context):
        if self._cancel_requested:
            return
        self._cancel_requested = True
        self._generation = getattr(self, "_generation", 0) + 1
        if getattr(self, "_dispatched", False):
            context.require_gripper().cancel_goal()


class MoveArmToState(ControllerAsyncAction):
    """Move the arm to a manifest-defined named state.

    Use this for controller-agnostic moves to known, reusable arm positions such
    as navigation-safe, observation, neutral, or retracted configurations. The
    target state is resolved by the arm controller from
    ``hardware.arm.named_targets`` in the robot manifest, so different robots or
    controller backends can provide their own joint positions for the same
    action interface.

    Params:
        state: str - Name of the target state from ``hardware.arm.named_targets``.
                     Common values include ``go``, ``neutral``, ``observe``, or
                     ``retract`` when defined by the manifest.
    """

    required_controllers = ["arm"]

    def __init__(
        self,
        state: str = 'neutral',
        use_live_collision: bool = False,
        clear_collision_after: bool = True,
        force_execution: bool = False,
    ) -> None:
        self.state = state
        self.use_live_collision = use_live_collision
        self.clear_collision_after = clear_collision_after
        self.force_execution = force_execution
        super().__init__()
        self._move_started = False
        self._dispatch_status = 'idle'
        self._failure_reason: str | None = None

    def start(self, context: Context) -> None:
        if not self.state:
            self._dispatch_status = "failure"
            self._failure_reason = "state is required"
            return

        self._dispatch_status = "dispatching"

        def send_goal() -> dict[str, Any]:
            try:
                arm = context.require_arm()
                arm.send_named_goal(self.state, force_execution=bool(self.force_execution))
                if arm.get_status() == "failed":
                    return {
                        "success": False,
                        "error": arm.get_error() or f"Failed to move arm to state '{self.state}'",
                    }
            except Exception as exc:  # noqa: BLE001
                return {"success": False, "error": str(exc)}
            return {"success": True}

        def on_dispatched(result: dict[str, Any]) -> None:
            if result.get("success"):
                self._move_started = True
                self._dispatch_status = "dispatched"
                log.info("Initiated arm movement to state '%s'", self.state)
            else:
                self._dispatch_status = "failure"
                self._failure_reason = str(result.get("error") or f"Failed to move arm to state '{self.state}'")

        def on_error(exc: Exception) -> None:
            self._dispatch_status = "failure"
            self._failure_reason = str(exc)

        try:
            context.run_in_context_loop(send_goal, callback=on_dispatched, error_callback=on_error)
        except Exception as exc:  # noqa: BLE001
            log.error("Failed to initiate arm movement to state '%s': %s", self.state, exc)
            self._dispatch_status = "failure"
            self._failure_reason = str(exc)

    def _state_dispatch_action(self) -> ControllerAsyncAction:
        action = MoveArmToState(state=self.state)
        return action

    def check_result(self, context: Context) -> ActionResult:
        if self._cancel_requested:
            return ActionResult(status="failure", reason="Action Cancelled")
        if self._dispatch_status == "failure":
            return ActionResult(
                status="failure",
                reason=self._failure_reason or f"Failed to move arm to state '{self.state}'",
                failure=replace(context.require_arm().get_failure() or FailureInfo(FailureCode.CONTROLLER_FAILED), origin=FailureOrigin.CONTROLLER),
            )
        if self._dispatch_status == "succeeded":
            return ActionResult(status="success", reason=f"Arm already at state '{self.state}'")
        if self._dispatch_status != "dispatched" or not self._move_started:
            return ActionResult(status="running")
        return _arm_status_result(context.require_arm())

    def reset(self) -> None:
        super().reset()
        self._move_started = False
        self._dispatch_status = "idle"
        self._failure_reason = None

    def cancel(self, context: Context) -> None:
        self._cancel_requested = True
        if context.arm:
            context.arm.cancel_goal()

    def build_node(self, context, name):
        if not self.use_live_collision:
            return super().build_node(context, name)
        children = [_AddCollisionMeshAction(name="live_scene_collision").build_node(context, "Model live scene"),
                    self._state_dispatch_action().build_node(context, "Move named state")]
        if self.clear_collision_after:
            children.append(_ClearCollisionSceneAction().build_node(context, "Clear live scene"))
        return py_trees.composites.Sequence(name=name, memory=True, children=children)

class PickTarget(AbstractAction):
    """Pick an object using the arm given its name and corresponding attributes in world state.

    This action retrieves estimated grasp poses from the world state (typically populated
    by EstimatePickPose or similar perception actions) and uses PickAtPose to execute
    the grasp. This is the preferred method for picking detected objects.

    Params:
        target: str - name of the target object to be picked
                (should have *_estimated_pick_pose_mat and *_estimated_width in world state)
        looking_hand_constraint: bool - if true, request HSRB hand-in-view
                planning for the generated PickAtPose action
    """

    required_controllers = ["arm", "gripper"]

    def __init__(
        self,
        target: str = '',
        *,
        looking_hand_constraint: bool = False,
        recovery_enabled: bool = False,
        collision_hull_inflation_factor: float = 1.1,
    ) -> None:
        self.target = target
        self.looking_hand_constraint = looking_hand_constraint
        self.recovery_enabled = recovery_enabled
        self.collision_hull_inflation_factor = validate_hull_inflation(collision_hull_inflation_factor)
        super().__init__()





    def build_node(self, context, name):
        self._pick = PickAtPose(object_id=self.target, looking_hand_constraint=self.looking_hand_constraint, recovery_enabled=self.recovery_enabled, collision_hull_inflation_factor=self.collision_hull_inflation_factor)
        self._pick_scope = self._pick.build_node(context, "Pick at observed pose")
        self._stow_node = MoveArmToState(state="go").build_node(context, "Stow arm")
        self._clear_node = _ClearCollisionSceneAction().build_node(context, "Clear pick scene")
        children = [self._pick_scope, self._stow_node, self._clear_node]
        return ActionScope(self, context, name, py_trees.composites.Sequence(name="PickTarget operations", memory=True, children=children))

    def on_enter(self, context):
        if not self.target:
            return ActionResult(status="failure", reason="PickTarget requires target", failure=FailureInfo(FailureCode.INVALID_REQUEST))
        ws = context.world_state
        matrix = ws.get(f"{self.target}_estimated_pick_pose_mat")
        if matrix is None:
            return ActionResult(status="failure", reason=f"No estimated pick pose for {self.target}", failure=FailureInfo(FailureCode.INVALID_STATE))
        x, y, z, roll, pitch, yaw = SensorDataCenter.get_tf_matrix_to_xyz_rpy(matrix)
        self._pick.x, self._pick.y, self._pick.z = x, y, z
        self._pick.roll, self._pick.pitch, self._pick.yaw = roll, pitch, yaw
        self._pick.frame_id = ws.get(f"{self.target}_estimated_pick_pose_frame") or context.hardware.get("odom", "odom")
        self._pick.object_id = self.target
        width = ws.get(f"{self.target}_estimated_width")
        self._pick.width = 0.3 if width is None else width

    def on_complete(self, context, status):
        if self._pick_scope.result is not None and self._pick_scope.result.status != "success":
            return self._pick_scope.result
        if status == py_trees.common.Status.FAILURE and self._pick_scope.result is not None:
            stage, cause = next(((stage, node.result) for stage, node in (("stow", self._stow_node), ("scene_cleanup", self._clear_node)) if node.result and node.result.status != "success"), ("post_pick", None))
            return ActionResult(status="needs_replan", reason=cause.reason if cause else "pick_target_failed_after_pick", force_replan=True, failure=cause.failure if cause else FailureInfo(),
                suggested_replan_context={"action": "PickTarget", "target": self.target, "failed_stage": stage, "grip_state": self._pick._hold_state()})
        return ActionResult(status="success" if status == py_trees.common.Status.SUCCESS else "failure")



class PickAtPose(AbstractAction):
    """Pick an object at given pose using the arm, steps as follows:
        > approach the object at an offset,
        > go to estimated grasp pose
        > close grip
        > retreat

    Params:
        x/y/z: float - position,
        roll/pitch/yaw: float - orientation,
        width: float - width to close the gripper
        frame_id: str - target grasp pose
        retreat_z: float - offset for lift before and after grasp (default 0.1 m)
        force_N: float - force used to close the gripper (default 0.5 N)
    """

    # retreat_mode: str - "world_z" or "grasp_minus_x" offset mode
    # looking_hand_constraint: bool - if true, request HSRB hand-in-view
    #     Cartesian planning for each pose move in this pick

    required_controllers = ["arm", "gripper"]

    def __init__(
        self,
        x: float = 0.0,
        y: float = 0.0,
        z: float = 0.0,
        roll: float = 0.0,
        pitch: float = 0.0,
        yaw: float = 0.0,
        frame_id: str = 'odom',
        width: float = 0.15,
        retreat_z: float = 0.0,
        force_N: float = 0.25,
        retreat_mode: str = 'world_z',
        looking_hand_constraint: bool = False,
        *,
        object_id: str = '',
        pick_height_clearance_m: float = 0.06,
        recovery_enabled: bool = False,
        collision_hull_inflation_factor: float = 1.1,
        approach_planning_scope: str = 'whole_body',
        approach_planning_motion: str = 'pose',
        grasp_planning_scope: str = 'whole_body',
        grasp_planning_motion: str = 'pose',
        retreat_planning_scope: str = 'whole_body',
        retreat_planning_motion: str = 'pose',
    ) -> None:
        self.x = x
        self.y = y
        self.z = z
        self.roll = roll
        self.pitch = pitch
        self.yaw = yaw
        self.frame_id = frame_id
        self.width = width
        self.retreat_z = retreat_z
        self.force_N = force_N
        self.retreat_mode = retreat_mode
        self.looking_hand_constraint = looking_hand_constraint
        self.object_id = object_id
        self.pick_height_clearance_m = pick_height_clearance_m
        self.recovery_enabled = recovery_enabled
        self.collision_hull_inflation_factor = validate_hull_inflation(collision_hull_inflation_factor)
        self.approach_planning_scope = approach_planning_scope
        self.approach_planning_motion = approach_planning_motion
        self.grasp_planning_scope = grasp_planning_scope
        self.grasp_planning_motion = grasp_planning_motion
        self.retreat_planning_scope = retreat_planning_scope
        self.retreat_planning_motion = retreat_planning_motion
        super().__init__()
        self._grasp_pose = _pose_from_params(self.param_dict(), default_frame=self.frame_id)
        self._active_pick_offset = abs(float(retreat_z))
        self.recovery_enabled = bool(recovery_enabled)
        log.warn('self.width', self.width)





    def _pick_z_increment(self, context: Context, target_names: list[str], object_pcd_array: Any | None) -> float:
        clearance = float(getattr(self, "pick_height_clearance_m", 0.06))
        requested = abs(float(self.retreat_z))
        ws = context.world_state
        if ws is not None:
            for target_name in target_names:
                if not hasattr(ws, "get_movable_object_pick_z_increment"):
                    continue
                selected, metadata = ws.get_movable_object_pick_z_increment(
                    target_name,
                    object_points=object_pcd_array,
                    requested_z_inc_m=requested,
                    clearance_m=clearance,
                )
                ws.set(f"{target_name}_pick_z_increment", metadata)
                log.info("Pick z increment for %s: %.3f metadata=%s", target_name, selected, metadata)
                return float(selected)
        return requested

    def _snapshot_pick_start_base(self, context: Context, target_names: list[str]) -> None:
        if context.world_state is None or context.sensor_data_center is None:
            return
        frame_id = str(context.hardware.get("odom", "odom")) or "odom"
        try:
            robot_pose = context.sensor_data_center.get_robot_pose_global(global_frame=frame_id)
            if not robot_pose:
                return
            pose = robot_pose.get("pose") if isinstance(robot_pose, dict) else robot_pose
            base_pose = {
                "x": float(getattr(pose, "x", 0.0)),
                "y": float(getattr(pose, "y", 0.0)),
                "yaw": float(getattr(pose, "yaw", 0.0)),
                "frame_id": frame_id,
            }
            context.world_state.set("last_pick_start_base_pose", base_pose)
            for target_name in target_names:
                context.world_state.set(f"{target_name}_pick_start_base_pose", base_pose)
        except Exception as exc:  # noqa: BLE001
            log.debug("Unable to snapshot pick start base pose: %s", exc)

    def _pick_line_axis(self, segment: str) -> list[float] | None:
        if self.retreat_mode != "grasp_minus_x":
            return None
        if segment == "grasp":
            return [1.0, 0.0, 0.0]
        if segment == "retreat":
            return [-1.0, 0.0, 0.0]
        return None

    def _pick_line_distance(self, segment: str) -> float | None:
        if self.retreat_mode != "grasp_minus_x" or segment not in {"grasp", "retreat"}:
            return None
        return abs(float(self._active_pick_offset))




    def _cleanup_collision_scene(self, context: Context) -> None:
        try:
            context.require_arm().clear_collision_scene()
        except Exception as exc:  # noqa: BLE001
            log.debug("PickAtPose collision cleanup skipped: %s", exc)

    def build_node(self, context, name):
        self._moveit_scene = type(context.require_arm()).__name__ == "MoveItArmController"
        self._approach = _MoveArmToPose(ArmPoseRequest(self._grasp_pose, segment="approach"))
        self._grasp = _MoveArmToPose(ArmPoseRequest(self._grasp_pose, segment="grasp"))
        self._retreat = _MoveArmToPose(ArmPoseRequest(self._grasp_pose, segment="retreat"))
        self._release = ReleaseObjectAction(distance=self.width + 0.02)
        self._close = GraspObjectAction(force_N=self.force_N, width_m=self.width - 0.02)
        self._target_mesh = _AddCollisionMeshAction(name="pick_target_collision", convex_hull=True, inflation_factor=self.collision_hull_inflation_factor)
        self._attach = _AttachCollisionObjectAction("pick_target_collision")
        self._motion_nodes = {}
        motions = {}
        for stage, action in (("approach", self._approach), ("grasp", self._grasp), ("retreat", self._retreat)):
            leaf = action.build_node(context, stage)
            self._motion_nodes[stage] = leaf
            if self.recovery_enabled:
                attempts = py_trees.decorators.Retry(name=f"Retry {stage}", child=leaf, num_failures=2)
                def eligible(leaf=leaf):
                    result = leaf.result
                    return result is None or result.status != "failure" or (result.failure is not None and result.failure.retryable)
                motions[stage] = py_trees.decorators.EternalGuard(name=f"Retryable {stage}", child=attempts, condition=eligible)
            else:
                motions[stage] = leaf
        self._close_node = self._close.build_node(context, "Close gripper")
        self._target_node = self._target_mesh.build_node(context, "Model target")
        self._attach_node = self._attach.build_node(context, "Attach target")
        children = []
        if not self._moveit_scene:
            self._initial_mesh = _AddCollisionMeshAction(name="pick_target_collision", convex_hull=True, inflation_factor=self.collision_hull_inflation_factor)
            self._initial_node = self._initial_mesh.build_node(context, "Initial target mesh")
            children.append(self._initial_node)
        children.extend([_ClearCollisionSceneAction().build_node(context, "Clear scene"),
                         self._target_node, motions["approach"],
                         self._release.build_node(context, "Open gripper")])
        if not self._moveit_scene:
            children.append(_RemoveCollisionObjectAction("pick_target_collision").build_node(context, "Remove contact target"))
        children.extend([motions["grasp"], self._close_node])
        if self._moveit_scene:
            children.extend([self._attach_node, _AllowCollisionPairsAction([("pick_target_collision", "pick_surround_collision")]).build_node(context, "Allow attached contact")])
        children.append(motions["retreat"])
        return ActionScope(self, context, name, py_trees.composites.Sequence(name="Pick operations", memory=True, children=children))

    def on_enter(self, context):
        self._grasp_pose = _pose_from_params(self.param_dict(), default_frame=self.frame_id)
        names = [self.object_id] if self.object_id else []
        ws = context.world_state
        pcd = next((ws.get(f"{target}_object_pcd") for target in names if ws.get(f"{target}_object_pcd") is not None), None)
        self._snapshot_pick_start_base(context, names)
        self._active_pick_offset = 0.15  # Preserve current effective pick offset.
        self._target_mesh.pcd = pcd
        self._target_mesh.frame_id = (ws.get(f"{self.object_id}_pcd_frame") if self.object_id else None) or self.frame_id
        if not self._moveit_scene:
            self._initial_mesh.pcd = pcd
            self._initial_mesh.frame_id = self._target_mesh.frame_id
        self._release.distance = self.width + 0.02
        self._close.width_m = self.width - 0.02
        self._close.force_N = self.force_N
        for stage, action, pose in (("approach", self._approach, _offset_pose(self._grasp_pose, 0.15, self.retreat_mode)),
                                   ("grasp", self._grasp, self._grasp_pose),
                                   ("retreat", self._retreat, _offset_pose(self._grasp_pose, 0.15, self.retreat_mode))):
            action.request = self._segment_request(stage, pose)

    def _hold_state(self):
        if not self._close_node.started:
            return "open"
        result = self._close_node.result
        return "commanded_hold" if result is not None and result.status == "success" else "uncertain"

    def on_complete(self, context, status):
        if status == py_trees.common.Status.SUCCESS:
            return ActionResult(status="success")
        stage, cause = next(((stage, node.result) for stage, node in self._motion_nodes.items() if node.result is not None and node.result.status == "failure"), ("close", self._close_node.result))
        if self._attach_node.result is not None and self._attach_node.result.status == "failure":
            stage, cause = "attach", self._attach_node.result
        hold = self._hold_state()
        if hold == "open":
            self._cleanup_collision_scene(context)
            for node in ([self._initial_node] if not self._moveit_scene else []) + [self._target_node]:
                if node.result is not None and node.result.status == "failure":
                    return node.result
        if self.recovery_enabled or hold != "open":
            return ActionResult(status="needs_replan", reason=cause.reason if cause else "pick_failed", force_replan=True,
                failure=cause.failure if cause else FailureInfo(), suggested_replan_context={"action": "PickAtPose", "target": self.object_id,
                    "failed_stage": stage, "grip_state": hold,
                    "attachment_requested": self._attach_node.result is not None and self._attach_node.result.status == "success"})
        return cause or ActionResult(status="failure", reason="pick_failed")

    def on_cancel(self, context):
        if self._hold_state() == "open":
            self._cleanup_collision_scene(context)

    def _segment_request(self, segment: str, pose: Pose3D) -> ArmPoseRequest:
        if not self._moveit_scene:
            scope, motion = {"approach": (self.approach_planning_scope, self.approach_planning_motion), "grasp": (self.grasp_planning_scope, self.grasp_planning_motion), "retreat": (self.retreat_planning_scope, self.retreat_planning_motion)}[segment]
            return ArmPoseRequest(pose, segment=segment, scope=scope, motion=motion,
                line=CartesianLine(axis=tuple(self._pick_line_axis(segment)) if self._pick_line_axis(segment) is not None else None, distance=self._pick_line_distance(segment)), looking_hand_constraint=self.looking_hand_constraint)
        approach = segment == "approach"
        return ArmPoseRequest(pose, segment=segment, moveit=MoveItPoseSettings(group_name="whole_body_weighted", goal_mode="ik_joint_goal", constrain_base=True,
            base_constraint_tolerance={"x": 0.1, "y": 0.1, "yaw": 0.5} if approach else {"x": 0.03, "y": 0.03, "yaw": 0.05},
            max_base_path_drift={"xy": 0.2, "yaw": 0.8} if approach else {"xy": 0.08, "yaw": 0.12}, collision_policy="full" if approach else "target_relaxed"))

class PlaceAtPose(AbstractAction):
    """Place an object at given pose using the arm, steps as follows:
        > approach the place pose at an offset,
        > go down close to the estimated place pose
        > open gripper
        > retreat

    Params:
        x/y/z: float - position,
        roll/pitch/yaw: float - orientation,
        width: float - width to open the gripper
        frame_id: str - target place pose frame
        retreat_z: float - offset for lift before and after place (default 0.1 m)
        retreat_mode: str - "world_z" or "grasp_minus_x" offset mode
        looking_hand_constraint: bool - if true, request HSRB hand-in-view
            Cartesian planning for each pose move in this place
        approach_moveit_group/place_moveit_group/retreat_moveit_group:
            optional MoveIt group override per arm-motion child.
        approach_planning_strategy/place_planning_strategy/retreat_planning_strategy:
            optional MoveIt strategy override per arm-motion child.
    """

    required_controllers = ["arm", "gripper"]

    def __init__(
        self,
        x: float = 0.0,
        y: float = 0.0,
        z: float = 0.0,
        roll: float = 0.0,
        pitch: float = 0.0,
        yaw: float = 0.0,
        frame_id: str = 'odom',
        width: float = 0.08,
        retreat_z: float = 0.1,
        retreat_mode: str = 'world_z',
        looking_hand_constraint: bool = False,
        approach_moveit_group: str | None = None,
        approach_planning_strategy: str | None = None,
        place_moveit_group: str | None = None,
        place_planning_strategy: str | None = None,
        retreat_moveit_group: str | None = None,
        retreat_planning_strategy: str | None = None,
    ) -> None:
        self.x = x
        self.y = y
        self.z = z
        self.roll = roll
        self.pitch = pitch
        self.yaw = yaw
        self.frame_id = frame_id
        self.width = width
        self.retreat_z = retreat_z
        self.retreat_mode = retreat_mode
        self.looking_hand_constraint = looking_hand_constraint
        self.approach_moveit_group = approach_moveit_group
        self.approach_planning_strategy = approach_planning_strategy
        self.place_moveit_group = place_moveit_group
        self.place_planning_strategy = place_planning_strategy
        self.retreat_moveit_group = retreat_moveit_group
        self.retreat_planning_strategy = retreat_planning_strategy
        super().__init__()
        self._place_pose = _pose_from_params(self.param_dict(), default_frame=self.frame_id)





    def build_node(self, context, name):
        self._approach = _MoveArmToPose(ArmPoseRequest(self._place_pose, segment="approach"))
        self._place = _MoveArmToPose(ArmPoseRequest(self._place_pose, segment="place"))
        self._retreat = _MoveArmToPose(ArmPoseRequest(self._place_pose, segment="retreat"))
        self._release = ReleaseObjectAction(distance=self.width)
        children = [_AddCollisionMeshAction().build_node(context, "Model place scene"), self._approach.build_node(context, "Approach place"), self._place.build_node(context, "Reach place"),
                    self._release.build_node(context, "Release object"), self._retreat.build_node(context, "Retreat")]
        return ActionScope(self, context, name, py_trees.composites.Sequence(name="Place operations", memory=True, children=children))

    def on_enter(self, context):
        self._place_pose = _pose_from_params(self.param_dict(), default_frame=self.frame_id)
        self._release.distance = self.width
        for stage, action, pose, group, strategy in (
            ("approach", self._approach, _offset_pose(self._place_pose, self.retreat_z, self.retreat_mode), self.approach_moveit_group, self.approach_planning_strategy),
            ("place", self._place, self._place_pose, self.place_moveit_group, self.place_planning_strategy),
            ("retreat", self._retreat, _offset_pose(self._place_pose, self.retreat_z, self.retreat_mode), self.retreat_moveit_group, self.retreat_planning_strategy)):
            action.request = ArmPoseRequest(pose, segment=stage, looking_hand_constraint=self.looking_hand_constraint, moveit=MoveItPoseSettings(group_name=group, planning_strategy=strategy))
