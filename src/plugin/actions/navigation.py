"""Navigation plugin actions."""

from __future__ import annotations

import math
from typing import Any, Callable

import numpy as np

import py_trees
from src.tree.bt_node_wrapper import ActionScope, BtConditionLeaf
from src.core.failure import FailureInfo, FailureCode
from src.core.action import AbstractAction, AsyncAction, ActionResult
from src.core.context import Context
from src.core.geometry import wrap_angle, yaw_delta
from src.core.manipulation_types import BasePose2D, FeasibilityResult, MoveItReachabilityRequest, ReachTarget
from src.core.planners import PlannerResult
from src.plugin.base_pose_selector import BasePoseSelectorConfig, ManipulationBasePoseSelector
from src.plugin.nav_feasibility import NavFeasibilityConfig, NavigationFeasibilityChecker
from src.plugin.actions._helpers import ControllerAsyncAction, _arm_status_result, _nav_status_result, scene_location_description
import src.core.logger as log 


def _head_joint_snapshot(context: Context, *, timeout_s: float = 5.) -> dict[str, float] | None:
    try:
        sensor = context.sensor_data_center
        if sensor is not None:
            read = sensor.peek_latest if timeout_s == 0. and callable(getattr(sensor,"peek_latest",None)) else sensor.get_latest
            msg = read("joint_states",timeout_s=timeout_s)
        else:
            msg = None
        data = msg.data if msg is not None else None
        names = list(getattr(data, "name", []) or [])
        positions = list(getattr(data, "position", []) or [])
        values = {
            name: float(positions[index])
            for index, name in enumerate(names)
            if name in {"head_pan_joint", "head_tilt_joint"} and index < len(positions)
        }
        return values or None
    except Exception as exc:  # noqa: BLE001
        log.debug("Unable to read head joint snapshot: %s", exc)
        return None


def _joint_snapshot(context: Context) -> dict[str, float] | None:
    try:
        msg = context.sensor_data_center.get_latest("joint_states") if context.sensor_data_center else None
        data = msg.data if msg is not None else None
        names = list(getattr(data, "name", []) or [])
        positions = list(getattr(data, "position", []) or [])
        values = {
            str(name): float(positions[index])
            for index, name in enumerate(names)
            if index < len(positions)
        }
        return values or None
    except Exception as exc:  # noqa: BLE001
        log.debug("Unable to read joint snapshot: %s", exc)
        return None


def _view_joint_limits(context: Context) -> dict[str, tuple[float, float]]:
    limits = {
        "arm_lift_joint": (0.0, 0.4),
        "head_pan_joint": (-1.75, 1.75),
        "head_tilt_joint": (-1.57, 0.52),
    }
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
    return limits


def _latest_camera_info(context: Context, sensor_name: str) -> Any | None:
    try:
        msg = context.sensor_data_center.get_latest(sensor_name) if context.sensor_data_center else None
        return msg.data if msg is not None else None
    except Exception:
        return None


def _camera_fov(camera_info: Any | None) -> tuple[float | None, float | None]:
    if camera_info is None:
        return None, None
    width = float(getattr(camera_info, "width", 0.0) or 0.0)
    height = float(getattr(camera_info, "height", 0.0) or 0.0)
    k = getattr(camera_info, "k", None)
    if k is None or len(k) < 6 or width <= 0.0 or height <= 0.0:
        return None, None
    fx = float(k[0])
    fy = float(k[4])
    if fx <= 0.0 or fy <= 0.0:
        return None, None
    return 2.0 * math.atan(width / (2.0 * fx)), 2.0 * math.atan(height / (2.0 * fy))


def _pan_tilt_to_point(point_base: np.ndarray, camera_z: float) -> tuple[float, float]:
    dx = float(point_base[0])
    dy = float(point_base[1])
    dz = float(point_base[2]) - float(camera_z)
    horizontal = max(1e-6, math.hypot(dx, dy))
    return math.atan2(dy, dx), math.atan2(dz, horizontal)


def _view_coverage(
    points_base: np.ndarray,
    camera_z: float,
    pan: float,
    tilt: float,
    hfov: float | None,
    vfov: float | None,
) -> float:
    if not hfov or not vfov:
        return 0.0
    inside = 0
    for point in points_base:
        yaw, pitch = _pan_tilt_to_point(point, camera_z)
        if abs(yaw_delta(yaw, pan)) <= hfov * 0.5 and abs(yaw_delta(pitch, tilt)) <= vfov * 0.5:
            inside += 1
    return float(inside) / float(max(1, len(points_base)))


def _center_view_error(point_base: np.ndarray, camera_z: float, pan: float, tilt: float) -> float:
    yaw, pitch = _pan_tilt_to_point(point_base, camera_z)
    return abs(yaw_delta(yaw, pan)) + abs(yaw_delta(pitch, tilt))


def _clamp(value: float, lo: float, hi: float) -> float:
    return float(min(max(float(value), float(lo)), float(hi)))

def _navigation_goal_reached(context: Context, x: float, y: float, yaw: float, frame_id: str) -> bool:
    if context.sensor_data_center is None:
        return False
    try:
        robot_pose = context.sensor_data_center.get_robot_pose_global(global_frame=frame_id)
        if robot_pose is None:
            return False
        pose = robot_pose.get("pose")
        current_x = float(getattr(pose, "x", robot_pose.get("x", 0.0)))
        current_y = float(getattr(pose, "y", robot_pose.get("y", 0.0)))
        current_yaw = float(getattr(pose, "yaw", robot_pose.get("yaw", 0.0)))
        nav_cfg = context.hardware.get("nav", {}) if isinstance(context.hardware, dict) else {}
        xy_tolerance = float(nav_cfg.get("xy_goal_tolerance", 0.10))
        yaw_tolerance = float(nav_cfg.get("yaw_goal_tolerance", 0.20))
        distance = math.hypot(current_x - float(x), current_y - float(y))
        yaw_error = abs(yaw_delta(current_yaw, float(yaw)))
        return distance <= xy_tolerance and yaw_error <= yaw_tolerance
    except Exception as exc:  # noqa: BLE001
        log.debug("Unable to evaluate navigation goal tolerance: %s", exc)
        return False

class MoveToPose(ControllerAsyncAction):
    """Navigate the mobile base to ``(x, y, yaw)``.
    Avoid using this directly in the behaviour tree. Use wrappers that build on it.
    Params:
        float: x, y, yaw - goal pose in *frame_id*
        str: frame_id - TF frame for the goal (default ``map``)
        bool: allow_goal_update - if true and nav is already running, hot-update goal
    """

    required_controllers = ["nav"]

    def __init__(
        self,
        x: float = 0.0,
        y: float = 0.0,
        yaw: float = 0.0,
        frame_id: str = 'map',
        allow_goal_update: bool = True,
    ) -> None:
        self.x = x
        self.y = y
        self.yaw = yaw
        self.frame_id = frame_id
        self.allow_goal_update = allow_goal_update
        super().__init__()
        self._dispatch_failed_reason = None

    def start(self, context: Context) -> None:
        try:
            nav = context.require_nav()
            target = (self.x, self.y, self.yaw)
            log.info('target:', target, len(target))
            head_before = _head_joint_snapshot(context)
            if head_before:
                log.info("MoveToPose head joints before nav goal: %s", head_before)
                
            if (self.allow_goal_update and nav.get_status() == "in_progress"):
                nav.update_goal(target, frame_id=self.frame_id)
            else:
                nav.send_goal(target, frame_id=self.frame_id)
        except Exception as exc:  # noqa: BLE001
            self._dispatch_failed_reason = str(exc)
            log.exception("Failed to dispatch navigation goal: %s", exc)

    def check_result(self, context: Context) -> ActionResult:
        if self._cancel_requested:
            return ActionResult(status="failure", reason="Action Cancelled")
        if self._dispatch_failed_reason:
            return ActionResult(status="failure", reason=self._dispatch_failed_reason)
        if _navigation_goal_reached(context, self.x, self.y, self.yaw, self.frame_id):
            if context.nav:
                context.nav.cancel_goal()
            return ActionResult(
                status="success",
                reason="Navigation goal reached within tolerance",
                payload={"pose": {"x": self.x, "y": self.y, "yaw": self.yaw}},
            )
        return _nav_status_result(context.require_nav())

    def reset(self) -> None:
        super().reset()
        self._dispatch_failed_reason = None
    
    def cancel(self, context):
        log.error("MoveToPose cancel requested")
        self._cancel_requested = True
        if context.nav:
            context.nav.cancel_goal()
        return


class MoveBaseWithWholeBody(ControllerAsyncAction):
    """Move the mobile base through the arm controller's whole-body MoveIt group.

    This is intended for short manipulation escape/undock motions where Nav2 may
    reject the current costmap pose. It plans base virtual joints through MoveIt
    and executes through ros2_control. With plan_motion=False, send only the
    resolved odom base-joint target directly through the controller.
    """

    required_controllers = ["arm"]

    def __init__(
        self,
        x: float | None = None,
        y: float | None = None,
        yaw: float | None = None,
        frame_id: str = 'odom',
        relative_x: float = -0.35,
        relative_y: float = 0.0,
        relative_yaw: float = 0.0,
        return_to_pick_start: bool = False,
        target: str = '',
        moveit_group: str = 'whole_body_weighted',
        pose_key: str | None = None,
        plan_motion: bool = True,
        relative_in_base_frame: bool = True,
    ) -> None:
        self.x = x
        self.y = y
        self.yaw = yaw
        self.frame_id = frame_id
        self.relative_x = relative_x
        self.relative_y = relative_y
        self.relative_yaw = relative_yaw
        self.return_to_pick_start = return_to_pick_start
        self.target = target
        self.moveit_group = moveit_group
        self.pose_key = pose_key
        self.plan_motion = plan_motion
        self.relative_in_base_frame = relative_in_base_frame
        super().__init__()
        self._dispatch_failed_reason: str | None = None
        self._target_pose: BasePose2D | None = None
        self._dispatched = False
        self._goal_sent = False

    def start(self, context: Context) -> None:
        self._generation = getattr(self, "_generation", 0) + 1
        generation = self._generation
        self._cancel_requested = False
        self._dispatched = False
        self._goal_sent = False
        self._dispatch_failed_reason = None
        def current():
            return generation == self._generation and not self._cancel_requested and context.execution.control is None and not context.execution.cancel_requested
        def work():
            if not current():
                return
            self._target_pose = self._resolve_base_goal(context)
            log.info("Base move target=%s plan_motion=%s hold_current_joints=True", self._target_pose,self.plan_motion)
            if not current():
                return
            if not self.plan_motion and self._target_pose.frame_id != 'odom':
                raise ValueError("direct base joint goals require an odom-frame target")
            self._goal_sent = True
            if self.plan_motion:
                context.require_arm().send_base_pose_goal(self._target_pose, moveit_group=str(self.moveit_group),
                    segment="whole_body_base_escape", collision_policy="full", hold_current_joints=True)
            else:
                context.require_arm().send_joint_goal({"odom_x":self._target_pose.x,"odom_y":self._target_pose.y,"odom_t":self._target_pose.yaw})
            if not current():
                context.require_arm().cancel_goal()
        def complete(_):
            if current():
                self._dispatched = True
                if context.require_arm().get_status() == "failed":
                    self._dispatch_failed_reason = context.require_arm().get_error() or "whole-body base move failed"
        def failed(exc):
            if current():
                self._dispatch_failed_reason = str(exc)
                log.warn("Base move failed: %s", exc)
        context.run_in_context_loop(work, callback=complete, error_callback=failed)

    def check_result(self, context: Context) -> ActionResult:
        if self._cancel_requested:
            return ActionResult(status="failure", reason="Action Cancelled")
        if self._dispatch_failed_reason:
            return ActionResult(status="failure", reason=self._dispatch_failed_reason)
        if not self._dispatched:
            return ActionResult(status="running")
        result = _arm_status_result(context.require_arm())
        if result.status == "success" and self._target_pose is not None:
            return ActionResult(
                status="success",
                reason="Whole-body base move completed",
                payload={"pose": self._target_pose.to_dict()},
            )
        return result

    def reset(self) -> None:
        super().reset()
        self._dispatch_failed_reason = None
        self._target_pose = None
        self._dispatched = False
        self._goal_sent = False

    def cancel(self, context: Context) -> None:
        if self._cancel_requested:
            return
        self._cancel_requested = True
        self._generation = getattr(self, "_generation", 0) + 1
        if context.arm and self._goal_sent:
            context.arm.cancel_goal()

    def _resolve_base_goal(self, context: Context) -> BasePose2D:
        if self.pose_key is not None:
            value = context.world_state.get(self.pose_key)
            if value is None:
                raise ValueError(f"base goal missing: {self.pose_key}")
            return BasePose2D.from_mapping(value, default_frame_id=str(self.frame_id))
        if self.return_to_pick_start:
            pose = self._pick_start_pose(context)
            if pose is not None:
                return pose
        if self.x is not None and self.y is not None and self.yaw is not None:
            return BasePose2D(
                x=float(self.x),
                y=float(self.y),
                yaw=float(self.yaw),
                frame_id=str(self.frame_id),
            )
        current = self._current_base_pose(context, str(self.frame_id))
        dx = float(self.relative_x)
        dy = float(self.relative_y)
        yaw = float(current.yaw)
        world_dx = math.cos(yaw) * dx - math.sin(yaw) * dy if self.relative_in_base_frame else dx
        world_dy = math.sin(yaw) * dx + math.cos(yaw) * dy if self.relative_in_base_frame else dy
        goal_yaw = yaw if not self.plan_motion and self.relative_yaw == 0. else wrap_angle(yaw + float(self.relative_yaw))
        return BasePose2D(
            x=float(current.x) + world_dx,
            y=float(current.y) + world_dy,
            yaw=goal_yaw,
            frame_id=current.frame_id,
        )

    def _pick_start_pose(self, context: Context) -> BasePose2D | None:
        if context.world_state is None:
            return None
        keys = []
        if self.target:
            keys.append(f"{self.target}_pick_start_base_pose")
        keys.append("last_pick_start_base_pose")
        for key in keys:
            value = context.world_state.get(key)
            if value is None:
                continue
            try:
                return BasePose2D.from_mapping(value, default_frame_id=str(self.frame_id))
            except Exception:
                continue
        return None

    def _current_base_pose(self, context: Context, frame_id: str) -> BasePose2D:
        if context.sensor_data_center is not None:
            robot_pose = context.sensor_data_center.get_robot_pose_global(global_frame=frame_id)
            if robot_pose:
                pose = robot_pose.get("pose") if isinstance(robot_pose, dict) else robot_pose
                return BasePose2D(
                    x=float(getattr(pose, "x", robot_pose.get("x", 0.0) if isinstance(robot_pose, dict) else 0.0)),
                    y=float(getattr(pose, "y", robot_pose.get("y", 0.0) if isinstance(robot_pose, dict) else 0.0)),
                    yaw=float(getattr(pose, "yaw", robot_pose.get("yaw", 0.0) if isinstance(robot_pose, dict) else 0.0)),
                    frame_id=frame_id,
                )
        raise RuntimeError("current_base_pose_unavailable")


class AlignViewToSceneTarget(ControllerAsyncAction):
    """Best-effort post-navigation posture to view a scene-graph target surface."""

    required_controllers = ["arm"]

    def __init__(
        self,
        target: str = '',
        frame_id: str = 'map',
        base_frame: str = 'base_link',
        camera_info_sensor: str = 'head_camera_info',
        camera_height_m: float = 0.9,
        lift_samples: int = 8,
        view_height_margin_m: float = 0.2,
        lift_height_weight: float = 8.0,
        lift_motion_weight: float = 0.5,
        coverage_weight: float = 10.0,
        tilt_motion_weight: float = 0.25,
        align_head: bool = True,
        align_lift: bool = True,
    ) -> None:
        self.target = target
        self.frame_id = frame_id
        self.base_frame = base_frame
        self.camera_info_sensor = camera_info_sensor
        self.camera_height_m = camera_height_m
        self.lift_samples = lift_samples
        self.view_height_margin_m = view_height_margin_m
        self.lift_height_weight = lift_height_weight
        self.lift_motion_weight = lift_motion_weight
        self.coverage_weight = coverage_weight
        self.tilt_motion_weight = tilt_motion_weight
        self.align_head = align_head
        self.align_lift = align_lift
        super().__init__()
        self._dispatch_status = 'idle'
        self._failure_reason: str | None = None
        self._move_started = False
        self.goal: dict[str, float] = {}
        self.metadata: dict[str, Any] = {}

    def start(self, context: Context) -> None:
        def solve_and_send() -> dict[str, Any]:
            try:
                solution = self._solve(context)
                self.metadata = solution.get("metadata", {})
                goal = solution.get("goal") or {}
                if not goal:
                    return {"success": True, "skipped": True}
                context.require_arm().send_joint_goal(goal)
                self.goal = goal
                if context.require_arm().get_status() == "failed":
                    return {"success": False, "error": context.require_arm().get_error() or "view_alignment_goal_failed"}
                return {"success": True, "goal": goal}
            except Exception as exc:  # noqa: BLE001
                log.warn("Skipping scene-target view alignment: %s", exc)
                return {"success": True, "skipped": True, "error": str(exc)}

        def on_dispatched(result: dict[str, Any]) -> None:
            if result.get("success") and not result.get("skipped"):
                self._move_started = True
                self._dispatch_status = "dispatched"
                return
            self._dispatch_status = "succeeded" if result.get("success") else "failure"
            self._failure_reason = str(result.get("error") or "view_alignment_failed") if not result.get("success") else None

        try:
            context.run_in_context_loop(solve_and_send, callback=on_dispatched, error_callback=lambda exc: on_dispatched({"success": True, "skipped": True, "error": str(exc)}))
        except Exception as exc:  # noqa: BLE001
            log.warn("Skipping scene-target view alignment: %s", exc)
            self._dispatch_status = "succeeded"

    def check_result(self, context: Context) -> ActionResult:
        if self._cancel_requested:
            return ActionResult(status="failure", reason="Action Cancelled")
        if self._dispatch_status == "failure":
            return ActionResult(status="success", reason=self._failure_reason or "view_alignment_skipped")
        if self._dispatch_status == "succeeded":
            return ActionResult(status="success", reason="view_alignment_skipped", payload=self.metadata)
        if self._dispatch_status != "dispatched" or not self._move_started:
            return ActionResult(status="running")
        result = _arm_status_result(context.require_arm())
        if result.status == "success":
            return ActionResult(status="success", reason="view_alignment_completed", payload={"goal": self.goal, "metadata": self.metadata})
        return result

    def reset(self) -> None:
        super().reset()
        self._dispatch_status = "idle"
        self._failure_reason = None
        self._move_started = False
        self.goal = {}
        self.metadata = {}

    def cancel(self, context: Context) -> None:
        self._cancel_requested = True
        if context.arm:
            context.arm.cancel_goal()

    def _solve(self, context: Context) -> dict[str, Any]:
        if not self.align_head and not self.align_lift:
            return {"goal": {}, "metadata": {"skipped": "alignment_disabled"}}
        if context.scene_graph is None:
            return {"goal": {}, "metadata": {"skipped": "missing_scene_graph"}}
        from src.core.scene_graph_query import SceneGraphQuery

        query = SceneGraphQuery(context.scene_graph)
        visit = query.get_best_visit_location(str(self.target))
        requested = visit.get("requested_target") if isinstance(visit, dict) else None
        navigation_target = visit.get("navigation_target") if isinstance(visit, dict) else None
        if navigation_target is not None:
            target_id = str(navigation_target)
        elif requested is not None:
            target_id = str(requested)
        else:
            target_id = str(self.target)
        bbox = query.get_object_bbox(target_id)
        if bbox is None and requested:
            bbox = query.get_object_bbox(str(requested))
            target_id = str(requested)
        if bbox is None:
            return {"goal": {}, "metadata": {"skipped": "missing_bbox", "target": str(self.target)}}
        top_points = self._bbox_top_points(np.asarray(bbox, dtype=np.float64))
        points_base = self._points_in_base(context, top_points)
        if points_base is None or points_base.size == 0:
            return {"goal": {}, "metadata": {"skipped": "missing_bbox_transform", "target": target_id}}
        current = _joint_snapshot(context)
        if not current:
            return {"goal": {}, "metadata": {"skipped": "missing_joint_state", "target": target_id}}
        limits = _view_joint_limits(context)
        camera_info = _latest_camera_info(context, str(self.camera_info_sensor))
        candidates = self._candidate_view_goals(points_base, current, limits, camera_info)
        if not candidates:
            return {"goal": {}, "metadata": {"skipped": "no_view_solution", "target": target_id}}
        selected = min(candidates, key=lambda item: float(item["score"]))
        return {
            "goal": selected["goal"],
            "metadata": {
                "target": target_id,
                "requested_target": requested,
                "navigation_target": navigation_target,
                "method": selected["method"],
                "score": float(selected["score"]),
                "coverage": float(selected.get("coverage", 0.0)),
                "focus_base": np.mean(points_base, axis=0).astype(float).tolist(),
                "bbox_top_z": float(np.max(points_base[:, 2])),
                "desired_camera_z": float(selected.get("desired_camera_z", float("nan"))),
                "selected_camera_z": float(selected.get("camera_z", float("nan"))),
                "current_lift": float(current.get("arm_lift_joint", 0.0)),
                "selected_lift": float(selected["goal"].get("arm_lift_joint", current.get("arm_lift_joint", 0.0))),
            },
        }

    @staticmethod
    def _bbox_top_points(bbox: np.ndarray) -> np.ndarray:
        xmin, ymin, _zmin, xmax, ymax, zmax = [float(v) for v in bbox[:6]]
        return np.asarray(
            [
                [xmin, ymin, zmax],
                [xmin, ymax, zmax],
                [xmax, ymin, zmax],
                [xmax, ymax, zmax],
                [(xmin + xmax) * 0.5, (ymin + ymax) * 0.5, zmax],
            ],
            dtype=np.float64,
        )

    def _points_in_base(self, context: Context, points: np.ndarray) -> np.ndarray | None:
        if str(self.frame_id) == str(self.base_frame):
            return points
        if context.sensor_data_center is None:
            return None
        try:
            return context.sensor_data_center.transform_data(points, str(self.frame_id), str(self.base_frame))
        except Exception as exc:  # noqa: BLE001
            log.warn("Unable to transform scene bbox for view alignment: %s", exc)
            return None

    def _candidate_view_goals(
        self,
        points_base: np.ndarray,
        current: dict[str, float],
        limits: dict[str, tuple[float, float]],
        camera_info: Any | None,
    ) -> list[dict[str, Any]]:
        lift_now = float(current.get("arm_lift_joint", 0.0))
        lift_min, lift_max = limits["arm_lift_joint"]
        bbox_top_z = float(np.max(points_base[:, 2]))
        desired_camera_z = _clamp(
            bbox_top_z + float(self.view_height_margin_m),
            float(self.camera_height_m) + lift_min,
            float(self.camera_height_m) + lift_max,
        )
        desired_lift = desired_camera_z - float(self.camera_height_m)
        lo = max(lift_min, min(lift_now - 0.08, desired_lift - 0.10))
        hi = min(lift_max, max(lift_now + 0.35, desired_lift + 0.10))
        if hi < lo:
            lo, hi = lift_min, lift_max
        lifts = np.linspace(lo, hi, max(1, int(self.lift_samples))) if self.align_lift else [lift_now]
        hfov, vfov = _camera_fov(camera_info)
        candidates: list[dict[str, Any]] = []
        for lift in lifts:
            camera_z = float(self.camera_height_m) + float(lift)
            focus = np.mean(points_base, axis=0)
            pan, tilt = _pan_tilt_to_point(focus, camera_z)
            pan = _clamp(pan, *limits["head_pan_joint"])
            tilt = _clamp(tilt, *limits["head_tilt_joint"])
            if not self.align_head:
                pan = float(current.get("head_pan_joint", 0.0))
                tilt = float(current.get("head_tilt_joint", 0.0))
            coverage = _view_coverage(points_base, camera_z, pan, tilt, hfov, vfov) if hfov and vfov else 0.0
            motion = (
                float(self.lift_motion_weight) * abs(float(lift) - lift_now)
                + 0.25 * abs(pan - float(current.get("head_pan_joint", 0.0)))
                + float(self.tilt_motion_weight) * abs(tilt - float(current.get("head_tilt_joint", 0.0)))
            )
            center_error = _center_view_error(focus, camera_z, pan, tilt)
            method = "fov_coverage" if hfov and vfov else "top_center"
            height_error = abs(float(camera_z) - desired_camera_z)
            score = (
                float(self.lift_height_weight) * height_error
                - float(self.coverage_weight) * coverage
                + center_error
                + motion
                + 0.5 * abs(float(tilt))
            )
            candidates.append(
                {
                    "goal": dict(
                        ({"arm_lift_joint": float(lift)} if self.align_lift else {}),
                        **({"head_pan_joint": float(pan), "head_tilt_joint": float(tilt)} if self.align_head else {}),
                    ),
                    "score": float(score),
                    "coverage": float(coverage),
                    "method": method,
                    "camera_z": float(camera_z),
                    "desired_camera_z": float(desired_camera_z),
                    "height_error": float(height_error),
                }
            )
        return candidates
    
class MoveToTarget(AbstractAction):
    """Navigate the mobile base to `(target object) whose location is known in the scenegraph`.
    Always prefer this over MoveToPose whenever possible so that the viable physical locations can be queried smartly by the action itself.
    Params:
        str: target - the exact same string name as available in the scene graph
        bool: allow_goal_update - if true and nav is already running, hot-update goal
    """

    required_controllers = ["nav", "arm"]

    def __init__(self, target: str='', allow_goal_update: bool=True, prefer_nearest: bool=False) -> None:
        self.target = target
        self.allow_goal_update = allow_goal_update
        self.prefer_nearest = prefer_nearest
        super().__init__()





    def build_node(self, context, name):
        from src.plugin.actions.manipulation import MoveArmToState
        from src.plugin.actions.head import MoveHeadToState
        self._navigation = MoveToPose()
        self._align = AlignViewToSceneTarget(target=self.target, align_head=False, align_lift=True)
        actions = [MoveArmToState(state="go"), MoveHeadToState(state="center"), self._navigation, self._align]
        return ActionScope(self, context, name, py_trees.composites.Sequence(name="MoveToTarget operations", memory=True, children=[action.build_node(context, label) for action, label in zip(actions, ["Stow arm", "Center head", "Navigate", "Align view"])]))

    def on_enter(self, context):
        from src.core.scene_graph_query import SceneGraphQuery
        reason = None
        if context.scene_graph is None:
            reason = "Scene graph not available"
        elif not isinstance(self.target, str):
            reason = "Target must be a string"
        else:
            query = SceneGraphQuery(context.scene_graph)
            current = context.require_nav().get_current_pose() if self.prefer_nearest else None
            pose = query.get_nearest_visit_location(self.target, [current.x, current.y]) if current else query.get_best_visit_location(self.target)
            if pose is None:
                reason = f"Viable move locations not known for '{self.target}'"
            else:
                self._navigation.x, self._navigation.y, self._navigation.yaw = pose["x"], pose["y"], pose["yaw"]
                self._navigation.frame_id = pose["frame_id"]
                self._align.frame_id = pose["frame_id"]
                self._align.target = self.target
                from ._helpers import scene_node_reference
                furniture_id=query.get_supporting_furniture(self.target) or self.target
                self._highlight=scene_node_reference(context,furniture_id)
                context.world_state.set("scene_navigation_target",self._highlight)
                furniture, room = scene_location_description(context, self.target)
                log.speak("Going to %s%s", furniture, f" in room {room}" if room else "")
        if reason:
            return ActionResult(status="needs_replan", reason=reason, suggested_replan_context={"target": self.target, "reason": reason}, failure=FailureInfo(FailureCode.INVALID_STATE))

    def on_complete(self,context,status):
        self.on_cancel(context)
        return super().on_complete(context,status)

    def on_cancel(self,context):
        if context.world_state.get("scene_navigation_target")==getattr(self,"_highlight",None):
            context.world_state.set("scene_navigation_target",None)


class MoveToManipulationPose(AbstractAction):
    """Navigate to a selected manipulation base pose for a target.

    Use this only when the robot is already near the target object of interest.
    Broad search/exploration should happen before this action. The action uses
    fresh movable-object state from ``WorldState``; when that state is missing
    or stale it runs ``ObserveScene(target=...)`` internally, then performs a
    coarse-to-fine base-pose search with nav feasibility before the configured
    manipulation planner's XYZ reachability check. This action only asks if the
    end effector can reach the target centroid position; grasp orientation is
    checked later by grasp estimation and pick execution.

    Params:
        target: str - target object id.
        use_ik: bool - if true, call the configured manipulation planner's IK
                       check during scoring.
        require_ik: bool - if true, reject candidates when IK is unavailable or
                           infeasible.
        object_pose_ttl_s: float - optional freshness window for object pose.
        reach_validate_joint_plan: bool - validate successful probes with a
                                          joint-constraint plan.
        reach_position_tolerance_m: float - position-only reach radius.
        
    """
    #     min_radius/max_radius/radius_step: float - coarse ring search bounds.
    #     preferred_radius: float - preferred object-to-base distance.
    #     angle_step_rad: float - coarse angular sampling step.
    #     refine_top_k/refine_iterations: int - local refinement controls.
    #     planner: str - optional planner name; defaults to context default.
    #     preferred_angles: list[float] - optional object-relative base angles to
    #                                   sample instead of a full ring.
    #     evaluate_ik_on_nav_rejected: bool - diagnostic flag to run IK even for
    #                                         nav-rejected poses.

    required_controllers = ["nav", "arm"]

    def __init__(
        self,
        target: str = '',
        frame_id: str = 'map',
        min_radius: float = 0.45,
        max_radius: float = 1.05,
        radius_step: float = 0.15,
        preferred_radius: float = 0.65,
        angle_step_rad: float = math.radians(30.0),
        refine_top_k: int = 5,
        refine_iterations: int = 1,
        use_ik: bool = True,
        require_ik: bool = False,
        planner: str | None = None,
        preferred_angles: list[float] | None = None,
        allow_goal_update: bool = True,
        object_pose_ttl_s: float | None = None,
        observe_if_stale: bool = True,
        evaluate_ik_on_nav_rejected: bool = False,
        nav_feasibility_missing_policy: str = 'reject',
        debug_visualization: bool = True,
        reach_validate_joint_plan: bool = True,
        reach_position_tolerance_m: float | None = None,
        *,
        reach_clearance_m: float | None = None,
        object_height_guess_m: float | None = None,
    ) -> None:
        self.target = target
        self.frame_id = frame_id
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.radius_step = radius_step
        self.preferred_radius = preferred_radius
        self.angle_step_rad = angle_step_rad
        self.refine_top_k = refine_top_k
        self.refine_iterations = refine_iterations
        self.use_ik = use_ik
        self.require_ik = require_ik
        self.planner = planner
        self.preferred_angles = preferred_angles
        self.allow_goal_update = allow_goal_update
        self.object_pose_ttl_s = object_pose_ttl_s
        self.observe_if_stale = observe_if_stale
        self.evaluate_ik_on_nav_rejected = evaluate_ik_on_nav_rejected
        self.nav_feasibility_missing_policy = nav_feasibility_missing_policy
        self.debug_visualization = debug_visualization
        self.reach_validate_joint_plan = reach_validate_joint_plan
        self.reach_position_tolerance_m = reach_position_tolerance_m
        self.reach_clearance_m = reach_clearance_m
        self.object_height_guess_m = object_height_guess_m
        super().__init__()




    def _select_manipulation_pose(self, context: Context, object_state, *, cancelled: Callable[[], bool] | None = None):
        target_pose = np.asarray(object_state.pose_matrix, dtype=float)
        if target_pose.shape != (4, 4):
            raise ValueError(f"Invalid object pose matrix for '{self.target}': {target_pose.shape}")
        target_pose_candidate_frame, candidate_frame = self._target_pose_in_candidate_frame(context, target_pose, object_state.frame_id)
        log.debug("Manipulation target frame: candidate=%s object=%s", candidate_frame, object_state.frame_id)
        object_position = (
            float(target_pose_candidate_frame[0][3]),
            float(target_pose_candidate_frame[1][3]),
            float(target_pose_candidate_frame[2][3]),
        )
        reach_target = self._resolve_reach_target(context, object_state, object_position, candidate_frame)
        object_xy = (object_position[0], object_position[1])
        selector = ManipulationBasePoseSelector(
            BasePoseSelectorConfig(
                min_radius=float(self.min_radius),
                max_radius=float(self.max_radius),
                radius_step=float(self.radius_step),
                preferred_radius=float(self.preferred_radius),
                angle_step_rad=float(self.angle_step_rad),
                refine_top_k=int(self.refine_top_k),
                refine_iterations=int(self.refine_iterations),
                robot_distance_weight=float(self._manipulation_cfg(context).get("robot_distance_weight", 0.0)),
                robot_yaw_change_weight=float(self._manipulation_cfg(context).get("robot_yaw_change_weight", 0.0)),
                base_drift_weight=float(self._manipulation_cfg(context).get("base_drift_weight", 1.0)),
                base_yaw_drift_weight=float(self._manipulation_cfg(context).get("base_yaw_drift_weight", 0.25)),
                require_ik=bool(self.require_ik),
                evaluate_ik_on_nav_rejected=bool(self.evaluate_ik_on_nav_rejected),
            ),
            nav_feasible_batch=self._nav_feasible_batch(context),
            ik_feasible=self._ik_feasible(
                context,
                reach_target,
                joint_state_seed=self._joint_state_seed(context),
                cancelled=cancelled,
            ) if bool(self.use_ik) else None,
            current_robot_pose=self._current_robot_pose(context, candidate_frame),
        )
        return selector.select(
            object_xy,
            frame_id=candidate_frame,
            preferred_angles=getattr(self, "preferred_angles", None),
        )



    def _get_fresh_object_state(self, context: Context):
        ws = context.world_state
        if ws is None or not hasattr(ws, "get_movable_object_state"):
            return None
        ttl = self._object_pose_ttl(context)
        return ws.get_movable_object_state(self.target, ttl_s=ttl)

    def _object_pose_ttl(self, context: Context) -> float | None:
        if self.object_pose_ttl_s is not None:
            return float(self.object_pose_ttl_s)
        cfg = self._manipulation_cfg(context)
        value = cfg.get("object_pose_ttl_s", 10.0)
        return None if value is None else float(value)

    def _target_pose_in_candidate_frame(
        self,
        context: Context,
        target_pose: np.ndarray,
        target_frame: str,
    ) -> tuple[np.ndarray, str]:
        candidate_frame = str(self.frame_id or target_frame)
        if target_frame == candidate_frame:
            return target_pose, candidate_frame
        sdc = context.sensor_data_center
        if sdc is None:
            raise RuntimeError("missing_sensor_data_center_for_object_frame_tf")
        tf = sdc.get_pose_at(target_frame, candidate_frame)
        if tf is None or tf.get("mat") is None:
            raise RuntimeError("missing_object_to_candidate_frame_tf")
        return np.asarray(tf["mat"], dtype=float) @ target_pose, candidate_frame

    def _resolve_reach_target(
        self,
        context: Context,
        object_state,
        object_position: tuple[float, float, float],
        frame_id: str,
    ) -> ReachTarget:
        cfg = self._manipulation_cfg(context)
        clearance = float(self.reach_clearance_m if self.reach_clearance_m is not None else cfg.get("reach_clearance_m", 0.06))
        height_guess = float(self.object_height_guess_m if self.object_height_guess_m is not None else cfg.get("object_height_guess_m", 0.12))
        if context.world_state is not None and hasattr(context.world_state, "get_movable_object_reach_target"):
            try:
                state_target = context.world_state.get_movable_object_reach_target(
                    self.target,
                    ttl_s=self._object_pose_ttl(context),
                    clearance_m=clearance,
                    height_guess_m=height_guess,
                    frame_id=object_state.frame_id,
                )
                if state_target is not None:
                    if state_target.frame_id == frame_id:
                        return state_target
                    sdc = context.sensor_data_center
                    if sdc is not None:
                        transformed = sdc.transform_data(
                            np.asarray([[state_target.x, state_target.y, state_target.z]], dtype=np.float32),
                            state_target.frame_id,
                            frame_id,
                        )
                        if transformed is not None:
                            point = np.asarray(transformed, dtype=float).reshape(-1, 3)[0]
                            return ReachTarget(
                                x=float(point[0]),
                                y=float(point[1]),
                                z=float(point[2]),
                                frame_id=frame_id,
                                source=state_target.source,
                            )
            except Exception as exc:  # noqa: BLE001
                log.debug("Unable to derive reach target from world state for %s: %s", self.target, exc)
        pcd_key = getattr(object_state, "object_pcd_key", None) or f"{self.target}_object_pcd"
        object_pcd = context.world_state.get(pcd_key) if context.world_state is not None else None
        if object_pcd is not None:
            try:
                points = np.asarray(object_pcd, dtype=float)
                log.debug("Object PCD shape for %s: ndim=%s shape=%s", self.target, points.ndim, points.shape)
                if points.ndim == 2 and points.shape[0] > 0 and points.shape[1] >= 3:
                    top_z = float(np.nanmax(points[:, 2]))
                    if math.isfinite(top_z):
                        return ReachTarget(
                            x=float(object_position[0]),
                            y=float(object_position[1]),
                            z=top_z + clearance,
                            frame_id=frame_id,
                            source="object_top_plus_margin",
                        )
            except Exception as exc:  # noqa: BLE001
                log.debug("Unable to derive reach target from object PCD %s: %s", pcd_key, exc)
        return ReachTarget(
            x=float(object_position[0]),
            y=float(object_position[1]),
            z=float(object_position[2]) + 0.5 * height_guess + clearance,
            frame_id=frame_id,
            source="centroid_plus_height_guess",
        )

    def _nav_feasible_batch(self, context: Context):
        sdc = context.sensor_data_center
        if sdc is None:
            def _missing(candidates):
                return [
                    FeasibilityResult(
                        self.nav_feasibility_missing_policy != "reject",
                        "missing_sensor_data_center",
                    )
                    for _ in candidates
                ]

            return _missing
        cfg = self._manipulation_cfg(context)
        hardware = context.hardware if isinstance(context.hardware, dict) else {}
        checker = NavigationFeasibilityChecker(
            sdc,
            NavFeasibilityConfig(
                local_costmap_sensor=str(cfg.get("local_costmap_sensor", "local_costmap")),
                map_sensor=str(cfg.get("map_sensor", "map")),
                footprint_sensor=str(cfg.get("footprint_sensor", "local_footprint")),
                missing_policy=str(getattr(self, "nav_feasibility_missing_policy", "allow_with_reason")),
                robot_radius=float(cfg.get("robot_radius", hardware.get("robot_radius", 0.3))),
                footprint=cfg.get("footprint"),
                footprint_inflation_m=float(cfg.get("footprint_inflation_m", 0.0)),
                occupied_threshold=int(cfg.get("occupied_threshold", 99)),
                near_goal_radius_m=float(cfg.get("near_goal_radius_m", 0.0)),
                near_goal_occupied_threshold=int(cfg.get("near_goal_occupied_threshold", cfg.get("occupied_threshold", 99))),
                # cost == 99    # footprint/inscribed collision
                # cost == 100   # lethal obstacle
                # cost == -1    # unknown
                unknown_is_blocked=bool(cfg.get("unknown_is_blocked", False)),
                timeout_s=float(cfg.get("costmap_timeout_s", 0.2)),
            ),
        )
        def _check(candidates):
            return [FeasibilityResult.from_legacy(item) for item in checker.check_candidates(candidates)]

        return _check

    def _current_robot_pose(self, context: Context, frame_id: str) -> BasePose2D | None:
        sdc = context.sensor_data_center
        try:
            robot_pose = sdc.get_robot_pose_global(global_frame=frame_id)
        except Exception as exc:  # noqa: BLE001
            log.debug("Unable to snapshot current robot pose for manipulation scoring: %s", exc)
            return None
        if not robot_pose:
            return None
        if isinstance(robot_pose, dict):
            pose = robot_pose.get("pose")
            fallback_x = robot_pose.get("x", 0.0)
            fallback_y = robot_pose.get("y", 0.0)
            fallback_yaw = robot_pose.get("yaw", 0.0)
            pose_frame = str(robot_pose.get("frame_id", frame_id))
        else:
            pose = robot_pose
            fallback_x = fallback_y = fallback_yaw = 0.0
            pose_frame = frame_id
        try:
            return BasePose2D(
                x=float(getattr(pose, "x", fallback_x)),
                y=float(getattr(pose, "y", fallback_y)),
                yaw=float(getattr(pose, "yaw", fallback_yaw)),
                frame_id=pose_frame,
            )
        except Exception as exc:  # noqa: BLE001
            log.debug("Invalid current robot pose snapshot for manipulation scoring: %s", exc)
            return None

    def _joint_state_seed(self, context: Context) -> dict[str, Any] | None:
        sdc = context.sensor_data_center
        if sdc is None:
            return None
        try:
            try:
                msg = sdc.get_latest("joint_states", timeout_s=0.05)
            except TypeError:
                msg = sdc.get_latest("joint_states")
            data = msg.data if msg is not None else None
            names = list(getattr(data, "name", []) or [])
            positions = list(getattr(data, "position", []) or [])
            if not names or not positions:
                return None
            return {"name": names, "position": [float(value) for value in positions[: len(names)]]}
        except Exception as exc:  # noqa: BLE001
            log.debug("Unable to snapshot joint state seed for manipulation IK: %s", exc)
            return None

    def _ik_feasible(self, context: Context, reach_target: ReachTarget, *, joint_state_seed: dict[str, Any] | None, cancelled: Callable[[], bool] | None = None):
        def _check(base_pose):
            if cancelled is not None and cancelled():
                return FeasibilityResult(False, "reachability_cancelled", metadata={"reach_method": "cancelled"})
            planner = context.get_planner(getattr(self, "planner", None))
            if planner is None:
                return FeasibilityResult(
                    False,
                    "planner_unavailable",
                    metadata={"planner": getattr(self, "planner", None) or "default"},
                )
            hardware = context.hardware if isinstance(context.hardware, dict) else {}
            manipulation_cfg = self._manipulation_cfg(context)
            reach_group = (manipulation_cfg.get("reach_group_name") or None)
            if reach_group in {"default", "moveit", "curobo"}:
                reach_group = None
            reach_tolerance = getattr(self, "reach_position_tolerance_m", None)
            if reach_tolerance is None:
                reach_tolerance = manipulation_cfg.get("reach_position_tolerance_m", 0.05)
            result = planner.check_reachability(
                MoveItReachabilityRequest(
                    reach_target=reach_target,
                    candidate_base=base_pose,
                    joint_state_seed=joint_state_seed,
                    group_name=str(reach_group or "whole_body_weighted"),
                    validate_plan=bool(getattr(self, "reach_validate_joint_plan", True)),
                    position_tolerance_m=float(reach_tolerance),
                    metadata={
                        "target": self.target,
                        "nav_base_frame": str(hardware.get("base") or hardware.get("base_frame") or "base_footprint"),
                        "cancelled": cancelled or (lambda: False),
                    },
                ),
                context=context,
            )
            if result.reason == "reachability_not_supported":
                result = PlannerResult(
                    success=False,
                    reason="planner_reachability_not_supported",
                    metadata={
                        "planner": type(planner).__name__,
                        "target": self.target,
                    },
                )
            return FeasibilityResult(
                bool(result.success),
                result.reason,
                score=result.score,
                metadata={
                    "planner_joint_positions": result.joint_positions,
                    "planner_metadata": {
                        **(result.metadata or {}),
                        "reach_target": reach_target.to_dict(),
                    },
                    "reach_method": (result.metadata or {}).get("method"),
                },
            )

        return _check

    def _selection_failure_reason(self, result) -> str:
        if result.accepted:
            return f"No selected manipulation base pose for '{self.target}'"
        reasons = {}
        for item in result.rejected:
            for reason in item.reasons or ["rejected"]:
                reasons[reason] = reasons.get(reason, 0) + 1
        if reasons:
            top = sorted(reasons.items(), key=lambda pair: pair[1], reverse=True)[:3]
            return "No feasible manipulation base pose for '%s': %s" % (
                self.target,
                ", ".join(f"{reason}={count}" for reason, count in top),
            )
        return f"No feasible manipulation base pose for '{self.target}'"

    def _compact_selection_summary(self, result) -> dict[str, Any] | None:
        if result is None:
            return None
        all_items = list(result.accepted) + list(result.rejected)
        reason_counts: dict[str, int] = {}
        nav_feasible = 0
        ik_feasible = 0
        debug_json_paths: list[str] = []
        for item in all_items:
            metadata = item.metadata if isinstance(item.metadata, dict) else {}
            if metadata.get("nav_feasible"):
                nav_feasible += 1
            if metadata.get("ik_feasible"):
                ik_feasible += 1
            for reason in item.reasons or []:
                reason_counts[str(reason)] = reason_counts.get(str(reason), 0) + 1
            ik_metadata = metadata.get("ik_metadata")
            if isinstance(ik_metadata, dict) and ik_metadata.get("debug_json"):
                debug_json_paths.append(str(ik_metadata["debug_json"]))
        top_rejections = sorted(reason_counts.items(), key=lambda pair: pair[1], reverse=True)[:5]
        return {
            "accepted_count": len(result.accepted),
            "rejected_count": len(result.rejected),
            "nav_feasible_candidates": nav_feasible,
            "ik_feasible_candidates": ik_feasible,
            "top_rejection_reasons": [
                {"reason": reason, "count": count}
                for reason, count in top_rejections
            ],
            "debug_json_files": debug_json_paths[:5],
        }

    def _log_selection_summary(self, result) -> None:
        counts: dict[str, int] = {}
        nav_sources: dict[str, int] = {}
        ik_evaluated = 0
        for item in list(result.accepted) + list(result.rejected):
            nav_metadata = item.metadata.get("nav_metadata") if isinstance(item.metadata, dict) else None
            if isinstance(nav_metadata, dict):
                source = nav_metadata.get("source") or nav_metadata.get("metadata", {}).get("source")
                if source:
                    nav_sources[str(source)] = nav_sources.get(str(source), 0) + 1
            if item.metadata.get("ik_reason") != "ik_skipped_nav_rejected":
                ik_evaluated += 1
            for reason in item.reasons or []:
                counts[str(reason)] = counts.get(str(reason), 0) + 1
        top_reasons = sorted(counts.items(), key=lambda pair: pair[1], reverse=True)[:5]
        log.info(
            "MoveToManipulationPose candidate summary for %s: total=%d accepted=%d rejected=%d nav_sources=%s ik_evaluated=%d top_rejections=%s",
            self.target,
            len(result.accepted) + len(result.rejected),
            len(result.accepted),
            len(result.rejected),
            nav_sources,
            ik_evaluated,
            top_reasons,
        )

    def _manipulation_cfg(self, context: Context) -> dict[str, Any]:
        hardware = context.hardware if isinstance(context.hardware, dict) else {}
        cfg = hardware.get("manipulation", {})
        return cfg if isinstance(cfg, dict) else {}

    def _publish_candidate_markers(self, context: Context, result) -> None:
        if not bool(getattr(self, "debug_visualization", True)):
            return
        node = getattr(context, "node", None) or getattr(context.sensor_data_center, "node", None)
        if node is None:
            return
        try:
            from geometry_msgs.msg import Point
            from std_msgs.msg import ColorRGBA
            from visualization_msgs.msg import Marker, MarkerArray
        except Exception as exc:  # noqa: BLE001
            log.debug("Candidate marker imports unavailable: %s", exc)
            return
        topic = str(self._manipulation_cfg(context).get("candidate_marker_topic", "/orchid/manipulation_base_candidates"))
        publisher = getattr(self, "_candidate_marker_publisher", None)
        if publisher is None:
            publisher = node.create_publisher(MarkerArray, topic, 10)
            self._candidate_marker_publisher = publisher
        markers = MarkerArray()
        reset = Marker()
        reset.action = Marker.DELETEALL
        markers.markers.append(reset)
        all_items = list(result.rejected) + list(result.accepted)
        selected_index = None
        if result.selected is not None:
            for idx, item in enumerate(all_items):
                if item is result.selected:
                    selected_index = idx
                    break
        for idx, item in enumerate(all_items):
            marker = Marker()
            marker.header.frame_id = item.pose.frame_id
            marker.header.stamp = node.get_clock().now().to_msg()
            marker.ns = self._candidate_namespace(item, selected=(idx == selected_index))
            marker.id = idx
            marker.type = Marker.ARROW
            marker.action = Marker.ADD
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.025
            marker.scale.y = 0.035
            marker.scale.z = 0.035
            start = Point()
            start.x = float(item.pose.x)
            start.y = float(item.pose.y)
            start.z = 0.05
            end = Point()
            end.x = float(item.pose.x + 0.1 * math.cos(item.pose.yaw))
            end.y = float(item.pose.y + 0.1 * math.sin(item.pose.yaw))
            end.z = 0.05
            marker.points = [start, end]
            marker.color = self._candidate_color(marker.ns, item.score)
            markers.markers.append(marker)
        publisher.publish(markers)

    @staticmethod
    def _candidate_namespace(item, *, selected: bool) -> str:
        if selected:
            return "selected"
        if item.accepted:
            return "accepted"
        reasons = set(item.reasons or [])
        if any("nav" in reason or "costmap" in reason or "map_" in reason for reason in reasons):
            return "nav_rejected"
        if any("ik" in reason or "curobo" in reason or "planner" in reason for reason in reasons):
            return "ik_rejected"
        return "unknown"

    @staticmethod
    def _candidate_color(namespace: str, score: float):
        from std_msgs.msg import ColorRGBA

        colors = {
            "selected": (0.1, 0.8, 1.0, 1.0),
            "accepted": (0.1, 0.85, 0.25, 0.75),
            "nav_rejected": (1.0, 0.25, 0.15, 0.55),
            "ik_rejected": (1.0, 0.75, 0.1, 0.55),
            "unknown": (0.6, 0.6, 0.6, 0.45),
        }
        r, g, b, a = colors.get(namespace, colors["unknown"])
        color = ColorRGBA()
        color.r = float(r)
        color.g = float(g)
        color.b = float(b)
        color.a = float(a)
        return color

    def build_node(self, context, name):
        from src.plugin.actions.observe import ObserveScene
        from src.plugin.actions.manipulation import MoveArmToState
        self._navigation = MoveToPose(allow_goal_update=self.allow_goal_update)
        fresh = BtConditionLeaf(lambda context: self._get_fresh_object_state(context) is not None, context)
        fresh.name = "Fresh object pose"
        prerequisites = [fresh]
        if self.observe_if_stale:
            prerequisites.append(ObserveScene(target=self.target).build_node(context, "Refresh object pose"))
        self._selection_leaf = _SelectManipulationBasePose(self)
        children = [py_trees.composites.Selector(name="Object pose", memory=True, children=prerequisites),
            self._selection_leaf.build_node(context, "Select manipulation base pose"),
            MoveArmToState(state="go").build_node(context, "Stow arm"), self._navigation.build_node(context, "Navigate to manipulation pose")]
        return ActionScope(self, context, name, py_trees.composites.Sequence(name="Manipulation navigation", memory=True, children=children))




    def _accept_selection(self, context, result):
        self._log_selection_summary(result)
        self._publish_candidate_markers(context, result)
        context.world_state.set(f"{self.target}_manipulation_base_selection", result.to_dict())
        if result.selected is None:
            return
        selected = result.selected.pose
        context.world_state.set(f"{self.target}_manipulation_base_pose", selected.to_dict())
        self._navigation.x, self._navigation.y, self._navigation.yaw = selected.x, selected.y, selected.yaw
        self._navigation.frame_id = selected.frame_id
        log.info("Selected manipulation base pose for %s: x=%.3f y=%.3f yaw=%.3f score=%.3f", self.target, selected.x, selected.y, selected.yaw, result.selected.score)

    def on_enter(self, context):
        self._selection_leaf.reset()
        if not self.target:
            return ActionResult(status="failure", reason="MoveToManipulationPose requires target", failure=FailureInfo(FailureCode.INVALID_REQUEST))

    def on_complete(self, context, status):
        if status == py_trees.common.Status.SUCCESS:
            return ActionResult(status="success")
        selection = self._selection_leaf
        return ActionResult(status="needs_replan", reason=selection.failure_reason or "No feasible manipulation base pose", suggested_replan_context={"target": self.target, "action": "MoveToManipulationPose", "selection_summary": self._compact_selection_summary(selection.selection_result), "selection_error": selection.error})


class _SelectManipulationBasePose(AsyncAction):
    """Own asynchronous base-pose computation, completion and cancellation state."""

    def __init__(self, inputs: MoveToManipulationPose):
        super().__init__()
        self.inputs = inputs
        self._generation = 0
        self.selection_result = None
        self.failure_reason = None
        self.error = None
        self._pending = False

    def param_dict(self):
        return {"inputs": self.inputs.param_dict()}

    def start(self, context):
        self._pending = True
        generation = self._generation
        object_state = self.inputs._get_fresh_object_state(context)

        def current():
            return generation == self._generation and not self._cancel_requested and context.execution.control is None and not context.execution.cancel_requested

        def on_error(exc):
            if current():
                self.error = str(exc)
                self.failure_reason = f"selector_exception: {exc}"
                self._pending = False

        def on_selected(result):
            if not current():
                return
            try:
                self.inputs._accept_selection(context, result)
                self.selection_result = result
                if result.selected is None:
                    self.failure_reason = self.inputs._selection_failure_reason(result)
                self._pending = False
            except Exception as exc:
                on_error(exc)

        try:
            context.run_in_context_loop(lambda: self.inputs._select_manipulation_pose(context, object_state, cancelled=lambda: not current()), callback=on_selected, error_callback=on_error)
        except Exception as exc:
            on_error(exc)

    def check_result(self, context):
        if self._cancel_requested:
            return ActionResult(status="failure", reason="Action cancelled", failure=FailureInfo(FailureCode.CANCELLED))
        if self._pending:
            return ActionResult(status="running")
        if self.failure_reason:
            return ActionResult(status="failure", reason=self.failure_reason, failure=FailureInfo(FailureCode.PLANNING_FAILED))
        return ActionResult(status="success")

    def reset(self):
        super().reset()
        self._generation += 1
        self.selection_result = None
        self.failure_reason = None
        self.error = None
        self._pending = False

    def cancel(self, context):
        self._cancel_requested = True
        self._generation += 1
