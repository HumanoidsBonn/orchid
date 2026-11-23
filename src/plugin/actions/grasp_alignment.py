"""Experimental HSRB grasp alignment actions.

These actions are intentionally additive. They provide a simpler perception ->
base/arm alignment path without changing the existing GGX/PickTarget flow.
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

import py_trees
from src.tree.bt_node_wrapper import ActionScope, BtConditionLeaf
from src.core.failure import FailureInfo, FailureCode
from src.core.arm_request import ArmPoseRequest, MoveItPoseSettings
from src.core.moveit_request import MoveItPlanRequest
from dataclasses import replace
from src.core.action import AbstractAction, ActionResult, AsyncAction
from src.core.collision_geometry import validate_hull_inflation
from src.core.context import Context
from src.core.controllers import Pose3D
from src.core.geometry import (
    matrix_to_xyz_rpy,
    planar_pose_matrix,
    pose_dict_to_matrix,
    wrap_angle,
    yaw_delta,
)
from src.core.manipulation_types import BasePose2D
from src.plugin.actions._helpers import ControllerAsyncAction, _arm_status_result
from src.plugin.nav_feasibility import NavFeasibilityConfig, NavigationFeasibilityChecker
import src.core.logger as log


DEFAULT_TMC_IK_MODULE = "tmc_solve_ik_with_collision"
_GRASP_ALIGNMENT_MARKER_PUBLISHERS: dict[tuple[int, str], Any] = {}


@dataclass(frozen=True)
class ServoCommand:
    vx: float
    vy: float
    wz: float
    xy_error: float
    yaw_error: float

    def to_dict(self) -> dict[str, float]:
        return {
            "vx": self.vx,
            "vy": self.vy,
            "wz": self.wz,
            "xy_error": self.xy_error,
            "yaw_error": self.yaw_error,
        }


def _validate_grasp_offset(value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError("grasp outward offset must be finite and nonnegative")
    return value


def _offset_grasp_outward(pose: np.ndarray, metadata: dict[str, Any], distance: float):
    from .perception_preview import capped_grasp_width
    metadata = capped_grasp_width(metadata)
    distance = _validate_grasp_offset(distance)
    original = pose[:3, 3].copy()
    direction = _approach_vector_from_pose(pose, metadata["approach_axis"])
    pose = pose.copy()
    pose[:3, 3] -= distance * direction
    metadata = {**metadata, "outward_offset_m": distance,
                "unoffset_grasp_position": original.tolist(), "grasp_position": pose[:3, 3].tolist()}
    log.info("Simple grasp outward offset_m=%.3f axis=%s original=%s resolved=%s",
             distance, metadata["approach_axis"], original.tolist(), pose[:3, 3].tolist())
    return pose, metadata


def simple_cylindrical_grasp_pose(
    object_points: Any,
    *,
    frame_id: str,
    robot_pose: BasePose2D,
    approach_axis: str = "+z",
    outward_offset_m: float = .05,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Build a simple robot-facing grasp pose from a segmented object cloud."""
    points = _as_points_array(object_points, "object_pcd")
    centroid = np.mean(points[:, :3], axis=0).astype(np.float64)
    mins = np.min(points[:, :3], axis=0)
    maxs = np.max(points[:, :3], axis=0)
    extents = maxs - mins
    yaw = math.atan2(float(centroid[1]) - robot_pose.y, float(centroid[0]) - robot_pose.x)
    rot = _hand_rotation_from_yaw(yaw, approach_axis)
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = rot
    pose[:3, 3] = centroid[:3]
    metadata = {
        "frame_id": frame_id,
        "centroid": centroid.astype(float).tolist(),
        "bounds_min": mins.astype(float).tolist(),
        "bounds_max": maxs.astype(float).tolist(),
        "estimated_radius_m": float(max(extents[0], extents[1]) * 0.5),
        "estimated_height_m": float(extents[2]),
        "yaw_rad": float(yaw),
        "approach_axis": approach_axis,
        "robot_pose": robot_pose.to_dict(),
    }
    return _offset_grasp_outward(pose, metadata, outward_offset_m)


def simple_pca_grasp_pose(
    object_points: Any,
    *,
    frame_id: str,
    robot_pose: BasePose2D,
    grasp_mode: str = "auto",
    approach_axis: str = "+z",
    cylinder_roundness_ratio: float = 1.25,
    cylinder_grasp_offset_m: float = .05,
    short_axis_grasp_offset_m: float = .05,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Build a simple PCA-based grasp pose from a segmented object cloud."""
    cylinder_grasp_offset_m = _validate_grasp_offset(cylinder_grasp_offset_m)
    short_axis_grasp_offset_m = _validate_grasp_offset(short_axis_grasp_offset_m)
    points = _as_points_array(object_points, "object_pcd")
    analysis = _pca_grasp_analysis(points, robot_pose=robot_pose)
    mode = str(grasp_mode or "auto").strip().lower()
    if mode not in {"auto", "cylinder_side", "top_down", "short_axis_side"}:
        raise ValueError(f"unsupported_grasp_mode:{grasp_mode}")
    is_cylinder_like = bool(analysis["horizontal_extent_ratio"] <= float(cylinder_roundness_ratio))
    if mode == "cylinder_side" or (mode in {"auto", "short_axis_side"} and (
            is_cylinder_like or analysis["horizontal_extent_ratio"] > 2.0)):
        pose, metadata = simple_cylindrical_grasp_pose(
            points,
            frame_id=frame_id,
            robot_pose=robot_pose,
            approach_axis=approach_axis,
            outward_offset_m=cylinder_grasp_offset_m,
        )
        metadata = {
            **metadata,
            "estimator": "simple_pca_grasp",
            "selected_strategy": "cylinder_side",
            "grasp_mode": mode,
            "is_cylinder_like": is_cylinder_like,
            "pca": analysis,
        }
        return pose, metadata
    pca_candidates = _pca_grasp_candidates(analysis)
    if mode in {"short_axis_side", "auto"}:
        pose, width, strategy = _short_axis_side_grasp_pose(analysis)
    else:  # Explicit top_down; auto selection depends on horizontal shape only.
        pose, width, strategy = _top_down_pca_grasp_pose(analysis)
    metadata = {
        "estimator": "simple_pca_grasp",
        "frame_id": frame_id,
        "centroid": analysis["centroid"],
        "bounds_min": analysis["bounds_min"],
        "bounds_max": analysis["bounds_max"],
        "estimated_radius_m": float(width) * 0.5,
        "estimated_width_m": float(width),
        "estimated_height_m": float(analysis["height_m"]),
        "yaw_rad": float(_approach_yaw_from_pose(pose, approach_axis)),
        "approach_axis": approach_axis,
        "robot_pose": robot_pose.to_dict(),
        "grasp_mode": mode,
        "selected_strategy": strategy,
        "candidate_strategies": pca_candidates,
        "is_cylinder_like": is_cylinder_like,
        "pca": analysis,
    }
    if strategy == "short_axis_side":
        return _offset_grasp_outward(pose, metadata, short_axis_grasp_offset_m)
    from .perception_preview import capped_grasp_width
    return pose, capped_grasp_width(metadata)


class EstimateSimplePcaGrasp(AbstractAction):
    """Estimate a PCA-based simple pick pose for uneven target objects.

    Use this after observation when a segmented object cloud exists for the
    target. Cylinder-like objects intentionally behave like
    EstimateSimpleCylindricalGrasp.

    Planner params:
    - target: object name from perception, such as "blue_jar".
    - grasp_mode: optional "auto", "cylinder_side", "top_down", or
      "short_axis_side". Auto and short_axis_side use the horizontal extent ratio:
      <=1.25 or >2.0 selects cylinder_side; between these selects short_axis_side,
      opening across the short dimension and approaching along the long axis.
      cylinder_side and top_down explicitly force their grasp family.
    - debug_visualization: optional bool to show or hide RViz debug markers.
    - cylinder_grasp_offset_m / short_axis_grasp_offset_m: outward displacement
      opposite insertion, default 0.02 m each; zero keeps the centroid grasp.
    """

    def __init__(
        self,
        target: str,
        object_pcd_key: str | None = None,
        pcd_frame_key: str | None = None,
        output_pose_key: str | None = None,
        grasp_mode: str = 'short_axis_side', # 'cylinder_side',
        approach_axis: str = '+z',
        frame_id: str | None = None,
        cylinder_roundness_ratio: float = 1.25,
        debug_visualization: bool = True,
        marker_topic: str = '/orchid/grasp_alignment/simple_pca_grasp',
        cylinder_grasp_offset_m: float = .02,
        short_axis_grasp_offset_m: float = .02,
    ) -> None:
        self.target = target
        self.object_pcd_key = object_pcd_key
        self.pcd_frame_key = pcd_frame_key
        self.output_pose_key = output_pose_key
        self.grasp_mode = grasp_mode
        self.approach_axis = approach_axis
        self.frame_id = frame_id
        self.cylinder_roundness_ratio = cylinder_roundness_ratio
        self.debug_visualization = debug_visualization
        self.marker_topic = marker_topic
        self.cylinder_grasp_offset_m = _validate_grasp_offset(cylinder_grasp_offset_m)
        self.short_axis_grasp_offset_m = _validate_grasp_offset(short_axis_grasp_offset_m)
        super().__init__()

    def execute(self, context: Context) -> ActionResult:
        ws = context.world_state
        if ws is None:
            return ActionResult(status="failure", reason="missing_world_state")
        target = str(self.target)
        movable = _movable_object_state(ws, target)
        object_key = str(self.object_pcd_key or getattr(movable, "object_pcd_key", None) or f"{target}_object_pcd")
        frame_key = str(self.pcd_frame_key or f"{target}_pcd_frame")
        output_key = str(self.output_pose_key or f"{target}_estimated_pick_pose_mat")
        points = ws.get(object_key)
        frame_id = str(self.frame_id or getattr(movable, "frame_id", None) or ws.get(frame_key) or "odom")
        if points is None:
            return ActionResult(status="failure", reason=f"missing_object_pcd:{object_key}")
        robot_pose = _current_base_pose(context, frame_id)
        if robot_pose is None:
            return ActionResult(status="failure", reason="missing_current_base_pose")
        try:
            pose_mat, metadata = simple_pca_grasp_pose(
                points,
                frame_id=frame_id,
                robot_pose=robot_pose,
                grasp_mode=str(self.grasp_mode),
                approach_axis=str(self.approach_axis),
                cylinder_roundness_ratio=float(self.cylinder_roundness_ratio),
                cylinder_grasp_offset_m=self.cylinder_grasp_offset_m,
                short_axis_grasp_offset_m=self.short_axis_grasp_offset_m,
            )
        except Exception as exc:  # noqa: BLE001
            return ActionResult(status="failure", reason=f"simple_pca_grasp_failed:{exc}")
        xyz_rpy = matrix_to_xyz_rpy(pose_mat)
        ws.set(output_key, pose_mat)
        ws.set(f"{target}_estimated_pick_pose_frame", frame_id)
        ws.set(f"{target}_estimated_width", float(metadata.get("estimated_width_m", float(metadata.get("estimated_radius_m", 0.0)) * 2.0)))
        ws.set(f"{target}_simple_grasp_metadata", metadata)
        from .perception_preview import publish_grasp_preview
        publish_grasp_preview(context,target,pose_mat,frame_id,self.approach_axis,metadata["estimated_width_m"])
        if bool(self.debug_visualization):
            _publish_pca_grasp_markers(context, str(self.marker_topic), frame_id, pose_mat, target, metadata)
        log.speak("Grasp pose detected")
        return ActionResult(
            status="success",
            reason="simple_pca_grasp_estimated",
            payload={"pose_key": output_key, "pose": xyz_rpy, "metadata": metadata},
        )


def score_tmc_ik_solution(solution: dict[str, Any], current_base: BasePose2D) -> float:
    """Lower score is better for TMC IK candidates."""
    base = BasePose2D.from_mapping(solution["origin_to_base"], default_frame_id=current_base.frame_id)
    joints = dict(solution.get("joint_positions") or {})
    base_xy = math.hypot(base.x - current_base.x, base.y - current_base.y)
    base_yaw = abs(yaw_delta(base.yaw, current_base.yaw))
    wrist_roll = abs(float(joints.get("wrist_roll_joint", 0.0)))
    arm_roll = abs(float(joints.get("arm_roll_joint", 0.0)))
    return float(base_xy + 0.25 * base_yaw + 0.10 * wrist_roll + 0.05 * arm_roll)


def compute_omni_servo_command(
    current: BasePose2D,
    target: BasePose2D,
    *,
    kx: float = 0.8,
    ky: float = 0.8,
    kyaw: float = 1.2,
    max_vx: float = 0.12,
    max_vy: float = 0.12,
    max_wz: float = 0.35,
) -> ServoCommand:
    """Compute a bounded body-frame omni command toward a target pose."""
    dx = target.x - current.x
    dy = target.y - current.y
    c = math.cos(-current.yaw)
    s = math.sin(-current.yaw)
    ex = c * dx - s * dy
    ey = s * dx + c * dy
    eyaw = yaw_delta(target.yaw, current.yaw)
    vx = _clamp(kx * ex, -max_vx, max_vx)
    vy = _clamp(ky * ey, -max_vy, max_vy)
    wz = _clamp(kyaw * eyaw, -max_wz, max_wz)
    return ServoCommand(vx=vx, vy=vy, wz=wz, xy_error=math.hypot(dx, dy), yaw_error=abs(eyaw))


class EstimateSimpleCylindricalGrasp(AbstractAction):
    """Estimate a simple cylindrical pick pose for a target object.

    Use this after an observation action has stored odom-frame object and
    surrounding point clouds in ``WorldState`` under ``<target>_object_pcd`` and
    ``<target>_surround_pcd``.

    Planner params:
    - target: object name from perception, such as "coke_bottle".
    - cylinder_grasp_offset_m: outward along the insertion axis, default 0.05 m.
    """

    # - approach_axis: optional hand-frame approach axis when testing orientation
    #   conventions. Use the default unless the grasp appears rotated in RViz.
    # - debug_visualization: optional bool to show or hide RViz debug markers.

    def __init__(
        self,
        target: str,
        object_pcd_key: str | None = None,
        pcd_frame_key: str | None = None,
        output_pose_key: str | None = None,
        approach_axis: str = '+z',
        frame_id: str | None = None,
        debug_visualization: bool = True,
        marker_topic: str = '/orchid/grasp_alignment/simple_grasp',
        cylinder_grasp_offset_m: float = .05,
    ) -> None:
        self.target = target
        self.object_pcd_key = object_pcd_key
        self.pcd_frame_key = pcd_frame_key
        self.output_pose_key = output_pose_key
        self.approach_axis = approach_axis
        self.frame_id = frame_id
        self.debug_visualization = debug_visualization
        self.marker_topic = marker_topic
        self.cylinder_grasp_offset_m = _validate_grasp_offset(cylinder_grasp_offset_m)
        super().__init__()

    def execute(self, context: Context) -> ActionResult:
        ws = context.world_state
        if ws is None:
            return ActionResult(status="failure", reason="missing_world_state")
        target = str(self.target)
        movable = _movable_object_state(ws, target)
        object_key = str(self.object_pcd_key or getattr(movable, "object_pcd_key", None) or f"{target}_object_pcd")
        frame_key = str(self.pcd_frame_key or f"{target}_pcd_frame")
        output_key = str(self.output_pose_key or f"{target}_estimated_pick_pose_mat")
        points = ws.get(object_key)
        frame_id = str(self.frame_id or getattr(movable, "frame_id", None) or ws.get(frame_key) or "odom")
        if points is None:
            return ActionResult(status="failure", reason=f"missing_object_pcd:{object_key}")
        robot_pose = _current_base_pose(context, frame_id)
        if robot_pose is None:
            return ActionResult(status="failure", reason="missing_current_base_pose")
        try:
            pose_mat, metadata = simple_cylindrical_grasp_pose(
                points,
                frame_id=frame_id,
                robot_pose=robot_pose,
                approach_axis=str(self.approach_axis),
                outward_offset_m=self.cylinder_grasp_offset_m,
            )
        except Exception as exc:  # noqa: BLE001
            return ActionResult(status="failure", reason=f"simple_grasp_failed:{exc}")
        xyz_rpy = matrix_to_xyz_rpy(pose_mat)
        ws.set(output_key, pose_mat)
        ws.set(f"{target}_estimated_pick_pose_frame", frame_id)
        ws.set(f"{target}_estimated_width", metadata["estimated_width_m"])
        ws.set(f"{target}_simple_grasp_metadata", metadata)
        from .perception_preview import publish_grasp_preview
        publish_grasp_preview(context,target,pose_mat,frame_id,self.approach_axis,metadata["estimated_width_m"])
        if bool(self.debug_visualization):
            _publish_simple_grasp_markers(context, str(self.marker_topic), frame_id, pose_mat, target)
        return ActionResult(
            status="success",
            reason="simple_cylindrical_grasp_estimated",
            payload={"pose_key": output_key, "pose": xyz_rpy, "metadata": metadata},
        )


class SolveTmcGraspIk(AbstractAction):
    """Use TMC IK to choose a base pose and arm joint goal for a target.

    Intended sequence:
    EstimateSimpleCylindricalGrasp(target=...) -> SolveTmcGraspIk(target=...)
    -> OmniBaseServoToPose(target=...).

    Planner params:
    - target: object name whose canonical grasp pose was estimated.
    - debug_visualization: optional bool to show or hide RViz debug markers.
    """

    required_modules = [DEFAULT_TMC_IK_MODULE]

    def __init__(
        self,
        target: str,
        grasp_pose_key: str | None = None,
        module_name: str = DEFAULT_TMC_IK_MODULE,
        output_prefix: str | None = None,
        timeout_s: float = 5.0,
        debug_visualization: bool = True,
        marker_topic: str = '/orchid/grasp_alignment/tmc_ik',
    ) -> None:
        self.target = target
        self.grasp_pose_key = grasp_pose_key
        self.module_name = module_name
        self.output_prefix = output_prefix
        self.timeout_s = timeout_s
        self.debug_visualization = debug_visualization
        self.marker_topic = marker_topic
        super().__init__()

    def execute(self, context: Context) -> ActionResult:
        ws = context.world_state
        if ws is None:
            return ActionResult(status="failure", reason="missing_world_state")
        if context.modules is None or not context.modules.has(str(self.module_name)):
            return ActionResult(status="failure", reason=f"tmc_ik_module_unavailable:{self.module_name}")
        target = str(self.target)
        prefix = str(self.output_prefix or target)
        pose_key = str(self.grasp_pose_key or f"{target}_pregrasp_pose")
        pose_mat = ws.get(pose_key)
        if pose_mat is None and not self.grasp_pose_key:
            pose_key = f"{target}_estimated_pick_pose_mat"
            pose_mat = ws.get(pose_key)
        if pose_mat is None:
            return ActionResult(status="failure", reason=f"missing_grasp_pose:{pose_key}")
        movable = _movable_object_state(ws, target)
        frame_id = str(getattr(movable, "frame_id", None) or ws.get(f"{target}_pcd_frame") or "odom")
        current_base = _current_base_pose(context, frame_id)
        if current_base is None:
            return ActionResult(status="failure", reason="missing_current_base_pose")
        request_metadata = _tmc_ik_request_metadata(context, pose_key, frame_id, current_base, str(self.module_name))
        ws.set(f"{prefix}_tmc_ik_request_metadata", request_metadata)
        try:
            request = _build_tmc_ik_request(context, np.asarray(pose_mat, dtype=np.float64), frame_id, current_base)
            response = context.modules.call(
                str(self.module_name),
                {"request": request, "raw_response": True},
                timeout=float(self.timeout_s),
            )
            solutions = _parse_tmc_ik_response(response, default_frame_id=frame_id)
        except Exception as exc:  # noqa: BLE001
            log.exception("TMC IK call failed: %s", exc)
            failure = {"request": request_metadata, "error": str(exc)}
            ws.set(f"{prefix}_tmc_ik_failure", failure)
            return ActionResult(status="failure", reason=f"tmc_ik_call_failed:{exc}", payload=failure)
        if not solutions:
            failure = {"request": request_metadata, "raw_response_summary": _response_summary(response)}
            ws.set(f"{prefix}_tmc_ik_failure", failure)
            return ActionResult(status="failure", reason="tmc_ik_no_solutions", payload=failure)
        ranked = sorted(
            ({**solution, "score": score_tmc_ik_solution(solution, current_base)} for solution in solutions),
            key=lambda item: float(item["score"]),
        )
        selected = ranked[0]
        ws.set(f"{prefix}_tmc_ik_solutions", ranked)
        ws.set(f"{prefix}_tmc_base_pose", selected["origin_to_base"])
        ws.set(f"{prefix}_tmc_joint_goal", selected["joint_positions"])
        ws.set(f"{prefix}_tmc_selected_index", int(selected.get("index", 0)))
        ws.set(f"{prefix}_tmc_ik_metadata", {"pose_key": pose_key, "solution_count": len(ranked), "request": request_metadata})
        if bool(self.debug_visualization):
            _publish_tmc_ik_markers(
                context,
                str(self.marker_topic),
                frame_id,
                np.asarray(pose_mat, dtype=np.float64),
                ranked,
                selected,
                target,
            )
        return ActionResult(status="success", reason="tmc_ik_solution_selected", payload=selected)


class SelectHsrGraspMotion(AbstractAction):
    """Select pregrasp, grasp, and retreat poses for an HSR target.

    Intended sequence:
    EstimateSimpleCylindricalGrasp(target=...) -> SelectHsrGraspMotion(target=...)
    -> SelectFkArmPosture or SolveTmcGraspIk -> OmniBaseServoToPose ->
    ExecuteHsrCartesianGraspMotion.

    Planner params:
    - target: object name whose canonical grasp pose was estimated.
    - pregrasp_mode: optional "simple" or "horizontal_projected" for tilted
      grasp testing. The default preserves the current pose backoff.
    - retreat_mode: optional "reverse" or "up" when testing retreat behavior.
    - debug_visualization: optional bool to show or hide RViz debug markers.
    """

    def __init__(
        self,
        target: str,
        grasp_pose_key: str | None = None,
        output_prefix: str | None = None,
        pregrasp_distance_m: float = 0.12,
        retreat_distance_m: float = 0.12,
        min_insertion_m: float = 0.04,
        max_insertion_m: float = 0.16,
        clearance_m: float = 0.03,
        pregrasp_mode: str = 'horizontal_projected',
        retreat_mode: str = 'reverse',
        debug_visualization: bool = True,
        marker_topic: str = '/orchid/grasp_alignment/grasp_motion',
    ) -> None:
        self.target = target
        self.grasp_pose_key = grasp_pose_key
        self.output_prefix = output_prefix
        self.pregrasp_distance_m = pregrasp_distance_m
        self.retreat_distance_m = retreat_distance_m
        self.min_insertion_m = min_insertion_m
        self.max_insertion_m = max_insertion_m
        self.clearance_m = clearance_m
        self.pregrasp_mode = pregrasp_mode
        self.retreat_mode = retreat_mode
        self.debug_visualization = debug_visualization
        self.marker_topic = marker_topic
        super().__init__()

    def execute(self, context: Context) -> ActionResult:
        ws = context.world_state
        if ws is None:
            return ActionResult(status="failure", reason="missing_world_state")
        target = str(self.target)
        prefix = str(self.output_prefix or target)
        pose_key = str(self.grasp_pose_key or f"{target}_estimated_pick_pose_mat")
        grasp_pose = ws.get(pose_key)
        if grasp_pose is None:
            return ActionResult(status="failure", reason=f"missing_grasp_pose:{pose_key}")
        grasp_pose = np.asarray(grasp_pose, dtype=np.float64)
        movable = _movable_object_state(ws, target)
        frame_id = str(
            getattr(movable, "frame_id", None)
            or ws.get(f"{target}_estimated_pick_pose_frame")
            or ws.get(f"{target}_pcd_frame")
            or "odom"
        )
        grasp_metadata = ws.get(f"{target}_simple_grasp_metadata") or {}
        approach_axis = str(grasp_metadata.get("approach_axis") or "+z")
        pregrasp_mode = str(self.pregrasp_mode or "simple").strip().lower()
        object_points, surround_points = _object_and_surround_points(ws, target, movable)
        approach = _approach_vector_from_pose(grasp_pose, approach_axis)
        pregrasp_direction, grasp_motion_type = _pregrasp_backoff_direction(
            grasp_pose,
            approach_axis=approach_axis,
            pregrasp_mode=pregrasp_mode,
        )
        scene = _estimate_grasp_scene_parameters(
            grasp_pose,
            approach_axis=approach_axis,
            object_points=object_points,
            surround_points=surround_points,
            clearance_m=float(self.clearance_m),
            min_insertion_m=float(self.min_insertion_m),
            max_insertion_m=float(self.max_insertion_m),
            depth_axis=pregrasp_direction if pregrasp_mode == "horizontal_projected" else None,
        )
        insertion = float(scene["suggested_insertion_m"])
        pregrasp_distance = max(float(self.pregrasp_distance_m), insertion)
        retreat_distance = float(self.retreat_distance_m)
        pregrasp_pose = grasp_pose.copy()
        pregrasp_pose[:3, 3] = grasp_pose[:3, 3] - pregrasp_direction * pregrasp_distance
        if bool(scene.get("is_vertical_grasp", False)):
            scene["required_lift_m"] = float(pregrasp_pose[2, 3])
            scene["lift_coupling_mode"] = "vertical_pregrasp_z"
        retreat_pose = grasp_pose.copy()
        if str(self.retreat_mode).strip().lower() == "up":
            retreat_pose[:3, 3] = grasp_pose[:3, 3] + np.array([0.0, 0.0, retreat_distance], dtype=np.float64)
        else:
            retreat_pose[:3, 3] = grasp_pose[:3, 3] - pregrasp_direction * retreat_distance
        approach_xy = approach[:2].astype(np.float64)
        approach_xy_norm = float(np.linalg.norm(approach_xy))
        if approach_xy_norm > 1e-9:
            approach_xy = approach_xy / approach_xy_norm
        metadata = {
            "pose_key": pose_key,
            "frame_id": frame_id,
            "approach_axis": approach_axis,
            "approach_vector": approach.astype(float).tolist(),
            "approach_vector_3d": approach.astype(float).tolist(),
            "approach_vector_xy": approach_xy.astype(float).tolist(),
            "depth_axis": pregrasp_direction.astype(float).tolist(),
            "pregrasp_mode": pregrasp_mode,
            "grasp_motion_type": grasp_motion_type,
            "pregrasp_distance_m": float(pregrasp_distance),
            "retreat_distance_m": float(retreat_distance),
            "retreat_mode": str(self.retreat_mode),
            "scene": scene,
        }
        ws.set(f"{prefix}_grasp_pose", grasp_pose)
        ws.set(f"{prefix}_pregrasp_pose", pregrasp_pose)
        ws.set(f"{prefix}_retreat_pose", retreat_pose)
        ws.set(f"{prefix}_grasp_motion_metadata", metadata)
        if bool(self.debug_visualization):
            _publish_grasp_motion_markers(
                context,
                str(self.marker_topic),
                frame_id,
                target,
                grasp_pose,
                pregrasp_pose,
                retreat_pose,
                scene,
            )
        return ActionResult(
            status="success",
            reason="hsr_grasp_motion_selected",
            payload={
                "grasp_pose_key": f"{prefix}_grasp_pose",
                "pregrasp_pose_key": f"{prefix}_pregrasp_pose",
                "retreat_pose_key": f"{prefix}_retreat_pose",
                "metadata": metadata,
            },
        )


class MoveToSelectedGraspBasePose(AbstractAction):
    """Move the base to the selected FK or TMC pregrasp base pose.

    Intended sequence:
    SelectFkArmPosture or SolveTmcGraspIk -> MoveToSelectedGraspBasePose ->
    SelectFkArmPosture(execute=True) or final approach.

    Planner params:
    - target: object name whose base pose was selected.
    - source: optional "fk", "tmc", or "auto" to choose the stored base pose.
    """

    def __init__(self, target: str, source: str='auto', moveit_group: str='whole_body_weighted') -> None:
        self.target = target
        self.source = source
        self.moveit_group = moveit_group
        super().__init__()
        self._payload: dict[str, Any] = {}




    def build_node(self, context, name):
        from src.plugin.actions.navigation import MoveBaseWithWholeBody
        self._base_move = MoveBaseWithWholeBody(moveit_group=self.moveit_group)
        return ActionScope(self, context, name, self._base_move.build_node(context, "Move to selected base pose"))

    def on_enter(self, context):
        base_pose, key = _selected_grasp_base_pose(context.world_state, str(self.target), str(self.source))
        self._base_move.x, self._base_move.y, self._base_move.yaw = base_pose.x, base_pose.y, base_pose.yaw
        self._base_move.frame_id = base_pose.frame_id
        self._payload = {"base_pose": base_pose.to_dict(), "base_pose_key": key, "source": str(self.source)}

    def on_complete(self, context, status):
        return ActionResult(status="success" if status == py_trees.common.Status.SUCCESS else "failure", payload=self._payload)


class SelectFkArmPosture(AsyncAction):
    """Select and optionally execute an FK pregrasp arm posture.

    Intended sequence:
    SelectHsrGraspMotion(target=...) -> SelectFkArmPosture(target=...) ->
    OmniBaseServoToPose(target=...).

    Planner params:
    - target: object name whose grasp motion was selected.
    - sampling_mode: "simple" or "multires_scene" (default). The latter
      adds scene-guided samples and refinement to the coarse FK grid.
    - execute: optional bool. True sends the selected joint goal to the arm;
      False only estimates and stores the FK result for debugging.
    - check_state_validity: whether to call the validity service (default True).
      Set both check_state_validity and enforce_state_validity False to skip it.
    - enforce_state_validity: reject invalid/unavailable checks (default True).
    - reach_preference_weight / lift_preference_weight: nonnegative excess
      preference coefficients, default 10. Geometry limits are hard gates.
    - validity_base_frame: frame of MoveIt odometry joints, default "odom".
    - debug_visualization: optional bool to show or hide RViz debug markers.
    """

    input_attributes = {"execute": "execute_arm"}

    def __init__(
        self,
        target: str,
        grasp_pose_key: str | None = None,
        output_prefix: str | None = None,
        arm_lift_min: float = 0.0,
        arm_lift_max: float = 0.55,
        arm_lift_samples: int = 8,
        arm_flex_min: float = -1.57,
        arm_flex_max: float = -0.2,
        arm_flex_samples: int = 9,
        arm_roll: float = 0.0,
        wrist_roll: float = 0.0,
        wrist_flex_offset: float = -1.57,
        desired_hand_x: float = 0.55,
        desired_hand_y: float = 0.0,
        arm_flex_compact_weight: float = 0.35,
        approach_alignment_weight: float = 1.25,
        reach_preference_weight: float = 10.0,
        lift_preference_weight: float = 10.0,
        sampling_mode: str = 'simple', # 'multires_scene', 'simple',
        fk_max_z_error_m: float = 0.03,
        fk_max_lateral_error_m: float = 0.06,
        fk_max_approach_error_rad: float = 0.25,
        fk_approach_standoff_m: float = 0.12,
        fk_min_reach_m: float = 0.25,
        fk_max_reach_m: float = 0.85,
        state_validity_group: str = 'whole_body_weighted',
        surround_pcd_mode: str = 'off', # 'soft',
        enforce_state_validity: bool = False,
        check_state_validity: bool = False,
        validity_base_frame: str = "odom",
        base_frame: str = 'base_footprint',
        world_frame: str | None = None,
        execute: bool = True,
        debug_visualization: bool = True,
        marker_topic: str = '/orchid/grasp_alignment/fk_postures',
        max_marker_candidates: int = 80,
    ) -> None:
        self.target = target
        self.grasp_pose_key = grasp_pose_key
        self.output_prefix = output_prefix
        self.arm_lift_min = arm_lift_min
        self.arm_lift_max = arm_lift_max
        self.arm_lift_samples = arm_lift_samples
        self.arm_flex_min = arm_flex_min
        self.arm_flex_max = arm_flex_max
        self.arm_flex_samples = arm_flex_samples
        self.arm_roll = arm_roll
        self.wrist_roll = wrist_roll
        self.wrist_flex_offset = wrist_flex_offset
        self.desired_hand_x = desired_hand_x
        self.desired_hand_y = desired_hand_y
        self.arm_flex_compact_weight = arm_flex_compact_weight
        self.approach_alignment_weight = approach_alignment_weight
        if not all(math.isfinite(value) and value >= 0 for value in (reach_preference_weight, lift_preference_weight)):
            raise ValueError("FK preference weights must be finite and nonnegative")
        self.reach_preference_weight = reach_preference_weight
        self.lift_preference_weight = lift_preference_weight
        self.sampling_mode = sampling_mode
        self.fk_max_z_error_m = fk_max_z_error_m
        self.fk_max_lateral_error_m = fk_max_lateral_error_m
        self.fk_max_approach_error_rad = fk_max_approach_error_rad
        self.fk_approach_standoff_m = fk_approach_standoff_m
        self.fk_min_reach_m = fk_min_reach_m
        self.fk_max_reach_m = fk_max_reach_m
        self.state_validity_group = state_validity_group
        self.surround_pcd_mode = surround_pcd_mode
        if enforce_state_validity and not check_state_validity:
            raise ValueError("enforce_state_validity requires check_state_validity")
        self.enforce_state_validity = enforce_state_validity
        self.check_state_validity = check_state_validity
        self.validity_base_frame = validity_base_frame
        self.base_frame = base_frame
        self.world_frame = world_frame
        self.debug_visualization = debug_visualization
        self.marker_topic = marker_topic
        self.max_marker_candidates = max_marker_candidates
        super().__init__()
        self.execute_arm = bool(execute)
        self._status = 'idle'
        self._reason: str | None = None
        self._payload: dict[str, Any] = {}
        self._executing = False
        self._selection_lock = threading.RLock()

    def start(self, context: Context) -> None:
        with self._selection_lock:
            self._status = "running"
            self._reason = None
            self._payload = {}
            self._executing = False
            self._generation = getattr(self, "_generation", 0) + 1
            generation = self._generation
            self._cancel_requested = False

        def stopped():
            return (generation != self._generation or self._cancel_requested
                    or context.execution.cancel_requested or context.execution.control is not None)

        def complete(best):
            with self._selection_lock:
                if stopped():
                    return
                log.error("SelectFkArmPosture complete: %s", best)
                publication = best.pop("_publication")
                publication_started = time.perf_counter()
                for key, value in publication.items():
                    context.world_state.set(key, value)
                publication[f"{self.output_prefix or self.target}_fk_posture_metadata"]["timings"]["publication_s"] = time.perf_counter() - publication_started
                self._payload = best
                if self.debug_visualization:
                    _publish_fk_posture_markers(
                        context, self.marker_topic, best["base_pose"]["frame_id"],
                        self.base_frame, publication[f"{self.output_prefix or self.target}_fk_posture_candidates"],
                        best, self.target, max_candidates=self.max_marker_candidates)
                if stopped():
                    return
                if not self.execute_arm:
                    self._status, self._reason = "success", "fk_posture_selected"
                else:
                    context.require_arm().send_joint_goal(best["joint_positions"])
                    self._executing = True

        def failed(exc):
            with self._selection_lock:
                if stopped():
                    return
                self._status, self._reason = "failure", str(exc)
                self._payload = getattr(exc, "diagnostics", {})

        inputs = type(self)(**self.param_dict())
        context.run_in_context_loop(inputs._select, context, cancelled=stopped,
                                    callback=complete, error_callback=failed)

    def check_result(self, context: Context) -> ActionResult:
        if self._status == "success":
            return ActionResult(status="success", reason=self._reason, payload=self._payload)
        if self._status == "failure":
            return ActionResult(status="failure", reason=self._reason or "fk_posture_failed", payload=self._payload,
                                failure=FailureInfo(FailureCode.PLANNING_FAILED, retryable=False))
        if not self._executing:
            return ActionResult(status="running", payload=self._payload)
        result = _arm_status_result(context.require_arm())
        if result.status == "success":
            self._status = "success"
            self._reason = "fk_posture_executed"
            return ActionResult(status="success", reason=self._reason, payload=self._payload)
        if result.status == "running":
            return ActionResult(status="running", payload=self._payload)
        self._status = "failure"
        self._reason = result.reason or "fk_posture_arm_execution_failed"
        return ActionResult(status="failure", reason=self._reason, payload=self._payload, failure=result.failure)

    def reset(self) -> None:
        with self._selection_lock:
            self._generation = getattr(self, "_generation", 0) + 1
            super().reset()
            self._status, self._reason = "idle", None
            self._payload = {}
            self._executing = False

    def cancel(self, context: Context) -> None:
        with self._selection_lock:
            self._cancel_requested = True
            self._generation = getattr(self, "_generation", 0) + 1
            if self._executing:
                context.require_arm().cancel_goal()
                self._executing = False
            self._status, self._reason = "failure", "fk_posture_cancelled"

    def _cached_joint_goal(self, context: Context) -> dict[str, float] | None:
        ws = context.world_state
        if ws is None:
            return None
        prefix = str(self.output_prefix or self.target)
        cached = ws.get(f"{prefix}_fk_joint_goal")
        if not isinstance(cached, dict) or not cached:
            return None
        try:
            return {str(name): float(value) for name, value in cached.items()}
        except Exception:  # noqa: BLE001
            return None

    def _select(self, context: Context, *, cancelled=lambda: False) -> dict[str, Any]:
        started_at = time.perf_counter()
        timings = {"fk_s": 0.0, "validity_s": 0.0, "geometry_scoring_s": 0.0}
        counts = {"fk_calls": 0, "validity_calls": 0, "state_snapshots": 1}
        snapshot_started = time.perf_counter()
        current = _current_joint_state_msg(context)
        seed = dict(zip(getattr(current, "name", []) or [], getattr(current, "position", []) or []))
        timings["state_snapshot_s"] = time.perf_counter() - snapshot_started
        def state(joints, base_pose=None):
            merged = {**seed, **joints}
            if base_pose is not None:
                merged.update(odom_x=base_pose.x, odom_y=base_pose.y, odom_t=base_pose.yaw)
            return _joint_state_msg(merged)
        def compute_fk(joints):
            if cancelled():
                raise RuntimeError("fk_posture_cancelled")
            begin = time.perf_counter()
            counts["fk_calls"] += 1
            result = planner.compute_fk(state(joints), context=context, link_names=["hand_palm_link"], frame_id=self.base_frame)
            timings["fk_s"] += time.perf_counter() - begin
            return result
        ws = context.world_state
        if hasattr(context, "get_planner"):
            planner = context.get_planner("manipulation") or context.get_planner("default") or context.get_planner()
        else:
            planner = context.planners.get("manipulation") or context.planners.get("default")
        if ws is None:
            raise RuntimeError("missing_world_state")
        if planner is None:
            raise RuntimeError("missing_arm_planner")
        target = str(self.target)
        prefix = str(self.output_prefix or target)
        pose_key = str(self.grasp_pose_key or f"{target}_pregrasp_pose")
        pregrasp_pose = ws.get(pose_key)
        if pregrasp_pose is None and not self.grasp_pose_key:
            pose_key = f"{target}_estimated_pick_pose_mat"
            pregrasp_pose = ws.get(pose_key)
        if pregrasp_pose is None:
            raise RuntimeError(f"missing_pregrasp_pose:{pose_key}")
        grasp_pose = ws.get(f"{prefix}_grasp_pose")
        if grasp_pose is None:
            grasp_pose = ws.get(f"{target}_estimated_pick_pose_mat")
        if grasp_pose is None:
            grasp_pose = pregrasp_pose
        motion_metadata = ws.get(f"{target}_grasp_motion_metadata") or {}
        grasp_metadata = ws.get(f"{target}_simple_grasp_metadata") or {}
        approach_axis = str(motion_metadata.get("approach_axis") or grasp_metadata.get("approach_axis") or "+z")
        movable = _movable_object_state(ws, target)
        frame_id = str(self.world_frame or getattr(movable, "frame_id", None) or ws.get(f"{target}_pcd_frame") or "odom")
        pregrasp_pose = np.array(pregrasp_pose, dtype=np.float64, copy=True)
        grasp_pose = np.array(grasp_pose, dtype=np.float64, copy=True)
        object_points, surround_points = _object_and_surround_points(ws, target, movable)
        surround_pcd_mode = str(getattr(self, "surround_pcd_mode", "off") or "off").strip().lower()
        scene = dict(motion_metadata.get("scene") or {})
        if not scene:
            scene = _estimate_grasp_scene_parameters(
                pregrasp_pose,
                approach_axis=approach_axis,
                object_points=object_points,
                surround_points=surround_points,
            )
        grasp_motion_type = str(motion_metadata.get("grasp_motion_type") or "").strip().lower()
        is_vertical_top_down = grasp_motion_type == "vertical_z"
        if not grasp_motion_type and _is_vertical_approach(pregrasp_pose, approach_axis):
            is_vertical_top_down = True
            grasp_motion_type = "vertical_z_fallback"
        world_from_base = _fk_frame_matrix(context, self.base_frame, frame_id)
        validity_from_world = _fk_frame_matrix(context, frame_id, self.validity_base_frame)
        base_height = float(world_from_base[2, 3])
        current_base = _current_base_pose(context, frame_id) if is_vertical_top_down else None
        signature = {
            "pregrasp": pregrasp_pose.tolist(), "grasp": grasp_pose.tolist(),
            "base_height": base_height, "scene": scene, "approach_axis": approach_axis,
            "inputs": {key: value for key, value in self.param_dict().items()
                       if key not in {"execute", "debug_visualization", "marker_topic", "max_marker_candidates"}},
        }
        cached_metadata = ws.get(f"{prefix}_fk_posture_metadata") or {}
        cached = self._cached_joint_goal(context) if self.execute_arm and cached_metadata.get("input_signature") == signature else None
        generation_started = time.perf_counter()
        candidates = [] if cached is not None else self._candidate_joint_sets(
            pregrasp_pose=pregrasp_pose,
            approach_axis=approach_axis,
            scene=scene,
            sampling_mode=self.sampling_mode,
            is_vertical_top_down=is_vertical_top_down,
            planner=planner,
            context=context,
            compute_fk=compute_fk,
            base_height=base_height,
        )
        timings["candidate_generation_s"] = time.perf_counter() - generation_started
        best: dict[str, Any] | None = None
        valid_candidates: list[dict[str, Any]] = []
        rejected_candidates: list[dict[str, Any]] = []
        seen_candidate_keys: set[tuple[float, float, float]] = set()

        def candidate_key(joints: dict[str, float]) -> tuple[float, float, float]:
            return (
                round(float(joints.get("arm_lift_joint", 0.0)), 5),
                round(float(joints.get("arm_flex_joint", 0.0)), 5),
                round(float(joints.get("wrist_roll_joint", 0.0)), 5),
            )

        def evaluate_candidate(joints: dict[str, float]) -> None:
            nonlocal best
            key = candidate_key(joints)
            if key in seen_candidate_keys:
                return
            seen_candidate_keys.add(key)
            log.debug("Testing candidate joints: %s", joints)
            fk = compute_fk(joints)
            geometry_started = time.perf_counter()
            if not fk.success:
                log.warning("FK failed for candidate joints: %s", joints)
                return
            hand_pose = _fk_pose_matrix(fk)
            if hand_pose is None or hand_pose.shape != (4, 4) or not np.isfinite(hand_pose).all():
                log.warning("Invalid hand pose for candidate joints: %s", joints)
                return
            if is_vertical_top_down:
                base_pose = _base_pose_from_vertical_fk_pregrasp_alignment(
                    pregrasp_pose,
                    hand_pose,
                    frame_id=frame_id,
                    current_base=current_base,
                )
            else:
                base_pose = _base_pose_from_fk_pregrasp_alignment(
                    pregrasp_pose,
                    hand_pose,
                    frame_id=frame_id,
                    approach_axis=approach_axis,
                    approach_standoff_m=float(self.fk_approach_standoff_m),
                )
            if is_vertical_top_down:
                final_grasp_base_pose = _base_pose_from_vertical_fk_pregrasp_alignment(
                    grasp_pose,
                    hand_pose,
                    frame_id=frame_id,
                    current_base=current_base,
                )
            else:
                final_grasp_base_pose = _base_pose_from_hand_alignment(
                    grasp_pose,
                    hand_pose,
                    frame_id=frame_id,
                    approach_axis=approach_axis,
                )
            resolved_hand_pose = _resolve_hand_pose_in_world(base_pose, hand_pose, base_height=base_height)
            resolved_grasp_hand_pose = _resolve_hand_pose_in_world(final_grasp_base_pose, hand_pose, base_height=base_height)
            errors = _fk_candidate_errors(
                resolved_hand_pose,
                pregrasp_pose,
                hand_pose_base=hand_pose,
                approach_axis=approach_axis,
            )
            gate = _fk_candidate_gate(
                errors,
                max_z_error_m=float(self.fk_max_z_error_m),
                max_lateral_error_m=float(self.fk_max_lateral_error_m),
                max_approach_error_rad=float(self.fk_max_approach_error_rad),
                min_reach_m=float(scene.get("min_reach_m", self.fk_min_reach_m) if isinstance(scene, dict) else self.fk_min_reach_m),
                max_reach_m=float(scene.get("max_reach_m", self.fk_max_reach_m) if isinstance(scene, dict) else self.fk_max_reach_m),
            )
            common = {
                "joint_positions": joints, "base_pose": base_pose.to_dict(),
                "final_grasp_base_pose": final_grasp_base_pose.to_dict(),
                "hand_pose_base": hand_pose.astype(float).tolist(),
                "resolved_hand_pose_world": resolved_hand_pose.astype(float).tolist(),
                "resolved_grasp_hand_pose_world": resolved_grasp_hand_pose.astype(float).tolist(),
                "errors": errors, "gate": gate,
            }
            timings["geometry_scoring_s"] += time.perf_counter() - geometry_started
            # if not gate["allowed"]:
            #     log.error(gate)
            #     rejected_candidates.append({**common, "validity": {"checked": False}})
            #     return
            from src.core.planners import PlannerResult
            validity = PlannerResult(success=True, metadata={"checked": False})
            if self.check_state_validity:
                if cancelled():
                    raise RuntimeError("fk_posture_cancelled")
                valid_mat = validity_from_world @ planar_pose_matrix(final_grasp_base_pose.x, final_grasp_base_pose.y, final_grasp_base_pose.yaw)
                validity_base = BasePose2D(x=float(valid_mat[0, 3]), y=float(valid_mat[1, 3]),
                    yaw=math.atan2(valid_mat[1, 0], valid_mat[0, 0]), frame_id=self.validity_base_frame)
                begin = time.perf_counter()
                counts["validity_calls"] += 1
                validity = _planner_state_validity(planner, state(joints, validity_base),
                    context=context, group_name=self.state_validity_group)
                timings["validity_s"] += time.perf_counter() - begin
                validity.metadata = {**validity.metadata, "checked": True, "reason": validity.reason}
                if self.enforce_state_validity and not validity.success:
                    rejected_candidates.append({**common, "validity": validity.metadata,
                        "gate": {**gate, "allowed": False, "reasons": ["state_validity"]}})
                    return
            geometry_started = time.perf_counter()
            if surround_pcd_mode == "off":
                sweep = {"allowed": True, "reason": "surround_pcd_sweep_off", "clearance_penalty": 0.0, "mode": "off"}
            else:
                sweep = _fk_sweep_clearance(
                    base_pose,
                    hand_pose,
                    surround_points=surround_points,
                    approach_axis=approach_axis,
                )
            if surround_pcd_mode == "diagnostic":
                sweep = {**sweep, "clearance_penalty": 0.0, "mode": surround_pcd_mode}
            elif surround_pcd_mode == "hard_gate" and not sweep["allowed"]:
                rejected_candidates.append(
                    {
                        "joint_positions": joints,
                        "base_pose": base_pose.to_dict(),
                        "final_grasp_base_pose": final_grasp_base_pose.to_dict(),
                        "hand_pose_base": hand_pose.astype(float).tolist(),
                        "resolved_hand_pose_world": resolved_hand_pose.astype(float).tolist(),
                        "resolved_grasp_hand_pose_world": resolved_grasp_hand_pose.astype(float).tolist(),
                        "validity": validity.metadata,
                        "errors": errors,
                        "gate": {
                            "allowed": False,
                            "reasons": ["surround_pcd_sweep"],
                            "error_norm": float("inf"),
                        },
                        "sweep": {**sweep, "mode": surround_pcd_mode},
                    }
                )
                return
            else:
                sweep = {**sweep, "mode": surround_pcd_mode}
            preferred_lift = (float(joints["arm_lift_joint"]) if is_vertical_top_down else
                _min_lift_for_reach(errors["reach_m"], scene, arm_lift_min=self.arm_lift_min, arm_lift_max=self.arm_lift_max))
            score_terms = self._score_terms(
                resolved_hand_pose,
                pregrasp_pose,
                joints=joints,
                approach_axis=approach_axis,
                scene=scene,
                sweep=sweep,
                hand_pose_base=hand_pose,
                errors=errors,
                arm_flex_compact_weight=float(self.arm_flex_compact_weight),
                approach_alignment_weight=float(self.approach_alignment_weight),
                reach_preference_weight=self.reach_preference_weight,
                lift_preference_weight=self.lift_preference_weight,
                preferred_lift=preferred_lift,
                arm_lift_min=self.arm_lift_min, arm_lift_max=self.arm_lift_max,
            )
            score = sum(score_terms.values())
            timings["geometry_scoring_s"] += time.perf_counter() - geometry_started
            item = {
                "joint_positions": joints,
                "score": score,
                "score_terms": score_terms,
                "base_pose": base_pose.to_dict(),
                "final_grasp_base_pose": final_grasp_base_pose.to_dict(),
                "hand_pose_base": hand_pose.astype(float).tolist(),
                "resolved_hand_pose_world": resolved_hand_pose.astype(float).tolist(),
                "resolved_grasp_hand_pose_world": resolved_grasp_hand_pose.astype(float).tolist(),
                "validity": validity.metadata,
                "sweep": sweep,
                "errors": errors,
                "gate": gate,
            }
            valid_candidates.append(item)
            if best is None or float(item["score"]) < float(best["score"]):
                best = item

        if cached is not None:
            evaluate_candidate(cached)
        cache_reused = best is not None
        if not cache_reused:
            if cached is not None:
                candidates = self._candidate_joint_sets(pregrasp_pose=pregrasp_pose,
                    approach_axis=approach_axis, scene=scene, sampling_mode=self.sampling_mode,
                    is_vertical_top_down=is_vertical_top_down, planner=planner, context=context,
                    compute_fk=compute_fk, base_height=base_height)
            for joints in candidates:
                evaluate_candidate(joints)

        if not cache_reused and not is_vertical_top_down and str(self.sampling_mode or "simple").strip().lower() == "multires_scene":
            anchors = sorted(
                valid_candidates + rejected_candidates,
                key=lambda item: float((item.get("gate") or {}).get("error_norm", 1e9)),
            )[:5]
            refinement_started = time.perf_counter()
            refined = self._refined_joint_sets_from_candidates(anchors, pregrasp_pose=pregrasp_pose, approach_axis=approach_axis)
            for joints in refined:
                evaluate_candidate(joints)
            timings["refinement_s"] = time.perf_counter() - refinement_started

        if best is None:
            exc = RuntimeError("fk_posture_no_valid_candidate")
            exc.diagnostics = {"rejected_candidates": rejected_candidates,
                               "counts": counts, "timings": timings}
            raise exc
        if cancelled():
            raise RuntimeError("fk_posture_cancelled")
        publication = {}

        publication[f"{prefix}_fk_posture_candidates"] = valid_candidates
        publication[f"{prefix}_fk_joint_goal"] = best["joint_positions"]
        publication[f"{prefix}_fk_base_pose"] = best["base_pose"]
        publication[f"{prefix}_fk_posture_metadata"] = {
            "pose_key": pose_key,
            "candidate_count": len(candidates),
            "evaluated_candidate_count": len(seen_candidate_keys),
            "valid_candidate_count": len(valid_candidates),
            "rejected_candidate_count": len(rejected_candidates),
            "selected": dict(best),
            "input_signature": signature,
            "cache_reused": cache_reused,
            "execute": self.execute_arm,
            "approach_axis": approach_axis,
            "grasp_motion_type": grasp_motion_type,
            "vertical_top_down_fk": bool(is_vertical_top_down),
            "base_yaw_mode": "preserve_current" if is_vertical_top_down else "approach_alignment",
            "fixed_arm_flex_joint": float(self.arm_flex_min) if is_vertical_top_down else None,
            "solved_arm_lift_joint": float(best["joint_positions"].get("arm_lift_joint", 0.0)) if is_vertical_top_down else None,
            "lift_clamped": (
                bool(
                    abs(float(best["joint_positions"].get("arm_lift_joint", 0.0)) - float(self.arm_lift_min)) < 1e-6
                    or abs(float(best["joint_positions"].get("arm_lift_joint", 0.0)) - float(self.arm_lift_max)) < 1e-6
                )
                if is_vertical_top_down
                else None
            ),
            "wrist_roll_sample_count": len({round(float(item["joint_positions"].get("wrist_roll_joint", 0.0)), 5) for item in valid_candidates}) if is_vertical_top_down else None,
            "sampling_mode": str(self.sampling_mode or "simple"),
            "surround_pcd_mode": surround_pcd_mode,
            "scene": scene,
            "state_validity_checked": bool(self.check_state_validity),
            "rejected_candidates": rejected_candidates,
            "counts": counts,
            "timings": {**timings, "total_s": time.perf_counter() - started_at},
            "base_height_m": base_height,
            "pregrasp_to_grasp_z_m": float(pregrasp_pose[2, 3] - grasp_pose[2, 3]),
            "state_validity_enforced": bool(self.enforce_state_validity),
            "state_validity_group": str(self.state_validity_group or "whole_body_weighted"),
        }
        return {**best, "_publication": publication}

    def _candidate_joint_sets(
        self,
        *,
        pregrasp_pose: np.ndarray | None = None,
        approach_axis: str = "+z",
        scene: dict[str, Any] | None = None,
        sampling_mode: str = "simple",
        is_vertical_top_down: bool = False,
        planner: Any | None = None,
        context: Context | None = None,
        compute_fk=None,
        base_height: float = 0.0,
    ) -> list[dict[str, float]]:
        wrist_bias = _wrist_flex_bias_for_approach(pregrasp_pose, approach_axis) if pregrasp_pose is not None else 0.0
        if bool(is_vertical_top_down):
            return self._top_down_candidate_joint_sets(
                pregrasp_pose=pregrasp_pose,
                wrist_bias=wrist_bias,
                planner=planner,
                context=context, compute_fk=compute_fk, base_height=base_height,
            )
        lifts = np.linspace(float(self.arm_lift_min), float(self.arm_lift_max), max(1, int(self.arm_lift_samples)))
        flexes = np.linspace(float(self.arm_flex_min), float(self.arm_flex_max), max(1, int(self.arm_flex_samples)))
        pairs: list[tuple[float, float]] = [(float(lift), float(flex)) for lift in lifts for flex in flexes]
        if str(sampling_mode).strip().lower() == "multires_scene":
            scene_data = scene or {}
            insertion = float((scene or {}).get("suggested_insertion_m", 0.08))
            depth = float(
                scene_data.get(
                    "scene_depth_for_reach_m",
                    scene_data.get("support_depth_m", scene_data.get("object_depth_m", 0.0)),
                )
                or 0.0
            )
            seed_reach = 0.5 * (
                float(scene_data.get("required_reach_min_m", scene_data.get("min_reach_m", 0.35)))
                + float(scene_data.get("required_reach_max_m", scene_data.get("max_reach_m", 0.75)))
            )
            preferred_lift = _clamp(
                _min_lift_for_reach(seed_reach, scene_data, arm_lift_min=float(self.arm_lift_min), arm_lift_max=float(self.arm_lift_max)),
                float(self.arm_lift_min),
                float(self.arm_lift_max),
            )
            flex_bias = 0.25 if bool(scene_data.get("is_vertical_grasp", False)) else 0.0
            preferred_flex = _clamp(
                -0.35 - flex_bias - 1.35 * _clamp(seed_reach + insertion - 0.45 + 0.20 * depth, 0.0, 0.65),
                float(self.arm_flex_min),
                float(self.arm_flex_max),
            )
            fine_lifts = np.linspace(preferred_lift - 0.06, preferred_lift + 0.06, 7)
            fine_flexes = np.linspace(preferred_flex - 0.22, preferred_flex + 0.22, 9)
            pairs.extend(
                (
                    _clamp(float(lift), float(self.arm_lift_min), float(self.arm_lift_max)),
                    _clamp(float(flex), float(self.arm_flex_min), float(self.arm_flex_max)),
                )
                for lift in fine_lifts
                for flex in fine_flexes
            )
        candidates: list[dict[str, float]] = []
        seen: set[tuple[float, float, float]] = set()
        wrist_rolls = [float(self.wrist_roll)]
        for lift, flex in pairs:
            for wrist_roll in wrist_rolls:
                key = (round(float(lift), 5), round(float(flex), 5), round(float(wrist_roll), 5))
                if key in seen:
                    continue
                seen.add(key)
                candidates.append(
                    {
                        "arm_lift_joint": float(lift),
                        "arm_flex_joint": float(flex),
                        "arm_roll_joint": float(self.arm_roll),
                        "wrist_flex_joint": float(self.wrist_flex_offset) + wrist_bias - float(flex),
                        "wrist_roll_joint": float(wrist_roll),
                    }
                )
        return candidates

    def _top_down_candidate_joint_sets(
        self,
        *,
        pregrasp_pose: np.ndarray | None,
        wrist_bias: float,
        planner: Any | None,
        context: Context | None,
        compute_fk=None,
        base_height: float = 0.0,
    ) -> list[dict[str, float]]:
        if pregrasp_pose is None or planner is None:
            return []
        fixed_flex = float(self.arm_flex_min)
        base_joints = {
            "arm_lift_joint": float(self.arm_lift_min),
            "arm_flex_joint": fixed_flex,
            "arm_roll_joint": float(self.arm_roll),
            "wrist_flex_joint": float(self.wrist_flex_offset) + float(wrist_bias) - fixed_flex,
            "wrist_roll_joint": float(self.wrist_roll),
        }
        if compute_fk is not None:
            fk = compute_fk(base_joints)
        else:
            fk_state = _merged_joint_state_msg(context, base_joints) if context is not None else _joint_state_msg(base_joints)
            fk = planner.compute_fk(fk_state, context=context, link_names=["hand_palm_link"], frame_id=str(self.base_frame))
        if not fk.success:
            log.warning("Top-down FK seed failed for candidate joints: %s", base_joints)
            return []
        hand_pose = _fk_pose_matrix(fk)
        if hand_pose is None:
            log.warning("Top-down FK seed returned no hand pose for candidate joints: %s", base_joints)
            return []
        target_z = float(np.asarray(pregrasp_pose, dtype=np.float64)[2, 3])
        solved_lift = float(self.arm_lift_min) + target_z - base_height - float(hand_pose[2, 3])
        solved_lift = _clamp(solved_lift, float(self.arm_lift_min), float(self.arm_lift_max))
        wrist_rolls = _angle_samples(float(self.wrist_roll), step=math.pi / 90.0, count=5)
        return [
            {
                "arm_lift_joint": float(solved_lift),
                "arm_flex_joint": fixed_flex,
                "arm_roll_joint": float(self.arm_roll),
                "wrist_flex_joint": float(self.wrist_flex_offset) + float(wrist_bias) - fixed_flex,
                "wrist_roll_joint": float(wrist_roll),
            }
            for wrist_roll in wrist_rolls
        ]

    def _refined_joint_sets_from_candidates(
        self,
        candidates: list[dict[str, Any]],
        *,
        pregrasp_pose: np.ndarray,
        approach_axis: str,
    ) -> list[dict[str, float]]:
        refined: list[dict[str, float]] = []
        lift_step = max(0.01, (float(self.arm_lift_max) - float(self.arm_lift_min)) / max(8.0, float(self.arm_lift_samples) * 3.0))
        flex_step = max(0.03, (float(self.arm_flex_max) - float(self.arm_flex_min)) / max(8.0, float(self.arm_flex_samples) * 2.0))
        wrist_bias = _wrist_flex_bias_for_approach(pregrasp_pose, approach_axis)
        wrist_roll_offsets = [0.0]
        for item in candidates:
            joints = dict(item.get("joint_positions") or {})
            if not joints:
                continue
            center_lift = float(joints.get("arm_lift_joint", 0.0))
            center_flex = float(joints.get("arm_flex_joint", 0.0))
            center_wrist_roll = float(joints.get("wrist_roll_joint", self.wrist_roll))
            for lift_delta in (-2, -1, 0, 1, 2):
                for flex_delta in (-2, -1, 0, 1, 2):
                    for wrist_delta in wrist_roll_offsets:
                        lift = _clamp(center_lift + lift_delta * lift_step, float(self.arm_lift_min), float(self.arm_lift_max))
                        flex = _clamp(center_flex + flex_delta * flex_step, float(self.arm_flex_min), float(self.arm_flex_max))
                        refined.append(
                            {
                                "arm_lift_joint": float(lift),
                                "arm_flex_joint": float(flex),
                                "arm_roll_joint": float(self.arm_roll),
                                "wrist_flex_joint": float(self.wrist_flex_offset) + wrist_bias - float(flex),
                                "wrist_roll_joint": float(wrap_angle(center_wrist_roll + wrist_delta)),
                            }
                        )
        return refined

    @staticmethod
    def _score_hand_pose(*args, **kwargs) -> float:
        return float(sum(SelectFkArmPosture._score_terms(*args, **kwargs).values()))

    @staticmethod
    def _score_terms(
        hand_pose: np.ndarray,
        grasp_pose: np.ndarray,
        *,
        joints: dict[str, float] | None = None,
        approach_axis: str = "+z",
        scene: dict[str, Any] | None = None,
        sweep: dict[str, Any] | None = None,
        hand_pose_base: np.ndarray | None = None,
        errors: dict[str, float] | None = None,
        arm_flex_compact_weight: float = 0.35,
        approach_alignment_weight: float = 1.25,
        reach_preference_weight: float = 10.0,
        lift_preference_weight: float = 10.0,
        preferred_lift: float | None = None,
        arm_lift_min: float = 0.0,
        arm_lift_max: float = 0.55,
    ) -> dict[str, float]:
        if errors is None:
            errors = _fk_candidate_errors(hand_pose, grasp_pose, hand_pose_base=hand_pose_base, approach_axis=approach_axis)
        base_hand = np.asarray(hand_pose_base, dtype=np.float64) if hand_pose_base is not None else np.asarray(hand_pose, dtype=np.float64)
        reach = float(base_hand[0, 3])
        min_reach = float((scene or {}).get("min_reach_m", 0.25))
        max_reach = float((scene or {}).get("max_reach_m", 0.85))
        reach_preference = max(0.0, reach - min_reach)
        reach_shortfall = max(0.0, min_reach - reach)
        reach_excess = max(0.0, reach - max_reach)
        arm_lift = float((joints or {}).get("arm_lift_joint", 0.0))
        min_lift = (preferred_lift if preferred_lift is not None else
                    _min_lift_for_reach(reach, scene, arm_lift_min=arm_lift_min, arm_lift_max=arm_lift_max))
        lift_shortfall = max(0.0, min_lift - arm_lift)
        lift_preference = max(0.0, arm_lift - min_lift)
        arm_flex = abs(float((joints or {}).get("arm_flex_joint", 0.0)))
        clearance_penalty = float((sweep or {}).get("clearance_penalty", 0.0))
        return {
            "z": 250.0 * float(errors["z_error_m"]),
            "lateral": 12.0 * float(errors["lateral_error_m"]),
            "approach": float(approach_alignment_weight) * 8.0 * float(errors["approach_error_rad"]),
            "inplane": 4.0 * float(errors.get("inplane_error_rad", 0.0)),
            "reach_shortfall": 8.0 * reach_shortfall,
            "reach_excess": reach_excess,
            "reach_preference": float(reach_preference_weight) * reach_preference,
            "lift_shortfall": 10.0 * lift_shortfall,
            "lift_preference": float(lift_preference_weight) * lift_preference,
            # This existing term prefers flex closer to zero, not geometric compactness.
            "arm_flex": float(arm_flex_compact_weight) * 0.25 * arm_flex,
            "clearance": clearance_penalty,
        }


class ExecuteHsrCartesianGraspMotion(AsyncAction):
    """Execute the short final motion from pregrasp to grasp.

    Intended sequence:
    OmniBaseServoToPose(target=...) -> ExecuteHsrCartesianGraspMotion(target=...).

    Planner params:
    - target: object name whose grasp motion and pregrasp posture were selected.
    - mode: optional "ik_waypoints", "arm_cartesian", or "base_servo".
    - debug_visualization: optional bool to show or hide RViz debug markers.
    """

    def __init__(
        self,
        target: str,
        mode: str = 'ik_waypoints',
        output_prefix: str | None = None,
        group_name: str = 'whole_body_weighted',
        ee_link: str = 'hand_palm_link',
        waypoint_count: int = 4,
        waypoint_base_source: str = 'auto',
        base_constraint_tolerance: dict[str, float] | None = None,
        max_base_path_drift: dict[str, float] | None = None,
        final_collision_policy: str = 'target_relaxed',
        cartesian_max_step: float = 0.005,
        cartesian_min_fraction: float = 0.95,
        avoid_collisions: bool = True,
        timeout_s: float = 52.0,
        debug_visualization: bool = True,
        marker_topic: str = '/orchid/grasp_alignment/final_approach',
    ) -> None:
        self.target = target
        self.mode = mode
        self.output_prefix = output_prefix
        self.group_name = group_name
        self.ee_link = ee_link
        self.waypoint_count = waypoint_count
        self.waypoint_base_source = waypoint_base_source
        self.base_constraint_tolerance = base_constraint_tolerance
        self.max_base_path_drift = max_base_path_drift
        self.final_collision_policy = final_collision_policy
        self.cartesian_max_step = cartesian_max_step
        self.cartesian_min_fraction = cartesian_min_fraction
        self.avoid_collisions = avoid_collisions
        self.timeout_s = timeout_s
        self.debug_visualization = debug_visualization
        self.marker_topic = marker_topic
        super().__init__()
        self._status = 'idle'
        self._reason: str | None = None
        self._payload: dict[str, Any] = {}
        self._base_servo: OmniBaseServoToPose | None = None
        self._executing_arm = False




    def _start_ik_waypoints(self, context: Context) -> None:
        ws = context.world_state
        if ws is None:
            raise RuntimeError("missing_world_state")
        target = str(self.target)
        prefix = str(self.output_prefix or target)
        grasp_pose, pregrasp_pose, frame_id, approach_axis = _resolve_grasp_motion_poses(ws, target, prefix)
        base_pose, base_key = _selected_grasp_base_pose(ws, target, str(self.waypoint_base_source))
        waypoints = _interpolate_pose_matrices(pregrasp_pose, grasp_pose, max(1, int(self.waypoint_count)))
        ws.set(f"{prefix}_final_approach_waypoints", [pose.astype(float).tolist() for pose in waypoints])
        ws.set(
            f"{prefix}_final_approach_metadata",
            {
                "mode": "ik_waypoints",
                "base_pose": base_pose.to_dict(),
                "base_pose_key": base_key,
                "waypoint_count": len(waypoints),
                "frame_id": frame_id,
                "approach_axis": approach_axis,
            },
        )
        from src.plugin.actions.manipulation import _MoveArmToPose

        for index, pose_mat in enumerate(waypoints, start=1):
            pose = _pose3d_from_matrix(pose_mat, frame_id)
            is_final = index == len(waypoints)
            collision_policy = str(self.final_collision_policy) if is_final else ("full" if bool(self.avoid_collisions) else "target_relaxed")
            action = self._waypoint_actions[index - 1]
            action.request = ArmPoseRequest(pose, segment=f"final_ik_waypoint_{index}", moveit=MoveItPoseSettings(group_name=str(self.group_name), goal_mode="ik_joint_goal", constrain_base=True,
                base_pose=base_pose, base_constraint_tolerance=dict(self.base_constraint_tolerance or {"x": 0.03, "y": 0.03, "yaw": 0.05}), max_base_path_drift=dict(self.max_base_path_drift or {"xy": 0.04, "yaw": 0.08}), collision_policy=collision_policy))
        self._payload = {
            "mode": "ik_waypoints",
            "waypoint_count": len(waypoints),
            "base_pose": base_pose.to_dict(),
            "base_pose_key": base_key,
        }
        if bool(self.debug_visualization):
            _publish_final_approach_markers(
                context,
                str(self.marker_topic),
                frame_id,
                target,
                pregrasp_pose,
                grasp_pose,
                approach_axis,
            )

    def _start_arm_cartesian(self, context: Context) -> None:
        ws = context.world_state
        if ws is None:
            raise RuntimeError("missing_world_state")
        planner = (
            context.get_planner("manipulation") or context.get_planner("default") or context.get_planner()
            if hasattr(context, "get_planner")
            else context.planners.get("manipulation") or context.planners.get("default")
        )
        if planner is None or not hasattr(planner, "plan_cartesian_request"):
            raise RuntimeError("cartesian_planner_unavailable")
        target = str(self.target)
        prefix = str(self.output_prefix or target)
        grasp_pose, pregrasp_pose, frame_id, approach_axis = _resolve_grasp_motion_poses(ws, target, prefix)
        seed = ws.get(f"{prefix}_fk_joint_goal") or ws.get(f"{prefix}_tmc_joint_goal")
        defaults = planner.execution_defaults
        request = MoveItPlanRequest(group_name=str(self.group_name), goal_mode="direct_pose", pose=_pose3d_from_matrix(grasp_pose, frame_id), ee_link=str(self.ee_link), start_state_seed=seed,
            settings=replace(defaults, cartesian_max_step=float(self.cartesian_max_step), cartesian_min_fraction=float(self.cartesian_min_fraction), avoid_collisions=bool(self.avoid_collisions)))
        plan = planner.plan_cartesian_request(request, context=context)
        self._payload = {
            "mode": "arm_cartesian",
            "plan": plan.to_dict() if hasattr(plan, "to_dict") else {},
            "grasp_pose": grasp_pose.astype(float).tolist(),
            "pregrasp_pose": pregrasp_pose.astype(float).tolist(),
        }
        if bool(self.debug_visualization):
            _publish_final_approach_markers(
                context,
                str(self.marker_topic),
                frame_id,
                target,
                pregrasp_pose,
                grasp_pose,
                approach_axis,
            )
        if not plan.success or plan.trajectory is None:
            self._status = "failure"
            self._reason = plan.reason or "final_cartesian_plan_failed"
            return
        arm = context.require_arm()
        arm.send_trajectory_goal(plan.trajectory)
        self._executing_arm = True

    def _start_base_servo(self, context: Context) -> None:
        ws = context.world_state
        if ws is None:
            raise RuntimeError("missing_world_state")
        target = str(self.target)
        prefix = str(self.output_prefix or target)
        grasp_pose, pregrasp_pose, frame_id, approach_axis = _resolve_grasp_motion_poses(ws, target, prefix)
        hand_pose = _selected_hand_pose_base(ws, prefix)
        if hand_pose is not None:
            motion_metadata = ws.get(f"{prefix}_grasp_motion_metadata") or ws.get(f"{target}_grasp_motion_metadata") or {}
            if str(motion_metadata.get("grasp_motion_type") or "").strip().lower() == "vertical_z":
                final_base = _base_pose_from_vertical_fk_pregrasp_alignment(
                    grasp_pose,
                    hand_pose,
                    frame_id=frame_id,
                    current_base=_current_base_pose(context, frame_id),
                )
            else:
                final_base = _base_pose_from_hand_alignment(
                    grasp_pose,
                    hand_pose,
                    frame_id=frame_id,
                    approach_axis=approach_axis,
                )
        else:
            final_base = _final_base_pose_from_pregrasp_delta(ws, prefix, grasp_pose, pregrasp_pose, frame_id)
            if final_base is None:
                raise RuntimeError("missing_selected_pregrasp_base_pose")
        key = f"{prefix}_final_approach_base_pose"
        ws.set(key, final_base.to_dict())
        self._payload = {
            "mode": "base_servo",
            "base_pose_key": key,
            "target_base_pose": final_base.to_dict(),
        }
        if bool(self.debug_visualization):
            _publish_final_approach_markers(
                context,
                str(self.marker_topic),
                frame_id,
                target,
                pregrasp_pose,
                grasp_pose,
                approach_axis,
                final_base=final_base,
            )
        self._base_servo.target = target
        self._base_servo.base_pose_key = key
        self._base_servo.frame_id = frame_id

    def build_node(self, context, name):
        from src.plugin.actions.manipulation import _MoveArmToPose
        if self.mode == "arm_cartesian":
            child = super().build_node(context, name)
        else:
            if self.mode == "ik_waypoints":
                self._waypoint_actions = [_MoveArmToPose(ArmPoseRequest(Pose3D(0, 0, 0), segment=f"final_ik_waypoint_{index + 1}")) for index in range(max(1, int(self.waypoint_count)))]
                root = py_trees.composites.Sequence(name="Final IK waypoints", memory=True, children=[action.build_node(context, action.request.segment) for action in self._waypoint_actions])
            elif self.mode == "base_servo":
                self._base_servo = OmniBaseServoToPose(target=self.target, timeout_s=self.timeout_s, debug_visualization=self.debug_visualization)
                root = self._base_servo.build_node(context, "Servo final approach")
            else:
                raise ValueError(f"unsupported_final_approach_mode:{self.mode}")
            child = ActionScope(self, context, name, root)
        return py_trees.decorators.Timeout(name=f"{name} timeout", child=child, duration=float(self.timeout_s))

    def on_enter(self, context):
        self._payload = {}
        if self.mode == "ik_waypoints":
            self._start_ik_waypoints(context)
        else:
            self._start_base_servo(context)

    def on_complete(self, context, status):
        return ActionResult(status="success" if status == py_trees.common.Status.SUCCESS else "failure", payload=self._payload)

    def start(self, context):
        self._status, self._reason = "running", None
        self._payload = {}
        self._executing_arm = False
        try:
            self._start_arm_cartesian(context)
        except Exception as exc:
            self._status, self._reason = "failure", str(exc)

    def check_result(self, context):
        if self._status != "running":
            return ActionResult(status=self._status, reason=self._reason, payload=self._payload)
        return _arm_status_result(context.require_arm()) if self._executing_arm else ActionResult(status="running")

    def cancel(self, context):
        if self._executing_arm:
            context.require_arm().cancel_goal()
        self._status, self._reason = "failure", "final_approach_cancelled"



class _TopDownPreGraspAdjustAction(ControllerAsyncAction):
    """Internal action for top-down wrist alignment and lift-down at pregrasp."""

    required_controllers = ["arm"]

    def __init__(self, target: str, lift_down_m: float | None=None, max_lift_down_m: float=0.08) -> None:
        self.target = target
        self.lift_down_m = lift_down_m
        self.max_lift_down_m = max_lift_down_m
        super().__init__()
        self.target = str(target)
        self.max_lift_down_m = float(max_lift_down_m)
        self._dispatch_status = 'idle'
        self._failure_reason: str | None = None
        self._move_started = False
        self.goal: dict[str, float] = {}

    def start(self, context: Context) -> None:
        ws = context.world_state
        if ws is None:
            self._dispatch_status = "failure"
            self._failure_reason = "missing_world_state"
            return
        metadata = ws.get(f"{self.target}_grasp_motion_metadata") or {}
        if str(metadata.get("grasp_motion_type") or "").strip().lower() != "vertical_z":
            self._dispatch_status = "succeeded"
            return
        cached = ws.get(f"{self.target}_fk_joint_goal")
        if not isinstance(cached, dict) or "wrist_roll_joint" not in cached:
            self._dispatch_status = "failure"
            self._failure_reason = f"missing_top_down_fk_joint_goal:{self.target}_fk_joint_goal"
            return

        def send_goal() -> dict[str, Any]:
            try:
                from src.plugin.actions.manipulation import _clamp_joint_value, _joint_limits_from_context, _latest_joint_positions

                current = _latest_joint_positions(context)
                if "arm_lift_joint" not in current:
                    return {"success": False, "error": "missing_joint_state:arm_lift_joint"}
                lift_down = self._resolve_lift_down(ws)
                limits = _joint_limits_from_context(context)
                goal = {
                    "wrist_roll_joint": float(cached["wrist_roll_joint"]),
                    "arm_lift_joint": _clamp_joint_value("arm_lift_joint", float(current["arm_lift_joint"]) - lift_down, limits),
                }
                context.require_arm().send_joint_goal(goal)
                self.goal = goal
                if context.require_arm().get_status() == "failed":
                    return {"success": False, "error": context.require_arm().get_error() or "top_down_pregrasp_adjust_failed"}
                return {"success": True}
            except Exception as exc:  # noqa: BLE001
                return {"success": False, "error": str(exc)}

        def on_dispatched(result: dict[str, Any]) -> None:
            if result.get("success"):
                self._move_started = True
                self._dispatch_status = "dispatched"
            else:
                self._dispatch_status = "failure"
                self._failure_reason = str(result.get("error") or "top_down_pregrasp_adjust_failed")

        try:
            context.run_in_context_loop(send_goal, callback=on_dispatched, error_callback=lambda exc: on_dispatched({"success": False, "error": str(exc)}))
        except Exception as exc:  # noqa: BLE001
            self._dispatch_status = "failure"
            self._failure_reason = str(exc)

    def check_result(self, context: Context) -> ActionResult:
        if self._cancel_requested:
            return ActionResult(status="failure", reason="Action Cancelled")
        if self._dispatch_status == "succeeded":
            return ActionResult(status="success", reason="top_down_pregrasp_adjust_noop")
        if self._dispatch_status == "failure":
            return ActionResult(status="failure", reason=self._failure_reason or "top_down_pregrasp_adjust_failed")
        if self._dispatch_status != "dispatched" or not self._move_started:
            return ActionResult(status="running")
        result = _arm_status_result(context.require_arm())
        if result.status == "success":
            return ActionResult(status="success", reason="top_down_pregrasp_adjust_completed", payload={"goal": self.goal})
        return result

    def reset(self) -> None:
        super().reset()
        self._dispatch_status = "idle"
        self._failure_reason = None
        self._move_started = False
        self.goal = {}

    def cancel(self, context: Context) -> None:
        self._cancel_requested = True

    def _resolve_lift_down(self, ws: Any) -> float:
        if self.lift_down_m is not None:
            return _clamp(float(self.lift_down_m), 0.0, float(self.max_lift_down_m))
        grasp = ws.get(f"{self.target}_grasp_pose")
        pregrasp = ws.get(f"{self.target}_pregrasp_pose")
        if grasp is None or pregrasp is None:
            return 0.0
        delta = float(np.asarray(pregrasp, dtype=np.float64)[2, 3]) - float(np.asarray(grasp, dtype=np.float64)[2, 3])
        return _clamp(max(0.0, delta), 0.0, float(self.max_lift_down_m))


class PickTargetWithGraspAlignment(AbstractAction):
    """Pick an object using the arm given its name and corresponding attributes in world state. 
    
    This action retrieves estimated grasp poses from the world state (typically populated
    by EstimateSimpleCylindricalGrasp or similar EstimatePickPose actions).

    Planner params:
    - target: object name from perception, such as "blue_jar".
      posture before moving the base.
    """

    # - force_N: optional gripper force for the final grasp.
    # - width: optional object width/gripper opening in meters.
    # - debug_visualization: optional bool to show or hide RViz debug markers.
    # - undock_mode: optional "return_to_start" or "relative_back" retreat mode.
    # - fk_execute_before_base: optional bool to unfold the selected FK arm
    # - pregrasp_mode: optional "simple" or "horizontal_projected".
    # - fk_sampling_mode: optional "simple" or "multires_scene".
    required_controllers = ["arm", "gripper"]

    def __init__(
        self,
        target: str = '',
        final_approach_mode: str = 'base_servo',
        undock_mode: str = 'return_to_start',
        undock_distance_m: float = 0.35,
        post_grasp_lift_m: float = 0.05,
        pregrasp_mode: str = 'horizontal_projected',
        fk_sampling_mode: str = 'multires_scene',
        fk_execute_before_base: bool = False,
        force_N: float = 0.5,
        width: float = 0.15,
        debug_visualization: bool = True,
        gripper_margin_m: float = 0.02,
        include_surround_collision: bool = True,
        collision_hull_inflation_factor: float = 2.0,
    ) -> None:
        self.target = target
        self.final_approach_mode = final_approach_mode
        self.undock_mode = undock_mode
        self.undock_distance_m = undock_distance_m
        self.post_grasp_lift_m = post_grasp_lift_m
        self.pregrasp_mode = pregrasp_mode
        self.fk_sampling_mode = fk_sampling_mode
        self.fk_execute_before_base = fk_execute_before_base
        self.force_N = force_N
        self.width = width
        self.debug_visualization = debug_visualization
        self.gripper_margin_m = gripper_margin_m
        self.include_surround_collision = include_surround_collision
        self.collision_hull_inflation_factor = validate_hull_inflation(collision_hull_inflation_factor)
        super().__init__()





    def _fail(self, reason: str) -> None:
        self._status = "failure"
        self._failure_reason = reason
        log.error("PickTargetWithGraspAlignment failed to initialize: %s", reason)

    def _snapshot_pick_start_base(self, context: Context, target: str, frame_id: str) -> None:
        if context.world_state is None or context.sensor_data_center is None:
            return
        try:
            robot_pose = context.sensor_data_center.get_robot_pose_global(global_frame=frame_id)
            if not robot_pose:
                return
            pose = robot_pose.get("pose", robot_pose) if isinstance(robot_pose, dict) else robot_pose
            base_pose = {
                "x": float(getattr(pose, "x", robot_pose.get("x", 0.0) if isinstance(robot_pose, dict) else 0.0)),
                "y": float(getattr(pose, "y", robot_pose.get("y", 0.0) if isinstance(robot_pose, dict) else 0.0)),
                "yaw": float(getattr(pose, "yaw", robot_pose.get("yaw", robot_pose.get("theta", 0.0)) if isinstance(robot_pose, dict) else 0.0)),
                "frame_id": frame_id,
            }
            context.world_state.set("last_pick_start_base_pose", base_pose)
            context.world_state.set(f"{target}_pick_start_base_pose", base_pose)
        except Exception as exc:  # noqa: BLE001
            log.debug("Unable to snapshot grasp-alignment pick start base pose: %s", exc)

    def _cleanup_collision_scene(self, context: Context) -> None:
        try:
            context.require_arm().clear_collision_scene()
        except Exception as exc:  # noqa: BLE001
            log.debug("PickTargetWithGraspAlignment collision cleanup skipped: %s", exc)

    def build_node(self, context, name):
        from src.plugin.actions.manipulation import GraspObjectAction, MoveArmToState, ReleaseObjectAction, _AddCollisionMeshAction, _AllowCollisionWithConfiguredLinksAction, _AllowCollisionPairsAction, _ClearOctomapAction, _ClearCollisionSceneAction, _MoveJointsRelativeAction
        target = str(self.target)
        self._object_mesh = _AddCollisionMeshAction(name="pick_target_collision", convex_hull=True, inflation_factor=self.collision_hull_inflation_factor)
        self._surround_mesh = _AddCollisionMeshAction(name="pick_surround_collision")
        self._open = ReleaseObjectAction(distance=self.width)
        self._close = GraspObjectAction(force_N=self.force_N, width_m=self.width)
        self._undock = OmniBaseServoToPose()
        self._post_links = _AllowCollisionWithConfiguredLinksAction(["pick_target_collision", "<octomap>"])
        def optional_surround(action, label):
            skip = BtConditionLeaf(lambda context: not self._has_surround, context)
            skip.name = "No surrounding collision geometry"
            return py_trees.composites.Selector(name=label, memory=True, children=[skip, action.build_node(context, label)])
        self._object_mesh_node = self._object_mesh.build_node(context, "Model target")
        children = [_ClearCollisionSceneAction().build_node(context, "Clear scene"), self._object_mesh_node]
        if self.include_surround_collision:
            children.append(optional_surround(self._surround_mesh, "Model surroundings"))
        children.extend([_ClearOctomapAction().build_node(context, "Clear octomap"), SelectHsrGraspMotion(target=target, pregrasp_mode=self.pregrasp_mode, debug_visualization=self.debug_visualization).build_node(context, "Select grasp motion"),
            _AllowCollisionWithConfiguredLinksAction(["pick_target_collision"]).build_node(context, "Allow hand contact"),
            SelectFkArmPosture(target=target, execute=False, sampling_mode=self.fk_sampling_mode, debug_visualization=self.debug_visualization).build_node(context, "Select pregrasp posture")])
        posture = SelectFkArmPosture(target=target, execute=True, sampling_mode=self.fk_sampling_mode, debug_visualization=self.debug_visualization).build_node(context, "Execute pregrasp posture")
        base = MoveToSelectedGraspBasePose(target=target, source="fk").build_node(context, "Align base")
        children.extend([posture, base] if self.fk_execute_before_base else [base, posture])
        self._close_node = self._close.build_node(context, "Close gripper")
        children.extend([self._open.build_node(context, "Open gripper"), ExecuteHsrCartesianGraspMotion(target=target, mode=self.final_approach_mode, debug_visualization=self.debug_visualization).build_node(context, "Final approach"), _TopDownPreGraspAdjustAction(target=target).build_node(context, "Adjust top down pregrasp"), self._close_node])
        if self.include_surround_collision:
            children.append(optional_surround(_AllowCollisionPairsAction([("pick_target_collision", "pick_surround_collision")]), "Allow held object contact"))
        children.append(self._post_links.build_node(context, "Allow held object links"))
        if self.post_grasp_lift_m != 0.0:
            children.append(_MoveJointsRelativeAction({"arm_lift_joint": self.post_grasp_lift_m}).build_node(context, "Lift held object"))
        children.extend([self._undock.build_node(context, "Undock"), MoveArmToState(state="go").build_node(context, "Stow arm"), _ClearCollisionSceneAction().build_node(context, "Clear scene after pick"), _ClearOctomapAction().build_node(context, "Clear octomap")])
        return ActionScope(self, context, name, py_trees.composites.Sequence(name="Grasp alignment pick", memory=True, children=children))

    def on_enter(self, context):
        ws, target = context.world_state, str(self.target or "")
        if not target:
            return ActionResult(status="failure", reason="missing_target", failure=FailureInfo(FailureCode.INVALID_REQUEST))
        movable = _movable_object_state(ws, target)
        object_key = str(getattr(movable, "object_pcd_key", None) or f"{target}_object_pcd")
        surround_key = str(getattr(movable, "surround_pcd_key", None) or f"{target}_surround_pcd")
        self._object_mesh.pcd = ws.get(object_key)
        if self._object_mesh.pcd is None:
            return ActionResult(status="failure", reason=f"missing_object_pcd:{object_key}", failure=FailureInfo(FailureCode.INVALID_STATE))
        self._surround_mesh.pcd = ws.get(surround_key)
        self._has_surround = bool(self.include_surround_collision and self._surround_mesh.pcd is not None)
        self._post_links.object_names = ["pick_target_collision", "<octomap>"] + (["pick_surround_collision"] if self._has_surround else [])
        frame = str(getattr(movable, "frame_id", None) or ws.get(f"{target}_pcd_frame") or context.hardware.get("odom", "odom") or "odom")
        self._object_mesh.frame_id = frame
        self._snapshot_pick_start_base(context, target, frame)
        start = ws.get("last_pick_start_base_pose")
        if start is None:
            return ActionResult(status="failure", reason="missing_pick_start_base_pose", failure=FailureInfo(FailureCode.INVALID_STATE))
        self._undock.x, self._undock.y, self._undock.yaw, self._undock.frame_id = start["x"], start["y"], start["yaw"], start["frame_id"]
        width = ws.get(f"{target}_estimated_width") or self.width
        margin = max(0.0, float(self.gripper_margin_m))
        self._open.distance, self._close.width_m = width + margin, width - margin

    def on_complete(self, context, status):
        if status == py_trees.common.Status.SUCCESS:
            return ActionResult(status="success")
        if self._close_node.started:
            return ActionResult(status="needs_replan", reason="pick_grasp_alignment_failed_after_close", force_replan=True, suggested_replan_context={"target": self.target, "grip_state": "commanded_hold" if self._close_node.result and self._close_node.result.status == "success" else "uncertain"})
        self._cleanup_collision_scene(context)
        if self._object_mesh_node.result is not None and self._object_mesh_node.result.status == "failure":
            return self._object_mesh_node.result
        return ActionResult(status="failure", reason="pick_grasp_alignment_failed")

    def on_cancel(self, context):
        if not self._close_node.started:
            self._cleanup_collision_scene(context)


class OmniBaseServoToPose(AsyncAction):
    """Servo the omni base to the selected base pose.

    Planner params:
    - target: object name. The action tracks the TMC or FK-derived base pose when available,
      otherwise can pass x, y, yaw and frame_id as well.
    """

    # - debug_visualization: optional bool to show or hide RViz debug markers.

    def __init__(
        self,
        target: str | None = None,
        base_pose_key: str | None = None,
        x: float | None = None,
        y: float | None = None,
        yaw: float | None = None,
        frame_id: str = 'odom',
        cmd_vel_topic: str = '/omni_base_controller/cmd_vel',
        rate_hz: float = 20.0,
        timeout_s: float = 50.0,
        xy_tolerance_m: float = 0.025,
        yaw_tolerance_rad: float = 0.04,
        settle_cycles: int = 5,
        projection_dt_s: float = 0.5,
        kx: float = 0.8,
        ky: float = 0.8,
        kyaw: float = 1.2,
        max_vx: float = 0.12,
        max_vy: float = 0.12,
        max_wz: float = 0.35,
        max_ax: float = 0.2,
        max_ay: float = 0.2,
        max_awz: float = 0.6,
        collision_gate_mode: str = 'diagnostic',
        debug_visualization: bool = True,
        marker_topic: str = '/orchid/grasp_alignment/servo',
        marker_publish_period_s: float = 0.2,
    ) -> None:
        self.target = target
        self.base_pose_key = base_pose_key
        self.x = x
        self.y = y
        self.yaw = yaw
        self.frame_id = frame_id
        self.cmd_vel_topic = cmd_vel_topic
        self.rate_hz = rate_hz
        self.timeout_s = timeout_s
        self.xy_tolerance_m = xy_tolerance_m
        self.yaw_tolerance_rad = yaw_tolerance_rad
        self.settle_cycles = settle_cycles
        self.projection_dt_s = projection_dt_s
        self.kx = kx
        self.ky = ky
        self.kyaw = kyaw
        self.max_vx = max_vx
        self.max_vy = max_vy
        self.max_wz = max_wz
        self.max_ax = max_ax
        self.max_ay = max_ay
        self.max_awz = max_awz
        self.collision_gate_mode = collision_gate_mode
        self.debug_visualization = debug_visualization
        self.marker_topic = marker_topic
        self.marker_publish_period_s = marker_publish_period_s
        super().__init__()
        self._thread: threading.Thread | None = None
        self._status = 'idle'
        self._reason: str | None = None
        self._payload: dict[str, Any] = {}
        self._publisher = None
        self._last_cmd = ServoCommand(0.0, 0.0, 0.0, float('inf'), float('inf'))

    def start(self, context: Context) -> None:
        try:
            target_pose = self._resolve_target_pose(context)
            node = context.ros2_node or getattr(getattr(context, "sensor_data_center", None), "node", None)
            if node is None:
                self._status = "failure"
                self._reason = "missing_ros2_node"
                return
            from geometry_msgs.msg import Twist

            self._Twist = Twist
            self._publisher = node.create_publisher(Twist, str(self.cmd_vel_topic), 10)
            self._status = "running"
            self._reason = None
            self._thread = threading.Thread(target=self._run_servo, args=(context, target_pose), daemon=True)
            self._thread.start()
        except Exception as exc:  # noqa: BLE001
            log.exception("Omni servo start failed: %s", exc)
            self._status = "failure"
            self._reason = str(exc)

    def check_result(self, context: Context) -> ActionResult:
        if self._status == "running":
            return ActionResult(status="running")
        if self._status == "success":
            return ActionResult(status="success", reason=self._reason, payload=self._payload)
        return ActionResult(status="failure", reason=self._reason or "omni_servo_failed", payload=self._payload)

    def cancel(self, context: Context) -> None:
        self._cancel_requested = True
        self._publish_zero()

    def _run_servo(self, context: Context, target_pose: BasePose2D) -> None:
        started = time.monotonic()
        dt = 1.0 / max(1.0, float(self.rate_hz))
        settled = 0
        last_time = time.monotonic()
        last_marker_time = 0.0
        try:
            while not self._cancel_requested:
                now = time.monotonic()
                if now - started > float(self.timeout_s):
                    self._finish("failure", "omni_servo_timeout")
                    return
                current = _current_base_pose(context, target_pose.frame_id)
                if current is None:
                    self._finish("failure", "missing_current_base_pose")
                    return
                raw = compute_omni_servo_command(
                    current,
                    target_pose,
                    kx=float(self.kx),
                    ky=float(self.ky),
                    kyaw=float(self.kyaw),
                    max_vx=float(self.max_vx),
                    max_vy=float(self.max_vy),
                    max_wz=float(self.max_wz),
                )
                if raw.xy_error <= float(self.xy_tolerance_m) and raw.yaw_error <= float(self.yaw_tolerance_rad):
                    settled += 1
                    self._publish_zero()
                    if settled >= int(self.settle_cycles):
                        self._finish("success", "omni_servo_reached", {"final_error": raw.to_dict()})
                        return
                    time.sleep(dt)
                    continue
                settled = 0
                elapsed = max(1e-3, now - last_time)
                last_time = now
                cmd = self._limit_acceleration(raw, elapsed)
                projected = _project_base_pose(current, cmd, float(self.projection_dt_s))
                if bool(self.debug_visualization) and now - last_marker_time >= float(self.marker_publish_period_s):
                    _publish_servo_markers(
                        context,
                        str(self.marker_topic),
                        target_pose.frame_id,
                        target_pose,
                        current,
                        projected,
                    )
                    last_marker_time = now
                collision = _servo_collision_allowed(context, projected, target=str(self.target or ""))
                if not collision.get("allowed", True):
                    if bool(self.debug_visualization):
                        _publish_servo_markers(
                            context,
                            str(self.marker_topic),
                            target_pose.frame_id,
                            target_pose,
                            current,
                            projected,
                            blocked=projected,
                        )
                    self._payload["collision_gate"] = collision
                    if str(self.collision_gate_mode).strip().lower() == "enforce":
                        self._publish_zero()
                        self._finish("failure", str(collision.get("reason") or "omni_servo_collision_blocked"), collision)
                        return
                self._publish_cmd(cmd)
                self._payload.update({"last_command": cmd.to_dict(), "target_pose": target_pose.to_dict()})
                time.sleep(dt)
            self._finish("failure", "omni_servo_cancelled")
        finally:
            self._publish_zero()

    def _resolve_target_pose(self, context: Context) -> BasePose2D:
        if self.base_pose_key:
            value = context.world_state.get(str(self.base_pose_key)) if context.world_state else None
            if value is None:
                raise ValueError(f"missing_base_pose_key:{self.base_pose_key}")
            return BasePose2D.from_mapping(value, default_frame_id=str(self.frame_id))
        if self.target:
            for suffix in ("_tmc_base_pose", "_fk_base_pose"):
                value = context.world_state.get(f"{self.target}{suffix}") if context.world_state else None
                if value is not None:
                    return BasePose2D.from_mapping(value, default_frame_id=str(self.frame_id))
        if self.x is None or self.y is None or self.yaw is None:
            raise ValueError("missing_target_pose")
        return BasePose2D(x=float(self.x), y=float(self.y), yaw=float(self.yaw), frame_id=str(self.frame_id))

    def _limit_acceleration(self, raw: ServoCommand, dt: float) -> ServoCommand:
        last = self._last_cmd
        cmd = ServoCommand(
            vx=_step_toward(last.vx, raw.vx, float(self.max_ax) * dt),
            vy=_step_toward(last.vy, raw.vy, float(self.max_ay) * dt),
            wz=_step_toward(last.wz, raw.wz, float(self.max_awz) * dt),
            xy_error=raw.xy_error,
            yaw_error=raw.yaw_error,
        )
        self._last_cmd = cmd
        return cmd

    def _publish_cmd(self, cmd: ServoCommand) -> None:
        if self._publisher is None:
            return
        msg = self._Twist()
        msg.linear.x = float(cmd.vx)
        msg.linear.y = float(cmd.vy)
        msg.angular.z = float(cmd.wz)
        self._publisher.publish(msg)

    def _publish_zero(self) -> None:
        self._last_cmd = ServoCommand(0.0, 0.0, 0.0, self._last_cmd.xy_error, self._last_cmd.yaw_error)
        self._publish_cmd(self._last_cmd)

    def _finish(self, status: str, reason: str, payload: dict[str, Any] | None = None) -> None:
        self._status = status
        self._reason = reason
        if payload:
            self._payload.update(payload)


def _build_tmc_ik_request(context: Context, pose_mat: np.ndarray, frame_id: str, current_base: BasePose2D) -> Any:
    from tmc_manipulation_msgs.srv import SolveIkWithCollision

    request = SolveIkWithCollision.Request()
    request.origin_to_hand_goal = _pose_msg_from_matrix(pose_mat)
    request.initial_origin_to_base = _pose_msg_from_matrix(planar_pose_matrix(current_base.x, current_base.y, current_base.yaw))
    joint_state = _current_joint_state_msg(context)
    if joint_state is not None:
        request.initial_joint_state = joint_state
    # Keep environment/obstacle map default for v1; the service accepts empty values.
    return request


def _tmc_ik_request_metadata(context: Context, pose_key: str, frame_id: str, current_base: BasePose2D, module_name: str) -> dict[str, Any]:
    joint_state = _current_joint_state_msg(context)
    names = [str(name) for name in getattr(joint_state, "name", [])] if joint_state is not None else []
    return {
        "pose_key": pose_key,
        "frame_id": frame_id,
        "initial_origin_to_base": current_base.to_dict(),
        "initial_joint_names": names,
        "initial_joint_count": len(names),
        "module_name": module_name,
    }


def _response_summary(response: Any) -> dict[str, Any]:
    raw = response.get("response", response) if isinstance(response, dict) else response
    summary: dict[str, Any] = {"type": type(raw).__name__}
    for name in ("error_code", "error", "message", "status"):
        if hasattr(raw, name):
            summary[name] = str(getattr(raw, name))
    for name in ("ik_results", "solutions"):
        if hasattr(raw, name):
            try:
                summary[f"{name}_count"] = len(getattr(raw, name) or [])
            except Exception:
                pass
    return summary


def _movable_object_state(ws: Any, target: str) -> Any | None:
    if ws is None or not target or not hasattr(ws, "get_movable_object_state"):
        return None
    try:
        return ws.get_movable_object_state(str(target))
    except Exception as exc:  # noqa: BLE001
        log.debug("Unable to read movable object state for %s: %s", target, exc)
        return None


def _ros_node(context: Context) -> Any | None:
    return getattr(context, "ros2_node", None) or getattr(getattr(context, "sensor_data_center", None), "node", None)


def _marker_publisher(context: Context, topic: str) -> Any | None:
    node = _ros_node(context)
    if node is None:
        return None
    key = (id(node), topic)
    cached = _GRASP_ALIGNMENT_MARKER_PUBLISHERS.get(key)
    if cached is not None:
        return cached
    try:
        from visualization_msgs.msg import MarkerArray

        qos: Any = 10
        try:
            from rclpy.qos import QoSDurabilityPolicy, QoSProfile

            qos = QoSProfile(depth=1, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
        except Exception:
            qos = 10
        publisher = node.create_publisher(MarkerArray, topic, qos)
        _GRASP_ALIGNMENT_MARKER_PUBLISHERS[key] = publisher
        return publisher
    except Exception as exc:  # noqa: BLE001
        log.debug("Unable to create marker publisher for %s: %s", topic, exc)
        return None


def _publish_simple_grasp_markers(context: Context, topic: str, frame_id: str, pose_mat: np.ndarray, target: str) -> None:
    try:
        markers = _pose_axis_markers(frame_id, pose_mat, ns=f"{target}_simple_grasp", start_id=0, scale=0.18, selected=True)
        markers.append(_text_marker(frame_id, pose_mat, ns=f"{target}_simple_grasp", marker_id=20, text="grasp"))
        _publish_marker_array(context, topic, markers)
    except Exception as exc:  # noqa: BLE001
        log.debug("Simple grasp marker publish skipped: %s", exc)


def _publish_pca_grasp_markers(context: Context, topic: str, frame_id: str, pose_mat: np.ndarray, target: str, metadata: dict[str, Any]) -> None:
    try:
        markers = _pose_axis_markers(frame_id, pose_mat, ns=f"{target}_pca_grasp", start_id=0, scale=0.18, selected=True)
        strategy = str(metadata.get("selected_strategy") or "pca")
        markers.append(_text_marker(frame_id, pose_mat, ns=f"{target}_pca_grasp", marker_id=20, text=strategy.replace("_", " ")))
        pca = metadata.get("pca") if isinstance(metadata.get("pca"), dict) else {}
        center = np.asarray(pca.get("top_center") or pca.get("centroid") or pose_mat[:3, 3], dtype=np.float64)
        long_axis = np.asarray(pca.get("long_axis_xy") or [1.0, 0.0], dtype=np.float64)
        short_axis = np.asarray(pca.get("short_axis_xy") or [0.0, 1.0], dtype=np.float64)
        markers.append(
            _arrow_points_marker(
                frame_id,
                center,
                center + np.array([long_axis[0], long_axis[1], 0.0], dtype=np.float64) * 0.14,
                ns=f"{target}_pca_axes",
                marker_id=30,
                color=(0.85, 0.85, 0.85, 0.7),
                shaft_diameter=0.008,
                head_diameter=0.02,
            )
        )
        markers.append(
            _arrow_points_marker(
                frame_id,
                center,
                center + np.array([short_axis[0], short_axis[1], 0.0], dtype=np.float64) * 0.14,
                ns=f"{target}_pca_axes",
                marker_id=31,
                color=(1.0, 0.8, 0.05, 0.85),
                shaft_diameter=0.01,
                head_diameter=0.025,
            )
        )
        _publish_marker_array(context, topic, markers)
    except Exception as exc:  # noqa: BLE001
        log.debug("PCA grasp marker publish skipped: %s", exc)


def _publish_grasp_motion_markers(
    context: Context,
    topic: str,
    frame_id: str,
    target: str,
    grasp_pose: np.ndarray,
    pregrasp_pose: np.ndarray,
    retreat_pose: np.ndarray,
    scene: dict[str, Any],
) -> None:
    try:
        markers: list[Any] = []
        markers.extend(_pose_axis_markers(frame_id, pregrasp_pose, ns=f"{target}_pregrasp", start_id=0, scale=0.14, selected=False, alpha=0.8))
        markers.extend(_pose_axis_markers(frame_id, grasp_pose, ns=f"{target}_grasp", start_id=20, scale=0.18, selected=True))
        markers.extend(_pose_axis_markers(frame_id, retreat_pose, ns=f"{target}_retreat", start_id=40, scale=0.12, selected=False, alpha=0.7))
        markers.append(
            _arrow_points_marker(
                frame_id,
                pregrasp_pose[:3, 3],
                grasp_pose[:3, 3],
                ns=f"{target}_approach_line",
                marker_id=60,
                color=(1.0, 0.8, 0.05, 1.0),
                shaft_diameter=0.012,
                head_diameter=0.035,
            )
        )
        markers.append(
            _arrow_points_marker(
                frame_id,
                grasp_pose[:3, 3],
                retreat_pose[:3, 3],
                ns=f"{target}_retreat_line",
                marker_id=61,
                color=(0.4, 0.8, 1.0, 0.8),
                shaft_diameter=0.01,
                head_diameter=0.03,
            )
        )
        support_depth = scene.get("support_depth_m")
        label = "grasp"
        if support_depth is not None:
            label = f"d={float(support_depth):.2f}m"
        markers.append(_text_marker(frame_id, pregrasp_pose, ns=f"{target}_pregrasp", marker_id=80, text="pre"))
        markers.append(_text_marker(frame_id, grasp_pose, ns=f"{target}_grasp_motion", marker_id=81, text=label))
        markers.append(_text_marker(frame_id, retreat_pose, ns=f"{target}_retreat", marker_id=82, text="retreat"))
        _publish_marker_array(context, topic, markers)
    except Exception as exc:  # noqa: BLE001
        log.debug("Grasp motion marker publish skipped: %s", exc)


def _publish_tmc_ik_markers(
    context: Context,
    topic: str,
    frame_id: str,
    grasp_pose: np.ndarray,
    ranked: list[dict[str, Any]],
    selected: dict[str, Any],
    target: str,
) -> None:
    try:
        markers: list[Any] = []
        markers.extend(_pose_axis_markers(frame_id, grasp_pose, ns=f"{target}_tmc_grasp", start_id=0, scale=0.13, selected=True))
        selected_index = int(selected.get("index", -1))
        marker_id = 100
        for item in ranked:
            base = BasePose2D.from_mapping(item["origin_to_base"], default_frame_id=frame_id)
            is_selected = int(item.get("index", -2)) == selected_index
            markers.append(
                _base_arrow_marker(
                    frame_id,
                    base,
                    ns=f"{target}_tmc_bases",
                    marker_id=marker_id,
                    color=(0.1, 0.8, 0.2, 1.0) if is_selected else (0.55, 0.55, 0.55, 0.45),
                    scale=0.28 if is_selected else 0.18,
                )
            )
            marker_id += 1
        _publish_marker_array(context, topic, markers)
    except Exception as exc:  # noqa: BLE001
        log.debug("TMC IK marker publish skipped: %s", exc)


def _publish_fk_posture_markers(
    context: Context,
    topic: str,
    world_frame: str,
    base_frame: str,
    candidates: list[dict[str, Any]],
    selected: dict[str, Any],
    target: str,
    *,
    max_candidates: int,
) -> None:
    try:
        markers: list[Any] = []
        selected_joints = selected.get("joint_positions", {})
        step = max(1, int(math.ceil(max(1, len(candidates)) / max(1, max_candidates))))
        marker_id = 0
        for item in candidates[::step]:
            is_selected = item.get("joint_positions") == selected_joints
            raw_pose = np.asarray(item["hand_pose_base"], dtype=np.float64)
            markers.extend(
                _pose_axis_markers(
                    base_frame,
                    raw_pose,
                    ns=f"{target}_fk_raw_candidates",
                    start_id=marker_id,
                    scale=0.16 if is_selected else 0.08,
                    selected=is_selected,
                    alpha=1.0 if is_selected else 0.35,
                )
            )
            marker_id += 3
        marker_id = 5000
        for item in candidates[::step]:
            resolved_value = item.get("resolved_hand_pose_world")
            if resolved_value is None:
                continue
            pose = np.asarray(resolved_value, dtype=np.float64)
            is_selected = item.get("joint_positions") == selected_joints
            markers.extend(
                _pose_axis_markers(
                    world_frame,
                    pose,
                    ns=f"{target}_fk_resolved_candidates",
                    start_id=marker_id,
                    scale=0.16 if is_selected else 0.08,
                    selected=is_selected,
                    alpha=1.0 if is_selected else 0.35,
                )
            )
            marker_id += 3
        base_pose = BasePose2D.from_mapping(selected["base_pose"], default_frame_id=world_frame)
        markers.append(
            _base_arrow_marker(
                world_frame,
                base_pose,
                ns=f"{target}_fk_base",
                marker_id=10000,
                color=(0.1, 0.8, 0.2, 1.0),
                scale=0.30,
            )
        )
        markers.append(_text_marker(world_frame, _planar_marker_pose(base_pose), ns=f"{target}_fk_base", marker_id=10001, text="fk base"))
        resolved = selected.get("resolved_hand_pose_world")
        if resolved is not None:
            resolved_pose = np.asarray(resolved, dtype=np.float64)
            markers.extend(
                _pose_axis_markers(
                    world_frame,
                    resolved_pose,
                    ns=f"{target}_fk_resolved_hand",
                    start_id=10100,
                    scale=0.18,
                    selected=True,
                    alpha=1.0,
                )
            )
            markers.append(
                _text_marker(
                    world_frame,
                    resolved_pose,
                    ns=f"{target}_fk_resolved_hand",
                    marker_id=10110,
                    text="resolved",
                )
            )
        _publish_marker_array(context, topic, markers)
    except Exception as exc:  # noqa: BLE001
        log.debug("FK posture marker publish skipped: %s", exc)


def _publish_final_approach_markers(
    context: Context,
    topic: str,
    frame_id: str,
    target: str,
    pregrasp_pose: np.ndarray,
    grasp_pose: np.ndarray,
    approach_axis: str,
    *,
    final_base: BasePose2D | None = None,
    failed_pose: np.ndarray | None = None,
) -> None:
    try:
        markers: list[Any] = []
        markers.extend(_pose_axis_markers(frame_id, pregrasp_pose, ns=f"{target}_final_pregrasp", start_id=0, scale=0.13, selected=False, alpha=0.75))
        markers.extend(_pose_axis_markers(frame_id, grasp_pose, ns=f"{target}_final_grasp", start_id=20, scale=0.17, selected=True))
        markers.append(
            _arrow_points_marker(
                frame_id,
                pregrasp_pose[:3, 3],
                grasp_pose[:3, 3],
                ns=f"{target}_final_path",
                marker_id=50,
                color=(0.1, 0.8, 0.2, 1.0),
                shaft_diameter=0.015,
                head_diameter=0.04,
            )
        )
        if final_base is not None:
            markers.append(
                _base_arrow_marker(
                    frame_id,
                    final_base,
                    ns=f"{target}_final_base",
                    marker_id=70,
                    color=(0.1, 0.8, 0.2, 1.0),
                    scale=0.28,
                )
            )
        if failed_pose is not None:
            markers.extend(_pose_axis_markers(frame_id, failed_pose, ns=f"{target}_final_failed", start_id=90, scale=0.16, selected=True))
        markers.append(_text_marker(frame_id, pregrasp_pose, ns=f"{target}_final_approach", marker_id=110, text="pre"))
        markers.append(_text_marker(frame_id, grasp_pose, ns=f"{target}_final_approach", marker_id=111, text="grasp"))
        _publish_marker_array(context, topic, markers)
    except Exception as exc:  # noqa: BLE001
        log.debug("Final approach marker publish skipped: %s", exc)


def _publish_servo_markers(
    context: Context,
    topic: str,
    frame_id: str,
    target: BasePose2D,
    current: BasePose2D,
    projected: BasePose2D,
    *,
    blocked: BasePose2D | None = None,
) -> None:
    try:
        markers = [
            _base_arrow_marker(frame_id, target, ns="servo_target", marker_id=0, color=(0.1, 0.8, 0.2, 1.0), scale=0.28),
            _base_arrow_marker(frame_id, current, ns="servo_current", marker_id=1, color=(0.1, 0.35, 0.95, 1.0), scale=0.24),
            _base_arrow_marker(frame_id, projected, ns="servo_projected", marker_id=2, color=(1.0, 0.8, 0.05, 1.0), scale=0.22),
        ]
        if blocked is not None:
            markers.append(_base_arrow_marker(frame_id, blocked, ns="servo_blocked", marker_id=3, color=(0.9, 0.05, 0.05, 1.0), scale=0.30))
        _publish_marker_array(context, topic, markers)
    except Exception as exc:  # noqa: BLE001
        log.debug("Servo marker publish skipped: %s", exc)


def _publish_marker_array(context: Context, topic: str, markers: list[Any]) -> None:
    publisher = _marker_publisher(context, topic)
    if publisher is None:
        return
    try:
        from visualization_msgs.msg import MarkerArray

        msg = MarkerArray()
        msg.markers = markers
        publisher.publish(msg)
    except Exception as exc:  # noqa: BLE001
        log.debug("Marker publish skipped for %s: %s", topic, exc)


def _pose_axis_markers(
    frame_id: str,
    pose_mat: np.ndarray,
    *,
    ns: str,
    start_id: int,
    scale: float,
    selected: bool,
    alpha: float = 1.0,
) -> list[Any]:
    pose = np.asarray(pose_mat, dtype=np.float64)
    origin = pose[:3, 3]
    colors = [(1.0, 0.05, 0.05, alpha), (0.05, 0.75, 0.1, alpha), (0.1, 0.25, 1.0, alpha)]
    axes = [pose[:3, 0], pose[:3, 1], pose[:3, 2]]
    markers: list[Any] = []
    for offset, (axis, color) in enumerate(zip(axes, colors, strict=False)):
        markers.append(
            _arrow_points_marker(
                frame_id,
                origin,
                origin + np.asarray(axis, dtype=np.float64) * float(scale),
                ns=ns,
                marker_id=start_id + offset,
                color=color,
                shaft_diameter=0.015 if selected else 0.008,
                head_diameter=0.035 if selected else 0.02,
            )
        )
    return markers


def _base_arrow_marker(
    frame_id: str,
    base: BasePose2D,
    *,
    ns: str,
    marker_id: int,
    color: tuple[float, float, float, float],
    scale: float,
) -> Any:
    start = np.array([base.x, base.y, 0.035], dtype=np.float64)
    end = start + np.array([math.cos(base.yaw) * float(scale), math.sin(base.yaw) * float(scale), 0.0], dtype=np.float64)
    return _arrow_points_marker(
        frame_id,
        start,
        end,
        ns=ns,
        marker_id=marker_id,
        color=color,
        shaft_diameter=0.035,
        head_diameter=0.075,
    )


def _arrow_points_marker(
    frame_id: str,
    start: np.ndarray,
    end: np.ndarray,
    *,
    ns: str,
    marker_id: int,
    color: tuple[float, float, float, float],
    shaft_diameter: float,
    head_diameter: float,
) -> Any:
    from geometry_msgs.msg import Point
    from visualization_msgs.msg import Marker

    marker = Marker()
    _fill_marker_header(marker, frame_id, ns, marker_id)
    marker.type = Marker.ARROW
    marker.action = getattr(Marker, "ADD", 0)
    marker.points = [_point(start, Point), _point(end, Point)]
    marker.scale.x = float(shaft_diameter)
    marker.scale.y = float(head_diameter)
    marker.scale.z = float(head_diameter)
    _set_color(marker, color)
    return marker


def _text_marker(frame_id: str, pose_mat: np.ndarray, *, ns: str, marker_id: int, text: str) -> Any:
    from visualization_msgs.msg import Marker

    marker = Marker()
    _fill_marker_header(marker, frame_id, ns, marker_id)
    marker.type = getattr(Marker, "TEXT_VIEW_FACING", 9)
    marker.action = getattr(Marker, "ADD", 0)
    marker.pose.position.x = float(pose_mat[0, 3])
    marker.pose.position.y = float(pose_mat[1, 3])
    marker.pose.position.z = float(pose_mat[2, 3]) + 0.025
    marker.scale.z = 0.035
    marker.text = text
    _set_color(marker, (1.0, 1.0, 1.0, 0.9))
    return marker


def _fill_marker_header(marker: Any, frame_id: str, ns: str, marker_id: int) -> None:
    marker.header.frame_id = frame_id
    marker.ns = ns
    marker.id = int(marker_id)
    marker.lifetime.sec = 0
    marker.lifetime.nanosec = 0


def _set_color(marker: Any, color: tuple[float, float, float, float]) -> None:
    marker.color.r = float(color[0])
    marker.color.g = float(color[1])
    marker.color.b = float(color[2])
    marker.color.a = float(color[3])


def _point(value: np.ndarray, point_cls: Any) -> Any:
    point = point_cls()
    point.x = float(value[0])
    point.y = float(value[1])
    point.z = float(value[2])
    return point


def _planar_marker_pose(base: BasePose2D) -> np.ndarray:
    mat = planar_pose_matrix(base.x, base.y, base.yaw)
    mat[2, 3] = 0.05
    return mat


def _parse_tmc_ik_response(response: Any, *, default_frame_id: str = "odom") -> list[dict[str, Any]]:
    raw = response.get("response", response) if isinstance(response, dict) else response
    results = list(getattr(raw, "ik_results", []) or getattr(raw, "solutions", []) or [])
    parsed: list[dict[str, Any]] = []
    for index, item in enumerate(results):
        names = [str(name) for name in getattr(item, "joint_names", [])]
        positions = [float(value) for value in getattr(item, "joint_positions", [])]
        base = _base_pose_from_pose_msg(getattr(item, "origin_to_base", None), default_frame_id=default_frame_id)
        if not names or len(names) != len(positions) or base is None:
            continue
        parsed.append(
            {
                "index": index,
                "joint_positions": dict(zip(names, positions, strict=False)),
                "origin_to_base": base.to_dict(),
            }
        )
    return parsed


def _planner_state_validity(planner: Any, joint_state: Any, *, context: Context, group_name: str) -> Any:
    if hasattr(planner, "check_state_validity"):
        return planner.check_state_validity(joint_state, context=context, group_name=group_name)
    from src.core.planners import PlannerResult

    return PlannerResult(success=False, reason="state_validity_not_supported", metadata={"error": "unsupported"})


def _servo_collision_allowed(context: Context, candidate: BasePose2D, *, target: str = "") -> dict[str, Any]:
    costmap = _costmap_candidate_allowed(context, candidate)
    if not costmap.get("allowed", True):
        return costmap
    pcd = _pointcloud_candidate_allowed(context, candidate, target=target)
    if not pcd.get("allowed", True):
        return pcd
    return {"allowed": True, "costmap": costmap, "pointcloud": pcd}


def _costmap_candidate_allowed(context: Context, candidate: BasePose2D) -> dict[str, Any]:
    return {"allowed": True, "reason": "missing_sensor_data_center", "missing_policy": "allow"}
    sdc = context.sensor_data_center
    if sdc is None:
        return {"allowed": True, "reason": "missing_sensor_data_center", "missing_policy": "allow"}
    nav_cfg = {}
    if isinstance(context.hardware, dict):
        nav_cfg = context.hardware.get("manipulation", {}) if isinstance(context.hardware.get("manipulation"), dict) else {}
    checker = NavigationFeasibilityChecker(
        sdc,
        NavFeasibilityConfig(
            local_costmap_sensor=str(nav_cfg.get("local_costmap_sensor", "local_costmap")),
            map_sensor="",
            footprint_sensor=str(nav_cfg.get("footprint_sensor", "local_footprint")),
            missing_policy="allow_with_reason",
            timeout_s=float(nav_cfg.get("costmap_timeout_s", 0.2)),
            footprint_inflation_m=float(nav_cfg.get("servo_footprint_inflation_m", 0.03)),
            near_goal_radius_m=float(nav_cfg.get("servo_near_goal_radius_m", 0.0)),
        ),
    )
    result = checker.check_candidate(candidate)
    return {"allowed": bool(result.get("feasible", result.get("success", True))), **result}


def _pointcloud_candidate_allowed(context: Context, candidate: BasePose2D, *, target: str = "") -> dict[str, Any]:
    return {"allowed": True, "reason": "pointcloud_gate_disabled"}
    sdc = context.sensor_data_center
    cfg = context.hardware.get("manipulation", {}) if isinstance(context.hardware, dict) and isinstance(context.hardware.get("manipulation"), dict) else {}
    if not bool(cfg.get("servo_use_pointcloud_gate", True)):
        return {"allowed": True, "reason": "pointcloud_gate_disabled"}
    points_blocks: list[np.ndarray] = []
    frame_blocks: list[str] = []
    ws = context.world_state
    if ws is not None and target:
        movable = _movable_object_state(ws, target)
        surround_key = getattr(movable, "surround_pcd_key", None) or f"{target}_surround_pcd"
        surround = ws.get(str(surround_key))
        if surround is not None:
            try:
                points_blocks.append(_as_points_array(surround, str(surround_key)))
                frame_blocks.append(str(getattr(movable, "frame_id", None) or ws.get(f"{target}_pcd_frame") or candidate.frame_id))
            except Exception as exc:  # noqa: BLE001
                log.debug("Ignoring invalid surround PCD for servo gate: %s", exc)
    if sdc is not None:
        try:
            msg = sdc.get_latest(str(cfg.get("servo_pointcloud_sensor", "head_pointcloud")), timeout_s=float(cfg.get("servo_pointcloud_timeout_s", 0.5)))
            data = getattr(msg, "data", msg)
            live_points, live_frame = _points_from_pointcloud_message(sdc, data)
            if live_points is not None and live_points.size:
                points_blocks.append(live_points)
                frame_blocks.append(live_frame or candidate.frame_id)
        except Exception as exc:  # noqa: BLE001
            log.debug("Live pointcloud unavailable for servo gate: %s", exc)
    if not points_blocks:
        return {"allowed": True, "reason": "pointcloud_unavailable"}
    transformed_blocks: list[np.ndarray] = []
    for points, frame_id in zip(points_blocks, frame_blocks, strict=False):
        if sdc is not None and frame_id != candidate.frame_id:
            try:
                transformed_blocks.append(sdc.transform_data(points, frame_id, candidate.frame_id))
                continue
            except Exception as exc:  # noqa: BLE001
                log.debug("Unable to transform pointcloud for servo gate: %s", exc)
        transformed_blocks.append(points)
    points_world = np.vstack([block for block in transformed_blocks if block.size])
    if points_world.size == 0:
        return {"allowed": True, "reason": "pointcloud_empty_after_transform"}
    local = _points_in_planar_base(points_world, candidate)
    front = float(cfg.get("servo_footprint_front_m", 0.55))
    back = float(cfg.get("servo_footprint_back_m", 0.45))
    half_width = float(cfg.get("servo_footprint_half_width_m", 0.35))
    min_z = float(cfg.get("servo_pointcloud_min_z_m", 0.02))
    max_z = float(cfg.get("servo_pointcloud_max_z_m", 1.20))
    mask = (
        (local[:, 0] >= -back)
        & (local[:, 0] <= front)
        & (np.abs(local[:, 1]) <= half_width)
        & (local[:, 2] >= min_z)
        & (local[:, 2] <= max_z)
    )
    blocked_count = int(np.count_nonzero(mask))
    if blocked_count:
        blocked = local[mask][:10].astype(float).tolist()
        return {
            "allowed": False,
            "reason": "pointcloud_footprint_blocked",
            "blocked_count": blocked_count,
            "sample_blocked_points_base": blocked,
            "candidate": candidate.to_dict(),
        }
    return {"allowed": True, "reason": "pointcloud_clear", "checked_points": int(local.shape[0])}


def _points_from_pointcloud_message(sensor_data_center: Any, data: Any) -> tuple[np.ndarray | None, str | None]:
    if data is None:
        return None, None
    if isinstance(data, np.ndarray) or isinstance(data, list):
        return _as_points_array(data, "head_pointcloud"), None
    width = int(getattr(data, "width", 0) or 0)
    height = int(getattr(data, "height", 0) or 0)
    point_step = int(getattr(data, "point_step", 0) or 0)
    row_step = int(getattr(data, "row_step", 0) or 0)
    raw = getattr(data, "data", None)
    if width <= 0 or height <= 0 or point_step <= 0 or row_step <= 0 or raw is None:
        return None, None
    max_samples = 1500
    stride = max(1, int(math.sqrt(max(1, (width * height) // max_samples))))
    points: list[tuple[float, float, float]] = []
    for v in range(0, height, stride):
        for u in range(0, width, stride):
            decoded = sensor_data_center.decode_pcd(data, u=u, v=v) if hasattr(sensor_data_center, "decode_pcd") else None
            if decoded is None:
                continue
            x, y, z, _color = decoded
            points.append((float(x), float(y), float(z)))
    frame_id = getattr(getattr(data, "header", None), "frame_id", None)
    return (np.asarray(points, dtype=np.float64), str(frame_id) if frame_id else None) if points else (None, frame_id)


def _points_in_planar_base(points: np.ndarray, base: BasePose2D) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    dx = points[:, 0] - base.x
    dy = points[:, 1] - base.y
    c = math.cos(-base.yaw)
    s = math.sin(-base.yaw)
    local = np.empty_like(points[:, :3])
    local[:, 0] = c * dx - s * dy
    local[:, 1] = s * dx + c * dy
    local[:, 2] = points[:, 2]
    return local


def _object_and_surround_points(ws: Any, target: str, movable: Any | None) -> tuple[np.ndarray | None, np.ndarray | None]:
    object_points = None
    surround_points = None
    object_key = getattr(movable, "object_pcd_key", None) or f"{target}_object_pcd"
    surround_key = getattr(movable, "surround_pcd_key", None) or f"{target}_surround_pcd"
    for key, attr in ((object_key, "object"), (surround_key, "surround")):
        value = ws.get(str(key)) if ws is not None else None
        if value is None:
            continue
        try:
            points = _as_points_array(value, str(key))
        except Exception as exc:  # noqa: BLE001
            log.debug("Ignoring invalid %s PCD for %s: %s", attr, target, exc)
            continue
        if attr == "object":
            object_points = points
        else:
            surround_points = points
    return object_points, surround_points


def _pca_grasp_analysis(points: np.ndarray, *, robot_pose: BasePose2D) -> dict[str, Any]:
    points = _as_points_array(points, "object_pcd")
    xyz = points[:, :3].astype(np.float64)
    centroid = np.mean(xyz, axis=0)
    bounds_min = np.percentile(xyz, 5.0, axis=0)
    bounds_max = np.percentile(xyz, 95.0, axis=0)
    xy = xyz[:, :2]
    xy_center = np.mean(xy, axis=0)
    centered = xy - xy_center
    if centered.shape[0] < 3 or float(np.linalg.norm(centered)) < 1e-9:
        long_axis_xy = np.array([1.0, 0.0], dtype=np.float64)
        short_axis_xy = np.array([0.0, 1.0], dtype=np.float64)
        eigenvalues = np.array([0.0, 0.0], dtype=np.float64)
    else:
        cov = np.cov(centered.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]
        long_axis_xy = _normalize_xy(eigenvectors[:, 0], fallback=np.array([1.0, 0.0], dtype=np.float64))
        short_axis_xy = _normalize_xy(eigenvectors[:, 1], fallback=np.array([0.0, 1.0], dtype=np.float64))
    if short_axis_xy[0] < 0.0 or (abs(float(short_axis_xy[0])) < 1e-9 and short_axis_xy[1] < 0.0):
        short_axis_xy = -short_axis_xy
    if float(np.cross(long_axis_xy, short_axis_xy)) < 0.0:
        long_axis_xy = -long_axis_xy
    long_extent = _robust_extent_along_axis(xy, long_axis_xy)
    short_extent = _robust_extent_along_axis(xy, short_axis_xy)
    max_extent = max(long_extent, short_extent, 1e-9)
    min_extent = max(min(long_extent, short_extent), 1e-9)
    robot_xy = np.array([robot_pose.x, robot_pose.y], dtype=np.float64)
    to_object = _normalize_xy(centroid[:2] - robot_xy, fallback=np.array([1.0, 0.0], dtype=np.float64))
    log.warn("Long extent: %f, Short extent: %f", long_extent, short_extent)
    return {
        "centroid": centroid.astype(float).tolist(),
        "top_center": [float(centroid[0]), float(centroid[1]), float(bounds_max[2])],
        "bounds_min": bounds_min.astype(float).tolist(),
        "bounds_max": bounds_max.astype(float).tolist(),
        "height_m": float(max(0.0, bounds_max[2] - bounds_min[2])),
        "long_axis_xy": long_axis_xy.astype(float).tolist(),
        "short_axis_xy": short_axis_xy.astype(float).tolist(),
        "horizontal_extents_m": {
            "long": float(long_extent),
            "short": float(short_extent),
        },
        "horizontal_extent_ratio": float(max_extent / min_extent),
        "horizontal_eigenvalues": eigenvalues.astype(float).tolist(),
        "robot_to_object_xy": to_object.astype(float).tolist(),
    }


def _top_down_pca_grasp_pose(analysis: dict[str, Any]) -> tuple[np.ndarray, float, str]:
    x_axis_xy = _normalize_xy(np.asarray(analysis["short_axis_xy"], dtype=np.float64), fallback=np.array([1.0, 0.0], dtype=np.float64))
    x_axis = np.array([x_axis_xy[0], x_axis_xy[1], 0.0], dtype=np.float64)
    z_axis = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    y_axis = _normalize3(np.cross(z_axis, x_axis), fallback=np.array([0.0, 1.0, 0.0], dtype=np.float64))
    x_axis = _normalize3(np.cross(y_axis, z_axis), fallback=x_axis)
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = np.column_stack((x_axis, y_axis, z_axis))
    # z_offset = float(analysis["height_m"]) * 0.5 + 0.02
    offset = np.array([0.02, 0.02, 0.0], dtype=np.float64)
    pose[:3, 3] = np.asarray(analysis["top_center"], dtype=np.float64) + offset
    width = float(analysis["horizontal_extents_m"]["short"])
    return pose, width, "top_down"


def _short_axis_side_grasp_pose(analysis: dict[str, Any]) -> tuple[np.ndarray, float, str]:
    # Short-axis refers to the span between the fingers, not insertion direction.
    # Keep the legacy lateral/up/approach basis consumed by both pickup workflows.
    long_axis_xy = _normalize_xy(np.asarray(analysis["long_axis_xy"], dtype=np.float64), fallback=np.array([1.0, 0.0], dtype=np.float64))
    to_object_xy = _normalize_xy(np.asarray(analysis["robot_to_object_xy"], dtype=np.float64), fallback=long_axis_xy)
    if float(np.dot(long_axis_xy, to_object_xy)) < 0.0:
        long_axis_xy = -long_axis_xy
    approach = np.array([long_axis_xy[0], long_axis_xy[1], 0.0], dtype=np.float64)
    up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    x_axis = _normalize3(np.cross(approach, up), fallback=np.array([0.0, -1.0, 0.0], dtype=np.float64))
    y_axis = up
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = np.column_stack((x_axis, y_axis, approach))
    pose[:3, 3] = np.asarray(analysis["centroid"], dtype=np.float64)
    width = float(analysis["horizontal_extents_m"]["short"])
    return pose, width, "short_axis_side"


def _pca_grasp_candidates(analysis: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "strategy": "top_down",
            "width_m": float(analysis["horizontal_extents_m"]["short"]),
            "origin": analysis["top_center"],
        },
        {
            "strategy": "short_axis_side",
            "width_m": float(analysis["horizontal_extents_m"]["short"]),
            "origin": analysis["centroid"],
        },
    ]


def _robust_extent_along_axis(points_xy: np.ndarray, axis_xy: np.ndarray) -> float:
    axis = _normalize_xy(axis_xy, fallback=np.array([1.0, 0.0], dtype=np.float64))
    projections = np.asarray(points_xy, dtype=np.float64) @ axis
    return float(max(0.0, np.percentile(projections, 95.0) - np.percentile(projections, 5.0)))


def _normalize_xy(value: np.ndarray, *, fallback: np.ndarray) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)[:2]
    norm = float(np.linalg.norm(vector))
    if norm < 1e-9:
        return np.asarray(fallback, dtype=np.float64)[:2]
    return (vector / norm).astype(np.float64)


def _normalize3(value: np.ndarray, *, fallback: np.ndarray) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)[:3]
    norm = float(np.linalg.norm(vector))
    if norm < 1e-9:
        return np.asarray(fallback, dtype=np.float64)[:3]
    return (vector / norm).astype(np.float64)


def _estimate_grasp_scene_parameters(
    grasp_pose: np.ndarray,
    *,
    approach_axis: str,
    object_points: np.ndarray | None,
    surround_points: np.ndarray | None,
    clearance_m: float = 0.03,
    min_insertion_m: float = 0.04,
    max_insertion_m: float = 0.16,
    depth_axis: np.ndarray | None = None,
) -> dict[str, Any]:
    grasp = np.asarray(grasp_pose, dtype=np.float64)
    approach = _approach_vector_from_pose(grasp, approach_axis)
    depth_direction = np.asarray(depth_axis, dtype=np.float64) if depth_axis is not None else approach
    depth_norm = float(np.linalg.norm(depth_direction))
    if depth_norm < 1e-9:
        depth_direction = approach
    else:
        depth_direction = (depth_direction / depth_norm).astype(np.float64)
    horizontal_depth = depth_direction.copy()
    horizontal_depth[2] = 0.0
    horizontal_norm = float(np.linalg.norm(horizontal_depth))
    is_vertical_grasp = horizontal_norm < 0.25
    if not is_vertical_grasp:
        horizontal_depth = (horizontal_depth / horizontal_norm).astype(np.float64)
    lateral_axis = np.array([-horizontal_depth[1], horizontal_depth[0], 0.0], dtype=np.float64) if not is_vertical_grasp else np.array([0.0, 1.0, 0.0], dtype=np.float64)
    scene: dict[str, Any] = {
        "clearance_m": float(clearance_m),
        "min_insertion_m": float(min_insertion_m),
        "max_insertion_m": float(max_insertion_m),
        "min_reach_m": 0.30 if is_vertical_grasp else 0.32,
        "max_reach_m": 0.68 if is_vertical_grasp else 0.82,
        "reach_margin_m": 0.08,
        "reach_tolerance_m": 0.18,
        "depth_axis": depth_direction.astype(float).tolist(),
        "projected_depth_axis": horizontal_depth.astype(float).tolist() if not is_vertical_grasp else [0.0, 0.0, float(np.sign(depth_direction[2]) or 1.0)],
        "lateral_axis": lateral_axis.astype(float).tolist(),
        "is_vertical_grasp": bool(is_vertical_grasp),
        "grasp_motion_type": "vertical_z" if is_vertical_grasp else "horizontal_projected",
    }
    if object_points is not None and object_points.size:
        object_xyz = np.asarray(object_points[:, :3], dtype=np.float64)
        object_xyz = object_xyz[np.all(np.isfinite(object_xyz), axis=1)]
    else:
        object_xyz = np.empty((0, 3), dtype=np.float64)
    if object_xyz.size:
        rel = object_xyz - grasp[:3, 3]
        depth_values = rel @ (depth_direction if is_vertical_grasp else horizontal_depth)
        lateral_values = rel @ lateral_axis
        mins = np.nanmin(object_xyz, axis=0)
        maxs = np.nanmax(object_xyz, axis=0)
        object_depth = float(np.nanmax(depth_values) - np.nanmin(depth_values))
        scene.update(
            {
                "object_bounds_min": mins.astype(float).tolist(),
                "object_bounds_max": maxs.astype(float).tolist(),
                "object_depth_min_m": float(np.nanmin(depth_values)),
                "object_depth_max_m": float(np.nanmax(depth_values)),
                "object_depth_m": object_depth,
                "object_lateral_min_m": float(np.nanmin(lateral_values)),
                "object_lateral_max_m": float(np.nanmax(lateral_values)),
                "object_height_m": float(maxs[2] - mins[2]),
                "object_width_m": float(max(maxs[0] - mins[0], maxs[1] - mins[1])),
                "object_top_z_m": float(maxs[2]),
            }
        )
    support_depth = None
    support_edge_along = None
    if surround_points is not None and surround_points.size:
        surround_xyz = np.asarray(surround_points[:, :3], dtype=np.float64)
        surround_xyz = surround_xyz[np.all(np.isfinite(surround_xyz), axis=1)]
    else:
        surround_xyz = np.empty((0, 3), dtype=np.float64)
    if surround_xyz.size and not is_vertical_grasp:
        rel = surround_xyz - grasp[:3, 3]
        along = rel @ horizontal_depth
        lateral = np.abs(rel @ lateral_axis)
        support_z = float(scene.get("object_bounds_min", [0.0, 0.0, grasp[2, 3] - 0.08])[2])
        z_window = (surround_xyz[:, 2] > support_z - 0.08) & (surround_xyz[:, 2] < float(grasp[2, 3]) + 0.12)
        behind_mask = (along < 0.0) & (along > -0.85) & (lateral < 0.35) & z_window
        ahead_mask = (along >= 0.0) & (along < 0.85) & (lateral < 0.35) & z_window
        if np.any(behind_mask):
            support_depth = abs(float(np.nanmin(along[behind_mask])))
            support_edge_along = float(np.nanmin(along[behind_mask]))
            scene.update(
                {
                    "support_depth_m": support_depth,
                    "support_depth_min_m": float(np.nanmin(along[behind_mask])),
                    "support_depth_max_m": float(np.nanmax(along[behind_mask])),
                    "support_point_count": int(np.count_nonzero(behind_mask)),
                }
            )
        if np.any(ahead_mask):
            scene.update(
                {
                    "front_support_extent_m": float(np.nanmax(along[ahead_mask])),
                    "front_support_point_count": int(np.count_nonzero(ahead_mask)),
                }
            )
    object_depth = float(scene.get("object_depth_m", min_insertion_m))
    geometric = object_depth + float(clearance_m)
    if support_depth is not None:
        geometric = min(geometric, max(0.0, support_depth - float(clearance_m)))
    suggested = _clamp(geometric, float(min_insertion_m), float(max_insertion_m))
    scene["suggested_insertion_m"] = float(suggested)
    if is_vertical_grasp:
        required_lift = _clamp(float(grasp[2, 3]), 0.0, 0.35)
        scene.update(
            {
                "required_lift_m": float(required_lift),
                "lift_coupling_mode": "vertical_pregrasp_z",
                "recommended_reach_min_m": float(scene["min_reach_m"]),
                "recommended_reach_max_m": float(scene["max_reach_m"]),
            }
        )
    else:
        reach_margin = float(scene["reach_margin_m"])
        reach_tolerance = float(scene["reach_tolerance_m"])
        fallback_depth = max(float(scene.get("object_depth_m", 0.0)), float(scene.get("suggested_insertion_m", 0.0)))
        if support_edge_along is not None:
            support_edge_to_grasp = abs(float(support_edge_along))
            support_edge_source = "surround_projected_edge"
        else:
            support_edge_to_grasp = fallback_depth
            support_edge_source = "object_projected_depth"
        support_edge_to_pregrasp = max(0.0, support_edge_to_grasp - float(suggested))
        required_reach_min = _clamp(support_edge_to_pregrasp + reach_margin, 0.25, 0.78)
        required_reach_max = _clamp(required_reach_min + reach_tolerance, required_reach_min + 0.05, 0.88)
        scene.update(
            {
                "scene_depth_for_reach_m": float(support_edge_to_grasp),
                "support_edge_source": support_edge_source,
                "support_edge_along_m": float(support_edge_along) if support_edge_along is not None else None,
                "support_edge_to_grasp_m": float(support_edge_to_grasp),
                "support_edge_to_pregrasp_m": float(support_edge_to_pregrasp),
                "required_reach_min_m": float(required_reach_min),
                "required_reach_max_m": float(required_reach_max),
                "recommended_reach_min_m": float(required_reach_min),
                "recommended_reach_max_m": float(required_reach_max),
                "min_reach_m": float(required_reach_min),
                "max_reach_m": float(required_reach_max),
                "lift_coupling_mode": "reach_linear",
                "lift_reach_threshold_m": 0.20,
                "lift_at_short_reach_m": 0.0,
                "lift_per_reach_m": 0.9,
            }
        )
    log.debug("Grasp scene parameters for", scene)
    return scene


def _min_lift_for_reach(
    reach_m: float,
    scene: dict[str, Any] | None,
    *,
    arm_lift_min: float = 0.0,
    arm_lift_max: float = 0.35,
) -> float:
    data = scene or {}
    if bool(data.get("is_vertical_grasp", False)) or str(data.get("lift_coupling_mode", "")).lower() == "vertical_pregrasp_z":
        return float(_clamp(float(data.get("required_lift_m", 0.0)), arm_lift_min, arm_lift_max))
    threshold = float(data.get("lift_reach_threshold_m", 0.42))
    base_lift = float(data.get("lift_at_short_reach_m", arm_lift_min))
    slope = float(data.get("lift_per_reach_m", 0.9))
    required = base_lift + max(0.0, float(reach_m) - threshold) * slope
    return float(_clamp(required, arm_lift_min, arm_lift_max))


def _fk_sweep_clearance(
    base_pose: BasePose2D,
    hand_pose_base: np.ndarray,
    *,
    surround_points: np.ndarray | None,
    approach_axis: str,
) -> dict[str, Any]:
    if surround_points is None or not surround_points.size:
        return {"allowed": True, "reason": "surround_pcd_unavailable", "clearance_penalty": 0.0}
    local = _points_in_planar_base(surround_points, base_pose)
    hand = np.asarray(hand_pose_base, dtype=np.float64)
    hand_x = max(0.05, float(hand[0, 3]))
    hand_z = float(hand[2, 3])
    half_width = 0.18
    z_margin = 0.16
    front_mask = (
        (local[:, 0] > 0.05)
        & (local[:, 0] < hand_x + 0.08)
        & (np.abs(local[:, 1]) < half_width)
        & (local[:, 2] > max(0.02, hand_z - z_margin))
        & (local[:, 2] < hand_z + z_margin)
    )
    blocked = int(np.count_nonzero(front_mask))
    if blocked:
        nearest = float(np.nanmin(local[front_mask, 0]))
        if nearest < max(0.12, hand_x - 0.20):
            return {
                "allowed": False,
                "reason": "fk_sweep_blocked",
                "blocked_count": blocked,
                "nearest_blocked_x_m": nearest,
                "approach_axis": approach_axis,
            }
        return {
            "allowed": True,
            "reason": "fk_sweep_near_surround",
            "blocked_count": blocked,
            "nearest_blocked_x_m": nearest,
            "clearance_penalty": 0.5,
            "approach_axis": approach_axis,
        }
    return {"allowed": True, "reason": "fk_sweep_clear", "clearance_penalty": 0.0, "approach_axis": approach_axis}


def _resolve_grasp_motion_poses(ws: Any, target: str, prefix: str) -> tuple[np.ndarray, np.ndarray, str, str]:
    grasp_pose = ws.get(f"{prefix}_grasp_pose")
    if grasp_pose is None:
        grasp_pose = ws.get(f"{target}_estimated_pick_pose_mat")
    pregrasp_pose = ws.get(f"{prefix}_pregrasp_pose")
    if pregrasp_pose is None:
        pregrasp_pose = grasp_pose
    if grasp_pose is None or pregrasp_pose is None:
        raise RuntimeError("missing_grasp_motion_pose")
    movable = _movable_object_state(ws, target)
    metadata = ws.get(f"{prefix}_grasp_motion_metadata") or ws.get(f"{target}_simple_grasp_metadata") or {}
    frame_id = str(metadata.get("frame_id") or getattr(movable, "frame_id", None) or ws.get(f"{target}_pcd_frame") or "odom")
    approach_axis = str(metadata.get("approach_axis") or "+z")
    return np.asarray(grasp_pose, dtype=np.float64), np.asarray(pregrasp_pose, dtype=np.float64), frame_id, approach_axis


def _selected_hand_pose_base(ws: Any, prefix: str) -> np.ndarray | None:
    metadata = ws.get(f"{prefix}_fk_posture_metadata") or {}
    selected = metadata.get("selected") if isinstance(metadata, dict) else None
    hand = selected.get("hand_pose_base") if isinstance(selected, dict) else None
    if hand is None:
        candidates = ws.get(f"{prefix}_fk_posture_candidates") or []
        if candidates:
            hand = candidates[0].get("hand_pose_base")
    return np.asarray(hand, dtype=np.float64) if hand is not None else None


def _selected_grasp_base_pose(ws: Any, target: str, source: str = "auto") -> tuple[BasePose2D, str]:
    if ws is None:
        raise RuntimeError("missing_world_state")
    normalized = str(source or "auto").strip().lower()
    if normalized == "fk":
        keys = [f"{target}_fk_base_pose"]
    elif normalized == "tmc":
        keys = [f"{target}_tmc_base_pose"]
    else:
        keys = [f"{target}_fk_base_pose", f"{target}_tmc_base_pose"]
    for key in keys:
        value = ws.get(key)
        if value is None:
            continue
        return BasePose2D.from_mapping(value, default_frame_id="odom"), key
    raise RuntimeError(f"missing_selected_grasp_base_pose:{','.join(keys)}")


def _interpolate_pose_matrices(start_pose: np.ndarray, goal_pose: np.ndarray, waypoint_count: int) -> list[np.ndarray]:
    start = np.asarray(start_pose, dtype=np.float64)
    goal = np.asarray(goal_pose, dtype=np.float64)
    count = max(1, int(waypoint_count))
    waypoints: list[np.ndarray] = []
    for index in range(1, count + 1):
        t = float(index) / float(count)
        pose = goal.copy()
        pose[:3, 3] = (1.0 - t) * start[:3, 3] + t * goal[:3, 3]
        waypoints.append(pose.astype(np.float64))
    return waypoints


def _pose3d_from_matrix(pose_mat: np.ndarray, frame_id: str) -> Pose3D:
    x, y, z, roll, pitch, yaw = matrix_to_xyz_rpy(np.asarray(pose_mat, dtype=np.float64))
    return Pose3D(x=x, y=y, z=z, roll=roll, pitch=pitch, yaw=yaw, frame_id=frame_id)


def _final_base_pose_from_pregrasp_delta(
    ws: Any,
    prefix: str,
    grasp_pose: np.ndarray,
    pregrasp_pose: np.ndarray,
    frame_id: str,
) -> BasePose2D | None:
    value = ws.get(f"{prefix}_tmc_base_pose")
    if value is None:
        value = ws.get(f"{prefix}_fk_base_pose")
    if value is None:
        return None
    base = BasePose2D.from_mapping(value, default_frame_id=frame_id)
    delta = np.asarray(grasp_pose, dtype=np.float64)[:2, 3] - np.asarray(pregrasp_pose, dtype=np.float64)[:2, 3]
    return BasePose2D(
        x=float(base.x + delta[0]),
        y=float(base.y + delta[1]),
        yaw=float(base.yaw),
        frame_id=base.frame_id or frame_id,
    )


def _project_base_pose(current: BasePose2D, cmd: ServoCommand, dt: float) -> BasePose2D:
    c = math.cos(current.yaw)
    s = math.sin(current.yaw)
    dx = (c * cmd.vx - s * cmd.vy) * float(dt)
    dy = (s * cmd.vx + c * cmd.vy) * float(dt)
    return BasePose2D(
        x=current.x + dx,
        y=current.y + dy,
        yaw=wrap_angle(current.yaw + cmd.wz * float(dt)),
        frame_id=current.frame_id,
    )


def _base_pose_from_hand_alignment(
    grasp_pose: np.ndarray,
    hand_pose_base: np.ndarray,
    *,
    frame_id: str,
    approach_axis: str = "+z",
) -> BasePose2D:
    grasp = np.asarray(grasp_pose, dtype=np.float64)
    hand = np.asarray(hand_pose_base, dtype=np.float64)
    grasp_yaw = _approach_yaw_from_pose(grasp, approach_axis)
    hand_yaw = _approach_yaw_from_pose(hand, approach_axis)
    base_yaw = wrap_angle(grasp_yaw - hand_yaw)
    c = math.cos(base_yaw)
    s = math.sin(base_yaw)
    hand_xy_world = np.array(
        [
            c * float(hand[0, 3]) - s * float(hand[1, 3]),
            s * float(hand[0, 3]) + c * float(hand[1, 3]),
        ],
        dtype=np.float64,
    )
    base_xy = grasp[:2, 3].astype(np.float64) - hand_xy_world
    return BasePose2D(x=float(base_xy[0]), y=float(base_xy[1]), yaw=base_yaw, frame_id=frame_id)


def _base_pose_from_fk_pregrasp_alignment(
    grasp_pose: np.ndarray,
    hand_pose_base: np.ndarray,
    *,
    frame_id: str,
    approach_axis: str = "+z",
    approach_standoff_m: float = 0.12,
) -> BasePose2D:
    grasp = np.asarray(grasp_pose, dtype=np.float64)
    hand = np.asarray(hand_pose_base, dtype=np.float64)
    grasp_yaw = _approach_yaw_from_pose(grasp, approach_axis)
    hand_yaw = _approach_yaw_from_pose(hand, approach_axis)
    base_yaw = wrap_angle(grasp_yaw - hand_yaw)
    c = math.cos(base_yaw)
    s = math.sin(base_yaw)
    hand_xy_world = np.array(
        [
            c * float(hand[0, 3]) - s * float(hand[1, 3]),
            s * float(hand[0, 3]) + c * float(hand[1, 3]),
        ],
        dtype=np.float64,
    )
    approach_xy = _approach_vector_from_pose(grasp, approach_axis)[:2].astype(np.float64)
    norm = float(np.linalg.norm(approach_xy))
    if norm < 1e-9:
        approach_xy = np.array([math.cos(grasp_yaw), math.sin(grasp_yaw)], dtype=np.float64)
    else:
        approach_xy = approach_xy / norm
    desired_hand_xy = grasp[:2, 3].astype(np.float64) - approach_xy * float(approach_standoff_m)
    base_xy = desired_hand_xy - hand_xy_world
    return BasePose2D(x=float(base_xy[0]), y=float(base_xy[1]), yaw=base_yaw, frame_id=frame_id)


def _base_pose_from_vertical_fk_pregrasp_alignment(
    grasp_pose: np.ndarray,
    hand_pose_base: np.ndarray,
    *,
    frame_id: str,
    current_base: BasePose2D | None,
) -> BasePose2D:
    grasp = np.asarray(grasp_pose, dtype=np.float64)
    hand = np.asarray(hand_pose_base, dtype=np.float64)
    base_yaw = float(current_base.yaw) if current_base is not None else 0.0
    c = math.cos(base_yaw)
    s = math.sin(base_yaw)
    hand_xy_world = np.array(
        [
            c * float(hand[0, 3]) - s * float(hand[1, 3]),
            s * float(hand[0, 3]) + c * float(hand[1, 3]),
        ],
        dtype=np.float64,
    )
    base_xy = grasp[:2, 3].astype(np.float64) - hand_xy_world
    return BasePose2D(x=float(base_xy[0]), y=float(base_xy[1]), yaw=base_yaw, frame_id=frame_id)


def _fk_candidate_errors(
    hand_pose: np.ndarray,
    target_pose: np.ndarray,
    *,
    hand_pose_base: np.ndarray | None,
    approach_axis: str,
) -> dict[str, float]:
    hand = np.asarray(hand_pose, dtype=np.float64)
    target = np.asarray(target_pose, dtype=np.float64)
    base_hand = np.asarray(hand_pose_base, dtype=np.float64) if hand_pose_base is not None else hand
    return {
        "signed_z_error_m": float(hand[2, 3]) - float(target[2, 3]),
        "z_error_m": abs(float(hand[2, 3]) - float(target[2, 3])),
        "lateral_error_m": abs(float(base_hand[1, 3])),
        "approach_error_rad": _approach_alignment_error(hand, target, approach_axis),
        "inplane_error_rad": _top_down_inplane_alignment_error(hand, target, approach_axis),
        "reach_m": float(base_hand[0, 3]),
    }


def _fk_candidate_gate(
    errors: dict[str, float],
    *,
    max_z_error_m: float,
    max_lateral_error_m: float,
    max_approach_error_rad: float,
    min_reach_m: float,
    max_reach_m: float,
) -> dict[str, Any]:
    z_error = float(errors.get("z_error_m", float("inf")))
    lateral_error = float(errors.get("lateral_error_m", float("inf")))
    approach_error = float(errors.get("approach_error_rad", float("inf")))
    reach = float(errors.get("reach_m", 0.0))
    reasons: list[str] = []
    if z_error > float(max_z_error_m):
        reasons.append("z_error")
    if lateral_error > float(max_lateral_error_m):
        reasons.append("lateral_error")
    if approach_error > float(max_approach_error_rad):
        reasons.append("approach_error")
    if reach < float(min_reach_m) or reach > float(max_reach_m):
        reasons.append("reach_range")
    error_norm = (
        z_error / max(1e-6, float(max_z_error_m))
        + lateral_error / max(1e-6, float(max_lateral_error_m))
        + approach_error / max(1e-6, float(max_approach_error_rad))
    )
    return {
        "allowed": not reasons,
        "reasons": reasons,
        "error_norm": float(error_norm),
        "limits": {
            "max_z_error_m": float(max_z_error_m),
            "max_lateral_error_m": float(max_lateral_error_m),
            "max_approach_error_rad": float(max_approach_error_rad),
            "min_reach_m": float(min_reach_m),
            "max_reach_m": float(max_reach_m),
        },
    }


def _pregrasp_backoff_direction(
    grasp_pose: np.ndarray,
    *,
    approach_axis: str,
    pregrasp_mode: str,
    near_vertical_xy_threshold: float = 0.25,
) -> tuple[np.ndarray, str]:
    approach = _approach_vector_from_pose(grasp_pose, approach_axis)
    if str(pregrasp_mode or "simple").strip().lower() != "horizontal_projected":
        return approach, "simple_axis"
    horizontal = approach.copy()
    horizontal[2] = 0.0
    horizontal_norm = float(np.linalg.norm(horizontal))
    if horizontal_norm < float(near_vertical_xy_threshold):
        z_sign = 1.0 if float(approach[2]) >= 0.0 else -1.0
        return np.array([0.0, 0.0, z_sign], dtype=np.float64), "vertical_z"
    return (horizontal / horizontal_norm).astype(np.float64), "horizontal_projected"


def _approach_yaw_from_pose(pose_mat: np.ndarray, approach_axis: str) -> float:
    pose = np.asarray(pose_mat, dtype=np.float64)
    vector = _approach_vector_from_pose(pose, approach_axis)[:2]
    norm = float(np.linalg.norm(vector))
    if norm < 1e-9:
        return 0.0
    return math.atan2(float(vector[1]), float(vector[0]))


def _approach_vector_from_pose(pose_mat: np.ndarray, approach_axis: str) -> np.ndarray:
    pose = np.asarray(pose_mat, dtype=np.float64)
    axis = str(approach_axis or "+z").strip().lower()
    if axis == "+x":
        vector = pose[:3, 0]
    elif axis == "-x":
        vector = -pose[:3, 0]
    elif axis == "-z":
        vector = -pose[:3, 2]
    else:
        vector = pose[:3, 2]
    norm = float(np.linalg.norm(vector))
    if norm < 1e-9:
        return np.array([1.0, 0.0, 0.0], dtype=np.float64)
    return (vector / norm).astype(np.float64)


def _approach_alignment_error(hand_pose: np.ndarray, target_pose: np.ndarray, approach_axis: str) -> float:
    hand_vec = _approach_vector_from_pose(hand_pose, approach_axis)
    target_vec = _approach_vector_from_pose(target_pose, approach_axis)
    dot = _clamp(float(np.dot(hand_vec, target_vec)), -1.0, 1.0)
    return float(math.acos(dot))


def _top_down_inplane_alignment_error(hand_pose: np.ndarray, target_pose: np.ndarray, approach_axis: str) -> float:
    if not _is_vertical_approach(target_pose, approach_axis):
        return 0.0
    hand_x = np.asarray(hand_pose, dtype=np.float64)[:3, 0].copy()
    target_x = np.asarray(target_pose, dtype=np.float64)[:3, 0].copy()
    hand_x[2] = 0.0
    target_x[2] = 0.0
    hand_norm = float(np.linalg.norm(hand_x))
    target_norm = float(np.linalg.norm(target_x))
    if hand_norm < 1e-9 or target_norm < 1e-9:
        return 0.0
    hand_x /= hand_norm
    target_x /= target_norm
    dot = abs(_clamp(float(np.dot(hand_x, target_x)), -1.0, 1.0))
    return float(math.acos(dot))


def _is_vertical_approach(pose_mat: np.ndarray, approach_axis: str, *, xy_threshold: float = 0.25) -> bool:
    approach = _approach_vector_from_pose(pose_mat, approach_axis)
    return float(np.linalg.norm(approach[:2])) < float(xy_threshold)


def _angle_samples(center: float, *, step: float, count: int) -> list[float]:
    half = max(0, int(count) // 2)
    values = [wrap_angle(float(center) + float(offset) * float(step)) for offset in range(-half, half + 1)]
    return list(dict.fromkeys(round(value, 8) for value in values))


def _wrist_flex_bias_for_approach(pose_mat: np.ndarray, approach_axis: str) -> float:
    approach = _approach_vector_from_pose(pose_mat, approach_axis)
    horizontal = math.hypot(float(approach[0]), float(approach[1]))
    pitch = math.atan2(float(approach[2]), max(1e-9, horizontal))
    # return float(_clamp(pitch, -0.8, 0.8))
    return float(_clamp(pitch, -1.57, 1.57))


def _fk_frame_matrix(context: Context, source: str, target: str) -> np.ndarray:
    if source == target:
        return np.eye(4)
    sdc = context.sensor_data_center
    transform = sdc.get_pose_at(source, target) if sdc is not None else None
    if transform is None:
        raise RuntimeError(f"fk_missing_transform:{source}->{target}")
    matrix = np.asarray(transform.get("mat") if transform.get("mat") is not None else pose_dict_to_matrix(transform), dtype=float)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise RuntimeError(f"fk_invalid_transform:{source}->{target}")
    if not np.allclose(matrix[:3, 2], [0, 0, 1], atol=1e-4):
        raise RuntimeError(f"fk_nonplanar_base_transform:{source}->{target}")
    return matrix


def _resolve_hand_pose_in_world(base_pose: BasePose2D, hand_pose_base: np.ndarray, *, base_height: float = 0.0) -> np.ndarray:
    base_mat = planar_pose_matrix(base_pose.x, base_pose.y, base_pose.yaw)
    base_mat[2, 3] = base_height
    return (base_mat @ np.asarray(hand_pose_base, dtype=np.float64)).astype(np.float64)


def _fk_pose_matrix(fk_result: Any) -> np.ndarray | None:
    metadata = getattr(fk_result, "metadata", {}) or {}
    pose = metadata.get("pose")
    if pose is not None:
        return np.asarray(pose, dtype=np.float64)
    response = metadata.get("response")
    if response is None:
        response = metadata.get("raw_response")
    poses = getattr(response, "pose_stamped", None) if response is not None else None
    if poses:
        return pose_dict_to_matrix(_pose_msg_to_dict(poses[0].pose)).astype(np.float64)
    return None


def _joint_state_msg(joints: dict[str, float]) -> Any:
    from sensor_msgs.msg import JointState

    msg = JointState()
    msg.name = list(joints)
    msg.position = [float(joints[name]) for name in msg.name]
    return msg


def _merged_joint_state_msg(
    context: Context,
    joints: dict[str, float],
    *,
    base_pose: BasePose2D | None = None,
) -> Any:
    merged: dict[str, float] = {}
    current = _current_joint_state_msg(context)
    if current is not None:
        for name, position in zip(getattr(current, "name", []) or [], getattr(current, "position", []) or [], strict=False):
            try:
                merged[str(name)] = float(position)
            except Exception:
                continue
    merged.update({str(name): float(value) for name, value in (joints or {}).items()})
    if base_pose is not None:
        merged.update({"odom_x": float(base_pose.x), "odom_y": float(base_pose.y), "odom_t": float(base_pose.yaw)})
    return _joint_state_msg(merged)


def _current_joint_state_msg(context: Context) -> Any | None:
    try:
        sdc = context.sensor_data_center
        if sdc is None:
            return None
        msg = sdc.peek_latest("joint_states", timeout_s=0.2, keep_active=True)
        return getattr(msg, "data", msg)
    except Exception:
        return None


def _current_base_pose(context: Context, frame_id: str) -> BasePose2D | None:
    try:
        sdc = context.sensor_data_center
        if sdc is not None:
            pose = sdc.get_robot_pose_global(global_frame=frame_id)
            if pose is not None:
                raw = pose.get("pose", pose) if isinstance(pose, dict) else pose
                return BasePose2D(
                    x=float(getattr(raw, "x", pose.get("x", 0.0) if isinstance(pose, dict) else 0.0)),
                    y=float(getattr(raw, "y", pose.get("y", 0.0) if isinstance(pose, dict) else 0.0)),
                    yaw=float(getattr(raw, "yaw", pose.get("yaw", pose.get("theta", 0.0)) if isinstance(pose, dict) else 0.0)),
                    frame_id=frame_id,
                )
        if context.nav is not None:
            pose = context.nav.get_current_pose()
            return BasePose2D(x=float(pose.x), y=float(pose.y), yaw=float(pose.yaw), frame_id=frame_id)
    except Exception as exc:  # noqa: BLE001
        log.debug("Unable to read current base pose: %s", exc)
    return None


def _hand_rotation_from_yaw(yaw: float, approach_axis: str) -> np.ndarray:
    approach = np.array([math.cos(yaw), math.sin(yaw), 0.0], dtype=np.float64)
    right = np.array([math.sin(yaw), -math.cos(yaw), 0.0], dtype=np.float64)
    up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    axis = str(approach_axis).strip().lower()
    if axis == "+x":
        cols = (approach, right, up)
    elif axis == "-x":
        cols = (-approach, -right, up)
    elif axis == "-z":
        cols = (right, up, -approach)
    else:
        cols = (right, up, approach)
    return np.column_stack(cols).astype(np.float64)


def _pose_msg_from_matrix(matrix: np.ndarray) -> Any:
    from geometry_msgs.msg import Pose
    from src.core.geometry import rotation_matrix_to_quaternion

    matrix = np.asarray(matrix, dtype=np.float64)
    pose = Pose()
    pose.position.x = float(matrix[0, 3])
    pose.position.y = float(matrix[1, 3])
    pose.position.z = float(matrix[2, 3])
    qx, qy, qz, qw = rotation_matrix_to_quaternion(matrix[:3, :3])
    pose.orientation.x = qx
    pose.orientation.y = qy
    pose.orientation.z = qz
    pose.orientation.w = qw
    return pose


def _base_pose_from_pose_msg(pose: Any, *, default_frame_id: str) -> BasePose2D | None:
    if pose is None:
        return None
    position = getattr(pose, "position", None)
    orientation = getattr(pose, "orientation", None)
    if position is None or orientation is None:
        return None
    from src.core.geometry import quaternion_to_yaw

    return BasePose2D(
        x=float(position.x),
        y=float(position.y),
        yaw=float(quaternion_to_yaw(float(orientation.x), float(orientation.y), float(orientation.z), float(orientation.w))),
        frame_id=default_frame_id,
    )


def _pose_msg_to_dict(pose: Any) -> dict[str, Any]:
    return {
        "position": {"x": float(pose.position.x), "y": float(pose.position.y), "z": float(pose.position.z)},
        "orientation": {
            "x": float(pose.orientation.x),
            "y": float(pose.orientation.y),
            "z": float(pose.orientation.z),
            "w": float(pose.orientation.w),
        },
    }


def _as_points_array(value: Any, label: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] < 3 or array.shape[0] == 0:
        raise ValueError(f"{label} must be a non-empty Nx3 array")
    finite = np.isfinite(array[:, :3]).all(axis=1)
    array = array[finite, :3]
    if array.shape[0] == 0:
        raise ValueError(f"{label} has no finite points")
    return array


def _clamp(value: float, low: float, high: float) -> float:
    return max(float(low), min(float(high), float(value)))


def _step_toward(current: float, target: float, max_step: float) -> float:
    delta = float(target) - float(current)
    if abs(delta) <= float(max_step):
        return float(target)
    return float(current) + math.copysign(float(max_step), delta)
