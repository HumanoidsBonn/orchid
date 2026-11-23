"""MoveIt manipulation planner backend.

This planner talks to MoveIt for IK and plan-only trajectory generation. It
does not execute; callers pass returned trajectories to the configured
execution controller.
"""

from __future__ import annotations
from dataclasses import replace
from src.core.moveit_request import MoveItPlanRequest, MoveItPlanningSettings
from src.core.failure import FailureInfo, FailureCode, moveit_failure

import math
import copy
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from src.core.geometry import (
    point_relative_to_planar_base,
    pose_dict_to_matrix,
    pose_matrix_relative_to_planar_base,
    rotation_matrix_to_quaternion,
    rpy_to_quaternion,
    wrap_angle,
)
from src.core.manipulation_types import BasePose2D, MoveItReachabilityRequest, ReachTarget
from src.core.planners import BaseManipulationPlanner, PlannerResult
from src.core.serialization import to_json_safe
from src.core.base_motion_bounds import validate_base_motion
import src.core.logger as log


_ReachTarget = ReachTarget
_CandidateBasePose = BasePose2D

_MOVEIT_ERROR_CODE_NAMES = {
    1: "SUCCESS",
    99999: "FAILURE",
    -1: "PLANNING_FAILED",
    -2: "INVALID_MOTION_PLAN",
    -3: "MOTION_PLAN_INVALIDATED_BY_ENVIRONMENT_CHANGE",
    -4: "CONTROL_FAILED",
    -5: "UNABLE_TO_AQUIRE_SENSOR_DATA",
    -6: "TIMED_OUT",
    -7: "PREEMPTED",
    -10: "START_STATE_IN_COLLISION",
    -11: "START_STATE_VIOLATES_PATH_CONSTRAINTS",
    -26: "START_STATE_INVALID",
    -12: "GOAL_IN_COLLISION",
    -13: "GOAL_VIOLATES_PATH_CONSTRAINTS",
    -14: "GOAL_CONSTRAINTS_VIOLATED",
    -27: "GOAL_STATE_INVALID",
    -28: "UNRECOGNIZED_GOAL_TYPE",
    -15: "INVALID_GROUP_NAME",
    -16: "INVALID_GOAL_CONSTRAINTS",
    -17: "INVALID_ROBOT_STATE",
    -18: "INVALID_LINK_NAME",
    -19: "INVALID_OBJECT_NAME",
    -21: "FRAME_TRANSFORM_FAILURE",
    -22: "COLLISION_CHECKING_UNAVAILABLE",
    -23: "ROBOT_STATE_STALE",
    -24: "SENSOR_INFO_STALE",
    -25: "COMMUNICATION_FAILURE",
    -29: "CRASH",
    -30: "ABORT",
    -31: "NO_IK_SOLUTION",
}




class MoveItManipulationPlanner(BaseManipulationPlanner):
    """MoveIt-backed planner that returns plans without executing them.

    Config keys:
        group_name: MoveIt planning group, default ``whole_body_weighted``.
        ee_link: end-effector link, default ``hand_palm_link``.
        frame_id: default target frame, default ``base_footprint``.
        ik_module: ModuleProxy endpoint for IK, default ``moveit_compute_ik``.
        fk_module: ModuleProxy endpoint for FK, default ``moveit_compute_fk``.
        cartesian_module: ModuleProxy endpoint for Cartesian paths, default
            ``moveit_compute_cartesian_path``.
        ik_service: IK service name, default ``/compute_ik``.
        fk_service: FK service name, default ``/compute_fk``.
        cartesian_service: Cartesian path service name, default
            ``/compute_cartesian_path``.
        move_group_action: MoveGroup action name, default ``/move_action``.
        allowed_planning_time: seconds, default ``5.0``.
        planning_attempts: attempts, default ``3``.
        planning_strategy: ``auto``, ``pose_constraints``, or
            ``ik_joint_constraints``. ``auto`` uses IK joint constraints for the
            arm group because live HSRB tests showed that direct pose
            constraints fail there while IK plus joint planning succeeds.
        goal_position_tolerance: meters, default ``0.01``.
        goal_orientation_tolerance: radians, default ``0.05``.
        reach_position_tolerance_m: meters, default ``0.10`` for
            position-only reachability checks.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self.execution_defaults = MoveItPlanningSettings.from_config(self.config)
        self._ros_node = self.config.get("ros2_node")
        self._owns_node = False
        self._move_group_client = None
        self._MoveGroup = None
        self._init_error: str | None = None
        self._last_collision_relaxation_metadata: dict[str, Any] = {}

    def check_ik(self, target: Any, *, context: Any | None = None, **options: Any) -> PlannerResult:
        if not self._ensure_ros(context, require_move_group=False):
            return PlannerResult(success=False, reason="moveit_unavailable", metadata={"error": self._init_error})
        pose_stamped = self._target_to_pose_stamped(target, context=context)
        if pose_stamped is None:
            return PlannerResult(
                success=False,
                reason="moveit_target_pose_unavailable",
                metadata={"target": to_json_safe(target)}
            )
        try:
            from moveit_msgs.srv import GetPositionIK

            request = GetPositionIK.Request()
            request.ik_request.group_name = self._group_name(target, options)
            request.ik_request.ik_link_name = self._ee_link(target, options)
            request.ik_request.pose_stamped = pose_stamped
            ik_timeout = max(0.0, float(self.config.get("ik_timeout_s", 5.0)))
            request.ik_request.timeout.sec = int(ik_timeout)
            request.ik_request.timeout.nanosec = int((ik_timeout - int(ik_timeout)) * 1_000_000_000)
            seed = self._joint_state_seed(target, options)
            if seed is not None:
                request.ik_request.robot_state.joint_state.name = list(seed)
                request.ik_request.robot_state.joint_state.position = [float(seed[name]) for name in seed]

            module_proxy = self._module_proxy(context)
            if module_proxy is None:
                return PlannerResult(success=False, reason="moveit_module_proxy_unavailable")
            response = module_proxy.call(
                str(self.config.get("ik_module", "moveit_compute_ik")),
                {"request": request, "raw_response": True},
                timeout=float(self.config.get("ik_timeout_s", 1.0)) + 1.0,
            )
            response_error = self._service_response_error(response, "ik")
            if response_error is not None:
                return PlannerResult(success=False, reason=response_error["reason"], metadata=response_error["metadata"])
            error_code = int(getattr(response.error_code, "val", 0))
            success = error_code == 1
            joints = self._joint_state_to_dict(response.solution.joint_state) if success else None
            error_name = self._moveit_error_name(error_code)
            seed_metadata = self._seed_metadata(seed)
            return PlannerResult(
                success=success,
                reason=None if success else self._moveit_reason("moveit_ik", error_code),
                joint_positions=joints,
                metadata={
                    "planner": "moveit",
                    "group_name": request.ik_request.group_name,
                    "ee_link": request.ik_request.ik_link_name,
                    "frame_id": pose_stamped.header.frame_id,
                    "error_code": error_code,
                    "error_code_name": error_name,
                    **seed_metadata,
                },
             failure=moveit_failure(error_code) if error_code != 1 else (None if success else FailureInfo(FailureCode.PLANNING_FAILED)))
        except Exception as exc:  # noqa: BLE001
            log.exception("MoveIt IK check failed: %s", exc)
            return PlannerResult(success=False, reason="moveit_ik_exception", metadata={"error": str(exc)})

    def check_reachability(self, request: MoveItReachabilityRequest, *, context: Any | None = None) -> PlannerResult:
        """Check manipulation reachability using the typed MoveIt request contract."""
        if not isinstance(request, MoveItReachabilityRequest):
            return PlannerResult(success=False, reason="moveit_reachability_request_invalid")
        options: dict[str, Any] = {
            **dict(request.metadata or {}),
            "base_drift_tolerance": request.base_drift_tolerance_dict(),
            "collision_relaxation": request.collision_policy == "target_relaxed",
            "reach_mode": request.mode,
        }
        if request.position_tolerance_m is not None:
            options["reach_position_tolerance_m"] = float(request.position_tolerance_m)
            options["goal_position_tolerance"] = float(request.position_tolerance_m)
        if request.orientation_tolerance_rad is not None:
            options["goal_orientation_tolerance"] = float(request.orientation_tolerance_rad)
        result = self._check_position_plan_reachability_from_base(
            request.reach_target,
            request.candidate_base,
            request.joint_state_seed,
            context=context,
            group_name=request.group_name,
            ee_link=request.ee_link,
            validate_plan=request.validate_plan,
            avoid_collisions_for_ik=request.avoid_collisions_for_ik,
            avoid_collisions_for_plan=request.avoid_collisions_for_plan,
            **options,
        )
        result.metadata.setdefault("request", request.to_dict())
        result.metadata.setdefault("method", "typed_moveit_reachability")
        return result

    def compute_fk(self, joint_state: Any, *, context: Any | None = None, link_names: list[str] | None = None, frame_id: str | None = None) -> PlannerResult:
        """Compute FK through the configured MoveIt FK module proxy."""
        if not self._ensure_ros(context, require_move_group=False):
            return PlannerResult(success=False, reason="moveit_unavailable", metadata={"error": self._init_error})
        module_proxy = self._module_proxy(context)
        if module_proxy is None:
            return PlannerResult(success=False, reason="moveit_module_proxy_unavailable")
        try:
            from moveit_msgs.srv import GetPositionFK

            request = GetPositionFK.Request()
            request.header.frame_id = str(frame_id or self.config.get("frame_id", "base_footprint"))
            if self._ros_node is not None:
                request.header.stamp = self._ros_node.get_clock().now().to_msg()
            request.fk_link_names = list(link_names or [str(self.config.get("ee_link", "hand_palm_link"))])
            request.robot_state.joint_state = joint_state
            response = module_proxy.call(
                str(self.config.get("fk_module", "moveit_compute_fk")),
                {"request": request, "raw_response": True},
                timeout=float(self.config.get("fk_timeout_s", self.config.get("ik_timeout_s", 1.0))) + 1.0,
            )
            response_error = self._service_response_error(response, "fk")
            if response_error is not None:
                return PlannerResult(success=False, reason=response_error["reason"], metadata=response_error["metadata"])
            error_code = int(getattr(response.error_code, "val", 0))
            success = error_code == 1
            error_name = self._moveit_error_name(error_code)
            pose_stamped = list(getattr(response, "pose_stamped", []) or [])
            first_pose = None
            if success and pose_stamped:
                first_pose = pose_dict_to_matrix(
                    {
                        "position": {
                            "x": float(pose_stamped[0].pose.position.x),
                            "y": float(pose_stamped[0].pose.position.y),
                            "z": float(pose_stamped[0].pose.position.z),
                        },
                        "orientation": {
                            "x": float(pose_stamped[0].pose.orientation.x),
                            "y": float(pose_stamped[0].pose.orientation.y),
                            "z": float(pose_stamped[0].pose.orientation.z),
                            "w": float(pose_stamped[0].pose.orientation.w),
                        },
                    }
                ).astype(float).tolist()
            return PlannerResult(
                success=success,
                reason=None if success else self._moveit_reason("moveit_fk", error_code),
                metadata={
                    "planner": "moveit",
                    "frame_id": request.header.frame_id,
                    "link_names": list(request.fk_link_names),
                    "error_code": error_code,
                    "error_code_name": error_name,
                    "pose": first_pose,
                    "pose_stamped": to_json_safe(pose_stamped) if success else [],
                },
             failure=moveit_failure(error_code) if error_code != 1 else (None if success else FailureInfo(FailureCode.PLANNING_FAILED)))
        except Exception as exc:  # noqa: BLE001
            log.exception("MoveIt FK compute failed: %s", exc)
            return PlannerResult(success=False, reason="moveit_fk_exception", metadata={"error": str(exc)})

    def check_state_validity(
        self,
        joint_state: Any,
        *,
        context: Any | None = None,
        group_name: str | None = None,
        constraints: Any | None = None,
    ) -> PlannerResult:
        """Check a RobotState through MoveIt's state-validity service."""
        if not self._ensure_ros(context, require_move_group=False):
            return PlannerResult(success=False, reason="moveit_unavailable", metadata={"error": self._init_error})
        module_proxy = self._module_proxy(context)
        if module_proxy is None:
            return PlannerResult(success=False, reason="moveit_module_proxy_unavailable")
        try:
            from moveit_msgs.srv import GetStateValidity

            request = GetStateValidity.Request()
            request.robot_state.joint_state = joint_state
            request.group_name = str(group_name or self.config.get("group_name", "arm"))
            if constraints is not None:
                request.constraints = constraints
            response = module_proxy.call(
                str(self.config.get("state_validity_module", "moveit_check_state_validity")),
                {"request": request, "raw_response": True},
                timeout=float(self.config.get("state_validity_timeout_s", self.config.get("planning_scene_timeout_s", 1.0))) + 1.0,
            )
            if response is None:
                return PlannerResult(
                    success=False,
                    reason="moveit_state_validity_service_no_response",
                    metadata={"error": "no_response"},
                 failure=FailureInfo())
            if isinstance(response, dict):
                error = response.get("error")
                if error:
                    return PlannerResult(
                        success=False,
                        reason="moveit_state_validity_service_error",
                        metadata={"error": str(error)},
                     failure=FailureInfo())
                response = response.get("response", response)
            if not hasattr(response, "valid"):
                return PlannerResult(
                    success=False,
                    reason="moveit_state_validity_service_invalid_response",
                    metadata={"response_type": type(response).__name__, "response": to_json_safe(response)},
                 failure=FailureInfo())
            valid = bool(getattr(response, "valid", False))
            return PlannerResult(
                success=valid,
                reason=None if valid else "moveit_state_invalid",
                metadata={
                    "planner": "moveit",
                    "group_name": request.group_name,
                    "valid": valid,
                    "contacts": to_json_safe(getattr(response, "contacts", [])),
                    "cost_sources": to_json_safe(getattr(response, "cost_sources", [])),
                },
            )
        except Exception as exc:  # noqa: BLE001
            log.exception("MoveIt state validity check failed: %s", exc)
            return PlannerResult(success=False, reason="moveit_state_validity_exception", metadata={"error": str(exc)})

    def plan_contract(self, contract: MoveItPlanRequest, *, context: Any | None = None) -> PlannerResult:
        """Plan with the explicit internal MoveIt request contract."""
        if not self._ensure_ros(context, require_move_group=True):
            return PlannerResult(success=False, reason="moveit_unavailable", metadata={"error": self._init_error})
        if not self._move_group_client.wait_for_server(timeout_sec=float(self.config.get("action_timeout_s", 2.0))):
            return PlannerResult(success=False, reason="moveit_action_unavailable")

        contract = self._resolve_planning_inputs(contract)
        if contract.constrain_base and contract.base_pose is None:
            return PlannerResult(success=False, reason="moveit_base_reference_missing", failure=FailureInfo(FailureCode.INVALID_STATE))
        if contract.constrain_base and not contract.start_state_seed:
            return PlannerResult(success=False, reason="moveit_start_state_missing", failure=FailureInfo(FailureCode.INVALID_STATE))
        if contract.constrain_base:
            seed = contract.start_state_seed
            if any(name not in seed or not math.isfinite(float(seed[name])) for name in self._base_joint_names()):
                return PlannerResult(success=False, reason="moveit_start_base_state_invalid", failure=FailureInfo(FailureCode.INVALID_STATE))
        target = {"ee_link": contract.ee_link} if contract.ee_link else {}
        pose_stamped = None
        request = None

        if contract.goal_mode == "direct_pose":
            pose_stamped = self._pose_to_pose_stamped(contract.pose)
            if pose_stamped is None:
                return PlannerResult(success=False, reason="moveit_contract_pose_invalid", metadata={"pose": to_json_safe(contract.pose)}, failure=FailureInfo(FailureCode.INVALID_REQUEST))
            request = self._build_pose_constraint_request(pose_stamped, contract.group_name, target, contract)
            if contract.start_state_seed:
                request.start_state = self._robot_state_msg_from_seed(contract.start_state_seed)
                request.start_state.is_diff = not self._start_state_complete(contract)
            if contract.constrain_base and contract.base_pose is not None and request.goal_constraints:
                constraints = request.goal_constraints[0]
                for joint_name, value in self._base_joint_values(contract.base_pose).items():
                    constraints.joint_constraints.append(self._joint_constraint(joint_name, value, contract, base_joint=True))
        elif contract.goal_mode == "ik_joint_goal":
            pose_stamped = self._pose_to_pose_stamped(contract.pose)
            if pose_stamped is None:
                return PlannerResult(success=False, reason="moveit_contract_pose_invalid", metadata={"pose": to_json_safe(contract.pose)}, failure=FailureInfo(FailureCode.INVALID_REQUEST))
            seed = dict(contract.start_state_seed or {})
            if not seed:
                return PlannerResult(
                    success=False,
                    reason="moveit_ik_start_state_missing",
                    metadata={
                        "planner": "moveit",
                        "contract": self._contract_metadata(contract),
                        "has_start_state_seed": False,
                        "seed_joint_count": 0,
                        "seed_joint_names": [],
                    },
                 failure=FailureInfo(FailureCode.INVALID_STATE))
            ik_result = self._call_ik_request(
                pose_stamped,
                group_name=contract.group_name,
                ee_link=str(contract.ee_link),
                seed=seed or None,
                avoid_collisions=contract.collision_policy != "target_relaxed",
                constraints=self._base_hold_constraints(contract.base_pose, contract)
                if contract.constrain_base and contract.base_pose is not None
                else None,
                context=context,
            )
            if not ik_result.success or not ik_result.joint_positions:
                return PlannerResult(
                    success=False,
                    reason="moveit_ik_before_plan_failed",
                    metadata={
                        "planner": "moveit",
                        "contract": self._contract_metadata(contract),
                        "ik_reason": ik_result.reason,
                        "ik_metadata": to_json_safe(ik_result.metadata),
                        "ik_attempt_count": 1,
                    },
                 failure=ik_result.failure or FailureInfo())
            joints = dict(ik_result.joint_positions)
            if contract.constrain_base:
                base_values = self._base_joint_values(contract.base_pose)
                if any(name not in joints or not math.isfinite(float(joints[name])) for name in base_values):
                    return PlannerResult(success=False, reason="moveit_ik_base_state_invalid", failure=FailureInfo(FailureCode.INVALID_STATE))
                yaw_name = self._base_joint_names()[2]
                joints[yaw_name] = float(seed[yaw_name]) + wrap_angle(float(joints[yaw_name]) - float(seed[yaw_name]))
                drift = self._base_drift_from_solution(joints, contract.base_pose)
                if not self._base_drift_within_tolerance(drift, contract):
                    return PlannerResult(success=False, reason="moveit_ik_base_drift_exceeded", metadata={"base_drift": drift}, failure=FailureInfo(FailureCode.LIMIT_VIOLATION))
            request = self._build_joint_plan_request(
                contract.group_name,
                joints,
                start_state_seed=seed or contract.start_state_seed,
                target=target,
                options=contract,
                base_hold_pose=contract.base_pose if contract.constrain_base else None,
            )
        elif contract.goal_mode == "joint_goal":
            if not contract.joint_positions:
                return PlannerResult(success=False, reason="moveit_contract_joint_positions_missing", failure=FailureInfo(FailureCode.INVALID_REQUEST))
            request = self._build_joint_plan_request(
                contract.group_name,
                contract.joint_positions,
                start_state_seed=contract.start_state_seed,
                target=target,
                options=contract,
                base_hold_pose=contract.base_pose if contract.constrain_base else None,
            )
        else:
            return PlannerResult(success=False, reason=f"moveit_contract_goal_mode_invalid:{contract.goal_mode}", failure=FailureInfo(FailureCode.INVALID_REQUEST))

        if contract.constrain_base:
            request.path_constraints = self._base_hold_constraints(contract.base_pose, contract)

        metadata = {
            "planner": "moveit",
            "group_name": contract.group_name,
            "ee_link": str(contract.ee_link),
            "planning_strategy": contract.goal_mode,
            "collision_policy": contract.collision_policy,
            "contract": self._contract_metadata(contract),
            **self._seed_metadata(contract.start_state_seed),
        }
        if pose_stamped is not None:
            metadata["frame_id"] = pose_stamped.header.frame_id
        if contract.metadata:
            metadata.update(contract.metadata)
        return self._send_move_group_plan(
            request,
            context=context,
            metadata=metadata,
            convert_trajectory=True,
            allow_collisions=contract.collision_policy == "target_relaxed",
            options=contract,
        )

    def plan_pose(self, target: Any, *, context: Any | None = None, **options: Any) -> PlannerResult:
        if not self._ensure_ros(context, require_move_group=True):
            return PlannerResult(success=False, reason="moveit_unavailable", metadata={"error": self._init_error})
        pose_stamped = self._target_to_pose_stamped(target, context=context)
        if pose_stamped is None:
            return PlannerResult(
                success=False,
                reason="moveit_target_pose_unavailable",
                metadata={"target": to_json_safe(target)}
            )
        try:
            if not self._move_group_client.wait_for_server(timeout_sec=float(self.config.get("action_timeout_s", 2.0))):
                return PlannerResult(success=False, reason="moveit_action_unavailable")

            if str(options.get("planning_motion", "")).strip() == "cartesian_line":
                return self.plan_cartesian_path(target, context=context, **options)

            group_name = self._group_name(target, options)
            strategy = self._planning_strategy(group_name, target, options)
            if options.get("constrain_base") or options.get("max_base_path_drift"):
                seed = self._joint_state_seed(target, options)
                reference = options.get("base_hold_pose") or options.get("base_pose")
                if reference is None and seed and all(name in seed for name in self._base_joint_names()):
                    reference = BasePose2D(*(seed[name] for name in self._base_joint_names()), frame_id="odom")
                if isinstance(reference, dict):
                    reference = BasePose2D.from_mapping(reference, default_frame_id="odom")
                return self.plan_contract(MoveItPlanRequest(
                    group_name=group_name, goal_mode="ik_joint_goal" if strategy == "ik_joint_constraints" else "direct_pose",
                    pose=pose_stamped, ee_link=self._ee_link(target, options),
                    constrain_base=bool(options.get("constrain_base", True)), base_pose=reference,
                    base_constraint_tolerance=options.get("base_constraint_tolerance") or options.get("base_drift_tolerance"),
                    max_base_path_drift=options.get("max_base_path_drift"), start_state_seed=seed,
                    collision_policy=str(options.get("collision_policy", "full")),
                ), context=context)
            if strategy == "ik_joint_constraints":
                ik_options = dict(options)
                ik_options["moveit_group"] = group_name
                ik_options["group_name"] = group_name
                ik_options["ee_link"] = self._ee_link(target, options)
                ik_result = self.check_ik(
                    target,
                    context=context,
                    **ik_options,
                )
                if not ik_result.success or not ik_result.joint_positions:
                    return PlannerResult(
                        success=False,
                        reason="moveit_ik_before_plan_failed",
                        metadata={
                            "planner": "moveit",
                            "group_name": group_name,
                            "strategy": strategy,
                            "ik_reason": ik_result.reason,
                            "ik_metadata": to_json_safe(ik_result.metadata),
                        },
                     failure=ik_result.failure or FailureInfo())
                request = self._build_joint_constraint_request(group_name, ik_result.joint_positions, target, options)
                ee_link = self._ee_link(target, options)
            else:
                request = self._build_pose_constraint_request(pose_stamped, group_name, target, options)
                ee_link = self._ee_link(target, options)

            return self._send_move_group_plan(
                request,
                context=context,
                metadata={
                    "planner": "moveit",
                    "group_name": group_name,
                    "ee_link": ee_link,
                    "frame_id": pose_stamped.header.frame_id,
                    "planning_strategy": strategy,
                    "collision_policy": str(options.get("collision_policy", "full")),
                },
                allow_collisions=bool(options.get("collision_relaxation", False))
                or str(options.get("collision_policy", "full")) == "target_relaxed",
                options=options,
            )
        except Exception as exc:  # noqa: BLE001
            log.exception("MoveIt pose planning failed: %s", exc)
            return PlannerResult(success=False, reason="moveit_plan_exception", metadata={"error": str(exc)})


    def _ensure_ros(self, context: Any | None, *, require_move_group: bool = True) -> bool:
        if not require_move_group and self._ros_node is not None:
            return True
        if require_move_group and self._move_group_client is not None:
            return True
        try:
            import rclpy
            from moveit_msgs.action import MoveGroup
            from rclpy.action import ActionClient

            if self._ros_node is None:
                self._ros_node = getattr(context, "ros2_node", None) or self.config.get("ros2_node")
            if self._ros_node is None:
                if not rclpy.ok():
                    rclpy.init()
                self._ros_node = rclpy.create_node("orchid_moveit_planner")
                self._owns_node = True
            self._MoveGroup = MoveGroup
            if require_move_group:
                self._move_group_client = ActionClient(
                    self._ros_node,
                    MoveGroup,
                    str(self.config.get("move_group_action", "/move_action")),
                )
            return True
        except Exception as exc:  # noqa: BLE001
            self._init_error = str(exc)
            log.warn("MoveIt planner unavailable: %s", exc)
            return False

    def _group_name(self, target: Any, options: dict[str, Any]) -> str:
        data = target if isinstance(target, dict) else {}
        return str(
            options.get("moveit_group")
            or options.get("group_name")
            or data.get("moveit_group")
            or data.get("group_name")
            or self.config.get("group_name", "whole_body_weighted")
        )

    def _ee_link(self, target: Any, options: dict[str, Any]) -> str:
        data = target if isinstance(target, dict) else {}
        return str(
            options.get("ee_link")
            or data.get("ee_link")
            or self.config.get("ee_link", "hand_palm_link")
        )

    def _planning_strategy(self, group_name: str, target: Any, options: dict[str, Any]) -> str:
        data = target if isinstance(target, dict) else {}
        strategy = str(
            options.get("planning_strategy")
            or data.get("planning_strategy")
            or self.config.get("planning_strategy", "auto")
        )
        if strategy == "auto":
            return "ik_joint_constraints" if group_name == "arm" else "pose_constraints"
        if strategy not in {"pose_constraints", "ik_joint_constraints"}:
            log.warn("Unsupported MoveIt planning_strategy '%s'; using pose_constraints", strategy)
            return "pose_constraints"
        return strategy

    def _check_position_plan_reachability_from_base(
        self,
        reach_target: _ReachTarget,
        candidate_base: _CandidateBasePose,
        joint_state_seed: Any | None,
        *,
        context: Any | None = None,
        group_name: str,
        ee_link: str | None = None,
        validate_plan: bool = True,
        avoid_collisions_for_ik: bool = False,
        avoid_collisions_for_plan: bool = True,
        **options: Any,
    ) -> PlannerResult:
        """Check candidate reachability with MoveGroup position constraints.

        This is used by ``MoveToManipulationPose`` where the question is only
        whether the hand can reach an XYZ point above the object from a
        hypothetical base pose. It deliberately avoids ``/compute_ik`` full-pose
        orientation probes.
        """
        if self._is_cancelled(options):
            return PlannerResult(success=False, reason="moveit_reach_cancelled")
        if not self._ensure_ros(context, require_move_group=True):
            return PlannerResult(success=False, reason="moveit_unavailable", metadata={"error": self._init_error})

        link_name = str(ee_link or options.get("ee_link") or self.config.get("ee_link", "hand_palm_link"))
        full_seed = self._build_full_robot_state_seed(joint_state_seed, candidate_base)
        request_target: dict[str, Any] = {"ee_link": link_name}
        base_hold_pose = candidate_base

        if group_name == "arm":
            local_target = self._reach_target_relative_to_candidate_base(reach_target, candidate_base, context=context)
            if local_target is None:
                return PlannerResult(
                    success=False,
                    reason="moveit_reach_target_transform_failed",
                    metadata={
                        "reach_target": self._reach_target_metadata(reach_target),
                        "candidate_base_pose": self._candidate_metadata(candidate_base),
                    },
                 failure=FailureInfo())
            frame_id = str(options.get("base_link_frame") or self.config.get("base_link_frame", "base_link"))
            log.debug("MoveIt reach target in %s: %s", frame_id, local_target)
            pose_stamped = self._pose_stamped_from_xyz_rpy(local_target, frame_id, (0.0, 0.0, 0.0))
            base_hold_pose = None
            target_meta = {"target_in_base_link": list(local_target)}
        else:
            pose_stamped = self._pose_stamped_from_xyz_rpy(
                (reach_target.x, reach_target.y, reach_target.z),
                reach_target.frame_id,
                (0.0, 0.0, 0.0),
            )
            target_meta = {"target_in_planning_frame": [reach_target.x, reach_target.y, reach_target.z]}

        if not self._move_group_client.wait_for_server(timeout_sec=float(self.config.get("action_timeout_s", 2.0))):
            return PlannerResult(success=False, reason="moveit_action_unavailable")

        base_metadata = {
            "planner": "moveit",
            "group_name": group_name,
            "ee_link": link_name,
            "frame_id": pose_stamped.header.frame_id,
            "method": "position_constraint_from_candidate_base",
            "candidate_base_pose": self._candidate_metadata(candidate_base),
            "reach_target": self._reach_target_metadata(reach_target),
            "reach_position_tolerance_m": self._reach_position_tolerance(options),
            "start_state_joints": sorted(full_seed),
            **target_meta,
        }

        kinematic_request = self._build_position_reach_request(
            pose_stamped,
            group_name,
            request_target,
            options,
            start_state_seed=full_seed,
            base_hold_pose=base_hold_pose,
        )
        kinematic_result = self._send_move_group_plan(
            kinematic_request,
            context=context,
            metadata={
                **base_metadata,
                "stage": "kinematic_position_plan",
                "requested_collision_checking": bool(avoid_collisions_for_ik),
            },
            convert_trajectory=False,
            allow_collisions=not bool(avoid_collisions_for_ik),
            options=options,
        )
        if not kinematic_result.success:
            kinematic_drift = self._base_drift_metadata(kinematic_result, candidate_base)
            return PlannerResult(
                success=False,
                reason=kinematic_result.reason or "moveit_kinematic_position_plan_failed",
                metadata={
                    **base_metadata,
                    "kinematic_result": to_json_safe(kinematic_result.metadata),
                    "kinematic_reason": kinematic_result.reason,
                    **kinematic_drift,
                    "plan_validation_attempted": False,
                },
            )
        kinematic_drift = self._base_drift_metadata(kinematic_result, candidate_base)
        if not bool(validate_plan):
            return PlannerResult(
                success=True,
                metadata={
                    **base_metadata,
                    "kinematic_result": to_json_safe(kinematic_result.metadata),
                    **kinematic_drift,
                    "plan_validation_attempted": False,
                },
            )

        collision_request = self._build_position_reach_request(
            pose_stamped,
            group_name,
            request_target,
            options,
            start_state_seed=full_seed,
            base_hold_pose=base_hold_pose,
        )
        collision_result = self._send_move_group_plan(
            collision_request,
            context=context,
            metadata={
                **base_metadata,
                "stage": "collision_position_plan",
                "requested_collision_checking": bool(avoid_collisions_for_plan),
            },
            convert_trajectory=False,
            allow_collisions=not bool(avoid_collisions_for_plan),
            options=options,
        )
        if not collision_result.success:
            collision_drift = self._base_drift_metadata(collision_result, candidate_base)
            return PlannerResult(
                success=False,
                reason=collision_result.reason or "moveit_collision_position_plan_failed",
                metadata={
                    **base_metadata,
                    "kinematic_result": to_json_safe(kinematic_result.metadata),
                    "collision_result": to_json_safe(collision_result.metadata),
                    "collision_reason": collision_result.reason,
                    **(collision_drift or kinematic_drift),
                    "plan_validation_attempted": True,
                },
            )
        collision_drift = self._base_drift_metadata(collision_result, candidate_base)

        return PlannerResult(
            success=True,
            metadata={
                **base_metadata,
                "kinematic_result": to_json_safe(kinematic_result.metadata),
                "collision_result": to_json_safe(collision_result.metadata),
                **(collision_drift or kinematic_drift),
                "plan_validation_attempted": True,
                "plan_success": True,
            },
        )
    
    # Not used anymore
    # def _check_position_constraint_reachability(self, target: Any, *, context: Any | None = None, **options: Any) -> PlannerResult:
    #     if self._is_cancelled(options):
    #         return PlannerResult(success=False, reason="moveit_reach_cancelled")
    #     if not self._ensure_ros(context, require_move_group=True):
    #         return PlannerResult(success=False, reason="moveit_unavailable", metadata={"error": self._init_error})
    #     pose_stamped = self._target_position_to_pose_stamped(target, context=context)
    #     if pose_stamped is None:
    #         return PlannerResult(success=False, reason="moveit_target_position_unavailable", metadata={"target": to_json_safe(target)})
    #     try:
    #         if not self._move_group_client.wait_for_server(timeout_sec=float(self.config.get("action_timeout_s", 2.0))):
    #             return PlannerResult(success=False, reason="moveit_action_unavailable")
    #         request = self._build_position_only_request(pose_stamped, self._group_name(target, options), target, options)
    #         return self._send_move_group_plan(
    #             request,
    #             context=context,
    #             metadata={
    #                 "planner": "moveit",
    #                 "group_name": request.group_name,
    #                 "ee_link": self._ee_link(target, options),
    #                 "frame_id": pose_stamped.header.frame_id,
    #                 "method": "position_constraint",
    #                 "reach_position_tolerance_m": self._reach_position_tolerance(options),
    #             },
    #             convert_trajectory=False,
    #         )
    #     except Exception as exc:  # noqa: BLE001
    #         log.exception("MoveIt position reachability check failed: %s", exc)
    #         return PlannerResult(success=False, reason="moveit_position_reach_exception", metadata={"error": str(exc)})

    def _build_pose_constraint_request(self, pose_stamped: Any, group_name: str, target: Any, options: MoveItPlanRequest | dict[str, Any]) -> Any:
        options = self._resolve_planning_inputs(options, target)
        from moveit_msgs.msg import Constraints, MotionPlanRequest, OrientationConstraint, PositionConstraint
        from shape_msgs.msg import SolidPrimitive
        constraints = Constraints()
        primitive = SolidPrimitive()
        primitive.type = SolidPrimitive.SPHERE
        primitive.dimensions = [float(options.settings.goal_position_tolerance)]
        pc = PositionConstraint()
        pc.header = pose_stamped.header
        pc.link_name = options.ee_link
        pc.constraint_region.primitives.append(primitive)
        pc.constraint_region.primitive_poses.append(pose_stamped.pose)
        pc.weight = 1.0
        constraints.position_constraints.append(pc)
        oc = OrientationConstraint()
        oc.header = pose_stamped.header
        oc.link_name = pc.link_name
        oc.orientation = pose_stamped.pose.orientation
        tol = float(options.settings.goal_orientation_tolerance)
        oc.absolute_x_axis_tolerance = tol
        oc.absolute_y_axis_tolerance = tol
        oc.absolute_z_axis_tolerance = tol
        oc.weight = 1.0
        constraints.orientation_constraints.append(oc)
        request = MotionPlanRequest()
        request.group_name = group_name
        self._apply_motion_request_defaults(request, options)
        request.goal_constraints.append(constraints)
        seed = options.start_state_seed
        if seed:
            request.start_state = self._robot_state_msg_from_seed(seed)
            request.start_state.is_diff = not self._start_state_complete(options)
        return request

    def _build_position_only_request(self, pose_stamped: Any, group_name: str, target: Any, options: dict[str, Any]) -> Any:
        from moveit_msgs.msg import Constraints, MotionPlanRequest, PositionConstraint
        from shape_msgs.msg import SolidPrimitive

        primitive = SolidPrimitive()
        primitive.type = SolidPrimitive.SPHERE
        primitive.dimensions = [self._reach_position_tolerance(options)]

        pc = PositionConstraint()
        pc.header = pose_stamped.header
        pc.link_name = self._ee_link(target, options)
        pc.constraint_region.primitives.append(primitive)
        pc.constraint_region.primitive_poses.append(pose_stamped.pose)
        pc.weight = 1.0

        constraints = Constraints()
        constraints.position_constraints.append(pc)

        request = MotionPlanRequest()
        request.group_name = group_name
        self._apply_motion_request_defaults(request, options)
        request.goal_constraints.append(constraints)
        return request

    def _build_position_reach_request(
        self,
        pose_stamped: Any,
        group_name: str,
        target: Any,
        options: dict[str, Any],
        *,
        start_state_seed: dict[str, float] | None,
        base_hold_pose: _CandidateBasePose | None = None,
    ) -> Any:
        request = self._build_position_only_request(pose_stamped, group_name, target, options)
        if start_state_seed:
            request.start_state = self._robot_state_msg_from_seed(start_state_seed)
            request.start_state.is_diff = not self._start_state_complete(options)
        if base_hold_pose is not None and request.goal_constraints:
            constraints = request.goal_constraints[0]
            for joint_name, value in self._base_joint_values(base_hold_pose).items():
                constraints.joint_constraints.append(self._joint_constraint(joint_name, value, options, base_joint=True))
            request.path_constraints = self._base_hold_constraints(base_hold_pose, options)
        return request

    def _build_joint_constraint_request(self, group_name: str, joint_positions: dict[str, float], target: Any, options: MoveItPlanRequest | dict[str, Any]) -> Any:
        options = self._resolve_planning_inputs(options, target)
        from moveit_msgs.msg import Constraints, JointConstraint, MotionPlanRequest
        constraints = Constraints()
        joint_names = self._active_joint_names_for_group(group_name, joint_positions)
        tolerance = float(options.settings.joint_goal_tolerance)
        for name in joint_names:
            if name not in joint_positions:
                continue
            jc = JointConstraint()
            jc.joint_name = name
            jc.position = float(joint_positions[name])
            jc.tolerance_above = tolerance
            jc.tolerance_below = tolerance
            jc.weight = 1.0
            constraints.joint_constraints.append(jc)
        request = MotionPlanRequest()
        request.group_name = group_name
        self._apply_motion_request_defaults(request, options)
        request.goal_constraints.append(constraints)
        return request

    def _build_joint_plan_request(self, group_name: str, joint_positions: dict[str, float], *, start_state_seed: dict[str, float] | None, target: Any, options: MoveItPlanRequest | dict[str, Any], base_hold_pose: _CandidateBasePose | None=None) -> Any:
        options = self._resolve_planning_inputs(options, target)
        request = self._build_joint_constraint_request(group_name, joint_positions, target, options)
        if start_state_seed:
            request.start_state = self._robot_state_msg_from_seed(start_state_seed)
            request.start_state.is_diff = not self._start_state_complete(options)
        if base_hold_pose is not None and request.goal_constraints:
            constraints = request.goal_constraints[0]
            existing = {jc.joint_name: jc for jc in constraints.joint_constraints}
            for joint_name, value in self._base_joint_values(base_hold_pose).items():
                if joint_name in existing:
                    window = self._joint_constraint(joint_name, value, options, base_joint=True)
                    goal = existing[joint_name]
                    goal.tolerance_above = max(0.0, min(goal.tolerance_above, value + window.tolerance_above - goal.position))
                    goal.tolerance_below = max(0.0, min(goal.tolerance_below, goal.position - value + window.tolerance_below))
                    continue
                constraints.joint_constraints.append(self._joint_constraint(joint_name, value, options, base_joint=True))
            request.path_constraints = self._base_hold_constraints(base_hold_pose, options)
        return request

    def _call_ik_request(
        self,
        pose_stamped: Any,
        *,
        group_name: str,
        ee_link: str,
        seed: dict[str, float] | None,
        avoid_collisions: bool,
        constraints: Any | None,
        context: Any | None,
    ) -> PlannerResult:
        try:
            from moveit_msgs.srv import GetPositionIK

            request = GetPositionIK.Request()
            request.ik_request.group_name = group_name
            request.ik_request.ik_link_name = ee_link
            request.ik_request.pose_stamped = pose_stamped
            request.ik_request.avoid_collisions = bool(avoid_collisions)
            if constraints is not None:
                request.ik_request.constraints = constraints
            ik_timeout = max(0.0, float(self.config.get("ik_timeout_s", 0.1)))
            log.error("IK_TIMEOUT:", ik_timeout)
            request.ik_request.timeout.sec = int(ik_timeout)
            request.ik_request.timeout.nanosec = int((ik_timeout - int(ik_timeout)) * 1_000_000_000)
            if seed:
                request.ik_request.robot_state = self._robot_state_msg_from_seed(seed)

            module_proxy = self._module_proxy(context)
            if module_proxy is None:
                return PlannerResult(success=False, reason="moveit_module_proxy_unavailable")
            response = module_proxy.call(
                str(self.config.get("ik_module", "moveit_compute_ik")),
                {"request": request, "raw_response": True},
                timeout=float(self.config.get("ik_timeout_s", 1.0)) + 1.0,
            )
            response_error = self._service_response_error(response, "ik")
            if response_error is not None:
                return PlannerResult(success=False, reason=response_error["reason"], metadata=response_error["metadata"])
            error_code = int(getattr(response.error_code, "val", 0))
            success = error_code == 1
            joints = self._joint_state_to_dict(response.solution.joint_state) if success else None
            error_name = self._moveit_error_name(error_code)
            seed_metadata = self._seed_metadata(seed)
            return PlannerResult(
                success=success,
                reason=None if success else self._moveit_reason("moveit_ik", error_code),
                joint_positions=joints,
                metadata={
                    "planner": "moveit",
                    "group_name": group_name,
                    "ee_link": ee_link,
                    "frame_id": pose_stamped.header.frame_id,
                    "avoid_collisions": bool(avoid_collisions),
                    "error_code": error_code,
                    "error_code_name": error_name,
                    **seed_metadata,
                },
             failure=moveit_failure(error_code) if error_code != 1 else (None if success else FailureInfo(FailureCode.PLANNING_FAILED)))
        except Exception as exc:  # noqa: BLE001
            log.exception("MoveIt IK request failed: %s", exc)
            return PlannerResult(success=False, reason="moveit_ik_exception", metadata={"error": str(exc)})

    def _build_full_robot_state_seed(
        self,
        joint_state_seed: Any | None,
        candidate_base: _CandidateBasePose,
    ) -> dict[str, float]:
        seed = self._normalize_joint_state_seed(joint_state_seed) or {}
        seed.update(self._base_joint_values(candidate_base))
        return seed

    def _robot_state_msg_from_seed(self, seed: dict[str, float]):
        from moveit_msgs.msg import RobotState

        state = RobotState()
        state.joint_state.name = list(seed)
        state.joint_state.position = [float(seed[name]) for name in seed]
        return state

    def _apply_motion_request_defaults(self, request: Any, options: MoveItPlanRequest | dict[str, Any]) -> None:
        options = self._resolve_planning_inputs(options)
        request.pipeline_id = str(options.settings.pipeline_id)
        planner_id = options.settings.planner_id
        if planner_id:
            request.planner_id = str(planner_id)
        request.num_planning_attempts = int(options.settings.planning_attempts)
        request.allowed_planning_time = float(options.settings.allowed_planning_time)
        request.max_velocity_scaling_factor = float(options.settings.max_velocity_scaling_factor)
        request.max_acceleration_scaling_factor = float(options.settings.max_acceleration_scaling_factor)
        workspace = options.settings.workspace
        if workspace is not None:
            self._apply_workspace_parameters(request.workspace_parameters, workspace)

    @staticmethod
    def _apply_workspace_parameters(workspace_parameters: Any, workspace: Any) -> None:
        if isinstance(workspace, dict):
            frame_id = workspace.get("frame_id") or workspace.get("header", {}).get("frame_id")
            min_corner = workspace.get("min_corner") or workspace.get("min") or [-1.0, -1.0, -1.0]
            max_corner = workspace.get("max_corner") or workspace.get("max") or [1.0, 1.0, 1.0]
        else:
            values = list(workspace)
            frame_id = "base_link"
            min_corner = values[:3] if len(values) >= 3 else [-1.0, -1.0, -1.0]
            max_corner = values[3:6] if len(values) >= 6 else [1.0, 1.0, 1.0]
        if frame_id:
            workspace_parameters.header.frame_id = str(frame_id)
        workspace_parameters.min_corner.x = float(min_corner[0])
        workspace_parameters.min_corner.y = float(min_corner[1])
        workspace_parameters.min_corner.z = float(min_corner[2])
        workspace_parameters.max_corner.x = float(max_corner[0])
        workspace_parameters.max_corner.y = float(max_corner[1])
        workspace_parameters.max_corner.z = float(max_corner[2])
        log.debug("MoveIt workspace parameters: frame_id=%s min=%s max=%s", workspace_parameters.header.frame_id, min_corner, max_corner)

    def _start_state_complete(self, options: MoveItPlanRequest | dict[str, Any]) -> bool:
        options = self._resolve_planning_inputs(options)
        value = options.settings.start_state_complete
        return bool(value)

    def _base_hold_constraints(self, candidate_base: _CandidateBasePose, options: MoveItPlanRequest | dict[str, Any]):
        options = self._resolve_planning_inputs(options)
        from moveit_msgs.msg import Constraints
        constraints = Constraints()
        for joint_name, value in self._base_joint_values(candidate_base).items():
            constraints.joint_constraints.append(self._joint_constraint(joint_name, value, options, base_joint=True))
        return constraints

    def _joint_constraint(self, joint_name: str, value: float, options: MoveItPlanRequest | dict[str, Any], *, base_joint: bool=False):
        options = self._resolve_planning_inputs(options)
        from moveit_msgs.msg import JointConstraint
        jc = JointConstraint()
        jc.joint_name = str(joint_name)
        jc.position = float(value)
        if base_joint:
            self._apply_base_joint_tolerance(jc, joint_name, options)
        else:
            tolerance = float(options.settings.joint_goal_tolerance)
            jc.tolerance_above = tolerance
            jc.tolerance_below = tolerance
        jc.weight = 1.0
        return jc

    def _apply_base_joint_tolerance(self, joint_constraint: Any, joint_name: str, options: MoveItPlanRequest | dict[str, Any]) -> None:
        options = self._resolve_planning_inputs(options)
        tolerances = options.base_constraint_tolerance or {}
        if not isinstance(tolerances, dict):
            tolerances = {}
        axis = self._base_joint_names().index(joint_name)
        default = 0.05 if axis == 2 else 0.03
        key = ('x', 'y', 'yaw')[axis]
        tolerance = float(tolerances.get(key, tolerances.get(joint_name, default)))
        if not math.isfinite(tolerance) or tolerance < 0:
            raise ValueError('base constraint tolerances must be finite and nonnegative')
        joint_constraint.tolerance_above = tolerance
        joint_constraint.tolerance_below = tolerance

    def _base_joint_values(self, candidate_base: _CandidateBasePose) -> dict[str, float]:
        names = self._base_joint_names()
        values = [candidate_base.x, candidate_base.y, candidate_base.yaw]
        return {name: float(values[index]) for index, name in enumerate(names[:3])}

    def _base_joint_names(self) -> list[str]:
        configured = self.config.get("base_joint_names", ["odom_x", "odom_y", "odom_t"])
        if isinstance(configured, (list, tuple)) and len(configured) >= 3:
            return [str(name) for name in configured[:3]]
        return ["odom_x", "odom_y", "odom_t"]

    def _base_drift_from_solution(self, joint_positions: dict[str, float], candidate_base: _CandidateBasePose) -> dict[str, float | None]:
        names = self._base_joint_names()
        target = [candidate_base.x, candidate_base.y, candidate_base.yaw]
        output: dict[str, float | None] = {}
        for index, name in enumerate(names[:3]):
            if name not in joint_positions:
                output[name] = None
                continue
            value = float(joint_positions[name])
            error = value - float(target[index])
            if index == 2:
                error = wrap_angle(error)
            output[name] = float(abs(error))
        return output

    def _base_drift_within_tolerance(self, drift: dict[str, float | None], options: MoveItPlanRequest | dict[str, Any]) -> bool:
        options = self._resolve_planning_inputs(options)
        names = self._base_joint_names()
        tolerances = options.base_constraint_tolerance or {}
        if not isinstance(tolerances, dict):
            tolerances = {}
        for index, name in enumerate(names[:3]):
            value = drift.get(name)
            if value is None:
                continue
            key = 'yaw' if index == 2 else 'x' if index == 0 else 'y'
            default = 0.05 if index == 2 else 0.03
            tolerance = float(tolerances.get(key, tolerances.get(name, default)))
            if float(value) > tolerance:
                return False
        return True

    def _base_drift_metadata(self, result: PlannerResult, candidate_base: _CandidateBasePose) -> dict[str, Any]:
        final_joints = (result.metadata or {}).get("final_joint_positions")
        if not isinstance(final_joints, dict):
            return {}
        drift = self._base_drift_from_solution(final_joints, candidate_base)
        dx = drift.get(self._base_joint_names()[0])
        dy = drift.get(self._base_joint_names()[1])
        yaw = drift.get(self._base_joint_names()[2])
        metadata: dict[str, Any] = {"base_drift": drift}
        if dx is not None and dy is not None:
            metadata["base_drift_xy_m"] = float(math.hypot(float(dx), float(dy)))
        if yaw is not None:
            metadata["base_drift_yaw_rad"] = float(yaw)
        return metadata

    def _reach_target_relative_to_candidate_base(
        self,
        reach_target: _ReachTarget,
        candidate_base: _CandidateBasePose,
        *,
        context: Any | None = None,
    ) -> np.ndarray | None:
        point = np.asarray([reach_target.x, reach_target.y, reach_target.z], dtype=float)
        log.debug("MoveIt reach target frame=%s candidate frame=%s", reach_target.frame_id, candidate_base.frame_id)
        if reach_target.frame_id != candidate_base.frame_id:
            sdc = getattr(context, "sensor_data_center", None)
            if sdc is None:
                return None
            try:
                transformed = sdc.transform_data(point.reshape(1, 3), reach_target.frame_id, candidate_base.frame_id)
                point = np.asarray(transformed, dtype=float).reshape(-1, 3)[0]
            except Exception as exc:  # noqa: BLE001
                log.warn("Unable to transform reach target %s -> %s: %s", reach_target.frame_id, candidate_base.frame_id, exc)
                return None
        return point_relative_to_planar_base(point, candidate_base)

    def _pose_stamped_from_xyz_rpy(self, xyz: tuple[float, float, float] | np.ndarray, frame_id: str, rpy: tuple[float, float, float]):
        from geometry_msgs.msg import PoseStamped

        pose = PoseStamped()
        pose.header.frame_id = str(frame_id)
        if self._ros_node is not None:
            pose.header.stamp = self._ros_node.get_clock().now().to_msg()
        values = np.asarray(xyz, dtype=float)
        qx, qy, qz, qw = rpy_to_quaternion(*rpy)
        pose.pose.position.x = float(values[0])
        pose.pose.position.y = float(values[1])
        pose.pose.position.z = float(values[2])
        pose.pose.orientation.x = qx
        pose.pose.orientation.y = qy
        pose.pose.orientation.z = qz
        pose.pose.orientation.w = qw
        return pose

    @staticmethod
    def _normalize_joint_state_seed(seed: Any | None) -> dict[str, float] | None:
        if seed is None:
            return None
        if isinstance(seed, dict) and "joints" in seed and isinstance(seed["joints"], dict):
            return {str(name): float(value) for name, value in seed["joints"].items()}
        if isinstance(seed, dict) and "name" in seed and "position" in seed:
            return {
                str(name): float(position)
                for name, position in zip(seed["name"], seed["position"], strict=False)
            }
        if hasattr(seed, "name") and hasattr(seed, "position"):
            return {
                str(name): float(position)
                for name, position in zip(seed.name, seed.position, strict=False)
            }
        if isinstance(seed, dict):
            try:
                return {str(name): float(value) for name, value in seed.items()}
            except Exception:
                return None
        return None

    @staticmethod
    def _reach_target_metadata(reach_target: _ReachTarget) -> dict[str, Any]:
        return {
            "x": reach_target.x,
            "y": reach_target.y,
            "z": reach_target.z,
            "frame_id": reach_target.frame_id,
            "source": reach_target.source,
        }

    @staticmethod
    def _candidate_metadata(candidate_base: _CandidateBasePose) -> dict[str, Any]:
        return {
            "x": candidate_base.x,
            "y": candidate_base.y,
            "yaw": candidate_base.yaw,
            "frame_id": candidate_base.frame_id,
        }

    def _active_joint_names_for_group(self, group_name: str, joint_positions: dict[str, float]) -> list[str]:
        configured = self.config.get("active_joints_by_group")
        if isinstance(configured, dict) and isinstance(configured.get(group_name), list):
            allowed = [str(name) for name in configured[group_name]]
            return [name for name in allowed if name in joint_positions]
        allowed_by_group = {
            "arm": [
                "arm_lift_joint",
                "arm_flex_joint",
                "arm_roll_joint",
                "wrist_flex_joint",
                "wrist_roll_joint",
            ],
            "base": [
                "odom_x",
                "odom_y",
                "odom_t",
            ],
            "whole_body": [
                "arm_lift_joint",
                "arm_flex_joint",
                "arm_roll_joint",
                "wrist_flex_joint",
                "wrist_roll_joint",
                "odom_x",
                "odom_y",
                "odom_t",
            ],
            "whole_body_light": [
                "arm_lift_joint",
                "arm_flex_joint",
                "arm_roll_joint",
                "wrist_flex_joint",
                "wrist_roll_joint",
                "odom_x",
                "odom_y",
                "odom_t",
            ],
            "whole_body_weighted": [
                "arm_lift_joint",
                "arm_flex_joint",
                "arm_roll_joint",
                "wrist_flex_joint",
                "wrist_roll_joint",
                "odom_x",
                "odom_y",
                "odom_t",
            ],
        }
        if group_name in allowed_by_group:
            return [name for name in allowed_by_group[group_name] if name in joint_positions]
        ignored = {str(name) for name in self.config.get("ignore_trajectory_joints", ["wrist_ft_sensor_frame_joint"])}
        return [name for name in joint_positions if name not in ignored]

    def _send_move_group_plan(self, request: Any, *, context: Any | None=None, metadata: dict[str, Any] | None=None, convert_trajectory: bool=True, allow_collisions: bool=False, options: MoveItPlanRequest | dict[str, Any] | None=None) -> PlannerResult:
        options = self._resolve_planning_inputs(options)
        goal = self._MoveGroup.Goal()
        goal.request = request
        goal.planning_options.plan_only = True
        goal.planning_options.replan = False
        if not list(getattr(goal.request.start_state.joint_state, 'name', []) or []):
            goal.request.start_state.is_diff = True
        goal.planning_options.planning_scene_diff.is_diff = True
        goal.planning_options.planning_scene_diff.robot_state.is_diff = True
        if allow_collisions:
            collisions_relaxed = self._relax_goal_collision_matrix(goal, options, context=context)
            collision_relaxation_metadata = dict(self._last_collision_relaxation_metadata)
        else:
            collisions_relaxed = False
            collision_relaxation_metadata = {'requested': False, 'applied': False, 'reason': 'collision_relaxation_not_requested', 'links': [], 'objects': []}
        send_future = self._move_group_client.send_goal_async(goal)
        goal_handle = self._spin_until_done(send_future, timeout_s=float(self.config.get('action_timeout_s', 2.0)))
        log.debug('MoveIt goal handle: %s', goal_handle)
        if goal_handle is None:
            return PlannerResult(success=False, reason='moveit_goal_send_timeout', metadata=metadata or {})
        if not getattr(goal_handle, 'accepted', False):
            return PlannerResult(success=False, reason='moveit_goal_rejected', metadata=metadata or {})
        result = self._spin_until_done(goal_handle.get_result_async(), timeout_s=float(self.config.get('allowed_planning_time', 5.0)) + 2.0)
        log.debug('MoveIt goal result: %s', result)
        if result is None:
            return PlannerResult(success=False, reason='moveit_plan_timeout', metadata=metadata or {})
        moveit_result = result.result
        error_code = int(getattr(moveit_result.error_code, 'val', 0))
        error_name = self._moveit_error_name(error_code)
        success = error_code == 1
        planned_trajectory = getattr(moveit_result, 'planned_trajectory', None)
        trajectory_summary = self._trajectory_summary(planned_trajectory)
        raw_trajectory = planned_trajectory if success and convert_trajectory else None
        final_joint_positions = self._trajectory_final_joint_positions(planned_trajectory)
        base_path_drift = self._base_path_drift_metadata(planned_trajectory, options.base_pose)
        bounds = self._base_validation_contract(options, request.group_name)
        validation = {}
        if success and bounds is not None:
            validation = validate_base_motion(getattr(planned_trajectory, 'joint_trajectory', planned_trajectory), **bounds)
            if validation.get('violation'):
                return PlannerResult(success=False, reason='moveit_plan_base_path_drift_exceeded', metadata={**(metadata or {}), **base_path_drift, **validation, 'base_motion_bounds': bounds}, failure=FailureInfo(FailureCode.LIMIT_VIOLATION))
        trajectory = self._to_ros2_control_batches(raw_trajectory, context=context) if raw_trajectory is not None else None
        result_metadata = dict(metadata or {})
        result_metadata.update({'error_code': error_code, 'error_code_name': error_name, 'request_summary': self._motion_plan_request_summary(goal.request), **trajectory_summary, 'trajectory_format': 'ros2_control_prepared_batches' if trajectory is not None else None, 'final_joint_positions': final_joint_positions, **base_path_drift, **validation, 'base_motion_bounds': bounds, 'collision_relaxation_applied': collisions_relaxed, 'collision_relaxation': collision_relaxation_metadata})
        drift_failure = self._base_path_drift_failure(base_path_drift, options.max_base_path_drift)
        if success and drift_failure is not None:
            result_metadata['base_path_drift_limit'] = drift_failure['limit']
            result_metadata['base_path_drift_value'] = drift_failure['value']
            return PlannerResult(success=False, reason='moveit_plan_base_path_drift_exceeded', metadata=result_metadata, failure=FailureInfo(FailureCode.LIMIT_VIOLATION))
        if success and convert_trajectory and (trajectory is None):
            return PlannerResult(success=False, reason='moveit_plan_no_ros2_control_trajectory', metadata=result_metadata)
        return PlannerResult(success=success, reason=None if success else self._moveit_reason('moveit_plan', error_code), trajectory=trajectory, metadata=result_metadata, failure=moveit_failure(error_code) if error_code != 1 else (None if success else FailureInfo(FailureCode.PLANNING_FAILED)))

    def _base_validation_contract(self, options: MoveItPlanRequest | dict[str, Any], group_name: str):
        options = self._resolve_planning_inputs(options)
        constrained = bool(options.constrain_base)
        limits = options.max_base_path_drift
        if not constrained and (not limits):
            return None
        reference = options.base_pose
        if isinstance(reference, BasePose2D):
            reference = reference.to_dict()
        reference = reference if isinstance(reference, dict) else {}
        tolerances = {}
        if constrained:
            for key, name in zip(('x', 'y', 'yaw'), self._base_joint_names()):
                tolerances[key] = self._joint_constraint(name, 0.0, options, base_joint=True).tolerance_above
        return {'reference': reference, 'joint_names': self._base_joint_names(), 'endpoint_tolerance': tolerances, 'path_tolerance': tolerances, 'path_limits': dict(limits or {}), 'require_base': group_name not in {'arm', 'gripper', 'head'}}


    @staticmethod
    def _contract_metadata(contract: MoveItPlanRequest) -> dict[str, Any]:
        return {
            "group_name": contract.group_name,
            "goal_mode": contract.goal_mode,
            "ee_link": contract.ee_link,
            "constrain_base": contract.constrain_base,
            "base_pose": contract.base_pose.to_dict() if contract.base_pose is not None else None,
            "base_constraint_tolerance": dict(contract.base_constraint_tolerance or {}),
            "collision_policy": contract.collision_policy,
            "max_base_path_drift": dict(contract.max_base_path_drift or {}),
            "goal_joint_names": sorted(contract.joint_positions or {}),
            **MoveItManipulationPlanner._seed_metadata(contract.start_state_seed),
            "metadata": to_json_safe(contract.metadata or {}),
        }

    @staticmethod
    def _seed_metadata(seed: dict[str, float] | None) -> dict[str, Any]:
        names = sorted(str(name) for name in (seed or {}))
        return {
            "has_start_state_seed": bool(names),
            "seed_joint_count": len(names),
            "seed_joint_names": names,
        }

    def _pose_to_pose_stamped(self, pose: Any):
        if pose is None:
            return None
        if hasattr(pose, "header") and hasattr(pose, "pose"):
            return pose
        return self._target_to_pose_stamped({"pose": pose})

    def _base_path_drift_metadata(self, trajectory: Any, base_pose: Any | None) -> dict[str, Any]:
        if base_pose is None:
            return {}
        try:
            if isinstance(base_pose, dict):
                candidate = BasePose2D.from_mapping(base_pose)
            elif isinstance(base_pose, BasePose2D):
                candidate = base_pose
            else:
                return {}
            joint_trajectory = getattr(trajectory, "joint_trajectory", trajectory)
            names = [str(name) for name in getattr(joint_trajectory, "joint_names", []) or []]
            points = list(getattr(joint_trajectory, "points", []) or [])
            base_names = self._base_joint_names()
            indexes = [names.index(name) if name in names else None for name in base_names]
            if not points or any(index is None for index in indexes[:3]):
                return {}
            max_xy = 0.0
            max_yaw = 0.0
            max_xy_index = 0
            max_yaw_index = 0
            for point_index, point in enumerate(points):
                positions = list(getattr(point, "positions", []) or [])
                if len(positions) <= max(indexes[:3]):
                    continue
                dx = float(positions[indexes[0]]) - candidate.x
                dy = float(positions[indexes[1]]) - candidate.y
                dyaw = abs(wrap_angle(float(positions[indexes[2]]) - candidate.yaw))
                xy = float(math.hypot(dx, dy))
                if xy > max_xy:
                    max_xy = xy
                    max_xy_index = point_index
                if dyaw > max_yaw:
                    max_yaw = dyaw
                    max_yaw_index = point_index
            return {
                "max_base_path_drift_xy_m": max_xy,
                "max_base_path_drift_xy_index": max_xy_index,
                "max_base_path_drift_yaw_rad": max_yaw,
                "max_base_path_drift_yaw_index": max_yaw_index,
            }
        except Exception as exc:  # noqa: BLE001
            log.debug("Unable to compute MoveIt base path drift: %s", exc)
            return {}

    @staticmethod
    def _base_path_drift_failure(metadata: dict[str, Any], limit: Any | None) -> dict[str, Any] | None:
        if not isinstance(limit, dict):
            return None
        xy_limit = limit.get("xy", limit.get("xy_m"))
        yaw_limit = limit.get("yaw", limit.get("yaw_rad"))
        xy_value = metadata.get("max_base_path_drift_xy_m")
        yaw_value = metadata.get("max_base_path_drift_yaw_rad")
        if xy_limit is not None and xy_value is not None and float(xy_value) > float(xy_limit):
            return {"limit": {"xy": float(xy_limit)}, "value": {"xy": float(xy_value)}}
        if yaw_limit is not None and yaw_value is not None and float(yaw_value) > float(yaw_limit):
            return {"limit": {"yaw": float(yaw_limit)}, "value": {"yaw": float(yaw_value)}}
        return None

    @staticmethod
    def _moveit_error_name(error_code: int) -> str:
        return _MOVEIT_ERROR_CODE_NAMES.get(int(error_code), f"UNKNOWN_{int(error_code)}")

    @classmethod
    def _moveit_reason(cls, prefix: str, error_code: int) -> str:
        return f"{prefix}_{cls._moveit_error_name(error_code).lower()}"

    @staticmethod
    def _trajectory_summary(trajectory: Any) -> dict[str, Any]:
        joint_trajectory = getattr(trajectory, "joint_trajectory", None)
        if joint_trajectory is None:
            return {
                "has_planned_trajectory": False,
                "trajectory_joint_names": [],
                "trajectory_point_count": 0,
            }
        joint_names = [str(name) for name in getattr(joint_trajectory, "joint_names", []) or []]
        points = list(getattr(joint_trajectory, "points", []) or [])
        return {
            "has_planned_trajectory": bool(joint_names and points),
            "trajectory_joint_names": joint_names,
            "trajectory_point_count": len(points),
        }

    @staticmethod
    def _trajectory_final_joint_positions(trajectory: Any) -> dict[str, float] | None:
        joint_trajectory = getattr(trajectory, "joint_trajectory", None)
        if joint_trajectory is None:
            return None
        names = list(getattr(joint_trajectory, "joint_names", []) or [])
        points = list(getattr(joint_trajectory, "points", []) or [])
        if not names or not points:
            return None
        positions = list(getattr(points[-1], "positions", []) or [])
        if not positions:
            return None
        return {
            str(name): float(position)
            for name, position in zip(names, positions, strict=False)
        }

    @staticmethod
    def _motion_plan_request_summary(request: Any) -> dict[str, Any]:
        workspace = getattr(request, "workspace_parameters", None)
        start_state = getattr(request, "start_state", None)
        joint_state = getattr(start_state, "joint_state", None)
        constraints = list(getattr(request, "goal_constraints", []) or [])
        return {
            "group_name": str(getattr(request, "group_name", "")),
            "pipeline_id": str(getattr(request, "pipeline_id", "")),
            "planner_id": str(getattr(request, "planner_id", "")),
            "num_planning_attempts": int(getattr(request, "num_planning_attempts", 0)),
            "allowed_planning_time": float(getattr(request, "allowed_planning_time", 0.0)),
            "max_velocity_scaling_factor": float(getattr(request, "max_velocity_scaling_factor", 0.0)),
            "max_acceleration_scaling_factor": float(getattr(request, "max_acceleration_scaling_factor", 0.0)),
            "workspace": MoveItManipulationPlanner._workspace_summary(workspace),
            "start_state": {
                "is_diff": bool(getattr(start_state, "is_diff", False)),
                "joint_count": len(list(getattr(joint_state, "name", []) or [])),
                "joint_names": [str(name) for name in list(getattr(joint_state, "name", []) or [])],
            },
            "goal_constraints": [MoveItManipulationPlanner._constraints_summary(item) for item in constraints],
            "path_constraints": MoveItManipulationPlanner._constraints_summary(getattr(request, "path_constraints", None)),
        }

    @staticmethod
    def _workspace_summary(workspace: Any) -> dict[str, Any] | None:
        if workspace is None:
            return None
        return {
            "frame_id": str(getattr(getattr(workspace, "header", None), "frame_id", "")),
            "min_corner": [
                float(getattr(getattr(workspace, "min_corner", None), "x", 0.0)),
                float(getattr(getattr(workspace, "min_corner", None), "y", 0.0)),
                float(getattr(getattr(workspace, "min_corner", None), "z", 0.0)),
            ],
            "max_corner": [
                float(getattr(getattr(workspace, "max_corner", None), "x", 0.0)),
                float(getattr(getattr(workspace, "max_corner", None), "y", 0.0)),
                float(getattr(getattr(workspace, "max_corner", None), "z", 0.0)),
            ],
        }

    @staticmethod
    def _constraints_summary(constraints: Any) -> dict[str, Any]:
        if constraints is None:
            return {
                "joint_constraints": 0,
                "position_constraints": 0,
                "orientation_constraints": 0,
                "visibility_constraints": 0,
            }
        return {
            "joint_constraints": len(list(getattr(constraints, "joint_constraints", []) or [])),
            "position_constraints": len(list(getattr(constraints, "position_constraints", []) or [])),
            "orientation_constraints": len(list(getattr(constraints, "orientation_constraints", []) or [])),
            "visibility_constraints": len(list(getattr(constraints, "visibility_constraints", []) or [])),
            "joint_names": [
                str(getattr(item, "joint_name", ""))
                for item in list(getattr(constraints, "joint_constraints", []) or [])
            ],
        }

    def _relax_goal_collision_matrix(self, goal: Any, options: MoveItPlanRequest | dict[str, Any], *, context: Any | None=None) -> bool:
        """Best-effort pairwise hand-target collision relaxation."""
        options = self._resolve_planning_inputs(options)
        self._last_collision_relaxation_metadata = {'requested': True, 'applied': False, 'reason': None, 'links': [], 'objects': []}
        link_names = options.settings.collision_relaxation_links
        object_names = options.settings.collision_relaxation_objects
        explicit_pairs = [[str(pair[0]), str(pair[1])] for pair in options.settings.collision_relaxation_pairs or [] if isinstance(pair, (list, tuple)) and len(pair) == 2]
        if not explicit_pairs and (not isinstance(link_names, (list, tuple)) or not link_names):
            self._last_collision_relaxation_metadata['reason'] = 'missing_collision_relaxation_links'
            return False
        if not explicit_pairs and (not isinstance(object_names, (list, tuple)) or not object_names):
            self._last_collision_relaxation_metadata['reason'] = 'missing_collision_relaxation_objects'
            return False
        try:
            links = [str(name) for name in link_names] if isinstance(link_names, (list, tuple)) else []
            objects = [str(name) for name in object_names] if isinstance(object_names, (list, tuple)) else []
            self._last_collision_relaxation_metadata.update({'links': links, 'objects': objects})
            current_acm = self._get_current_allowed_collision_matrix(context)
            if current_acm is None:
                self._last_collision_relaxation_metadata['reason'] = 'current_acm_unavailable'
                return False
            acm = copy.deepcopy(current_acm)
            original_entry_names = list(getattr(acm, 'entry_names', []) or [])
            allowed_pairs = [[link, object_name] for link in links for object_name in objects] + explicit_pairs
            for name in [*links, *objects, *(name for pair in explicit_pairs for name in pair)]:
                self._ensure_acm_entry(acm, name)
            for link, object_name in allowed_pairs:
                self._set_allowed_collision_pair(acm, link, object_name, True)
            scene = goal.planning_options.planning_scene_diff
            scene.is_diff = True
            scene.allowed_collision_matrix = acm
            self._last_collision_relaxation_metadata.update({'applied': True, 'reason': None, 'allowed_pairs': allowed_pairs, 'entry_names': list(getattr(acm, 'entry_names', []) or []), 'current_acm_entry_count': len(original_entry_names), 'merged_acm_entry_count': len(list(getattr(acm, 'entry_names', []) or []))})
            return True
        except Exception as exc:
            self._last_collision_relaxation_metadata['reason'] = f'exception: {exc}'
            log.warn('Unable to apply MoveIt collision relaxation: %s', exc)
            return False

    def _get_current_allowed_collision_matrix(self, context: Any | None):
        module_proxy = self._module_proxy(context)
        if module_proxy is None:
            self._last_collision_relaxation_metadata["reason"] = "moveit_module_proxy_unavailable"
            return None
        module_name = str(self.config.get("planning_scene_module", "moveit_get_planning_scene"))
        if hasattr(module_proxy, "has") and not module_proxy.has(module_name):
            self._last_collision_relaxation_metadata["reason"] = f"module_unavailable:{module_name}"
            return None
        try:
            from moveit_msgs.msg import PlanningSceneComponents
            from moveit_msgs.srv import GetPlanningScene

            request = GetPlanningScene.Request()
            request.components.components = PlanningSceneComponents.ALLOWED_COLLISION_MATRIX
            response = module_proxy.call(
                module_name,
                {"request": request, "raw_response": True},
                timeout=float(self.config.get("planning_scene_timeout_s", 2.0)),
            )
            if response is None:
                self._last_collision_relaxation_metadata["reason"] = "moveit_planning_scene_service_no_response"
                return None
            if isinstance(response, dict):
                error = response.get("error")
                if error:
                    self._last_collision_relaxation_metadata["reason"] = "moveit_planning_scene_service_error"
                    self._last_collision_relaxation_metadata["service_error"] = {"error": str(error)}
                else:
                    self._last_collision_relaxation_metadata["reason"] = "moveit_planning_scene_service_invalid_response"
                    self._last_collision_relaxation_metadata["service_error"] = {"response": to_json_safe(response)}
                return None
            scene = getattr(response, "scene", None)
            acm = getattr(scene, "allowed_collision_matrix", None)
            if acm is None:
                self._last_collision_relaxation_metadata["reason"] = "planning_scene_missing_acm"
                return None
            return acm
        except Exception as exc:  # noqa: BLE001
            self._last_collision_relaxation_metadata["reason"] = f"planning_scene_exception: {exc}"
            log.warn("Unable to fetch MoveIt planning scene ACM: %s", exc)
            return None

    @staticmethod
    def _set_allowed_collision_pair(acm: Any, name_a: str, name_b: str, enabled: bool) -> None:
        if name_a == name_b:
            return
        MoveItManipulationPlanner._ensure_acm_entry(acm, name_a)
        MoveItManipulationPlanner._ensure_acm_entry(acm, name_b)
        names = list(getattr(acm, "entry_names", []) or [])
        index_a = names.index(name_a)
        index_b = names.index(name_b)
        acm.entry_values[index_a].enabled[index_b] = bool(enabled)
        acm.entry_values[index_b].enabled[index_a] = bool(enabled)

    @staticmethod
    def _ensure_acm_entry(acm: Any, name: str) -> None:
        from moveit_msgs.msg import AllowedCollisionEntry

        names = list(getattr(acm, "entry_names", []) or [])
        if name in names:
            MoveItManipulationPlanner._normalize_acm_shape(acm)
            return

        names.append(name)
        acm.entry_names = names
        for entry in list(getattr(acm, "entry_values", []) or []):
            values = list(getattr(entry, "enabled", []) or [])
            values.append(False)
            entry.enabled = values
        new_entry = AllowedCollisionEntry()
        new_entry.enabled = [False] * len(names)
        acm.entry_values.append(new_entry)
        MoveItManipulationPlanner._normalize_acm_shape(acm)

    @staticmethod
    def _normalize_acm_shape(acm: Any) -> None:
        from moveit_msgs.msg import AllowedCollisionEntry

        names = list(getattr(acm, "entry_names", []) or [])
        values = list(getattr(acm, "entry_values", []) or [])
        while len(values) < len(names):
            entry = AllowedCollisionEntry()
            entry.enabled = []
            values.append(entry)
        for entry in values:
            row = list(getattr(entry, "enabled", []) or [])
            if len(row) < len(names):
                row.extend([False] * (len(names) - len(row)))
            elif len(row) > len(names):
                row = row[: len(names)]
            entry.enabled = row
        acm.entry_values = values[: len(names)]

    @staticmethod
    def _module_proxy(context: Any | None):
        return getattr(context, "modules", None) if context is not None else None

    def _target_to_pose_stamped(self, target: Any, *, context: Any | None = None):
        try:
            from geometry_msgs.msg import PoseStamped
        except Exception:
            return None
        data = target if isinstance(target, dict) else {}
        pose_stamped = PoseStamped()
        pose_stamped.header.frame_id = str(
            data.get("frame_id")
            or data.get("target_pose_frame")
            or self.config.get("frame_id", "base_footprint")
        )
        if self._ros_node is not None:
            pose_stamped.header.stamp = self._ros_node.get_clock().now().to_msg()

        matrix = data.get("target_pose_matrix")
        if matrix is not None:
            array = np.asarray(matrix, dtype=float)
            if array.shape != (4, 4):
                return None
            if isinstance(data.get("base_pose"), dict):
                array = pose_matrix_relative_to_planar_base(array, data["base_pose"])
                pose_stamped.header.frame_id = str(self.config.get("frame_id", "base_footprint"))
            qx, qy, qz, qw = rotation_matrix_to_quaternion(array[:3, :3])
            pose_stamped.pose.position.x = float(array[0, 3])
            pose_stamped.pose.position.y = float(array[1, 3])
            pose_stamped.pose.position.z = float(array[2, 3])
            pose_stamped.pose.orientation.x = float(qx)
            pose_stamped.pose.orientation.y = float(qy)
            pose_stamped.pose.orientation.z = float(qz)
            pose_stamped.pose.orientation.w = float(qw)
            return pose_stamped

        pose = data.get("pose") or data.get("ee_pose") or target
        if isinstance(pose, dict) and "pose" in pose:
            header = pose.get("header") or {}
            inner_pose = pose.get("pose") or {}
            if isinstance(header, dict) and header.get("frame_id"):
                pose_stamped.header.frame_id = str(header["frame_id"])
            pose = inner_pose
        if isinstance(pose, dict) and "position" in pose:
            position = pose["position"]
            orientation = pose.get("orientation", {})
            if isinstance(position, dict):
                px = position.get("x", 0.0)
                py = position.get("y", 0.0)
                pz = position.get("z", 0.0)
            else:
                values = list(position or [])
                px = values[0] if len(values) > 0 else 0.0
                py = values[1] if len(values) > 1 else 0.0
                pz = values[2] if len(values) > 2 else 0.0
            pose_stamped.pose.position.x = float(px)
            pose_stamped.pose.position.y = float(py)
            pose_stamped.pose.position.z = float(pz)
            pose_stamped.pose.orientation.x = float(orientation.get("x", 0.0))
            pose_stamped.pose.orientation.y = float(orientation.get("y", 0.0))
            pose_stamped.pose.orientation.z = float(orientation.get("z", 0.0))
            pose_stamped.pose.orientation.w = float(orientation.get("w", 1.0))
            return pose_stamped
        if isinstance(pose, dict) and all(key in pose for key in ("x", "y", "z")):
            qx, qy, qz, qw = rpy_to_quaternion(
                float(pose.get("roll", 0.0)),
                float(pose.get("pitch", 0.0)),
                float(pose.get("yaw", 0.0)),
            )
            pose_stamped.pose.position.x = float(pose["x"])
            pose_stamped.pose.position.y = float(pose["y"])
            pose_stamped.pose.position.z = float(pose["z"])
            pose_stamped.pose.orientation.x = qx
            pose_stamped.pose.orientation.y = qy
            pose_stamped.pose.orientation.z = qz
            pose_stamped.pose.orientation.w = qw
            pose_stamped.header.frame_id = str(pose.get("frame_id", pose_stamped.header.frame_id))
            return pose_stamped
        if all(hasattr(pose, key) for key in ("x", "y", "z")):
            qx, qy, qz, qw = rpy_to_quaternion(
                float(getattr(pose, "roll", 0.0)),
                float(getattr(pose, "pitch", 0.0)),
                float(getattr(pose, "yaw", 0.0)),
            )
            pose_stamped.pose.position.x = float(getattr(pose, "x"))
            pose_stamped.pose.position.y = float(getattr(pose, "y"))
            pose_stamped.pose.position.z = float(getattr(pose, "z"))
            pose_stamped.pose.orientation.x = qx
            pose_stamped.pose.orientation.y = qy
            pose_stamped.pose.orientation.z = qz
            pose_stamped.pose.orientation.w = qw
            pose_stamped.header.frame_id = str(getattr(pose, "frame_id", pose_stamped.header.frame_id))
            return pose_stamped
        if isinstance(pose, (list, tuple)) and len(pose) >= 7:
            pose_stamped.pose.position.x = float(pose[0])
            pose_stamped.pose.position.y = float(pose[1])
            pose_stamped.pose.position.z = float(pose[2])
            pose_stamped.pose.orientation.w = float(pose[3])
            pose_stamped.pose.orientation.x = float(pose[4])
            pose_stamped.pose.orientation.y = float(pose[5])
            pose_stamped.pose.orientation.z = float(pose[6])
            return pose_stamped
        return None

    def _target_position_to_pose_stamped(self, target: Any, *, context: Any | None = None):
        try:
            from geometry_msgs.msg import PoseStamped
        except Exception:
            return None
        data = target if isinstance(target, dict) else {}
        position = data.get("target_position") or data.get("object_position") or data.get("position")
        if position is None:
            return None
        values = list(position) if not isinstance(position, dict) else [
            position.get("x", 0.0),
            position.get("y", 0.0),
            position.get("z", 0.0),
        ]
        if len(values) < 3:
            return None
        pose_stamped = PoseStamped()
        pose_stamped.header.frame_id = str(
            data.get("target_position_frame")
            or data.get("target_frame")
            or data.get("frame_id")
            or self.config.get("frame_id", "base_footprint")
        )
        log.debug("Position target PoseStamped frame_id: %s", pose_stamped.header.frame_id)
        if self._ros_node is not None:
            pose_stamped.header.stamp = self._ros_node.get_clock().now().to_msg()
        point = np.array([float(values[0]), float(values[1]), float(values[2])], dtype=float)
        if isinstance(data.get("base_pose"), dict):
            point = point_relative_to_planar_base(point, data["base_pose"])
            pose_stamped.header.frame_id = str(self.config.get("frame_id", "base_footprint"))
        log.debug("Position target PoseStamped frame_id after base transform: %s", pose_stamped.header.frame_id)
        pose_stamped.pose.position.x = float(point[0])
        pose_stamped.pose.position.y = float(point[1])
        pose_stamped.pose.position.z = float(point[2])
        pose_stamped.pose.orientation.w = 1.0
        return pose_stamped

    def _reach_position_tolerance(self, options: dict[str, Any]) -> float:
        return float(
            options.get(
                "reach_position_tolerance_m",
                self.config.get("reach_position_tolerance_m", self.config.get("goal_position_tolerance", 0.10)),
            )
        )

    @staticmethod
    def _is_cancelled(options: dict[str, Any]) -> bool:
        value = options.get("cancelled", False)
        if callable(value):
            try:
                return bool(value())
            except Exception:  # noqa: BLE001
                return False
        return bool(value)

    @staticmethod
    def _service_response_error(response: Any, service_kind: str) -> dict[str, Any] | None:
        prefix = f"moveit_{service_kind}"
        if response is None:
            return {"reason": f"{prefix}_service_no_response", "metadata": {"error": "no_response"}}
        if isinstance(response, dict):
            error = response.get("error")
            if error:
                return {"reason": f"{prefix}_service_error", "metadata": {"error": str(error)}}
            return {
                "reason": f"{prefix}_service_invalid_response",
                "metadata": {"response": to_json_safe(response)},
            }
        if not hasattr(response, "error_code"):
            return {
                "reason": f"{prefix}_service_invalid_response",
                "metadata": {"response_type": type(response).__name__},
            }
        return None

    @staticmethod
    def _joint_state_seed(target: Any, options: dict[str, Any]) -> dict[str, float] | None:
        for source in (options or {}, target if isinstance(target, dict) else {}):
            seed = source.get("joint_state_seed") or source.get("seed_state")
            if seed is None:
                continue
            return MoveItManipulationPlanner._normalize_joint_state_seed(seed)
        return None

    @staticmethod
    def _joint_state_to_dict(joint_state: Any) -> dict[str, float]:
        names = list(getattr(joint_state, "name", []) or [])
        positions = list(getattr(joint_state, "position", []) or [])
        return {
            str(name): float(positions[index])
            for index, name in enumerate(names)
            if index < len(positions)
        }

    def _spin_until_done(self, future: Any, *, timeout_s: float) -> Any | None:
        deadline = time.monotonic() + max(float(timeout_s), 0.01)
        while not future.done() and time.monotonic() < deadline:
            self._spin_once(0.01)
        if not future.done():
            return None
        return future.result()

    def _spin_once(self, timeout_sec: float = 0.0) -> None:
        if self._owns_node:
            import rclpy

            rclpy.spin_once(self._ros_node, timeout_sec=timeout_sec)
        elif timeout_sec > 0:
            time.sleep(timeout_sec)

    def _to_ros2_control_batches(self, robot_trajectory: Any, *, context: Any | None = None) -> list[list[tuple[dict[str, Any], Any]]] | None:
        joint_trajectory = getattr(robot_trajectory, "joint_trajectory", robot_trajectory)
        source_names = [str(name) for name in getattr(joint_trajectory, "joint_names", []) or []]
        if not source_names:
            return None
        controllers = self._ros2_control_controller_configs(context)
        if not controllers:
            return None
        ignored = {str(name) for name in self.config.get("ignore_trajectory_joints", ["wrist_ft_sensor_frame_joint"])}
        remaining = set(source_names)
        batch: list[tuple[dict[str, Any], Any]] = []
        for controller in controllers:
            owned = [str(name) for name in controller.get("joints", []) or []]
            present = [name for name in owned if name in source_names]
            if not present:
                continue
            remaining -= set(present)
            controller_entry = dict(controller)
            controller_entry["name"] = str(controller.get("name") or controller.get("action"))
            controller_entry["action"] = str(controller["action"])
            batch.append(
                (
                    controller_entry,
                    self._extract_trajectory_subset(joint_trajectory, present),
                )
            )
        unhandled = sorted(name for name in remaining if name not in ignored)
        if unhandled:
            log.warn("MoveIt trajectory has no ros2_control owner for joints: %s", unhandled)
        return [batch] if batch else None

    def _ros2_control_controller_configs(self, context: Any | None) -> list[dict[str, Any]]:
        configured = self.config.get("ros2_control_controllers")
        if configured is None and context is not None:
            hardware = getattr(context, "hardware", {}) or {}
            arm_cfg = hardware.get("arm", {}) if isinstance(hardware, dict) else {}
            ros2_cfg = arm_cfg.get("ros2_control", {}) if isinstance(arm_cfg, dict) else {}
            configured = ros2_cfg.get("controllers")
        if isinstance(configured, dict):
            return [
                {"name": name, **value}
                for name, value in configured.items()
                if isinstance(value, dict) and value.get("action")
            ]
        if isinstance(configured, list):
            return [item for item in configured if isinstance(item, dict) and item.get("action")]
        return []

    @staticmethod
    def _extract_trajectory_subset(trajectory: Any, subset_names: list[str]) -> Any:
        from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

        source_names = [str(name) for name in getattr(trajectory, "joint_names", []) or []]
        subset = JointTrajectory()
        subset.header = getattr(trajectory, "header", subset.header)
        subset.joint_names = list(subset_names)
        subset.points = []
        for point in getattr(trajectory, "points", []) or []:
            new_point = JointTrajectoryPoint()
            new_point.positions = MoveItManipulationPlanner._extract_point_values(source_names, subset_names, getattr(point, "positions", []))
            new_point.velocities = MoveItManipulationPlanner._extract_point_values(source_names, subset_names, getattr(point, "velocities", []))
            new_point.accelerations = MoveItManipulationPlanner._extract_point_values(source_names, subset_names, getattr(point, "accelerations", []))
            new_point.time_from_start = point.time_from_start
            subset.points.append(new_point)
        return subset

    @staticmethod
    def _extract_point_values(source_names: list[str], subset_names: list[str], values: Any) -> list[float]:
        values = list(values or [])
        output: list[float] = []
        for name in subset_names:
            if name in source_names:
                index = source_names.index(name)
                if index < len(values):
                    output.append(float(values[index]))
                    continue
            output.append(0.0)
        return output

    def _resolve_planning_inputs(self, inputs, target=None):
        """Convert legacy feasibility API inputs explicitly at shared-builder entry."""
        defaults = getattr(self, "execution_defaults", None)
        if defaults is None:
            defaults = MoveItPlanningSettings.from_config(self.config)
            self.execution_defaults = defaults
        if isinstance(inputs, MoveItPlanRequest):
            resolved = inputs.settings or defaults
            if inputs.settings is not None and inputs.ee_link is not None and inputs.base_constraint_tolerance is not None:
                return inputs
            return replace(inputs, settings=resolved, ee_link=inputs.ee_link or resolved.ee_link,
                           base_constraint_tolerance=inputs.base_constraint_tolerance if inputs.base_constraint_tolerance is not None else (resolved.base_constraint_tolerance or {}))
        inputs = inputs or {}
        target = target if isinstance(target, dict) else {}
        from dataclasses import fields
        overrides = {item.name: inputs[item.name] for item in fields(MoveItPlanningSettings) if item.name in inputs}
        if "workspace_parameters" in inputs:
            overrides["workspace"] = inputs["workspace_parameters"]
        if "target_collision_objects" in inputs:
            overrides["collision_relaxation_objects"] = tuple(inputs["target_collision_objects"])
        settings = replace(defaults, **overrides)
        base = inputs.get("base_hold_pose", inputs.get("base_pose"))
        if isinstance(base, dict):
            base = BasePose2D.from_mapping(base, default_frame_id="odom")
        return MoveItPlanRequest(group_name=self._group_name(target, inputs), goal_mode="direct_pose", ee_link=self._ee_link(target, inputs),
            pose=target.get("pose"), start_state_seed=self._joint_state_seed(target, inputs),
            constrain_base=bool(inputs.get("constrain_base", False)), base_pose=base,
            base_constraint_tolerance=inputs.get("base_drift_tolerance", inputs.get("base_constraint_tolerance", defaults.base_constraint_tolerance)),
            max_base_path_drift=inputs.get("max_base_path_drift"), collision_policy=str(inputs.get("collision_policy", "full")), settings=settings)

    def plan_cartesian_path(self, target, *, context=None, **options):
        contract = self._resolve_planning_inputs(options, target)
        pose = self._target_to_pose_stamped(target, context=context)
        return self.plan_cartesian_request(replace(contract, pose=pose), context=context)
    def plan_cartesian_request(self, contract: MoveItPlanRequest, *, context: Any | None = None) -> PlannerResult:
        """Plan a MoveIt Cartesian path and return ros2_control-ready batches."""
        contract = self._resolve_planning_inputs(contract)
        if not self._ensure_ros(context, require_move_group=False):
            return PlannerResult(success=False, reason="moveit_unavailable", metadata={"error": self._init_error})
        pose_stamped = self._pose_to_pose_stamped(contract.pose)
        if pose_stamped is None:
            return PlannerResult(success=False, reason="moveit_target_pose_unavailable", metadata={"target": to_json_safe(contract.pose)})
        module_proxy = self._module_proxy(context)
        if module_proxy is None:
            return PlannerResult(success=False, reason="moveit_module_proxy_unavailable")
        try:
            from moveit_msgs.srv import GetCartesianPath

            group_name = contract.group_name
            request = GetCartesianPath.Request()
            request.header = pose_stamped.header
            request.group_name = group_name
            request.link_name = contract.ee_link
            request.waypoints = [pose_stamped.pose]
            request.max_step = float(contract.settings.cartesian_max_step)
            request.jump_threshold = float(contract.settings.cartesian_jump_threshold)
            request.avoid_collisions = bool(contract.settings.avoid_collisions)
            seed = contract.start_state_seed
            if seed is not None:
                request.start_state.joint_state.name = list(seed)
                request.start_state.joint_state.position = [float(seed[name]) for name in seed]

            if contract.constrain_base or contract.max_base_path_drift:
                reference = contract.base_pose
                if reference is None and seed and all(name in seed for name in self._base_joint_names()):
                    reference = BasePose2D(*(seed[name] for name in self._base_joint_names()), frame_id="odom")
                if reference is None:
                    return PlannerResult(success=False, reason="moveit_base_reference_missing", failure=FailureInfo(FailureCode.INVALID_STATE))
                contract = replace(contract, base_pose=reference)
                if contract.constrain_base:
                    request.path_constraints = self._base_hold_constraints(reference, contract)

            timeout_s = float(self.config.get("cartesian_timeout_s", self.config.get("allowed_planning_time", 5.0))) + 1.0
            response = module_proxy.call(
                str(self.config.get("cartesian_module", "moveit_compute_cartesian_path")),
                {"request": request, "raw_response": True},
                timeout=timeout_s,
            )
            response_error = self._service_response_error(response, "cartesian")
            if response_error is not None:
                return PlannerResult(success=False, reason=response_error["reason"], metadata=response_error["metadata"])
            error_code = int(getattr(getattr(response, "error_code", None), "val", 0))
            error_name = self._moveit_error_name(error_code)
            fraction = float(getattr(response, "fraction", 0.0))
            min_fraction = float(contract.settings.cartesian_min_fraction)
            success = error_code == 1 and fraction >= min_fraction
            raw_solution = getattr(response, "solution", None)
            trajectory_summary = self._trajectory_summary(raw_solution)
            bounds = self._base_validation_contract(contract, group_name)
            validation = validate_base_motion(getattr(raw_solution, "joint_trajectory", raw_solution), **bounds) if success and bounds else {}
            if validation.get("violation"):
                return PlannerResult(success=False, reason="moveit_plan_base_path_drift_exceeded",
                                     metadata={**validation, "base_motion_bounds": bounds}, failure=FailureInfo(FailureCode.LIMIT_VIOLATION))
            trajectory = self._to_ros2_control_batches(raw_solution, context=context) if success else None
            if success and trajectory is None:
                return PlannerResult(
                    success=False,
                    reason="moveit_cartesian_no_ros2_control_trajectory",
                    metadata={
                        "planner": "moveit",
                        "group_name": group_name,
                        "fraction": fraction,
                        "error_code": error_code,
                        "error_code_name": error_name,
                        **trajectory_summary,
                    },
                 failure=FailureInfo())
            return PlannerResult(
                success=success,
                reason=None if success else f"{self._moveit_reason('moveit_cartesian', error_code)}_fraction_{fraction:.3f}",
                trajectory=trajectory,
                metadata={
                    "planner": "moveit",
                    "group_name": group_name,
                    "ee_link": request.link_name,
                    "frame_id": pose_stamped.header.frame_id,
                    "fraction": fraction,
                    "min_fraction": min_fraction,
                    **validation,
                    "base_motion_bounds": bounds,
                    "error_code": error_code,
                    "error_code_name": error_name,
                    **trajectory_summary,
                    "trajectory_format": "ros2_control_prepared_batches" if trajectory is not None else None,
                },
             failure=moveit_failure(error_code) if error_code != 1 else (None if success else FailureInfo(FailureCode.PLANNING_FAILED)))
        except Exception as exc:  # noqa: BLE001
            log.exception("MoveIt Cartesian path planning failed: %s", exc)
            return PlannerResult(success=False, reason="moveit_cartesian_exception", metadata={"error": str(exc)})
