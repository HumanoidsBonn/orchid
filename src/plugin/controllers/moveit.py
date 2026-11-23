"""MoveIt move-group controller plugins (ROS 2 Humble)."""

from __future__ import annotations
from dataclasses import replace
from src.core.arm_request import ArmPoseRequest
from src.core.moveit_request import MoveItPlanRequest, MoveItPlanningSettings
from src.core.failure import FailureInfo, FailureCode, MotionFailure

import math
import threading
import time
from types import SimpleNamespace
from typing import Iterable

from src.core.controllers import (
    ArmStatus,
    BaseArmController,
    BaseGripperController,
    BaseHeadController,
    ControllerResult,
    HeadStatus,
    Pose3D,
)
from src.core.geometry import rpy_to_quaternion
from src.core.manipulation_types import BasePose2D
from src.core.planners import PlannerResult
from src.plugin.controllers.base_trajectory_sanitizer import (
    is_base_trajectory_config,
    sanitize_base_trajectory,
)

import src.core.logger as log


def _euler_to_quaternion(roll: float, pitch: float, yaw: float):
    from geometry_msgs.msg import Quaternion

    qx, qy, qz, qw = rpy_to_quaternion(roll, pitch, yaw)
    return Quaternion(x=qx, y=qy, z=qz, w=qw)


class _MoveItMoveGroupClient:
    """Shared MoveIt2 ``MoveGroup`` action client for a single move_group."""

    def __init__(self, hardware_config: dict, node_name: str):
        self.hardware_config = hardware_config
        self.move_group = hardware_config.get("move_group", "arm")
        self._action_name = hardware_config.get(
            "action_name", f"/{self.move_group}/move_action"
        )
        self._status: ArmStatus = "idle"
        self._error: str | None = None
        self._goal_handle = None
        self._lock = threading.Lock()
        self._ros_node = hardware_config.get("ros2_node")
        self._owns_node = False
        self._action_client = None
        self._MoveGroup = None
        self._init_ros(node_name)
        self.counter = 10

    def _init_ros(self, node_name: str) -> None:
        try:
            import rclpy
            from moveit_msgs.action import MoveGroup
            from rclpy.action import ActionClient
            from rclpy.node import Node
        except ImportError as exc:
            log.warning("MoveIt ROS dependencies unavailable: %s", exc)
            return

        if self._ros_node is None:
            if not rclpy.ok():
                rclpy.init()
            self._ros_node = Node(node_name)
            self._owns_node = True

        self._MoveGroup = MoveGroup
        self._action_client = ActionClient(
            self._ros_node,
            MoveGroup,
            self._action_name,
        )

    def check_client_ready(self) -> bool:
        if self._action_client is None:
            with self._lock:
                self._status = "failed"
                self._error = "MoveIt action client not initialized"
            return False

        if not self._action_client.wait_for_server(timeout_sec=2.0):
            with self._lock:
                self._status = "failed"
                self._error = f"MoveIt action server '{self._action_name}' unavailable"
            return False
        
        return True
    
    def send_pose_goal(self, request: ArmPoseRequest) -> None:
        pose = request.pose
        if not self.check_client_ready():
            return
    
        from geometry_msgs.msg import Pose
        from moveit_msgs.msg import Constraints, MotionPlanRequest, PositionConstraint

        goal_pose = Pose()
        goal_pose.position.x = pose.x
        goal_pose.position.y = pose.y
        goal_pose.position.z = pose.z
        goal_pose.orientation = _euler_to_quaternion(pose.roll, pose.pitch, pose.yaw)

        req = MotionPlanRequest()
        req.group_name = self.move_group
        req.num_planning_attempts = int(self.hardware_config.get("planning_attempts", 3))
        req.allowed_planning_time = float(
            self.hardware_config.get("allowed_planning_time", 5.0)
        )

        constraint = Constraints()
        pos = PositionConstraint()
        pos.header.frame_id = pose.frame_id
        pos.header.stamp = self._ros_node.get_clock().now().to_msg()
        pos.constraint_region.primitive_poses.append(goal_pose)
        constraint.position_constraints.append(pos)
        req.goal_constraints.append(constraint)

        goal_msg = self._MoveGroup.Goal()
        goal_msg.request = req
        goal_msg.planning_options.plan_only = False
        goal_msg.planning_options.replan = True

        with self._lock:
            self._status = "in_progress"
            self._error = None
            if not self._action_client.wait_for_server(timeout_sec=2.0):
                self._status = "failed"
                self._error = f"MoveIt action server '{self._action_name}' unavailable"
                return

            send_future = self._action_client.send_goal_async(goal_msg)
            send_future.add_done_callback(self._on_goal_response)

    def send_joint_goal(
        self,
        joint_names: Iterable[str],
        joint_positions: Iterable[float],
    ) -> None:
        if not self.check_client_ready():
            return

        from moveit_msgs.msg import Constraints, JointConstraint, MotionPlanRequest

        req = MotionPlanRequest()
        req.group_name = self.move_group
        req.num_planning_attempts = int(self.hardware_config.get("planning_attempts", 3))
        req.allowed_planning_time = float(
            self.hardware_config.get("allowed_planning_time", 5.0)
        )

        constraint = Constraints()
        for name, position in zip(joint_names, joint_positions, strict=True):
            jc = JointConstraint()
            jc.joint_name = name
            jc.position = float(position)
            jc.tolerance_above = 0.01
            jc.tolerance_below = 0.01
            jc.weight = 1.0
            constraint.joint_constraints.append(jc)
        req.goal_constraints.append(constraint)

        goal_msg = self._MoveGroup.Goal()
        goal_msg.request = req
        goal_msg.planning_options.plan_only = False
        goal_msg.planning_options.replan = True

        with self._lock:
            self._status = "in_progress"
            self._error = None
            if not self._action_client.wait_for_server(timeout_sec=2.0):
                self._status = "failed"
                self._error = f"MoveIt action server '{self._action_name}' unavailable"
                return
            send_future = self._action_client.send_goal_async(goal_msg)
            send_future.add_done_callback(self._on_goal_response)

    def send_named_goal(self, config_name: str) -> None:
        named_targets = self.hardware_config.get("named_targets", {})
        target = named_targets.get(config_name)
        if target is None:
            with self._lock:
                self._status = "failed"
                self._error = f"Unknown named target '{config_name}'"
            return
        if "joint_names" in target and "joint_positions" in target:
            self.send_joint_goal(target["joint_names"], target["joint_positions"])
            return
        if all(k in target for k in ("x", "y", "z")):
            self.send_pose_goal(ArmPoseRequest(Pose3D(x=target['x'], y=target['y'], z=target['z'], roll=target.get('roll', 0.0), pitch=target.get('pitch', 0.0), yaw=target.get('yaw', 0.0), frame_id=target.get('frame_id', 'base_link'))))
            return
        with self._lock:
            self._status = "failed"
            self._error = f"Invalid named target config for '{config_name}'"

    def update_pose_goal(self, pose: Pose3D) -> None:
        self.send_pose_goal(ArmPoseRequest(pose))

    def get_status(self) -> ArmStatus:
        with self._lock:
            return self._status

    def get_error(self) -> str | None:
        with self._lock:
            return self._error

    def cancel_goal(self) -> None:
        with self._lock:
            handle = self._goal_handle
        if handle is not None:
            handle.cancel_goal_async()
        with self._lock:
            self._status = "idle"
            self._goal_handle = None

    def _on_goal_response(self, future) -> None:
        goal_handle = future.result()
        if not goal_handle.accepted:
            with self._lock:
                self._status = "failed"
                self._error = "MoveIt goal rejected"
            return
        with self._lock:
            self._goal_handle = goal_handle
            result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._on_result)

    def _on_result(self, future) -> None:
        result = future.result()
        with self._lock:
            error_code = result.result.error_code.val
            if error_code == 1:  # SUCCESS
                self._status = "succeeded"
                self._error = None
            elif result.status == 6:
                self._status = "idle"
            else:
                self._status = "failed"
                self._error = f"MoveIt error_code={error_code}"

    def destroy(self) -> None:
        if self._owns_node and self._ros_node is not None:
            self._ros_node.destroy_node()


class MoveItArmController(BaseArmController):
    """Action-facing arm controller using MoveIt planning and ros2_control execution."""

    def __init__(self, hardware_config: dict):
        super().__init__(hardware_config)
        from src.plugin.controllers.ros2_control_trajectory import Ros2ControlController
        from src.plugin.planners.moveit_planner import MoveItManipulationPlanner

        planner_config = self._planner_config(hardware_config)
        self._planner = MoveItManipulationPlanner(planner_config)
        self._pose_defaults = MoveItPlanningSettings.from_config(planner_config)
        self._last_failure = None
        self._executor = Ros2ControlController(
            hardware_config,
            joint_state_provider=hardware_config.get("joint_state_provider"),
        )
        self._context = SimpleNamespace(
            modules=hardware_config.get("module_proxy"),
            ros2_node=hardware_config.get("ros2_node"),
            hardware={"arm": hardware_config},
            sensor_data_center=hardware_config.get("sensor_data_center"),
        )
        self._status: ArmStatus = "idle"
        self._error: str | None = None
        self._phase = "idle"
        self._base_retry_contract_factory = None
        self._base_retry_attempts = 0
        self._base_retry_max = 0
        self._ros_node = hardware_config.get("ros2_node")
        self._owns_node = False
        node_name = hardware_config.get("ros2_node_name", "moveit_arm_controller")
        self._init_ros(node_name)
    
        self._collision_object_pub = None
        self._attached_collision_object_pub = None
        self._pose_marker_pub = None
        self._apply_planning_scene_client = None
        self._owned_collision_objects: dict[str, object] = {}
        self._attached_collision_objects: set[str] = set()
        self._persistent_acm_applied = False
        self._persistent_acm_attempted = False

    def _init_ros(self, node_name: str) -> None:
        try:
            import rclpy
            from rclpy.node import Node
        except ImportError as exc:
            log.warning("ROS dependencies unavailable: %s", exc)
            return

        if self._ros_node is None:
            if not rclpy.ok():
                rclpy.init()
            self._ros_node = Node(node_name)
            self._owns_node = True



    def _plan_contract_with_retries(self, contract_factory, *, context, failure_reason: str):
        attempts = []
        plan_retries = max(0, int(self.hardware_config.get("moveit_plan_retries", 2)))
        ik_retries = max(0, int(self.hardware_config.get("moveit_ik_retries", 2)))
        max_attempts = 1 + max(plan_retries, ik_retries)
        delay_s = max(0.0, float(self.hardware_config.get("moveit_retry_delay_s", 0.15)))
        last_result = None
        for attempt_index in range(max_attempts):
            try:
                contract = contract_factory()
            except RuntimeError as exc:
                reason = str(exc) or failure_reason
                metadata = {
                    "attempt_index": attempt_index,
                    "reason": reason,
                    "has_start_state_seed": False,
                    "seed_joint_count": 0,
                    "seed_joint_names": [],
                }
                attempts.append(metadata)
                last_result = PlannerResult(success=False, reason=reason, metadata=metadata)
            else:
                seed_metadata = self._seed_metadata(getattr(contract, "start_state_seed", None))
                result = self._planner.plan_contract(contract, context=context)
                goal_mode = str(getattr(contract, "goal_mode", ""))
                metadata = dict(getattr(result, "metadata", {}) or {})
                metadata.setdefault("attempt_index", attempt_index)
                metadata.update({k: v for k, v in seed_metadata.items() if k not in metadata})
                attempts.append(
                    {
                        "attempt_index": attempt_index,
                        "success": bool(result.success and result.trajectory is not None),
                        "reason": result.reason,
                        **seed_metadata,
                    }
                )
                if result.success and result.trajectory is not None:
                    metadata["plan_attempt_count"] = attempt_index + 1
                    if goal_mode == "ik_joint_goal":
                        metadata["ik_attempt_count"] = attempt_index + 1
                    metadata["attempts"] = attempts
                    result.metadata = metadata
                    return result
                result.metadata = metadata
                last_result = result
            if last_result is not None and not (last_result.failure and last_result.failure.retryable):
                break
            if attempt_index + 1 < max_attempts and delay_s > 0.0:
                time.sleep(delay_s)

        if last_result is None:
            return PlannerResult(success=False, reason=failure_reason, metadata={"attempts": attempts})
        metadata = dict(getattr(last_result, "metadata", {}) or {})
        metadata["plan_attempt_count"] = len(attempts)
        if any(
            attempt.get("reason") in {"moveit_ik_before_plan_failed", "moveit_ik_start_state_missing", "moveit_start_state_unavailable"}
            for attempt in attempts
        ):
            metadata["ik_attempt_count"] = len(attempts)
        metadata["attempts"] = attempts
        last_result.metadata = metadata
        return last_result

    @staticmethod
    def _seed_metadata(seed: dict[str, float] | None) -> dict[str, object]:
        names = sorted(str(name) for name in (seed or {}))
        return {
            "has_start_state_seed": bool(names),
            "seed_joint_count": len(names),
            "seed_joint_names": names,
        }

    def add_collision_mesh(self, mesh_file_name, name: str = "collision_mesh", *, frame_id: str | None = None) -> bool:
        try:
            collision_object = self._collision_object_from_mesh(mesh_file_name, name, frame_id=frame_id)
            # Keep existing ownership bookkeeping even when apply/readback fails:
            # the topic publication may still reach the scene and needs cleanup.
            self._owned_collision_objects[name] = collision_object
            if not self._apply_collision_object(collision_object):
                log.warn("MoveIt collision object apply failed name=%s", name)
                return False
            if not self._verify_collision_object(collision_object):
                log.warn("MoveIt collision object readback failed name=%s frame=%s", name, collision_object.header.frame_id)
                return False
            log.info("MoveIt collision object verified name=%s frame=%s", name, collision_object.header.frame_id)
            return True
        except Exception as exc:
            log.exception("MoveIt collision mesh integration failed for %s: %s", name, exc)
            return False

    def _verify_collision_object(self, expected) -> bool:
        """Read the existing scene service; compare geometry in its returned frame."""
        from moveit_msgs.msg import PlanningSceneComponents
        from moveit_msgs.srv import GetPlanningScene
        from src.core.geometry import pose_dict_to_matrix
        import numpy as np

        proxy = self._planner._module_proxy(self._context)
        if proxy is None:
            return False
        request = GetPlanningScene.Request()
        request.components.components = PlanningSceneComponents.WORLD_OBJECT_GEOMETRY
        response = proxy.call(
            str(self._planner.config.get("planning_scene_module", "moveit_get_planning_scene")),
            {"request": request, "raw_response": True},
            timeout=float(self._planner.config.get("planning_scene_timeout_s", 2.0)),
        )
        scene = getattr(response, "scene", None)
        if scene is None:
            return False
        actual = next((obj for obj in scene.world.collision_objects if obj.id == expected.id), None)
        if actual is None or len(actual.meshes) != 1 or len(actual.mesh_poses) != 1 or actual.primitives or actual.planes:
            return False
        transform = np.eye(4)
        if actual.header.frame_id != expected.header.frame_id:
            sdc = getattr(self._context, "sensor_data_center", None)
            if sdc is None:
                return False
            resolved = sdc.get_pose_at(expected.header.frame_id, actual.header.frame_id)
            if not resolved:
                return False
            transform = pose_dict_to_matrix(resolved)
        def vertices(obj, matrix):
            xyz = np.array([[p.x, p.y, p.z] for p in obj.meshes[0].vertices])
            pose = obj.mesh_poses[0]
            matrix = matrix @ pose_dict_to_matrix({"position": {k: getattr(pose.position, k) for k in ("x", "y", "z")},
                                                  "orientation": {k: getattr(pose.orientation, k) for k in ("x", "y", "z", "w")}})
            return (np.c_[xyz, np.ones(len(xyz))] @ matrix.T)[:, :3]
        wanted, observed = vertices(expected, transform), vertices(actual, np.eye(4))
        if wanted.shape != observed.shape or not np.isfinite(observed).all():
            return False
        # MoveIt retains mesh vertex/triangle indices; verify complete geometry,
        # including winding/connectivity, rather than accepting matching bounds.
        return bool(np.allclose(wanted, observed, atol=1e-6, rtol=0) and
                    [list(t.vertex_indices) for t in expected.meshes[0].triangles] ==
                    [list(t.vertex_indices) for t in actual.meshes[0].triangles])

    def remove_collision_object(self, name: str) -> None:
        try:
            self.detach_collision_object(name)
            collision_object = self._collision_object_remove_msg(name)
            applied = self._apply_collision_object(collision_object)
            self._owned_collision_objects.pop(name, None)
            log.info("MoveIt collision object removed name=%s applied=%s", name, applied)
        except Exception as exc:  # noqa: BLE001
            log.exception("MoveIt collision object removal failed for %s: %s", name, exc)

    def clear_collision_scene(self) -> None:
        for name in set(self._attached_collision_objects) | set(self._owned_collision_objects):
            self.remove_collision_object(name)

    def allow_collision_pairs(self, pairs: list[tuple[str, str]]) -> bool:
        return self._apply_allowed_collision_pairs([(str(a), str(b)) for a, b in pairs])

    def attach_collision_object(
        self,
        name: str,
        link_name: str | None = None,
        touch_links: list[str] | None = None,
    ) -> bool:
        try:
            collision_object = self._owned_collision_objects.get(name)
            if collision_object is None:
                log.warn("MoveIt attach skipped; unknown collision object: %s", name)
                return False
            applied = self._apply_attach_collision_object(
                collision_object,
                link_name=link_name,
                touch_links=touch_links,
            )
            if not applied:
                return False
            self._attached_collision_objects.add(name)
            self._owned_collision_objects.pop(name, None)
            log.info("MoveIt collision object attached name=%s applied=%s", name, applied)
            return True
        except Exception as exc:  # noqa: BLE001
            log.exception("MoveIt collision object attach failed for %s: %s", name, exc)
            return False

    def detach_collision_object(self, name: str) -> None:
        if name not in self._attached_collision_objects:
            return
        try:
            attached = self._attached_collision_object_remove_msg(name)
            applied = self._apply_attached_collision_object(attached)
            self._attached_collision_objects.discard(name)
            log.info("MoveIt collision object detached name=%s applied=%s", name, applied)
        except Exception as exc:  # noqa: BLE001
            log.exception("MoveIt collision object detach failed for %s: %s", name, exc)

    def send_joint_goal(self, joints: dict[str, float]) -> None:
        self._last_failure = None
        self._base_retry_contract_factory = None
        self._status = "in_progress"
        self._error = None
        self._phase = "execution"
        self._executor.send_joint_goal(joints)
        if self._executor.get_status() == "failed":
            self._status = "failed"
            self._error = self._executor.get_error() or "ros2_control_joint_goal_failed"

    def send_trajectory_goal(self, trajectory) -> None:
        self._last_failure = None
        self._base_retry_contract_factory = None
        self._status = "in_progress"
        self._error = None
        self._phase = "execution"
        self._send_executor_trajectory(trajectory)

    def send_synchronized_trajectory_goal(self, trajectory, *, cancelled=None) -> None:
        self._last_failure = None
        self._base_retry_contract_factory = None
        self._status = "in_progress"
        self._error = None
        self._phase = "execution"
        self._executor.send_synchronized_trajectory_goal(trajectory, cancelled=cancelled)
        if self._executor.get_status() == "failed":
            self._status = "failed"
            self._error = self._executor.get_error()
            self._last_failure = self._executor.get_failure()

    def send_named_goal(self, config_name: str, *, force_execution: bool = False) -> None:
        self._last_failure = None
        self._base_retry_contract_factory = None
        try:
            self._ensure_persistent_acm()
            target = self.resolve_named_target(config_name)
            if target is None:
                raise ValueError(f"Unknown named target '{config_name}'")
            joints = self._named_target_joints(target)
            if joints is None:
                super().send_named_goal(config_name, force_execution=force_execution)
                return
            if self._joints_within_tolerance(joints):
                self._status = "succeeded"
                self._error = None
                self._phase = "idle"
                log.info("MoveIt named target '%s' already within tolerance", config_name)
                return
            if force_execution:
                self.send_joint_goal(joints)
                return
            self._status = "in_progress"
            self._error = None
            self._phase = "planning"
            def contract_factory():
                base_pose = self._current_base_pose()
                return MoveItPlanRequest(
                    group_name=str(self.hardware_config.get("named_target_move_group") or "arm"),
                    goal_mode="joint_goal",
                    joint_positions=joints,
                    constrain_base=base_pose is not None,
                    base_pose=base_pose,
                    base_constraint_tolerance={"x": 0.1, "y": 0.1, "yaw": 0.1},
                    max_base_path_drift={"xy": 0.08, "yaw": 0.12},
                    start_state_seed=self._current_joint_positions(),
                    metadata={"segment": "named_state", "named_target": config_name},
                    settings=replace(self._pose_defaults, joint_goal_tolerance=0.1),
                )

            plan = self._plan_contract_with_retries(
                contract_factory,
                context=self._context,
                failure_reason="moveit_named_target_planning_failed",
            )
            if not plan.success or plan.trajectory is None:
                self._status = "failed"
                self._error = plan.reason or "moveit_named_target_planning_failed"
                self._last_failure = plan.failure
                if plan.metadata:
                    log.warn("MoveIt named target planning failed: %s metadata=%s", self._error, plan.metadata)
                return
            self._phase = "execution"
            self._send_executor_trajectory(plan.trajectory, base_bounds=(plan.metadata or {}).get("base_motion_bounds"))
        except Exception as exc:  # noqa: BLE001
            self._status = "failed"
            self._error = str(exc)

    def send_base_pose_goal(self, pose, **options) -> None:
        self._last_failure = None
        try:
            from src.core.manipulation_types import BasePose2D

            self._ensure_persistent_acm()
            target_pose = BasePose2D.from_mapping(
                pose,
                default_frame_id=str(options.get("frame_id") or self.hardware_config.get("odom", "odom")),
            )
            self._status = "in_progress"
            self._error = None
            self._phase = "planning"
            self._base_retry_attempts = 0
            self._base_retry_max = max(0, int(self.hardware_config.get("whole_body_base_path_tolerance_retries", 1)))

            def contract_factory():
                start_state = self._current_joint_positions() or {}
                joint_positions = self._base_joint_positions(target_pose)
                if bool(options.get("hold_current_joints", True)):
                    joint_positions.update(self._current_hold_joint_positions(start_state))
                return MoveItPlanRequest(
                    group_name=str(self.hardware_config.get("base_escape_move_group", "base")),
                    goal_mode="joint_goal",
                    joint_positions=joint_positions,
                    constrain_base=False,
                    base_pose=(BasePose2D(*(start_state[name] for name in ("odom_x", "odom_y", "odom_t")), frame_id=str(self.hardware_config.get("odom", "odom")))
                               if options.get("max_base_path_drift") and all(name in start_state for name in ("odom_x", "odom_y", "odom_t")) else None),
                    collision_policy=str(options.get("collision_policy", "full")),
                    max_base_path_drift=options.get("max_base_path_drift"),
                    start_state_seed=start_state or None,
                    metadata={"segment": str(options.get("segment", "base_escape")), "base_goal": target_pose.to_dict()},
                )

            self._base_retry_contract_factory = contract_factory
            self._plan_and_execute_base_pose(contract_factory)
        except Exception as exc:  # noqa: BLE001
            self._status = "failed"
            self._error = str(exc)

    def _plan_and_execute_base_pose(self, contract_factory) -> None:
        self._phase = "planning"
        plan = self._plan_contract_with_retries(
            contract_factory,
            context=self._context,
            failure_reason="moveit_base_pose_planning_failed",
        )
        if not plan.success or plan.trajectory is None:
            self._status = "failed"
            self._error = plan.reason or "moveit_base_pose_planning_failed"
            self._last_failure = plan.failure
            if plan.metadata:
                log.warn("MoveIt base pose planning failed: %s metadata=%s", self._error, plan.metadata)
            return
        self._status = "in_progress"
        self._error = None
        self._phase = "execution"
        self._send_executor_trajectory(plan.trajectory, base_bounds=(plan.metadata or {}).get("base_motion_bounds"))

    def _retry_whole_body_base_after_path_tolerance(self, error: str | None) -> bool:
        if self._base_retry_contract_factory is None:
            return False
        if not self._is_omni_base_path_tolerance_failure(error):
            return False
        if self._base_retry_attempts >= self._base_retry_max:
            return False
        self._base_retry_attempts += 1
        delay_s = max(0.0, float(self.hardware_config.get("whole_body_base_retry_delay_s", 0.25)))
        if delay_s > 0.0:
            time.sleep(delay_s)
        log.warn(
            "Retrying whole-body base move after omni_base PATH_TOLERANCE_VIOLATED attempt=%s/%s",
            self._base_retry_attempts,
            self._base_retry_max,
        )
        self._plan_and_execute_base_pose(self._base_retry_contract_factory)
        return self._status == "in_progress"

    @staticmethod
    def _is_omni_base_path_tolerance_failure(error: str | None) -> bool:
        text = str(error or "")
        if "PATH_TOLERANCE_VIOLATED(-4)" not in text:
            return False
        return "omni_base" in text or "/omni_base_controller/follow_joint_trajectory" in text

    @staticmethod
    def _named_target_joints(target: dict) -> dict[str, float] | None:
        if target.get("type") not in (None, "joint_positions"):
            return None
        joints = target.get("joints")
        if isinstance(joints, dict):
            return {str(name): float(position) for name, position in joints.items()}
        if "joint_names" in target and "joint_positions" in target:
            return {
                str(name): float(position)
                for name, position in zip(target["joint_names"], target["joint_positions"], strict=True)
            }
        return None

    def _joints_within_tolerance(self, joints: dict[str, float]) -> bool:
        current = self._current_joint_positions()
        if current is None:
            return False
        for name, target in joints.items():
            if name not in current:
                return False
            tolerance = float(
                (self.hardware_config.get("goal_tolerance_by_joint") or {}).get(
                    name,
                    self.hardware_config.get("named_target_tolerance", self.hardware_config.get("goal_tolerance", 0.03)),
                )
            )
            if abs(float(current[name]) - float(target)) > tolerance:
                return False
        return True

    def _current_joint_positions(self) -> dict[str, float] | None:
        provider = self.hardware_config.get("joint_state_provider")
        if not callable(provider):
            return None
        try:
            joint_state = provider()
        except Exception as exc:  # noqa: BLE001
            log.debug("Unable to read current joint state for MoveIt controller: %s", exc)
            return None
        names = list(getattr(joint_state, "name", []) or [])
        positions = list(getattr(joint_state, "position", []) or [])
        if not names or not positions:
            return None
        return {
            str(name): float(positions[index])
            for index, name in enumerate(names)
            if index < len(positions)
        }

    def _current_base_pose(self):
        from src.core.manipulation_types import BasePose2D

        current = self._current_joint_positions()
        if current is None:
            return None
        if not all(name in current for name in ("odom_x", "odom_y", "odom_t")):
            return None
        return BasePose2D(
            x=float(current["odom_x"]),
            y=float(current["odom_y"]),
            yaw=float(current["odom_t"]),
            frame_id=str(self.hardware_config.get("odom", "odom")),
        )

    @staticmethod
    def _base_joint_positions(base_pose) -> dict[str, float]:
        return {
            "odom_x": float(base_pose.x),
            "odom_y": float(base_pose.y),
            "odom_t": float(base_pose.yaw),
        }

    def _current_hold_joint_positions(self, current: dict[str, float]) -> dict[str, float]:
        configured = self.hardware_config.get("base_escape_hold_joints")
        if configured is None:
            configured = [
                "arm_lift_joint",
                "arm_flex_joint",
                "arm_roll_joint",
                "wrist_flex_joint",
                "wrist_roll_joint",
                "head_pan_joint",
                "head_tilt_joint",
            ]
        return {
            str(name): float(current[str(name)])
            for name in configured
            if str(name) in current
        }

    def publish_pose_goal_marker(self, pose: Pose3D, *, label: str = "pose") -> None:
        if not bool(self.hardware_config.get("publish_pose_goal_markers", True)):
            return
        if self._ros_node is None:
            return
        try:
            from visualization_msgs.msg import Marker, MarkerArray

            if self._pose_marker_pub is None:
                self._pose_marker_pub = self._ros_node.create_publisher(
                    MarkerArray,
                    str(self.hardware_config.get("pose_goal_marker_topic", "/orchid/debug/arm_pose_goals")),
                    10,
                )
            stamp = self._ros_node.get_clock().now().to_msg()
            marker = Marker()
            marker.header.frame_id = str(pose.frame_id)
            marker.header.stamp = stamp
            marker.ns = "orchid_arm_pose_goals"
            marker.id = self._marker_id(label)
            marker.type = Marker.ARROW
            marker.action = Marker.ADD
            marker.pose.position.x = float(pose.x)
            marker.pose.position.y = float(pose.y)
            marker.pose.position.z = float(pose.z)
            marker.pose.orientation = _euler_to_quaternion(pose.roll, pose.pitch, pose.yaw)
            marker.scale.x = 0.14
            marker.scale.y = 0.025
            marker.scale.z = 0.025
            marker.color.r, marker.color.g, marker.color.b = self._marker_color(label)
            marker.color.a = 0.9

            text = Marker()
            text.header.frame_id = marker.header.frame_id
            text.header.stamp = stamp
            text.ns = "orchid_arm_pose_goal_labels"
            text.id = marker.id
            text.type = Marker.TEXT_VIEW_FACING
            text.action = Marker.ADD
            text.pose.position.x = float(pose.x)
            text.pose.position.y = float(pose.y)
            text.pose.position.z = float(pose.z) + 0.08
            text.pose.orientation.w = 1.0
            text.scale.z = 0.05
            text.color.r = text.color.g = text.color.b = 1.0
            text.color.a = 0.95
            text.text = str(label)
            self._pose_marker_pub.publish(MarkerArray(markers=[marker, text]))
        except Exception as exc:  # noqa: BLE001
            log.debug("Unable to publish MoveIt pose goal marker: %s", exc)

    @staticmethod
    def _marker_id(label: str) -> int:
        labels = {"approach": 1, "grasp": 2, "retreat": 3, "manual_pose": 4, "pose": 5}
        return labels.get(str(label), 99)

    @staticmethod
    def _marker_color(label: str) -> tuple[float, float, float]:
        colors = {
            "approach": (0.1, 0.7, 1.0),
            "grasp": (1.0, 0.3, 0.1),
            "retreat": (0.2, 1.0, 0.35),
            "manual_pose": (1.0, 0.9, 0.1),
        }
        return colors.get(str(label), (0.8, 0.8, 0.8))

    def update_pose_goal(self, pose: Pose3D) -> None:
        self.cancel_goal()
        self.send_pose_goal(ArmPoseRequest(pose))

    def get_status(self) -> ArmStatus:
        if self._status != "in_progress":
            return self._status
        if self._phase == "execution":
            status = self._executor.get_status()
            if status in ("succeeded", "failed", "idle"):
                if status == "failed":
                    error = self._executor.get_error()
                    if self._retry_whole_body_base_after_path_tolerance(error):
                        return self._status
                self._status = status
                if status == "failed":
                    self._error = self._executor.get_error()
        return self._status

    def get_error(self) -> str | None:
        return self._executor.get_error() if self._phase == "execution" and self._executor.get_error() else self._error

    def cancel_goal(self) -> None:
        self._executor.cancel_goal()
        self._status = "idle"
        self._phase = "idle"
        self._base_retry_contract_factory = None

    def _send_executor_trajectory(self, trajectory, *, base_bounds=None) -> None:
        if self._is_prepared_batches(trajectory):
            if bool(self.hardware_config.get("sanitize_base_trajectory", False)):
                trajectory = self._sanitize_prepared_base_batches(trajectory)
            if base_bounds is not None:
                from src.core.base_motion_bounds import validate_base_motion
                names = base_bounds["joint_names"]
                base_trajectories = [item for batch in trajectory for _goal, item in batch
                                     if any(name in list(item.joint_names) for name in names)]
                if not base_trajectories and base_bounds.get("require_base", True):
                    raise MotionFailure("moveit_dispatch_base_trajectory_missing", FailureInfo(FailureCode.INVALID_STATE))
                for item in base_trajectories:
                    validation = validate_base_motion(item, **base_bounds)
                    if validation.get("violation"):
                        log.warn("MoveIt dispatch blocked: %s", validation)
                        raise MotionFailure("moveit_dispatch_base_bounds_exceeded:" + str(validation["violation"]), FailureInfo(FailureCode.LIMIT_VIOLATION))
            self._executor.send_prepared_trajectory_goals(trajectory)
        else:
            if base_bounds is not None:
                from src.core.base_motion_bounds import validate_base_motion
                validation = validate_base_motion(getattr(trajectory, "joint_trajectory", trajectory), **base_bounds)
                if validation.get("violation"):
                    raise MotionFailure("moveit_dispatch_base_bounds_exceeded:" + str(validation["violation"]), FailureInfo(FailureCode.LIMIT_VIOLATION))
            self._executor.send_trajectory_goal(trajectory)
        if self._executor.get_status() == "failed":
            self._status = "failed"
            self._error = self._executor.get_error() or "ros2_control_execution_failed"

    def _sanitize_prepared_base_batches(self, batches):
        has_base = any(
            is_base_trajectory_config(dict(goal) if isinstance(goal, dict) else {}, trajectory)
            for batch in batches
            for goal, trajectory in batch
        )
        if not has_base:
            return batches

        current_positions = self._current_joint_positions()
        if current_positions is None:
            log.warn("MoveIt base trajectory sanitizer using first waypoint fallback; current joint state unavailable")
            current_positions = {}
        sanitized_batches = []
        for batch in batches:
            sanitized_batch = []
            for goal, trajectory in batch:
                config = dict(goal) if isinstance(goal, dict) else {}
                if not config and hasattr(goal, "name"):
                    config = {
                        "name": getattr(goal, "name", ""),
                        "action": getattr(goal, "action", ""),
                    }
                if is_base_trajectory_config(config, trajectory):
                    trajectory = sanitize_base_trajectory(
                        trajectory,
                        current_positions=current_positions,
                        controller_config=config,
                        log_prefix="MoveIt",
                    )
                sanitized_batch.append((goal, trajectory))
            if sanitized_batch:
                sanitized_batches.append(sanitized_batch)
        return sanitized_batches

    @staticmethod
    def _is_prepared_batches(trajectory) -> bool:
        if not isinstance(trajectory, list) or not trajectory:
            return False
        first_batch = trajectory[0]
        return isinstance(first_batch, list) and bool(first_batch)

    @staticmethod
    def _planner_config(hardware_config: dict) -> dict:
        planners = hardware_config.get("planners", {})
        config: dict = {}
        if isinstance(planners, dict):
            moveit_cfg = planners.get("moveit")
            if isinstance(moveit_cfg, dict):
                config.update(moveit_cfg)
        moveit_cfg = hardware_config.get("moveit")
        if isinstance(moveit_cfg, dict):
            config.update(moveit_cfg)
        for key in (
            "group_name",
            "ee_link",
            "frame_id",
            "move_group_action",
            "pipeline_id",
            "planner_id",
            "ik_module",
            "fk_module",
            "cartesian_module",
            "planning_scene_module",
            "ik_timeout_s",
            "fk_timeout_s",
            "planning_scene_timeout_s",
            "allowed_planning_time",
            "planning_attempts",
            "max_velocity_scaling_factor",
            "max_acceleration_scaling_factor",
            "start_state_complete",
            "workspace",
            "goal_position_tolerance",
            "goal_orientation_tolerance",
            "reach_position_tolerance_m",
            "reach_validate_joint_plan",
            "planning_strategy",
            "collision_frame_id",
            "collision_relaxation_links",
            "collision_relaxation_objects",
            "target_collision_objects",
            "attached_object_link",
            "attached_object_touch_links",
            "persistent_octomap_collision_links",
            "persistent_octomap_collision_object",
            "base_escape_hold_joints",
            "base_escape_move_group",
            "base_joint_names",
            "ignore_trajectory_joints",
        ):
            if key in hardware_config:
                config[key] = hardware_config[key]
        ros2_cfg = hardware_config.get("ros2_control")
        if isinstance(ros2_cfg, dict) and "controllers" in ros2_cfg:
            config.setdefault("ros2_control_controllers", ros2_cfg["controllers"])
        if "ros2_node" in hardware_config:
            config["ros2_node"] = hardware_config["ros2_node"]
        return config

    def _ensure_scene_interfaces(self) -> bool:
        if self._ros_node is None:
            return False
        if self._collision_object_pub is not None:
            return True
        try:
            from moveit_msgs.msg import AttachedCollisionObject, CollisionObject, PlanningScene
            from moveit_msgs.srv import ApplyPlanningScene

            self._collision_object_pub = self._ros_node.create_publisher(
                CollisionObject,
                str(self.hardware_config.get("collision_object_topic", "/collision_object")),
                10,
            )
            self._attached_collision_object_pub = self._ros_node.create_publisher(
                AttachedCollisionObject,
                str(self.hardware_config.get("attached_collision_object_topic", "/attached_collision_object")),
                10,
            )
            self._apply_planning_scene_client = self._ros_node.create_client(
                ApplyPlanningScene,
                str(self.hardware_config.get("apply_planning_scene_service", "/apply_planning_scene")),
            )
            PlanningScene
            return True
        except Exception as exc:  # noqa: BLE001
            log.warn("MoveIt planning scene interfaces unavailable: %s", exc)
            return False

    def _ensure_persistent_acm(self) -> None:
        if self._persistent_acm_applied:
            return
        if self._persistent_acm_attempted and not bool(self.hardware_config.get("retry_persistent_acm", True)):
            return
        self._persistent_acm_attempted = True
        links = self._persistent_octomap_collision_links()
        if not links:
            return
        octomap_name = str(
            self.hardware_config.get("persistent_octomap_collision_object")
            or (getattr(self._planner, "config", {}) or {}).get("persistent_octomap_collision_object")
            or "octomap"
        )
        pairs = [(link, octomap_name) for link in links]
        self._persistent_acm_applied = self._apply_allowed_collision_pairs(pairs)
        if self._persistent_acm_applied:
            log.info("MoveIt persistent octomap ACM applied for links=%s object=%s", links, octomap_name)

    def _persistent_octomap_collision_links(self) -> list[str]:
        configured = (
            self.hardware_config.get("persistent_octomap_collision_links")
            or (getattr(self._planner, "config", {}) or {}).get("persistent_octomap_collision_links")
            or []
        )
        if not isinstance(configured, (list, tuple)):
            return []
        return [str(name) for name in configured if str(name)]

    def _apply_allowed_collision_pairs(self, pairs: list[tuple[str, str]]) -> bool:
        if not pairs:
            return True
        if not self._ensure_scene_interfaces():
            return False
        try:
            current_acm = self._planner._get_current_allowed_collision_matrix(self._context)
            if current_acm is None:
                return False
            import copy

            acm = copy.deepcopy(current_acm)
            for name_a, name_b in pairs:
                self._planner._set_allowed_collision_pair(acm, str(name_a), str(name_b), True)
            return self._apply_planning_scene_diff(allowed_collision_matrix=acm)
        except Exception as exc:  # noqa: BLE001
            log.warn("MoveIt allowed-collision pair apply failed: %s", exc)
            return False

    def _collision_object_from_mesh(self, mesh_file_name, name: str, *, frame_id: str | None = None):
        from geometry_msgs.msg import Point, Pose
        from moveit_msgs.msg import CollisionObject
        from shape_msgs.msg import Mesh, MeshTriangle
        import open3d as o3d

        mesh_data = o3d.io.read_triangle_mesh(str(mesh_file_name))
        vertices = list(mesh_data.vertices)
        triangles = list(mesh_data.triangles)
        if not vertices or not triangles:
            raise ValueError(f"Collision mesh is empty: {mesh_file_name}")

        mesh = Mesh()
        mesh.vertices = [
            Point(x=float(vertex[0]), y=float(vertex[1]), z=float(vertex[2]))
            for vertex in vertices
        ]
        mesh.triangles = [
            MeshTriangle(vertex_indices=[int(triangle[0]), int(triangle[1]), int(triangle[2])])
            for triangle in triangles
        ]

        pose = Pose()
        pose.orientation.w = 1.0

        collision_object = CollisionObject()
        collision_object.header.frame_id = str(frame_id or self.hardware_config.get("collision_frame_id", self.hardware_config.get("odom", "odom")))
        if self._ros_node is not None:
            collision_object.header.stamp = self._ros_node.get_clock().now().to_msg()
        collision_object.id = str(name)
        collision_object.meshes = [mesh]
        collision_object.mesh_poses = [pose]
        collision_object.operation = CollisionObject.ADD
        return collision_object

    def _collision_object_remove_msg(self, name: str):
        from moveit_msgs.msg import CollisionObject

        collision_object = CollisionObject()
        collision_object.header.frame_id = str(self.hardware_config.get("collision_frame_id", self.hardware_config.get("odom", "odom")))
        if self._ros_node is not None:
            collision_object.header.stamp = self._ros_node.get_clock().now().to_msg()
        collision_object.id = str(name)
        collision_object.operation = CollisionObject.REMOVE
        return collision_object

    def _attached_collision_object_remove_msg(self, name: str):
        from moveit_msgs.msg import AttachedCollisionObject, CollisionObject

        attached = AttachedCollisionObject()
        attached.object.id = str(name)
        attached.object.operation = CollisionObject.REMOVE
        return attached

    def _apply_attach_collision_object(
        self,
        collision_object,
        *,
        link_name: str | None = None,
        touch_links: list[str] | None = None,
    ) -> bool:
        from moveit_msgs.msg import AttachedCollisionObject, CollisionObject

        attached = AttachedCollisionObject()
        attached.link_name = str(
            link_name
            or self.hardware_config.get("attached_object_link")
            or self.hardware_config.get("ee_link")
            or "hand_palm_link"
        )
        configured_touch_links = (
            touch_links
            or self.hardware_config.get("attached_object_touch_links")
            or self.hardware_config.get("collision_relaxation_links")
            or []
        )
        attached.touch_links = [str(item) for item in configured_touch_links]
        attached.object = collision_object
        attached.object.operation = CollisionObject.ADD

        remove_from_world = self._collision_object_remove_msg(str(collision_object.id))
        applied = self._apply_planning_scene_diff(
            collision_objects=[remove_from_world],
            attached_collision_objects=[attached],
        )
        # ApplyPlanningScene is authoritative; do not publish a second, potentially
        # contradictory attachment after reporting a failed service operation.
        if not applied:
            log.warn("MoveIt attachment apply was not acknowledged: %s", collision_object.id)
        return self._verify_attachment(collision_object.id, attached.link_name)

    def _verify_attachment(self, name: str, link_name: str) -> bool:
        from moveit_msgs.srv import GetPlanningScene
        from moveit_msgs.msg import PlanningSceneComponents
        proxy = self._planner._module_proxy(self._context)
        if proxy is None:
            return False
        req = GetPlanningScene.Request()
        req.components.components = PlanningSceneComponents.ROBOT_STATE_ATTACHED_OBJECTS
        response = proxy.call(str(self._planner.config.get("planning_scene_module", "moveit_get_planning_scene")),
                              {"request": req, "raw_response": True},
                              timeout=float(self.hardware_config.get("planning_scene_timeout_s", 2.)))
        scene = getattr(response, "scene", None)
        verified = scene is not None and any(o.object.id == name and o.link_name == link_name for o in scene.robot_state.attached_collision_objects)
        log.info("MoveIt attachment readback name=%s link=%s verified=%s", name, link_name, verified)
        return verified

    def _apply_collision_object(self, collision_object) -> bool:
        applied = self._apply_planning_scene_diff(collision_objects=[collision_object])
        if self._collision_object_pub is not None:
            self._collision_object_pub.publish(collision_object)
        return applied

    def _apply_attached_collision_object(self, attached_collision_object) -> bool:
        applied = self._apply_planning_scene_diff(attached_collision_objects=[attached_collision_object])
        if self._attached_collision_object_pub is not None:
            self._attached_collision_object_pub.publish(attached_collision_object)
        return applied

    def _apply_planning_scene_diff(
        self,
        *,
        collision_objects: list | None = None,
        attached_collision_objects: list | None = None,
        allowed_collision_matrix: object | None = None,
    ) -> bool:
        if not self._ensure_scene_interfaces():
            return False
        try:
            from moveit_msgs.msg import PlanningScene
            from moveit_msgs.srv import ApplyPlanningScene
            import rclpy

            scene = PlanningScene()
            scene.is_diff = True
            scene.world.collision_objects = list(collision_objects or [])
            if allowed_collision_matrix is not None:
                scene.allowed_collision_matrix = allowed_collision_matrix
            scene.robot_state.is_diff = True
            scene.robot_state.attached_collision_objects = list(attached_collision_objects or [])
            client = self._apply_planning_scene_client
            if client is None or not client.wait_for_service(timeout_sec=1.0):
                return False
            request = ApplyPlanningScene.Request()
            request.scene = scene
            future = client.call_async(request)
            deadline = time.monotonic() + float(self.hardware_config.get("planning_scene_timeout_s", 2.0))
            while not future.done() and time.monotonic() < deadline:
                if self._owns_node:
                    rclpy.spin_once(self._ros_node, timeout_sec=0.02)
                else:
                    time.sleep(0.02)
            return bool(future.done() and future.result() and future.result().success)
        except Exception as exc:  # noqa: BLE001
            log.warn("MoveIt planning scene apply failed: %s", exc)
            return False

    def send_pose_goal(self, request: ArmPoseRequest) -> None:
        self._base_retry_contract_factory = None
        self.publish_pose_goal_marker(request.pose, label=request.segment)
        self._ensure_persistent_acm()
        self._status, self._error, self._phase = "in_progress", None, "planning"
        self._last_failure = None
        try:
            contract = self._pose_plan_contract(request)
            log.info("Resolved arm request: %s; defaults: %s", contract, self._pose_defaults)
            if request.motion == "cartesian_line":
                plan = self._planner.plan_cartesian_request(contract, context=self._context)
            else:
                attempt = 0
                def invocation_state():
                    nonlocal attempt
                    measured = contract if attempt == 0 else replace(contract, start_state_seed=self._current_joint_positions())
                    attempt += 1
                    return measured
                plan = self._plan_contract_with_retries(invocation_state, context=self._context, failure_reason="moveit_pose_planning_failed")
            if not plan.success or plan.trajectory is None:
                self._status, self._error = "failed", plan.reason or "moveit_pose_planning_failed"
                self._last_failure = plan.failure or FailureInfo()
                log.warn("MoveIt pose planning failed: %s metadata=%s", self._error, plan.metadata)
                return
            self._phase = "execution"
            self._send_executor_trajectory(plan.trajectory, base_bounds=(plan.metadata or {}).get("base_motion_bounds"))
        except Exception as exc:
            self._status, self._error = "failed", str(exc)
            self._last_failure = exc.failure if isinstance(exc, MotionFailure) else FailureInfo(FailureCode.INVALID_REQUEST if isinstance(exc, ValueError) else FailureCode.INVALID_STATE)
            log.exception("MoveItArmController pose goal failed: %s", exc)

    def _pose_plan_contract(self, request: ArmPoseRequest):
        settings = request.moveit
        state = self._current_joint_positions()
        constrained = bool(settings.constrain_base)
        mode = settings.goal_mode
        if mode is None:
            strategy = settings.planning_strategy or self._pose_defaults.planning_strategy
            group = settings.group_name or self._pose_defaults.group_name
            mode = "ik_joint_goal" if strategy == "ik_joint_constraints" or (strategy == "auto" and group == "arm") else "direct_pose"
        if mode == "ik_joint_goal" and not state:
            raise RuntimeError("moveit_start_state_unavailable")
        base = settings.base_pose
        if (constrained or settings.max_base_path_drift) and base is None:
            names = self._planner.config.get("base_joint_names", ["odom_x", "odom_y", "odom_t"])
            if state and all(name in state for name in names):
                base = BasePose2D(*(float(state[name]) for name in names), frame_id=str(self.hardware_config.get("odom", "odom")))
        if constrained and base is None:
            raise RuntimeError("moveit_base_constraint_requested_but_current_base_pose_unavailable")
        if base is not None and base.frame_id != str(self.hardware_config.get("odom", "odom")):
            raise ValueError("moveit_base_reference_frame_mismatch")
        overrides = {}
        for field, value in (("cartesian_max_step", request.line.max_step), ("cartesian_min_fraction", request.line.min_fraction), ("cartesian_jump_threshold", request.line.jump_threshold), ("avoid_collisions", request.line.avoid_collisions)):
            if value is not None:
                overrides[field] = value
        resolved = replace(self._pose_defaults, **overrides)
        # Keep the existing target-relaxed contact policy for pick segments.
        if settings.collision_policy == "target_relaxed":
            resolved = replace(resolved, collision_relaxation_objects=("pick_target_collision",))
        return MoveItPlanRequest(group_name=settings.group_name or self._pose_defaults.group_name, goal_mode=mode, pose=request.pose,
            ee_link=settings.ee_link or self._pose_defaults.ee_link, constrain_base=constrained, base_pose=base,
            base_constraint_tolerance=settings.base_constraint_tolerance if settings.base_constraint_tolerance is not None else self._pose_defaults.base_constraint_tolerance,
            max_base_path_drift=settings.max_base_path_drift, collision_policy=settings.collision_policy,
            start_state_seed=state, metadata={"segment": request.segment}, settings=resolved)

    def get_failure(self):
        if self._last_failure is not None:
            return self._last_failure
        executor = getattr(self, "_executor", None)
        if self._phase == "execution" and executor is not None and hasattr(executor, "get_failure"):
            return executor.get_failure()
        return FailureInfo() if self.get_status() == "failed" else None


class MoveItHeadController(BaseHeadController):
    """MoveIt2 head controller using the semantic ``head`` move group."""

    def __init__(self, hardware_config: dict):
        super().__init__(hardware_config)
        cfg = dict(hardware_config)
        cfg.setdefault("move_group", "head")
        cfg.setdefault("action_name", "/move_action")
        self._client = _MoveItMoveGroupClient(cfg, "moveit_head_client")
        self._pan_joint = str(cfg.get("pan_joint", "head_pan_joint"))
        self._tilt_joint = str(cfg.get("tilt_joint", "head_tilt_joint"))
        self._named_targets = cfg.get("named_targets", {})
        self._error: str | None = None

    def send_pan_tilt(self, pan: float, tilt: float) -> None:
        self._error = None
        self._client.send_joint_goal(
            [self._pan_joint, self._tilt_joint],
            [float(pan), float(tilt)],
        )
        if self._client.get_status() == "failed":
            self._error = self._client.get_error()

    def send_joint_goal(self, joints: dict[str, float]) -> None:
        names = []
        positions = []
        for joint_name in (self._pan_joint, self._tilt_joint):
            if joint_name in joints:
                names.append(joint_name)
                positions.append(float(joints[joint_name]))
        if not names:
            self._error = "no head joints supplied"
            return
        self._error = None
        self._client.send_joint_goal(names, positions)
        if self._client.get_status() == "failed":
            self._error = self._client.get_error()

    def send_named_goal(self, config_name: str) -> None:
        target = self._named_targets.get(config_name)
        if isinstance(target, dict):
            joints = target.get("joints")
            if isinstance(joints, dict):
                self.send_joint_goal({str(k): float(v) for k, v in joints.items()})
                return
            if self._pan_joint in target and self._tilt_joint in target:
                self.send_pan_tilt(float(target[self._pan_joint]), float(target[self._tilt_joint]))
                return
            if "pan" in target and "tilt" in target:
                self.send_pan_tilt(float(target["pan"]), float(target["tilt"]))
                return
        self._client.send_named_goal(config_name)
        if self._client.get_status() == "failed":
            self._error = self._client.get_error()

    def get_status(self) -> HeadStatus:
        return self._client.get_status()

    def get_error(self) -> str | None:
        return self._client.get_error() or self._error

    def cancel_goal(self) -> None:
        self._client.cancel_goal()


class MoveItGripperController(BaseGripperController):
    """MoveIt2 position gripper controller using the semantic ``gripper`` group.

    Idle close/open goals can be confirmed from measured opening (<= 5 cm for
    close; +/- 3 cm of the commanded width for open). Missing feedback and
    cancelled goals cannot be confirmed this way.
    """

    def __init__(self, hardware_config: dict):
        super().__init__(hardware_config)
        cfg = dict(hardware_config)
        cfg.setdefault("move_group", "gripper")
        cfg.setdefault("action_name", "/move_action")
        self._client = _MoveItMoveGroupClient(cfg, "moveit_gripper_client")
        self._state: str = "open"
        self._joint_name = str(cfg.get("joint_name", "hand_motor_joint"))
        self._min_width = float(cfg.get("min_width", cfg.get("closed_distance", 0.0)))
        self._max_width = float(cfg.get("max_width", cfg.get("open_distance", 0.08)))
        self._min_position = float(cfg.get("min_position", cfg.get("closed_position", 0.0)))
        self._max_position = float(cfg.get("max_position", cfg.get("open_position", 1.0)))
        self._open_distance = float(cfg.get("open_distance", self._max_width))
        self._closed_distance = float(cfg.get("closed_distance", self._min_width))
        self._joint_state_provider = cfg.get("joint_state_provider")
        self._target_width: float | None = None
        self._target_position: float | None = None
        self._target_state: str | None = None

    def _clamp_width(self, width_m: float) -> float:
        low = min(self._min_width, self._max_width)
        high = max(self._min_width, self._max_width)
        return max(low, min(high, float(width_m)))

    def _width_to_position(self, width_m: float) -> float:
        width = self._clamp_width(width_m)
        span = self._max_width - self._min_width
        if abs(span) < 1e-9:
            return self._min_position
        ratio = (width - self._min_width) / span
        return self._min_position + ratio * (self._max_position - self._min_position)

    def _position_to_width(self, position: float) -> float:
        span = self._max_position - self._min_position
        if abs(span) < 1e-9:
            return self._min_width
        ratio = (float(position) - self._min_position) / span
        return self._min_width + ratio * (self._max_width - self._min_width)

    def _send_width_goal(self, width_m: float, state_after_success: str) -> ControllerResult:
        width = self._clamp_width(width_m)
        position = self._width_to_position(width)
        log.warn('in send_width_goal width_m --', width_m, 'width --', width, 'position --', position)
        self._target_width = width
        self._target_position = position
        self._target_state = None
        self._state = "moving"
        self._client.send_joint_goal([self._joint_name], [position])
        if self._client.get_status() == "failed":
            return ControllerResult(success=False, error=self._client.get_error() or "moveit gripper goal failed")
        self._target_state = state_after_success
        self._state = state_after_success if self._client.get_status() == "succeeded" else "moving"
        return ControllerResult(success=True)

    def open(self, width_m: float | None = None) -> ControllerResult:
        return self._send_width_goal(
            self._open_distance if width_m is None else float(width_m),
            "open",
        )

    def close(self, force_n: float | None = None, width_m: float | None = None) -> ControllerResult:
        # MoveIt gripper is position-only. ``force_n`` is accepted for interface
        # compatibility but is not interpreted as force.
        log.warn('in close fn width_m --', width_m)
        target_width = self._closed_distance if width_m is None else float(width_m)
        return self._send_width_goal(target_width, "closed")

    def get_state(self) -> str:
        status = self._client.get_status()
        if status == "in_progress":
            return "moving"
        if status == "failed":
            return "stalled"
        if status == "succeeded":
            if self._state == "moving":
                distance = self.get_distance()
                midpoint = (self._min_width + self._max_width) * 0.5
                self._state = "open" if distance is not None and distance >= midpoint else "closed"
            return self._state
        return self._state if self._state in ("open", "closed") else "closed"

    def get_status(self) -> ArmStatus:
        status = self._client.get_status()
        if status == "idle" and self._target_state is not None:
            distance = self._measured_distance()
            matched = distance is not None and (
                (self._target_state == "closed" and distance <= .05) or
                (self._target_state == "open" and abs(distance - self._target_width) <= .03 + 1e-9))
            if matched:
                self._state = self._target_state
                log.info("Gripper idle accepted from measured opening: command=%s distance=%.4f m target=%.4f m",
                         self._target_state, distance, self._target_width)
                return "succeeded"
        if status == "succeeded":
            self.get_state()
        return status

    def get_error(self) -> str | None:
        return self._client.get_error()

    def cancel_goal(self) -> None:
        self._target_state = None
        self._client.cancel_goal()

    def get_distance(self) -> float | None:
        distance = self._measured_distance()
        return distance if distance is not None else self._target_width

    def _measured_distance(self) -> float | None:
        """Read feedback only; a commanded width cannot confirm an idle goal."""
        if self._joint_state_provider is None:
            return None
        try:
            joint_state = self._joint_state_provider()
        except Exception as exc:  # noqa: BLE001
            log.debug("MoveIt gripper joint state unavailable: %s", exc)
            return None
        names = list(getattr(joint_state, "name", []) or [])
        positions = list(getattr(joint_state, "position", []) or [])
        for index, name in enumerate(names):
            if str(name) == self._joint_name and index < len(positions):
                try:
                    position = float(positions[index])
                except (TypeError, ValueError):
                    return None
                if not math.isfinite(position):
                    return None
                return self._clamp_width(self._position_to_width(position))
        return None
