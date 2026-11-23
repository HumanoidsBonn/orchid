"""HSRB arm controller adapter using `hsrb_interface` JointGroup/whole_body."""

from __future__ import annotations

from contextlib import contextmanager
from threading import Lock
from typing import Any, Optional

from src.core.arm_request import ArmPoseRequest
from src.core.failure import FailureInfo, FailureCode
from src.core.controllers import BaseArmController, ArmStatus, Pose3D
from src.plugin.controllers.base_trajectory_sanitizer import (
    base_trajectory_diagnostics,
    duration_to_sec,
    required_base_segment_dt,
    sanitize_base_trajectory,
    sec_to_duration,
    unwrap_angle_near,
)

import src.core.logger as log


class _HsrbInterfacePlanner:
    """HSRB-specific plan-only wrapper around ``hsrb_interface`` JointGroup."""

    VALID_SCOPES = {"whole_body", "arm_only"}
    VALID_MOTIONS = {"pose", "cartesian_line"}

    def __init__(self, joint_group: Any):
        self._jg = joint_group

    def plan_pose(self, request: ArmPoseRequest) -> Any:
        pose = request.pose
        planning_scope = request.scope
        planning_motion = request.motion
        looking_hand_constraint = request.looking_hand_constraint
        with self._temporary_looking_hand_constraint(looking_hand_constraint):
            with self._temporary_base_movement_scope(planning_scope):
                if planning_motion == "cartesian_line":
                    return self._plan_cartesian_line(request)
                geom = __import__("hsrb_interface").geometry
                gp = geom.pose(pose.x, pose.y, pose.z, pose.roll, pose.pitch, pose.yaw)
                return self._jg.move_end_effector_pose(
                    gp,
                    ref_frame_id=pose.frame_id,
                    plan_only=True,
                )

    def plan_joint_positions(self, joints: dict[str, float]) -> Any:
        return self._jg.move_to_joint_positions(joints, plan_only=True)

    def _plan_cartesian_line(self, request: ArmPoseRequest) -> Any:
        axis = request.line.axis
        distance = request.line.distance
        if axis is None or distance is None:
            raise ValueError("cartesian_line planning requires line_axis and line_distance")
        if len(axis) != 3:
            raise ValueError("line_axis must contain exactly three values")
        return self._jg.move_end_effector_by_line(
            [float(axis[0]), float(axis[1]), float(axis[2])],
            float(distance),
            plan_only=True,
        )

    @contextmanager
    def _temporary_looking_hand_constraint(self, enabled: bool):
        previous = getattr(self._jg, "looking_hand_constraint", None)
        if previous is not None:
            self._jg.looking_hand_constraint = bool(enabled)
        try:
            yield
        finally:
            if previous is not None:
                self._jg.looking_hand_constraint = previous

    @contextmanager
    def _temporary_base_movement_scope(self, planning_scope: str):
        if planning_scope != "arm_only":
            yield
            return

        original = self._jg._generate_planning_request

        def _generate_arm_only_request(request_type):
            request = original(request_type)
            base_movement_type = getattr(request, "base_movement_type", None)
            if base_movement_type is not None and hasattr(type(base_movement_type), "NONE"):
                base_movement_type.val = type(base_movement_type).NONE
            elif base_movement_type is not None and hasattr(base_movement_type, "NONE"):
                base_movement_type.val = base_movement_type.NONE
            return request

        self._jg._generate_planning_request = _generate_arm_only_request
        try:
            yield
        finally:
            self._jg._generate_planning_request = original





class HsrbArmController(BaseArmController):
    """Adapter that uses `hsrb_interface`'s JointGroup/whole_body for arm motion.

    hardware_config keys:
    - name: name of joint_group resource (default: "whole_body")
    - named_targets: optional mapping of named config to joint dicts
    """

    DEFAULT_ROS2_CONTROL_CONTROLLERS = [
        {
            "name": "arm",
            "action": "/arm_trajectory_controller/follow_joint_trajectory",
            "joints": [
                "arm_lift_joint",
                "arm_flex_joint",
                "arm_roll_joint",
                "wrist_flex_joint",
                "wrist_roll_joint",
            ],
            "group_type": "primary",
        },
        {
            "name": "head",
            "action": "/head_trajectory_controller/follow_joint_trajectory",
            "joints": ["head_pan_joint", "head_tilt_joint"],
            "group_type": "primary",
        },
        {
            "name": "omni_base",
            "action": "/omni_base_controller/follow_joint_trajectory",
            "joints": ["odom_x", "odom_y", "odom_t"],
            "group_type": "base_trajectory",
        },
    ]

    def __init__(self, hardware_config: dict):
        super().__init__(hardware_config)
        self._available = False
        self._hsrb_available = False
        self._jg = None
        self._planner: _HsrbInterfacePlanner | None = None
        self._lock = Lock()
        self._execution_backend = str(hardware_config.get("execution_backend", "hsrb_interface"))
        self._planning_backend = str(hardware_config.get("planning_backend", "hsrb_interface_plan_only"))
        self._ros2_control = None
        if self._execution_backend == "ros2_control":
            from src.plugin.controllers.ros2_control_trajectory import Ros2ControlController

            self._ros2_control = Ros2ControlController(
                hardware_config,
                joint_state_provider=self._current_joint_state,
            )
            self._available = True
        try:
            import hsrb_interface
            self._robot = hsrb_interface.Robot()
            # log.info('Robot List', self._robot.list())
            self._collision_world = self._robot.get("global_collision_world", None)
            if self._collision_world is None:
                log.error("self._collision_world is None")
            name = hardware_config.get("name", "whole_body")
            self._jg = self._robot.get(name)
            if self._jg is None:
                log.error("self._jg is None")
            log.info(
                "HSRB arm resources: collision_world=%s joint_group=%s",
                type(self._collision_world).__name__,
                type(self._jg).__name__,
            )
            log.info("HSRB arm end_effector_frame: %s", self._jg.end_effector_frame)
            self._jg.collision_world = self._collision_world
            self._jg.collision_world.remove_all()
            self._planner = _HsrbInterfacePlanner(self._jg)
            self._available = True
            self._hsrb_available = True
        except Exception as exc:  # pragma: no cover - adapter best-effort
            log.exception("hsrb_interface JointGroup unavailable: %s", exc)
        self.status: ArmStatus = "idle"
        self._error: str | None = None
    
    def add_collision_mesh(self, mesh_file_name, name: str = "precise_obstacle_mesh", *, frame_id: str | None = None) -> bool:
        try:
            self._collision_world.add_mesh(
                filename=mesh_file_name,
                frame_id=frame_id or 'odom',
                name=name,
            )
            return True
        except Exception as exc:  # pragma: no cover - adapter best-effort
            log.exception("hsrb_interface Failed to add mesh file: %s", exc)
            return False

    def remove_collision_object(self, name: str) -> None:
        try:
            if self._collision_world is not None:
                self._collision_world.remove(name)
        except Exception as exc:  # pragma: no cover - adapter best-effort
            log.warn("Failed to remove HSRB collision object %s: %s", name, exc)

    def clear_collision_scene(self) -> None:
        try:
            if self._collision_world is not None:
                self._collision_world.remove_all()
        except Exception as exc:  # pragma: no cover - adapter best-effort
            log.warn("Failed to clear HSRB collision scene: %s", exc)

    def send_pose_goal(self, request: ArmPoseRequest) -> None:
        self._last_failure = None
        if not self._hsrb_available or self._jg is None:
            self.status = "failed"
            self._error = "HSRB interface is unavailable for Cartesian pose planning"
            return
        try:
            self._error = None
            self.status = "in_progress"
            planner = getattr(self, "_planner", None) or _HsrbInterfacePlanner(self._jg)
            traj = planner.plan_pose(request)
            if traj is None:
                self.status = "failed"
                self._error = "HSRB pose planner returned no trajectory"
                return 
            if self._execution_backend == "ros2_control":
                if self._ros2_control is None:
                    self.status = "failed"
                    self._error = "ros2_control backend was not initialized"
                    return
                self._ros2_control.send_prepared_trajectory_goals(
                    self._prepare_hsrb_ros2_control_trajectory_batches(traj)
                )
                self.status = self._ros2_control.get_status()
                self._error = self._ros2_control.get_error()
                return
            self._execute_trajectory_with_pending_goal_retry(traj)
        except Exception as exc:
            self._last_failure = FailureInfo(FailureCode.INVALID_REQUEST if isinstance(exc, ValueError) else FailureCode.UNKNOWN)
            self._error = str(exc)
            log.exception("Failed to send HSRB arm pose goal: %s", exc)
            self.status = "failed"

    def get_failure(self):
        if self.get_status() != "failed":
            return None
        executor = getattr(self, "_ros2_control", None)
        if executor is not None and executor.get_status() == "failed":
            return executor.get_failure()
        return getattr(self, "_last_failure", None) or FailureInfo()

    def send_joint_goal(self, joints: dict[str, float]) -> None:
        log.info(
            "HSRB arm send_joint_goal backend=%s joints=%s",
            self._execution_backend,
            {name: float(value) for name, value in joints.items()},
        )
        if self._execution_backend == "ros2_control":
            self.status = "in_progress"
            self._error = None
            if self._ros2_control is None:
                self.status = "failed"
                self._error = "ros2_control backend was not initialized"
                return
            self._ros2_control.send_joint_goal(joints)
            self.status = self._ros2_control.get_status()
            self._error = self._ros2_control.get_error()
            return
        if not self._available or self._jg is None:
            self.status = "failed"
            self._error = "HSRB interface is unavailable"
            return
        try:
            self._error = None
            self.status = "in_progress"
            planner = getattr(self, "_planner", None) or _HsrbInterfacePlanner(self._jg)
            traj = planner.plan_joint_positions(joints)
            if traj is None:
                self.status = "failed"
                self._error = "HSRB joint planner returned no trajectory"
                return
            self._execute_trajectory_with_pending_goal_retry(traj)
        except Exception as exc:
            self._error = str(exc)
            log.exception("Failed to send HSRB joint goal: %s", exc)
            self.status = "failed"

    def send_synchronized_trajectory_goal(self, trajectory, *, cancelled=None) -> None:
        if self._execution_backend != "ros2_control" or self._ros2_control is None:
            raise NotImplementedError("synchronized pickup requires the ros2_control execution backend")
        self.status = "in_progress"
        self._error = None
        self._ros2_control.send_synchronized_trajectory_goal(trajectory, cancelled=cancelled)
        self.status = self._ros2_control.get_status()
        self._error = self._ros2_control.get_error()

    def send_trajectory_goal(self, trajectory) -> None:
        if self._execution_backend == "ros2_control":
            self.status = "in_progress"
            self._error = None
            if self._ros2_control is None:
                self.status = "failed"
                self._error = "ros2_control backend was not initialized"
                return
            if isinstance(trajectory, dict):
                self.send_joint_goal(trajectory)
                return
            if self._is_prepared_ros2_control_batches(trajectory):
                self._ros2_control.send_prepared_trajectory_goals(trajectory)
            else:
                self._ros2_control.send_prepared_trajectory_goals(
                    self._prepare_hsrb_ros2_control_trajectory_batches(trajectory)
                )
            self.status = self._ros2_control.get_status()
            self._error = self._ros2_control.get_error()
            return
        if not self._available or self._jg is None:
            self.status = "failed"
            return
        try:
            self._error = None
            self.status = "in_progress"
            self._execute_trajectory_with_pending_goal_retry(trajectory)
        except Exception as exc:
            self._error = str(exc)
            log.exception("Failed to execute HSRB trajectory goal: %s", exc)
            self.status = "failed"

    @staticmethod
    def _is_prepared_ros2_control_batches(value: Any) -> bool:
        if not isinstance(value, list) or not value:
            return False
        first_batch = value[0]
        if not isinstance(first_batch, list) or not first_batch:
            return False
        first_item = first_batch[0]
        return (
            isinstance(first_item, tuple)
            and len(first_item) == 2
            and isinstance(first_item[0], dict)
            and "action" in first_item[0]
        )

    def send_named_goal(self, config_name: str, *, force_execution: bool = False) -> None:
        _ = force_execution
        if self._execution_backend != "ros2_control" and (not self._available or self._jg is None):
            self.status = "failed"
            self._error = "HSRB interface is unavailable"
            return
        try:
            log.info("HSRB arm executing named target '%s' via %s", config_name, self._execution_backend)
            super().send_named_goal(config_name)
        except Exception as exc:  # noqa: BLE001
            self._error = str(exc)
            log.exception("Failed to execute named target %s", config_name)
            self.status = "failed"

    def update_pose_goal(self, pose: Pose3D) -> None:
        # HSRB joint_group doesn't support goal update; resend
        self.send_pose_goal(ArmPoseRequest(pose))

    def get_status(self) -> ArmStatus:
        if self.status == "failed":
            return "failed"
        if self._execution_backend == "ros2_control":
            if self._ros2_control is None:
                return "failed"
            if self.status != "in_progress":
                return self.status
            self.status = self._ros2_control.get_status()
            self._error = self._ros2_control.get_error()
            return self.status
        if not self._available or self._jg is None:
            return "failed"
        try:
            if self.status == "failed":
                return "failed"
            elif self._jg.is_moving():
                return "in_progress"
            elif self._jg.is_succeeded():
                self.status = "succeeded"
                return "succeeded"
            return self.status
        except Exception as exc:
            if self._is_pending_goal_handle_error(exc) and self.status == "in_progress":
                log.debug("HSRB arm goal handle is not ready yet; treating as in_progress")
                return "in_progress"
            self._error = str(exc)
            log.exception("Failed to poll HSRB arm status: %s", exc)
            return "failed"

    def get_error(self) -> str | None:
        return self._error

    def cancel_goal(self) -> None:
        if self._execution_backend == "ros2_control":
            if self._ros2_control is not None:
                self._ros2_control.cancel_goal()
            return
        if not self._available or self._jg is None:
            return
        try:
            self._jg.cancel_goal()
        except Exception as exc:
            if self._is_pending_goal_handle_error(exc):
                log.debug("Ignoring HSRB arm cancel before goal handle is ready")
                return
            log.exception("Failed to cancel HSRB arm goal")

    def _execute_trajectory_with_pending_goal_retry(self, trajectory) -> None:
        try:
            self._jg.execute(trajectory)
            return
        except Exception as exc:
            if not self._is_pending_goal_handle_error(exc):
                raise

        log.debug(
            "HSRB JointGroup.execute hit pending goal handle during internal cancel; "
            "clearing stale clients and retrying once"
        )
        self._clear_pending_current_clients()
        self._jg.execute(trajectory)

    def _clear_pending_current_clients(self) -> None:
        try:
            setattr(self._jg, "_current_clients", None)
        except Exception as exc:  # pragma: no cover - defensive adapter guard
            log.debug("Unable to clear HSRB pending current clients: %s", exc)

    def _current_joint_state(self):
        try:
            if self._jg is not None and hasattr(self._jg, "_get_joint_state"):
                return self._jg._get_joint_state()
        except Exception as exc:  # pragma: no cover - diagnostic only
            log.debug("Unable to read HSRB current joint state for trajectory extraction: %s", exc)
        return None

    def _prepare_hsrb_ros2_control_trajectory_batches(self, trajectory: Any) -> list[list[tuple[dict[str, Any], Any]]]:
        """Adapt a HSRB whole-body trajectory into controller-ready batches.

        This mirrors ``hsrb_interface.trajectory.extract`` closely, but keeps the
        HSRB-specific joint ownership and base dispatch ordering out of the
        generic ros2_control sender.
        """
        source_names = [str(name) for name in getattr(trajectory, "joint_names", []) or []]
        if not source_names:
            raise ValueError("HSRB planned trajectory has no joint names")

        current_joint_state = self._current_joint_state()
        current_positions = self._joint_state_positions(current_joint_state)
        primary: list[tuple[dict[str, Any], Any]] = []
        base: list[tuple[dict[str, Any], Any]] = []
        remaining = set(source_names)

        for cfg in self._hsrb_ros2_control_controller_configs():
            controller_joints = [str(name) for name in cfg.get("joints", [])]
            present_names = [name for name in source_names if name in controller_joints]
            if not present_names:
                continue
            missing = [name for name in controller_joints if name not in source_names]
            missing_without_state = [name for name in missing if name not in current_positions]
            if missing_without_state:
                raise ValueError(
                    "Cannot adapt HSRB trajectory for controller "
                    f"{cfg.get('name') or cfg.get('action')}: missing joints "
                    f"{missing_without_state} and no current joint state fallback"
                )

            remaining -= set(present_names)
            subset = self._extract_hsrb_controller_trajectory(
                trajectory,
                controller_joints,
                current_positions=current_positions,
                controller_config=cfg,
            )
            if str(cfg.get("group_type")) == "base_trajectory" and bool(
                cfg.get("sanitize_base_trajectory", self.hardware_config.get("sanitize_base_trajectory", False))
            ):
                subset = sanitize_base_trajectory(
                    subset,
                    current_positions=current_positions,
                    controller_config=cfg,
                    log_prefix="HSRB",
                )
            entry = (cfg, subset)
            if str(cfg.get("group_type")) == "base_trajectory":
                base.append(entry)
            else:
                primary.append(entry)

        if remaining:
            log.warn("HSRB trajectory joints not owned by configured ros2_control controllers: %s", sorted(remaining))
        if not primary and not base:
            raise ValueError(f"No HSRB ros2_control controller owns trajectory joints: {source_names}")

        dispatch_mode = self._hsrb_ros2_control_dispatch_mode()
        if dispatch_mode == "synchronized":
            synchronized = [*primary, *base]
            log.info(
                "HSRB ros2_control dispatch_mode=synchronized controllers=%s",
                [entry[0].get("name") or entry[0].get("action") for entry in synchronized],
            )
            return [synchronized]
        if dispatch_mode != "staged":
            log.warn(
                "Unknown HSRB ros2_control dispatch_mode=%s; using staged dispatch",
                dispatch_mode,
            )

        batches: list[list[tuple[dict[str, Any], Any]]] = []
        if primary:
            batches.append(primary)
        if base:
            batches.append(base)
        log.info(
            "HSRB ros2_control dispatch_mode=staged batches=%s",
            [[entry[0].get("name") or entry[0].get("action") for entry in batch] for batch in batches],
        )
        return batches

    def _hsrb_ros2_control_controller_configs(self) -> list[dict[str, Any]]:
        cfg = self.hardware_config.get("ros2_control") or {}
        controllers = cfg.get("controllers") or self.hardware_config.get("trajectory_controllers")
        defaults = {
            key: cfg[key]
            for key in ("time_scale", "velocity_scale", "acceleration_scale")
            if key in cfg
        }
        if isinstance(controllers, dict):
            return [
                {"name": name, **defaults, **value}
                for name, value in controllers.items()
                if isinstance(value, dict) and value.get("action")
            ]
        if isinstance(controllers, list):
            return [{**defaults, **c} for c in controllers if isinstance(c, dict) and c.get("action")]
        return [{**defaults, **c} for c in self.DEFAULT_ROS2_CONTROL_CONTROLLERS]

    def _hsrb_ros2_control_dispatch_mode(self) -> str:
        cfg = self.hardware_config.get("ros2_control") or {}
        mode = cfg.get("dispatch_mode", self.hardware_config.get("trajectory_dispatch_mode", "synchronized"))
        return str(mode).strip().lower()

    @staticmethod
    def _extract_hsrb_controller_trajectory(
        trajectory: Any,
        joint_names: list[str],
        *,
        current_positions: dict[str, float],
        controller_config: dict[str, Any] | None = None,
    ) -> Any:
        from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

        controller_config = controller_config or {}
        time_scale = HsrbArmController._positive_scale(controller_config.get("time_scale"), default=1.0)
        velocity_scale = HsrbArmController._nonnegative_scale(controller_config.get("velocity_scale"), default=1.0)
        acceleration_scale = HsrbArmController._nonnegative_scale(controller_config.get("acceleration_scale"), default=1.0)
        source_names = [str(name) for name in getattr(trajectory, "joint_names", []) or []]
        num_source_joints = len(source_names)
        index_map = {
            joint_name: source_names.index(joint_name)
            for joint_name in joint_names
            if joint_name in source_names
        }

        trajectory_out = JointTrajectory()
        trajectory_out.header = getattr(trajectory, "header", trajectory_out.header)
        trajectory_out.joint_names = list(joint_names)
        trajectory_out.points = []

        for source in getattr(trajectory, "points", []) or []:
            target = JointTrajectoryPoint()
            target.positions = [0.0] * len(joint_names)
            target.velocities = [0.0] * len(joint_names)
            target.accelerations = [0.0] * len(joint_names)
            target.time_from_start = HsrbArmController._scaled_duration(
                source.time_from_start,
                time_scale,
            )

            source_positions = list(getattr(source, "positions", []) or [])
            source_velocities = list(getattr(source, "velocities", []) or [])
            source_accelerations = list(getattr(source, "accelerations", []) or [])
            has_velocities = len(source_velocities) == num_source_joints
            has_accelerations = len(source_accelerations) == num_source_joints

            for joint_index, joint_name in enumerate(joint_names):
                source_index = index_map.get(joint_name)
                if source_index is not None:
                    if source_index >= len(source_positions):
                        raise ValueError(
                            f"HSRB trajectory point lacks position for joint {joint_name}"
                        )
                    target.positions[joint_index] = float(source_positions[source_index])
                    if has_velocities:
                        target.velocities[joint_index] = float(source_velocities[source_index]) * velocity_scale
                    if has_accelerations:
                        target.accelerations[joint_index] = float(source_accelerations[source_index]) * acceleration_scale
                else:
                    target.positions[joint_index] = float(current_positions[joint_name])

            trajectory_out.points.append(target)

        return trajectory_out

    @staticmethod
    def _sanitize_hsrb_base_trajectory(
        trajectory: Any,
        *,
        current_positions: dict[str, float],
        controller_config: dict[str, Any],
    ) -> Any:
        return sanitize_base_trajectory(
            trajectory,
            current_positions=current_positions,
            controller_config=controller_config,
            log_prefix="HSRB",
        )

    @staticmethod
    def _base_trajectory_diagnostics(trajectory: Any, *, limits: tuple[float, float, float] | None = None) -> dict[str, Any]:
        return base_trajectory_diagnostics(trajectory, limits=limits)

    @staticmethod
    def _required_base_segment_dt(
        previous: list[float],
        current: list[float],
        *,
        min_dt: float,
        max_vx: float,
        max_vy: float,
        max_wz: float,
    ) -> float:
        return required_base_segment_dt(
            previous,
            current,
            min_dt=min_dt,
            max_vx=max_vx,
            max_vy=max_vy,
            max_wz=max_wz,
        )

    @staticmethod
    def _unwrap_angle_near(angle: float, reference: float) -> float:
        return unwrap_angle_near(angle, reference)

    @staticmethod
    def _duration_to_sec(duration: Any) -> float:
        return duration_to_sec(duration)

    @staticmethod
    def _sec_to_duration(seconds: float) -> Any:
        return sec_to_duration(seconds)

    @staticmethod
    def _positive_scale(value: Any, *, default: float) -> float:
        try:
            scale = float(value)
            return scale if scale > 0.0 else default
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _nonnegative_scale(value: Any, *, default: float) -> float:
        try:
            scale = float(value)
            return scale if scale >= 0.0 else default
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _scaled_duration(duration: Any, scale: float) -> Any:
        try:
            total_ns = (
                int(getattr(duration, "sec", 0)) * 1_000_000_000
                + int(getattr(duration, "nanosec", 0))
            )
            scaled_ns = max(0, int(round(total_ns * scale)))
            scaled = type(duration)()
            scaled.sec = scaled_ns // 1_000_000_000
            scaled.nanosec = scaled_ns % 1_000_000_000
            return scaled
        except Exception:  # noqa: BLE001
            return duration

    @staticmethod
    def _joint_state_positions(joint_state: Any | None) -> dict[str, float]:
        if joint_state is None:
            return {}
        names = list(getattr(joint_state, "name", []) or [])
        positions = list(getattr(joint_state, "position", []) or [])
        return {
            str(name): float(positions[index])
            for index, name in enumerate(names)
            if index < len(positions)
        }

    @staticmethod
    def _is_pending_goal_handle_error(exc: Exception) -> bool:
        return (
            isinstance(exc, AttributeError)
            and "'NoneType' object has no attribute 'get_result_async'" in str(exc)
        )
