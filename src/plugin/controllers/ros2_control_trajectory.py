"""Small direct ros2_control FollowJointTrajectory dispatcher."""

from __future__ import annotations
from src.core.failure import FailureInfo, FailureCode, MotionFailure, trajectory_failure

import time
import copy
import math
from dataclasses import dataclass
from typing import Any, Callable

import src.core.logger as log


@dataclass
class _ControllerGoal:
    name: str
    action: str
    joints: list[str]
    goal_handle_future: Any = None
    result_future: Any = None
    accepted: bool | None = None
    trajectory: Any = None
    acceptance_attempts: int = 0
    tracking_only: bool = False
    acceptance_error_norm: float | None = None
    trajectory_diagnostics: dict[str, Any] | None = None
    expanded_from_partial: bool = False
    requested_joints: list[str] | None = None
    cancel_requested: bool = False


class Ros2ControlController:
    """Generic ros2_control FollowJointTrajectory controller.

    This class owns action clients, result/cancel tracking, tolerance-based
    completion, and readable FollowJointTrajectory failures. Robot-specific
    trajectory shaping belongs in the caller; inputs here are already
    controller-ready joint goals or JointTrajectory messages.
    """

    DEFAULT_CONTROLLERS: list[dict[str, Any]] = []

    def __init__(
        self,
        hardware_config: dict[str, Any],
        *,
        joint_state_provider: Callable[[], Any | None] | None = None,
    ):
        self.hardware_config = hardware_config
        self.base_path_tolerance = {'odom_x': 0.25, 'odom_y': 0.25, 'odom_t': math.pi / 6.0}
        self.base_path_tolerance.update(hardware_config.get('base_path_tolerance_by_joint') or {})
        for name, value in self.base_path_tolerance.items():
            value = float(value)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f'base_path_tolerance_by_joint.{name} must be finite and positive')
            self.base_path_tolerance[name] = value
        self.duration_s = float(hardware_config.get("trajectory_duration_s", 3.0))
        self.wait_for_server_s = float(hardware_config.get("action_server_timeout_s", 2.0))
        self.synchronized_start_lead_s = float(hardware_config.get("synchronized_start_lead_s", 5.0))
        if not math.isfinite(self.synchronized_start_lead_s) or self.synchronized_start_lead_s < 1.0:
            raise ValueError("synchronized_start_lead_s must be finite and at least 1 second")
        self.goal_tolerance = float(
            hardware_config.get(
                "goal_tolerance",
                hardware_config.get("named_target_tolerance", 0.03),
            )
        )
        self.goal_tolerance_by_joint = {
            str(name): float(value)
            for name, value in (hardware_config.get("goal_tolerance_by_joint") or {}).items()
        }
        self._joint_state_provider = joint_state_provider
        self._node = hardware_config.get("ros2_node")
        self._owns_node = False
        self._clients: dict[str, Any] = {}
        self._active_goals: list[_ControllerGoal] = []
        self._cancel_futures: list[Any] = []
        self._final_targets: dict[str, float] = {}
        self._error: str | None = None
        self._failure = None
        self._status = "idle"
        self.progress_timeout_s = float(hardware_config.get("progress_timeout_s", 8.0))
        self.joint_state_stale_timeout_s = float(hardware_config.get("joint_state_stale_timeout_s", 2.0))
        self.progress_epsilon = float(hardware_config.get("progress_epsilon", 0.002))
        self.goal_response_retries = int(hardware_config.get("goal_response_retries", 2))
        self.goal_response_retry_settle_s = float(hardware_config.get("goal_response_retry_settle_s", 0.25))
        self._last_progress_time = time.monotonic()
        self._last_joint_state_time = time.monotonic()
        self._last_error_norm: float | None = None

    def send_joint_goal(self, joints: dict[str, float]) -> None:
        self._failure = None
        self._error = None
        self._active_goals = []
        self._cancel_futures = []
        self._final_targets = {}
        try:
            goal_map = self._split_by_controller(joints)
            if not goal_map:
                self._status = "failed"
                self._error = f"No ros2_control controller owns requested joints: {sorted(joints)}"
                log.error(self._error)
                return
            for controller, subset in goal_map:
                self._send_controller_goal(controller, subset)
                self._record_joint_final_targets(subset)
            if not self._wait_for_goal_acceptance(self._active_goals):
                return
            self._reset_progress_tracking()
            self._status = "in_progress"
        except Exception as exc:  # noqa: BLE001
            self._fail(str(exc))
            log.exception("Failed to send ros2_control joint goal: %s", exc)

    def send_trajectory_goal(self, trajectory: Any) -> None:
        self._failure = None
        """Dispatch a controller-ready JointTrajectory through owned ros2_control actions."""
        self._error = None
        self._active_goals = []
        self._cancel_futures = []
        self._final_targets = {}
        goal_map = self._split_trajectory_by_controller(trajectory)
        if not goal_map:
            joint_names = list(getattr(trajectory, "joint_names", []) or [])
            self._status = "failed"
            self._error = f"No ros2_control controller owns trajectory joints: {joint_names}"
            log.error(self._error)
            return
        self.send_prepared_trajectory_goals([goal_map])

    def send_prepared_trajectory_goals(
        self,
        batches: list[list[tuple[dict[str, Any] | _ControllerGoal, Any]]],
    ) -> None:
        self._failure = None
        """Dispatch already prepared controller trajectories.

        Each inner list is sent as one acceptance batch. Later batches are sent
        only after all earlier batches are accepted, allowing higher layers to
        encode robot-specific safety ordering without making this client know
        about robot-specific joint names or controller groups.
        """
        self._error = None
        self._active_goals = []
        self._cancel_futures = []
        self._final_targets = {}
        normalized_batches = [
            [self._normalize_prepared_goal(goal, trajectory) for goal, trajectory in batch]
            for batch in batches
            if batch
        ]
        if not normalized_batches:
            self._status = "failed"
            self._error = "No ros2_control prepared trajectories supplied"
            log.error(self._error)
            return
        try:
            for batch in normalized_batches:
                for controller, subset_trajectory in batch:
                    self._send_controller_trajectory_goal(controller, subset_trajectory)
                    self._record_trajectory_final_targets(subset_trajectory)
                if not self._wait_for_goal_acceptance(
                    [goal for goal, _trajectory in batch]
                ):
                    return
            self._reset_progress_tracking()
            self._status = "in_progress"
        except Exception as exc:  # noqa: BLE001
            self._fail(str(exc))
            log.exception("Failed to send ros2_control trajectory goal: %s", exc)

    def send_synchronized_trajectory_goal(self, trajectory: Any, *, cancelled=None) -> None:
        """One clock/timeline; no independent retries or tracking-only acceptance."""
        from rclpy.duration import Duration
        self._active_goals = []
        self._cancel_futures = []
        self._final_targets = {}
        self._failure = None
        self._error = None
        self._status = "dispatching"
        cancelled = cancelled or (lambda: False)
        try:
            if cancelled():
                raise RuntimeError("synchronized dispatch cancelled")
            if not getattr(trajectory, "points", None):
                raise ValueError("empty synchronized trajectory")
            goal_map = self._split_trajectory_by_controller(copy.deepcopy(trajectory))
            owned = {name for _, part in goal_map for name in part.joint_names}
            if not goal_map or owned != set(trajectory.joint_names):
                raise ValueError("synchronized trajectory has unowned joints")
            for goal, _ in goal_map:
                if not self._client(goal.action).wait_for_server(timeout_sec=self.wait_for_server_s):
                    raise MotionFailure(f"controller unavailable before dispatch: {goal.action}",
                        FailureInfo(FailureCode.TIMEOUT, retryable=True, backend_code="controller_unavailable"))
            lead = max(self.synchronized_start_lead_s,
                       self.wait_for_server_s + len(goal_map) * self.goal_response_retry_settle_s + .5)
            clock = self._get_node().get_clock()
            issued = clock.now()
            start = issued + Duration(seconds=lead)
            deadline = time.monotonic() + lead - .2
            # An explicit zero-derivative hold starts now. Future-dating the
            # entire message lets JTC interpolate from stale commanded
            # derivatives before waypoint zero, even when the robot is static.
            prepared = self._stationary_prefix(trajectory, lead, issued)
            goal_map = self._split_trajectory_by_controller(prepared)
            log.info("Synchronized trajectory hold_s=%.3f issued_ns=%s motion_start_ns=%s",
                     lead, issued.nanoseconds, start.nanoseconds)
            for goal, part in goal_map:
                if cancelled():
                    raise RuntimeError("synchronized dispatch cancelled")
                self._send_controller_trajectory_goal(goal, part)
                self._record_trajectory_final_targets(part)
            pending = list(self._active_goals)
            while pending and time.monotonic() < deadline and clock.now() < start - Duration(seconds=.2):
                if cancelled():
                    raise RuntimeError("synchronized dispatch cancelled")
                self._spin_once(.01)
                for goal in list(pending):
                    if goal.goal_handle_future.done():
                        if not self._accept_goal(goal):
                            return
                        pending.remove(goal)
            if cancelled():
                raise RuntimeError("synchronized dispatch cancelled")
            if pending or clock.now() >= start - Duration(seconds=.2):
                pending_details = [f"{goal.name} ({goal.action})" for goal in pending]
                accepted_details = [f"{goal.name} ({goal.action})" for goal in self._active_goals if goal.accepted]
                raise RuntimeError(
                    "synchronized batch not accepted before start deadline; "
                    f"pending={pending_details}; accepted={accepted_details}; "
                    f"lead_s={lead:.3f}; clock_now_ns={clock.now().nanoseconds}; "
                    f"scheduled_start_ns={start.nanoseconds}"
                )
            self._reset_progress_tracking()
            self._status = "in_progress"
        except Exception as exc:
            self._failure = (FailureInfo(FailureCode.CANCELLED) if cancelled() else
                             getattr(exc,"failure",FailureInfo(FailureCode.CONTROLLER_FAILED)))
            self._fail(str(exc))

    @staticmethod
    def _stationary_prefix(trajectory: Any, lead_s: float, issued: Any):
        """Keep prepared motion intact after an explicit stationary reference."""
        from rclpy.duration import Duration
        result = copy.deepcopy(trajectory)
        size = len(result.joint_names)
        if not result.points or len(result.points[0].positions) != size:
            raise ValueError("invalid synchronized trajectory start")
        hold = copy.deepcopy(result.points[0])
        hold.velocities = [0.] * size
        hold.accelerations = [0.] * size
        hold.time_from_start = Duration().to_msg()
        offset = round(lead_s * 1e9)
        for point in result.points:
            ns = point.time_from_start.sec * 10**9 + point.time_from_start.nanosec
            point.time_from_start = Duration(nanoseconds=ns + offset).to_msg()
        for point in (result.points[0], result.points[-1]):
            point.velocities = [0.] * size
            point.accelerations = [0.] * size
        result.points.insert(0, hold)
        result.header.stamp = issued.to_msg()
        return result

    def get_status(self) -> str:
        if self._status != "in_progress":
            return self._status
        try:
            try:
                from action_msgs.msg import GoalStatus
            except Exception:  # noqa: BLE001
                GoalStatus = None

            all_done = True
            fallback_tracking_needed = False
            for goal in self._active_goals:
                if goal.goal_handle_future is not None and not goal.goal_handle_future.done():
                    self._spin_once(0.0)
                    all_done = False
                    continue
                if goal.tracking_only:
                    all_done = False
                    fallback_tracking_needed = True
                    continue
                if goal.accepted is None:
                    if not self._accept_goal(goal):
                        return self._status
                if goal.result_future is None:
                    all_done = False
                    fallback_tracking_needed = True
                    continue
                if goal.result_future is not None and not goal.result_future.done():
                    self._spin_once(0.0)
                    all_done = False
                    continue
                result = goal.result_future.result()
                status = getattr(result, "status", None)
                succeeded = getattr(GoalStatus, "STATUS_SUCCEEDED", 4)
                if status != succeeded:
                    self._mark_failed_result(goal, result, status)
                    return self._status
                action_result = getattr(result, "result", None)
                error_code = getattr(action_result, "error_code", 0)
                if action_result is not None and error_code not in (0, getattr(type(action_result), "SUCCESSFUL", 0)):
                    self._mark_failed_result(goal, result, status)
                    return self._status
            if all_done:
                self._status = "succeeded"
                self._error = None
                log.info("ros2_control trajectory completed from action result")
                return self._status
            if not fallback_tracking_needed:
                return self._status
            if self._final_targets_reached():
                self._status = "succeeded"
                self._error = None
                log.info(
                    "ros2_control trajectory completed from joint tolerance: %s",
                    self._final_target_error_summary(),
                )
                self._cancel_active_goals()
            elif self._detect_stalled_progress():
                self._cancel_active_goals()
            elif self._final_targets and self._joint_state_provider is not None:
                log.debug(
                    "ros2_control final target error while waiting: %s",
                    self._final_target_error_summary(),
                )
            return self._status
        except Exception as exc:  # noqa: BLE001
            self._status = "failed"
            self._error = str(exc)
            log.exception("Failed to poll ros2_control trajectory status: %s", exc)
            return self._status

    def get_error(self) -> str | None:
        return self._error

    def get_failure(self):
        return getattr(self, "_failure", None) or (FailureInfo() if self._status == "failed" else None)

    def _record_joint_final_targets(self, joints: dict[str, float]) -> None:
        for name, position in joints.items():
            self._final_targets[str(name)] = float(position)

    def _record_trajectory_final_targets(self, trajectory: Any) -> None:
        points = list(getattr(trajectory, "points", []) or [])
        if not points:
            return
        joint_names = [str(name) for name in getattr(trajectory, "joint_names", []) or []]
        positions = list(getattr(points[-1], "positions", []) or [])
        for index, joint_name in enumerate(joint_names):
            if index < len(positions):
                self._final_targets[joint_name] = float(positions[index])

    def _reset_progress_tracking(self) -> None:
        now = time.monotonic()
        self._last_progress_time = now
        self._last_joint_state_time = now
        self._last_error_norm = self._final_target_error_norm()

    def _final_targets_reached(self) -> bool:
        if not self._final_targets or self._joint_state_provider is None:
            return False
        current = self._current_joint_positions()
        if current is None:
            return False
        for joint_name, target in self._final_targets.items():
            if joint_name not in current:
                return False
            tolerance = self.goal_tolerance_by_joint.get(joint_name, self.goal_tolerance)
            if abs(current[joint_name] - target) > tolerance:
                return False
        return True

    def _final_target_error_norm(self) -> float | None:
        if not self._final_targets or self._joint_state_provider is None:
            return None
        current = self._current_joint_positions()
        if current is None:
            return None
        errors = []
        for joint_name, target in self._final_targets.items():
            if joint_name not in current:
                return None
            errors.append(abs(float(current[joint_name]) - float(target)))
        return max(errors) if errors else None

    def _detect_stalled_progress(self) -> bool:
        if not self._final_targets or self._joint_state_provider is None:
            return False
        now = time.monotonic()
        error_norm = self._final_target_error_norm()
        if error_norm is None:
            if now - self._last_joint_state_time > self.joint_state_stale_timeout_s:
                self._fail("ros2_control_stalled_joint_state_unavailable")
                return True
            return False
        self._last_joint_state_time = now
        if self._last_error_norm is None or error_norm < self._last_error_norm - self.progress_epsilon:
            self._last_error_norm = error_norm
            self._last_progress_time = now
            return False
        if now - self._last_progress_time > self.progress_timeout_s:
            self._fail(
                "ros2_control_stalled_no_joint_progress: "
                f"target_error={error_norm:.4f}, last_progress_age_s={now - self._last_progress_time:.2f}"
            )
            return True
        return False

    def _current_joint_positions(self) -> dict[str, float] | None:
        if self._joint_state_provider is None:
            return None
        try:
            joint_state = self._joint_state_provider()
        except Exception as exc:  # noqa: BLE001
            log.debug("Unable to read joint state for ros2_control tolerance check: %s", exc)
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

    def _final_target_error_summary(self) -> str:
        current = self._current_joint_positions()
        if current is None:
            return "joint_state unavailable"
        errors = []
        for joint_name, target in self._final_targets.items():
            if joint_name not in current:
                errors.append(f"{joint_name}=missing")
                continue
            errors.append(f"{joint_name}={current[joint_name] - target:+.4f}")
        return ", ".join(errors)

    def _mark_failed_result(self, goal: _ControllerGoal, result: Any, status: Any) -> None:
        action_result = getattr(result, "result", None)
        error_code = getattr(action_result, "error_code", None)
        self._failure = trajectory_failure(error_code, status)
        error_string = getattr(action_result, "error_string", "")
        status_name = self._goal_status_name(status)
        detail = f"status {status_name}({status})"
        if error_code is not None:
            detail += f", error_code {self._trajectory_error_name(error_code)}({error_code})"
        if error_string:
            detail += f", error_string {error_string}"
        detail += f", action {goal.action}, joints {goal.joints}"
        if goal.expanded_from_partial:
            detail += f", expanded_from_partial requested_joints={goal.requested_joints}"
        self._fail(f"ros2_control goal on {goal.name} ended with {detail}")

    def _fail(self, error: str) -> None:
        self._status = "failed"
        self._error = error
        log.warn("ros2_control failure: %s", error)
        self._cancel_active_goals()

    def cancel_goal(self) -> None:
        self._cancel_active_goals()

    def _cancel_active_goals(self) -> None:
        try:
            for goal in self._active_goals:
                if goal.cancel_requested:
                    continue
                goal.cancel_requested = True
                if goal.goal_handle_future is None:
                    continue
                if not goal.goal_handle_future.done():
                    def cancel_later(future):
                        handle = future.result()
                        if handle is not None and getattr(handle, "accepted", False):
                            self._cancel_futures.append(handle.cancel_goal_async())
                    goal.goal_handle_future.add_done_callback(cancel_later)
                    continue
                handle = goal.goal_handle_future.result()
                if handle is not None and getattr(handle, "accepted", False):
                    self._cancel_futures.append(handle.cancel_goal_async())
        except Exception as exc:  # noqa: BLE001
            log.warn("Failed to cancel ros2_control trajectory goal: %s", exc)

    def _wait_for_goal_acceptance(self, goals: list[_ControllerGoal]) -> bool:
        pending = set(id(goal) for goal in goals)
        while pending:
            deadline = time.monotonic() + max(self.wait_for_server_s, 0.1)
            while pending and time.monotonic() < deadline:
                self._spin_once(0.01)
                for goal in goals:
                    if id(goal) not in pending:
                        continue
                    if goal.goal_handle_future is None or not goal.goal_handle_future.done():
                        continue
                    if not self._accept_goal(goal):
                        return False
                    pending.remove(id(goal))
            if not pending:
                break
            for goal in goals:
                if id(goal) not in pending:
                    continue
                if self._goal_may_be_executing(goal):
                    goal.tracking_only = True
                    goal.accepted = True
                    goal.goal_handle_future = None
                    pending.remove(id(goal))
                    log.warn(
                        "ros2_control goal response lost for %s; tracking by joint tolerance",
                        goal.name,
                    )
                    continue
                if goal.acceptance_attempts >= self.goal_response_retries + 1:
                    names = [g.name for g in goals if id(g) in pending]
                    self._fail(f"Timed out waiting for ros2_control goal acceptance: {names}")
                    return False
                self._retry_controller_goal(goal)
        return True

    def _accept_goal(self, goal: _ControllerGoal) -> bool:
        handle = goal.goal_handle_future.result()
        goal.accepted = bool(getattr(handle, "accepted", False))
        if not goal.accepted:
            self._failure = FailureInfo(FailureCode.CONTROLLER_FAILED, backend_code="goal_rejected")
            self._fail(f"ros2_control goal rejected by {goal.name}; action={goal.action}; joints={goal.joints}")
            return False
        log.info("Accepted ros2_control goal controller=%s action=%s", goal.name, goal.action)
        goal.result_future = handle.get_result_async()
        return True

    def _split_by_controller(
        self,
        joints: dict[str, float],
    ) -> list[tuple[_ControllerGoal, dict[str, float]]]:
        remaining = set(joints)
        current_positions = self._current_joint_positions()
        result: list[tuple[_ControllerGoal, dict[str, float]]] = []
        for cfg in self._controller_configs():
            names = [str(j) for j in cfg.get("joints", [])]
            requested = [name for name in names if name in joints]
            if not requested:
                continue
            missing_current = [
                name
                for name in names
                if name not in joints and (current_positions is None or name not in current_positions)
            ]
            if missing_current:
                raise ValueError(
                    "Cannot expand ros2_control partial joint goal for "
                    f"{cfg.get('name') or cfg.get('action')}: missing current joint state for "
                    f"{missing_current}; requested={requested}, controller_joints={names}"
                )
            subset = {
                name: float(joints[name]) if name in joints else float(current_positions[name])
                for name in names
            }
            remaining -= set(requested)
            expanded = len(requested) != len(names)
            result.append(
                (
                    _ControllerGoal(
                        name=str(cfg.get("name") or cfg.get("action")),
                        action=str(cfg["action"]),
                        joints=list(subset),
                        expanded_from_partial=expanded,
                        requested_joints=list(requested),
                    ),
                    subset,
                )
            )
        if remaining:
            log.warn("No ros2_control owner configured for joints: %s", sorted(remaining))
        return result

    def _split_trajectory_by_controller(self, trajectory: Any) -> list[tuple[_ControllerGoal, Any]]:
        joint_names = [str(name) for name in getattr(trajectory, "joint_names", []) or []]
        if not joint_names:
            return []

        remaining = set(joint_names)
        result: list[tuple[_ControllerGoal, Any]] = []
        for cfg in self._controller_configs():
            owned = [str(j) for j in cfg.get("joints", [])]
            present_names = [name for name in joint_names if name in owned]
            if not present_names:
                continue
            remaining -= set(present_names)
            subset_trajectory = self._extract_trajectory_subset(
                trajectory,
                present_names,
            )
            result.append(
                (
                    _ControllerGoal(
                        name=str(cfg.get("name") or cfg.get("action")),
                        action=str(cfg["action"]),
                        joints=list(present_names),
                    ),
                    subset_trajectory,
                )
            )
        if remaining:
            log.warn("No ros2_control owner configured for trajectory joints: %s", sorted(remaining))
        return result

    @staticmethod
    def _extract_trajectory_subset(
        trajectory: Any,
        subset_names: list[str],
    ) -> Any:
        from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

        source_names = [str(name) for name in getattr(trajectory, "joint_names", []) or []]

        subset = JointTrajectory()
        subset.header = getattr(trajectory, "header", subset.header)
        subset.joint_names = list(subset_names)
        subset.points = []
        for point in getattr(trajectory, "points", []) or []:
            new_point = JointTrajectoryPoint()
            new_point.positions = Ros2ControlController._extract_point_sequence(
                source_names,
                subset_names,
                getattr(point, "positions", []),
                default_value=0.0,
            )
            new_point.velocities = Ros2ControlController._extract_point_sequence(
                source_names,
                subset_names,
                getattr(point, "velocities", []),
                default_value=0.0,
            )
            new_point.accelerations = Ros2ControlController._extract_point_sequence(
                source_names,
                subset_names,
                getattr(point, "accelerations", []),
                default_value=0.0,
            )
            new_point.time_from_start = point.time_from_start
            subset.points.append(new_point)
        return subset

    @staticmethod
    def _extract_point_sequence(
        source_names: list[str],
        output_names: list[str],
        values: Any,
        *,
        default_value: float,
    ) -> list[float]:
        values = list(values or [])
        output = []
        for name in output_names:
            if name in source_names:
                index = source_names.index(name)
                if index < len(values):
                    output.append(float(values[index]))
                    continue
            output.append(float(default_value))
        return output

    @staticmethod
    def _normalize_prepared_goal(
        goal: dict[str, Any] | _ControllerGoal,
        trajectory: Any,
    ) -> tuple[_ControllerGoal, Any]:
        if isinstance(goal, _ControllerGoal):
            return goal, trajectory
        joint_names = [str(name) for name in getattr(trajectory, "joint_names", []) or []]
        return (
            _ControllerGoal(
                name=str(goal.get("name") or goal.get("action")),
                action=str(goal["action"]),
                joints=joint_names,
            ),
            trajectory,
        )

    @staticmethod
    def _trajectory_diagnostics(trajectory: Any) -> dict[str, Any] | None:
        attached = getattr(trajectory, "_orchid_base_trajectory_diagnostics", None)
        if isinstance(attached, dict):
            return attached
        joint_names = [str(name) for name in getattr(trajectory, "joint_names", []) or []]
        points = list(getattr(trajectory, "points", []) or [])
        if not points:
            return {"point_count": 0}
        if not all(name in joint_names for name in ("odom_x", "odom_y", "odom_t")):
            return {"point_count": len(points)}
        indices = [joint_names.index(name) for name in ("odom_x", "odom_y", "odom_t")]
        rows: list[tuple[list[float], float]] = []
        for point in points:
            positions = list(getattr(point, "positions", []) or [])
            if max(indices) >= len(positions):
                continue
            duration = getattr(point, "time_from_start", None)
            seconds = float(getattr(duration, "sec", 0)) + float(getattr(duration, "nanosec", 0)) / 1_000_000_000.0
            rows.append(([float(positions[index]) for index in indices], seconds))
        if not rows:
            return {"point_count": len(points), "reason": "missing_base_positions"}
        summary: dict[str, Any] = {
            "point_count": len(rows),
            "duration_s": float(rows[-1][1]),
            "first": rows[0][0],
            "last": rows[-1][0],
            "min_dt_s": None,
            "max_abs_vx_mps": 0.0,
            "max_abs_vy_mps": 0.0,
            "max_abs_wz_radps": 0.0,
        }
        min_dt: float | None = None
        for index in range(1, len(rows)):
            previous, previous_time = rows[index - 1]
            current, current_time = rows[index]
            dt = float(current_time) - float(previous_time)
            if min_dt is None or dt < min_dt:
                min_dt = dt
            if dt > 0.0:
                summary["max_abs_vx_mps"] = max(float(summary["max_abs_vx_mps"]), abs((current[0] - previous[0]) / dt))
                summary["max_abs_vy_mps"] = max(float(summary["max_abs_vy_mps"]), abs((current[1] - previous[1]) / dt))
                summary["max_abs_wz_radps"] = max(float(summary["max_abs_wz_radps"]), abs((current[2] - previous[2]) / dt))
        summary["min_dt_s"] = min_dt
        return summary

    @staticmethod
    def _goal_status_name(status: Any) -> str:
        names = {
            0: "STATUS_UNKNOWN",
            1: "STATUS_ACCEPTED",
            2: "STATUS_EXECUTING",
            3: "STATUS_CANCELING",
            4: "STATUS_SUCCEEDED",
            5: "STATUS_CANCELED",
            6: "STATUS_ABORTED",
        }
        return names.get(status, "STATUS_UNRECOGNIZED")

    @staticmethod
    def _trajectory_error_name(error_code: Any) -> str:
        names = {
            0: "SUCCESSFUL",
            -1: "INVALID_GOAL",
            -2: "INVALID_JOINTS",
            -3: "OLD_HEADER_TIMESTAMP",
            -4: "PATH_TOLERANCE_VIOLATED",
            -5: "GOAL_TOLERANCE_VIOLATED",
        }
        return names.get(error_code, "UNKNOWN_ERROR")

    def _controller_configs(self) -> list[dict[str, Any]]:
        cfg = self.hardware_config.get("ros2_control") or {}
        controllers = cfg.get("controllers") or self.hardware_config.get("trajectory_controllers")
        if isinstance(controllers, dict):
            return [
                {"name": name, **value}
                for name, value in controllers.items()
                if isinstance(value, dict) and value.get("action")
            ]
        if isinstance(controllers, list):
            return [c for c in controllers if isinstance(c, dict) and c.get("action")]
        return list(self.DEFAULT_CONTROLLERS)

    def _send_controller_goal(self, goal: _ControllerGoal, joints: dict[str, float]) -> None:
        from builtin_interfaces.msg import Duration
        from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

        client = self._client(goal.action)
        if not client.wait_for_server(timeout_sec=self.wait_for_server_s):
            raise RuntimeError(f"ros2_control action server unavailable: {goal.action}")

        trajectory = JointTrajectory()
        trajectory.joint_names = list(joints)
        point = JointTrajectoryPoint()
        point.positions = [float(joints[name]) for name in trajectory.joint_names]
        point.time_from_start = Duration(sec=int(self.duration_s), nanosec=int((self.duration_s % 1.0) * 1e9))
        trajectory.points = [point]

        self._send_controller_trajectory_goal(goal, trajectory)
        self._spin_once(0.0)
        log.info(
            "Sent ros2_control goal to %s via %s joints=%s expanded_from_partial=%s requested_joints=%s",
            goal.name,
            goal.action,
            trajectory.joint_names,
            goal.expanded_from_partial,
            goal.requested_joints,
        )

    def _send_controller_trajectory_goal(self, goal: _ControllerGoal, trajectory: Any) -> None:
        from control_msgs.action import FollowJointTrajectory
        from control_msgs.msg import JointTolerance

        # These are independently completed stop-to-stop goals. In particular,
        # IPTP can leave nonzero endpoint acceleration despite zero velocity;
        # JTC retains that value as its next interpolation reference.
        trajectory = copy.deepcopy(trajectory)
        for point in (trajectory.points[0], trajectory.points[-1]):
            point.velocities = [0.] * len(trajectory.joint_names)
            point.accelerations = [0.] * len(trajectory.joint_names)
        goal.trajectory = trajectory
        goal.acceptance_attempts += 1
        client = self._client(goal.action)
        if not client.wait_for_server(timeout_sec=self.wait_for_server_s):
            raise RuntimeError(f"ros2_control action server unavailable: {goal.action}")
        if self.goal_response_retry_settle_s > 0.0:
            time.sleep(self.goal_response_retry_settle_s)

        action_goal = FollowJointTrajectory.Goal()
        action_goal.trajectory = trajectory
        action_goal.path_tolerance = [
            JointTolerance(name=name, position=self.base_path_tolerance[name])
            for name in trajectory.joint_names if name in self.base_path_tolerance
        ]
        goal.goal_handle_future = client.send_goal_async(action_goal)
        if goal not in self._active_goals:
            self._active_goals.append(goal)
        self._spin_once(0.0)
        log.info(
            "Sent ros2_control trajectory to %s via %s joints=%s points=%s path_tolerance=%s",
            goal.name,
            goal.action,
            list(getattr(trajectory, "joint_names", []) or []),
            len(getattr(trajectory, "points", []) or []),
            {t.name: t.position for t in action_goal.path_tolerance},
        )

    def _retry_controller_goal(self, goal: _ControllerGoal) -> None:
        if goal.trajectory is None:
            return
        log.warn(
            "Retrying ros2_control goal response timeout for %s via %s attempt=%s/%s",
            goal.name,
            goal.action,
            goal.acceptance_attempts + 1,
            self.goal_response_retries + 1,
        )
        self._clients.pop(goal.action, None)
        goal.goal_handle_future = None
        goal.accepted = None
        self._send_controller_trajectory_goal(goal, goal.trajectory)

    def _goal_may_be_executing(self, goal: _ControllerGoal) -> bool:
        if self._joint_state_provider is None:
            return False
        error_norm = self._goal_target_error_norm(goal)
        if error_norm is None:
            return False
        if goal.acceptance_error_norm is None:
            goal.acceptance_error_norm = error_norm
            return False
        improving = error_norm < goal.acceptance_error_norm - self.progress_epsilon
        if improving:
            goal.acceptance_error_norm = error_norm
        return improving

    def _goal_target_error_norm(self, goal: _ControllerGoal) -> float | None:
        current = self._current_joint_positions()
        if current is None:
            return None
        errors = []
        for joint_name in goal.joints:
            if joint_name not in self._final_targets or joint_name not in current:
                continue
            errors.append(abs(float(current[joint_name]) - float(self._final_targets[joint_name])))
        return max(errors) if errors else None

    def _client(self, action_name: str):
        if action_name not in self._clients:
            from control_msgs.action import FollowJointTrajectory
            from rclpy.action import ActionClient

            self._clients[action_name] = ActionClient(
                self._get_node(),
                FollowJointTrajectory,
                action_name,
            )
        return self._clients[action_name]

    def _get_node(self):
        if self._node is None:
            import rclpy

            if not rclpy.ok():
                raise RuntimeError("rclpy is not initialized")
            safe_name = f"orchid_ros2_control_{int(time.time() * 1000)}"
            self._node = rclpy.create_node(safe_name)
            self._owns_node = True
        return self._node

    def _spin_once(self, timeout_sec: float = 0.0) -> None:
        if self._owns_node:
            import rclpy

            rclpy.spin_once(self._get_node(), timeout_sec=timeout_sec)
        elif timeout_sec > 0:
            time.sleep(timeout_sec)


class Ros2ControlTrajectoryClient(Ros2ControlController):
    """Backward-compatible alias for the generic ros2_control controller."""
