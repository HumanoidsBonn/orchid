"""Nav2 controller plugin (ROS 2 Humble)."""

from __future__ import annotations

import threading
import time
from typing import Tuple

from src.core.controllers import BaseNavController, NavStatus, Pose
from src.core.geometry import rpy_to_quaternion

import src.core.logger as log


def _yaw_to_quaternion(yaw: float):
    from geometry_msgs.msg import Quaternion

    qx, qy, qz, qw = rpy_to_quaternion(0.0, 0.0, yaw)
    return Quaternion(x=qx, y=qy, z=qz, w=qw)


class Nav2Controller(BaseNavController):
    """Non-blocking Nav2 ``NavigateToPose`` action client."""

    def __init__(self, hardware_config: dict):
        super().__init__(hardware_config)
        self._action_name = hardware_config.get("action_name", "navigate_to_pose")
        self._default_frame = hardware_config.get("frame_id", "map")
        self._goal_timeout_s = float(hardware_config.get("goal_timeout_s", 300.0))
        self._status: NavStatus = "idle"
        self._error: str | None = None
        self._goal_handle = None
        self._result_future = None
        self._send_goal_future = None
        self._cancel_future = None
        self._cancel_requested = False
        self._pending_goal: tuple[Tuple[float, float, float], str | None] | None = None
        self._lock = threading.Lock()
        self._ros_node = hardware_config.get("ros2_node")
        self._owns_node = False
        self._action_client = None
        self._init_ros()
        self.counter = 10

    def _init_ros(self) -> None:
        try:
            import rclpy
            from nav2_msgs.action import NavigateToPose
            from rclpy.action import ActionClient
            from rclpy.node import Node
        except ImportError as exc:
            log.warning("Nav2Controller ROS dependencies unavailable: %s", exc)
            return

        if self._ros_node is None:
            if not rclpy.ok():
                rclpy.init()
            self._ros_node = Node("nav2_controller_client")
            self._owns_node = True

        self._NavigateToPose = NavigateToPose
        self._action_client = ActionClient(
            self._ros_node,
            NavigateToPose,
            self._action_name,
        )

    def send_goal(self, target: Tuple[float, float, float], frame_id: str | None = None) -> None:
        if self._action_client is None:
            self._status = "failed"
            self._error = "Nav2 action client not initialized"
            return
                
        from geometry_msgs.msg import PoseStamped

        pose = PoseStamped()
        pose.header.frame_id = frame_id or self._default_frame
        pose.header.stamp = self._ros_node.get_clock().now().to_msg()
        pose.pose.position.x = float(target[0])
        pose.pose.position.y = float(target[1])
        pose.pose.orientation = _yaw_to_quaternion(float(target[2]))

        goal_msg = self._NavigateToPose.Goal()
        goal_msg.pose = pose

        with self._lock:
            self._status = "in_progress"
            self._error = None
            self._cancel_requested = False
            self._pending_goal = None
            if not self._action_client.wait_for_server(timeout_sec=2.0):
                self._status = "failed"
                self._error = f"Nav2 action server '{self._action_name}' unavailable"
                return

            send_future = self._action_client.send_goal_async(
                goal_msg,
                feedback_callback=self._on_feedback,
            )
            self._send_goal_future = send_future
            log.info(
                "Sent Nav2 goal to %s frame=%s x=%.3f y=%.3f yaw=%.3f",
                self._action_name,
                pose.header.frame_id,
                float(target[0]),
                float(target[1]),
                float(target[2]),
            )
            send_future.add_done_callback(self._on_goal_response)

    def update_goal(self, target: Tuple[float, float, float], frame_id: str | None = None) -> None:
        """Cancel the active goal and resend once cancellation is acknowledged."""
        log.info("Nav2Controller.update_goal requested target=%s frame=%s", target, frame_id)
        with self._lock:
            active = self._goal_handle is not None or self._send_goal_future is not None
        if active:
            log.error("Cancel goal called in update_goal")
            self.cancel_goal()
            self._pending_goal = (target, frame_id)
            return
        self.send_goal(target, frame_id=frame_id)

    def get_status(self) -> NavStatus:
        self._progress_futures()
        with self._lock:
            return self._status

    def get_error(self) -> str | None:
        with self._lock:
            return self._error

    def cancel_goal(self) -> None:
        with self._lock:
            handle = self._goal_handle
            self._cancel_requested = True
            if handle is None and self._send_goal_future is not None:
                self._status = "idle"
                self._send_goal_future = None
        if handle is not None:
            log.info("Requesting Nav2 goal cancel on %s", self._action_name)
            cancel_future = handle.cancel_goal_async()
            with self._lock:
                self._cancel_future = cancel_future
            cancel_future.add_done_callback(self._on_cancel_response)
            return
        with self._lock:
            self._status = "idle"
            self._result_future = None
            self._send_goal_future = None

    def get_current_pose(self) -> Pose:
        provider = self.hardware_config.get("pose_provider")
        if callable(provider):
            try:
                pose = provider()
                if isinstance(pose, dict):
                    return Pose(
                        x=float(pose.get("x", 0.0)),
                        y=float(pose.get("y", 0.0)),
                        yaw=float(pose.get("yaw", 0.0)),
                    )
                return Pose(
                    x=float(getattr(pose, "x", 0.0)),
                    y=float(getattr(pose, "y", 0.0)),
                    yaw=float(getattr(pose, "yaw", 0.0)),
                )
            except Exception as exc:  # noqa: BLE001
                log.debug("Nav2 pose provider failed: %s", exc)
        return Pose(x=0.0, y=0.0, yaw=0.0)

    def _on_goal_response(self, future) -> None:
        with self._lock:
            if self._send_goal_future is not None and future is not self._send_goal_future:
                return
        try:
            goal_handle = future.result()
        except Exception as exc:  # noqa: BLE001
            with self._lock:
                self._status = "failed"
                self._error = f"Nav2 goal response failed: {exc}"
            log.exception("Nav2 goal response failed: %s", exc)
            return
        if not goal_handle.accepted:
            with self._lock:
                self._status = "failed"
                self._error = "Nav2 goal rejected"
                self._send_goal_future = None
            log.warn("Nav2 goal rejected by %s", self._action_name)
            return

        with self._lock:
            self._goal_handle = goal_handle
            self._result_future = goal_handle.get_result_async()
            self._send_goal_future = None
        log.info("Nav2 goal accepted by %s", self._action_name)
        self._result_future.add_done_callback(self._on_result)

    def _on_result(self, future) -> None:
        with self._lock:
            if self._result_future is not None and future is not self._result_future:
                return
        try:
            result = future.result()
        except Exception as exc:  # noqa: BLE001
            with self._lock:
                self._status = "failed"
                self._error = f"Nav2 result failed: {exc}"
            log.exception("Nav2 result callback failed: %s", exc)
            return
        try:
            from action_msgs.msg import GoalStatus
        except Exception:  # noqa: BLE001
            GoalStatus = None
        succeeded = getattr(GoalStatus, "STATUS_SUCCEEDED", 4)
        canceled = getattr(GoalStatus, "STATUS_CANCELED", 5)
        aborted = getattr(GoalStatus, "STATUS_ABORTED", 6)
        status_name = self._goal_status_name(getattr(result, "status", None))
        with self._lock:
            if result.status == succeeded:
                self._status = "succeeded"
                self._error = None
            elif result.status == canceled:
                self._status = "idle"
                self._error = None
            elif result.status == aborted:
                self._status = "failed"
                self._error = f"Nav2 goal aborted with status={status_name}({result.status})"
            else:
                self._status = "failed"
                self._error = f"Nav2 result status={status_name}({result.status})"
            self._goal_handle = None
            self._result_future = None
            self._send_goal_future = None
            self._cancel_future = None
            self._cancel_requested = False
            pending_goal = self._pending_goal
            self._pending_goal = None
        terminal_status = self._status
        terminal_error = self._error
        if terminal_status == "failed":
            log.warn(terminal_error)
        else:
            log.info("Nav2 goal finished with %s", status_name)
        if pending_goal is not None:
            self.send_goal(*pending_goal)

    def _on_cancel_response(self, future) -> None:
        with self._lock:
            if self._cancel_future is not None and future is not self._cancel_future:
                return
        try:
            response = future.result()
            goals_canceling = list(getattr(response, "goals_canceling", []) or [])
        except Exception as exc:  # noqa: BLE001
            with self._lock:
                self._status = "failed"
                self._error = f"Nav2 cancel response failed: {exc}"
            log.exception("Nav2 cancel response failed: %s", exc)
            return
        with self._lock:
            self._cancel_future = None
            pending_goal = self._pending_goal
            if not goals_canceling and self._result_future is None:
                self._status = "idle"
                self._goal_handle = None
                self._pending_goal = None
        log.info("Nav2 cancel response goals_canceling=%d", len(goals_canceling))
        if pending_goal is not None and not goals_canceling:
            self.send_goal(*pending_goal)

    def _on_feedback(self, feedback_msg) -> None:
        log.debug("Nav2 feedback: distance_remaining=%s", feedback_msg.feedback.distance_remaining)

    def destroy(self) -> None:
        if self._owns_node and self._ros_node is not None:
            self._ros_node.destroy_node()

    def _progress_futures(self) -> None:
        if self._ros_node is None:
            return
        try:
            self._spin_once(0.0)
            with self._lock:
                send_future = self._send_goal_future
                result_future = self._result_future
                cancel_future = self._cancel_future
            if send_future is not None and hasattr(send_future, "done") and send_future.done():
                self._on_goal_response(send_future)
            if cancel_future is not None and hasattr(cancel_future, "done") and cancel_future.done():
                self._on_cancel_response(cancel_future)
            if result_future is not None and hasattr(result_future, "done") and result_future.done():
                self._on_result(result_future)
        except Exception as exc:  # noqa: BLE001
            log.debug("Unable to progress Nav2 futures: %s", exc)

    def _spin_once(self, timeout_sec: float = 0.0) -> None:
        if self._owns_node and self._ros_node is not None:
            import rclpy

            rclpy.spin_once(self._ros_node, timeout_sec=timeout_sec)
        elif timeout_sec > 0:
            time.sleep(timeout_sec)

    @staticmethod
    def _goal_status_name(status) -> str:
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
