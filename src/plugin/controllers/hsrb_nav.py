"""HSRB navigation controller adapter using `hsrb_interface` MobileBase."""

from __future__ import annotations

from typing import Tuple

from src.core.controllers import BaseNavController, NavStatus, Pose

import src.core.logger as log


class HsrbNavController(BaseNavController):
    """Adapter that uses `hsrb_interface`'s MobileBase for navigation.

    hardware_config keys:
    - name: resource name in hsrb_interface settings (default: "omni_base")
    """

    def __init__(self, hardware_config: dict):
        super().__init__(hardware_config)
        self._available = False
        self._base = None
        try:
            import hsrb_interface
            from hsrb_interface import geometry
            self._geometry = geometry
            self._robot = hsrb_interface.Robot()
            name = hardware_config.get("name", "omni_base")
            self._base = self._robot.get(name)
            self._available = True
        except Exception as exc:  # pragma: no cover - best effort adapter
            log.warning("hsrb_interface not available: %s", exc)

    def send_goal(self, target: Tuple[float, float, float], frame_id: str = "map") -> None:
        if not self._available or self._base is None:
            return
        x, y, yaw = float(target[0]), float(target[1]), float(target[2])
        pose = self._geometry.pose(x, y, 0.0, 0.0, 0.0, yaw)
        goal = self._base.create_go_pose_goal(pose, frame_id)
        # execute() sends the goal asynchronously
        self._base.execute(goal)

    def update_goal(self, target: Tuple[float, float, float], frame_id: str | None = None) -> None:
        # HSRB MobileBase does not support goal-update; resend.
        self.send_goal(target, frame_id=frame_id)

    def get_status(self) -> NavStatus:
        if not self._available or self._base is None:
            return "failed"
        try:
            if self._base.is_moving():
                return "in_progress"
            if self._base.is_succeeded():
                return "succeeded"
            return "idle"
        except Exception:
            return "failed"

    def get_error(self) -> str | None:
        return None

    def cancel_goal(self) -> None:
        if not self._available or self._base is None:
            return
        try:
            self._base.cancel_goal()
        except Exception:
            log.exception("Failed to cancel HSRB nav goal")

    def get_current_pose(self) -> Pose:
        if not self._available or self._base is None:
            return Pose(0.0, 0.0, 0.0)
        try:
            p = self._base.pose
            return Pose(x=float(p[0]), y=float(p[1]), yaw=float(p[2]))
        except Exception:
            return Pose(0.0, 0.0, 0.0)
