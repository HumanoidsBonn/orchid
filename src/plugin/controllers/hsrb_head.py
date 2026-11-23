"""HSRB head controller adapter."""

from __future__ import annotations

from src.core.controllers import BaseHeadController, HeadStatus

import src.core.logger as log



class HsrbHeadController(BaseHeadController):
    """Move HSRB head pan/tilt joints through a selectable backend.

    ``execution_backend="ros2_control"`` sends direct
    ``FollowJointTrajectory`` goals to ``/head_trajectory_controller``.
    ``execution_backend="hsrb_interface"`` keeps the older whole-body
    ``JointGroup.move_to_joint_positions`` path.
    """

    def __init__(self, hardware_config: dict):
        super().__init__(hardware_config)
        self._available = False
        self._hsrb_available = False
        self._jg = None
        self._error: str | None = None
        self.status: HeadStatus = "idle"
        self._execution_backend = str(hardware_config.get("execution_backend", "ros2_control"))
        self._ros2_control = None
        if self._execution_backend == "ros2_control":
            from src.plugin.controllers.ros2_control_trajectory import Ros2ControlTrajectoryClient

            cfg = dict(hardware_config)
            cfg.setdefault(
                "ros2_control",
                {
                    "controllers": [
                        {
                            "name": "head",
                            "action": "/head_trajectory_controller/follow_joint_trajectory",
                            "joints": [
                                cfg.get("pan_joint", "head_pan_joint"),
                                cfg.get("tilt_joint", "head_tilt_joint"),
                            ],
                        }
                    ]
                },
            )
            self._ros2_control = Ros2ControlTrajectoryClient(
                cfg,
                joint_state_provider=self._current_joint_state,
            )
            self._available = True

        try:
            import hsrb_interface

            self._robot = hsrb_interface.Robot()
            name = hardware_config.get("name", "whole_body")
            self._jg = self._robot.get(name)
            self._available = True
            self._hsrb_available = True
        except Exception as exc:  # pragma: no cover - best-effort adapter
            log.warning("hsrb_interface JointGroup unavailable for head: %s", exc)

    def send_pan_tilt(self, pan: float, tilt: float) -> None:
        pan_name = self.hardware_config.get("pan_joint", "head_pan_joint")
        tilt_name = self.hardware_config.get("tilt_joint", "head_tilt_joint")
        self.send_joint_goal({pan_name: float(pan), tilt_name: float(tilt)})

    def send_joint_goal(self, joints: dict[str, float]) -> None:
        log.info(
            "HSRB head send_joint_goal backend=%s joints=%s",
            self._execution_backend,
            {name: float(value) for name, value in joints.items()},
        )
        if self._execution_backend == "ros2_control":
            if self._ros2_control is None:
                self.status = "failed"
                self._error = "ros2_control head backend was not initialized"
                return
            self._ros2_control.send_joint_goal(joints)
            self.status = self._ros2_control.get_status()
            self._error = self._ros2_control.get_error()
            return

        if not self._hsrb_available or self._jg is None:
            self.status = "failed"
            self._error = "HSRB interface is unavailable for head control"
            return
        try:
            self._error = None
            self.status = "in_progress"
            traj = self._jg.move_to_joint_positions(joints, plan_only=True)
            if traj is None:
                self.status = "failed"
                self._error = "HSRB head planner returned no trajectory"
                return
            self._jg.execute(traj)
        except Exception as exc:  # noqa: BLE001
            self.status = "failed"
            self._error = str(exc)
            log.exception("Failed to move HSRB head: %s", exc)

    def get_status(self) -> HeadStatus:
        if self._execution_backend == "ros2_control":
            if self._ros2_control is None:
                return "failed"
            self.status = self._ros2_control.get_status()
            self._error = self._ros2_control.get_error()
            return self.status
        if not self._hsrb_available or self._jg is None:
            return "failed"
        try:
            if self.status == "failed":
                return "failed"
            if self._jg.is_moving():
                return "in_progress"
            if self._jg.is_succeeded():
                self.status = "succeeded"
                return "succeeded"
            return self.status
        except Exception as exc:  # noqa: BLE001
            self.status = "failed"
            self._error = str(exc)
            return "failed"

    def get_error(self) -> str | None:
        return self._error

    def cancel_goal(self) -> None:
        if self._execution_backend == "ros2_control":
            if self._ros2_control is not None:
                self._ros2_control.cancel_goal()
            return
        if not self._hsrb_available or self._jg is None:
            return
        try:
            self._jg.cancel_goal()
        except Exception as exc:  # noqa: BLE001
            log.warn("Failed to cancel HSRB head goal: %s", exc)

    def _current_joint_state(self):
        try:
            if self._jg is not None and hasattr(self._jg, "_get_joint_state"):
                return self._jg._get_joint_state()
        except Exception as exc:  # pragma: no cover - diagnostic only
            log.debug("Unable to read HSRB current joint state for head tolerance check: %s", exc)
        return None
