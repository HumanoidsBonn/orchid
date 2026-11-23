"""HSRB gripper controller adapter using `hsrb_interface` EndEffector.Gripper."""

from __future__ import annotations
import math

from src.core.controllers import BaseGripperController, ControllerResult

import src.core.logger as log


class HsrbGripperController(BaseGripperController):
    """Adapter for HSRB gripper.

    hardware_config keys:
    - name: end_effector resource name (default: "gripper")
    - open_distance: meters when open
    - closed_distance: meters when closed
    """

    def __init__(self, hardware_config: dict):
        super().__init__(hardware_config)
        self._available = False
        self._gripper = None
        self._last_command: str | None = None
        self._open_width: float | None = None
        try:
            import hsrb_interface
            self._robot = hsrb_interface.Robot()
            name = hardware_config.get("name", "gripper")
            self._gripper = self._robot.get(name)
            self._available = True
        except Exception as exc:  # pragma: no cover - best-effort adapter
            log.warning("hsrb_interface Gripper unavailable: %s", exc)

    def open(self, width_m: float | None = None, sync=False) -> ControllerResult:
        self._last_command = None
        if not self._available or self._gripper is None:
            return ControllerResult(success=False, error="gripper unavailable")
        try:
            if width_m is None:
                width = float(self.hardware_config.get("open_distance", 0.08))
            else:
                width = float(width_m)
            self._gripper.set_distance(width, sync=sync)
            self._open_width = width
            self._last_command = "open"
            return ControllerResult(success=True)
        except Exception as exc:
            log.exception("Failed to open gripper: %s", exc)
            return ControllerResult(success=False, error=str(exc))

    def close(self, force_n: float | None = None, sync=False) -> ControllerResult:
        self._last_command = None
        if not self._available or self._gripper is None:
            return ControllerResult(success=False, error="gripper unavailable")
        try:
            if force_n is None:
                # use closed_distance config
                dist = float(self.hardware_config.get("closed_distance", 0.0))
                self._gripper.set_distance(dist, sync=sync)
            else:
                # apply_force only engages real force-controlled closing when
                # delicate=True (and effort < 0.8N, hsrb_interface's own
                # threshold) - otherwise it silently falls back to a blunt,
                # non-force-limited close regardless of force_n.
                self._gripper.apply_force(float(force_n), delicate=False, sync=sync)
            self._last_command = "closed"
            return ControllerResult(success=True)
        except Exception as exc:
            log.exception("Failed to close gripper: %s", exc)
            return ControllerResult(success=False, error=str(exc))

    def get_state(self) -> str:
        status = self.get_status()
        if status == "in_progress":
            return "moving"
        if status == "failed":
            return "stalled"
        if status == "succeeded":
            return self._last_command or "closed"
        return "open"

    def get_status(self) -> str:
        if not self._gripper:
            return "failed"
        try:
            moving = self._gripper.is_moving()
            succeeded = self._gripper.is_succeeded()
            log.debug("Gripper moving=%s success=%s", moving, succeeded)
            if moving:
                return "in_progress"
            if succeeded:
                return "succeeded"
            if self._last_command is not None:
                distance = self.get_distance()
                if distance is not None and math.isfinite(distance) and distance >= 0:
                    matched = (self._last_command == "closed" and distance <= .05) or (
                        self._last_command == "open" and abs(distance - self._open_width) <= .03 + 1e-9)
                    if matched:
                        log.info("HSR gripper idle accepted from measured opening: command=%s distance=%.4f m open_target=%s",
                                 self._last_command, distance, self._open_width)
                        return "succeeded"
                    log.warn("HSR gripper idle outside success range: command=%s distance=%.4f m open_target=%s",
                             self._last_command, distance, self._open_width)
                else:
                    log.warn("HSR gripper idle cannot be confirmed: measured opening=%s", distance)
            return "idle"
        except Exception:
            return "failed"

    def get_error(self) -> str | None:
        return None

    def cancel_goal(self) -> None:
        self._last_command = None
        if not self._available or self._gripper is None:
            return
        try:
            self._gripper.cancel_goal()
        except Exception as e:
            log.exception("Failed to cancel gripper goal", e)

    def get_distance(self) -> float | None:
        if not self._available or self._gripper is None:
            return None
        try:
            return float(self._gripper.get_distance())
        except Exception as exc:
            log.exception("Failed to read gripper distance: %s", exc)
            return None
