"""Head movement actions."""

from __future__ import annotations

from typing import Any

from src.core.action import ActionResult, AsyncAction
from src.core.context import Context
from src.core.failure import FailureInfo, FailureCode, FailureOrigin
from src.plugin.actions._helpers import _head_status_result


class MoveHeadToPanTilt(AsyncAction):
    """Move the robot head to an explicit pan/tilt view.

    Use this for camera view changes before perception. The action dispatches a
    non-blocking head goal and polls the configured head controller.

    Params:
        pan: float - head pan/yaw joint target in radians.
        tilt: float - head tilt/pitch joint target in radians.
    """

    required_controllers = ["head"]

    def __init__(self, pan: float=0.0, tilt: float=0.0) -> None:
        self.pan = pan
        self.tilt = tilt
        super().__init__()
        self._status = 'idle'
        self._failure_reason: str | None = None

    def start(self, context: Context) -> None:
        try:
            context.require_head().send_pan_tilt(float(self.pan), float(self.tilt))
            self._status = "in_progress"
        except Exception as exc:  # noqa: BLE001
            self._status = "failure"
            self._failure_reason = str(exc)

    def check_result(self, context: Context) -> ActionResult:
        if self._cancel_requested:
            return ActionResult(status="failure", reason="Action Cancelled")
        if self._status == "failure":
            return ActionResult(status="failure", reason=self._failure_reason,
                                failure=FailureInfo(FailureCode.CONTROLLER_FAILED, origin=FailureOrigin.CONTROLLER))
        return _head_status_result(context.require_head())

    def reset(self) -> None:
        super().reset()
        self._status = "idle"
        self._failure_reason = None

    def cancel(self, context: Context) -> None:
        self._cancel_requested = True
        if context.head:
            context.head.cancel_goal()


class MoveHeadToState(AsyncAction):
    """Move the head to a manifest-defined named view state.

    Params:
        state: str - name under ``hardware.head.named_targets``.
    """

    required_controllers = ["head"]

    def __init__(self, state: str='center') -> None:
        self.state = state
        super().__init__()
        self._status = 'idle'
        self._failure_reason: str | None = None

    def start(self, context: Context) -> None:
        try:
            context.require_head().send_named_goal(self.state)
            self._status = "in_progress"
        except Exception as exc:  # noqa: BLE001
            self._status = "failure"
            self._failure_reason = str(exc)

    def check_result(self, context: Context) -> ActionResult:
        if self._cancel_requested:
            return ActionResult(status="failure", reason="Action Cancelled")
        if self._status == "failure":
            return ActionResult(status="failure", reason=self._failure_reason,
                                failure=FailureInfo(FailureCode.CONTROLLER_FAILED, origin=FailureOrigin.CONTROLLER))
        return _head_status_result(context.require_head())

    def reset(self) -> None:
        super().reset()
        self._status = "idle"
        self._failure_reason = None

    def cancel(self, context: Context) -> None:
        self._cancel_requested = True
        if context.head:
            context.head.cancel_goal()
