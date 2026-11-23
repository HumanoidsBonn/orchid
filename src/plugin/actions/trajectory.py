"""Execute an already planned joint trajectory; no planning or scene ownership."""
from __future__ import annotations
import copy
import threading
import time
import src.core.logger as log
from src.core.action import AsyncAction, ActionResult
from src.core.failure import FailureInfo, FailureCode
from ._helpers import _arm_status_result


def trajectory_subset(trajectory, joint_names):
    result = copy.deepcopy(trajectory)
    if not joint_names:
        return result
    names = list(result.joint_names)
    selected = [n for n in names if n in joint_names]
    if set(selected) != set(joint_names):
        raise ValueError("prepared trajectory is missing requested controller joints")
    indices = [names.index(n) for n in selected]
    result.joint_names = selected
    for point in result.points:
        for field in ("positions", "velocities", "accelerations", "effort"):
            values = list(getattr(point, field))
            if values:
                setattr(point, field, [values[i] for i in indices])
    return result


class ExecuteJointTrajectory(AsyncAction):
    """Dispatch a WorldState trajectory without replanning or changing the scene.

    joint_names optionally selects the subsystem from a fixed-joint full-body
    plan. Controller batching preserves the prepared arm/base timeline.
    """
    required_controllers = ["arm"]

    def __init__(self, trajectory_key: str, joint_names: list[str] | None = None, start_message: str | None = None):
        super().__init__()
        self.trajectory_key, self.joint_names = trajectory_key, joint_names
        self.start_message = start_message
        self._lock = threading.RLock()
        self._generation = 0
        self._dispatched = self._dispatch_complete = self._cancel_sent = False
        self._result = None

    def start(self, context):
        with self._lock:
            self._generation += 1
            generation = self._generation
            self._cancel_requested = False
            self._dispatched = self._dispatch_complete = self._cancel_sent = False
            self._result = None
            self._began = self._last_log = time.monotonic()
        def current():
            return (generation == self._generation and not self._cancel_requested and
                    context.execution.control is None and not context.execution.cancel_requested)
        try:
            source = context.world_state.get(self.trajectory_key)
            if source is None or not source.points:
                raise ValueError(f"missing prepared trajectory: {self.trajectory_key}")
            trajectory = trajectory_subset(source, self.joint_names)
            t = trajectory.points[-1].time_from_start
            self._duration = t.sec+t.nanosec*1e-9
        except Exception as exc:
            self._result = ActionResult(status="failure", reason=str(exc), failure=FailureInfo(FailureCode.INVALID_REQUEST))
            log.warn("Trajectory input failed: %s", exc)
            return
        arm = context.require_arm()
        def work():
            if not current():
                return
            log.info("Execute prepared trajectory key=%s joints=%s points=%s duration_s=%.3f first=%s",
                     self.trajectory_key,list(trajectory.joint_names),len(trajectory.points),self._duration,list(trajectory.points[0].positions))
            with self._lock:
                if not current():
                    return
                self._dispatched = True
            if self.start_message:
                log.speak(self.start_message)
            # This route preserves timing and bypasses backend-specific replanning,
            # independent controller scaling and sanitization, even for one owner.
            arm.send_synchronized_trajectory_goal(trajectory, cancelled=lambda:not current())
            with self._lock:
                self._dispatch_complete = True
                if not current() and not self._cancel_sent:
                    self._cancel_sent = True
                    arm.cancel_goal()
        def fail(exc):
            if current():
                log.warn("Trajectory dispatch failed key=%s: %s",self.trajectory_key,exc)
                self._result = ActionResult(status="failure",reason=str(exc),failure=getattr(exc,"failure",FailureInfo()))
        context.run_in_context_loop(work,error_callback=fail)

    def check_result(self, context):
        if self._result:
            return self._result
        now = time.monotonic()
        if now-self._last_log>=2.:
            log.info("Trajectory waiting key=%s dispatched=%s elapsed_s=%.1f planned_s=%.2f",
                     self.trajectory_key,self._dispatch_complete,now-self._began,self._duration)
            self._last_log=now
        if not self._dispatch_complete:
            return ActionResult(status="running")
        result = _arm_status_result(context.require_arm())
        if result.status != "running":
            log.info("Trajectory completed key=%s status=%s reason=%s elapsed_s=%.3f",self.trajectory_key,result.status,result.reason,now-self._began)
            self._result=result
        return result

    def cancel(self, context):
        with self._lock:
            if self._cancel_requested:
                return
            self._cancel_requested=True
            self._generation+=1
            cancel=self._dispatched and self._dispatch_complete and not self._cancel_sent
            if cancel:
                self._cancel_sent=True
        if cancel:
            context.require_arm().cancel_goal()
