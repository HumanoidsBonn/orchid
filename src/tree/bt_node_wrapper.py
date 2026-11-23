"""Native action leaves and a single workflow lifecycle decorator."""
from __future__ import annotations
import time
import py_trees
from src.core.action import ActionResult, AsyncAction
from src.core.failure import FailureInfo, FailureCode
from src.core.world_state import ActionRecord
from src.core.serialization import to_json_safe
import src.core.logger as log


def _begin(node):
    node._result = None
    node._recorded = False
    node._start_time = time.time()
    node._monotonic_start_time = time.monotonic()
    node._last_duration_s = None
    node._last_operation = None
    node._last_console_progress_s = 0.0


def _record(node):
    result = node._result
    if result is None or result.status == "running" or node._recorded:
        return
    node._recorded = True
    if result.status not in {"success", "running"}:
        log.warn("Action failed name=%s type=%s status=%s reason=%s failure=%s",
                 node.name, type(node.action).__name__, result.status, result.reason, result.failure)
    node.binding.result = result
    node._last_duration_s = max(0.0, time.monotonic() - node._monotonic_start_time)
    record = ActionRecord(action_name=node.binding.name, params=node.action.param_dict(), result=result,
        timestamp=node._start_time, duration_s=node._last_duration_s, bt_node_name=node.name,
        action_class=type(node.action).__name__, action_type=node.binding.action_type,
        planner_visible=node.binding.planner_visible)
    node.context.world_state.append_action_record(record)
    node.context.execution.request_control(node, result)
    control = node.context.execution.control
    if control is None or control.node_id == node.id:
        node.context.world_state.set("last_action_result", {
            "action_name": record.action_name, "bt_node_name": node.name,
            "action_class": record.action_class, "action_type": record.action_type,
            "status": result.status, "reason": result.reason,
            "suggested_replan_context": to_json_safe(result.suggested_replan_context),
            "payload": to_json_safe(result.payload), "failure": to_json_safe(result.failure),
            "force_replan": result.force_replan, "duration_s": record.duration_s,
            "params": to_json_safe(record.params)})


def _progress(node, status="running"):
    elapsed = max(0.0, time.monotonic() - node._monotonic_start_time)
    if status == "running" and isinstance(node.action, AsyncAction) and elapsed-node._last_console_progress_s >= 2.:
        log.info("Action running name=%s type=%s elapsed_s=%.1f", node.name, type(node.action).__name__, elapsed)
        node._last_console_progress_s = elapsed
    operation = {"sequence_label": node.name, "current_action_name": node.name,
                 "current_action_index": 1, "total_actions": 1, "elapsed_s": elapsed,
                 "current_action_elapsed_s": elapsed, "status": status,
                 "message": f"{node.name} ({elapsed:.1f}s)"}
    node._last_operation = operation
    node.context.world_state.set("current_operation", operation)


def _status(node):
    _record(node)
    if node.context.execution.control is not None:
        return py_trees.common.Status.RUNNING
    return {"success": py_trees.common.Status.SUCCESS, "running": py_trees.common.Status.RUNNING}.get(
        node._result.status, py_trees.common.Status.FAILURE)


def terminal_failure(context, root):
    """Read the latest terminal descendant failure, including native-invalidated leaves."""
    failures = []
    for node in root.iterate():
        binding = context.execution.bindings.get(node.id)
        result = binding.result if binding else getattr(node, "result", None)
        if result is not None and result.status not in {"success", "running"}:
            failures.append((getattr(node, "_monotonic_start_time", 0.), result))
    return max(failures, key=lambda row: row[0])[1] if failures else None


def retry_eligible(context, child, retry=None):
    """Native Retry guard: permit active work, retry only explicitly eligible failures."""
    if context.execution.control is not None or context.execution.cancel_requested:
        return False
    if retry is not None and retry.status != py_trees.common.Status.RUNNING:
        return True
    if child.status == py_trees.common.Status.RUNNING:
        return True
    tip = child.tip()
    binding = context.execution.bindings.get(child.id) or (context.execution.bindings.get(tip.id) if tip else None)
    result = getattr(child, "result", None) or (binding.result if binding else None) or getattr(tip, "result", None)
    if result is None:
        return child.status != py_trees.common.Status.FAILURE
    return result.status in {"success", "running"} or bool(result.failure and result.failure.retryable)


class BtActionLeaf(py_trees.behaviour.Behaviour):
    """Adapter for a single operation; it never ticks another behaviour."""
    def __init__(self, action, context, name):
        super().__init__(name=name)
        self.action = action
        self.context = context
        context.execution.bind(self, action)
        self.binding = context.execution.bindings[self.id]
        self._result = None
        self._start_time = 0.0
        self._last_duration_s = None
        self._last_operation = None
        self._active = False
        self._invoked = False

    @property
    def result(self):
        return self._result

    @property
    def started(self):
        return self._invoked

    def initialise(self):
        _begin(self)
        self._invoked = False
        if isinstance(self.action, AsyncAction):
            self.action.reset()
        self._active = False

    def update(self):
        if self.context.execution.control is not None or self.context.execution.cancel_requested:
            return py_trees.common.Status.RUNNING
        try:
            if not self.action.pre_condition(self.context):
                self._result = ActionResult(status="failure", reason="Pre-condition failed",
                    failure=FailureInfo(FailureCode.CONDITION_FAILED))
            else:
                self._active = True
                self._invoked = True
                _progress(self)
                self._result = self.action.execute(self.context)
                if self._result.status == "success" and not self.action.post_condition(self.context):
                    self._result = ActionResult(status="failure", reason="Post-condition failed",
                        failure=FailureInfo(FailureCode.CONDITION_FAILED))
        except Exception as exc:
            log.exception("%s execution failed: %s", self.name, exc)
            self._result = ActionResult(status="failure", reason=str(exc), failure=FailureInfo())
        if self._result.status != "running":
            _progress(self, self._result.status)
        return _status(self)

    def terminate(self, new_status):
        if self._active and (self._result is None or self._result.status == "running"):
            if isinstance(self.action, AsyncAction):
                try:
                    self.action.cancel(self.context)
                except Exception as exc:
                    log.exception("%s cancellation failed: %s", self.name, exc)
            self._result = ActionResult(status="failure", reason="Action cancelled",
                failure=FailureInfo(FailureCode.CANCELLED))
            _record(self)
        self._active = False

    _map_action_result_to_bt_status = staticmethod(lambda result: {
        "success": py_trees.common.Status.SUCCESS, "running": py_trees.common.Status.RUNNING
    }.get(result.status, py_trees.common.Status.FAILURE))


class ActionScope(py_trees.decorators.Decorator):
    """Invocation preparation and outcome handling around an existing native root."""
    def __init__(self, action, context, name, child, *, on_enter=None, on_complete=None, on_cancel=None):
        super().__init__(name=name, child=child)
        self.action = action
        self.context = context
        context.execution.bind(self, action)
        self.binding = context.execution.bindings[self.id]
        self._result = None
        self._start_time = 0.0
        self._on_enter = on_enter or action.on_enter
        self._on_complete = on_complete or action.on_complete
        self._on_cancel = on_cancel or action.on_cancel
        self._entry_failure = None
        self._completed = False
        self._active = False

    @property
    def result(self):
        return self._result

    def initialise(self):
        _begin(self)
        self._completed = False
        self._entry_failure = None
        self._active = False
        if self.context.execution.control is not None or self.context.execution.cancel_requested:
            return
        self.decorated.stop(py_trees.common.Status.INVALID)
        for node in self.decorated.iterate():
            binding = self.context.execution.bindings.get(node.id)
            if binding:
                binding.result = None
            if isinstance(node, (BtActionLeaf, ActionScope)):
                node._result = None
                node._start_time = 0.0
            if isinstance(node, BtActionLeaf):
                node._invoked = False
        try:
            if not self.action.pre_condition(self.context):
                self._entry_failure = ActionResult(status="failure", reason="Pre-condition failed",
                    failure=FailureInfo(FailureCode.CONDITION_FAILED))
            else:
                self._active = True
                self._entry_failure = self._on_enter(self.context)
                if self._entry_failure is not None and self._entry_failure.status == "running":
                    raise ValueError("Workflow preparation must be an executable leaf when it runs asynchronously")
        except Exception as exc:
            self._entry_failure = ActionResult(status="failure", reason=str(exc),
                failure=FailureInfo(FailureCode.INVALID_REQUEST))

    def tick(self):
        if self.status != py_trees.common.Status.RUNNING:
            self.initialise()
        if self.context.execution.control is not None or self.context.execution.cancel_requested:
            self.status = py_trees.common.Status.RUNNING
            yield self
            return
        if self._entry_failure is None:
            yield from self.decorated.tick()
        new_status = self.update()
        if new_status != py_trees.common.Status.RUNNING:
            self.stop(new_status)
        self.status = new_status
        yield self

    def update(self):
        if self.context.execution.control is not None or self.context.execution.cancel_requested:
            return py_trees.common.Status.RUNNING
        status = self.decorated.status
        if self._entry_failure is not None:
            self._result = self._entry_failure
        elif status == py_trees.common.Status.RUNNING:
            return status
        elif not self._completed:
            self._completed = True
            try:
                self._result = self._on_complete(self.context, status)
                if self._result.status == "running":
                    raise ValueError("Workflow completion must return a terminal ActionResult")
                tip = self.decorated.tip()
                cause = getattr(tip, "result", None) or terminal_failure(self.context, self.decorated)
                if self._result.status in {"failure", "needs_replan"} and cause is not None:
                    self._result.reason = self._result.reason or cause.reason
                    if self._result.failure is None or self._result.failure == FailureInfo():
                        self._result.failure = cause.failure
                if self._result.status == "success" and not self.action.post_condition(self.context):
                    self._result = ActionResult(status="failure", reason="Post-condition failed",
                        failure=FailureInfo(FailureCode.CONDITION_FAILED))
            except Exception as exc:
                self._result = ActionResult(status="failure", reason=str(exc), failure=FailureInfo())
        return _status(self)

    def terminate(self, new_status):
        if new_status == py_trees.common.Status.INVALID and self._active and not self._completed:
            try:
                self._on_cancel(self.context)
            except Exception as exc:
                log.exception("%s cancellation hook failed: %s", self.name, exc)
            if self._result is None or self._result.status == "running":
                control = self.context.execution.control
                self._result = control.result if control else ActionResult(status="failure", reason="Workflow cancelled", failure=FailureInfo(FailureCode.CANCELLED))
                _record(self)
        self._active = False


class BtConditionLeaf(py_trees.behaviour.Behaviour):
    """Wraps a condition check (callable) in a py_trees leaf.

    Condition should be a simple callable: (context) -> bool
    """

    def __init__(self, condition, context: Context, name: str | None = None):
        """Initialize BtConditionLeaf.

        Args:
            condition: Callable that returns bool given context
            context: Context to pass to condition
        """
        super().__init__(name=name or getattr(condition, "__name__", condition.__class__.__name__))
        self.condition = condition
        self.context = context

    def update(self) -> py_trees.common.Status:
        """Evaluate condition and return SUCCESS or FAILURE.

        Returns:
            py_trees.common.Status.SUCCESS if condition true, else FAILURE
        """
        try:
            result = self.condition(self.context)
            if not isinstance(result, bool):
                raise TypeError("Conditions must return bool")
            self.feedback_message = "true" if result else "false"
            status = (
                py_trees.common.Status.SUCCESS
                if result
                else py_trees.common.Status.FAILURE
            )
            log.debug("%s.update() -> %s (result=%s)", self.name, status.value, result)
            return status
        except Exception as e:
            self.feedback_message = f"{type(e).__name__}: {e}"
            log.exception("%s.update() raised exception: %s", self.name, e)
            return py_trees.common.Status.FAILURE
