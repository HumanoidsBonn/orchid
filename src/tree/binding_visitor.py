"""Engine observations for action-defined native roots without an ActionScope."""
import time
import py_trees
from src.core.action import ActionResult
from src.core.world_state import ActionRecord
from src.core.serialization import to_json_safe


class ActionBindingsVisitor(py_trees.visitors.VisitorBase):
    def __init__(self, context):
        super().__init__(full=False)
        self.context = context
        self.started = {}
        self.previous = {}

    def run(self, node):
        binding = self.context.execution.bindings.get(node.id)
        if binding is None or hasattr(node, "result"):
            return
        status = node.status
        if self.previous.get(node.id) != py_trees.common.Status.RUNNING:
            self.started[node.id] = time.monotonic()
        self.previous[node.id] = status
        if status not in {py_trees.common.Status.SUCCESS, py_trees.common.Status.FAILURE}:
            return
        result = None
        for descendant in node.iterate():
            if getattr(descendant, "action", None) is binding.action:
                result = getattr(descendant, "result", None)
                if result is not None:
                    break
        if result is None or status == py_trees.common.Status.FAILURE:
            tip = node.tip()
            cause = getattr(tip, "result", None)
            result = ActionResult(status="success" if status == py_trees.common.Status.SUCCESS else "failure",
                                  reason=cause.reason if cause else None, failure=cause.failure if cause else None)
        binding.result = result
        duration = max(0., time.monotonic() - self.started[node.id])
        record = ActionRecord(action_name=binding.name, params=binding.action.param_dict(), result=result,
            timestamp=time.time() - duration, duration_s=duration, bt_node_name=node.name,
            action_class=type(binding.action).__name__, action_type=binding.action_type, planner_visible=binding.planner_visible)
        self.context.world_state.append_action_record(record)
        self.context.execution.request_control(node, result)
        if self.context.execution.control is None:
            self.context.world_state.set("last_action_result", {"action_name": record.action_name, "bt_node_name": node.name,
                "action_class": record.action_class, "action_type": record.action_type, "status": result.status,
                "reason": result.reason, "force_replan": result.force_replan, "suggested_replan_context": to_json_safe(result.suggested_replan_context),
                "payload": to_json_safe(result.payload), "failure": to_json_safe(result.failure), "duration_s": duration})
