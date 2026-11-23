"""Native traversal, invocation lifecycle, execution control and physical recovery."""
from unittest.mock import Mock
import ast
from pathlib import Path
import py_trees
import pytest
from src.core.action import AbstractAction, AsyncAction, ActionResult
from src.core.context import Context
from src.core.failure import FailureInfo, FailureCode
from src.tree.bt_node_wrapper import ActionScope, BtActionLeaf
from src.tree.builder import TreeBuilder
from src.tree.plan import Plan, ActionNode
from tests.fixtures.native_tree import tick_action, action_node


class ScriptedAction(AsyncAction):
    def __init__(self, name, attempts, trace):
        super().__init__()
        self.name, self.attempts, self.trace = name, attempts, trace
        self.starts = self.cancels = 0

    def start(self, context):
        self.trace.append(self.name)
        self.states = iter(self.attempts[min(self.starts, len(self.attempts) - 1)])
        self.starts += 1

    def check_result(self, context):
        value = next(self.states)
        return value if isinstance(value, ActionResult) else ActionResult(status=value)

    def cancel(self, context):
        self.cancels += 1


class Workflow(AbstractAction):
    def __init__(self, factory, preparation=None):
        self.factory, self.preparation = factory, preparation
        self.entries = self.completions = self.cancellations = 0

    def build_node(self, context, name):
        return ActionScope(self, context, name, self.factory(context))

    def on_enter(self, context):
        self.entries += 1
        return self.preparation(context) if self.preparation else None

    def on_complete(self, context, status):
        self.completions += 1
        return super().on_complete(context, status)

    def on_cancel(self, context):
        self.cancellations += 1


def test_native_descendants_lifecycle_reentry_and_leaf_defaults():
    ctx, trace = Context(), []
    first = ScriptedAction("first", [["running", "success"]], trace)
    failed = ScriptedAction("failed", [["failure"]], trace)
    final = ScriptedAction("final", [["success"]], trace)
    def topology(context):
        return py_trees.composites.Sequence("Operations", memory=True, children=[first.build_node(context, "First"), py_trees.composites.Selector("Fallback", memory=False, children=[failed.build_node(context, "Failure"), final.build_node(context, "Final")])])
    action = Workflow(topology)
    root = action.build_node(ctx, "Task")
    ids = [node.id for node in root.iterate()]
    assert isinstance(first.build_node(ctx, "Ordinary action"), BtActionLeaf)
    list(root.tick())
    visited = list(root.tick())
    assert root.status == py_trees.common.Status.SUCCESS
    assert any(isinstance(node, py_trees.composites.Selector) for node in visited)
    assert trace == ["first", "failed", "final"]
    assert action.completions == 1
    root.stop(py_trees.common.Status.INVALID)
    list(root.tick()); list(root.tick())
    assert action.entries == action.completions == 2
    assert [node.id for node in root.iterate()] == ids
    assert trace == ["first", "failed", "final"] * 2
    assert len(ctx.world_state.get_action_history()) == 8


@pytest.mark.parametrize("result", [ActionResult(status="needs_user_input"), ActionResult(status="needs_replan", force_replan=True)])
@pytest.mark.parametrize("ancestor", ["selector", "retry", "timeout", "parallel"])
def test_control_latch_prevents_dispatch_even_if_ancestor_changes_status(result, ancestor, monkeypatch):
    ctx, trace = Context(), []
    first = ScriptedAction("request", [[result]], trace).build_node(ctx, "Request")
    other = ScriptedAction("other", [["success"]], trace).build_node(ctx, "Other")
    if ancestor == "selector":
        root = py_trees.composites.Selector("Fallback", memory=False, children=[first, other])
    elif ancestor == "retry":
        root = py_trees.decorators.Retry("Retry", child=first, num_failures=3)
    elif ancestor == "parallel":
        root = py_trees.composites.Parallel("Parallel", policy=py_trees.common.ParallelPolicy.SuccessOnAll(synchronise=False), children=[first, other])
    else:
        clock = [0.0]
        monkeypatch.setattr(py_trees.decorators.time, "monotonic", lambda: clock[0])
        timed = py_trees.decorators.Timeout("Timeout", child=first, duration=1.0)
        root = py_trees.composites.Selector("Fallback after timeout", memory=False, children=[timed, other])
        list(root.tick()); clock[0] = 2.0
    list(root.tick())
    assert trace == ["request"]
    assert ctx.execution.control.result is result
    root.stop(py_trees.common.Status.INVALID)
    assert ctx.execution.control.result is result


def test_preparation_failure_blocks_children_and_fresh_binding_on_reentry():
    ctx, trace = Context(), []
    child = ScriptedAction("old", [["success"]], trace)
    def prepare(context):
        value = context.world_state.get("input")
        if value is None:
            return ActionResult(status="failure", reason="missing input")
        child.name = value
    owner = Workflow(lambda context: child.build_node(context, "Bound operation"), prepare)
    root = owner.build_node(ctx, "Workflow")
    list(root.tick()); assert trace == []
    for value in ["first", "second"]:
        ctx.world_state.set("input", value)
        root.stop(py_trees.common.Status.INVALID); list(root.tick())
    assert trace == ["first", "second"]
    assert owner.completions == 2


def test_timeout_cancels_active_operation_once(monkeypatch):
    ctx, clock = Context(), [0.0]
    monkeypatch.setattr(py_trees.decorators.time, "monotonic", lambda: clock[0])
    child = ScriptedAction("active", [["running", "running"]], [])
    owner = Workflow(lambda context: child.build_node(context, "Active"))
    root = py_trees.decorators.Timeout("Timeout", child=owner.build_node(ctx, "Owner"), duration=1.0)
    list(root.tick()); clock[0] = 2.0; list(root.tick())
    root.stop(py_trees.common.Status.INVALID)
    assert child.cancels == owner.cancellations == 1


def test_native_parallel_success_policy():
    ctx = Context()
    slow = ScriptedAction("slow", [["running"]], [])
    fast = ScriptedAction("fast", [["success"]], [])
    root = py_trees.composites.Parallel("First success", policy=py_trees.common.ParallelPolicy.SuccessOnOne(), children=[slow.build_node(ctx, "Slow"), fast.build_node(ctx, "Fast")])
    list(root.tick())
    assert root.status == py_trees.common.Status.SUCCESS
    assert slow.cancels == 1


@pytest.mark.parametrize("params", [{"composition_backend": "subtree"}, {"unknown": True}])
def test_json_boundary_rejects_obsolete_and_unknown_inputs(params):
    from src.plugin.actions.manipulation import PickAtPose
    ctx = Context(supported_actions={"Pick": PickAtPose})
    with pytest.raises((TypeError, ValueError)):
        TreeBuilder(ctx).build(Plan(root=ActionNode(name="Pick", action_type="Pick", params=params)))


def test_workflows_do_not_schedule_or_execute_children():
    for filename in ["manipulation", "navigation", "observe", "inspect", "grasp_alignment"]:
        source = ast.parse(Path(f"src/plugin/actions/{filename}.py").read_text())
        calls = [node for node in ast.walk(source) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)]
        assert not any(node.func.attr in {"tick", "tick_once", "action_sequence", "action_selector"} for node in calls)
        assert not any(node.func.attr == "execute" and isinstance(node.func.value, ast.Attribute) and node.func.value.attr in {"sequence", "_sequence", "_delegate", "_stage", "selector"} for node in calls)


def _pick_context(monkeypatch, failures, retryable=True):
    from src.plugin.actions import manipulation
    from tests.fixtures.mock_controllers import MockArmController, MockGripperController
    class MoveItArmController(MockArmController):
        def send_pose_goal(self, request):
            super().send_pose_goal(request)
            remaining = failures.get(request.segment, 0)
            if remaining:
                failures[request.segment] -= 1
                self.current_status = "failed"
        def get_error(self):
            return "motion failed" if self.current_status == "failed" else None
        def get_failure(self):
            return FailureInfo(FailureCode.PLANNING_FAILED if retryable else FailureCode.LIMIT_VIOLATION, retryable=retryable) if self.current_status == "failed" else None
    ctx = Context(arm=MoveItArmController(), gripper=MockGripperController())
    def immediately(fn, callback=None, error_callback=None):
        try:
            value = fn()
            if callback: callback(value)
        except Exception as exc:
            if error_callback: error_callback(exc)
            else: raise
    ctx.run_in_context_loop = immediately
    monkeypatch.setattr(manipulation, "_create_mesh_for_collision_world", lambda *a, **k: "target.ply")
    return ctx


def _finish(action, context):
    for _ in range(30):
        result = tick_action(action, context)
        if result.status != "running": return result
    pytest.fail("action did not finish")


def test_pick_without_recovery_command_order(monkeypatch):
    from src.plugin.actions.manipulation import PickAtPose
    ctx = _pick_context(monkeypatch, {})
    assert _finish(PickAtPose(width=.1), ctx).status == "success"
    assert [request.segment for request in ctx.arm.pose_requests] == ["approach", "grasp", "retreat"]
    assert ("attach", "pick_target_collision") in ctx.arm.collision_events


@pytest.mark.parametrize("stage", ["approach", "grasp", "retreat"])
def test_bounded_motion_retry_does_not_replay_closure(stage, monkeypatch):
    from src.plugin.actions.manipulation import PickAtPose
    ctx = _pick_context(monkeypatch, {stage: 1})
    original = ctx.gripper.close
    ctx.gripper.close = Mock(side_effect=lambda force=None, width_m=None: original(force))
    assert _finish(PickAtPose(recovery_enabled=True, width=.1), ctx).status == "success"
    segments = [request.segment for request in ctx.arm.pose_requests]
    assert segments.count(stage) == 2 and len(segments) == 4
    assert ctx.gripper.close.call_count == 1


def test_retreat_exhaustion_preserves_hold_and_prevents_stow(monkeypatch):
    import numpy as np
    from src.plugin.actions.manipulation import PickTarget
    ctx = _pick_context(monkeypatch, {"retreat": 3})
    ctx.world_state.set("cup_estimated_pick_pose_mat", np.eye(4))
    action = PickTarget(target="cup", recovery_enabled=True)
    result = _finish(action, ctx)
    assert result.force_replan and result.suggested_replan_context["grip_state"] == "commanded_hold"
    assert result.suggested_replan_context["failed_stage"] == "retreat"
    assert ctx.arm.named_goal_history == []
    assert ctx.arm.collision_events.count(("clear", None)) == 1
    action_node(action, ctx).stop(py_trees.common.Status.INVALID)
    assert ctx.arm.collision_events.count(("clear", None)) == 1


@pytest.mark.parametrize("failure", ["close", "attach"])
@pytest.mark.parametrize("recovery_enabled", [False, True])
def test_post_close_failure_preserves_scene_and_never_replays_close(failure, recovery_enabled, monkeypatch):
    from src.core.controllers import ControllerResult
    from src.plugin.actions.manipulation import PickAtPose
    ctx = _pick_context(monkeypatch, {})
    if failure == "close":
        ctx.gripper.close = Mock(return_value=ControllerResult(False, error="uncertain closure"))
    else:
        ctx.arm.attach_collision_object = Mock(return_value=False)
    result = _finish(PickAtPose(recovery_enabled=recovery_enabled), ctx)
    assert result.force_replan
    assert result.suggested_replan_context["grip_state"] == ("uncertain" if failure == "close" else "commanded_hold")
    assert [request.segment for request in ctx.arm.pose_requests] == ["approach", "grasp"]
    assert ctx.arm.collision_events.count(("clear", None)) == 1


def test_nonretryable_motion_failure_is_not_repeated(monkeypatch):
    from src.plugin.actions.manipulation import PickAtPose
    ctx = _pick_context(monkeypatch, {"approach": 3}, retryable=False)
    result = _finish(PickAtPose(recovery_enabled=True), ctx)
    assert result.force_replan
    assert len(ctx.arm.pose_requests) == 1


def test_cancel_request_during_parallel_tick_blocks_subsequent_dispatch():
    ctx, trace = Context(), []
    class RequestCancel(AbstractAction):
        def execute(self, context):
            context.execution.cancel_requested = True
            return ActionResult(status="running")
    other = ScriptedAction("later", [["success"]], trace)
    root = py_trees.composites.Parallel("Concurrent", policy=py_trees.common.ParallelPolicy.SuccessOnAll(), children=[RequestCancel().build_node(ctx, "Cancel"), other.build_node(ctx, "Later")])
    list(root.tick())
    assert trace == []
    root.stop(py_trees.common.Status.INVALID)


def test_raw_native_root_records_engine_binding_and_public_outcome():
    trace = []
    class NativeWorkflow(AbstractAction):
        def build_node(self, context, name):
            return py_trees.composites.Sequence(name, memory=False, children=[ScriptedAction("child", [["success"]], trace).build_node(context, "Child")])
    ctx = Context(supported_actions={"Native": NativeWorkflow})
    tree = TreeBuilder(ctx).build(Plan(root=ActionNode(name="Public action", action_type="Native")))
    tree.tick()
    history = ctx.world_state.get_action_history()
    assert [record.planner_visible for record in history] == [False, True]
    assert history[-1].action_type == "Native" and history[-1].result.status == "success"


def test_invocation_reentry_clears_nested_scope_outcomes():
    ctx, trace = Context(), []
    child = ScriptedAction("child", [["failure"], ["success"]], trace)
    inner = Workflow(lambda context: child.build_node(context, "Child"))
    outer = Workflow(lambda context: inner.build_node(context, "Inner"))
    root = outer.build_node(ctx, "Outer")
    list(root.tick()); assert root.status == py_trees.common.Status.FAILURE
    root.stop(py_trees.common.Status.INVALID)
    list(root.tick()); assert root.status == py_trees.common.Status.SUCCESS
    assert inner.completions == outer.completions == 2


def test_async_selection_leaf_discards_cancelled_invocation_results():
    from types import SimpleNamespace
    from src.plugin.actions.navigation import MoveToManipulationPose, _SelectManipulationBasePose
    ctx = Context()
    inputs = MoveToManipulationPose(target="cup")
    inputs._get_fresh_object_state = Mock(return_value=object())
    inputs._accept_selection = Mock()
    queued = []
    ctx.run_in_context_loop = lambda fn, callback=None, error_callback=None: queued.append(callback)
    leaf = _SelectManipulationBasePose(inputs)
    node = leaf.build_node(ctx, "Select")
    list(node.tick())
    old_callback = queued[0]
    node.stop(py_trees.common.Status.INVALID)
    list(node.tick())
    result = SimpleNamespace(selected=object())
    old_callback(result)
    inputs._accept_selection.assert_not_called()
    assert leaf.check_result(ctx).status == "running"
    queued[1](result)
    assert leaf.check_result(ctx).status == "success"
    inputs._accept_selection.assert_called_once_with(ctx, result)
    assert not hasattr(inputs, "status")


def test_pick_runtime_pose_updates_declared_child_inputs_and_requests(monkeypatch):
    import numpy as np
    from src.plugin.actions.manipulation import PickTarget
    ctx = _pick_context(monkeypatch, {})
    action = PickTarget(target="cup")
    root = action_node(action, ctx)
    ids = [node.id for node in root.iterate()]
    for x in [.3, .8]:
        matrix = np.eye(4)
        matrix[0, 3] = x
        ctx.world_state.set("cup_estimated_pick_pose_mat", matrix)
        root.stop(py_trees.common.Status.INVALID)
        assert _finish(action, ctx).status == "success"
        assert action._pick.param_dict()["x"] == pytest.approx(x)
        assert action._pick.param_dict()["object_id"] == "cup"
        assert ctx.arm.pose_requests[-2].pose.x == pytest.approx(x)
        assert [node.id for node in root.iterate()] == ids


@pytest.mark.parametrize("retryable,expected", [(False, 1), (True, 2)])
def test_json_retry_obeys_structured_failure_eligibility(retryable, expected):
    from src.tree.plan import DecoratorNode
    trace = []
    class Operation(AbstractAction):
        def execute(self, context):
            trace.append("attempt")
            return ActionResult(status="success") if len(trace) == 2 else ActionResult(status="failure", failure=FailureInfo(FailureCode.PLANNING_FAILED, retryable=retryable))
    ctx = Context(supported_actions={"Operation": Operation})
    tree = TreeBuilder(ctx).build(Plan(root=DecoratorNode(type="retry", name="Retry operation", params={"num_attempts": 2}, child=ActionNode(name="Operation", action_type="Operation"))))
    tree.tick(); tree.tick()
    assert len(trace) == expected
    assert tree.root.status == (py_trees.common.Status.SUCCESS if retryable else py_trees.common.Status.FAILURE)


def test_parent_retry_cannot_replay_closure_after_stow_failure(monkeypatch):
    import numpy as np
    from src.plugin.actions.manipulation import PickTarget
    from src.tree.plan import DecoratorNode
    ctx = _pick_context(monkeypatch, {})
    ctx.supported_actions = {"Pick": PickTarget}
    ctx.world_state.set("cup_estimated_pick_pose_mat", np.eye(4))
    original_close = ctx.gripper.close
    ctx.gripper.close = Mock(side_effect=lambda force=None, width_m=None: original_close(force))
    ctx.arm.send_named_goal = Mock(side_effect=lambda *args, **kwargs: setattr(ctx.arm, "current_status", "failed"))
    tree = TreeBuilder(ctx).build(Plan(root=DecoratorNode(type="retry", name="Retry pick", params={"num_attempts": 3}, child=ActionNode(name="Pick", action_type="Pick", params={"target": "cup"}))))
    for _ in range(30):
        tree.tick()
        if ctx.execution.control is not None:
            break
    assert ctx.execution.control.result.suggested_replan_context["failed_stage"] == "stow"
    tree.tick()
    assert ctx.gripper.close.call_count == 1
    assert ctx.arm.collision_events.count(("clear", None)) == 1
