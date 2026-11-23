"""Tests for async plan execution and execution state reporting."""

import time

from src.tree.execution import ExecutionEngine, ExecutionStatus
from src.tree.plan import ActionNode, Plan
from tests.fixtures.mock_contexts import context_full_controllers


class TestExecutionEngineAsync:
    def test_start_plan_runs_in_background(self, context_full_controllers):
        engine = ExecutionEngine(context=context_full_controllers)
        action = ActionNode(id="a1", action_type="SuccessAction", name="Task1")
        plan = Plan(goal="background task", root=action)

        engine.start_plan(plan)
        assert engine.is_running

        assert engine.join(timeout=2.0)
        snapshot = engine.get_snapshot()
        assert snapshot.status == ExecutionStatus.SUCCEEDED
        assert snapshot.goal == "background task"
        assert snapshot.success is True

    def test_stop_cancels_running_plan(self, context_full_controllers):
        engine = ExecutionEngine(context=context_full_controllers)
        action = ActionNode(id="a1", action_type="SuccessAction", name="Task1")
        plan = Plan(goal="cancel me", root=action)

        engine.start_plan(plan)
        deadline = time.time() + 1.0
        while not engine.is_running and time.time() < deadline:
            time.sleep(0.01)

        engine.stop(wait=True)
        snapshot = engine.get_snapshot()
        assert snapshot.status in {
            ExecutionStatus.CANCELLED,
            ExecutionStatus.SUCCEEDED,
        }

    def test_start_plan_replaces_previous_execution(self, context_full_controllers):
        engine = ExecutionEngine(context=context_full_controllers)
        first = Plan(
            goal="first",
            root=ActionNode(id="a1", action_type="SuccessAction", name="First"),
        )
        second = Plan(
            goal="second",
            root=ActionNode(id="a2", action_type="SuccessAction", name="Second"),
        )

        engine.start_plan(first)
        engine.start_plan(second)
        assert engine.join(timeout=2.0)

        snapshot = engine.get_snapshot()
        assert snapshot.goal == "second"
        assert snapshot.status == ExecutionStatus.SUCCEEDED
