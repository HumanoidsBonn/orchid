"""Tests for session task intake and LLM planning integration."""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from src.core.action import ActionResult
from src.core.llm_planner import PlanGenerationError, UserInputRequired
from src.core.world_state import ActionRecord
from src.orchestrator.session import RobotSession
from src.tree.execution import ExecutionStatus
from src.tree.plan import ActionNode, CompositeNode, Plan
from tests.fixtures.mock_actions import SuccessAction
from tests.fixtures.mock_manifests import basic_registry, rich_manifest


class TestSessionTaskIntake:
    def test_submit_task_queues_new_command(self, rich_manifest, basic_registry):
        session = RobotSession(rich_manifest, basic_registry)
        assert session.submit_task("go to the kitchen") is True
        assert session._consume_pending_task() == "go to the kitchen"

    def test_submit_task_rejects_empty_command(self, rich_manifest, basic_registry):
        session = RobotSession(rich_manifest, basic_registry)
        assert session.submit_task("   ") is False

    def test_submit_task_ignores_duplicate_pending(self, rich_manifest, basic_registry):
        session = RobotSession(rich_manifest, basic_registry)
        assert session.submit_task("pick up the cup") is True
        assert session.submit_task("pick up the cup") is False

    def test_submit_task_ignores_task_already_executing(
        self, rich_manifest, basic_registry
    ):
        session = RobotSession(rich_manifest, basic_registry)
        session._active_task = "navigate home"
        session.executor._snapshot.status = ExecutionStatus.RUNNING

        assert session.submit_task("navigate home") is False


class TestSessionRunLoop:
    def test_run_loop_plans_and_starts_execution(
        self, rich_manifest, basic_registry
    ):
        session = RobotSession(rich_manifest, basic_registry)
        session.context.supported_actions["SuccessAction"] = SuccessAction
        action = ActionNode(id="a1", action_type="SuccessAction", name="Task1")
        plan = Plan(goal="do something", root=action)

        with patch.object(session.planner, "generate_plan", return_value=plan):
            session.submit_task("do something")
            session._handle_new_task("do something")

        assert session.executor.join(timeout=2.0)
        snapshot = session.get_execution_snapshot()
        assert snapshot.goal == "do something"
        assert snapshot.status == ExecutionStatus.SUCCEEDED

    def test_run_loop_cancels_active_plan_for_new_task(
        self, rich_manifest, basic_registry
    ):
        session = RobotSession(rich_manifest, basic_registry)
        session.context.supported_actions["SuccessAction"] = SuccessAction
        fast_action = ActionNode(id="a2", action_type="SuccessAction", name="Fast")
        fast_plan = Plan(goal="fast task", root=fast_action)
        session.executor._snapshot.status = ExecutionStatus.RUNNING

        with patch.object(session.executor, "stop") as mock_stop:
            with patch.object(session.executor, "start_plan") as mock_start:
                with patch.object(
                    session.planner, "generate_plan", return_value=fast_plan
                ):
                    session._handle_new_task("fast task")

                mock_stop.assert_called_once_with(wait=True)
                mock_start.assert_called_once()
                start_args, start_kwargs = mock_start.call_args
                assert start_args[0] == fast_plan
                assert "tree" in start_kwargs
                assert start_kwargs["tree"] is not None

    def test_run_loop_reports_planning_failure(
        self, rich_manifest, basic_registry
    ):
        session = RobotSession(rich_manifest, basic_registry)

        with patch.object(
            session.planner,
            "generate_plan",
            side_effect=PlanGenerationError("bad json"),
        ):
            session._handle_new_task("broken task")

        assert session._active_task is None
        assert session.get_execution_status() == ExecutionStatus.IDLE

    def test_handle_new_task_reports_planning_build_and_execution_phases(
        self, rich_manifest, basic_registry
    ):
        session = RobotSession(rich_manifest, basic_registry)
        session.context.supported_actions["SuccessAction"] = SuccessAction
        plan = Plan(
            goal="fast task",
            root=ActionNode(id="a2", action_type="SuccessAction", name="Fast"),
        )

        with patch.object(session.planner, "generate_plan", return_value=plan):
            with patch.object(session, "_set_current_operation", wraps=session._set_current_operation) as mock_operation:
                session._handle_new_task("fast task")

        labels = [call.args[0] for call in mock_operation.call_args_list]
        messages = [call.args[1] for call in mock_operation.call_args_list]

        assert "Task" in labels
        assert "Planning" in labels
        assert "Build Tree" in labels
        assert "Execution" in labels
        assert "Planning with LLM" in messages
        assert "Building behavior tree" in messages
        assert "Executing: fast task" in messages

    def test_handle_new_task_reports_tree_build_failure_before_replan(
        self, rich_manifest, basic_registry
    ):
        session = RobotSession(rich_manifest, basic_registry)
        session.context.supported_actions["SuccessAction"] = SuccessAction
        first_plan = Plan(
            goal="bad tree",
            root=ActionNode(id="a1", action_type="MissingAction", name="Missing"),
        )
        second_plan = Plan(
            goal="finish",
            root=ActionNode(id="a2", action_type="SuccessAction", name="Finish"),
        )

        with patch.object(
            session.planner,
            "generate_plan",
            side_effect=[first_plan, second_plan],
        ):
            with patch.object(session.executor, "build_tree", side_effect=[RuntimeError("bad action"), MagicMock()]):
                with patch.object(session.executor, "join", return_value=True):
                    with patch.object(session.executor, "start_plan") as mock_start:
                        session.executor._snapshot.status = ExecutionStatus.SUCCEEDED
                        with patch.object(session, "_set_current_operation", wraps=session._set_current_operation) as mock_operation:
                            session._handle_new_task("recover")

        messages = [call.args[1] for call in mock_operation.call_args_list]
        assert "Tree build failed: bad action" in messages
        assert "Replanning after tree build failed" in messages
        mock_start.assert_called_once()

    def test_handle_new_task_preserves_original_command_for_clarification(
        self, rich_manifest, basic_registry
    ):
        session = RobotSession(rich_manifest, basic_registry)
        session.context.supported_actions["SuccessAction"] = SuccessAction
        plan = Plan(
            goal="fast task",
            root=ActionNode(id="a2", action_type="SuccessAction", name="Fast"),
        )

        with patch.object(
            session.planner,
            "generate_plan",
            side_effect=[UserInputRequired("Where should I go?"), plan],
        ) as mock_generate_plan:
            with patch.object(session, "request_user_input", return_value="kitchen"):
                with patch.object(session.executor, "build_tree", return_value=MagicMock()):
                    with patch.object(session.executor, "start_plan") as mock_start:
                        session._handle_new_task("find the cup")

        assert mock_generate_plan.call_count == 2
        assert mock_generate_plan.call_args_list[0][0][0] == "find the cup"
        assert mock_generate_plan.call_args_list[1][0][0] == "find the cup"
        assert mock_generate_plan.call_args_list[1][1]["user_feedback"] == "kitchen"
        mock_start.assert_called_once()

    def test_handle_new_task_replans_with_user_input_payload(
        self, rich_manifest, basic_registry
    ):
        session = RobotSession(rich_manifest, basic_registry)
        session.context.supported_actions["SuccessAction"] = SuccessAction
        first_plan = Plan(
            goal="ask",
            root=ActionNode(id="a1", action_type="SuccessAction", name="Ask"),
        )
        second_plan = Plan(
            goal="finish",
            root=ActionNode(id="a2", action_type="SuccessAction", name="Finish"),
        )

        def start_side_effect(*args, **kwargs):
            plan = args[0]
            if plan.goal != "finish":
                session.context.world_state.append_action_record(
                    ActionRecord(
                        action_name="MoveToTarget",
                        params={"target": "table"},
                        result=ActionResult(status="success"),
                        timestamp=time.time(),
                        duration_s=0.1,
                    )
                )
                session.context.world_state.set(
                    "last_action_result",
                    {
                        "action_name": "AskUser",
                        "status": "needs_replan",
                        "reason": "Which shelf?",
                        "suggested_replan_context": {"user_input": "top shelf"},
                        "payload": {"user_input": "top shelf"},
                    },
                )
                session.executor._snapshot.status = ExecutionStatus.FAILED
            else:
                session.executor._snapshot.status = ExecutionStatus.SUCCEEDED

        with patch.object(
            session.planner,
            "generate_plan",
            side_effect=[first_plan, second_plan],
        ) as mock_generate_plan:
            with patch.object(session.executor, "build_tree", return_value=MagicMock()):
                with patch.object(session.executor, "join", return_value=True):
                    with patch.object(session.executor, "start_plan", side_effect=start_side_effect):
                        session._handle_new_task("find the cup")

        assert mock_generate_plan.call_count == 2
        assert mock_generate_plan.call_args_list[1][1]["user_feedback"] == "top shelf"
        action_history = mock_generate_plan.call_args_list[1][1]["action_history"]
        assert action_history[0]["action_name"] == "MoveToTarget"
        assert action_history[0]["result"]["status"] == "success"
