import builtins
import json
import sys
import time
import types

import pytest

from src.orchestrator.session import RobotSession
from src.orchestrator.session import Ros2RobotSessionNode
from src.tree.execution import PlanExecutionSnapshot
from tests.fixtures.mock_manifests import rich_manifest, basic_registry

def test_robot_session_node_optional(rich_manifest, basic_registry):
    """Test RobotSession conditionally creates ROS2 node if available."""
    session = RobotSession(rich_manifest, basic_registry)
    if pytest.HAS_RCLPY:
        assert session.ros2_wrapper is not None
        # user input callback path (node wrapper)
        called = {}
        def cb(prompt):
            called["prompt"] = prompt
            return "ok"
        session.ros2_wrapper.set_user_input_callback(cb)
        res = session.ros2_wrapper.request_user_input("hello")
        assert res == "ok"
        session.ros2_wrapper.node.destroy_node()
    else:
        assert session.ros2_wrapper is None
        # base fallback returns empty string unless terminal input is available
        assert session.request_user_input("x") == ""


def test_request_user_input_terminal_fallback(monkeypatch, rich_manifest, basic_registry):
    """Terminal input is used when ROS2 user input is not available."""
    session = RobotSession(rich_manifest, basic_registry)
    session.ros2_wrapper = None

    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(builtins, "input", lambda prompt: "typed answer")

    assert session.request_user_input("Where should I go?") == "typed answer"


def test_ros2_feedback_uses_current_operation():
    operation = {
        "sequence_label": "PickAtPose",
        "current_action_name": "_MoveArmToPose[grasp]",
        "current_action_index": 4,
        "total_actions": 6,
        "elapsed_s": 2.3,
        "message": "PickAtPose: _MoveArmToPose[grasp] (4/6, 2.3s)",
        "status": "running",
    }
    snapshot = PlanExecutionSnapshot(goal="pick mug")

    assert Ros2RobotSessionNode._feedback_message(operation, snapshot, "pick mug") == (
        "PickAtPose: _MoveArmToPose[grasp] (4/6, 2.3s)"
    )
    assert Ros2RobotSessionNode._feedback_percent(operation, snapshot) == pytest.approx(52.5)
    assert Ros2RobotSessionNode._active_node(operation, snapshot) == (
        "PickAtPose: _MoveArmToPose[grasp]"
    )
    assert Ros2RobotSessionNode._elapsed_label(operation) == "Elapsed 2.3s"


def test_ros2_feedback_uses_session_phase_elapsed_and_percent():
    operation = {
        "sequence_label": "Planning",
        "current_action_name": "Planning with LLM",
        "current_action_index": 1,
        "total_actions": 1,
        "elapsed_s": 0.0,
        "started_monotonic": time.monotonic() - 1.2,
        "message": "Planning with LLM",
        "status": "running",
        "percent_completed": 5.0,
    }
    snapshot = PlanExecutionSnapshot(goal="pick mug")

    assert Ros2RobotSessionNode._feedback_message(
        operation, snapshot, "pick mug"
    ).startswith("Planning with LLM (")
    assert Ros2RobotSessionNode._feedback_percent(operation, snapshot) == pytest.approx(5.0)
    assert Ros2RobotSessionNode._active_node(operation, snapshot) == (
        "Planning: Planning with LLM"
    )
    assert Ros2RobotSessionNode._elapsed_label(operation).startswith("Elapsed 1.")


def test_publish_gui_state_includes_operation_and_task_elapsed(monkeypatch):
    std_msgs = types.ModuleType("std_msgs")
    msg_module = types.ModuleType("std_msgs.msg")

    class _String:
        def __init__(self):
            self.data = ""

    msg_module.String = _String
    std_msgs.msg = msg_module
    monkeypatch.setitem(sys.modules, "std_msgs", std_msgs)
    monkeypatch.setitem(sys.modules, "std_msgs.msg", msg_module)

    node = object.__new__(Ros2RobotSessionNode)
    node.robot_id = "test_robot"
    node._goal_started_monotonic = time.monotonic() - 3.0
    node.gui_state_publisher = None
    node.node = None
    published = {}

    class _Publisher:
        def publish(self, msg):
            published["data"] = msg.data

    node.gui_state_publisher = _Publisher()
    node._current_operation = lambda: {
        "sequence_label": "PickAtPose",
        "current_action_name": "_MoveArmToPose[grasp]",
        "current_action_elapsed_s": 1.2,
        "elapsed_s": 2.0,
        "started_monotonic": time.monotonic() - 2.0,
    }

    node.publish_gui_state(
        robot_status="Running",
        active_node="PickAtPose: _MoveArmToPose[grasp]",
        eta="Elapsed 2.0s",
    )

    payload = json.loads(published["data"])
    assert payload["task_elapsed_s"] >= 2.9
    assert payload["operation"]["current_action_name"] == "_MoveArmToPose[grasp]"
    assert payload["operation"]["current_action_elapsed_s"] == pytest.approx(1.2)
    assert "started_monotonic" not in payload["operation"]


@pytest.mark.requires_ros2
def test_robot_session_node_lifecycle_and_cleanup(rich_manifest, basic_registry):
    """Test ROS2 node lifecycle and cleanup."""
    session = RobotSession(rich_manifest, basic_registry)
    assert session.ros2_wrapper is not None
    session.ros2_wrapper.node.destroy_node()
