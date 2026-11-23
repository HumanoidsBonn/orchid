"""Tests for direct ros2_control trajectory routing."""

from __future__ import annotations

import sys
import types

import pytest

from src.plugin.controllers.ros2_control_trajectory import (
    Ros2ControlTrajectoryClient,
    _ControllerGoal,
)


def test_synchronized_start_lead_default_and_override():
    assert Ros2ControlTrajectoryClient({}).synchronized_start_lead_s == 5.
    assert Ros2ControlTrajectoryClient({'synchronized_start_lead_s': 12.}).synchronized_start_lead_s == 12.


@pytest.mark.parametrize('lead', [0., .5, float('nan'), float('inf')])
def test_invalid_synchronized_start_lead_is_rejected(lead):
    with pytest.raises(ValueError, match='synchronized_start_lead_s'):
        Ros2ControlTrajectoryClient({'synchronized_start_lead_s': lead})


def test_split_by_controller_routes_arm_and_head_joints():
    """Partial joint goals should expand to each controller's full joint list."""
    joint_state = types.SimpleNamespace(
        name=["arm_lift_joint", "arm_flex_joint", "head_pan_joint", "head_tilt_joint"],
        position=[0.0, -0.6, 0.2, -0.1],
    )
    client = Ros2ControlTrajectoryClient(
        {
            "ros2_control": {
                "controllers": [
                    {
                        "name": "arm",
                        "action": "/arm_trajectory_controller/follow_joint_trajectory",
                        "joints": ["arm_lift_joint", "arm_flex_joint"],
                    },
                    {
                        "name": "head",
                        "action": "/head_trajectory_controller/follow_joint_trajectory",
                        "joints": ["head_pan_joint", "head_tilt_joint"],
                    },
                ]
            }
        },
        joint_state_provider=lambda: joint_state,
    )

    split = client._split_by_controller(
        {
            "arm_lift_joint": 0.1,
            "head_tilt_joint": -0.4,
            "unknown_joint": 1.0,
        }
    )

    assert [(goal.name, joints) for goal, joints in split] == [
        ("arm", {"arm_lift_joint": 0.1, "arm_flex_joint": -0.6}),
        ("head", {"head_pan_joint": 0.2, "head_tilt_joint": -0.4}),
    ]
    assert split[0][0].expanded_from_partial is True
    assert split[0][0].requested_joints == ["arm_lift_joint"]


def test_split_by_controller_fails_partial_goal_without_current_state():
    client = Ros2ControlTrajectoryClient(
        {
            "ros2_control": {
                "controllers": [
                    {
                        "name": "arm",
                        "action": "/arm_trajectory_controller/follow_joint_trajectory",
                        "joints": ["arm_lift_joint", "arm_flex_joint"],
                    },
                ]
            }
        }
    )

    with pytest.raises(ValueError, match="Cannot expand ros2_control partial joint goal"):
        client._split_by_controller({"arm_lift_joint": 0.1})


def test_split_trajectory_by_controller_preserves_timing():
    """Planned trajectories should be split by controller with point timing intact."""
    from builtin_interfaces.msg import Duration
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

    client = Ros2ControlTrajectoryClient(
        {
            "ros2_control": {
                "controllers": [
                    {
                        "name": "arm",
                        "action": "/arm_trajectory_controller/follow_joint_trajectory",
                        "joints": ["arm_lift_joint", "arm_flex_joint"],
                    },
                    {
                        "name": "head",
                        "action": "/head_trajectory_controller/follow_joint_trajectory",
                        "joints": ["head_pan_joint"],
                    },
                ]
            }
        }
    )
    trajectory = JointTrajectory()
    trajectory.joint_names = ["arm_lift_joint", "head_pan_joint", "unknown_joint"]
    point = JointTrajectoryPoint()
    point.positions = [0.1, -0.2, 0.3]
    point.velocities = [0.01, 0.02, 0.03]
    point.time_from_start = Duration(sec=2, nanosec=100)
    trajectory.points = [point]

    split = client._split_trajectory_by_controller(trajectory)

    assert [goal.name for goal, _trajectory in split] == ["arm", "head"]
    arm_traj = split[0][1]
    head_traj = split[1][1]
    assert arm_traj.joint_names == ["arm_lift_joint"]
    assert arm_traj.points[0].positions == pytest.approx([0.1])
    assert arm_traj.points[0].velocities == pytest.approx([0.01])
    assert arm_traj.points[0].time_from_start.sec == 2
    assert arm_traj.points[0].time_from_start.nanosec == 100
    assert head_traj.joint_names == ["head_pan_joint"]
    assert head_traj.points[0].positions == pytest.approx([-0.2])


def test_split_trajectory_by_controller_sends_only_present_controller_joints():
    """Generic ros2_control splitting should not apply robot-specific joint filling."""
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

    client = Ros2ControlTrajectoryClient(
        {
            "ros2_control": {
                "controllers": [
                    {
                        "name": "arm",
                        "action": "/arm_trajectory_controller/follow_joint_trajectory",
                        "joints": ["arm_lift_joint", "arm_flex_joint"],
                    },
                ]
            }
        }
    )
    trajectory = JointTrajectory()
    trajectory.joint_names = ["arm_lift_joint"]
    point = JointTrajectoryPoint()
    point.positions = [0.4]
    trajectory.points = [point]
    split = client._split_trajectory_by_controller(trajectory)

    arm_traj = split[0][1]
    assert arm_traj.joint_names == ["arm_lift_joint"]
    assert arm_traj.points[0].positions == pytest.approx([0.4])


def test_send_prepared_trajectory_goals_preserves_batch_order(monkeypatch):
    """Prepared trajectory batches should be sent in caller-supplied stages."""
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

    monkeypatch.setitem(
        sys.modules,
        "rclpy",
        types.SimpleNamespace(spin_once=lambda node, timeout_sec=0.0: None),
    )
    client = Ros2ControlTrajectoryClient({})
    sent = []
    client._get_node = lambda: object()

    def fake_send(goal, trajectory):
        sent.append(goal.name)
        goal.goal_handle_future = _DoneFuture(_FakeGoalHandle())
        client._active_goals.append(goal)

    monkeypatch.setattr(client, "_send_controller_trajectory_goal", fake_send)
    arm_traj = JointTrajectory()
    arm_traj.joint_names = ["joint_a"]
    arm_traj.points = [JointTrajectoryPoint()]
    base_traj = JointTrajectory()
    base_traj.joint_names = ["joint_b"]
    base_traj.points = [JointTrajectoryPoint()]

    client.send_prepared_trajectory_goals([
        [({"name": "arm", "action": "/arm"}, arm_traj)],
        [({"name": "base", "action": "/base"}, base_traj)],
    ])

    assert sent == ["arm", "base"]
    assert client._status == "in_progress"


def test_goal_acceptance_timeout_retries_with_recreated_client(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "rclpy",
        types.SimpleNamespace(spin_once=lambda node, timeout_sec=0.0: None),
    )
    client = Ros2ControlTrajectoryClient({"action_server_timeout_s": 0.01, "goal_response_retries": 1})
    client._get_node = lambda: object()
    goal = _ControllerGoal(name="arm", action="/arm", joints=["joint_a"], trajectory=object())
    attempts = []

    def fake_send(retry_goal, trajectory):
        attempts.append(trajectory)
        retry_goal.acceptance_attempts += 1
        retry_goal.goal_handle_future = _DoneFuture(_FakeGoalHandle())

    monkeypatch.setattr(client, "_send_controller_trajectory_goal", fake_send)
    monkeypatch.setattr(client, "_goal_may_be_executing", lambda retry_goal: False)

    goal.goal_handle_future = _PendingFuture()
    goal.acceptance_attempts = 1

    assert client._wait_for_goal_acceptance([goal])
    assert len(attempts) == 1
    assert goal.accepted is True


def test_goal_acceptance_timeout_tracks_progress_without_resend(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "rclpy",
        types.SimpleNamespace(spin_once=lambda node, timeout_sec=0.0: None),
    )
    joint_state = type("JointState", (), {"name": ["joint_a"], "position": [0.5]})()
    client = Ros2ControlTrajectoryClient(
        {"action_server_timeout_s": 0.01, "goal_response_retries": 1},
        joint_state_provider=lambda: joint_state,
    )
    client._get_node = lambda: object()
    client._final_targets = {"joint_a": 0.0}
    goal = _ControllerGoal(
        name="arm",
        action="/arm",
        joints=["joint_a"],
        goal_handle_future=_PendingFuture(),
        trajectory=object(),
        acceptance_attempts=1,
        acceptance_error_norm=1.0,
    )
    retry_calls = []
    monkeypatch.setattr(client, "_retry_controller_goal", lambda retry_goal: retry_calls.append(retry_goal))

    assert client._wait_for_goal_acceptance([goal])
    assert goal.tracking_only is True
    assert goal.accepted is True
    assert retry_calls == []


class _DoneFuture:
    def __init__(self, result):
        self._result = result

    def done(self):
        return True

    def result(self):
        return self._result


class _PendingFuture:
    def done(self):
        return False

    def result(self):
        raise AssertionError("pending future has no result")


class _FakeGoalHandle:
    accepted = True

    def __init__(self):
        self.cancelled = False
        self.result_future = _DoneFuture(types.SimpleNamespace(status=4, result=types.SimpleNamespace(error_code=0)))

    def get_result_async(self):
        return self.result_future

    def cancel_goal_async(self):
        self.cancelled = True


def test_failed_controller_result_cancels_active_peer_goals(monkeypatch):
    """A whole-body trajectory failure should cancel sibling controllers promptly."""
    monkeypatch.setitem(
        sys.modules,
        "rclpy",
        types.SimpleNamespace(spin_once=lambda node, timeout_sec=0.0: None),
    )
    client = Ros2ControlTrajectoryClient({})
    client._get_node = lambda: object()
    arm_handle = _FakeGoalHandle()
    base_handle = _FakeGoalHandle()
    arm_goal = _ControllerGoal(
        name="arm",
        action="/arm_trajectory_controller/follow_joint_trajectory",
        joints=["arm_lift_joint"],
        goal_handle_future=_DoneFuture(arm_handle),
        accepted=True,
    )
    base_goal = _ControllerGoal(
        name="omni_base",
        action="/omni_base_controller/follow_joint_trajectory",
        joints=["odom_x"],
        goal_handle_future=_DoneFuture(base_handle),
        accepted=True,
    )
    client._active_goals = [arm_goal, base_goal]

    client._mark_failed_result(
        arm_goal,
        types.SimpleNamespace(
            result=types.SimpleNamespace(error_code=-1, error_string="invalid trajectory")
        ),
        status=4,
    )

    assert client.get_status() == "failed"
    assert "arm" in (client.get_error() or "")
    assert "INVALID_GOAL(-1)" in (client.get_error() or "")
    assert arm_handle.cancelled
    assert base_handle.cancelled


def test_failed_controller_result_omits_trajectory_diagnostics(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "rclpy",
        types.SimpleNamespace(spin_once=lambda node, timeout_sec=0.0: None),
    )
    client = Ros2ControlTrajectoryClient({})
    client._get_node = lambda: object()
    goal = _ControllerGoal(
        name="omni_base",
        action="/omni_base_controller/follow_joint_trajectory",
        joints=["odom_x", "odom_y", "odom_t"],
        goal_handle_future=_DoneFuture(_FakeGoalHandle()),
        accepted=True,
        trajectory_diagnostics={"sanitized": {"max_abs_wz_radps": 0.3}, "raw": {"max_abs_wz_radps": 9.0}},
    )
    client._active_goals = [goal]

    client._mark_failed_result(
        goal,
        types.SimpleNamespace(
            result=types.SimpleNamespace(error_code=-4, error_string="path tolerance")
        ),
        status=6,
    )

    assert client.get_status() == "failed"
    assert "PATH_TOLERANCE_VIOLATED(-4)" in (client.get_error() or "")
    assert "trajectory_diagnostics" not in (client.get_error() or "")
    assert "max_abs_wz_radps" not in (client.get_error() or "")


def test_follow_joint_trajectory_error_code_names_are_decoded():
    """Generic trajectory errors should be human-readable."""
    assert Ros2ControlTrajectoryClient._trajectory_error_name(-2) == "INVALID_JOINTS"
    assert Ros2ControlTrajectoryClient._trajectory_error_name(-5) == "GOAL_TOLERANCE_VIOLATED"


def test_accepted_pending_goal_does_not_use_joint_tracking(monkeypatch):
    """Accepted goals with pending result futures should wait for action result."""
    monkeypatch.setitem(
        sys.modules,
        "rclpy",
        types.SimpleNamespace(spin_once=lambda node, timeout_sec=0.0: None),
    )
    joint_state = type(
        "JointState",
        (),
        {"name": ["joint_a"], "position": [1.01]},
    )()
    provider_calls = []
    client = Ros2ControlTrajectoryClient(
        {"goal_tolerance": 0.03},
        joint_state_provider=lambda: provider_calls.append(True) or joint_state,
    )
    client._get_node = lambda: object()
    handle = _FakeGoalHandle()
    handle.result_future = _PendingFuture()
    client._active_goals = [
        _ControllerGoal(
            name="arm",
            action="/arm",
            joints=["joint_a"],
            goal_handle_future=_DoneFuture(handle),
        )
    ]
    client._final_targets = {"joint_a": 1.0}
    client._status = "in_progress"

    assert client.get_status() == "in_progress"
    assert client.get_error() is None
    assert not handle.cancelled
    assert provider_calls == []


def test_tracking_only_goal_succeeds_when_final_targets_are_within_tolerance(monkeypatch):
    """Tolerance completion remains available for lost-response fallback goals."""
    monkeypatch.setitem(
        sys.modules,
        "rclpy",
        types.SimpleNamespace(spin_once=lambda node, timeout_sec=0.0: None),
    )
    joint_state = type(
        "JointState",
        (),
        {"name": ["joint_a"], "position": [1.01]},
    )()
    client = Ros2ControlTrajectoryClient(
        {"goal_tolerance": 0.03},
        joint_state_provider=lambda: joint_state,
    )
    client._get_node = lambda: object()
    client._active_goals = [
        _ControllerGoal(
            name="arm",
            action="/arm",
            joints=["joint_a"],
            tracking_only=True,
            accepted=True,
        )
    ]
    client._final_targets = {"joint_a": 1.0}
    client._status = "in_progress"

    assert client.get_status() == "succeeded"
    assert client.get_error() is None


def test_in_progress_goal_stays_running_when_final_targets_are_outside_tolerance(monkeypatch):
    """Accepted pending action result should not use tolerance tracking."""
    monkeypatch.setitem(
        sys.modules,
        "rclpy",
        types.SimpleNamespace(spin_once=lambda node, timeout_sec=0.0: None),
    )
    joint_state = type(
        "JointState",
        (),
        {"name": ["joint_a"], "position": [1.2]},
    )()
    client = Ros2ControlTrajectoryClient(
        {"goal_tolerance": 0.03},
        joint_state_provider=lambda: joint_state,
    )
    client._get_node = lambda: object()
    handle = _FakeGoalHandle()
    handle.result_future = _PendingFuture()
    client._active_goals = [
        _ControllerGoal(
            name="arm",
            action="/arm",
            joints=["joint_a"],
            goal_handle_future=_DoneFuture(handle),
        )
    ]
    client._final_targets = {"joint_a": 1.0}
    client._status = "in_progress"

    assert client.get_status() == "in_progress"
    assert not handle.cancelled


def test_per_joint_tolerance_override_is_honored(monkeypatch):
    """Per-joint tolerance should override the generic tolerance in tracking fallback."""
    monkeypatch.setitem(
        sys.modules,
        "rclpy",
        types.SimpleNamespace(spin_once=lambda node, timeout_sec=0.0: None),
    )
    joint_state = type(
        "JointState",
        (),
        {"name": ["joint_a"], "position": [1.08]},
    )()
    client = Ros2ControlTrajectoryClient(
        {"goal_tolerance": 0.03, "goal_tolerance_by_joint": {"joint_a": 0.1}},
        joint_state_provider=lambda: joint_state,
    )
    client._get_node = lambda: object()
    handle = _FakeGoalHandle()
    handle.result_future = _PendingFuture()
    client._active_goals = [
        _ControllerGoal(
            name="arm",
            action="/arm",
            joints=["joint_a"],
            goal_handle_future=_DoneFuture(handle),
            tracking_only=True,
            accepted=True,
        )
    ]
    client._final_targets = {"joint_a": 1.0}
    client._status = "in_progress"

    assert client.get_status() == "succeeded"


def test_in_progress_goal_fails_when_joint_error_stops_improving(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "rclpy",
        types.SimpleNamespace(spin_once=lambda node, timeout_sec=0.0: None),
    )
    joint_state = type(
        "JointState",
        (),
        {"name": ["joint_a"], "position": [0.0]},
    )()
    client = Ros2ControlTrajectoryClient(
        {"goal_tolerance": 0.03, "progress_timeout_s": 0.01},
        joint_state_provider=lambda: joint_state,
    )
    client._get_node = lambda: object()
    handle = _FakeGoalHandle()
    handle.result_future = _PendingFuture()
    client._active_goals = [
        _ControllerGoal(
            name="arm",
            action="/arm",
            joints=["joint_a"],
            goal_handle_future=_DoneFuture(handle),
            tracking_only=True,
            accepted=True,
        )
    ]
    client._final_targets = {"joint_a": 1.0}
    client._status = "in_progress"
    client._reset_progress_tracking()
    client._last_progress_time -= 1.0

    assert client.get_status() == "failed"
    assert "ros2_control_stalled_no_joint_progress" in (client.get_error() or "")
    assert handle.cancelled
