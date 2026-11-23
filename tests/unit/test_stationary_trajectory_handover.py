"""Stop reference and safe staging retry regressions; no ROS goals are sent."""
import numpy as np
import py_trees
import pytest
from rclpy.time import Time

from src.core.action import ActionResult
from src.core.failure import FailureCode, FailureInfo
from src.plugin.actions.hsr_pickup import HsrPickup
from src.plugin.actions.trajectory import ExecuteJointTrajectory
from src.plugin.controllers.ros2_control_trajectory import Ros2ControlController
from tests.unit.test_hsr_pickup import sync_client, trajectory_message
from tests.unit.test_workflow_catalog import make_context


def test_explicit_hold_is_stationary_and_preserves_motion_positions_and_spacing():
    source = trajectory_message({'joint_names': ['a', 'b'], 'points': [
        {'positions': [.127, -.201], 'velocities': [0., 0.], 'accelerations': [-.04, -.89], 'time_from_start': 0.},
        {'positions': [.2, -.6], 'velocities': [.1, -.2], 'accelerations': [.02, -.03], 'time_from_start': 1.},
        {'positions': [.35, -1.2], 'velocities': [0., 0.], 'accelerations': [-.28, .86], 'time_from_start': 2.}]})
    prepared = Ros2ControlController._stationary_prefix(source, 5., Time(seconds=10))
    assert prepared.header.stamp.sec == 10
    assert [p.time_from_start.sec for p in prepared.points] == [0, 5, 6, 7]
    assert prepared.points[0].positions == prepared.points[1].positions == source.points[0].positions
    for point in (prepared.points[0], prepared.points[1], prepared.points[-1]):
        assert list(point.velocities) == list(point.accelerations) == [0., 0.]
    assert prepared.points[2].velocities == source.points[1].velocities
    assert prepared.points[2].accelerations == source.points[1].accelerations
    assert [p.positions for p in prepared.points[1:]] == [p.positions for p in source.points]
    assert list(source.points[0].accelerations) == [-.04, -.89]
    # A quintic between identical positions and zero endpoint derivatives is constant.
    for duration in (2.8, 5., 10.):
        for index in range(2):
            a, b = prepared.points[:2]
            c = np.zeros(6); c[0] = a.positions[index]
            target = [b.positions[index] - c[0], 0., 0.]
            c[3:] = np.linalg.solve([[1, 1, 1], [3, 4, 5], [6, 12, 20]], target)
            np.testing.assert_allclose(np.polynomial.polynomial.polyval(np.linspace(0, 1, 50), c), a.positions[index])


def test_zero_stop_derivatives_are_transmitted_to_each_controller():
    client, clients, _ = sync_client()
    trajectory = trajectory_message({'joint_names': ['a', 'b'], 'points': [
        {'positions': [0., 0.], 'time_from_start': 0.},
        {'positions': [1., 1.], 'velocities': [0., 0.], 'accelerations': [-.04, -.89], 'time_from_start': 2.}]})
    client.send_synchronized_trajectory_goal(trajectory)
    for target in clients.values():
        sent = target.send_goal_async.call_args.args[0].trajectory
        assert sent.header.stamp.sec == 10
        assert list(sent.points[0].positions) == list(sent.points[1].positions) == [0.]
        for point in (sent.points[0], sent.points[1], sent.points[-1]):
            assert list(point.velocities) == list(point.accelerations) == [0.]
    assert list(trajectory.points[-1].accelerations) == [-.04, -.89]


def test_controller_unavailable_is_retryable_only_before_any_send():
    client, clients, _ = sync_client()
    clients['/base'].wait_for_server.return_value = False
    trajectory = trajectory_message({'joint_names': ['a', 'b'], 'points': [
        {'positions': [0., 0.], 'time_from_start': 0.}]})
    client.send_synchronized_trajectory_goal(trajectory)
    assert client.get_status() == 'failed'
    assert client.get_failure().retryable
    assert client.get_failure().backend_code == 'controller_unavailable'
    assert not any(c.send_goal_async.called for c in clients.values())


@pytest.mark.parametrize('retryable', [True, False])
def test_staging_is_a_direct_leaf_without_action_retry(monkeypatch, retryable):
    context = make_context(); attempts = []
    def start(action, ctx):
        attempts.append(action.trajectory_key)
        action._test_result = ActionResult(status='failure',
            failure=FailureInfo(FailureCode.TIMEOUT, retryable=retryable))
    monkeypatch.setattr(ExecuteJointTrajectory, 'start', start)
    monkeypatch.setattr(ExecuteJointTrajectory, 'check_result', lambda action, ctx: action._test_result)
    root = HsrPickup('jar').build_node(context, 'Pickup')
    leaf = next(node for node in root.iterate() if node.name == 'Stage arm and base')
    assert type(leaf.action) is ExecuteJointTrajectory
    assert not any(node.name.startswith('Retry staging dispatch:') for node in root.iterate())
    leaf.tick_once()
    assert len(attempts) == 1
    assert leaf.status == py_trees.common.Status.FAILURE
    assert not any(isinstance(node, py_trees.decorators.Retry) and 'Insert' in node.name for node in root.iterate())
