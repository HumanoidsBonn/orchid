"""HSR hardware adapter checks with a mocked hsrb_interface; no ROS or motion."""
import sys
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest
from src.plugin.controllers.hsrb_gripper import HsrbGripperController


@pytest.fixture
def gripper(monkeypatch):
    hardware = Mock()
    hardware.is_moving.return_value = False
    hardware.is_succeeded.return_value = False
    robot = Mock()
    robot.get.return_value = hardware
    monkeypatch.setitem(sys.modules, 'hsrb_interface', NS(Robot=lambda: robot))
    return HsrbGripperController({'open_distance': .10}), hardware


@pytest.mark.parametrize('command,distance,expected', [
    ('open', .12, 'succeeded'), ('open', .15, 'succeeded'), ('open', .18, 'succeeded'),
    ('open', .119, 'idle'), ('open', .181, 'idle'), ('open', .04, 'idle'),
    ('closed', 0., 'succeeded'), ('closed', .05, 'succeeded'), ('closed', .051, 'idle'),
    ('closed', float('nan'), 'idle'), ('closed', float('inf'), 'idle'), ('closed', -.01, 'idle'),
])
def test_idle_uses_last_command_and_measured_distance(gripper, command, distance, expected):
    controller, hardware = gripper
    hardware.get_distance.return_value = distance
    result = controller.open(.15) if command == 'open' else controller.close(.5)
    assert result.success
    assert controller.get_status() == expected
    if expected == 'succeeded':
        assert controller.get_state() == command
    if command == 'open':
        hardware.set_distance.assert_called_once_with(.15, sync=False)


def test_default_open_width_and_cancel_and_missing_measurement(gripper):
    controller, hardware = gripper
    hardware.get_distance.return_value = .10
    assert controller.get_status() == 'idle'
    controller.open()
    hardware.set_distance.assert_called_once_with(.10, sync=False)
    assert controller.get_status() == 'succeeded'
    controller.cancel_goal()
    assert controller.get_status() == 'idle'
    controller.close(.5)
    hardware.get_distance.side_effect = RuntimeError('no joint feedback')
    assert controller.get_status() == 'idle'


def test_moving_and_reported_success_keep_precedence(gripper):
    controller, hardware = gripper
    controller.open(.15)
    hardware.get_distance.return_value = .15
    hardware.is_moving.return_value = True
    assert controller.get_status() == 'in_progress'
    hardware.is_moving.return_value = False
    hardware.is_succeeded.return_value = True
    assert controller.get_status() == 'succeeded'
    hardware.get_distance.assert_not_called()


def test_failed_new_dispatch_cannot_reuse_previous_command(gripper):
    controller, hardware = gripper
    controller.close(.5)
    hardware.get_distance.return_value = .04
    assert controller.get_status() == 'succeeded'
    hardware.set_distance.side_effect = RuntimeError('dispatch failed')
    assert not controller.open(.15).success
    assert controller.get_status() == 'idle'
