"""Tests for MoveIt-backed head and gripper controller contracts."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import src.plugin.controllers.moveit as moveit_module


class _FakeMoveGroupClient:
    instances: list["_FakeMoveGroupClient"] = []

    def __init__(self, hardware_config: dict, node_name: str):
        self.hardware_config = hardware_config
        self.node_name = node_name
        self.status = "idle"
        self.error = None
        self.joint_goals = []
        self.named_goals = []
        _FakeMoveGroupClient.instances.append(self)

    def send_joint_goal(self, joint_names, joint_positions):
        self.joint_goals.append((list(joint_names), [float(v) for v in joint_positions]))
        self.status = "in_progress"

    def send_named_goal(self, config_name: str):
        self.named_goals.append(config_name)
        self.status = "in_progress"

    def get_status(self):
        return self.status

    def get_error(self):
        return self.error

    def cancel_goal(self):
        self.status = "idle"


@pytest.fixture(autouse=True)
def fake_move_group_client(monkeypatch):
    _FakeMoveGroupClient.instances = []
    monkeypatch.setattr(moveit_module, "_MoveItMoveGroupClient", _FakeMoveGroupClient)


def test_moveit_head_controller_uses_head_group_and_semantic_joints():
    controller = moveit_module.MoveItHeadController({})

    controller.send_pan_tilt(0.4, -0.6)

    fake = _FakeMoveGroupClient.instances[-1]
    assert fake.hardware_config["move_group"] == "head"
    assert fake.joint_goals == [(["head_pan_joint", "head_tilt_joint"], [0.4, -0.6])]


def test_moveit_head_named_target_accepts_pan_tilt_aliases():
    controller = moveit_module.MoveItHeadController(
        {"named_targets": {"look_down": {"pan": 0.1, "tilt": -0.8}}}
    )

    controller.send_named_goal("look_down")

    fake = _FakeMoveGroupClient.instances[-1]
    assert fake.joint_goals == [(["head_pan_joint", "head_tilt_joint"], [0.1, -0.8])]


def test_moveit_gripper_maps_open_width_to_hand_motor_joint():
    controller = moveit_module.MoveItGripperController(
        {
            "min_width": 0.0,
            "max_width": 0.08,
            "min_position": 0.0,
            "max_position": 1.2,
            "open_distance": 0.06,
        }
    )

    result = controller.open(0.04)

    fake = _FakeMoveGroupClient.instances[-1]
    assert result.success
    assert fake.hardware_config["move_group"] == "gripper"
    assert fake.joint_goals == [(["hand_motor_joint"], [pytest.approx(0.6)])]
    assert controller.get_distance() == pytest.approx(0.04)


def test_moveit_gripper_close_uses_requested_width_keyword():
    controller = moveit_module.MoveItGripperController(
        {
            "joint_name": "hand_motor_joint",
            "min_width": 0.0,
            "max_width": 0.10,
            "min_position": -0.2,
            "max_position": 0.8,
            "closed_distance": 0.01,
        }
    )

    result = controller.close(force_n=0.5, width_m=0.03)

    fake = _FakeMoveGroupClient.instances[-1]
    assert result.success
    assert fake.joint_goals == [(["hand_motor_joint"], [pytest.approx(0.1)])]
    assert controller.get_distance() == pytest.approx(0.03)


def test_moveit_gripper_get_distance_reads_joint_state_provider():
    joint_state = SimpleNamespace(name=["hand_motor_joint"], position=[0.5])
    controller = moveit_module.MoveItGripperController(
        {
            "min_width": 0.0,
            "max_width": 0.08,
            "min_position": 0.0,
            "max_position": 1.0,
            "joint_state_provider": lambda: joint_state,
        }
    )

    assert controller.get_distance() == pytest.approx(0.04)


@pytest.mark.parametrize('command,measured,expected',[
    ('close',0.,'succeeded'),('close',.05,'succeeded'),('close',.0501,'idle'),
    ('open',.12,'succeeded'),('open',.15,'succeeded'),('open',.18,'succeeded'),
    ('open',.119,'idle'),('open',.181,'idle'),('open',.04,'idle'),
])
def test_gripper_idle_uses_measured_opening_and_command(command,measured,expected):
    state=SimpleNamespace(name=['hand_motor_joint'],position=[measured])
    controller=moveit_module.MoveItGripperController({
        'min_width':0.,'max_width':.20,'min_position':0.,'max_position':.20,
        'open_distance':.15,'joint_state_provider':lambda:state})
    result=controller.open() if command=='open' else controller.close()
    assert result.success
    fake=_FakeMoveGroupClient.instances[-1];fake.status='idle'
    assert controller.get_status()==expected
    if expected=='succeeded':assert controller.get_state()==('open' if command=='open' else 'closed')


@pytest.mark.parametrize('provider',[
    None,lambda:None,lambda:SimpleNamespace(name=[],position=[]),
    lambda:SimpleNamespace(name=['hand_motor_joint'],position=[float('nan')]),
    lambda:SimpleNamespace(name=['hand_motor_joint'],position=[float('inf')]),
    lambda:SimpleNamespace(name=['hand_motor_joint'],position=['invalid']),
])
def test_gripper_idle_does_not_accept_requested_width_as_feedback(provider):
    controller=moveit_module.MoveItGripperController({'joint_state_provider':provider})
    controller.open(.08)
    _FakeMoveGroupClient.instances[-1].status='idle'
    assert controller.get_distance()==.08  # Existing reporting fallback remains.
    assert controller.get_status()=='idle'  # It cannot confirm execution.


@pytest.mark.parametrize('status',['in_progress','failed','succeeded'])
def test_gripper_measured_width_does_not_override_nonidle_status(status):
    state=SimpleNamespace(name=['hand_motor_joint'],position=[0.])
    controller=moveit_module.MoveItGripperController({'joint_state_provider':lambda:state})
    controller.close();_FakeMoveGroupClient.instances[-1].status=status
    assert controller.get_status()==status


def test_gripper_idle_before_command_and_after_cancel_is_not_success():
    state=SimpleNamespace(name=['hand_motor_joint'],position=[0.])
    controller=moveit_module.MoveItGripperController({'joint_state_provider':lambda:state})
    assert controller.get_status()=='idle'
    controller.close();controller.cancel_goal()
    assert controller.get_status()=='idle'
    controller.close();_FakeMoveGroupClient.instances[-1].status='idle'
    assert controller.get_status()=='succeeded'


def test_gripper_idle_open_uses_requested_width_and_resets_previous_close():
    state=SimpleNamespace(name=['hand_motor_joint'],position=[.04])
    controller=moveit_module.MoveItGripperController({
        'min_width':0.,'max_width':.20,'min_position':0.,'max_position':.20,
        'open_distance':.15,'joint_state_provider':lambda:state})
    controller.close();_FakeMoveGroupClient.instances[-1].status='idle'
    assert controller.get_status()=='succeeded'
    controller.open(.10);_FakeMoveGroupClient.instances[-1].status='idle'
    assert controller.get_status()=='idle'
    state.position=[.10]
    assert controller.get_status()=='succeeded'
    assert controller.get_state()=='open'
