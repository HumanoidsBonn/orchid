"""Tests for HSRB arm controller target delegation."""

from __future__ import annotations
from src.core.arm_request import ArmPoseRequest, MoveItPoseSettings, CartesianLine
from src.core.manipulation_types import BasePose2D

import sys
import types
import math

import pytest

from src.plugin.controllers.hsrb_arm import HsrbArmController
from src.plugin.controllers.hsrb_head import HsrbHeadController


class FakeJointGroup:
    def __init__(self):
        self.joint_calls = []
        self.executed = []
        self.status_error = None
        self.cancel_error = None
        self.execute_error = None
        self.pose_error = None
        self._current_clients = ["stale-client"]
        self.pose_calls = []
        self.line_calls = []
        self.generated_requests = []
        self.looking_hand_constraint = False
        self.looking_hand_constraint_during_pose_calls = []
        self.joint_state = type(
            "JointState",
            (),
            {
                "name": ["arm_lift_joint", "wrist_flex_joint"],
                "position": [0.0, -1.0],
            },
        )()

    def move_to_joint_positions(self, joints, plan_only=False):
        self.joint_calls.append((joints, plan_only))
        return {"trajectory": joints}

    def execute(self, trajectory):
        if self.execute_error is not None:
            error = self.execute_error
            self.execute_error = None
            raise error
        self.executed.append(trajectory)

    def move_end_effector_pose(self, pose, ref_frame_id=None, plan_only=False):
        self.pose_calls.append((pose, ref_frame_id, plan_only))
        self.looking_hand_constraint_during_pose_calls.append(self.looking_hand_constraint)
        if self.pose_error is not None:
            raise self.pose_error
        self._generate_planning_request(_FakeHandLineRequest)
        return FakeTrajectory(["arm_lift_joint", "wrist_flex_joint"])

    def move_end_effector_by_line(self, axis, distance, plan_only=False):
        self.line_calls.append((axis, distance, plan_only))
        self._generate_planning_request(_FakeHandLineRequest)
        return FakeTrajectory(["arm_lift_joint", "wrist_flex_joint"])

    def _generate_planning_request(self, request_type):
        request = request_type()
        request.base_movement_type.val = _FakeBaseMovementType.PLANAR
        self.generated_requests.append(request)
        return request

    def _get_joint_state(self):
        return self.joint_state

    def is_moving(self):
        if self.status_error is not None:
            raise self.status_error
        return False

    def is_succeeded(self):
        return bool(self.executed)

    def cancel_goal(self):
        if self.cancel_error is not None:
            raise self.cancel_error


def _controller_with_fake_joint_group() -> tuple[HsrbArmController, FakeJointGroup]:
    controller = HsrbArmController.__new__(HsrbArmController)
    controller.hardware_config = {
        "named_targets": {
            "go": {
                "type": "joint_positions",
                "joints": {"arm_flex_joint": 0.0},
            }
        }
    }
    controller._available = True
    controller._hsrb_available = True
    controller._execution_backend = "hsrb_interface"
    controller._ros2_control = None
    controller._error = None
    controller._jg = FakeJointGroup()
    controller.status = "idle"
    return controller, controller._jg


class FakeTrajectory:
    def __init__(self, joint_names):
        self.joint_names = joint_names
        self.points = []


class _FakeBaseMovementType:
    NONE = 0
    PLANAR = 1

    def __init__(self):
        self.val = self.PLANAR


class _FakeHandLineRequest:
    def __init__(self):
        self.base_movement_type = _FakeBaseMovementType()


class FakeRos2Control:
    def __init__(self):
        self.joint_goals = []
        self.trajectory_goals = []
        self.prepared_batches = []
        self.status = "succeeded"
        self.error = None

    def send_joint_goal(self, joints):
        self.joint_goals.append(joints)
        self.status = "in_progress"

    def send_trajectory_goal(self, trajectory, current_joint_state=None):
        self.trajectory_goals.append((trajectory, current_joint_state))
        self.status = "in_progress"

    def send_prepared_trajectory_goals(self, batches):
        self.prepared_batches.append(batches)
        self.status = "in_progress"

    def get_status(self):
        return self.status

    def get_error(self):
        return self.error


def _controller_with_ros2_control() -> tuple[HsrbArmController, FakeJointGroup, FakeRos2Control]:
    controller, joint_group = _controller_with_fake_joint_group()
    controller._execution_backend = "ros2_control"
    controller._planning_backend = "hsrb_interface_plan_only"
    controller._ros2_control = FakeRos2Control()
    return controller, joint_group, controller._ros2_control


def test_send_joint_goal_plans_then_executes():
    controller, joint_group = _controller_with_fake_joint_group()
    controller.send_joint_goal({"arm_flex_joint": 0.0})
    assert joint_group.joint_calls == [({"arm_flex_joint": 0.0}, True)]
    assert joint_group.executed == [{"trajectory": {"arm_flex_joint": 0.0}}]
    assert controller.status == "in_progress"


def test_send_named_goal_resolves_manifest_joint_target():
    controller, joint_group = _controller_with_fake_joint_group()
    controller.send_named_goal("go")
    assert joint_group.joint_calls == [({"arm_flex_joint": 0.0}, True)]
    assert joint_group.executed == [{"trajectory": {"arm_flex_joint": 0.0}}]


def test_send_trajectory_goal_executes_directly():
    controller, joint_group = _controller_with_fake_joint_group()
    controller.send_trajectory_goal("trajectory")
    assert joint_group.executed == ["trajectory"]
    assert controller.status == "in_progress"


def test_pending_hsrb_goal_handle_polls_as_in_progress():
    controller, joint_group = _controller_with_fake_joint_group()
    controller.status = "in_progress"
    joint_group.status_error = AttributeError(
        "'NoneType' object has no attribute 'get_result_async'"
    )

    assert controller.get_status() == "in_progress"


def test_pending_hsrb_goal_handle_cancel_is_idempotent():
    controller, joint_group = _controller_with_fake_joint_group()
    joint_group.cancel_error = AttributeError(
        "'NoneType' object has no attribute 'get_result_async'"
    )

    controller.cancel_goal()


def test_execute_retries_after_internal_cancel_pending_goal_handle():
    controller, joint_group = _controller_with_fake_joint_group()
    joint_group.execute_error = AttributeError(
        "'NoneType' object has no attribute 'get_result_async'"
    )

    controller.send_joint_goal({"arm_flex_joint": 0.0})

    assert joint_group._current_clients is None
    assert joint_group.executed == [{"trajectory": {"arm_flex_joint": 0.0}}]
    assert controller.status == "in_progress"


def test_ros2_control_named_target_uses_direct_joint_goal():
    controller, joint_group, ros2_control = _controller_with_ros2_control()

    controller.send_named_goal("go")

    assert ros2_control.joint_goals == [{"arm_flex_joint": 0.0}]
    assert joint_group.joint_calls == []
    assert joint_group.executed == []
    assert controller.status == "in_progress"


def test_ros2_control_pose_goal_uses_hsrb_plan_only_then_ros2_execution(monkeypatch):
    from src.core.controllers import Pose3D
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

    fake_hsrb = types.SimpleNamespace(
        geometry=types.SimpleNamespace(
            pose=lambda x, y, z, roll, pitch, yaw: (x, y, z, roll, pitch, yaw)
        )
    )
    monkeypatch.setitem(sys.modules, "hsrb_interface", fake_hsrb)
    controller, joint_group, ros2_control = _controller_with_ros2_control()
    controller.hardware_config["ros2_control"] = {
        "controllers": [
            {
                "name": "arm",
                "action": "/arm",
                "joints": ["arm_lift_joint", "wrist_flex_joint"],
            }
        ]
    }
    trajectory = JointTrajectory()
    trajectory.joint_names = ["arm_lift_joint", "wrist_flex_joint"]
    point = JointTrajectoryPoint()
    point.positions = [0.2, -0.3]
    trajectory.points = [point]
    joint_group.move_end_effector_pose = lambda pose, ref_frame_id=None, plan_only=False: trajectory

    controller.send_pose_goal(ArmPoseRequest(Pose3D(x=0.1, y=0.2, z=0.3, frame_id='odom')))

    assert len(ros2_control.prepared_batches) == 1
    sent_trajectory = ros2_control.prepared_batches[0][0][0][1]
    assert sent_trajectory.joint_names == ["arm_lift_joint", "wrist_flex_joint"]
    assert sent_trajectory.points[0].positions == pytest.approx([0.2, -0.3])
    assert joint_group.executed == []
    assert controller.status == "in_progress"


def test_hsrb_trajectory_extraction_fills_missing_controller_joints_from_state():
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

    controller, joint_group, _ros2_control = _controller_with_ros2_control()
    controller.hardware_config["ros2_control"] = {
        "controllers": [
            {
                "name": "arm",
                "action": "/arm",
                "joints": ["arm_lift_joint", "wrist_flex_joint"],
            }
        ]
    }
    trajectory = JointTrajectory()
    trajectory.joint_names = ["arm_lift_joint"]
    point = JointTrajectoryPoint()
    point.positions = [0.4]
    point.velocities = [0.1]
    trajectory.points = [point]

    batches = controller._prepare_hsrb_ros2_control_trajectory_batches(trajectory)

    extracted = batches[0][0][1]
    assert extracted.joint_names == ["arm_lift_joint", "wrist_flex_joint"]
    assert extracted.points[0].positions == pytest.approx([0.4, -1.0])
    assert extracted.points[0].velocities == pytest.approx([0.1, 0.0])
    assert extracted.points[0].accelerations == pytest.approx([0.0, 0.0])
    assert len(extracted.points[0].effort) == 0


def test_hsrb_trajectory_extraction_batches_base_synchronously_by_default():
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

    controller, _joint_group, _ros2_control = _controller_with_ros2_control()
    controller.hardware_config["ros2_control"] = {
        "controllers": [
            {
                "name": "arm",
                "action": "/arm",
                "joints": ["arm_lift_joint"],
            },
            {
                "name": "omni_base",
                "action": "/base",
                "joints": ["odom_x", "odom_y", "odom_t"],
                "group_type": "base_trajectory",
            },
        ]
    }
    controller._jg.joint_state = type(
        "JointState",
        (),
        {
            "name": ["arm_lift_joint", "odom_x", "odom_y", "odom_t"],
            "position": [0.0, 0.0, 0.0, 0.0],
        },
    )()
    trajectory = JointTrajectory()
    trajectory.joint_names = ["odom_x", "odom_y", "odom_t", "arm_lift_joint"]
    point = JointTrajectoryPoint()
    point.positions = [1.0, 2.0, 0.3, 0.4]
    trajectory.points = [point]

    batches = controller._prepare_hsrb_ros2_control_trajectory_batches(trajectory)

    assert [[entry[0]["name"] for entry in batch] for batch in batches] == [["arm", "omni_base"]]
    assert batches[0][0][1].joint_names == ["arm_lift_joint"]
    assert batches[0][1][1].joint_names == ["odom_x", "odom_y", "odom_t"]


def test_hsrb_trajectory_extraction_batches_base_after_primary_when_staged():
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

    controller, _joint_group, _ros2_control = _controller_with_ros2_control()
    controller.hardware_config["ros2_control"] = {
        "dispatch_mode": "staged",
        "controllers": [
            {
                "name": "arm",
                "action": "/arm",
                "joints": ["arm_lift_joint"],
            },
            {
                "name": "omni_base",
                "action": "/base",
                "joints": ["odom_x", "odom_y", "odom_t"],
                "group_type": "base_trajectory",
            },
        ],
    }
    controller._jg.joint_state = type(
        "JointState",
        (),
        {
            "name": ["arm_lift_joint", "odom_x", "odom_y", "odom_t"],
            "position": [0.0, 0.0, 0.0, 0.0],
        },
    )()
    trajectory = JointTrajectory()
    trajectory.joint_names = ["odom_x", "odom_y", "odom_t", "arm_lift_joint"]
    point = JointTrajectoryPoint()
    point.positions = [1.0, 2.0, 0.3, 0.4]
    trajectory.points = [point]

    batches = controller._prepare_hsrb_ros2_control_trajectory_batches(trajectory)

    assert [[entry[0]["name"] for entry in batch] for batch in batches] == [["arm"], ["omni_base"]]


def test_hsrb_trajectory_extraction_includes_head_joints_in_primary_batch():
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

    controller, _joint_group, _ros2_control = _controller_with_ros2_control()
    controller.hardware_config["ros2_control"] = {
        "controllers": [
            {
                "name": "arm",
                "action": "/arm",
                "joints": ["arm_lift_joint"],
            },
            {
                "name": "head",
                "action": "/head",
                "joints": ["head_pan_joint", "head_tilt_joint"],
            },
        ]
    }
    controller._jg.joint_state = type(
        "JointState",
        (),
        {
            "name": ["arm_lift_joint", "head_pan_joint", "head_tilt_joint"],
            "position": [0.0, 0.0, 0.0],
        },
    )()
    trajectory = JointTrajectory()
    trajectory.joint_names = ["arm_lift_joint", "head_pan_joint", "head_tilt_joint"]
    point = JointTrajectoryPoint()
    point.positions = [0.4, 0.2, -0.5]
    trajectory.points = [point]

    batches = controller._prepare_hsrb_ros2_control_trajectory_batches(trajectory)

    assert [[entry[0]["name"] for entry in batch] for batch in batches] == [["arm", "head"]]
    assert batches[0][1][1].joint_names == ["head_pan_joint", "head_tilt_joint"]
    assert batches[0][1][1].points[0].positions == pytest.approx([0.2, -0.5])


def test_hsrb_base_trajectory_sanitizer_inserts_current_state_and_rescales_time():
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

    controller, _joint_group, _ros2_control = _controller_with_ros2_control()
    controller.hardware_config["ros2_control"] = {
        "time_scale": 1.0,
        "velocity_scale": 1.0,
        "acceleration_scale": 1.0,
        "controllers": [
            {
                "name": "omni_base",
                "action": "/base",
                "joints": ["odom_x", "odom_y", "odom_t"],
                "group_type": "base_trajectory",
                "time_scale": 2.0,
                "velocity_scale": 0.5,
                "acceleration_scale": 0.25,
            },
        ],
    }
    controller._jg.joint_state = type(
        "JointState",
        (),
        {
            "name": ["odom_x", "odom_y", "odom_t"],
            "position": [0.0, 0.0, 0.0],
        },
    )()
    trajectory = JointTrajectory()
    trajectory.joint_names = ["odom_x", "odom_y", "odom_t"]
    point = JointTrajectoryPoint()
    point.positions = [1.0, 2.0, 0.3]
    point.velocities = [0.4, 0.6, 0.8]
    point.accelerations = [0.2, 0.4, 0.6]
    point.time_from_start.sec = 1
    point.time_from_start.nanosec = 500_000_000
    trajectory.points = [point]

    batches = controller._prepare_hsrb_ros2_control_trajectory_batches(trajectory)

    base_trajectory = batches[0][0][1]
    assert len(base_trajectory.points) == 2
    assert base_trajectory.points[0].positions == pytest.approx([0.0, 0.0, 0.0])
    assert base_trajectory.points[0].time_from_start.sec == 0
    scaled = base_trajectory.points[1]
    assert scaled.positions == pytest.approx([1.0, 2.0, 0.3])
    assert scaled.time_from_start.sec == 10
    assert scaled.time_from_start.nanosec == 0
    assert scaled.velocities == pytest.approx([0.0, 0.0, 0.0])
    assert scaled.accelerations == pytest.approx([0.0, 0.0, 0.0])
    diagnostics = HsrbArmController._base_trajectory_diagnostics(
        base_trajectory,
        limits=(0.2, 0.2, 0.35),
    )
    assert diagnostics["point_count"] == 2
    assert diagnostics["limit_violations"] == []
    assert diagnostics["max_abs_vy_mps"] <= 0.2


def test_hsrb_base_trajectory_sanitizer_unwraps_yaw_near_current_state():
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

    controller, _joint_group, _ros2_control = _controller_with_ros2_control()
    controller.hardware_config["ros2_control"] = {
        "controllers": [
            {
                "name": "omni_base",
                "action": "/base",
                "joints": ["odom_x", "odom_y", "odom_t"],
                "group_type": "base_trajectory",
                "base_trajectory_max_wz_radps": 0.35,
            },
        ],
    }
    controller._jg.joint_state = type(
        "JointState",
        (),
        {
            "name": ["odom_x", "odom_y", "odom_t"],
            "position": [0.0, 0.0, 3.10],
        },
    )()
    trajectory = JointTrajectory()
    trajectory.joint_names = ["odom_x", "odom_y", "odom_t"]
    point = JointTrajectoryPoint()
    point.positions = [0.0, 0.0, -3.10]
    point.time_from_start.sec = 1
    trajectory.points = [point]

    batches = controller._prepare_hsrb_ros2_control_trajectory_batches(trajectory)

    yaw = batches[0][0][1].points[-1].positions[2]
    assert yaw == pytest.approx(3.10 + (2.0 * math.pi - 6.20))
    diagnostics = HsrbArmController._base_trajectory_diagnostics(
        batches[0][0][1],
        limits=(0.2, 0.2, 0.35),
    )
    assert diagnostics["max_abs_dyaw_rad"] < 0.1


def test_hsrb_base_trajectory_diagnostics_reports_velocity_limit_violations():
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

    trajectory = JointTrajectory()
    trajectory.joint_names = ["odom_x", "odom_y", "odom_t"]
    start = JointTrajectoryPoint()
    start.positions = [0.0, 0.0, 0.0]
    trajectory.points.append(start)
    end = JointTrajectoryPoint()
    end.positions = [1.0, 0.0, 0.5]
    end.time_from_start.nanosec = 10_000_000
    trajectory.points.append(end)

    diagnostics = HsrbArmController._base_trajectory_diagnostics(
        trajectory,
        limits=(0.2, 0.2, 0.35),
    )

    assert diagnostics["min_dt_s"] == pytest.approx(0.01)
    assert diagnostics["max_abs_vx_mps"] == pytest.approx(100.0)
    assert diagnostics["max_abs_wz_radps"] == pytest.approx(50.0)
    assert {violation["joint"] for violation in diagnostics["limit_violations"]} == {"odom_x", "odom_t"}


def test_hsrb_pose_goal_applies_looking_hand_constraint_only_for_planning_call(monkeypatch):
    from src.core.controllers import Pose3D

    fake_hsrb = types.SimpleNamespace(
        geometry=types.SimpleNamespace(
            pose=lambda x, y, z, roll, pitch, yaw: (x, y, z, roll, pitch, yaw)
        )
    )
    monkeypatch.setitem(sys.modules, "hsrb_interface", fake_hsrb)
    controller, joint_group, _ros2_control = _controller_with_ros2_control()
    joint_group.looking_hand_constraint = False

    controller.send_pose_goal(ArmPoseRequest(Pose3D(x=0.1, y=0.2, z=0.3, frame_id='odom'), looking_hand_constraint=True))

    assert joint_group.looking_hand_constraint_during_pose_calls == [True]
    assert joint_group.looking_hand_constraint is False


def test_hsrb_pose_goal_arm_only_forces_no_base_movement(monkeypatch):
    from src.core.controllers import Pose3D

    fake_hsrb = types.SimpleNamespace(
        geometry=types.SimpleNamespace(
            pose=lambda x, y, z, roll, pitch, yaw: (x, y, z, roll, pitch, yaw)
        )
    )
    monkeypatch.setitem(sys.modules, "hsrb_interface", fake_hsrb)
    controller, joint_group, _ros2_control = _controller_with_ros2_control()

    controller.send_pose_goal(ArmPoseRequest(Pose3D(x=0.1, y=0.2, z=0.3, frame_id='odom'), scope='arm_only'))

    assert joint_group.generated_requests
    assert joint_group.generated_requests[-1].base_movement_type.val == _FakeBaseMovementType.NONE


def test_hsrb_pose_goal_whole_body_preserves_planar_base_movement(monkeypatch):
    from src.core.controllers import Pose3D

    fake_hsrb = types.SimpleNamespace(
        geometry=types.SimpleNamespace(
            pose=lambda x, y, z, roll, pitch, yaw: (x, y, z, roll, pitch, yaw)
        )
    )
    monkeypatch.setitem(sys.modules, "hsrb_interface", fake_hsrb)
    controller, joint_group, _ros2_control = _controller_with_ros2_control()

    controller.send_pose_goal(ArmPoseRequest(Pose3D(x=0.1, y=0.2, z=0.3, frame_id='odom'), scope='whole_body'))

    assert joint_group.generated_requests
    assert joint_group.generated_requests[-1].base_movement_type.val == _FakeBaseMovementType.PLANAR


def test_hsrb_cartesian_line_goal_uses_line_planner(monkeypatch):
    from src.core.controllers import Pose3D

    fake_hsrb = types.SimpleNamespace(geometry=types.SimpleNamespace())
    monkeypatch.setitem(sys.modules, "hsrb_interface", fake_hsrb)
    controller, joint_group, _ros2_control = _controller_with_ros2_control()

    controller.send_pose_goal(ArmPoseRequest(Pose3D(x=0.1, y=0.2, z=0.3, frame_id='odom'), motion='cartesian_line', scope='arm_only', line=CartesianLine(axis=[1.0, 0.0, 0.0], distance=0.05)))

    assert joint_group.pose_calls == []
    assert joint_group.line_calls == [([1.0, 0.0, 0.0], 0.05, True)]
    assert joint_group.generated_requests[-1].base_movement_type.val == _FakeBaseMovementType.NONE


def test_ros2_control_pose_planning_failure_is_not_overwritten_by_stale_backend(monkeypatch):
    from src.core.controllers import Pose3D

    fake_hsrb = types.SimpleNamespace(
        geometry=types.SimpleNamespace(
            pose=lambda x, y, z, roll, pitch, yaw: (x, y, z, roll, pitch, yaw)
        )
    )
    monkeypatch.setitem(sys.modules, "hsrb_interface", fake_hsrb)
    controller, joint_group, ros2_control = _controller_with_ros2_control()
    ros2_control.status = "succeeded"
    joint_group.pose_error = RuntimeError("GOAL_IN_COLLISION")

    controller.send_pose_goal(ArmPoseRequest(Pose3D(x=0.1, y=0.2, z=0.3, frame_id='odom')))

    assert controller.get_status() == "failed"
    assert "GOAL_IN_COLLISION" in (controller.get_error() or "")
    assert ros2_control.trajectory_goals == []


def test_hsrb_head_passes_joint_state_provider_to_ros2_control(monkeypatch):
    import src.plugin.controllers.ros2_control_trajectory as ros2_control_module

    captured = {}

    class FakeRos2ControlClient:
        def __init__(self, hardware_config, *, joint_state_provider=None):
            captured["hardware_config"] = hardware_config
            captured["joint_state_provider"] = joint_state_provider

        def send_joint_goal(self, joints):
            return None

        def get_status(self):
            return "idle"

        def get_error(self):
            return None

        def cancel_goal(self):
            return None

    monkeypatch.setattr(
        ros2_control_module,
        "Ros2ControlTrajectoryClient",
        FakeRos2ControlClient,
    )

    controller = HsrbHeadController({"execution_backend": "ros2_control"})

    assert callable(captured["joint_state_provider"])
    assert controller._ros2_control is not None
