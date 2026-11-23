"""Enforce base limits on planner outputs and final controller payloads."""
import math
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from src.core.base_motion_bounds import validate_base_motion
from src.core.controllers import Pose3D
from src.core.manipulation_types import BasePose2D
from src.core.planners import PlannerResult
from src.plugin.planners.moveit_planner import MoveItManipulationPlanner, MoveItPlanRequest


NAMES = ["odom_x", "odom_y", "odom_t"]
BOUNDS = {"reference": {"x": 0.0, "y": 0.0, "yaw": 0.0}, "joint_names": NAMES,
          "endpoint_tolerance": {"x": 0.03, "y": 0.03, "yaw": 0.05},
          "path_limits": {"xy": 0.08, "yaw": 0.12}, "require_base": True}


def trajectory(rows, names=NAMES):
    result = JointTrajectory()
    result.joint_names = list(names)
    result.points = [JointTrajectoryPoint(positions=[float(value) for value in row]) for row in rows]
    return result


@pytest.mark.parametrize("rows,reason", [
    ([[0, 0, 0], [0.4, 0, 0], [0, 0, 0]], "base_motion_limit_exceeded"),
    ([[0, 0, 0], [0.04, 0, 0]], "base_motion_limit_exceeded"),
    ([[0, 0, 0], [0, 0, 2 * math.pi]], "base_motion_limit_exceeded"),
    ([[0, 0, 0], [float("nan"), 0, 0]], "base_trajectory_nonfinite"),
    ([[0, 0]], "base_trajectory_positions_invalid"),
    ([], "base_trajectory_points_missing"),
])
def test_rejects_bad_trajectory_even_when_endpoint_returns_home(rows, reason):
    result = validate_base_motion(trajectory(rows), **BOUNDS)
    assert result["violation"]["reason"] == reason


def test_exact_limits_pass_and_arm_only_is_explicitly_stationary():
    assert validate_base_motion(trajectory([[0, 0, 0], [0.08, 0, 0.12], [0.03, 0, 0.05]]), **BOUNDS)["base_validation"] == "passed"
    arm = trajectory([[0.1]], ["arm_lift_joint"])
    assert validate_base_motion(arm, **BOUNDS)["violation"]["reason"] == "base_trajectory_joints_missing"
    assert validate_base_motion(arm, **{**BOUNDS, "require_base": False})["base_validation"] == "stationary"


def _planner(monkeypatch):
    planner = MoveItManipulationPlanner({"start_state_complete": True})
    monkeypatch.setattr(planner, "_ensure_ros", lambda *a, **k: True)
    planner._move_group_client = SimpleNamespace(wait_for_server=lambda **k: True)
    planner._send_move_group_plan = Mock(return_value=PlannerResult(success=True, trajectory=object()))
    return planner


@pytest.mark.parametrize("mode", ["direct_pose", "ik_joint_goal"])
def test_both_modes_send_current_seed_goal_and_path_bounds(mode, monkeypatch):
    planner = _planner(monkeypatch)
    seed = dict(zip(NAMES + ["arm_lift_joint"], [0.02, 0.0, 0.0, 0.2]))
    solved = {**seed, "odom_x": 0.025}
    planner._call_ik_request = Mock(return_value=PlannerResult(success=True, joint_positions=solved))
    result = planner.plan_contract(MoveItPlanRequest(
        group_name="whole_body_weighted", goal_mode=mode, pose=Pose3D(0.3, 0, 0.8),
        constrain_base=True, base_pose=BasePose2D(0, 0, 0, "odom"), start_state_seed=seed,
        base_constraint_tolerance={"x": 0.03, "y": 0.03, "yaw": 0.05}), context=None)
    assert result.success
    request = planner._send_move_group_plan.call_args.args[0]
    assert dict(zip(request.start_state.joint_state.name, request.start_state.joint_state.position)) == seed
    assert request.start_state.is_diff is False
    assert [item.joint_name for item in request.path_constraints.joint_constraints] == NAMES
    goal = {item.joint_name: item for item in request.goal_constraints[0].joint_constraints}
    assert set(NAMES) <= set(goal)
    if mode == "ik_joint_goal":
        assert goal["odom_x"].position == 0.025  # preserve IK endpoint
        assert goal["odom_x"].tolerance_above == pytest.approx(0.005)
        assert planner._call_ik_request.call_args.kwargs["seed"] == seed
        assert planner._call_ik_request.call_args.kwargs["avoid_collisions"] is True
    else:
        planner._call_ik_request.assert_not_called()


def test_ik_normalizes_yaw_and_rejects_excessive_base_displacement(monkeypatch):
    planner = _planner(monkeypatch)
    seed = dict(zip(NAMES, [0, 0, 3.31]))
    contract = MoveItPlanRequest(group_name="whole_body_weighted", goal_mode="ik_joint_goal",
        pose=Pose3D(0.3, 0, 0.8), constrain_base=True, base_pose=BasePose2D(0, 0, 3.31, "odom"),
        start_state_seed=seed)
    planner._call_ik_request = Mock(return_value=PlannerResult(success=True, joint_positions={**seed, "odom_t": 3.31 - 2 * math.pi}))
    assert planner.plan_contract(contract).success
    request = planner._send_move_group_plan.call_args.args[0]
    assert next(item for item in request.goal_constraints[0].joint_constraints if item.joint_name == "odom_t").position == pytest.approx(3.31)
    planner._send_move_group_plan.reset_mock()
    planner._call_ik_request.return_value.joint_positions = {**seed, "odom_x": 0.5}
    assert planner.plan_contract(contract).reason == "moveit_ik_base_drift_exceeded"
    planner._send_move_group_plan.assert_not_called()


def test_dispatch_rejects_invalid_prepared_payload_after_sanitization():
    from src.plugin.controllers.moveit import MoveItArmController
    controller = MoveItArmController.__new__(MoveItArmController)
    controller.hardware_config = {"sanitize_base_trajectory": True}
    controller._executor = Mock()
    good = [[({"name": "base"}, trajectory([[0, 0, 0], [0.01, 0, 0]]))]]
    bad = [[({"name": "base"}, trajectory([[0, 0, 0], [0.5, 0, 0]]))]]
    controller._sanitize_prepared_base_batches = Mock(return_value=bad)
    with pytest.raises(RuntimeError, match="moveit_dispatch_base_bounds_exceeded"):
        controller._send_executor_trajectory(good, base_bounds=BOUNDS)
    controller._executor.send_prepared_trajectory_goals.assert_not_called()


def test_dispatch_rejects_missing_base_batch():
    from src.plugin.controllers.moveit import MoveItArmController
    controller = MoveItArmController.__new__(MoveItArmController)
    controller.hardware_config = {}
    controller._executor = Mock()
    with pytest.raises(RuntimeError, match="base_trajectory_missing"):
        controller._send_executor_trajectory([[({}, trajectory([[0.2]], ["arm_lift_joint"]))]], base_bounds=BOUNDS)
    controller._executor.send_prepared_trajectory_goals.assert_not_called()


@pytest.mark.parametrize("mode", ["direct_pose", "ik_joint_goal"])
def test_constraints_reach_actual_move_group_goal(mode, monkeypatch):
    from moveit_msgs.action import MoveGroup
    planner = MoveItManipulationPlanner()
    monkeypatch.setattr(planner, "_ensure_ros", lambda *a, **k: True)
    planner._MoveGroup = MoveGroup
    planner._move_group_client = SimpleNamespace(wait_for_server=lambda **k: True,
                                                 send_goal_async=Mock(return_value=object()))
    raw = SimpleNamespace(joint_trajectory=trajectory([[0, 0, 0], [0.01, 0, 0]]))
    handle = SimpleNamespace(accepted=True, get_result_async=lambda: object())
    planner._spin_until_done = Mock(side_effect=[handle, SimpleNamespace(result=SimpleNamespace(
        error_code=SimpleNamespace(val=1), planned_trajectory=raw))])
    planner._to_ros2_control_batches = Mock(return_value=[[({}, raw.joint_trajectory)]])
    seed = dict(zip(NAMES, [0.0, 0.0, 0.0]))
    planner._call_ik_request = Mock(return_value=PlannerResult(success=True, joint_positions={**seed, "odom_x": 0.01}))
    result = planner.plan_contract(MoveItPlanRequest(group_name="whole_body_weighted", goal_mode=mode,
        pose=Pose3D(0.3, 0, 0.8), constrain_base=True, base_pose=BasePose2D(0, 0, 0, "odom"),
        start_state_seed=seed, max_base_path_drift={"xy": 0.08, "yaw": 0.12}))
    assert result.success
    goal = planner._move_group_client.send_goal_async.call_args.args[0]
    assert goal.planning_options.plan_only
    assert [item.joint_name for item in goal.request.path_constraints.joint_constraints] == NAMES
    assert set(NAMES) <= {item.joint_name for item in goal.request.goal_constraints[0].joint_constraints}
    assert result.metadata["base_validation"] == "passed"


def test_cartesian_route_keeps_primitive_and_enforces_bounds(monkeypatch):
    planner = MoveItManipulationPlanner()
    monkeypatch.setattr(planner, "_ensure_ros", lambda *a, **k: True)
    solution = SimpleNamespace(joint_trajectory=trajectory([[0, 0, 0], [0.01, 0, 0]]))
    modules = SimpleNamespace(call=Mock(return_value=SimpleNamespace(error_code=SimpleNamespace(val=1),
                                         fraction=1.0, solution=solution)))
    planner._to_ros2_control_batches = Mock(return_value=[[({}, solution.joint_trajectory)]])
    options = {"constrain_base": True, "joint_state_seed": dict(zip(NAMES, [0.0, 0.0, 0.0])),
               "base_constraint_tolerance": {"x": 0.03, "y": 0.03, "yaw": 0.05}}
    result = planner.plan_cartesian_path({"pose": Pose3D(0.3, 0, 0.8)}, context=SimpleNamespace(modules=modules), **options)
    assert result.success
    request = modules.call.call_args.args[1]["request"]
    assert [item.joint_name for item in request.path_constraints.joint_constraints] == NAMES
    planner._to_ros2_control_batches.reset_mock()
    solution.joint_trajectory = trajectory([[0, 0, 0], [0.1, 0, 0], [0, 0, 0]])
    result = planner.plan_cartesian_path({"pose": Pose3D(0.3, 0, 0.8)}, context=SimpleNamespace(modules=modules), **options)
    assert not result.success
    planner._to_ros2_control_batches.assert_not_called()
