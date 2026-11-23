"""Typed arm execution and deterministic setup/default resolution."""
from types import SimpleNamespace
from unittest.mock import Mock
import pytest
from src.core.action import ActionResult
from src.core.arm_request import ArmPoseRequest, CartesianLine, MoveItPoseSettings
from src.core.controllers import Pose3D
from src.core.failure import FailureCode, FailureInfo, moveit_failure
from src.core.moveit_request import MoveItPlanRequest, MoveItPlanningSettings
from src.core.planners import PlannerResult
from src.plugin.controllers.moveit import MoveItArmController
from src.plugin.planners.moveit_planner import MoveItManipulationPlanner


def _controller():
    controller = MoveItArmController.__new__(MoveItArmController)
    controller.hardware_config = {"odom": "odom", "moveit_plan_retries": 1, "moveit_retry_delay_s": 0.0}
    controller._pose_defaults = MoveItPlanningSettings(allowed_planning_time=2., goal_position_tolerance=.02)
    controller._pose_group = "whole_body_weighted"
    controller._pose_ee_link = "hand_palm_link"
    controller._planner = Mock()
    controller._planner.config = {}
    controller._context = SimpleNamespace()
    controller._current_joint_positions = Mock(return_value={"odom_x": 0., "odom_y": 0., "odom_t": 0., "arm_lift_joint": .3})
    controller._ensure_persistent_acm = Mock()
    controller.publish_pose_goal_marker = Mock()
    controller._executor = Mock()
    controller._executor.get_status.return_value = "in_progress"
    return controller


@pytest.mark.parametrize("mode", ["direct_pose", "ik_joint_goal"])
def test_pose_execution_passes_typed_request_with_separate_measured_seed(mode):
    controller = _controller()
    prepared = [[({"name": "arm"}, object())]]
    controller._planner.plan_contract.return_value = PlannerResult(success=True, trajectory=prepared)
    request = ArmPoseRequest(Pose3D(.3, 0, .8), segment="test", moveit=MoveItPoseSettings(goal_mode=mode, constrain_base=True))
    controller.send_pose_goal(request)
    contract = controller._planner.plan_contract.call_args.args[0]
    assert isinstance(contract, MoveItPlanRequest)
    assert contract.pose is request.pose and contract.goal_mode == mode
    assert contract.start_state_seed == controller._current_joint_positions.return_value
    assert contract.base_pose.x == 0. and contract.constrain_base
    assert contract.settings.allowed_planning_time == 2.
    assert contract.settings.goal_position_tolerance == .02
    controller._executor.send_prepared_trajectory_goals.assert_called_once_with(prepared)


def test_cartesian_execution_applies_declared_override_once():
    controller = _controller()
    controller._planner.plan_cartesian_request.return_value = PlannerResult(success=True, trajectory=[[({"name": "arm"}, object())]])
    request = ArmPoseRequest(Pose3D(.3, 0, .8), motion="cartesian_line", line=CartesianLine(axis=(1., 0., 0.), distance=.05, max_step=.005, min_fraction=.95))
    controller.send_pose_goal(request)
    contract = controller._planner.plan_cartesian_request.call_args.args[0]
    assert contract.settings.cartesian_max_step == .005
    assert contract.settings.cartesian_min_fraction == .95
    assert controller._pose_defaults.cartesian_max_step == .01
    controller._planner.plan_contract.assert_not_called()


def test_named_arm_posture_uses_relaxed_basin_without_changing_pose_defaults():
    controller = _controller()
    controller.resolve_named_target = Mock(return_value={"joints": {"arm_lift_joint": .1}})
    controller._named_target_joints = Mock(return_value={"arm_lift_joint": .1})
    controller._joints_within_tolerance = Mock(return_value=False)
    controller._planner.plan_contract.return_value = PlannerResult(success=False, reason="fixture stop")
    controller.send_named_goal("go")
    contract = controller._planner.plan_contract.call_args.args[0]
    assert contract.group_name == "arm"
    assert contract.settings.joint_goal_tolerance == .1
    assert contract.base_constraint_tolerance == dict.fromkeys(("x", "y", "yaw"), .1)
    assert contract.settings.allowed_planning_time == 2.
    assert controller._pose_defaults.joint_goal_tolerance == .01


def test_defaults_are_setup_values_not_live_config_sources():
    config = {"goal_position_tolerance": .025, "base_drift_tolerance": {"x": .04}, "collision_relaxation_links": ["hand"]}
    planner = MoveItManipulationPlanner(config)
    config["base_drift_tolerance"]["x"] = .9
    planner.config["goal_position_tolerance"] = .5
    contract = planner._resolve_planning_inputs(MoveItPlanRequest(group_name="arm", goal_mode="direct_pose"))
    assert contract.settings.goal_position_tolerance == .025
    assert contract.base_constraint_tolerance["x"] == .04
    assert contract.ee_link == "hand_palm_link"
    assert planner._resolve_planning_inputs(contract) is contract


@pytest.mark.parametrize("code,classification,retryable", [(-1, FailureCode.PLANNING_FAILED, True), (-2, FailureCode.PLANNING_FAILED, True), (-4, FailureCode.CONTROLLER_FAILED, False), (-7, FailureCode.CANCELLED, False), (-10, FailureCode.INVALID_STATE, False), (-31, FailureCode.PLANNING_FAILED, True), (99999, FailureCode.UNKNOWN, False)])
def test_backend_codes_define_retry_eligibility(code, classification, retryable):
    failure = moveit_failure(code)
    assert failure.code == classification and failure.retryable is retryable
    assert failure.backend_code == code


def test_unknown_backend_failure_is_not_retried_by_controller():
    controller = _controller()
    controller._planner.plan_contract.return_value = PlannerResult(False, reason="contains planning_failed but unknown")
    controller.send_pose_goal(ArmPoseRequest(Pose3D(.3, 0, .8)))
    assert controller._planner.plan_contract.call_count == 1
    assert controller.get_failure().retryable is False
    controller._executor.send_prepared_trajectory_goals.assert_not_called()


def test_declared_action_inputs_validate_and_serialize_execute_alias():
    from src.plugin.actions.grasp_alignment import SelectFkArmPosture
    from src.plugin.actions.manipulation import PickAtPose
    action = SelectFkArmPosture.from_params(target="mug", execute=False)
    assert callable(action.execute)
    assert action.param_dict()["execute"] is False
    with pytest.raises(ValueError):
        PickAtPose.from_params(x="invalid coordinate")
    with pytest.raises(ValueError):
        PickAtPose.from_params(options={})


def test_invalid_line_request_fails_before_dispatch():
    with pytest.raises(ValueError, match="requires"):
        ArmPoseRequest(Pose3D(0, 0, 0), motion="cartesian_line")
    with pytest.raises(ValueError, match="planning_scope"):
        ArmPoseRequest(Pose3D(0, 0, 0), scope="invalid")


@pytest.mark.parametrize("code,classification", [(-1, FailureCode.INVALID_REQUEST), (-4, FailureCode.LIMIT_VIOLATION), (-5, FailureCode.LIMIT_VIOLATION), (99, FailureCode.UNKNOWN)])
def test_controller_result_failure_codes_are_nonretryable(code, classification):
    from src.core.failure import trajectory_failure
    result = trajectory_failure(code, 6)
    assert result.code == classification and result.retryable is False
    assert result.backend_code == code
