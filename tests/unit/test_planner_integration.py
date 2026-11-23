"""Tests for manipulation planner registry/session integration."""

from __future__ import annotations
from src.core.failure import FailureInfo, FailureCode
from src.core.arm_request import ArmPoseRequest, MoveItPoseSettings, CartesianLine
from src.core.manipulation_types import BasePose2D

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from src.core.manifest import ControllerMap, PlannerMap, RobotManifest
from src.core.manipulation_types import BasePose2D, MoveItReachabilityRequest, ReachTarget
from src.core.planners import BaseManipulationPlanner, PlannerResult
from src.orchestrator.manifest_listener import load_manifest_from_file
import src.orchestrator.session as session_module
from src.orchestrator.session import RobotSession
from src.plugin.planners.curobo_planner import CuroboManipulationPlanner
from src.plugin.planners.moveit_planner import MoveItManipulationPlanner, MoveItPlanRequest
from src.plugin.registry import PluginRegistry
from tests.fixtures.mock_controllers import MockArmController, MockGripperController, MockNavController


class MockPlanner(BaseManipulationPlanner):
    def check_ik(self, target, *, context=None, **options):
        return PlannerResult(success=True, reason=None, metadata={"target": target})


def _registry_with_mock_planner() -> PluginRegistry:
    registry = PluginRegistry()
    registry.register_controller("nav2", MockNavController)
    registry.register_controller("moveit", MockArmController)
    registry.register_controller("hsr_gripper", MockGripperController)
    registry.register_planner("mock_planner", MockPlanner)
    return registry


def test_registry_registers_and_returns_planner():
    registry = PluginRegistry()
    registry.register_planner("mock_planner", MockPlanner)

    assert registry.get_planner("mock_planner") is MockPlanner
    assert "mock_planner" in registry.available_planners


def test_manifest_file_loads_planner_map(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "robot_id": "bot",
                "robot_type": "mobile_manipulation",
                "controllers": {},
                "planners": {"manipulation": "mock_planner", "default": "mock_planner"},
                "hardware": {},
                "modules": [],
            }
        )
    )

    manifest = load_manifest_from_file(manifest_path)

    assert manifest.planners.manipulation == "mock_planner"
    assert manifest.planners.default == "mock_planner"


def test_hsrb_manifest_uses_nav2_by_default():
    manifest = load_manifest_from_file("manifests/example_manifest_hsrb.json")

    assert manifest.controllers.nav == "nav2"
    assert manifest.hardware["nav"]["action_name"] == "navigate_to_pose"
    assert manifest.hardware["nav"]["frame_id"] == "map"


def test_hsrb_manifest_uses_moveit_manipulation_planner_by_default():
    manifest = load_manifest_from_file("manifests/example_manifest_hsrb.json")

    assert manifest.planners.manipulation == "moveit"
    assert manifest.planners.default == "moveit"


def test_hsrb_manifest_exposes_grasp_alignment_modules_and_actions():
    manifest = load_manifest_from_file("manifests/example_manifest_hsrb.json")
    module_names = {module.name for module in manifest.modules}

    assert "EstimateSimpleCylindricalGrasp" in manifest.capabilities
    assert "EstimateSimplePcaGrasp" in manifest.capabilities
    assert "SelectHsrGraspMotion" in manifest.capabilities
    assert "SolveTmcGraspIk" in manifest.capabilities
    assert "SelectFkArmPosture" in manifest.capabilities
    assert "MoveToSelectedGraspBasePose" in manifest.capabilities
    assert "OmniBaseServoToPose" in manifest.capabilities
    assert "ExecuteHsrCartesianGraspMotion" in manifest.capabilities
    assert "PickTargetWithGraspAlignment" in manifest.capabilities
    assert "moveit_check_state_validity" in module_names
    assert "moveit_clear_octomap" in module_names
    assert "tmc_solve_ik_with_collision" in module_names


def test_hsrb_manifest_moveit_config_matches_current_defaults():
    manifest = load_manifest_from_file("manifests/example_manifest_hsrb.json")
    moveit_cfg = manifest.hardware["planners"]["moveit"]

    assert moveit_cfg["group_name"] == "whole_body_weighted"
    assert moveit_cfg["base_drift_tolerance"] == {"x": 1.3, "y": 1.3, "yaw": 1.5}
    assert moveit_cfg["pipeline_id"] == "move_group"
    assert moveit_cfg["planning_attempts"] == 10
    assert moveit_cfg["max_velocity_scaling_factor"] == 0.5
    assert moveit_cfg["max_acceleration_scaling_factor"] == 0.5
    assert moveit_cfg["start_state_complete"] is True
    assert manifest.hardware["arm"]["whole_body_base_path_tolerance_retries"] == 1
    assert manifest.hardware["arm"]["whole_body_base_retry_delay_s"] == 0.25
    assert moveit_cfg["workspace"] == {
        "frame_id": "base_link",
        "min_corner": [-1.0, -1.0, -1.0],
        "max_corner": [1.0, 1.0, 1.0],
    }
    assert "hand_palm_link" in moveit_cfg["collision_relaxation_links"]
    assert moveit_cfg["collision_relaxation_objects"] == ["pick_target_collision"]
    assert manifest.hardware["manipulation"]["reach_position_tolerance_m"] == 0.1
    assert moveit_cfg["ik_module"] == "moveit_compute_ik"
    assert moveit_cfg["fk_module"] == "moveit_compute_fk"
    assert {module.name for module in manifest.modules} >= {
        "moveit_compute_ik",
        "moveit_compute_fk",
        "moveit_get_planning_scene",
    }


def test_robot_session_builds_planners_into_context(monkeypatch):
    monkeypatch.setattr(session_module, "_try_import_rclpy", lambda: None)
    manifest = RobotManifest(
        robot_id="bot",
        robot_type="mobile_manipulation",
        capabilities=[],
        controllers=ControllerMap(nav="nav2", arm="moveit", gripper="hsr_gripper"),
        planners=PlannerMap(manipulation="mock_planner"),
        hardware={"planners": {"mock_planner": {"foo": "bar"}}},
    )

    session = RobotSession(manifest, _registry_with_mock_planner())

    planner = session.context.require_planner("manipulation")
    assert isinstance(planner, MockPlanner)
    assert session.context.require_planner().config["foo"] == "bar"
    assert session.supported_planners["manipulation"] == "mock_planner"
    assert session.supported_planners["default"] == "mock_planner"


def test_curobo_planner_fails_cleanly_when_unavailable(monkeypatch):
    def _raise(_name):
        raise ImportError("missing curobo")

    monkeypatch.setattr("importlib.import_module", _raise)
    planner = CuroboManipulationPlanner({})

    result = planner.check_ik({"pose": "target"})

    assert result.success is False
    assert result.reason == "curobo_not_available"
    assert "missing curobo" in result.metadata["import_error"]


def test_moveit_planner_converts_candidate_matrix_to_base_frame(monkeypatch):
    from builtin_interfaces.msg import Time

    planner = MoveItManipulationPlanner({"frame_id": "base_footprint"})
    planner._ros_node = SimpleNamespace(get_clock=lambda: SimpleNamespace(now=lambda: SimpleNamespace(to_msg=lambda: Time())))
    target = np.eye(4)
    target[0, 3] = 2.0
    target[1, 3] = 3.0
    target[2, 3] = 0.7

    pose = planner._target_to_pose_stamped(
        {
            "target_pose_matrix": target,
            "target_pose_frame": "map",
            "base_pose": {"x": 1.0, "y": 1.0, "yaw": 0.0, "frame_id": "map"},
        }
    )

    assert pose is not None
    assert pose.header.frame_id == "base_footprint"
    assert pose.pose.position.x == pytest.approx(1.0)
    assert pose.pose.position.y == pytest.approx(2.0)
    assert pose.pose.position.z == pytest.approx(0.7)


def test_moveit_planner_uses_module_proxy_for_ik(monkeypatch):
    from moveit_msgs.srv import GetPositionIK
    from sensor_msgs.msg import JointState

    planner = MoveItManipulationPlanner(
        {
            "group_name": "whole_body_weighted",
            "ee_link": "hand_palm_link",
            "frame_id": "base_footprint",
            "ik_module": "moveit_compute_ik",
        }
    )
    monkeypatch.setattr(planner, "_ensure_ros", lambda context, require_move_group=True: True)
    response = GetPositionIK.Response()
    response.error_code.val = 1
    response.solution.joint_state = JointState(name=["arm_lift_joint"], position=[0.2])
    modules = SimpleNamespace(call=Mock(return_value=response))

    result = planner.check_ik(
        {"ee_pose": [0.4, 0.0, 0.8, 1.0, 0.0, 0.0, 0.0]},
        context=SimpleNamespace(modules=modules),
        joint_state_seed={"arm_lift_joint": 0.1},
    )

    assert result.success is True
    assert result.joint_positions == {"arm_lift_joint": 0.2}
    module_name, payload = modules.call.call_args.args[:2]
    assert module_name == "moveit_compute_ik"
    assert payload["raw_response"] is True
    request = payload["request"]
    assert request.ik_request.group_name == "whole_body_weighted"
    assert request.ik_request.ik_link_name == "hand_palm_link"
    assert request.ik_request.pose_stamped.header.frame_id == "base_footprint"
    assert request.ik_request.robot_state.joint_state.name == ["arm_lift_joint"]


def test_moveit_planner_rejects_invalid_dict_ik_response(monkeypatch):
    planner = MoveItManipulationPlanner({"ik_module": "moveit_compute_ik"})
    monkeypatch.setattr(planner, "_ensure_ros", lambda context, require_move_group=True: True)
    modules = SimpleNamespace(call=Mock(return_value={"error": ""}))

    result = planner.check_ik(
        {"ee_pose": [0.4, 0.0, 0.8, 1.0, 0.0, 0.0, 0.0]},
        context=SimpleNamespace(modules=modules),
    )

    assert result.success is False
    assert result.reason == "moveit_ik_service_invalid_response"


def test_moveit_arm_reachability_from_candidate_base_uses_position_constraint(monkeypatch):
    planner = MoveItManipulationPlanner({"group_name": "arm", "base_link_frame": "base_link"})
    monkeypatch.setattr(planner, "_ensure_ros", lambda context, require_move_group=True: True)
    planner._move_group_client = SimpleNamespace(wait_for_server=lambda timeout_sec: True)
    captured = {}

    def fake_send(request, *, context=None, metadata=None, convert_trajectory=True, **kwargs):
        captured.setdefault("requests", []).append(request)
        captured.setdefault("metadata", []).append(metadata)
        captured.setdefault("convert_trajectory", []).append(convert_trajectory)
        captured.setdefault("kwargs", []).append(kwargs)
        return PlannerResult(success=True)

    monkeypatch.setattr(planner, "_send_move_group_plan", fake_send)
    modules = SimpleNamespace(call=Mock())

    result = planner.check_reachability(
        MoveItReachabilityRequest(
            reach_target=ReachTarget(2.0, 3.0, 0.7, "map"),
            candidate_base=BasePose2D(1.0, 1.0, 0.0, "map"),
            joint_state_seed={"arm_lift_joint": 0.1},
            group_name="arm",
        ),
        context=SimpleNamespace(modules=modules),
    )

    assert result.success is True
    assert result.metadata["method"] == "position_constraint_from_candidate_base"
    assert modules.call.call_count == 0
    assert captured["convert_trajectory"] == [False, False]
    request = captured["requests"][0]
    assert request.group_name == "arm"
    assert dict(zip(request.start_state.joint_state.name, request.start_state.joint_state.position, strict=False)) == {
        "arm_lift_joint": 0.1,
        "odom_x": 1.0,
        "odom_y": 1.0,
        "odom_t": 0.0,
    }
    constraints = request.goal_constraints[0]
    assert constraints.joint_constraints == []
    assert len(constraints.position_constraints) == 1
    assert constraints.orientation_constraints == []
    pc = constraints.position_constraints[0]
    assert pc.header.frame_id == "base_link"
    assert pc.link_name == "hand_palm_link"
    assert pc.constraint_region.primitive_poses[0].position.x == pytest.approx(1.0)
    assert pc.constraint_region.primitive_poses[0].position.y == pytest.approx(2.0)


def test_moveit_whole_body_reachability_seeds_base_and_holds_base_in_position_plan(monkeypatch):
    planner = MoveItManipulationPlanner({"group_name": "whole_body_weighted", "frame_id": "map"})
    monkeypatch.setattr(planner, "_ensure_ros", lambda context, require_move_group=True: True)
    planner._move_group_client = SimpleNamespace(wait_for_server=lambda timeout_sec: True)
    modules = SimpleNamespace(call=Mock())
    captured = {}

    def fake_send(request, *, context=None, metadata=None, convert_trajectory=True, **kwargs):
        captured.setdefault("requests", []).append(request)
        captured.setdefault("metadata", []).append(metadata)
        captured.setdefault("convert_trajectory", []).append(convert_trajectory)
        return PlannerResult(success=True)

    monkeypatch.setattr(planner, "_send_move_group_plan", fake_send)

    result = planner.check_reachability(
        MoveItReachabilityRequest(
            reach_target=ReachTarget(2.0, 3.0, 0.7, "map"),
            candidate_base=BasePose2D(1.0, 1.0, 0.0, "map"),
            joint_state_seed={"arm_lift_joint": 0.1},
            group_name="whole_body_weighted",
        ),
        context=SimpleNamespace(modules=modules),
    )

    assert result.success is True
    assert modules.call.call_count == 0
    assert captured["convert_trajectory"] == [False, False]
    request = captured["requests"][0]
    assert request.group_name == "whole_body_weighted"
    assert dict(zip(request.start_state.joint_state.name, request.start_state.joint_state.position, strict=False))["odom_x"] == pytest.approx(1.0)
    joint_constraints = {jc.joint_name: jc for jc in request.goal_constraints[0].joint_constraints}
    assert joint_constraints["odom_x"].position == pytest.approx(1.0)
    assert joint_constraints["odom_y"].position == pytest.approx(1.0)
    assert joint_constraints["odom_t"].position == pytest.approx(0.0)


def test_moveit_typed_reachability_request_forwards_contract_fields(monkeypatch):
    planner = MoveItManipulationPlanner({"group_name": "whole_body_weighted", "frame_id": "map"})
    captured = {}

    def fake_check(reach_target, candidate_base, joint_state_seed, *, context=None, group_name=None, ee_link=None, validate_plan=True, avoid_collisions_for_ik=True, avoid_collisions_for_plan=True, **options):
        captured["reach_target"] = reach_target
        captured["candidate_base"] = candidate_base
        captured["joint_state_seed"] = joint_state_seed
        captured["group_name"] = group_name
        captured["ee_link"] = ee_link
        captured["validate_plan"] = validate_plan
        captured["avoid_collisions_for_ik"] = avoid_collisions_for_ik
        captured["avoid_collisions_for_plan"] = avoid_collisions_for_plan
        captured["options"] = options
        return PlannerResult(success=True, metadata={})

    monkeypatch.setattr(planner, "_check_position_plan_reachability_from_base", fake_check)

    request = MoveItReachabilityRequest(
        reach_target=ReachTarget(2.0, 3.0, 0.7, "map", source="test"),
        candidate_base=BasePose2D(1.0, 1.0, 0.2, "map"),
        joint_state_seed={"arm_lift_joint": 0.1},
        group_name="whole_body_weighted",
        ee_link="hand_palm_link",
        validate_plan=False,
        avoid_collisions_for_ik=False,
        avoid_collisions_for_plan=True,
        position_tolerance_m=0.1,
        orientation_tolerance_rad=0.5,
        base_drift_tolerance={"x": 0.2, "y": 0.25, "yaw": 0.4},
        metadata={"target": "mug"},
    )

    result = planner.check_reachability(request)

    assert result.success is True
    assert captured["reach_target"] is request.reach_target
    assert captured["candidate_base"] is request.candidate_base
    assert captured["joint_state_seed"] == {"arm_lift_joint": 0.1}
    assert captured["group_name"] == "whole_body_weighted"
    assert captured["ee_link"] == "hand_palm_link"
    assert captured["validate_plan"] is False
    assert captured["avoid_collisions_for_ik"] is False
    assert captured["avoid_collisions_for_plan"] is True
    assert captured["options"]["reach_position_tolerance_m"] == pytest.approx(0.1)
    assert captured["options"]["goal_orientation_tolerance"] == pytest.approx(0.5)
    assert captured["options"]["base_drift_tolerance"] == {"x": 0.2, "y": 0.25, "yaw": 0.4}
    assert result.metadata["method"] == "typed_moveit_reachability"
    assert result.metadata["request"]["metadata"]["target"] == "mug"


def test_moveit_reachability_kinematic_failure_skips_collision_stage(monkeypatch):
    planner = MoveItManipulationPlanner({"group_name": "arm", "base_link_frame": "base_link"})
    monkeypatch.setattr(planner, "_ensure_ros", lambda context, require_move_group=True: True)
    planner._move_group_client = SimpleNamespace(wait_for_server=lambda timeout_sec: True)
    send = Mock(return_value=PlannerResult(success=False, reason="moveit_plan_error_99999"))
    monkeypatch.setattr(planner, "_send_move_group_plan", send)

    result = planner.check_reachability(
        MoveItReachabilityRequest(
            reach_target=ReachTarget(2.0, 3.0, 0.7, "map"),
            candidate_base=BasePose2D(1.0, 1.0, 0.0, "map"),
            joint_state_seed={"arm_lift_joint": 0.1},
            group_name="arm",
        ),
        context=SimpleNamespace(modules=SimpleNamespace(call=Mock())),
    )

    assert result.success is False
    assert result.reason == "moveit_plan_error_99999"
    assert send.call_count == 1


def test_moveit_reachability_collision_failure_rejects_candidate(monkeypatch):
    planner = MoveItManipulationPlanner({"group_name": "arm", "base_link_frame": "base_link"})
    monkeypatch.setattr(planner, "_ensure_ros", lambda context, require_move_group=True: True)
    planner._move_group_client = SimpleNamespace(wait_for_server=lambda timeout_sec: True)
    send = Mock(side_effect=[PlannerResult(success=True), PlannerResult(success=False, reason="moveit_plan_error_31")])
    monkeypatch.setattr(planner, "_send_move_group_plan", send)

    result = planner.check_reachability(
        MoveItReachabilityRequest(
            reach_target=ReachTarget(2.0, 3.0, 0.7, "map"),
            candidate_base=BasePose2D(1.0, 1.0, 0.0, "map"),
            joint_state_seed={"arm_lift_joint": 0.1},
            group_name="arm",
        ),
        context=SimpleNamespace(modules=SimpleNamespace(call=Mock())),
    )

    assert result.success is False
    assert result.reason == "moveit_plan_error_31"
    assert send.call_count == 2


def test_moveit_planner_accepts_fk_pose_stamped_dict_for_ik(monkeypatch):
    from moveit_msgs.srv import GetPositionIK

    planner = MoveItManipulationPlanner({"ik_module": "moveit_compute_ik"})
    monkeypatch.setattr(planner, "_ensure_ros", lambda context, require_move_group=True: True)
    response = GetPositionIK.Response()
    response.error_code.val = 1
    modules = SimpleNamespace(call=Mock(return_value=response))

    result = planner.check_ik(
        {
            "pose": {
                "header": {"frame_id": "base_footprint"},
                "pose": {
                    "position": {"x": 0.3, "y": 0.0, "z": 0.8},
                    "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
                },
            }
        },
        context=SimpleNamespace(modules=modules),
    )

    assert result.success is True
    request = modules.call.call_args.args[1]["request"]
    assert request.ik_request.pose_stamped.header.frame_id == "base_footprint"
    assert request.ik_request.pose_stamped.pose.position.x == pytest.approx(0.3)


def test_moveit_planner_uses_module_proxy_for_fk(monkeypatch):
    from moveit_msgs.srv import GetPositionFK
    from sensor_msgs.msg import JointState

    planner = MoveItManipulationPlanner({"frame_id": "base_footprint", "fk_module": "moveit_compute_fk"})
    monkeypatch.setattr(planner, "_ensure_ros", lambda context, require_move_group=True: True)
    response = GetPositionFK.Response()
    response.error_code.val = 1
    modules = SimpleNamespace(call=Mock(return_value=response))
    seed = JointState(name=["arm_lift_joint"], position=[0.2])

    result = planner.compute_fk(seed, context=SimpleNamespace(modules=modules), link_names=["hand_palm_link"])

    assert result.success is True
    module_name, payload = modules.call.call_args.args[:2]
    assert module_name == "moveit_compute_fk"
    assert payload["raw_response"] is True
    request = payload["request"]
    assert request.header.frame_id == "base_footprint"
    assert request.fk_link_names == ["hand_palm_link"]
    assert request.robot_state.joint_state.name == ["arm_lift_joint"]


def test_moveit_planner_state_validity_parses_valid_response(monkeypatch):
    from moveit_msgs.srv import GetStateValidity
    from sensor_msgs.msg import JointState

    planner = MoveItManipulationPlanner({"state_validity_module": "moveit_check_state_validity"})
    monkeypatch.setattr(planner, "_ensure_ros", lambda context, require_move_group=True: True)
    response = GetStateValidity.Response()
    response.valid = True
    modules = SimpleNamespace(call=Mock(return_value=response))
    joint_state = JointState(name=["arm_lift_joint"], position=[0.2])

    result = planner.check_state_validity(
        joint_state,
        context=SimpleNamespace(modules=modules),
        group_name="whole_body_weighted",
    )

    assert result.success is True
    assert result.metadata["valid"] is True
    module_name, payload = modules.call.call_args.args[:2]
    assert module_name == "moveit_check_state_validity"
    assert payload["raw_response"] is True
    assert payload["request"].group_name == "whole_body_weighted"


def test_moveit_planner_state_validity_parses_invalid_response(monkeypatch):
    from moveit_msgs.srv import GetStateValidity
    from sensor_msgs.msg import JointState

    planner = MoveItManipulationPlanner({"state_validity_module": "moveit_check_state_validity"})
    monkeypatch.setattr(planner, "_ensure_ros", lambda context, require_move_group=True: True)
    response = GetStateValidity.Response()
    response.valid = False
    modules = SimpleNamespace(call=Mock(return_value=response))
    joint_state = JointState(name=["arm_lift_joint"], position=[0.2])

    result = planner.check_state_validity(
        joint_state,
        context=SimpleNamespace(modules=modules),
        group_name="whole_body_weighted",
    )

    assert result.success is False
    assert result.reason == "moveit_state_invalid"
    assert result.metadata["valid"] is False


def test_moveit_planner_converts_robot_trajectory_to_ros2_control_batches():
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

    planner = MoveItManipulationPlanner(
        {
            "ros2_control_controllers": [
                {
                    "name": "arm",
                    "action": "/arm_trajectory_controller/follow_joint_trajectory",
                    "joints": ["arm_lift_joint", "wrist_flex_joint"],
                },
                {
                    "name": "head",
                    "action": "/head_trajectory_controller/follow_joint_trajectory",
                    "joints": ["head_pan_joint"],
                },
                {
                    "name": "omni_base",
                    "action": "/omni_base_controller/follow_joint_trajectory",
                    "joints": ["odom_x", "odom_y", "odom_t"],
                },
            ]
        }
    )
    trajectory = JointTrajectory()
    trajectory.joint_names = [
        "odom_y",
        "arm_lift_joint",
        "wrist_ft_sensor_frame_joint",
        "head_pan_joint",
        "odom_x",
        "wrist_flex_joint",
        "odom_t",
    ]
    point = JointTrajectoryPoint()
    point.positions = [1.0, 2.0, 9.0, 3.0, 4.0, 5.0, 6.0]
    point.velocities = [0.1, 0.2, 0.9, 0.3, 0.4, 0.5, 0.6]
    trajectory.points.append(point)

    batches = planner._to_ros2_control_batches(SimpleNamespace(joint_trajectory=trajectory))

    assert batches is not None
    batch = batches[0]
    assert [controller["name"] for controller, _traj in batch] == ["arm", "head", "omni_base"]
    assert batch[0][1].joint_names == ["arm_lift_joint", "wrist_flex_joint"]
    assert list(batch[0][1].points[0].positions) == [2.0, 5.0]
    assert batch[1][1].joint_names == ["head_pan_joint"]
    assert list(batch[1][1].points[0].positions) == [3.0]
    assert batch[2][1].joint_names == ["odom_x", "odom_y", "odom_t"]
    assert list(batch[2][1].points[0].positions) == [4.0, 1.0, 6.0]


def test_moveit_reachability_base_drift_metadata_uses_planned_endpoint():
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

    planner = MoveItManipulationPlanner()
    trajectory = JointTrajectory()
    trajectory.joint_names = ["odom_x", "odom_y", "odom_t", "arm_lift_joint"]
    point = JointTrajectoryPoint()
    point.positions = [1.12, 0.91, 0.20, 0.08]
    trajectory.points.append(point)
    result = PlannerResult(
        success=True,
        metadata={
            "final_joint_positions": planner._trajectory_final_joint_positions(
                SimpleNamespace(joint_trajectory=trajectory)
            )
        },
    )

    metadata = planner._base_drift_metadata(result, BasePose2D(1.0, 1.0, 0.0, "map"))

    assert metadata["base_drift"]["odom_x"] == pytest.approx(0.12)
    assert metadata["base_drift"]["odom_y"] == pytest.approx(0.09)
    assert metadata["base_drift"]["odom_t"] == pytest.approx(0.20)
    assert metadata["base_drift_xy_m"] == pytest.approx(0.15)
    assert metadata["base_drift_yaw_rad"] == pytest.approx(0.20)


def test_moveit_base_path_drift_metadata_scans_all_points():
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

    planner = MoveItManipulationPlanner()
    trajectory = JointTrajectory()
    trajectory.joint_names = ["odom_x", "odom_y", "odom_t"]
    for values in ([1.0, 1.0, 0.0], [0.5, 1.0, 0.2], [1.02, 1.01, 0.01]):
        point = JointTrajectoryPoint()
        point.positions = list(values)
        trajectory.points.append(point)

    metadata = planner._base_path_drift_metadata(
        SimpleNamespace(joint_trajectory=trajectory),
        BasePose2D(1.0, 1.0, 0.0, "odom"),
    )

    assert metadata["max_base_path_drift_xy_m"] == pytest.approx(0.5)
    assert metadata["max_base_path_drift_xy_index"] == 1
    assert metadata["max_base_path_drift_yaw_rad"] == pytest.approx(0.2)
    assert metadata["max_base_path_drift_yaw_index"] == 1


def test_moveit_base_path_drift_failure_detects_intermediate_excursion(monkeypatch):
    from moveit_msgs.action import MoveGroup
    from moveit_msgs.msg import MotionPlanRequest
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

    planner = MoveItManipulationPlanner()
    planner._MoveGroup = MoveGroup
    planner._move_group_client = SimpleNamespace(send_goal_async=Mock(return_value=object()))
    trajectory = JointTrajectory()
    trajectory.joint_names = ["odom_x", "odom_y", "odom_t"]
    for values in ([1.0, 1.0, 0.0], [0.4, 1.0, 0.0], [1.0, 1.0, 0.0]):
        point = JointTrajectoryPoint()
        point.positions = list(values)
        trajectory.points.append(point)
    moveit_result = SimpleNamespace(
        error_code=SimpleNamespace(val=1),
        planned_trajectory=SimpleNamespace(joint_trajectory=trajectory),
    )
    goal_handle = SimpleNamespace(accepted=True, get_result_async=Mock(return_value=object()))
    monkeypatch.setattr(
        planner,
        "_spin_until_done",
        Mock(side_effect=[goal_handle, SimpleNamespace(result=moveit_result)]),
    )

    result = planner._send_move_group_plan(
        MotionPlanRequest(),
        convert_trajectory=False,
        options={
            "base_hold_pose": BasePose2D(1.0, 1.0, 0.0, "odom"),
            "max_base_path_drift": {"xy": 0.1, "yaw": 0.2},
        },
    )

    assert result.success is False
    assert result.reason == "moveit_plan_base_path_drift_exceeded"
    assert result.metadata["max_base_path_drift_xy_m"] == pytest.approx(0.6)


def test_moveit_error_code_names_and_reasons_are_readable():
    assert MoveItManipulationPlanner._moveit_error_name(1) == "SUCCESS"
    assert MoveItManipulationPlanner._moveit_error_name(-2) == "INVALID_MOTION_PLAN"
    assert MoveItManipulationPlanner._moveit_error_name(-31) == "NO_IK_SOLUTION"
    assert MoveItManipulationPlanner._moveit_error_name(-12345) == "UNKNOWN_-12345"
    assert MoveItManipulationPlanner._moveit_reason("moveit_plan", -2) == "moveit_plan_invalid_motion_plan"


def test_moveit_request_defaults_match_rviz_parity_fields():
    from moveit_msgs.msg import MotionPlanRequest

    planner = MoveItManipulationPlanner(
        {
            "pipeline_id": "move_group",
            "planning_attempts": 10,
            "max_velocity_scaling_factor": 0.1,
            "max_acceleration_scaling_factor": 0.1,
            "workspace": {
                "frame_id": "odom",
                "min_corner": [-1.0, -1.0, -1.0],
                "max_corner": [1.0, 1.0, 1.0],
            },
        }
    )
    request = MotionPlanRequest()

    planner._apply_motion_request_defaults(request, {})
    summary = planner._motion_plan_request_summary(request)

    assert summary["pipeline_id"] == "move_group"
    assert summary["num_planning_attempts"] == 10
    assert summary["max_velocity_scaling_factor"] == pytest.approx(0.1)
    assert summary["max_acceleration_scaling_factor"] == pytest.approx(0.1)
    assert summary["workspace"] == {
        "frame_id": "odom",
        "min_corner": [-1.0, -1.0, -1.0],
        "max_corner": [1.0, 1.0, 1.0],
    }


def test_moveit_complete_start_state_sets_is_diff_false():
    from geometry_msgs.msg import PoseStamped

    planner = MoveItManipulationPlanner({"start_state_complete": True})
    pose = PoseStamped()
    pose.header.frame_id = "map"

    request = planner._build_position_reach_request(
        pose,
        "whole_body_weighted",
        {"ee_link": "hand_palm_link"},
        {},
        start_state_seed={"odom_x": 1.0, "odom_y": 2.0, "odom_t": 0.3, "arm_lift_joint": 0.1},
    )

    assert request.start_state.is_diff is False
    assert request.start_state.joint_state.name == ["odom_x", "odom_y", "odom_t", "arm_lift_joint"]


def test_moveit_invalid_motion_plan_reports_visible_trajectory(monkeypatch):
    from moveit_msgs.action import MoveGroup
    from moveit_msgs.msg import MotionPlanRequest
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

    planner = MoveItManipulationPlanner()
    planner._MoveGroup = MoveGroup
    planner._move_group_client = SimpleNamespace(send_goal_async=Mock(return_value=object()))
    trajectory = JointTrajectory()
    trajectory.joint_names = ["odom_x", "odom_y", "odom_t"]
    point = JointTrajectoryPoint()
    point.positions = [1.0, 2.0, 0.3]
    trajectory.points.append(point)
    moveit_result = SimpleNamespace(
        error_code=SimpleNamespace(val=-2),
        planned_trajectory=SimpleNamespace(joint_trajectory=trajectory),
    )
    goal_handle = SimpleNamespace(accepted=True, get_result_async=Mock(return_value=object()))
    monkeypatch.setattr(
        planner,
        "_spin_until_done",
        Mock(side_effect=[goal_handle, SimpleNamespace(result=moveit_result)]),
    )

    result = planner._send_move_group_plan(MotionPlanRequest(), convert_trajectory=False)

    assert result.success is False
    assert result.reason == "moveit_plan_invalid_motion_plan"
    assert result.metadata["error_code"] == -2
    assert result.metadata["error_code_name"] == "INVALID_MOTION_PLAN"
    assert result.metadata["has_planned_trajectory"] is True
    assert result.metadata["trajectory_point_count"] == 1
    assert result.metadata["final_joint_positions"] == {"odom_x": 1.0, "odom_y": 2.0, "odom_t": 0.3}


def test_moveit_contract_joint_goal_adds_base_hold_constraints():
    from moveit_msgs.msg import MotionPlanRequest

    planner = MoveItManipulationPlanner()
    captured = {}

    def fake_send(request, **kwargs):
        captured["request"] = request
        captured["kwargs"] = kwargs
        return PlannerResult(success=True, trajectory=[[({"name": "arm"}, object())]], metadata={})

    planner._MoveGroup = object
    planner._move_group_client = SimpleNamespace(wait_for_server=Mock(return_value=True))
    planner._ensure_ros = Mock(return_value=True)
    planner._send_move_group_plan = fake_send

    result = planner.plan_contract(
        MoveItPlanRequest(
            group_name="whole_body_weighted",
            goal_mode="joint_goal",
            joint_positions={"arm_lift_joint": 0.1},
            constrain_base=True,
            base_pose=BasePose2D(1.0, 2.0, 0.3, "odom"),
            start_state_seed={"odom_x": 1.0, "odom_y": 2.0, "odom_t": 0.3, "arm_lift_joint": 0.0},
            base_constraint_tolerance={"x": 0.03, "y": 0.04, "yaw": 0.05},
        )
    )

    assert result.success is True
    request = captured["request"]
    assert isinstance(request, MotionPlanRequest)
    constraints = {jc.joint_name: jc for jc in request.goal_constraints[0].joint_constraints}
    assert constraints["arm_lift_joint"].position == pytest.approx(0.1)
    assert constraints["odom_x"].position == pytest.approx(1.0)
    assert constraints["odom_x"].tolerance_above == pytest.approx(0.03)
    assert constraints["odom_y"].tolerance_above == pytest.approx(0.04)
    assert constraints["odom_t"].tolerance_above == pytest.approx(0.05)


def test_moveit_contract_whole_body_joint_goal_filters_non_planning_joints():
    from moveit_msgs.msg import MotionPlanRequest

    planner = MoveItManipulationPlanner()
    captured = {}

    def fake_send(request, **kwargs):
        captured["request"] = request
        captured["kwargs"] = kwargs
        return PlannerResult(success=True, trajectory=[[({"name": "arm"}, object())]], metadata={})

    planner._MoveGroup = object
    planner._move_group_client = SimpleNamespace(wait_for_server=Mock(return_value=True))
    planner._ensure_ros = Mock(return_value=True)
    planner._send_move_group_plan = fake_send

    result = planner.plan_contract(
        MoveItPlanRequest(
            group_name="whole_body_weighted",
            goal_mode="joint_goal",
            joint_positions={
                "odom_x": 1.0,
                "odom_y": 0.0,
                "odom_t": 0.0,
                "arm_lift_joint": 0.1,
                "wrist_flex_joint": -1.2,
                "hand_l_proximal_joint": -0.8,
                "hand_l_spring_proximal_joint": 0.7,
                "hand_l_mimic_distal_joint": -0.7,
                "hand_motor_joint": -0.8,
                "base_l_drive_wheel_joint": 2.0,
                "base_f_bumper_joint": 0.0,
                "wrist_ft_sensor_frame_joint": 0.0,
            },
            start_state_seed={
                "odom_x": 0.0,
                "hand_l_proximal_joint": -0.9,
                "hand_l_spring_proximal_joint": 0.8,
                "base_l_drive_wheel_joint": 2.0,
            },
        )
    )

    assert result.success is True
    request = captured["request"]
    assert isinstance(request, MotionPlanRequest)
    constrained = {jc.joint_name for jc in request.goal_constraints[0].joint_constraints}
    assert constrained == {
        "odom_x",
        "odom_y",
        "odom_t",
        "arm_lift_joint",
        "wrist_flex_joint",
    }
    start_state = dict(zip(request.start_state.joint_state.name, request.start_state.joint_state.position, strict=True))
    assert start_state["hand_l_proximal_joint"] == pytest.approx(-0.9)
    assert start_state["hand_l_spring_proximal_joint"] == pytest.approx(0.8)
    assert start_state["base_l_drive_wheel_joint"] == pytest.approx(2.0)


def test_moveit_contract_base_joint_goal_filters_to_virtual_base_joints():
    from moveit_msgs.msg import MotionPlanRequest

    planner = MoveItManipulationPlanner()
    captured = {}

    def fake_send(request, **kwargs):
        captured["request"] = request
        return PlannerResult(success=True, trajectory=[[({"name": "omni_base"}, object())]], metadata={})

    planner._MoveGroup = object
    planner._move_group_client = SimpleNamespace(wait_for_server=Mock(return_value=True))
    planner._ensure_ros = Mock(return_value=True)
    planner._send_move_group_plan = fake_send

    result = planner.plan_contract(
        MoveItPlanRequest(
            group_name="base",
            goal_mode="joint_goal",
            joint_positions={
                "odom_x": 1.0,
                "odom_y": 2.0,
                "odom_t": 0.5,
                "arm_lift_joint": 0.1,
                "hand_motor_joint": -0.8,
            },
        )
    )

    assert result.success is True
    request = captured["request"]
    assert isinstance(request, MotionPlanRequest)
    constrained = {jc.joint_name for jc in request.goal_constraints[0].joint_constraints}
    assert constrained == {"odom_x", "odom_y", "odom_t"}


def test_moveit_contract_arm_joint_goal_filters_to_arm_chain_joints():
    from moveit_msgs.msg import MotionPlanRequest

    planner = MoveItManipulationPlanner()
    captured = {}

    def fake_send(request, **kwargs):
        captured["request"] = request
        return PlannerResult(success=True, trajectory=[[({"name": "arm"}, object())]], metadata={})

    planner._MoveGroup = object
    planner._move_group_client = SimpleNamespace(wait_for_server=Mock(return_value=True))
    planner._ensure_ros = Mock(return_value=True)
    planner._send_move_group_plan = fake_send

    result = planner.plan_contract(
        MoveItPlanRequest(
            group_name="arm",
            goal_mode="joint_goal",
            joint_positions={
                "arm_lift_joint": 0.1,
                "arm_flex_joint": -0.2,
                "wrist_roll_joint": 1.0,
                "odom_x": 1.0,
                "hand_motor_joint": -0.8,
            },
        )
    )

    assert result.success is True
    request = captured["request"]
    assert isinstance(request, MotionPlanRequest)
    constrained = {jc.joint_name for jc in request.goal_constraints[0].joint_constraints}
    assert constrained == {"arm_lift_joint", "arm_flex_joint", "wrist_roll_joint"}


def test_robot_session_controller_hardware_injects_joint_state_provider():
    joint_state = SimpleNamespace(name=["arm_lift_joint"], position=[0.1])
    sensor_data_center = SimpleNamespace(
        peek_latest=Mock(return_value=SimpleNamespace(data=joint_state)),
        get_latest=Mock(side_effect=AssertionError("joint_states should use persistent peek_latest")),
    )
    session = object.__new__(RobotSession)
    session.ros2_wrapper = None
    manifest = RobotManifest(
        robot_id="hsrb",
        robot_type="mobile_manipulation",
        hardware={"arm": {}},
    )

    hardware = session._controller_hardware(manifest, "arm", sensor_data_center=sensor_data_center)

    assert callable(hardware["joint_state_provider"])
    assert hardware["joint_state_provider"]() is joint_state
    sensor_data_center.peek_latest.assert_called_once_with("joint_states", timeout_s=0.2)
    sensor_data_center.get_latest.assert_not_called()


def test_moveit_contract_ik_joint_goal_rejects_missing_start_state():
    planner = MoveItManipulationPlanner()
    planner._MoveGroup = object
    planner._move_group_client = SimpleNamespace(wait_for_server=Mock(return_value=True))
    planner._ensure_ros = Mock(return_value=True)
    planner._send_move_group_plan = Mock()

    result = planner.plan_contract(
        MoveItPlanRequest(
            group_name="whole_body_weighted",
            goal_mode="ik_joint_goal",
            pose=SimpleNamespace(x=0.3, y=0.0, z=0.8, frame_id="odom"),
            ee_link="hand_palm_link",
        )
    )

    assert result.success is False
    assert result.reason == "moveit_ik_start_state_missing"
    assert result.metadata["seed_joint_count"] == 0
    planner._send_move_group_plan.assert_not_called()


def test_moveit_arm_controller_pose_goal_fails_before_ik_when_start_state_missing(monkeypatch):
    import src.plugin.controllers.ros2_control_trajectory as ros2_control_module
    import src.plugin.planners.moveit_planner as moveit_planner_module
    from src.core.controllers import Pose3D
    from src.plugin.controllers.moveit import MoveItArmController

    calls = {}

    class FakePlanner:
        config = {"group_name": "whole_body_weighted"}

        def __init__(self, config):
            self.config = config

        def plan_contract(self, *args, **kwargs):
            calls["planned"] = True
            return PlannerResult(success=True, trajectory=[[({"name": "arm"}, object())]])

    class FakeExecutor:
        def __init__(self, hardware_config, joint_state_provider=None):
            pass

    monkeypatch.setattr(moveit_planner_module, "MoveItManipulationPlanner", FakePlanner)
    monkeypatch.setattr(ros2_control_module, "Ros2ControlController", FakeExecutor)
    controller = MoveItArmController(
        {
            "joint_state_provider": lambda: SimpleNamespace(name=[], position=[]),
            "moveit_plan_retries": 0,
            "moveit_ik_retries": 0,
        }
    )

    controller.send_pose_goal(ArmPoseRequest(Pose3D(x=0.3, y=0.0, z=0.8, frame_id='odom'), moveit=MoveItPoseSettings(goal_mode='ik_joint_goal', group_name='whole_body_weighted')))

    assert controller.get_status() == "failed"
    assert controller.get_error() == "moveit_start_state_unavailable"
    assert "planned" not in calls


def test_moveit_arm_controller_constrained_pose_goal_fails_without_base_pose(monkeypatch):
    import src.plugin.controllers.ros2_control_trajectory as ros2_control_module
    import src.plugin.planners.moveit_planner as moveit_planner_module
    from src.core.controllers import Pose3D
    from src.plugin.controllers.moveit import MoveItArmController

    calls = {}

    class FakePlanner:
        config = {"group_name": "whole_body_weighted"}

        def __init__(self, config):
            self.config = config

        def plan_contract(self, *args, **kwargs):
            calls["planned"] = True
            return PlannerResult(success=True, trajectory=[])

    class FakeExecutor:
        def __init__(self, hardware_config, joint_state_provider=None):
            pass

        def get_status(self):
            return "idle"

        def get_error(self):
            return None

    monkeypatch.setattr(moveit_planner_module, "MoveItManipulationPlanner", FakePlanner)
    monkeypatch.setattr(ros2_control_module, "Ros2ControlController", FakeExecutor)
    controller = MoveItArmController(
        {
            "joint_state_provider": lambda: SimpleNamespace(name=["arm_lift_joint"], position=[0.1]),
            "moveit_plan_retries": 0,
            "moveit_ik_retries": 0,
        }
    )

    controller.send_pose_goal(ArmPoseRequest(Pose3D(x=0.3, y=0.0, z=0.8, frame_id='odom'), moveit=MoveItPoseSettings(goal_mode='ik_joint_goal', group_name='whole_body_weighted', constrain_base=True)))

    assert controller.get_status() == "failed"
    assert controller.get_error() == "moveit_base_constraint_requested_but_current_base_pose_unavailable"
    assert "planned" not in calls


def test_moveit_arm_controller_pose_goal_retry_refreshes_start_state(monkeypatch):
    import src.plugin.controllers.ros2_control_trajectory as ros2_control_module
    import src.plugin.planners.moveit_planner as moveit_planner_module
    from src.core.controllers import Pose3D
    from src.plugin.controllers.moveit import MoveItArmController

    calls = {}
    states = iter(
        [
            SimpleNamespace(name=[], position=[]),
            SimpleNamespace(name=["arm_lift_joint"], position=[0.1]),
        ]
    )
    prepared = [[({"name": "arm", "action": "/arm"}, object())]]

    class FakePlanner:
        config = {"group_name": "whole_body_weighted"}

        def __init__(self, config):
            self.config = config

        def plan_contract(self, contract, *, context=None):
            calls.setdefault("contracts", []).append(contract)
            return PlannerResult(success=True, trajectory=prepared)

    class FakeExecutor:
        def __init__(self, hardware_config, joint_state_provider=None):
            self.status = "idle"

        def send_prepared_trajectory_goals(self, trajectory):
            calls["prepared"] = trajectory
            self.status = "succeeded"

        def get_status(self):
            return self.status

        def get_error(self):
            return None

    monkeypatch.setattr(moveit_planner_module, "MoveItManipulationPlanner", FakePlanner)
    monkeypatch.setattr(ros2_control_module, "Ros2ControlController", FakeExecutor)
    controller = MoveItArmController(
        {
            "joint_state_provider": lambda: next(states),
            "moveit_plan_retries": 1,
            "moveit_ik_retries": 1,
            "moveit_retry_delay_s": 0.0,
            "ros2_control": {"controllers": []},
        }
    )

    controller.send_pose_goal(ArmPoseRequest(Pose3D(x=0.3, y=0.0, z=0.8, frame_id='odom'), moveit=MoveItPoseSettings(goal_mode='ik_joint_goal', group_name='whole_body_weighted')))

    assert controller.get_status() == "failed"
    assert controller.get_failure().retryable is False
    assert "contracts" not in calls
    assert "prepared" not in calls


def test_moveit_arm_controller_pose_goal_retry_refreshes_state_after_plan_failure(monkeypatch):
    import src.plugin.controllers.ros2_control_trajectory as ros2_control_module
    import src.plugin.planners.moveit_planner as moveit_planner_module
    from src.core.controllers import Pose3D
    from src.plugin.controllers.moveit import MoveItArmController

    calls = {}
    states = iter(
        [
            SimpleNamespace(name=["arm_lift_joint"], position=[0.1]),
            SimpleNamespace(name=["arm_lift_joint"], position=[0.2]),
        ]
    )
    prepared = [[({"name": "arm", "action": "/arm"}, object())]]

    class FakePlanner:
        config = {"group_name": "whole_body_weighted"}

        def __init__(self, config):
            self.config = config

        def plan_contract(self, contract, *, context=None):
            contracts = calls.setdefault("contracts", [])
            contracts.append(contract)
            if len(contracts) == 1:
                return PlannerResult(success=False, reason="moveit_plan_invalid_motion_plan", failure=FailureInfo(FailureCode.PLANNING_FAILED, retryable=True, backend_code=-2))
            return PlannerResult(success=True, trajectory=prepared)

    class FakeExecutor:
        def __init__(self, hardware_config, joint_state_provider=None):
            self.status = "idle"

        def send_prepared_trajectory_goals(self, trajectory):
            calls["prepared"] = trajectory
            self.status = "succeeded"

        def get_status(self):
            return self.status

        def get_error(self):
            return None

    monkeypatch.setattr(moveit_planner_module, "MoveItManipulationPlanner", FakePlanner)
    monkeypatch.setattr(ros2_control_module, "Ros2ControlController", FakeExecutor)
    controller = MoveItArmController(
        {
            "joint_state_provider": lambda: next(states),
            "moveit_plan_retries": 1,
            "moveit_ik_retries": 1,
            "moveit_retry_delay_s": 0.0,
            "ros2_control": {"controllers": []},
        }
    )

    controller.send_pose_goal(ArmPoseRequest(Pose3D(x=0.3, y=0.0, z=0.8, frame_id='odom'), moveit=MoveItPoseSettings(goal_mode='ik_joint_goal', group_name='whole_body_weighted')))

    assert controller.get_status() == "succeeded"
    assert calls["contracts"][0].start_state_seed == {"arm_lift_joint": pytest.approx(0.1)}
    assert calls["contracts"][1].start_state_seed == {"arm_lift_joint": pytest.approx(0.2)}
    assert calls["prepared"] is prepared


def test_moveit_arm_controller_base_pose_retries_on_omni_path_tolerance(monkeypatch):
    import src.plugin.controllers.ros2_control_trajectory as ros2_control_module
    import src.plugin.planners.moveit_planner as moveit_planner_module
    from src.plugin.controllers.moveit import MoveItArmController

    calls = {}
    states = iter(
        [
            SimpleNamespace(name=["odom_x", "odom_y", "odom_t", "arm_lift_joint"], position=[0.0, 0.0, 0.0, 0.1]),
            SimpleNamespace(name=["odom_x", "odom_y", "odom_t", "arm_lift_joint"], position=[0.1, 0.0, 0.0, 0.2]),
        ]
    )
    prepared = [[({"name": "omni_base", "action": "/omni_base_controller/follow_joint_trajectory"}, object())]]

    class FakePlanner:
        config = {"group_name": "whole_body_weighted"}

        def __init__(self, config):
            self.config = config

        def plan_contract(self, contract, *, context=None):
            calls.setdefault("contracts", []).append(contract)
            return PlannerResult(success=True, trajectory=prepared)

    class FakeExecutor:
        def __init__(self, hardware_config, joint_state_provider=None):
            self.statuses = ["in_progress", "failed", "in_progress", "succeeded"]
            self.error = (
                "ros2_control goal on omni_base ended with status STATUS_ABORTED(6), "
                "error_code PATH_TOLERANCE_VIOLATED(-4), action /omni_base_controller/follow_joint_trajectory"
            )

        def send_prepared_trajectory_goals(self, trajectory):
            calls.setdefault("prepared", []).append(trajectory)

        def get_status(self):
            return self.statuses.pop(0)

        def get_error(self):
            return self.error

        def cancel_goal(self):
            pass

    monkeypatch.setattr(moveit_planner_module, "MoveItManipulationPlanner", FakePlanner)
    monkeypatch.setattr(ros2_control_module, "Ros2ControlController", FakeExecutor)
    controller = MoveItArmController(
        {
            "joint_state_provider": lambda: next(states),
            "whole_body_base_path_tolerance_retries": 1,
            "whole_body_base_retry_delay_s": 0.0,
            "ros2_control": {"controllers": []},
        }
    )

    controller.send_base_pose_goal({"x": 0.5, "y": 0.0, "yaw": 0.0, "frame_id": "odom"})

    assert controller.get_status() == "in_progress"
    assert controller.get_status() == "succeeded"
    assert len(calls["contracts"]) == 2
    assert calls["contracts"][0].start_state_seed["arm_lift_joint"] == pytest.approx(0.1)
    assert calls["contracts"][1].start_state_seed["arm_lift_joint"] == pytest.approx(0.2)
    assert len(calls["prepared"]) == 2


def test_moveit_arm_controller_base_pose_does_not_retry_non_omni_failure(monkeypatch):
    import src.plugin.controllers.ros2_control_trajectory as ros2_control_module
    import src.plugin.planners.moveit_planner as moveit_planner_module
    from src.plugin.controllers.moveit import MoveItArmController

    calls = {}
    prepared = [[({"name": "arm", "action": "/arm_trajectory_controller/follow_joint_trajectory"}, object())]]

    class FakePlanner:
        def __init__(self, config):
            self.config = config

        def plan_contract(self, contract, *, context=None):
            calls.setdefault("contracts", []).append(contract)
            return PlannerResult(success=True, trajectory=prepared)

    class FakeExecutor:
        def __init__(self, hardware_config, joint_state_provider=None):
            self.statuses = ["in_progress", "failed"]

        def send_prepared_trajectory_goals(self, trajectory):
            calls.setdefault("prepared", []).append(trajectory)

        def get_status(self):
            return self.statuses.pop(0)

        def get_error(self):
            return "ros2_control goal on arm ended with error_code PATH_TOLERANCE_VIOLATED(-4)"

        def cancel_goal(self):
            pass

    monkeypatch.setattr(moveit_planner_module, "MoveItManipulationPlanner", FakePlanner)
    monkeypatch.setattr(ros2_control_module, "Ros2ControlController", FakeExecutor)
    controller = MoveItArmController(
        {
            "joint_state_provider": lambda: SimpleNamespace(name=["odom_x", "odom_y", "odom_t"], position=[0.0, 0.0, 0.0]),
            "whole_body_base_path_tolerance_retries": 1,
            "whole_body_base_retry_delay_s": 0.0,
            "ros2_control": {"controllers": []},
        }
    )

    controller.send_base_pose_goal({"x": 0.5, "y": 0.0, "yaw": 0.0, "frame_id": "odom"})

    assert controller.get_status() == "failed"
    assert len(calls["contracts"]) == 1
    assert len(calls["prepared"]) == 1


def test_moveit_send_preserves_complete_start_state_is_diff_false(monkeypatch):
    from moveit_msgs.action import MoveGroup
    from moveit_msgs.msg import MotionPlanRequest

    planner = MoveItManipulationPlanner()
    planner._MoveGroup = MoveGroup
    captured = {}

    def send_goal_async(goal):
        captured["goal"] = goal
        return object()

    planner._move_group_client = SimpleNamespace(send_goal_async=send_goal_async)
    request = MotionPlanRequest()
    request.start_state.joint_state.name = ["odom_x", "odom_y", "odom_t", "arm_lift_joint"]
    request.start_state.joint_state.position = [1.0, 2.0, 0.3, 0.1]
    request.start_state.is_diff = False
    goal_handle = SimpleNamespace(accepted=True, get_result_async=Mock(return_value=object()))
    moveit_result = SimpleNamespace(error_code=SimpleNamespace(val=1), planned_trajectory=None)
    monkeypatch.setattr(
        planner,
        "_spin_until_done",
        Mock(side_effect=[goal_handle, SimpleNamespace(result=moveit_result)]),
    )

    planner._send_move_group_plan(request, convert_trajectory=False)

    assert captured["goal"].request.start_state.is_diff is False


def test_moveit_planner_arm_pose_uses_ik_joint_constraints(monkeypatch):
    from geometry_msgs.msg import PoseStamped

    planner = MoveItManipulationPlanner({"group_name": "arm"})
    monkeypatch.setattr(planner, "_ensure_ros", lambda context, require_move_group=True: True)
    planner._move_group_client = SimpleNamespace(wait_for_server=lambda timeout_sec: True)
    pose = PoseStamped()
    pose.header.frame_id = "base_footprint"
    monkeypatch.setattr(planner, "_target_to_pose_stamped", lambda target, context=None: pose)
    monkeypatch.setattr(
        planner,
        "check_ik",
        Mock(return_value=PlannerResult(success=True, joint_positions={"arm_lift_joint": 0.2, "wrist_ft_sensor_frame_joint": 9.0})),
    )
    captured = {}

    def fake_send(request, *, context=None, metadata=None, allow_collisions=False, options=None):
        captured["request"] = request
        captured["metadata"] = metadata
        captured["allow_collisions"] = allow_collisions
        captured["options"] = options
        return PlannerResult(success=True, trajectory=[[({"name": "arm", "action": "/arm"}, object())]])

    monkeypatch.setattr(planner, "_send_move_group_plan", fake_send)

    result = planner.plan_pose({"pose": [0.4, 0.0, 0.8, 1.0, 0.0, 0.0, 0.0]})

    assert result.success is True
    assert captured["metadata"]["planning_strategy"] == "ik_joint_constraints"
    assert captured["metadata"]["collision_policy"] == "full"
    assert captured["allow_collisions"] is False
    request = captured["request"]
    assert request.group_name == "arm"
    joint_names = [jc.joint_name for jc in request.goal_constraints[0].joint_constraints]
    assert joint_names == ["arm_lift_joint"]


def test_moveit_collision_relaxation_is_pairwise_hand_to_target_only():
    from moveit_msgs.msg import AllowedCollisionEntry, PlanningScene

    planner = MoveItManipulationPlanner(
        {
            "collision_relaxation_links": ["hand_palm_link", "hand_l_distal_link"],
            "collision_relaxation_objects": ["pick_target_collision"],
        }
    )
    current_acm = PlanningScene().allowed_collision_matrix
    current_acm.entry_names = ["base_link", "torso_lift_link"]
    base_row = AllowedCollisionEntry()
    base_row.enabled = [False, True]
    torso_row = AllowedCollisionEntry()
    torso_row.enabled = [True, False]
    current_acm.entry_values = [base_row, torso_row]
    current_acm.default_entry_names = ["octomap"]
    current_acm.default_entry_values = [False]
    planner._get_current_allowed_collision_matrix = Mock(return_value=current_acm)
    goal = SimpleNamespace(
        planning_options=SimpleNamespace(
            planning_scene_diff=PlanningScene(),
        )
    )

    assert planner._relax_goal_collision_matrix(goal, {}) is True

    acm = goal.planning_options.planning_scene_diff.allowed_collision_matrix
    names = list(acm.entry_names)
    assert names == [
        "base_link",
        "torso_lift_link",
        "hand_palm_link",
        "hand_l_distal_link",
        "pick_target_collision",
    ]
    rows = {
        name: list(acm.entry_values[index].enabled)
        for index, name in enumerate(names)
    }
    assert rows["base_link"] == [False, True, False, False, False]
    assert rows["torso_lift_link"] == [True, False, False, False, False]
    assert rows["hand_palm_link"] == [False, False, False, False, True]
    assert rows["hand_l_distal_link"] == [False, False, False, False, True]
    assert rows["pick_target_collision"] == [False, False, True, True, False]
    assert list(acm.default_entry_names) == ["octomap"]
    assert list(acm.default_entry_values) == [False]
    assert planner._last_collision_relaxation_metadata["applied"] is True
    assert planner._last_collision_relaxation_metadata["allowed_pairs"] == [
        ["hand_palm_link", "pick_target_collision"],
        ["hand_l_distal_link", "pick_target_collision"],
    ]
    assert planner._last_collision_relaxation_metadata["current_acm_entry_count"] == 2
    assert planner._last_collision_relaxation_metadata["merged_acm_entry_count"] == 5


def test_moveit_collision_relaxation_does_not_apply_partial_acm_without_current_scene():
    from moveit_msgs.msg import PlanningScene

    planner = MoveItManipulationPlanner(
        {
            "collision_relaxation_links": ["hand_palm_link"],
            "collision_relaxation_objects": ["pick_target_collision"],
        }
    )
    planner._get_current_allowed_collision_matrix = Mock(return_value=None)
    goal = SimpleNamespace(
        planning_options=SimpleNamespace(
            planning_scene_diff=PlanningScene(),
        )
    )

    assert planner._relax_goal_collision_matrix(goal, {}) is False
    assert list(goal.planning_options.planning_scene_diff.allowed_collision_matrix.entry_names) == []
    assert planner._last_collision_relaxation_metadata["applied"] is False
    assert planner._last_collision_relaxation_metadata["reason"] == "current_acm_unavailable"


def test_moveit_get_current_acm_uses_planning_scene_response_without_error_code():
    from moveit_msgs.msg import PlanningScene

    planner = MoveItManipulationPlanner({"planning_scene_module": "moveit_get_planning_scene"})
    current_acm = PlanningScene().allowed_collision_matrix
    current_acm.entry_names = ["base_link"]

    modules = SimpleNamespace(
        has=Mock(return_value=True),
        call=Mock(return_value=SimpleNamespace(scene=SimpleNamespace(allowed_collision_matrix=current_acm))),
    )
    context = SimpleNamespace(modules=modules)

    assert planner._get_current_allowed_collision_matrix(context) is current_acm
    modules.has.assert_called_once_with("moveit_get_planning_scene")
    called_payload = modules.call.call_args.args[1]
    assert called_payload["raw_response"] is True


def test_moveit_collision_relaxation_reports_missing_config():
    from moveit_msgs.msg import PlanningScene

    planner = MoveItManipulationPlanner({})
    goal = SimpleNamespace(
        planning_options=SimpleNamespace(
            planning_scene_diff=PlanningScene(),
        )
    )

    assert planner._relax_goal_collision_matrix(goal, {}) is False
    assert planner._last_collision_relaxation_metadata == {
        "requested": True,
        "applied": False,
        "reason": "missing_collision_relaxation_links",
        "links": [],
        "objects": [],
    }


def test_moveit_arm_controller_planner_config_includes_collision_contract(monkeypatch):
    import src.plugin.controllers.ros2_control_trajectory as ros2_control_module
    import src.plugin.planners.moveit_planner as moveit_planner_module
    from src.plugin.controllers.moveit import MoveItArmController

    calls = {}

    class FakePlanner:
        def __init__(self, config):
            calls["planner_config"] = config

    class FakeExecutor:
        def __init__(self, hardware_config, joint_state_provider=None):
            pass

    monkeypatch.setattr(moveit_planner_module, "MoveItManipulationPlanner", FakePlanner)
    monkeypatch.setattr(ros2_control_module, "Ros2ControlController", FakeExecutor)

    MoveItArmController(
        {
            "module_proxy": object(),
            "collision_relaxation_links": ["hand_palm_link"],
            "collision_relaxation_objects": ["pick_target_collision"],
        }
    )

    assert calls["planner_config"]["collision_relaxation_links"] == ["hand_palm_link"]
    assert calls["planner_config"]["collision_relaxation_objects"] == ["pick_target_collision"]


def test_moveit_arm_controller_collision_mesh_add_remove_uses_scene_diff(monkeypatch):
    import src.plugin.controllers.ros2_control_trajectory as ros2_control_module
    import src.plugin.planners.moveit_planner as moveit_planner_module
    from src.plugin.controllers.moveit import MoveItArmController

    calls = {}

    class FakePlanner:
        def __init__(self, config):
            pass

    class FakeExecutor:
        def __init__(self, hardware_config, joint_state_provider=None):
            pass

    monkeypatch.setattr(moveit_planner_module, "MoveItManipulationPlanner", FakePlanner)
    monkeypatch.setattr(ros2_control_module, "Ros2ControlController", FakeExecutor)
    controller = MoveItArmController({"module_proxy": object()})
    fake_object = SimpleNamespace(id="pick_target_collision")
    monkeypatch.setattr(controller, "_collision_object_from_mesh", lambda mesh, name: fake_object)
    monkeypatch.setattr(controller, "_apply_collision_object", lambda obj: calls.setdefault("applied", []).append(obj) or True)
    monkeypatch.setattr(controller, "_collision_object_remove_msg", lambda name: SimpleNamespace(id=name, operation="REMOVE"))

    controller.add_collision_mesh("/tmp/object.stl", name="pick_target_collision")
    controller.remove_collision_object("pick_target_collision")

    assert calls["applied"][0] is fake_object
    assert calls["applied"][1].id == "pick_target_collision"
    assert calls["applied"][1].operation == "REMOVE"


def test_moveit_arm_controller_applies_persistent_octomap_acm(monkeypatch):
    import src.plugin.controllers.ros2_control_trajectory as ros2_control_module
    import src.plugin.planners.moveit_planner as moveit_planner_module
    from moveit_msgs.msg import AllowedCollisionEntry, PlanningScene
    from src.plugin.controllers.moveit import MoveItArmController

    class FakeExecutor:
        def __init__(self, hardware_config, joint_state_provider=None):
            pass

    monkeypatch.setattr(ros2_control_module, "Ros2ControlController", FakeExecutor)
    controller = MoveItArmController(
        {
            "module_proxy": object(),
            "persistent_octomap_collision_links": ["base_l_drive_wheel_link"],
        }
    )
    current_acm = PlanningScene().allowed_collision_matrix
    current_acm.entry_names = ["base_link", "octomap"]
    row_a = AllowedCollisionEntry()
    row_a.enabled = [False, False]
    row_b = AllowedCollisionEntry()
    row_b.enabled = [False, False]
    current_acm.entry_values = [row_a, row_b]
    monkeypatch.setattr(controller, "_ensure_scene_interfaces", lambda: True)
    monkeypatch.setattr(controller._planner, "_get_current_allowed_collision_matrix", lambda context: current_acm)
    applied = {}
    monkeypatch.setattr(
        controller,
        "_apply_planning_scene_diff",
        lambda **kwargs: applied.setdefault("acm", kwargs.get("allowed_collision_matrix")) or True,
    )

    controller._ensure_persistent_acm()

    acm = applied["acm"]
    names = list(acm.entry_names)
    wheel_index = names.index("base_l_drive_wheel_link")
    octomap_index = names.index("octomap")
    assert acm.entry_values[wheel_index].enabled[octomap_index] is True
    assert acm.entry_values[octomap_index].enabled[wheel_index] is True


def test_moveit_arm_controller_base_pose_goal_uses_whole_body_joint_contract(monkeypatch):
    import src.plugin.controllers.ros2_control_trajectory as ros2_control_module
    import src.plugin.planners.moveit_planner as moveit_planner_module
    from src.plugin.controllers.moveit import MoveItArmController

    calls = {}

    class FakePlanner:
        config = {"group_name": "whole_body_weighted"}

        def __init__(self, config):
            self.config = config

        def plan_contract(self, contract, *, context=None):
            calls["contract"] = contract
            return PlannerResult(success=True, trajectory=[[({"name": "base", "action": "/base"}, object())]])

    class FakeExecutor:
        def __init__(self, hardware_config, joint_state_provider=None):
            self.sent = None
            self.status = "idle"

        def send_prepared_trajectory_goals(self, trajectory):
            self.sent = trajectory
            self.status = "succeeded"

        def send_trajectory_goal(self, trajectory):
            self.sent = trajectory
            self.status = "succeeded"

        def get_status(self):
            return self.status

        def get_error(self):
            return None

        def cancel_goal(self):
            self.status = "idle"

    joint_state = SimpleNamespace(
        name=["odom_x", "odom_y", "odom_t", "arm_lift_joint"],
        position=[0.0, 0.0, 0.0, 0.1],
    )
    monkeypatch.setattr(moveit_planner_module, "MoveItManipulationPlanner", FakePlanner)
    monkeypatch.setattr(ros2_control_module, "Ros2ControlController", FakeExecutor)
    controller = MoveItArmController({"joint_state_provider": lambda: joint_state})

    controller.send_base_pose_goal({"x": 1.0, "y": 2.0, "yaw": 0.5, "frame_id": "odom"})

    contract = calls["contract"]
    assert contract.group_name == "base"
    assert contract.goal_mode == "joint_goal"
    assert contract.joint_positions["odom_x"] == pytest.approx(1.0)
    assert contract.joint_positions["odom_y"] == pytest.approx(2.0)
    assert contract.joint_positions["odom_t"] == pytest.approx(0.5)
    assert contract.joint_positions["arm_lift_joint"] == pytest.approx(0.1)


def test_moveit_arm_controller_publishes_pose_goal_marker(monkeypatch):
    import src.plugin.controllers.ros2_control_trajectory as ros2_control_module
    import src.plugin.planners.moveit_planner as moveit_planner_module
    from src.core.controllers import Pose3D
    from src.plugin.controllers.moveit import MoveItArmController

    published = []

    class FakePublisher:
        def publish(self, msg):
            published.append(msg)

    class FakeClock:
        def now(self):
            from builtin_interfaces.msg import Time

            return SimpleNamespace(to_msg=lambda: Time())

    class FakeNode:
        def create_publisher(self, *args):
            return FakePublisher()

        def get_clock(self):
            return FakeClock()

    class FakePlanner:
        def __init__(self, config):
            self.config = config

    class FakeExecutor:
        def __init__(self, hardware_config, joint_state_provider=None):
            pass

    monkeypatch.setattr(moveit_planner_module, "MoveItManipulationPlanner", FakePlanner)
    monkeypatch.setattr(ros2_control_module, "Ros2ControlController", FakeExecutor)
    controller = MoveItArmController({"ros2_node": FakeNode()})

    controller.publish_pose_goal_marker(Pose3D(x=0.1, y=0.2, z=0.3, frame_id="odom"), label="grasp")

    assert published
    assert published[0].markers[0].header.frame_id == "odom"
    assert published[0].markers[1].text == "grasp"


def test_moveit_planner_cartesian_path_uses_module_proxy(monkeypatch):
    from moveit_msgs.srv import GetCartesianPath
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

    planner = MoveItManipulationPlanner(
        {
            "cartesian_module": "moveit_compute_cartesian_path",
            "ros2_control_controllers": [
                {"name": "arm", "action": "/arm", "joints": ["arm_lift_joint"]},
            ],
        }
    )
    monkeypatch.setattr(planner, "_ensure_ros", lambda context, require_move_group=False: True)
    trajectory = JointTrajectory()
    trajectory.joint_names = ["arm_lift_joint"]
    point = JointTrajectoryPoint()
    point.positions = [0.2]
    trajectory.points = [point]
    response = GetCartesianPath.Response()
    response.error_code.val = 1
    response.fraction = 1.0
    response.solution.joint_trajectory = trajectory
    modules = SimpleNamespace(call=Mock(return_value=response))

    result = planner.plan_cartesian_path(
        {"ee_pose": [0.4, 0.0, 0.8, 1.0, 0.0, 0.0, 0.0]},
        context=SimpleNamespace(modules=modules),
    )

    assert result.success is True
    assert result.trajectory[0][0][0]["name"] == "arm"
    module_name, payload = modules.call.call_args.args[:2]
    assert module_name == "moveit_compute_cartesian_path"
    assert payload["raw_response"] is True


def test_moveit_arm_controller_joint_goal_uses_ros2_control_directly(monkeypatch):
    from src.plugin.controllers import moveit as moveit_module

    calls = {}

    class FakePlanner:
        def __init__(self, config):
            calls["planner_config"] = config

        def plan_pose(self, *args, **kwargs):
            raise AssertionError("joint goals should not call MoveIt planning")

    class FakeExecutor:
        def __init__(self, hardware_config, joint_state_provider=None):
            calls["executor_config"] = hardware_config
            self.status = "in_progress"

        def send_joint_goal(self, joints):
            calls["joints"] = joints

        def get_status(self):
            return self.status

        def get_error(self):
            return None

        def cancel_goal(self):
            pass

    monkeypatch.setattr(moveit_module, "MoveItManipulationPlanner", FakePlanner, raising=False)
    monkeypatch.setattr(moveit_module, "Ros2ControlController", FakeExecutor, raising=False)
    monkeypatch.setattr("src.plugin.planners.moveit_planner.MoveItManipulationPlanner", FakePlanner)
    monkeypatch.setattr("src.plugin.controllers.ros2_control_trajectory.Ros2ControlController", FakeExecutor)

    controller = moveit_module.MoveItArmController({"module_proxy": object()})
    controller.send_joint_goal({"arm_lift_joint": 0.2})

    assert calls["joints"] == {"arm_lift_joint": 0.2}


def test_moveit_arm_controller_sanitizes_prepared_base_batch(monkeypatch):
    import src.plugin.controllers.ros2_control_trajectory as ros2_control_module
    import src.plugin.planners.moveit_planner as moveit_planner_module
    from builtin_interfaces.msg import Duration
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
    from src.plugin.controllers.moveit import MoveItArmController

    captured = {}
    joint_state = SimpleNamespace(
        name=["odom_x", "odom_y", "odom_t"],
        position=[1.0, 2.0, 3.10],
    )

    class FakePlanner:
        def __init__(self, config):
            pass

    class FakeExecutor:
        def __init__(self, hardware_config, joint_state_provider=None):
            self.status = "in_progress"

        def send_prepared_trajectory_goals(self, trajectory):
            captured["prepared"] = trajectory

        def send_trajectory_goal(self, trajectory):
            raise AssertionError("prepared batches should stay prepared")

        def get_status(self):
            return self.status

        def get_error(self):
            return None

        def cancel_goal(self):
            pass

    monkeypatch.setattr(moveit_planner_module, "MoveItManipulationPlanner", FakePlanner)
    monkeypatch.setattr(ros2_control_module, "Ros2ControlController", FakeExecutor)

    controller = MoveItArmController({"joint_state_provider": lambda: joint_state, "sanitize_base_trajectory": True})
    arm_traj = JointTrajectory()
    arm_traj.joint_names = ["arm_lift_joint"]
    arm_point = JointTrajectoryPoint()
    arm_point.positions = [0.2]
    arm_point.time_from_start = Duration(sec=1)
    arm_traj.points = [arm_point]

    base_traj = JointTrajectory()
    base_traj.joint_names = ["odom_x", "odom_y", "odom_t"]
    base_point = JointTrajectoryPoint()
    base_point.positions = [1.6, 2.0, -3.10]
    base_point.velocities = [99.0, 99.0, 99.0]
    base_point.accelerations = [99.0, 99.0, 99.0]
    base_point.time_from_start = Duration(sec=0, nanosec=10_000_000)
    base_traj.points = [base_point]

    controller._send_executor_trajectory([
        [
            ({"name": "arm", "action": "/arm"}, arm_traj),
            (
                {
                    "name": "omni_base",
                    "action": "/base",
                    "group_type": "base_trajectory",
                    "base_trajectory_max_vx_mps": 0.2,
                    "base_trajectory_max_vy_mps": 0.2,
                    "base_trajectory_max_wz_radps": 0.35,
                    "base_trajectory_min_dt_s": 0.12,
                },
                base_traj,
            ),
        ]
    ])

    sent_arm = captured["prepared"][0][0][1]
    sent_base = captured["prepared"][0][1][1]
    assert sent_arm is arm_traj
    assert sent_base.joint_names == ["odom_x", "odom_y", "odom_t"]
    assert sent_base.points[0].positions == pytest.approx([1.0, 2.0, 3.10])
    assert sent_base.points[1].positions[2] > 3.10
    assert sent_base.points[1].velocities == pytest.approx([0.0, 0.0, 0.0])
    assert sent_base.points[1].accelerations == pytest.approx([0.0, 0.0, 0.0])
    duration = sent_base.points[1].time_from_start.sec + sent_base.points[1].time_from_start.nanosec / 1e9
    assert duration >= 3.0


def test_moveit_arm_controller_leaves_prepared_base_batch_when_sanitizer_disabled(monkeypatch):
    import src.plugin.controllers.ros2_control_trajectory as ros2_control_module
    import src.plugin.planners.moveit_planner as moveit_planner_module
    from trajectory_msgs.msg import JointTrajectory
    from src.plugin.controllers.moveit import MoveItArmController

    captured = {}

    class FakePlanner:
        def __init__(self, config):
            pass

    class FakeExecutor:
        def __init__(self, hardware_config, joint_state_provider=None):
            self.status = "in_progress"

        def send_prepared_trajectory_goals(self, trajectory):
            captured["prepared"] = trajectory

        def get_status(self):
            return self.status

        def get_error(self):
            return None

        def cancel_goal(self):
            pass

    monkeypatch.setattr(moveit_planner_module, "MoveItManipulationPlanner", FakePlanner)
    monkeypatch.setattr(ros2_control_module, "Ros2ControlController", FakeExecutor)

    controller = MoveItArmController({"sanitize_base_trajectory": False})
    base_traj = JointTrajectory()
    base_traj.joint_names = ["odom_x", "odom_y", "odom_t"]

    controller._send_executor_trajectory([[({"name": "omni_base", "action": "/base", "group_type": "base_trajectory"}, base_traj)]])

    assert captured["prepared"][0][0][1] is base_traj


def test_moveit_arm_controller_named_goal_succeeds_without_dispatch_when_already_reached(monkeypatch):
    import src.plugin.controllers.ros2_control_trajectory as ros2_control_module
    import src.plugin.planners.moveit_planner as moveit_planner_module
    from src.plugin.controllers.moveit import MoveItArmController

    calls = {}
    joint_state = SimpleNamespace(
        name=["arm_lift_joint", "odom_x", "odom_y", "odom_t"],
        position=[0.105, 1.0, 2.0, 0.3],
    )

    class FakePlanner:
        def __init__(self, config):
            pass

        def plan_contract(self, *args, **kwargs):
            calls["planned"] = True
            return PlannerResult(success=False)

    class FakeExecutor:
        def __init__(self, hardware_config, joint_state_provider=None):
            pass

        def send_joint_goal(self, joints):
            calls["direct"] = joints

        def get_status(self):
            return "idle"

        def get_error(self):
            return None

        def cancel_goal(self):
            pass

    monkeypatch.setattr(moveit_planner_module, "MoveItManipulationPlanner", FakePlanner)
    monkeypatch.setattr(ros2_control_module, "Ros2ControlController", FakeExecutor)
    controller = MoveItArmController(
        {
            "named_target_tolerance": 0.02,
            "joint_state_provider": lambda: joint_state,
            "named_targets": {"ready": {"type": "joint_positions", "joints": {"arm_lift_joint": 0.1}}},
        }
    )

    controller.send_named_goal("ready")

    assert controller.get_status() == "succeeded"
    assert "planned" not in calls
    assert "direct" not in calls


def test_moveit_arm_controller_named_goal_force_execution_uses_direct_ros2_control(monkeypatch):
    import src.plugin.controllers.ros2_control_trajectory as ros2_control_module
    import src.plugin.planners.moveit_planner as moveit_planner_module
    from src.plugin.controllers.moveit import MoveItArmController

    calls = {}
    joint_state = SimpleNamespace(name=["arm_lift_joint"], position=[0.0])

    class FakePlanner:
        def __init__(self, config):
            pass

        def plan_contract(self, *args, **kwargs):
            raise AssertionError("force_execution should skip MoveIt planning")

    class FakeExecutor:
        def __init__(self, hardware_config, joint_state_provider=None):
            self.status = "in_progress"

        def send_joint_goal(self, joints):
            calls["direct"] = joints

        def get_status(self):
            return self.status

        def get_error(self):
            return None

        def cancel_goal(self):
            pass

    monkeypatch.setattr(moveit_planner_module, "MoveItManipulationPlanner", FakePlanner)
    monkeypatch.setattr(ros2_control_module, "Ros2ControlController", FakeExecutor)
    controller = MoveItArmController(
        {
            "joint_state_provider": lambda: joint_state,
            "named_targets": {"ready": {"type": "joint_positions", "joints": {"arm_lift_joint": 0.2}}},
        }
    )

    controller.send_named_goal("ready", force_execution=True)

    assert calls["direct"] == {"arm_lift_joint": 0.2}
    assert controller.get_status() == "in_progress"


def test_moveit_arm_controller_named_goal_plans_joint_target(monkeypatch):
    import src.plugin.controllers.ros2_control_trajectory as ros2_control_module
    import src.plugin.planners.moveit_planner as moveit_planner_module
    from src.plugin.controllers.moveit import MoveItArmController

    calls = {}
    prepared = [[({"name": "arm", "action": "/arm"}, object())]]
    joint_state = SimpleNamespace(
        name=["arm_lift_joint", "odom_x", "odom_y", "odom_t"],
        position=[0.0, 1.0, 2.0, 0.3],
    )

    class FakePlanner:
        def __init__(self, config):
            pass

        def plan_contract(self, contract, *, context=None):
            calls["contract"] = contract
            calls["context"] = context
            return PlannerResult(success=True, trajectory=prepared)

    class FakeExecutor:
        def __init__(self, hardware_config, joint_state_provider=None):
            self.status = "in_progress"

        def send_prepared_trajectory_goals(self, trajectory):
            calls["prepared"] = trajectory

        def send_trajectory_goal(self, trajectory):
            calls["raw"] = trajectory

        def get_status(self):
            return self.status

        def get_error(self):
            return None

        def cancel_goal(self):
            pass

    monkeypatch.setattr(moveit_planner_module, "MoveItManipulationPlanner", FakePlanner)
    monkeypatch.setattr(ros2_control_module, "Ros2ControlController", FakeExecutor)
    controller = MoveItArmController(
        {
            "group_name": "whole_body_weighted",
            "joint_state_provider": lambda: joint_state,
            "named_targets": {"ready": {"type": "joint_positions", "joints": {"arm_lift_joint": 0.2}}},
            "ros2_control": {"controllers": []},
        }
    )

    controller.send_named_goal("ready")

    assert calls["contract"].goal_mode == "joint_goal"
    assert calls["contract"].group_name == "whole_body_weighted"
    assert calls["contract"].joint_positions == {"arm_lift_joint": 0.2}
    assert calls["contract"].constrain_base is True
    assert calls["contract"].base_pose.x == pytest.approx(1.0)
    assert calls["prepared"] is prepared


def test_moveit_arm_controller_pose_goal_plans_then_executes_prepared_batches(monkeypatch):
    import src.plugin.controllers.ros2_control_trajectory as ros2_control_module
    import src.plugin.planners.moveit_planner as moveit_planner_module
    from src.core.controllers import Pose3D
    from src.plugin.controllers.moveit import MoveItArmController

    calls = {}
    prepared = [[({"name": "arm", "action": "/arm"}, object())]]

    class FakePlanner:
        def __init__(self, config):
            calls["planner_config"] = config
            self.config = config

        def plan_contract(self, contract, *, context=None):
            calls["contract"] = contract
            calls["context"] = context
            return PlannerResult(success=True, trajectory=prepared)

    class FakeExecutor:
        def __init__(self, hardware_config, joint_state_provider=None):
            self.status = "in_progress"

        def send_prepared_trajectory_goals(self, trajectory):
            calls["prepared"] = trajectory

        def send_trajectory_goal(self, trajectory):
            calls["raw_trajectory"] = trajectory

        def get_status(self):
            return self.status

        def get_error(self):
            return None

        def cancel_goal(self):
            pass

    monkeypatch.setattr(moveit_planner_module, "MoveItManipulationPlanner", FakePlanner)
    monkeypatch.setattr(ros2_control_module, "Ros2ControlController", FakeExecutor)

    controller = MoveItArmController({"module_proxy": object(), "ros2_control": {"controllers": []}, "joint_state_provider": lambda: SimpleNamespace(name=["arm_lift_joint"], position=[.1])})
    pose = Pose3D(x=0.3, y=0.0, z=0.8, frame_id="base_footprint")
    controller.send_pose_goal(ArmPoseRequest(pose, moveit=MoveItPoseSettings(group_name='arm', planning_strategy='ik_joint_constraints')))

    assert calls["contract"].pose is pose
    assert calls["contract"].group_name == "arm"
    assert calls["contract"].goal_mode == "ik_joint_goal"
    assert calls["prepared"] is prepared
    assert "raw_trajectory" not in calls


def test_ros2_control_shared_node_does_not_spin(monkeypatch):
    import src.plugin.controllers.ros2_control_trajectory as ros2_control_module

    spun = []
    fake_rclpy = SimpleNamespace(spin_once=lambda node, timeout_sec=0.0: spun.append((node, timeout_sec)))
    monkeypatch.setitem(__import__("sys").modules, "rclpy", fake_rclpy)
    controller = ros2_control_module.Ros2ControlController({"ros2_node": object()})

    controller._spin_once(0.01)

    assert spun == []


def test_curobo_planner_builds_solver_and_checks_ik(monkeypatch, tmp_path):
    robot_config = tmp_path / "hsrc_robot.yml"
    robot_config.write_text("robot_cfg: {}\n")
    calls = {}

    class FakeTensor:
        def __init__(self, value):
            self.value = value

        def item(self):
            return self.value

    class FakeDeviceCfg:
        pass

    class FakePose:
        def __init__(self, values):
            self.values = values

        @staticmethod
        def from_list(values, device_cfg=None):
            calls["pose"] = values
            calls["device_cfg"] = device_cfg
            return FakePose(values)

        def unsqueeze(self, dim):
            calls["unsqueeze_dim"] = dim
            return self

    class FakeIKSolverConfig:
        @staticmethod
        def create(robot, device_cfg=None, **options):
            calls["robot"] = robot
            calls["device_cfg_for_config"] = device_cfg
            calls["ik_options"] = options
            return {"ik_config": True}

    class FakeGoalToolPose:
        @staticmethod
        def from_poses(pose_dict, ordered_tool_frames=None):
            calls["goal_pose_dict"] = pose_dict
            calls["ordered_tool_frames"] = ordered_tool_frames
            return {"goal": pose_dict}

    class FakeIKSolver:
        def __init__(self, config):
            calls["solver_config"] = config
            self.joint_names = ["arm_lift_joint", "wrist_flex_joint"]
            self.kinematics = SimpleNamespace(tool_frames=["hand_palm_link"])

        def solve_pose(self, goal_tool_poses, return_seeds=1):
            calls["solved_goal"] = goal_tool_poses
            calls["return_seeds"] = return_seeds
            return SimpleNamespace(
                success=FakeTensor(True),
                score=FakeTensor(0.25),
                solution=[0.1, 0.2],
            )

    def _import(name):
        modules = {
            "curobo": SimpleNamespace(__version__="test"),
            "curobo._src.types.device_cfg": SimpleNamespace(DeviceCfg=FakeDeviceCfg),
            "curobo._src.types.pose": SimpleNamespace(Pose=FakePose),
            "curobo._src.types.tool_pose": SimpleNamespace(GoalToolPose=FakeGoalToolPose),
            "curobo._src.solver.solver_ik": SimpleNamespace(
                IKSolver=FakeIKSolver,
            ),
            "curobo._src.solver.solver_ik_cfg": SimpleNamespace(
                IKSolverCfg=FakeIKSolverConfig,
            ),
            "curobo.config_io": SimpleNamespace(
                load_yaml=lambda _path: {
                    "robot_cfg": {"kinematics": {"ee_link": "hand_palm_link"}},
                    "ik": {"num_seeds": 8},
                    "orchid": {"target_z_offset": 0.1, "orientation_wxyz": [1, 0, 0, 0]},
                }
            ),
        }
        if name not in modules:
            raise ImportError(name)
        return modules[name]

    monkeypatch.setattr("importlib.import_module", _import)
    planner = CuroboManipulationPlanner({"robot_config": str(robot_config), "debug_dir": str(tmp_path)})

    result = planner.check_ik(
        {
            "object_position": (1.0, 0.0, 0.4),
            "base_pose": {"x": 0.5, "y": 0.0, "yaw": 0.0},
        }
    )

    assert result.success is True
    assert result.reason is None
    assert result.joint_positions == {"arm_lift_joint": 0.1, "wrist_flex_joint": 0.2}
    assert result.score == 0.25
    assert calls["ik_options"]["num_seeds"] == 8
    assert calls["ik_options"]["random_seed"] == 0
    assert calls["ik_options"]["use_cuda_graph"] is False
    assert calls["pose"] == [0.5, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0]
    assert calls["ordered_tool_frames"] == ["hand_palm_link"]
    assert "raw_result" not in result.metadata
    assert result.metadata["debug_json"]
    assert json.loads(Path(result.metadata["debug_json"]).read_text())["raw_result"]


def test_curobo_planner_uses_provided_seed_without_reading_joint_states(monkeypatch, tmp_path):
    robot_config = tmp_path / "hsrc_robot.yml"
    robot_config.write_text("robot_cfg: {}\n")

    class FakeTensor:
        def __init__(self, value):
            self.value = value

        def item(self):
            return self.value

    class FakePose:
        @staticmethod
        def from_list(values, device_cfg=None):
            return FakePose()

        def unsqueeze(self, dim):
            return self

    class FakeIKSolverConfig:
        @staticmethod
        def create(robot, device_cfg=None, **options):
            return {}

    class FakeGoalToolPose:
        @staticmethod
        def from_poses(pose_dict, ordered_tool_frames=None):
            return pose_dict

    class FakeIKSolver:
        def __init__(self, config):
            self.joint_names = ["arm_lift_joint", "arm_flex_joint"]
            self.kinematics = SimpleNamespace(tool_frames=["hand_palm_link"])

        def solve_pose(self, goal_tool_poses, return_seeds=1):
            return SimpleNamespace(success=FakeTensor(True), position_error=FakeTensor(0.01), rotation_error=FakeTensor(0.02))

    def _import(name):
        modules = {
            "curobo": SimpleNamespace(__version__="test"),
            "curobo._src.types.device_cfg": SimpleNamespace(DeviceCfg=lambda: object()),
            "curobo._src.types.pose": SimpleNamespace(Pose=FakePose),
            "curobo._src.types.tool_pose": SimpleNamespace(GoalToolPose=FakeGoalToolPose),
            "curobo._src.solver.solver_ik": SimpleNamespace(IKSolver=FakeIKSolver),
            "curobo._src.solver.solver_ik_cfg": SimpleNamespace(IKSolverCfg=FakeIKSolverConfig),
            "curobo.config_io": SimpleNamespace(
                load_yaml=lambda _path: {
                    "robot_cfg": {
                        "kinematics": {
                            "cspace": {"joint_names": ["arm_lift_joint", "arm_flex_joint"]},
                        }
                    },
                }
            ),
        }
        if name not in modules:
            raise ImportError(name)
        return modules[name]

    sdc = SimpleNamespace(get_latest=Mock(side_effect=AssertionError("joint state should be snapped by action")))
    monkeypatch.setattr("importlib.import_module", _import)
    planner = CuroboManipulationPlanner({"robot_config": str(robot_config)})

    result = planner.check_ik(
        {
            "ee_pose": [0.1, 0.0, 0.2, 1.0, 0.0, 0.0, 0.0],
            "joint_state_seed": {
                "name": ["arm_lift_joint", "arm_flex_joint"],
                "position": [0.1, -0.2],
            },
        },
        context=SimpleNamespace(sensor_data_center=sdc),
    )

    assert result.success is True
    assert result.metadata["seed_state"] == {
        "available": True,
        "source": "provided",
        "joints": {"arm_lift_joint": 0.1, "arm_flex_joint": -0.2},
    }
    assert result.metadata["solver_errors"]["position_error"] == 0.01
    assert result.metadata["solver_errors"]["rotation_error"] == 0.02
    sdc.get_latest.assert_not_called()


def test_curobo_planner_transforms_target_pose_matrix_into_candidate_base(monkeypatch, tmp_path):
    robot_config = tmp_path / "hsrc_robot.yml"
    robot_config.write_text("robot_cfg: {}\n")
    calls = {}

    class FakeTensor:
        def __init__(self, value):
            self.value = value

        def item(self):
            return self.value

    class FakePose:
        def __init__(self, values):
            self.values = values

        @staticmethod
        def from_list(values, device_cfg=None):
            calls["pose"] = values
            return FakePose(values)

        def unsqueeze(self, dim):
            return self

    class FakeIKSolverConfig:
        @staticmethod
        def create(robot, device_cfg=None, **options):
            return {}

    class FakeGoalToolPose:
        @staticmethod
        def from_poses(pose_dict, ordered_tool_frames=None):
            return pose_dict

    class FakeIKSolver:
        def __init__(self, config):
            self.joint_names = []
            self.kinematics = SimpleNamespace(tool_frames=["hand_palm_link"])

        def solve_pose(self, goal_tool_poses, return_seeds=1):
            return SimpleNamespace(success=FakeTensor(True))

    def _import(name):
        modules = {
            "curobo": SimpleNamespace(__version__="test"),
            "curobo._src.types.device_cfg": SimpleNamespace(DeviceCfg=lambda: object()),
            "curobo._src.types.pose": SimpleNamespace(Pose=FakePose),
            "curobo._src.types.tool_pose": SimpleNamespace(GoalToolPose=FakeGoalToolPose),
            "curobo._src.solver.solver_ik": SimpleNamespace(
                IKSolver=FakeIKSolver,
            ),
            "curobo._src.solver.solver_ik_cfg": SimpleNamespace(
                IKSolverCfg=FakeIKSolverConfig,
            ),
            "curobo.config_io": SimpleNamespace(
                load_yaml=lambda _path: {
                    "robot_cfg": {},
                    "orchid": {"target_z_offset": 0.1},
                }
            ),
        }
        if name not in modules:
            raise ImportError(name)
        return modules[name]

    monkeypatch.setattr("importlib.import_module", _import)
    planner = CuroboManipulationPlanner({"robot_config": str(robot_config)})
    target_pose = np.eye(4)
    target_pose[:3, :3] = [
        [0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
    target_pose[:3, 3] = [1.0, 0.0, 0.4]

    result = planner.check_ik(
        {
            "target_pose_matrix": target_pose,
            "base_pose": {"x": 0.5, "y": 0.0, "yaw": 0.0},
        }
    )

    assert result.success is True
    assert calls["pose"][:3] == [0.5, 0.0, 0.4]
    assert calls["pose"][3:] == pytest.approx([0.70710678, 0.0, 0.0, 0.70710678], abs=1e-6)


def test_curobo_planner_prefers_robot_cfg_base_frame(monkeypatch, tmp_path):
    robot_config = tmp_path / "hsrc_robot.yml"
    robot_config.write_text("robot_cfg: {}\n")

    def _import(name):
        modules = {
            "curobo": SimpleNamespace(__version__="test"),
            "curobo._src.types.device_cfg": SimpleNamespace(DeviceCfg=lambda: object()),
            "curobo._src.types.pose": SimpleNamespace(Pose=SimpleNamespace),
            "curobo._src.types.tool_pose": SimpleNamespace(GoalToolPose=SimpleNamespace),
            "curobo._src.solver.solver_ik": SimpleNamespace(IKSolver=lambda _cfg: object()),
            "curobo._src.solver.solver_ik_cfg": SimpleNamespace(IKSolverCfg=SimpleNamespace),
            "curobo.config_io": SimpleNamespace(
                load_yaml=lambda _path: {
                    "robot_cfg": {"kinematics": {"base_link": "base_footprint"}},
                    "orchid": {"base_frame": "base_link"},
                }
            ),
        }
        if name not in modules:
            raise ImportError(name)
        return modules[name]

    monkeypatch.setattr("importlib.import_module", _import)
    planner = CuroboManipulationPlanner({"robot_config": str(robot_config)})

    assert planner.robot_base_frame() == "base_footprint"
    assert planner.config_summary()["curobo_base_frame"] == "base_footprint"


def test_curobo_planner_reports_missing_robot_config(monkeypatch):
    monkeypatch.setattr("importlib.import_module", lambda _name: SimpleNamespace())
    planner = CuroboManipulationPlanner({})

    result = planner.check_ik({"ee_pose": [0, 0, 0, 1, 0, 0, 0]})

    assert result.success is False
    assert result.reason == "curobo_ik_not_configured"
    assert "missing robot_config" in result.metadata["config_error"]
