from __future__ import annotations

import math
import sys
import time
from types import ModuleType
from types import SimpleNamespace

import numpy as np
import pytest

from tests.fixtures.native_tree import tick_action, prepare_action, native_actions, action_node
from src.core.context import Context
from src.core.manipulation_types import BasePose2D
from src.core.action import AbstractAction
from src.core.moveit_request import MoveItPlanningSettings
from src.core.planners import PlannerResult
from src.core.world_state import WorldState
from src.plugin.actions import grasp_alignment as ga
from src.plugin.actions.grasp_alignment import (
    ExecuteHsrCartesianGraspMotion,
    EstimateSimpleCylindricalGrasp,
    EstimateSimplePcaGrasp,
    MoveToSelectedGraspBasePose,
    PickTargetWithGraspAlignment,
    SelectHsrGraspMotion,
    SelectFkArmPosture,
    compute_omni_servo_command,
    score_tmc_ik_solution,
    simple_cylindrical_grasp_pose,
    simple_pca_grasp_pose,
)
from src.plugin.actions.manipulation import _AllowCollisionWithConfiguredLinksAction, _MoveJointsRelativeAction


class _FakeSensorDataCenter:
    def __init__(self, pose=None, local_costmap=None, joint_state=None):
        self.pose = pose
        self.local_costmap = local_costmap
        self.joint_state = joint_state

    def get_pose_at(self, source_frame, target_frame):
        return {"mat": np.eye(4)}

    def get_robot_pose_global(self, global_frame=None, base_frame=None):
        return self.pose

    def get_latest(self, sensor_name, timeout_s=0.2):
        if sensor_name == "local_costmap" and self.local_costmap is not None:
            return SimpleNamespace(data=self.local_costmap)
        if sensor_name == "joint_states" and self.joint_state is not None:
            return SimpleNamespace(data=self.joint_state)
        return None

    def peek_latest(self, sensor_name, timeout_s=0.2, keep_active=False):
        return self.get_latest(sensor_name, timeout_s=timeout_s)


class _FakePublisher:
    def __init__(self):
        self.messages = []

    def publish(self, msg):
        self.messages.append(msg)


class _FakeNode:
    def __init__(self):
        self.publishers = {}

    def create_publisher(self, _msg_type, topic, _qos):
        publisher = _FakePublisher()
        self.publishers[topic] = publisher
        return publisher


class _FakeArm:
    def __init__(self):
        self.goals = []
        self.status = "in_progress"
        self.cleared = False
        self.hardware_config = {}
        self.allowed_pairs = []

    def send_joint_goal(self, joints):
        self.goals.append(dict(joints))
        self.status = "in_progress"

    def send_trajectory_goal(self, trajectory):
        self.goals.append({"trajectory": trajectory})
        self.status = "in_progress"

    def get_status(self):
        return self.status

    def get_error(self):
        return None

    def clear_collision_scene(self):
        self.cleared = True

    def allow_collision_pairs(self, pairs):
        self.allowed_pairs.extend(list(pairs))


def _free_grid(width=20, height=20, resolution=0.1, frame_id="odom"):
    return SimpleNamespace(
        header=SimpleNamespace(frame_id=frame_id),
        info=SimpleNamespace(
            width=width,
            height=height,
            resolution=resolution,
            origin=SimpleNamespace(position=SimpleNamespace(x=-1.0, y=-1.0)),
        ),
        data=[0] * (width * height),
    )


def _install_marker_stubs(monkeypatch):
    class Marker:
        ARROW = 0
        TEXT_VIEW_FACING = 9
        ADD = 0

        def __init__(self):
            self.header = SimpleNamespace(frame_id="")
            self.ns = ""
            self.id = 0
            self.type = 0
            self.action = 0
            self.points = []
            self.scale = SimpleNamespace(x=0.0, y=0.0, z=0.0)
            self.color = SimpleNamespace(r=0.0, g=0.0, b=0.0, a=0.0)
            self.pose = SimpleNamespace(
                position=SimpleNamespace(x=0.0, y=0.0, z=0.0),
                orientation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
            )
            self.lifetime = SimpleNamespace(sec=0, nanosec=0)
            self.text = ""

    class MarkerArray:
        def __init__(self):
            self.markers = []

    class Point:
        def __init__(self):
            self.x = 0.0
            self.y = 0.0
            self.z = 0.0

    class Twist:
        def __init__(self):
            self.linear = SimpleNamespace(x=0.0, y=0.0, z=0.0)
            self.angular = SimpleNamespace(x=0.0, y=0.0, z=0.0)

    visualization_msgs = ModuleType("visualization_msgs")
    visualization_msgs_msg = ModuleType("visualization_msgs.msg")
    visualization_msgs_msg.Marker = Marker
    visualization_msgs_msg.MarkerArray = MarkerArray
    geometry_msgs = ModuleType("geometry_msgs")
    geometry_msgs_msg = ModuleType("geometry_msgs.msg")
    geometry_msgs_msg.Point = Point
    geometry_msgs_msg.Twist = Twist
    monkeypatch.setitem(sys.modules, "visualization_msgs", visualization_msgs)
    monkeypatch.setitem(sys.modules, "visualization_msgs.msg", visualization_msgs_msg)
    monkeypatch.setitem(sys.modules, "geometry_msgs", geometry_msgs)
    monkeypatch.setitem(sys.modules, "geometry_msgs.msg", geometry_msgs_msg)
    ga._GRASP_ALIGNMENT_MARKER_PUBLISHERS.clear()


def test_simple_cylindrical_grasp_offsets_centroid_and_keeps_robot_facing_yaw():
    points = np.array(
        [
            [1.0, 0.9, 0.2],
            [1.2, 1.1, 0.6],
            [0.8, 1.0, 0.4],
        ],
        dtype=np.float32,
    )

    pose, metadata = simple_cylindrical_grasp_pose(
        points,
        frame_id="odom",
        robot_pose=BasePose2D(0.0, 0.0, 0.0, "odom"),
        approach_axis="+z",
    )

    centre = np.mean(points, axis=0)
    direction = np.array([centre[0], centre[1], 0.0])
    direction /= np.linalg.norm(direction)
    assert pose[:3, 3] == pytest.approx(centre - .05 * direction)
    assert metadata["frame_id"] == "odom"
    assert metadata["estimated_height_m"] == pytest.approx(0.4)
    expected_yaw = math.atan2(float(np.mean(points[:, 1])), float(np.mean(points[:, 0])))
    assert metadata["yaw_rad"] == pytest.approx(expected_yaw)


def test_estimate_simple_cylindrical_grasp_writes_world_state():
    ws = WorldState()
    ws.set("mug_object_pcd", np.array([[1.0, 0.0, 0.2], [1.0, 0.2, 0.4]], dtype=np.float32))
    ws.set("mug_pcd_frame", "odom")
    context = Context(
        world_state=ws,
        sensor_data_center=_FakeSensorDataCenter(pose={"pose": SimpleNamespace(x=0.0, y=0.0, yaw=0.0)}),
    )

    result = tick_action(EstimateSimpleCylindricalGrasp(target='mug'), context)

    assert result.status == "success"
    assert ws.get("mug_estimated_pick_pose_mat").shape == (4, 4)
    assert ws.get("mug_estimated_pick_pose_frame") == "odom"
    assert ws.get("mug_estimated_width") > 0.0
    assert ws.get("mug_simple_grasp_metadata")["centroid"] == pytest.approx([1.0, 0.1, 0.3])


def test_estimate_simple_grasp_resolves_movable_state_target_only(monkeypatch):
    _install_marker_stubs(monkeypatch)
    ws = WorldState()
    ws.set("custom_mug_pcd", np.array([[1.0, 0.0, 0.2], [1.2, 0.2, 0.6]], dtype=np.float32))
    ws.set_movable_object_state("mug", pose_matrix=np.eye(4), frame_id="odom", object_pcd_key="custom_mug_pcd")
    node = _FakeNode()
    context = Context(
        world_state=ws,
        sensor_data_center=_FakeSensorDataCenter(pose={"pose": SimpleNamespace(x=0.0, y=0.0, yaw=0.0)}),
        ros2_node=node,
    )

    result = tick_action(EstimateSimpleCylindricalGrasp(target='mug'), context)

    assert result.status == "success"
    assert ws.get("mug_simple_grasp_metadata")["centroid"] == pytest.approx([1.1, 0.1, 0.4])
    published = node.publishers["/orchid/grasp_alignment/simple_grasp"].messages[-1]
    assert len(published.markers) >= 3


def test_simple_pca_grasp_cylinder_like_matches_cylindrical_behavior():
    angles = np.linspace(0.0, 2.0 * math.pi, 24, endpoint=False)
    zs = np.array([0.2, 0.5], dtype=float)
    points = np.array([[1.0 + 0.05 * math.cos(a), 0.2 + 0.05 * math.sin(a), z] for a in angles for z in zs], dtype=np.float32)
    robot = BasePose2D(0.0, 0.0, 0.0, "odom")

    cyl_pose, cyl_meta = simple_cylindrical_grasp_pose(points, frame_id="odom", robot_pose=robot, approach_axis="+z")
    pca_pose, pca_meta = simple_pca_grasp_pose(points, frame_id="odom", robot_pose=robot, grasp_mode="auto", approach_axis="+z")

    assert pca_meta["selected_strategy"] == "cylinder_side"
    assert pca_pose == pytest.approx(cyl_pose)
    assert pca_meta["centroid"] == pytest.approx(cyl_meta["centroid"])
    assert pca_meta["estimated_radius_m"] == pytest.approx(cyl_meta["estimated_radius_m"])


def test_estimate_simple_pca_grasp_top_down_writes_canonical_keys(monkeypatch):
    _install_marker_stubs(monkeypatch)
    ws = WorldState()
    xs = np.linspace(0.8, 1.2, 5)
    ys = np.linspace(-0.04, 0.04, 3)
    zs = np.linspace(0.2, 0.36, 3)
    ws.set("bar_object_pcd", np.array([[x, y, z] for x in xs for y in ys for z in zs], dtype=np.float32))
    ws.set("bar_pcd_frame", "odom")
    node = _FakeNode()
    context = Context(
        world_state=ws,
        sensor_data_center=_FakeSensorDataCenter(pose={"pose": SimpleNamespace(x=0.0, y=-0.5, yaw=0.0)}),
        ros2_node=node,
    )

    result = tick_action(EstimateSimplePcaGrasp(target='bar', grasp_mode='top_down'), context)

    assert result.status == "success"
    pose = ws.get("bar_estimated_pick_pose_mat")
    metadata = ws.get("bar_simple_grasp_metadata")
    assert pose[2, 3] == pytest.approx(0.36)
    assert pose[:3, 2] == pytest.approx([0.0, 0.0, -1.0])
    assert abs(float(np.dot(pose[:2, 0], [0.0, 1.0]))) == pytest.approx(1.0)
    assert ws.get("bar_estimated_pick_pose_frame") == "odom"
    assert ws.get("bar_estimated_width") == pytest.approx(0.08)
    assert metadata["selected_strategy"] == "top_down"
    assert node.publishers["/orchid/grasp_alignment/simple_pca_grasp"].messages[-1].markers


def test_select_hsr_grasp_motion_classifies_pca_top_down_from_pose_only():
    ws = WorldState()
    pose = np.eye(4, dtype=float)
    pose[:3, 2] = [0.0, 0.0, -1.0]
    pose[:3, 1] = [0.0, 1.0, 0.0]
    pose[:3, 0] = [1.0, 0.0, 0.0]
    pose[:3, 3] = [1.0, 0.0, 0.4]
    ws.set("bar_estimated_pick_pose_mat", pose)
    context = Context(world_state=ws)

    result = tick_action(SelectHsrGraspMotion(target='bar', pregrasp_mode='horizontal_projected', pregrasp_distance_m=0.12), context)

    assert result.status == "success"
    pregrasp = ws.get("bar_pregrasp_pose")
    assert pregrasp[:3, 3] == pytest.approx([1.0, 0.0, 0.52])
    assert ws.get("bar_grasp_motion_metadata")["grasp_motion_type"] == "vertical_z"


@pytest.mark.parametrize('angle', [0.0, math.pi / 5])
@pytest.mark.parametrize('robot_side', [-1.0, 1.0])
def test_simple_pca_short_axis_side_faces_robot_side(angle, robot_side):
    from src.plugin.actions.hsr_pickup.planning import import_simple_grasp
    long_axis = np.array([math.cos(angle), math.sin(angle), 0.0])
    short_axis = np.array([-math.sin(angle), math.cos(angle), 0.0])
    center = np.array([1.0, 0.0, 0.3])
    points = np.array([center + x * long_axis + y * short_axis
                       for x in np.linspace(-.1, .1, 5) for y in np.linspace(-.05, .05, 3)])
    robot_xy = center[:2] + robot_side * long_axis[:2]
    pose, metadata = simple_pca_grasp_pose(
        points,
        frame_id="odom",
        robot_pose=BasePose2D(float(robot_xy[0]), float(robot_xy[1]), 0.0, "odom"),
        grasp_mode="short_axis_side",
    )

    assert metadata["selected_strategy"] == "short_axis_side"
    assert float(np.dot(pose[:3, 2], -robot_side * long_axis)) == pytest.approx(1.0)
    assert abs(float(np.dot(pose[:3, 0], short_axis))) == pytest.approx(1.0)
    palm, corrected = import_simple_grasp(pose, '+z')
    assert corrected
    assert palm[:3, 2] == pytest.approx(pose[:3, 2])
    assert abs(float(np.dot(palm[:3, 1], short_axis))) == pytest.approx(1.0)
    assert np.linalg.det(palm[:3, :3]) == pytest.approx(1.0)
    assert metadata['estimated_width_m'] == pytest.approx(np.ptp(points @ palm[:3, 1]))
    assert metadata['estimated_width_m'] == pytest.approx(.1)


@pytest.mark.parametrize('mode', ['cylinder_side', 'top_down', 'short_axis_side'])
def test_simple_pca_constructor_respects_explicit_mode(mode):
    action = EstimateSimplePcaGrasp.from_params(target='bar', grasp_mode=mode)
    assert action.grasp_mode == mode
    assert action.param_dict()['grasp_mode'] == mode


def test_fk_top_down_inplane_alignment_prefers_matching_x_axis():
    target = np.eye(4, dtype=float)
    target[:3, 0] = [1.0, 0.0, 0.0]
    target[:3, 1] = [0.0, 1.0, 0.0]
    target[:3, 2] = [0.0, 0.0, -1.0]
    hand_good = target.copy()
    hand_bad = target.copy()
    hand_bad[:3, 0] = [0.0, 1.0, 0.0]
    hand_bad[:3, 1] = [-1.0, 0.0, 0.0]

    good_errors = ga._fk_candidate_errors(hand_good, target, hand_pose_base=hand_good, approach_axis="+z")
    bad_errors = ga._fk_candidate_errors(hand_bad, target, hand_pose_base=hand_bad, approach_axis="+z")

    assert good_errors["inplane_error_rad"] == pytest.approx(0.0)
    assert bad_errors["inplane_error_rad"] == pytest.approx(math.pi / 2.0)
    assert SelectFkArmPosture._score_hand_pose(hand_good, target, errors=good_errors) < SelectFkArmPosture._score_hand_pose(hand_bad, target, errors=bad_errors)


def test_tmc_solution_scoring_prefers_closer_base_and_simple_wrist():
    current = BasePose2D(0.0, 0.0, 0.0, "odom")
    close = {
        "origin_to_base": {"x": 0.2, "y": 0.0, "yaw": 0.0, "frame_id": "odom"},
        "joint_positions": {"wrist_roll_joint": 0.1},
    }
    far = {
        "origin_to_base": {"x": 1.0, "y": 0.0, "yaw": 0.0, "frame_id": "odom"},
        "joint_positions": {"wrist_roll_joint": 0.0},
    }

    assert score_tmc_ik_solution(close, current) < score_tmc_ik_solution(far, current)


def test_parse_tmc_ik_response_returns_base_and_joint_goal():
    result = SimpleNamespace(
        joint_names=["arm_lift_joint", "wrist_roll_joint"],
        joint_positions=[0.2, 0.0],
        origin_to_base=SimpleNamespace(
            position=SimpleNamespace(x=1.0, y=2.0, z=0.0),
            orientation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
        ),
    )

    parsed = ga._parse_tmc_ik_response(SimpleNamespace(ik_results=[result]), default_frame_id="odom")

    assert parsed == [
        {
            "index": 0,
            "joint_positions": {"arm_lift_joint": 0.2, "wrist_roll_joint": 0.0},
            "origin_to_base": {"x": 1.0, "y": 2.0, "yaw": 0.0, "frame_id": "odom"},
        }
    ]


def test_select_hsr_grasp_motion_writes_pregrasp_and_retreat(monkeypatch):
    _install_marker_stubs(monkeypatch)
    ws = WorldState()
    grasp = np.eye(4, dtype=float)
    grasp[0, 3] = 1.0
    grasp[2, 3] = 0.4
    grasp[:3, 2] = [1.0, 0.0, 0.0]
    grasp[:3, 1] = [0.0, 0.0, 1.0]
    grasp[:3, 0] = [0.0, -1.0, 0.0]
    ws.set("mug_estimated_pick_pose_mat", grasp)
    ws.set("mug_simple_grasp_metadata", {"approach_axis": "+z", "frame_id": "odom"})
    ws.set("mug_object_pcd", np.array([[0.96, 0.0, 0.2], [1.04, 0.0, 0.6]], dtype=np.float32))
    ws.set("mug_surround_pcd", np.array([[0.82, 0.02, 0.4], [1.4, 0.4, 0.4]], dtype=np.float32))
    node = _FakeNode()
    context = Context(world_state=ws, ros2_node=node)

    result = tick_action(SelectHsrGraspMotion(target='mug', pregrasp_distance_m=0.12), context)

    assert result.status == "success"
    pregrasp = ws.get("mug_pregrasp_pose")
    retreat = ws.get("mug_retreat_pose")
    assert pregrasp[:3, 3] == pytest.approx([0.88, 0.0, 0.4])
    assert retreat[:3, 3] == pytest.approx([0.88, 0.0, 0.4])
    metadata = ws.get("mug_grasp_motion_metadata")
    assert metadata["scene"]["suggested_insertion_m"] >= 0.04
    assert node.publishers["/orchid/grasp_alignment/grasp_motion"].messages[-1].markers


def test_select_hsr_grasp_motion_horizontal_projected_keeps_side_pregrasp_height():
    ws = WorldState()
    grasp = np.eye(4, dtype=float)
    grasp[0, 3] = 1.0
    grasp[2, 3] = 0.4
    approach = np.array([1.0, 0.0, 1.0], dtype=float)
    approach /= np.linalg.norm(approach)
    grasp[:3, 2] = approach
    ws.set("mug_estimated_pick_pose_mat", grasp)
    ws.set("mug_simple_grasp_metadata", {"approach_axis": "+z", "frame_id": "odom"})
    context = Context(world_state=ws)

    result = tick_action(SelectHsrGraspMotion(target='mug', pregrasp_mode='horizontal_projected', pregrasp_distance_m=0.12), context)

    assert result.status == "success"
    pregrasp = ws.get("mug_pregrasp_pose")
    assert pregrasp[2, 3] == pytest.approx(grasp[2, 3])
    assert pregrasp[0, 3] == pytest.approx(0.88)
    metadata = ws.get("mug_grasp_motion_metadata")
    assert metadata["grasp_motion_type"] == "horizontal_projected"


def test_select_hsr_grasp_motion_uses_projected_axis_for_support_depth():
    ws = WorldState()
    grasp = np.eye(4, dtype=float)
    grasp[0, 3] = 1.0
    grasp[2, 3] = 0.4
    approach = np.array([1.0, 0.0, 1.0], dtype=float)
    approach /= np.linalg.norm(approach)
    grasp[:3, 2] = approach
    ws.set("mug_estimated_pick_pose_mat", grasp)
    ws.set("mug_simple_grasp_metadata", {"approach_axis": "+z", "frame_id": "odom"})
    ws.set("mug_object_pcd", np.array([[0.98, 0.0, 0.35], [1.04, 0.0, 0.46]], dtype=np.float32))
    ws.set("mug_surround_pcd", np.array([[0.70, 0.02, 0.4], [1.35, 0.40, 0.4]], dtype=np.float32))
    context = Context(world_state=ws)

    result = tick_action(SelectHsrGraspMotion(target='mug', pregrasp_mode='horizontal_projected', pregrasp_distance_m=0.12), context)

    assert result.status == "success"
    metadata = ws.get("mug_grasp_motion_metadata")
    assert metadata["depth_axis"] == pytest.approx([1.0, 0.0, 0.0])
    assert metadata["scene"]["depth_axis"] == pytest.approx([1.0, 0.0, 0.0])
    assert metadata["scene"]["support_depth_m"] == pytest.approx(0.30, abs=0.01)
    assert metadata["scene"]["is_vertical_grasp"] is False
    assert metadata["scene"]["support_edge_to_grasp_m"] == pytest.approx(0.30, abs=0.01)
    assert metadata["scene"]["required_reach_min_m"] == pytest.approx(0.29, abs=0.02)
    assert metadata["scene"]["min_reach_m"] == pytest.approx(metadata["scene"]["required_reach_min_m"])


def test_projected_support_edge_depth_increases_required_reach():
    grasp = np.eye(4, dtype=float)
    grasp[0, 3] = 1.0
    grasp[2, 3] = 0.4
    grasp[:3, 2] = [1.0, 0.0, 0.0]
    obj = np.array([[0.98, 0.0, 0.35], [1.04, 0.0, 0.46]], dtype=np.float64)

    shallow = ga._estimate_grasp_scene_parameters(
        grasp,
        approach_axis="+z",
        object_points=obj,
        surround_points=np.array([[0.86, 0.0, 0.36], [1.10, 0.2, 0.36]], dtype=np.float64),
        depth_axis=np.array([1.0, 0.0, 0.0], dtype=np.float64),
    )
    deep = ga._estimate_grasp_scene_parameters(
        grasp,
        approach_axis="+z",
        object_points=obj,
        surround_points=np.array([[0.55, 0.0, 0.36], [1.10, 0.2, 0.36]], dtype=np.float64),
        depth_axis=np.array([1.0, 0.0, 0.0], dtype=np.float64),
    )

    assert shallow["support_depth_m"] == pytest.approx(0.14, abs=0.01)
    assert deep["support_depth_m"] == pytest.approx(0.45, abs=0.01)
    assert deep["required_reach_min_m"] > shallow["required_reach_min_m"] + 0.08


def test_select_hsr_grasp_motion_horizontal_projected_top_uses_vertical_pregrasp():
    ws = WorldState()
    grasp = np.eye(4, dtype=float)
    grasp[2, 3] = 0.4
    grasp[:3, 2] = [0.0, 0.0, -1.0]
    ws.set("mug_estimated_pick_pose_mat", grasp)
    ws.set("mug_simple_grasp_metadata", {"approach_axis": "+z", "frame_id": "odom"})
    context = Context(world_state=ws)

    result = tick_action(SelectHsrGraspMotion(target='mug', pregrasp_mode='horizontal_projected', pregrasp_distance_m=0.12), context)

    assert result.status == "success"
    pregrasp = ws.get("mug_pregrasp_pose")
    assert pregrasp[:2, 3] == pytest.approx([0.0, 0.0])
    assert pregrasp[2, 3] == pytest.approx(0.52)
    metadata = ws.get("mug_grasp_motion_metadata")
    assert metadata["grasp_motion_type"] == "vertical_z"
    assert metadata["scene"]["is_vertical_grasp"] is True
    assert metadata["scene"]["projected_depth_axis"] == pytest.approx([0.0, 0.0, -1.0])
    assert metadata["scene"]["required_lift_m"] == pytest.approx(pregrasp[2, 3])


def test_move_to_selected_grasp_base_pose_delegates_to_whole_body_base_move(monkeypatch):
    calls = {}

    class FakeMoveBaseWithWholeBody(ga.AsyncAction):
        def __init__(self, **kwargs):
            super().__init__()
            self.moveit_group = kwargs.get("moveit_group")

        def start(self, context):
            calls["init"] = {"x": self.x, "y": self.y, "yaw": self.yaw, "frame_id": self.frame_id, "moveit_group": self.moveit_group}
            calls["started"] = True

        def check_result(self, context):
            return ga.ActionResult(status="success", reason="base_done", payload={"pose": calls["init"]})

        def cancel(self, context):
            calls["cancelled"] = True

    import src.plugin.actions.navigation as navigation

    monkeypatch.setattr(navigation, "MoveBaseWithWholeBody", FakeMoveBaseWithWholeBody)
    ws = WorldState()
    ws.set("mug_fk_base_pose", {"x": 0.3, "y": 0.1, "yaw": 0.2, "frame_id": "odom"})
    context = Context(world_state=ws)

    action = MoveToSelectedGraspBasePose(target="mug", source="fk")
    result = tick_action(action, context)

    assert result.status == "success"
    assert calls["started"] is True
    assert calls["init"]["x"] == pytest.approx(0.3)
    assert calls["init"]["moveit_group"] == "whole_body_weighted"


def test_pick_target_with_grasp_alignment_builds_expected_sequence(monkeypatch):
    ws = WorldState()
    ws.set("mug_object_pcd", np.array([[1.0, 0.0, 0.2], [1.1, 0.1, 0.5]], dtype=np.float32))
    ws.set("mug_surround_pcd", np.array([[0.7, 0.0, 0.2], [1.4, 0.2, 0.4]], dtype=np.float32))
    context = Context(
        world_state=ws,
        sensor_data_center=_FakeSensorDataCenter(pose={"pose": SimpleNamespace(x=0.2, y=-0.1, yaw=0.3)}),
        hardware={"odom": "odom"},
    )

    action = PickTargetWithGraspAlignment(
        target="mug",
        final_approach_mode="base_servo",
        force_N=0.6,
        width=0.12,
        debug_visualization=False,
    )
    prepare_action(action, context)

    assert action_node(action, context) is not None
    names = [type(child).__name__ for child in native_actions(action, context)]
    assert names == [
        "_ClearCollisionSceneAction",
        "_AddCollisionMeshAction",
        "_AddCollisionMeshAction",
        "SelectHsrGraspMotion",
        "_AllowCollisionWithConfiguredLinksAction",
        "SelectFkArmPosture",
        "MoveToSelectedGraspBasePose",
        "SelectFkArmPosture",
        "ReleaseObjectAction",
        "ExecuteHsrCartesianGraspMotion",
        "_TopDownPreGraspAdjustAction",
        "GraspObjectAction",
        "_AllowCollisionPairsAction",
        "_AllowCollisionWithConfiguredLinksAction",
        "_MoveJointsRelativeAction",
        "OmniBaseServoToPose",
        "MoveArmToState",
        "_ClearCollisionSceneAction",
        "_ClearOctomapAction",
    ]
    assert native_actions(action, context)[5].execute_arm is False
    assert native_actions(action, context)[7].execute_arm is True
    assert native_actions(action, context)[10].target == "mug"
    assert native_actions(action, context)[9].mode == "base_servo"
    assert native_actions(action, context)[13].object_names == ["pick_target_collision", "<octomap>", "pick_surround_collision"]
    assert native_actions(action, context)[14].joint_deltas == {"arm_lift_joint": pytest.approx(0.05)}
    assert native_actions(action, context)[15].target is None
    assert native_actions(action, context)[15].frame_id == "odom"
    assert native_actions(action, context)[16].state == "go"
    assert ws.get("last_pick_start_base_pose") == {"x": 0.2, "y": -0.1, "yaw": 0.3, "frame_id": "odom"}
    assert ws.get("mug_pick_start_base_pose") == {"x": 0.2, "y": -0.1, "yaw": 0.3, "frame_id": "odom"}


def test_pick_target_with_grasp_alignment_uses_movable_pcd_keys():
    ws = WorldState()
    ws.set("custom_object_pcd", np.array([[1.0, 0.0, 0.2], [1.1, 0.1, 0.5]], dtype=np.float32))
    ws.set("custom_surround_pcd", np.array([[0.7, 0.0, 0.2], [1.4, 0.2, 0.4]], dtype=np.float32))
    ws.set_movable_object_state(
        "mug",
        pose_matrix=np.eye(4),
        frame_id="odom",
        object_pcd_key="custom_object_pcd",
        surround_pcd_key="custom_surround_pcd",
    )
    context = Context(
        world_state=ws,
        sensor_data_center=_FakeSensorDataCenter(pose={"pose": SimpleNamespace(x=0.0, y=0.0, yaw=0.0)}),
    )

    action = PickTargetWithGraspAlignment(target="mug")
    prepare_action(action, context)

    assert action_node(action, context) is not None
    assert native_actions(action, context)[1].pcd is ws.get("custom_object_pcd")
    assert native_actions(action, context)[2].pcd is ws.get("custom_surround_pcd")


def test_pick_target_with_grasp_alignment_missing_object_pcd_fails_cleanly():
    ws = WorldState()
    context = Context(world_state=ws)

    result = tick_action(PickTargetWithGraspAlignment(target='mug'), context)

    assert result.status == "failure"
    assert result.reason == "missing_object_pcd:mug_object_pcd"


def test_pick_target_with_grasp_alignment_without_surround_skips_surround_mesh():
    ws = WorldState()
    ws.set("mug_object_pcd", np.array([[1.0, 0.0, 0.2], [1.1, 0.1, 0.5]], dtype=np.float32))
    context = Context(
        world_state=ws,
        sensor_data_center=_FakeSensorDataCenter(pose={"pose": SimpleNamespace(x=0.0, y=0.0, yaw=0.0)}),
    )

    action = PickTargetWithGraspAlignment(target="mug")
    prepare_action(action, context)

    assert action_node(action, context) is not None
    names = [type(child).__name__ for child in native_actions(action, context)]
    assert names.count("_AddCollisionMeshAction") == 2
    assert action._has_surround is False
    assert "_AllowCollisionWithConfiguredLinksAction" in names
    assert names[-5:] == ["_MoveJointsRelativeAction", "OmniBaseServoToPose", "MoveArmToState", "_ClearCollisionSceneAction", "_ClearOctomapAction"]


def test_clear_octomap_action_uses_module_proxy():
    from src.plugin.actions.manipulation import _ClearOctomapAction

    calls = []

    class FakeModules:
        def has(self, name):
            return name == "moveit_clear_octomap"

        def call(self, name, payload, timeout=None):
            calls.append((name, payload, timeout))
            return {}

    context = Context(modules=FakeModules(), arm=_FakeArm())
    context.run_in_context_loop = lambda func, callback=None, error_callback=None, **kwargs: callback(func())

    action = _ClearOctomapAction(timeout_s=1.5)
    result = tick_action(action, context)

    assert result.status == "success"
    assert calls == [("moveit_clear_octomap", {}, pytest.approx(1.5))]


def test_clear_octomap_action_missing_module_is_best_effort():
    from src.plugin.actions.manipulation import _ClearOctomapAction

    class FakeModules:
        def has(self, name):
            return False

        def call(self, name, payload, timeout=None):
            raise AssertionError("missing octomap module should not be called")

    context = Context(modules=FakeModules(), arm=_FakeArm())
    context.run_in_context_loop = lambda func, callback=None, error_callback=None, **kwargs: callback(func())

    result = tick_action(_ClearOctomapAction(), context)

    assert result.status == "success"


def test_pick_target_with_grasp_alignment_relative_back_undock():
    ws = WorldState()
    ws.set("mug_object_pcd", np.array([[1.0, 0.0, 0.2], [1.1, 0.1, 0.5]], dtype=np.float32))
    context = Context(
        world_state=ws,
        sensor_data_center=_FakeSensorDataCenter(pose={"pose": SimpleNamespace(x=0.0, y=0.0, yaw=0.0)}),
    )

    action = PickTargetWithGraspAlignment(target="mug", undock_mode="relative_back", undock_distance_m=0.42, post_grasp_lift_m=0.0)
    prepare_action(action, context)

    assert action_node(action, context) is not None
    names = [type(child).__name__ for child in native_actions(action, context)]
    assert "_MoveJointsRelativeAction" not in names
    undock = native_actions(action, context)[-4]
    assert type(undock).__name__ == "MoveBaseWithWholeBody"
    assert undock.return_to_pick_start is False
    assert undock.relative_x == pytest.approx(-0.42)


def test_allow_collision_with_configured_links_builds_all_pairs():
    arm = _FakeArm()
    arm.hardware_config = {"collision_relaxation_links": ["hand_palm_link", "wrist_roll_link"]}
    context = Context(arm=arm)
    action = _AllowCollisionWithConfiguredLinksAction(["pick_target_collision", "pick_surround_collision"])

    action.start(context)
    for _ in range(50):
        result = action.check_result(context)
        if result.status == "success":
            break
        time.sleep(0.01)
    else:
        pytest.fail("configured-link ACM action did not complete")

    assert arm.allowed_pairs == [
        ("hand_palm_link", "pick_target_collision"),
        ("hand_palm_link", "pick_surround_collision"),
        ("wrist_roll_link", "pick_target_collision"),
        ("wrist_roll_link", "pick_surround_collision"),
    ]
    assert action.pairs == arm.allowed_pairs


def test_allow_collision_with_configured_links_missing_links_is_success():
    arm = _FakeArm()
    arm.hardware_config = {}
    context = Context(arm=arm)
    action = _AllowCollisionWithConfiguredLinksAction(["pick_target_collision"])

    action.start(context)
    for _ in range(50):
        result = action.check_result(context)
        if result.status == "success":
            break
        time.sleep(0.01)
    else:
        pytest.fail("configured-link ACM action did not complete")

    assert arm.allowed_pairs == []


def test_move_joints_relative_action_uses_latest_joint_state():
    arm = _FakeArm()
    joint_state = SimpleNamespace(
        name=["arm_lift_joint", "head_pan_joint"],
        position=[0.20, 0.10],
    )
    context = Context(
        arm=arm,
        sensor_data_center=_FakeSensorDataCenter(joint_state=joint_state),
    )
    action = _MoveJointsRelativeAction({"arm_lift_joint": 0.05, "head_pan_joint": -0.2})

    first = tick_action(action, context)
    for _ in range(50):
        if arm.goals:
            break
        time.sleep(0.01)

    assert first.status in {"running", "success"}
    assert arm.goals[-1] == {"arm_lift_joint": pytest.approx(0.25), "head_pan_joint": pytest.approx(-0.10)}
    arm.status = "succeeded"
    assert tick_action(action, context).status == "success"


def test_tmc_ik_defaults_to_pregrasp_and_records_failure_metadata(monkeypatch):
    class FakeModules:
        def has(self, name):
            return True

        def call(self, name, payload, timeout=None):
            return SimpleNamespace(ik_results=[])

    ws = WorldState()
    pregrasp = np.eye(4, dtype=float)
    pregrasp[0, 3] = 0.8
    simple = np.eye(4, dtype=float)
    simple[0, 3] = 1.0
    ws.set("mug_pregrasp_pose", pregrasp)
    ws.set("mug_estimated_pick_pose_mat", simple)
    ws.set("mug_pcd_frame", "odom")
    context = Context(
        world_state=ws,
        modules=FakeModules(),
        sensor_data_center=_FakeSensorDataCenter(pose={"pose": SimpleNamespace(x=0.0, y=0.0, yaw=0.0)}),
    )
    monkeypatch.setattr(ga, "_build_tmc_ik_request", lambda *args, **kwargs: object())

    result = tick_action(ga.SolveTmcGraspIk(target='mug'), context)

    assert result.status == "failure"
    assert result.reason == "tmc_ik_no_solutions"
    assert ws.get("mug_tmc_ik_request_metadata")["pose_key"] == "mug_pregrasp_pose"
    assert result.payload["request"]["initial_origin_to_base"]["frame_id"] == "odom"


def test_fk_base_pose_alignment_uses_actual_hand_transform():
    expected_base = BasePose2D(0.6, 0.9, 0.4, "odom")
    hand_pose = np.eye(4, dtype=float)
    hand_pose[0, 3] = 0.55
    hand_pose[1, 3] = 0.08
    hand_pose[:3, 2] = [1.0, 0.0, 0.0]
    hand_pose[:3, 1] = [0.0, 0.0, 1.0]
    hand_pose[:3, 0] = [0.0, -1.0, 0.0]
    grasp_pose = ga._resolve_hand_pose_in_world(expected_base, hand_pose)

    actual = ga._base_pose_from_hand_alignment(grasp_pose, hand_pose, frame_id="odom", approach_axis="+z")

    assert actual.x == pytest.approx(expected_base.x)
    assert actual.y == pytest.approx(expected_base.y)
    assert actual.yaw == pytest.approx(expected_base.yaw)


def test_fk_base_pose_alignment_regression_for_rotated_grasp():
    hand_pose = np.eye(4, dtype=float)
    hand_pose[0, 3] = 0.60
    hand_pose[1, 3] = 0.08
    hand_pose[:3, 2] = [1.0, 0.0, 0.0]
    hand_pose[:3, 1] = [0.0, 0.0, 1.0]
    hand_pose[:3, 0] = [0.0, -1.0, 0.0]
    expected_base = BasePose2D(0.62, 0.96, 0.55, "odom")
    grasp_pose = ga._resolve_hand_pose_in_world(expected_base, hand_pose)

    actual = ga._base_pose_from_hand_alignment(grasp_pose, hand_pose, frame_id="odom", approach_axis="+z")

    assert actual.x == pytest.approx(expected_base.x)
    assert actual.y == pytest.approx(expected_base.y)
    assert actual.yaw == pytest.approx(expected_base.yaw)
    old_shortcut_y = float(grasp_pose[1, 3]) - math.sin(ga._approach_yaw_from_pose(grasp_pose, "+x")) * 0.55
    assert abs(old_shortcut_y - expected_base.y) > 0.1


def test_fk_pregrasp_base_alignment_keeps_approach_standoff():
    hand_pose = np.eye(4, dtype=float)
    hand_pose[0, 3] = 0.55
    hand_pose[1, 3] = 0.0
    hand_pose[2, 3] = 0.30
    hand_pose[:3, 2] = [1.0, 0.0, 0.0]
    hand_pose[:3, 1] = [0.0, 0.0, 1.0]
    hand_pose[:3, 0] = [0.0, -1.0, 0.0]
    pregrasp = np.eye(4, dtype=float)
    pregrasp[0, 3] = 0.88
    pregrasp[1, 3] = 0.20
    pregrasp[2, 3] = 0.30
    pregrasp[:3, 2] = [1.0, 0.0, 0.0]
    pregrasp[:3, 1] = [0.0, 0.0, 1.0]
    pregrasp[:3, 0] = [0.0, -1.0, 0.0]

    base = ga._base_pose_from_fk_pregrasp_alignment(pregrasp, hand_pose, frame_id="odom", approach_axis="+z", approach_standoff_m=0.12)
    resolved = ga._resolve_hand_pose_in_world(base, hand_pose)

    assert resolved[0, 3] == pytest.approx(0.76)
    assert resolved[1, 3] == pytest.approx(pregrasp[1, 3])
    assert resolved[2, 3] == pytest.approx(pregrasp[2, 3])
    assert ga._approach_alignment_error(resolved, pregrasp, "+z") == pytest.approx(0.0)


def test_fk_candidate_gate_rejects_z_lateral_and_approach_errors():
    allowed = ga._fk_candidate_gate(
        {"z_error_m": 0.01, "lateral_error_m": 0.02, "approach_error_rad": 0.1, "reach_m": 0.5},
        max_z_error_m=0.03,
        max_lateral_error_m=0.06,
        max_approach_error_rad=0.25,
        min_reach_m=0.25,
        max_reach_m=0.85,
    )
    rejected = ga._fk_candidate_gate(
        {"z_error_m": 0.15, "lateral_error_m": 0.10, "approach_error_rad": 0.5, "reach_m": 0.5},
        max_z_error_m=0.03,
        max_lateral_error_m=0.06,
        max_approach_error_rad=0.25,
        min_reach_m=0.25,
        max_reach_m=0.85,
    )

    assert allowed["allowed"] is True
    assert rejected["allowed"] is False
    assert rejected["reasons"] == ["z_error", "lateral_error", "approach_error"]


def test_fk_score_reach_preference_is_explicit_tradeoff():
    pose = np.eye(4, dtype=float)
    target = np.eye(4, dtype=float)
    base_hand_good_xyz = np.eye(4, dtype=float)
    base_hand_good_xyz[0, 3] = 0.55
    base_hand_bad_x = np.eye(4, dtype=float)
    base_hand_bad_x[0, 3] = 0.85

    good_zyaw_bad_x = SelectFkArmPosture._score_hand_pose(
        pose,
        target,
        hand_pose_base=base_hand_bad_x,
        errors={"z_error_m": 0.005, "lateral_error_m": 0.005, "approach_error_rad": 0.01, "reach_m": 0.85},
        scene={"min_reach_m": 0.25, "max_reach_m": 0.85},
    )
    bad_zyaw_good_x = SelectFkArmPosture._score_hand_pose(
        pose,
        target,
        hand_pose_base=base_hand_good_xyz,
        errors={"z_error_m": 0.025, "lateral_error_m": 0.05, "approach_error_rad": 0.20, "reach_m": 0.55},
        scene={"min_reach_m": 0.25, "max_reach_m": 0.85},
    )

    # With the configured reach coefficient 10, shortening reach can outweigh
    # smaller within-gate geometric errors. Hard acceptance is tested separately.
    assert bad_zyaw_good_x < good_zyaw_bad_x


def test_fk_score_does_not_prefer_farther_reach_inside_envelope():
    pose = np.eye(4, dtype=float)
    target = np.eye(4, dtype=float)
    enough_reach = np.eye(4, dtype=float)
    enough_reach[0, 3] = 0.44
    far_reach = np.eye(4, dtype=float)
    far_reach[0, 3] = 0.68
    errors = {"z_error_m": 0.005, "lateral_error_m": 0.005, "approach_error_rad": 0.01, "reach_m": 0.0}
    scene = {"min_reach_m": 0.40, "max_reach_m": 0.82, "lift_reach_threshold_m": 0.7}

    close_score = SelectFkArmPosture._score_hand_pose(
        pose,
        target,
        joints={"arm_lift_joint": 0.0, "arm_flex_joint": -0.4},
        hand_pose_base=enough_reach,
        errors={**errors, "reach_m": 0.44},
        scene=scene,
    )
    far_score = SelectFkArmPosture._score_hand_pose(
        pose,
        target,
        joints={"arm_lift_joint": 0.0, "arm_flex_joint": -0.4},
        hand_pose_base=far_reach,
        errors={**errors, "reach_m": 0.68},
        scene=scene,
    )

    assert close_score <= far_score


def test_fk_score_penalizes_too_short_reach_and_low_lift_for_long_reach():
    pose = np.eye(4, dtype=float)
    target = np.eye(4, dtype=float)
    short = np.eye(4, dtype=float)
    short[0, 3] = 0.42
    long = np.eye(4, dtype=float)
    long[0, 3] = 0.66
    errors = {"z_error_m": 0.005, "lateral_error_m": 0.005, "approach_error_rad": 0.01}
    scene = {
        "min_reach_m": 0.58,
        "max_reach_m": 0.78,
        "lift_reach_threshold_m": 0.42,
        "lift_at_short_reach_m": 0.0,
        "lift_per_reach_m": 1.0,
    }

    short_score = SelectFkArmPosture._score_hand_pose(
        pose,
        target,
        joints={"arm_lift_joint": 0.25, "arm_flex_joint": -0.5},
        hand_pose_base=short,
        errors={**errors, "reach_m": 0.42},
        scene=scene,
    )
    long_low_lift = SelectFkArmPosture._score_hand_pose(
        pose,
        target,
        joints={"arm_lift_joint": 0.02, "arm_flex_joint": -0.5},
        hand_pose_base=long,
        errors={**errors, "reach_m": 0.66},
        scene=scene,
    )
    long_good_lift = SelectFkArmPosture._score_hand_pose(
        pose,
        target,
        joints={"arm_lift_joint": 0.25, "arm_flex_joint": -0.5},
        hand_pose_base=long,
        errors={**errors, "reach_m": 0.66},
        scene=scene,
    )

    assert long_good_lift < short_score
    assert long_good_lift < long_low_lift


def test_fk_score_vertical_grasp_requires_pregrasp_z_lift():
    pose = np.eye(4, dtype=float)
    target = np.eye(4, dtype=float)
    hand = np.eye(4, dtype=float)
    hand[0, 3] = 0.45
    errors = {"z_error_m": 0.005, "lateral_error_m": 0.005, "approach_error_rad": 0.01, "reach_m": 0.45}
    scene = {"is_vertical_grasp": True, "required_lift_m": 0.30, "min_reach_m": 0.30, "max_reach_m": 0.70}

    low = SelectFkArmPosture._score_hand_pose(
        pose,
        target,
        joints={"arm_lift_joint": 0.05, "arm_flex_joint": -0.5},
        hand_pose_base=hand,
        errors=errors,
        scene=scene,
    )
    enough = SelectFkArmPosture._score_hand_pose(
        pose,
        target,
        joints={"arm_lift_joint": 0.30, "arm_flex_joint": -0.5},
        hand_pose_base=hand,
        errors=errors,
        scene=scene,
    )

    assert enough < low


@pytest.fixture(autouse=True)
def _inline_fk_selection_for_contract_tests(request, monkeypatch):
    # Existing selector tests assert outcomes, not worker scheduling. Real async
    # lifecycle and cancellation are tested in test_serial_fk_actions.py.
    if "fk_posture_selector" not in request.node.name:
        return
    def run(self, work, *args, callback=None, error_callback=None, **kwargs):
        try:
            result = work(*args, **kwargs)
        except Exception as exc:
            error_callback(exc)
        else:
            callback(result)
    monkeypatch.setattr(Context, "run_in_context_loop", run)
    monkeypatch.setattr(ga, "_fk_frame_matrix", lambda *a: np.eye(4))


def test_fk_posture_selector_uses_pregrasp_and_diagnostic_state_validity(monkeypatch):
    class FakePlanner:
        def __init__(self):
            self.validity_calls = 0
            self.fk_calls = 0

        def check_state_validity(self, joint_state, *, context=None, group_name=None):
            self.validity_calls += 1
            return PlannerResult(success=True, metadata={"group_name": group_name})

        def compute_fk(self, joint_state, *, context=None, link_names=None, frame_id=None):
            self.fk_calls += 1
            pose = np.eye(4, dtype=float)
            pose[0, 3] = 0.55
            pose[2, 3] = 0.3
            pose[:3, 2] = [1.0, 0.0, 0.0]
            pose[:3, 1] = [0.0, 0.0, 1.0]
            pose[:3, 0] = [0.0, -1.0, 0.0]
            return PlannerResult(success=True, metadata={"pose": pose.tolist()})

    monkeypatch.setattr(ga, "_joint_state_msg", lambda joints: SimpleNamespace(name=list(joints), position=list(joints.values())))
    ws = WorldState()
    grasp = np.eye(4, dtype=float)
    grasp[0, 3] = 1.0
    grasp[2, 3] = 0.3
    grasp[:3, 2] = [1.0, 0.0, 0.0]
    grasp[:3, 1] = [0.0, 0.0, 1.0]
    grasp[:3, 0] = [0.0, -1.0, 0.0]
    ws.set("mug_estimated_pick_pose_mat", grasp)
    ws.set("mug_simple_grasp_metadata", {"approach_axis": "+z", "scene": {"min_reach_m": .25, "max_reach_m": .85}})
    pregrasp = grasp.copy()
    pregrasp[0, 3] = 0.88
    ws.set("mug_pregrasp_pose", pregrasp)
    ws.set("mug_grasp_motion_metadata", {"approach_axis": "+z", "scene": {"preferred_hand_x_m": 0.55}})
    ws.set("mug_pcd_frame", "odom")
    planner = FakePlanner()
    context = Context(world_state=ws, planners={"manipulation": planner})

    result = tick_action(SelectFkArmPosture(target='mug', sampling_mode='simple', arm_lift_samples=1, arm_flex_samples=1, execute=False), context)

    assert result.status == "success", (result.reason, result.payload)
    assert planner.validity_calls == 1
    assert planner.fk_calls == 1
    assert ws.get("mug_fk_joint_goal")["arm_lift_joint"] == pytest.approx(0.0)
    assert ws.get("mug_fk_base_pose")["x"] == pytest.approx(0.21)
    assert ws.get("mug_fk_base_pose")["y"] == pytest.approx(0.0)
    assert len(ws.get("mug_fk_posture_candidates")) == 1
    assert ws.get("mug_fk_posture_metadata")["approach_axis"] == "+z"
    assert ws.get("mug_fk_posture_metadata")["state_validity_group"] == "whole_body_weighted"
    resolved = np.asarray(ws.get("mug_fk_posture_candidates")[0]["resolved_hand_pose_world"], dtype=float)
    assert resolved[0, 3] == pytest.approx(pregrasp[0, 3] - 0.12)
    assert resolved[1, 3] == pytest.approx(pregrasp[1, 3])
    assert resolved[2, 3] == pytest.approx(pregrasp[2, 3])


def test_fk_posture_selector_state_validity_diagnostic_mode_is_not_blocking(monkeypatch):
    class FakePlanner:
        def check_state_validity(self, joint_state, *, context=None, group_name=None):
            return PlannerResult(success=False, metadata={"group_name": group_name, "valid": False})

        def compute_fk(self, joint_state, *, context=None, link_names=None, frame_id=None):
            pose = np.eye(4, dtype=float)
            pose[0, 3] = 0.55
            pose[2, 3] = 0.3
            pose[:3, 2] = [1.0, 0.0, 0.0]
            pose[:3, 1] = [0.0, 0.0, 1.0]
            pose[:3, 0] = [0.0, -1.0, 0.0]
            return PlannerResult(success=True, metadata={"pose": pose.tolist()})

    monkeypatch.setattr(ga, "_joint_state_msg", lambda joints: SimpleNamespace(name=list(joints), position=list(joints.values())))
    ws = WorldState()
    pregrasp = np.eye(4, dtype=float)
    pregrasp[0, 3] = 0.88
    pregrasp[2, 3] = 0.3
    pregrasp[:3, 2] = [1.0, 0.0, 0.0]
    pregrasp[:3, 1] = [0.0, 0.0, 1.0]
    pregrasp[:3, 0] = [0.0, -1.0, 0.0]
    ws.set("mug_pregrasp_pose", pregrasp)
    ws.set("mug_grasp_motion_metadata", {"approach_axis": "+z", "scene": {"min_reach_m": .25, "max_reach_m": .85}})
    context = Context(world_state=ws, planners={"manipulation": FakePlanner()})

    result = tick_action(SelectFkArmPosture(target='mug', sampling_mode='simple', arm_lift_samples=1, arm_flex_samples=1, execute=False, enforce_state_validity=False), context)

    assert result.status == "success", (result.reason, result.payload)
    assert ws.get("mug_fk_posture_metadata")["state_validity_enforced"] is False


def test_fk_posture_selector_surround_pcd_off_skips_sweep(monkeypatch):
    class FakePlanner:
        def check_state_validity(self, joint_state, *, context=None, group_name=None):
            return PlannerResult(success=True, metadata={"group_name": group_name})

        def compute_fk(self, joint_state, *, context=None, link_names=None, frame_id=None):
            pose = np.eye(4, dtype=float)
            pose[0, 3] = 0.55
            pose[2, 3] = 0.3
            pose[:3, 2] = [1.0, 0.0, 0.0]
            pose[:3, 1] = [0.0, 0.0, 1.0]
            pose[:3, 0] = [0.0, -1.0, 0.0]
            return PlannerResult(success=True, metadata={"pose": pose.tolist()})

    monkeypatch.setattr(ga, "_joint_state_msg", lambda joints: SimpleNamespace(name=list(joints), position=list(joints.values())))

    def fail_sweep(*args, **kwargs):
        raise AssertionError("surround sweep should not run in off mode")

    monkeypatch.setattr(ga, "_fk_sweep_clearance", fail_sweep)
    ws = WorldState()
    pregrasp = np.eye(4, dtype=float)
    pregrasp[0, 3] = 0.88
    pregrasp[2, 3] = 0.3
    pregrasp[:3, 2] = [1.0, 0.0, 0.0]
    pregrasp[:3, 1] = [0.0, 0.0, 1.0]
    pregrasp[:3, 0] = [0.0, -1.0, 0.0]
    ws.set("mug_pregrasp_pose", pregrasp)
    ws.set("mug_grasp_motion_metadata", {"approach_axis": "+z", "scene": {"min_reach_m": .25, "max_reach_m": .85}})
    context = Context(world_state=ws, planners={"manipulation": FakePlanner()})

    result = tick_action(SelectFkArmPosture(target='mug', sampling_mode='simple', arm_lift_samples=1, arm_flex_samples=1, execute=False), context)

    assert result.status == "success", (result.reason, result.payload)
    candidate = ws.get("mug_fk_posture_candidates")[0]
    assert candidate["sweep"]["allowed"] is True
    assert candidate["sweep"]["mode"] == "off"
    assert candidate["sweep"]["clearance_penalty"] == pytest.approx(0.0)
    assert ws.get("mug_fk_posture_metadata")["surround_pcd_mode"] == "off"


def test_fk_posture_selector_surround_pcd_soft_does_not_hard_reject(monkeypatch):
    class FakePlanner:
        def check_state_validity(self, joint_state, *, context=None, group_name=None):
            return PlannerResult(success=True, metadata={"group_name": group_name})

        def compute_fk(self, joint_state, *, context=None, link_names=None, frame_id=None):
            pose = np.eye(4, dtype=float)
            pose[0, 3] = 0.55
            pose[2, 3] = 0.3
            pose[:3, 2] = [1.0, 0.0, 0.0]
            pose[:3, 1] = [0.0, 0.0, 1.0]
            pose[:3, 0] = [0.0, -1.0, 0.0]
            return PlannerResult(success=True, metadata={"pose": pose.tolist()})

    monkeypatch.setattr(ga, "_joint_state_msg", lambda joints: SimpleNamespace(name=list(joints), position=list(joints.values())))
    monkeypatch.setattr(
        ga,
        "_fk_sweep_clearance",
        lambda *args, **kwargs: {"allowed": False, "reason": "fk_sweep_blocked", "clearance_penalty": 2.0},
    )
    ws = WorldState()
    pregrasp = np.eye(4, dtype=float)
    pregrasp[0, 3] = 0.88
    pregrasp[2, 3] = 0.3
    pregrasp[:3, 2] = [1.0, 0.0, 0.0]
    pregrasp[:3, 1] = [0.0, 0.0, 1.0]
    pregrasp[:3, 0] = [0.0, -1.0, 0.0]
    ws.set("mug_pregrasp_pose", pregrasp)
    ws.set("mug_grasp_motion_metadata", {"approach_axis": "+z", "scene": {"min_reach_m": .25, "max_reach_m": .85}})
    context = Context(world_state=ws, planners={"manipulation": FakePlanner()})

    result = tick_action(SelectFkArmPosture(target='mug', sampling_mode='simple', arm_lift_samples=1, arm_flex_samples=1, execute=False, surround_pcd_mode='soft'), context)

    assert result.status == "success", (result.reason, result.payload)
    candidate = ws.get("mug_fk_posture_candidates")[0]
    assert candidate["sweep"]["allowed"] is False
    assert candidate["sweep"]["mode"] == "soft"
    assert ws.get("mug_fk_posture_metadata")["surround_pcd_mode"] == "soft"


def test_fk_posture_selector_surround_pcd_diagnostic_records_without_score_penalty(monkeypatch):
    class FakePlanner:
        def check_state_validity(self, joint_state, *, context=None, group_name=None):
            return PlannerResult(success=True, metadata={"group_name": group_name})

        def compute_fk(self, joint_state, *, context=None, link_names=None, frame_id=None):
            pose = np.eye(4, dtype=float)
            pose[0, 3] = 0.55
            pose[2, 3] = 0.3
            pose[:3, 2] = [1.0, 0.0, 0.0]
            pose[:3, 1] = [0.0, 0.0, 1.0]
            pose[:3, 0] = [0.0, -1.0, 0.0]
            return PlannerResult(success=True, metadata={"pose": pose.tolist()})

    monkeypatch.setattr(ga, "_joint_state_msg", lambda joints: SimpleNamespace(name=list(joints), position=list(joints.values())))
    monkeypatch.setattr(
        ga,
        "_fk_sweep_clearance",
        lambda *args, **kwargs: {"allowed": False, "reason": "fk_sweep_blocked", "clearance_penalty": 2.0},
    )
    ws = WorldState()
    pregrasp = np.eye(4, dtype=float)
    pregrasp[0, 3] = 0.88
    pregrasp[2, 3] = 0.3
    pregrasp[:3, 2] = [1.0, 0.0, 0.0]
    pregrasp[:3, 1] = [0.0, 0.0, 1.0]
    pregrasp[:3, 0] = [0.0, -1.0, 0.0]
    ws.set("mug_pregrasp_pose", pregrasp)
    ws.set("mug_grasp_motion_metadata", {"approach_axis": "+z", "scene": {"min_reach_m": .25, "max_reach_m": .85}})
    context = Context(world_state=ws, planners={"manipulation": FakePlanner()})

    result = tick_action(SelectFkArmPosture(target='mug', sampling_mode='simple', arm_lift_samples=1, arm_flex_samples=1, execute=False, surround_pcd_mode='diagnostic'), context)

    assert result.status == "success", (result.reason, result.payload)
    candidate = ws.get("mug_fk_posture_candidates")[0]
    assert candidate["sweep"]["mode"] == "diagnostic"
    assert candidate["sweep"]["clearance_penalty"] == pytest.approx(0.0)


def test_fk_posture_selector_surround_pcd_hard_gate_rejects_blocked_sweep(monkeypatch):
    class FakePlanner:
        def check_state_validity(self, joint_state, *, context=None, group_name=None):
            return PlannerResult(success=True, metadata={"group_name": group_name})

        def compute_fk(self, joint_state, *, context=None, link_names=None, frame_id=None):
            pose = np.eye(4, dtype=float)
            pose[0, 3] = 0.55
            pose[2, 3] = 0.3
            pose[:3, 2] = [1.0, 0.0, 0.0]
            pose[:3, 1] = [0.0, 0.0, 1.0]
            pose[:3, 0] = [0.0, -1.0, 0.0]
            return PlannerResult(success=True, metadata={"pose": pose.tolist()})

    monkeypatch.setattr(ga, "_joint_state_msg", lambda joints: SimpleNamespace(name=list(joints), position=list(joints.values())))
    monkeypatch.setattr(
        ga,
        "_fk_sweep_clearance",
        lambda *args, **kwargs: {"allowed": False, "reason": "fk_sweep_blocked", "clearance_penalty": 2.0},
    )
    ws = WorldState()
    pregrasp = np.eye(4, dtype=float)
    pregrasp[0, 3] = 0.88
    pregrasp[2, 3] = 0.3
    pregrasp[:3, 2] = [1.0, 0.0, 0.0]
    pregrasp[:3, 1] = [0.0, 0.0, 1.0]
    pregrasp[:3, 0] = [0.0, -1.0, 0.0]
    ws.set("mug_pregrasp_pose", pregrasp)
    ws.set("mug_grasp_motion_metadata", {"approach_axis": "+z", "scene": {"min_reach_m": .25, "max_reach_m": .85}})
    context = Context(world_state=ws, planners={"manipulation": FakePlanner()})

    result = tick_action(SelectFkArmPosture(target='mug', sampling_mode='simple', arm_lift_samples=1, arm_flex_samples=1, execute=False, surround_pcd_mode='hard_gate'), context)

    assert result.status == "failure"
    assert result.reason == "fk_posture_no_valid_candidate"


def test_fk_posture_selector_enforced_whole_body_invalid_candidate_is_rejected(monkeypatch):
    class FakePlanner:
        def check_state_validity(self, joint_state, *, context=None, group_name=None):
            assert group_name == "whole_body_weighted"
            return PlannerResult(success=False, metadata={"group_name": group_name, "valid": False})

        def compute_fk(self, joint_state, *, context=None, link_names=None, frame_id=None):
            pose = np.eye(4, dtype=float)
            pose[0, 3] = 0.55
            pose[2, 3] = 0.3
            pose[:3, 2] = [1.0, 0.0, 0.0]
            pose[:3, 1] = [0.0, 0.0, 1.0]
            pose[:3, 0] = [0.0, -1.0, 0.0]
            return PlannerResult(success=True, metadata={"pose": pose.tolist()})

    monkeypatch.setattr(ga, "_joint_state_msg", lambda joints: SimpleNamespace(name=list(joints), position=list(joints.values())))
    ws = WorldState()
    pregrasp = np.eye(4, dtype=float)
    pregrasp[0, 3] = 0.88
    pregrasp[2, 3] = 0.3
    pregrasp[:3, 2] = [1.0, 0.0, 0.0]
    pregrasp[:3, 1] = [0.0, 0.0, 1.0]
    pregrasp[:3, 0] = [0.0, -1.0, 0.0]
    ws.set("mug_pregrasp_pose", pregrasp)
    ws.set("mug_grasp_motion_metadata", {"approach_axis": "+z", "scene": {"min_reach_m": .25, "max_reach_m": .85}})
    context = Context(world_state=ws, planners={"manipulation": FakePlanner()})

    result = tick_action(SelectFkArmPosture(target='mug', sampling_mode='simple', arm_lift_samples=1, arm_flex_samples=1, execute=False, enforce_state_validity=True), context)

    assert result.status == "failure"
    assert result.reason == "fk_posture_no_valid_candidate"


def test_fk_posture_selector_multires_adds_refined_samples():
    action = SelectFkArmPosture(
        target="mug",
        arm_lift_samples=2,
        arm_flex_samples=2,
        execute=False,
        sampling_mode="multires_scene",
    )
    pregrasp = np.eye(4, dtype=float)
    pregrasp[2, 3] = 0.42

    candidates = action._candidate_joint_sets(
        pregrasp_pose=pregrasp,
        approach_axis="+z",
        scene={"support_depth_m": 0.35, "suggested_insertion_m": 0.12, "preferred_hand_x_m": 0.45},
        sampling_mode="multires_scene",
    )

    assert len(candidates) > 4


def test_fk_posture_selector_state_validity_uses_full_state_with_base_override(monkeypatch):
    captured = {}

    class FakePlanner:
        def check_state_validity(self, joint_state, *, context=None, group_name=None):
            captured["group_name"] = group_name
            captured["validity_names"] = list(joint_state.name)
            captured["validity_positions"] = dict(zip(joint_state.name, joint_state.position, strict=False))
            return PlannerResult(success=True, metadata={"group_name": group_name})

        def compute_fk(self, joint_state, *, context=None, link_names=None, frame_id=None):
            pose = np.eye(4, dtype=float)
            pose[0, 3] = 0.55
            pose[2, 3] = 0.3
            pose[:3, 2] = [1.0, 0.0, 0.0]
            pose[:3, 1] = [0.0, 0.0, 1.0]
            pose[:3, 0] = [0.0, -1.0, 0.0]
            return PlannerResult(success=True, metadata={"pose": pose.tolist()})

    ws = WorldState()
    pregrasp = np.eye(4, dtype=float)
    pregrasp[0, 3] = 0.88
    pregrasp[2, 3] = 0.3
    pregrasp[:3, 2] = [1.0, 0.0, 0.0]
    pregrasp[:3, 1] = [0.0, 0.0, 1.0]
    pregrasp[:3, 0] = [0.0, -1.0, 0.0]
    ws.set("mug_pregrasp_pose", pregrasp)
    ws.set("mug_grasp_motion_metadata", {"approach_axis": "+z", "scene": {"min_reach_m": .25, "max_reach_m": .85}})
    joint_state = SimpleNamespace(
        name=["head_pan_joint", "odom_x", "odom_y", "odom_t", "arm_lift_joint", "arm_flex_joint"],
        position=[0.2, 9.0, 9.0, 1.0, 0.22, -0.3],
    )
    context = Context(
        world_state=ws,
        planners={"manipulation": FakePlanner()},
        sensor_data_center=_FakeSensorDataCenter(joint_state=joint_state),
    )

    monkeypatch.setattr(ga, "_joint_state_msg", lambda joints: SimpleNamespace(name=list(joints), position=list(joints.values())))

    result = tick_action(SelectFkArmPosture(target='mug', sampling_mode='simple', arm_lift_samples=1, arm_flex_samples=1, execute=False), context)

    assert result.status == "success", (result.reason, result.payload)
    assert captured["group_name"] == "whole_body_weighted"
    assert captured["validity_positions"]["head_pan_joint"] == pytest.approx(0.2)
    assert captured["validity_positions"]["arm_lift_joint"] == pytest.approx(0.0)
    selected = ws.get("mug_fk_posture_candidates")[0]
    assert captured["validity_positions"]["odom_x"] == pytest.approx(selected["final_grasp_base_pose"]["x"])
    assert captured["validity_positions"]["odom_t"] == pytest.approx(ws.get("mug_fk_base_pose")["yaw"])


def test_fk_posture_selector_validity_uses_final_grasp_base_but_stores_pregrasp_base(monkeypatch):
    captured = {}

    class FakePlanner:
        def check_state_validity(self, joint_state, *, context=None, group_name=None):
            captured["positions"] = dict(zip(joint_state.name, joint_state.position, strict=False))
            return PlannerResult(success=True, metadata={"group_name": group_name})

        def compute_fk(self, joint_state, *, context=None, link_names=None, frame_id=None):
            pose = np.eye(4, dtype=float)
            pose[0, 3] = 0.55
            pose[2, 3] = 0.3
            pose[:3, 2] = [1.0, 0.0, 0.0]
            pose[:3, 1] = [0.0, 0.0, 1.0]
            pose[:3, 0] = [0.0, -1.0, 0.0]
            return PlannerResult(success=True, metadata={"pose": pose.tolist()})

    monkeypatch.setattr(ga, "_joint_state_msg", lambda joints: SimpleNamespace(name=list(joints), position=list(joints.values())))
    ws = WorldState()
    grasp = np.eye(4, dtype=float)
    grasp[0, 3] = 1.0
    grasp[2, 3] = 0.3
    grasp[:3, 2] = [1.0, 0.0, 0.0]
    grasp[:3, 1] = [0.0, 0.0, 1.0]
    grasp[:3, 0] = [0.0, -1.0, 0.0]
    pregrasp = grasp.copy()
    pregrasp[0, 3] = 0.88
    ws.set("mug_grasp_pose", grasp)
    ws.set("mug_pregrasp_pose", pregrasp)
    ws.set("mug_grasp_motion_metadata", {"approach_axis": "+z", "scene": {"min_reach_m": .25, "max_reach_m": .85}})
    context = Context(world_state=ws, planners={"manipulation": FakePlanner()})

    result = tick_action(SelectFkArmPosture(target='mug', sampling_mode='simple', arm_lift_samples=1, arm_flex_samples=1, execute=False), context)

    assert result.status == "success", (result.reason, result.payload)
    candidate = ws.get("mug_fk_posture_candidates")[0]
    assert ws.get("mug_fk_base_pose")["x"] == pytest.approx(0.21)
    assert candidate["final_grasp_base_pose"]["x"] == pytest.approx(0.45)
    assert captured["positions"]["odom_x"] == pytest.approx(0.45)


def test_fk_posture_selector_execute_true_sends_arm_goal(monkeypatch):
    class FakePlanner:
        def check_state_validity(self, joint_state, *, context=None, group_name=None):
            return PlannerResult(success=True)

        def compute_fk(self, joint_state, *, context=None, link_names=None, frame_id=None):
            pose = np.eye(4, dtype=float)
            pose[0, 3] = 0.55
            pose[2, 3] = 0.3
            pose[:3, 2] = [1.0, 0.0, 0.0]
            pose[:3, 1] = [0.0, 0.0, 1.0]
            pose[:3, 0] = [0.0, -1.0, 0.0]
            return PlannerResult(success=True, metadata={"pose": pose.tolist()})

    monkeypatch.setattr(ga, "_joint_state_msg", lambda joints: SimpleNamespace(name=list(joints), position=list(joints.values())))
    ws = WorldState()
    grasp = np.eye(4, dtype=float)
    grasp[0, 3] = 1.0
    grasp[2, 3] = 0.3
    grasp[:3, 2] = [1.0, 0.0, 0.0]
    grasp[:3, 1] = [0.0, 0.0, 1.0]
    grasp[:3, 0] = [0.0, -1.0, 0.0]
    ws.set("mug_estimated_pick_pose_mat", grasp)
    ws.set("mug_simple_grasp_metadata", {"approach_axis": "+z", "scene": {"min_reach_m": .25, "max_reach_m": .85}})
    ws.set("mug_grasp_motion_metadata", {"approach_axis": "+z", "scene": {"min_reach_m": .25, "max_reach_m": .85}})
    arm = _FakeArm()
    context = Context(world_state=ws, planners={"manipulation": FakePlanner()}, arm=arm)
    action = SelectFkArmPosture(target="mug", sampling_mode="simple", arm_lift_samples=1, arm_flex_samples=1)

    first = tick_action(action, context)
    assert first.status == "running", (first.reason, first.payload)
    assert arm.goals and arm.goals[0]["arm_lift_joint"] == pytest.approx(0.0)

    arm.status = "succeeded"
    second = tick_action(action, context)
    assert second.status == "success"
    assert second.reason == "fk_posture_executed"


def test_fk_posture_selector_bare_cached_goal_cannot_bypass_runtime_inputs():
    class FakePlanner:
        def compute_fk(self, *args, **kwargs):
            raise AssertionError("missing input must fail before FK")
    ws = WorldState()
    ws.set("mug_fk_joint_goal", {"arm_lift_joint": .12})
    arm = _FakeArm()
    context = Context(world_state=ws, planners={"manipulation": FakePlanner()}, arm=arm)
    result = tick_action(SelectFkArmPosture(target="mug", execute=True), context)
    assert result.status == "failure"
    assert "missing_pregrasp_pose" in result.reason
    assert arm.goals == []


def test_fk_top_down_uses_full_reach_solved_lift_and_preserves_base_yaw(monkeypatch):
    class FakePlanner:
        def check_state_validity(self, joint_state, *, context=None, group_name=None):
            return PlannerResult(success=True, metadata={"group_name": group_name})

        def compute_fk(self, joint_state, *, context=None, link_names=None, frame_id=None):
            joints = dict(zip(joint_state.name, joint_state.position, strict=False))
            wrist_roll = float(joints.get("wrist_roll_joint", 0.0))
            pose = np.eye(4, dtype=float)
            pose[0, 3] = 0.55
            pose[1, 3] = 0.02
            pose[2, 3] = 0.20 + float(joints.get("arm_lift_joint", 0.0))
            pose[:3, 2] = [0.0, 0.0, -1.0]
            pose[:3, 0] = [math.cos(wrist_roll), math.sin(wrist_roll), 0.0]
            pose[:3, 1] = [-math.sin(wrist_roll), math.cos(wrist_roll), 0.0]
            return PlannerResult(success=True, metadata={"pose": pose.tolist()})

    monkeypatch.setattr(ga, "_joint_state_msg", lambda joints: SimpleNamespace(name=list(joints), position=list(joints.values())))
    ws = WorldState()
    grasp = np.eye(4, dtype=float)
    grasp[:3, 0] = [1.0, 0.0, 0.0]
    grasp[:3, 1] = [0.0, 1.0, 0.0]
    grasp[:3, 2] = [0.0, 0.0, -1.0]
    grasp[:3, 3] = [1.0, 0.2, 0.28]
    pregrasp = grasp.copy()
    pregrasp[2, 3] = 0.36
    ws.set("mug_grasp_pose", grasp)
    ws.set("mug_pregrasp_pose", pregrasp)
    ws.set("mug_grasp_motion_metadata", {"approach_axis": "+z", "grasp_motion_type": "vertical_z", "scene": {"min_reach_m": 0.25, "max_reach_m": 0.85}})
    context = Context(
        world_state=ws,
        planners={"manipulation": FakePlanner()},
        sensor_data_center=_FakeSensorDataCenter(pose={"pose": SimpleNamespace(x=0.0, y=0.0, yaw=0.7)}),
    )

    result = tick_action(SelectFkArmPosture(target='mug', arm_lift_min=0.0, arm_lift_max=0.35, arm_flex_min=-1.57, arm_flex_max=-0.2, arm_lift_samples=6, arm_flex_samples=6, execute=False), context)

    assert result.status == "success"
    goal = ws.get("mug_fk_joint_goal")
    assert goal["arm_flex_joint"] == pytest.approx(-1.57)
    assert goal["arm_lift_joint"] == pytest.approx(0.16)
    candidates = ws.get("mug_fk_posture_candidates")
    assert len({round(c["joint_positions"]["arm_lift_joint"], 5) for c in candidates}) == 1
    assert len({round(c["joint_positions"]["arm_flex_joint"], 5) for c in candidates}) == 1
    assert len({round(c["joint_positions"]["wrist_roll_joint"], 5) for c in candidates}) > 1
    assert ws.get("mug_fk_base_pose")["yaw"] == pytest.approx(0.7)
    metadata = ws.get("mug_fk_posture_metadata")
    assert metadata["vertical_top_down_fk"] is True
    assert metadata["base_yaw_mode"] == "preserve_current"
    assert metadata["fixed_arm_flex_joint"] == pytest.approx(-1.57)
    assert metadata["solved_arm_lift_joint"] == pytest.approx(0.16)


def test_fk_metadata_not_pose_matrix_controls_top_down_branch(monkeypatch):
    class FakePlanner:
        def check_state_validity(self, joint_state, *, context=None, group_name=None):
            return PlannerResult(success=True)

        def compute_fk(self, joint_state, *, context=None, link_names=None, frame_id=None):
            pose = np.eye(4, dtype=float)
            pose[0, 3] = 0.55
            pose[2, 3] = 0.3
            pose[:3, 2] = [0.0, 0.0, -1.0]
            pose[:3, 0] = [1.0, 0.0, 0.0]
            pose[:3, 1] = [0.0, 1.0, 0.0]
            return PlannerResult(success=True, metadata={"pose": pose.tolist()})

    monkeypatch.setattr(ga, "_joint_state_msg", lambda joints: SimpleNamespace(name=list(joints), position=list(joints.values())))
    ws = WorldState()
    pregrasp = np.eye(4, dtype=float)
    pregrasp[:3, 2] = [0.0, 0.0, -1.0]
    pregrasp[:3, 3] = [1.0, 0.0, 0.3]
    ws.set("mug_pregrasp_pose", pregrasp)
    ws.set("mug_grasp_motion_metadata", {"approach_axis": "+z", "grasp_motion_type": "horizontal_projected"})
    context = Context(world_state=ws, planners={"manipulation": FakePlanner()})

    result = tick_action(SelectFkArmPosture(target='mug', arm_lift_samples=2, arm_flex_samples=2, execute=False), context)

    assert result.status == "success"
    metadata = ws.get("mug_fk_posture_metadata")
    assert metadata["vertical_top_down_fk"] is False
    assert metadata["base_yaw_mode"] == "approach_alignment"


def test_top_down_pregrasp_adjust_noops_for_non_vertical():
    ws = WorldState()
    ws.set("mug_grasp_motion_metadata", {"grasp_motion_type": "horizontal_projected"})
    arm = _FakeArm()
    context = Context(world_state=ws, arm=arm)
    action = ga._TopDownPreGraspAdjustAction("mug")

    action.start(context)
    arm.status = "succeeded"
    result = action.check_result(context)

    assert result.status == "success"
    assert result.reason == "top_down_pregrasp_adjust_noop"
    assert arm.goals == []


def test_top_down_pregrasp_adjust_sets_wrist_roll_and_lowers_lift(monkeypatch):
    ws = WorldState()
    grasp = np.eye(4, dtype=float)
    grasp[2, 3] = 0.30
    pregrasp = np.eye(4, dtype=float)
    pregrasp[2, 3] = 0.42
    ws.set("mug_grasp_pose", grasp)
    ws.set("mug_pregrasp_pose", pregrasp)
    ws.set("mug_grasp_motion_metadata", {"grasp_motion_type": "vertical_z"})
    ws.set("mug_fk_joint_goal", {"wrist_roll_joint": 0.75, "arm_lift_joint": 0.2})
    arm = _FakeArm()
    arm.status = "succeeded"
    context = Context(
        world_state=ws,
        arm=arm,
        sensor_data_center=_FakeSensorDataCenter(joint_state=SimpleNamespace(name=["arm_lift_joint"], position=[0.20])),
    )
    monkeypatch.setattr(context, "run_in_context_loop", lambda func, *args, callback=None, error_callback=None, **kwargs: callback(func()))
    action = ga._TopDownPreGraspAdjustAction("mug")

    action.start(context)
    arm.status = "succeeded"
    result = action.check_result(context)

    assert result.status == "success"
    assert arm.goals[-1] == {"wrist_roll_joint": pytest.approx(0.75), "arm_lift_joint": pytest.approx(0.12)}


def test_top_down_pregrasp_adjust_fails_without_cached_wrist_roll():
    ws = WorldState()
    ws.set("mug_grasp_motion_metadata", {"grasp_motion_type": "vertical_z"})
    ws.set("mug_fk_joint_goal", {"arm_lift_joint": 0.2})
    context = Context(world_state=ws, arm=_FakeArm())
    action = ga._TopDownPreGraspAdjustAction("mug")

    action.start(context)
    result = action.check_result(context)

    assert result.status == "failure"
    assert result.reason == "missing_top_down_fk_joint_goal:mug_fk_joint_goal"


def test_tmc_fk_and_servo_marker_publishers_are_best_effort(monkeypatch):
    _install_marker_stubs(monkeypatch)
    node = _FakeNode()
    context = Context(ros2_node=node)
    pose = np.eye(4, dtype=float)
    ranked = [
        {"index": 0, "origin_to_base": {"x": 0.0, "y": 0.0, "yaw": 0.0, "frame_id": "odom"}},
        {"index": 1, "origin_to_base": {"x": 0.5, "y": 0.0, "yaw": 0.0, "frame_id": "odom"}},
    ]

    ga._publish_tmc_ik_markers(context, "/orchid/grasp_alignment/tmc_ik", "odom", pose, ranked, ranked[1], "mug")
    assert node.publishers["/orchid/grasp_alignment/tmc_ik"].messages[-1].markers

    candidate = {
        "joint_positions": {"arm_lift_joint": 0.0},
        "hand_pose_base": pose.tolist(),
        "base_pose": {"x": 0.3, "y": 0.0, "yaw": 0.0, "frame_id": "odom"},
        "resolved_hand_pose_world": pose.tolist(),
    }
    ga._publish_fk_posture_markers(
        context,
        "/orchid/grasp_alignment/fk_postures",
        "odom",
        "base_footprint",
        [candidate],
        candidate,
        "mug",
        max_candidates=10,
    )
    fk_markers = node.publishers["/orchid/grasp_alignment/fk_postures"].messages[-1].markers
    assert fk_markers
    assert any(marker.ns == "mug_fk_raw_candidates" for marker in fk_markers)
    assert any(marker.ns == "mug_fk_resolved_candidates" for marker in fk_markers)
    assert any(marker.ns == "mug_fk_resolved_hand" for marker in fk_markers)

    ga._publish_servo_markers(
        context,
        "/orchid/grasp_alignment/servo",
        "odom",
        BasePose2D(1.0, 0.0, 0.0, "odom"),
        BasePose2D(0.0, 0.0, 0.0, "odom"),
        BasePose2D(0.1, 0.0, 0.0, "odom"),
        blocked=BasePose2D(0.2, 0.0, 0.0, "odom"),
    )
    servo_markers = node.publishers["/orchid/grasp_alignment/servo"].messages[-1].markers
    assert len(servo_markers) == 4


def test_final_approach_arm_cartesian_sends_planned_trajectory(monkeypatch):
    class FakePlanner:
        execution_defaults = MoveItPlanningSettings()
        def plan_cartesian_request(self, request, *, context=None):
            self.request = request
            return PlannerResult(success=True, trajectory={"prepared": True}, metadata={"fraction": 1.0})

    ws = WorldState()
    pregrasp = np.eye(4, dtype=float)
    grasp = np.eye(4, dtype=float)
    pregrasp[0, 3] = 0.88
    grasp[0, 3] = 1.0
    for pose in (pregrasp, grasp):
        pose[2, 3] = 0.3
        pose[:3, 2] = [1.0, 0.0, 0.0]
        pose[:3, 1] = [0.0, 0.0, 1.0]
        pose[:3, 0] = [0.0, -1.0, 0.0]
    ws.set("mug_pregrasp_pose", pregrasp)
    ws.set("mug_grasp_pose", grasp)
    ws.set("mug_grasp_motion_metadata", {"approach_axis": "+z", "frame_id": "odom"})
    ws.set("mug_fk_joint_goal", {"arm_lift_joint": 0.1})
    arm = _FakeArm()
    planner = FakePlanner()
    context = Context(world_state=ws, planners={"manipulation": planner}, arm=arm)

    action = ExecuteHsrCartesianGraspMotion(target="mug", mode="arm_cartesian", debug_visualization=False)
    first = tick_action(action, context)
    assert first.status == "running"
    assert arm.goals[-1] == {"trajectory": {"prepared": True}}
    assert planner.request.group_name == "whole_body_weighted"

    arm.status = "succeeded"
    second = tick_action(action, context)
    assert second.status == "success"


def test_final_approach_ik_waypoints_builds_constrained_move_arm_actions(monkeypatch):
    created = []

    class FakeMoveArmToPose(AbstractAction):
        def __init__(self, request):
            self.request = request
            self.done = False
            created.append(self)

        def execute(self, context):
            if not self.done:
                self.done = True
                return ga.ActionResult(status="success")
            return ga.ActionResult(status="success")

        def cancel(self, context):
            pass

    import src.plugin.actions.manipulation as manipulation

    monkeypatch.setattr(manipulation, "_MoveArmToPose", FakeMoveArmToPose)
    ws = WorldState()
    pregrasp = np.eye(4, dtype=float)
    grasp = np.eye(4, dtype=float)
    pregrasp[0, 3] = 0.88
    grasp[0, 3] = 1.0
    for pose in (pregrasp, grasp):
        pose[2, 3] = 0.3
        pose[:3, 2] = [1.0, 0.0, 0.0]
        pose[:3, 1] = [0.0, 0.0, 1.0]
        pose[:3, 0] = [0.0, -1.0, 0.0]
    ws.set("mug_pregrasp_pose", pregrasp)
    ws.set("mug_grasp_pose", grasp)
    ws.set("mug_grasp_motion_metadata", {"approach_axis": "+z", "frame_id": "odom"})
    ws.set("mug_fk_base_pose", {"x": 0.3, "y": 0.0, "yaw": 0.0, "frame_id": "odom"})
    context = Context(world_state=ws)

    result = tick_action(ExecuteHsrCartesianGraspMotion(target='mug', waypoint_count=3, debug_visualization=False), context)

    assert result.status == "success"
    assert len(created) == 3
    assert created[0].request.moveit.group_name == "whole_body_weighted"
    assert created[0].request.moveit.goal_mode == "ik_joint_goal"
    assert created[0].request.moveit.constrain_base is True
    assert created[0].request.moveit.base_pose.to_dict() == {"x": 0.3, "y": 0.0, "yaw": 0.0, "frame_id": "odom"}
    assert created[-1].request.moveit.collision_policy == "target_relaxed"
    assert ws.get("mug_final_approach_waypoints")[-1][0][3] == pytest.approx(1.0)


def test_final_approach_base_servo_stores_final_base_pose(monkeypatch):
    monkeypatch.setattr(ga.OmniBaseServoToPose, "start", lambda self, context: setattr(self, "_status", "success"))
    monkeypatch.setattr(
        ga.OmniBaseServoToPose,
        "check_result",
        lambda self, context: ga.ActionResult(status="success", reason="omni_servo_reached"),
    )
    ws = WorldState()
    pregrasp = np.eye(4, dtype=float)
    grasp = np.eye(4, dtype=float)
    grasp[0, 3] = 1.0
    pregrasp[0, 3] = 0.88
    for pose in (pregrasp, grasp):
        pose[2, 3] = 0.3
        pose[:3, 2] = [1.0, 0.0, 0.0]
        pose[:3, 1] = [0.0, 0.0, 1.0]
        pose[:3, 0] = [0.0, -1.0, 0.0]
    hand = np.eye(4, dtype=float)
    hand[0, 3] = 0.55
    hand[2, 3] = 0.3
    hand[:3, 2] = [1.0, 0.0, 0.0]
    hand[:3, 1] = [0.0, 0.0, 1.0]
    hand[:3, 0] = [0.0, -1.0, 0.0]
    ws.set("mug_pregrasp_pose", pregrasp)
    ws.set("mug_grasp_pose", grasp)
    ws.set("mug_grasp_motion_metadata", {"approach_axis": "+z", "frame_id": "odom"})
    ws.set("mug_fk_posture_metadata", {"selected": {"hand_pose_base": hand.tolist()}})
    context = Context(world_state=ws)

    result = tick_action(ExecuteHsrCartesianGraspMotion(target='mug', mode='base_servo', debug_visualization=False), context)

    assert result.status == "success"
    assert ws.get("mug_final_approach_base_pose")["x"] == pytest.approx(0.45)


def test_final_approach_base_servo_falls_back_to_tmc_base_delta(monkeypatch):
    monkeypatch.setattr(ga.OmniBaseServoToPose, "start", lambda self, context: setattr(self, "_status", "success"))
    monkeypatch.setattr(
        ga.OmniBaseServoToPose,
        "check_result",
        lambda self, context: ga.ActionResult(status="success", reason="omni_servo_reached"),
    )
    ws = WorldState()
    pregrasp = np.eye(4, dtype=float)
    grasp = np.eye(4, dtype=float)
    pregrasp[0, 3] = 0.88
    grasp[0, 3] = 1.0
    ws.set("mug_pregrasp_pose", pregrasp)
    ws.set("mug_grasp_pose", grasp)
    ws.set("mug_grasp_motion_metadata", {"approach_axis": "+z", "frame_id": "odom"})
    ws.set("mug_tmc_base_pose", {"x": 0.3, "y": 0.1, "yaw": 0.2, "frame_id": "odom"})
    context = Context(world_state=ws)

    result = tick_action(ExecuteHsrCartesianGraspMotion(target='mug', mode='base_servo', debug_visualization=False), context)

    assert result.status == "success"
    final_base = ws.get("mug_final_approach_base_pose")
    assert final_base["x"] == pytest.approx(0.42)
    assert final_base["y"] == pytest.approx(0.1)
    assert final_base["yaw"] == pytest.approx(0.2)
    assert final_base["frame_id"] == "odom"


def test_planner_facing_docstrings_hide_internal_keys():
    for cls in (
        EstimateSimpleCylindricalGrasp,
        SelectHsrGraspMotion,
        ga.SolveTmcGraspIk,
        SelectFkArmPosture,
        MoveToSelectedGraspBasePose,
        ga.OmniBaseServoToPose,
        ExecuteHsrCartesianGraspMotion,
        PickTargetWithGraspAlignment,
    ):
        doc = cls.__doc__ or ""
        assert "target" in doc
        assert "object_pcd_key" not in doc
        assert "grasp_pose_key" not in doc
        assert "marker_topic" not in doc
        assert "object_pcd_key" not in doc


def test_omni_servo_command_rotates_error_into_base_frame_and_clamps():
    cmd = compute_omni_servo_command(
        BasePose2D(0.0, 0.0, math.pi / 2.0, "odom"),
        BasePose2D(1.0, 0.0, math.pi / 2.0, "odom"),
        max_vx=0.2,
        max_vy=0.2,
    )

    assert abs(cmd.vx) < 1e-6
    assert cmd.vy == pytest.approx(-0.2)
    assert cmd.xy_error == pytest.approx(1.0)


def test_omni_servo_collision_gate_diagnostic_does_not_block(monkeypatch):
    _install_marker_stubs(monkeypatch)
    states = [
        BasePose2D(0.0, 0.0, 0.0, "odom"),
        BasePose2D(0.01, 0.0, 0.0, "odom"),
    ]

    def fake_pose(context, frame_id):
        return states.pop(0) if states else BasePose2D(0.01, 0.0, 0.0, "odom")

    monkeypatch.setattr(ga, "_current_base_pose", fake_pose)
    monkeypatch.setattr(ga, "_servo_collision_allowed", lambda context, candidate, target="": {"allowed": False, "reason": "blocked"})
    ws = WorldState()
    ws.set("mug_fk_base_pose", {"x": 0.01, "y": 0.0, "yaw": 0.0, "frame_id": "odom"})
    node = _FakeNode()
    context = Context(world_state=ws, ros2_node=node)

    action = ga.OmniBaseServoToPose(
        target="mug",
        rate_hz=200.0,
        timeout_s=0.2,
        xy_tolerance_m=0.02,
        settle_cycles=1,
        collision_gate_mode="diagnostic",
    )
    first = tick_action(action, context)
    deadline = 20
    result = first
    while result.status == "running" and deadline:
        result = tick_action(action, context)
        deadline -= 1

    assert result.status == "success"
    assert node.publishers["/omni_base_controller/cmd_vel"].messages


def test_costmap_collision_gate_rejects_projected_blocked_footprint():
    grid = _free_grid()
    # Occupy the cell around x=0.5, y=0.0 in the odom frame.
    u = int((0.5 - (-1.0)) / 0.1)
    v = int((0.0 - (-1.0)) / 0.1)
    data = list(grid.data)
    data[v * grid.info.width + u] = 100
    grid.data = data
    context = Context(sensor_data_center=_FakeSensorDataCenter(local_costmap=grid), hardware={"manipulation": {"costmap_timeout_s": 0.01}})

    result = ga._costmap_candidate_allowed(context, BasePose2D(0.5, 0.0, 0.0, "odom"))

    assert result["allowed"] is False
    assert result["reason"] in {"footprint_costmap_blocked", "candidate_out_of_grid"}


def test_pointcloud_collision_gate_rejects_surround_points_inside_projected_footprint():
    ws = WorldState()
    ws.set("mug_surround_pcd", np.array([[0.2, 0.0, 0.4], [1.5, 1.5, 0.4]], dtype=np.float32))
    ws.set("mug_pcd_frame", "odom")
    context = Context(
        world_state=ws,
        sensor_data_center=_FakeSensorDataCenter(),
        hardware={"manipulation": {"servo_use_pointcloud_gate": True}},
    )

    result = ga._pointcloud_candidate_allowed(context, BasePose2D(0.0, 0.0, 0.0, "odom"), target="mug")

    assert result["allowed"] is False
    assert result["reason"] == "pointcloud_footprint_blocked"
    assert result["blocked_count"] == 1
