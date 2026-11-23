"""Tests for discoverable plugin actions."""

from __future__ import annotations

import time
import math
from types import SimpleNamespace

import numpy as np
import pytest

from src.core.action import ActionResult, AsyncAction
from tests.fixtures.native_tree import tick_action, prepare_action, native_actions, action_node
from src.core.context import Context
from src.core.controllers import Pose
from src.core.planners import BaseManipulationPlanner, PlannerResult
from src.core.module_proxy import ModuleProxy
from src.core.scene_graph import SceneGraph
from src.core.world_state import WorldState
from src.plugin.actions.manipulation import (
    GraspObjectAction,
    MoveArmToState,
    PickTarget,
    PickAtPose,
    PlaceAtPose,
    ReleaseObjectAction,
)
from src.plugin.actions.navigation import AlignViewToSceneTarget, MoveBaseWithWholeBody, MoveToManipulationPose, MoveToPose, MoveToTarget
from src.plugin.actions.head import MoveHeadToPanTilt, MoveHeadToState
from src.plugin.actions.observe import DescribeScene, ScanScene
from src.plugin.actions.user_input import AskUser
from src.plugin.registry import PluginRegistry
from src.tree.builder import TreeBuilder
from src.tree.execution import ExecutionEngine
from src.tree.plan import ActionNode, Plan
from tests.fixtures.mock_controllers import MockArmController, MockGripperController, MockNavController


class _RunThenSucceed(AsyncAction):
    def __init__(self, label: str = "run"):
        super().__init__()
        self.label = label
        self.ticks = 0

    def start(self, context):
        return None

    def check_result(self, context):
        self.ticks += 1
        if self.ticks == 1:
            return ActionResult(status="running")
        return ActionResult(status="success")

    def cancel(self, context):
        self._cancel_requested = True


class _SceneGraph:
    def get_nodes(self, node_type=None):
        node = self.get_node("table")
        if node_type in (None, "furniture"):
            return [node]
        return []

    def get_node(self, target):
        if target == "table":
            return SimpleNamespace(id="table", type="furniture", attributes={"label": "table", "name": "table"})
        return None

    def find_nodes(self, **kwargs):
        label = kwargs.get("label") or kwargs.get("name")
        if label == "table":
            return [self.get_node("table")]
        return []

    def get_node_access_poses(self, target):
        if target == "table":
            return [{"x": 1.0, "y": 2.0, "yaw": 0.5, "frame_id": "map", "rank": 0, "valid": True}]
        return []

    def get_best_access_pose(self, target):
        poses = self.get_node_access_poses(target)
        return poses[0] if poses else None

    def get_edges_to(self, target, relation_type=None):
        return []

    def get_node_bbox(self, target):
        return None

    def get_node_visit_locations(self, target):
        if target == "table":
            return [(1.0, 2.0, 0.5)]
        return []


class _FakeSensorDataCenter:
    def __init__(self):
        self.joint_state = None
        self.camera_info = None
        self.robot_pose = None
        self.local_costmap = None
        self.map = None
        self.latest_calls = []
        self.robot_pose_calls = 0

    def get_latest(self, sensor_name, timeout_s=0.2):
        self.latest_calls.append(sensor_name)
        if sensor_name == "joint_states" and self.joint_state is not None:
            return type("SensorMsg", (), {"data": self.joint_state})()
        if sensor_name == "head_camera_info" and self.camera_info is not None:
            return type("SensorMsg", (), {"data": self.camera_info})()
        if sensor_name == "local_costmap" and self.local_costmap is not None:
            return type("SensorMsg", (), {"data": self.local_costmap})()
        if sensor_name == "map" and self.map is not None:
            return type("SensorMsg", (), {"data": self.map})()
        return None

    def get_robot_pose_global(self, global_frame=None, base_frame=None):
        self.robot_pose_calls += 1
        return self.robot_pose

    def transform_data(self, data, src_frame, target_frame, timestamp=None):
        return np.asarray(data, dtype=float)


def _free_grid(width=40, height=40, resolution=0.1, frame_id="map"):
    return SimpleNamespace(
        header=SimpleNamespace(frame_id=frame_id),
        info=SimpleNamespace(
            width=width,
            height=height,
            resolution=resolution,
            origin=SimpleNamespace(position=SimpleNamespace(x=0.0, y=0.0)),
        ),
        data=[0] * (width * height),
    )


class _SlowNavController(MockNavController):
    def __init__(self):
        super().__init__({})
        self.cancel_count = 0

    def get_status(self):
        return self.current_status

    def cancel_goal(self):
        self.cancel_count += 1
        super().cancel_goal()


class _AlwaysIkPlanner(BaseManipulationPlanner):
    def check_ik(self, target, *, context=None, **options):
        return PlannerResult(success=True, metadata={"target": target})

    def check_reachability(self, request, *, context=None):
        return PlannerResult(
            success=True,
            metadata={
                "reach_target": request.reach_target.to_dict(),
                "candidate_base_pose": request.candidate_base.to_dict(),
                "joint_state_seed": request.joint_state_seed,
                "method": "base_reach_test",
            },
        )


class _SoftFailIkPlanner(BaseManipulationPlanner):
    def __init__(self, config=None):
        super().__init__(config)
        self.targets = []

    def check_ik(self, target, *, context=None, **options):
        self.targets.append(target)
        return PlannerResult(
            success=False,
            reason="curobo_ik_not_configured",
            metadata={"target": target, "solver_trace": np.array([1, 2, 3])},
        )

    def check_reachability(self, request, *, context=None):
        target = {
            "reach_target": request.reach_target.to_dict(),
            "candidate_base_pose": request.candidate_base.to_dict(),
            "joint_state_seed": request.joint_state_seed,
        }
        self.targets.append(target)
        return PlannerResult(
            success=False,
            reason="position_reach_plan_failed",
            metadata={"target": target, "solver_trace": np.array([1, 2, 3]), "method": "base_reach_test"},
        )


def _run_until_done(action, context, max_ticks: int = 50):
    result = tick_action(action, context)
    for _ in range(max_ticks - 1):
        if result.status != "running":
            break
        time.sleep(0.05)
        result = tick_action(action, context)
    return result


@pytest.fixture
def manipulation_context() -> Context:
    return Context(
        robot_id="test_bot",
        hardware={"frame_id": "map"},
        modules=ModuleProxy([]),
        nav=MockNavController({}),
        arm=MockArmController({}),
        gripper=MockGripperController({}),
        scene_graph=_SceneGraph(),
        supported_actions={
            "move": MoveToPose,
            "MoveToTarget": MoveToTarget,
            "MoveToManipulationPose": MoveToManipulationPose,
            "grasp_object": GraspObjectAction,
            "release_object": ReleaseObjectAction,
            "MoveArmToState": MoveArmToState,
            "pick_object": PickAtPose,
            "place_object": PlaceAtPose,
            "AskUser": AskUser,
        },
        world_state=WorldState(max_history=10),
    )


class TestPluginActions:
    def test_move_action_sends_nav_goal(self, manipulation_context):
        action = MoveToPose(x=1.0, y=2.0, yaw=0.5, frame_id="map")
        result = tick_action(action, manipulation_context)
        assert result.status == "success"
        assert manipulation_context.nav.last_target == (1.0, 2.0, 0.5)
        assert manipulation_context.nav.last_frame_id == "map"

    def test_move_action_updates_goal_when_already_navigating(self, manipulation_context):
        manipulation_context.nav.current_status = "in_progress"
        action = MoveToPose(x=3.0, y=4.0, yaw=0.0, allow_goal_update=True)
        action.start(manipulation_context)
        assert manipulation_context.nav.last_target == (3.0, 4.0, 0.0)

    def test_move_to_target_puts_arm_in_go_state_before_navigation(self, manipulation_context):
        action = MoveToTarget(target="table")
        assert _run_until_done(action, manipulation_context).status == "success"
        assert manipulation_context.arm.named_goal_history == ["go"]
        assert manipulation_context.nav.last_target == (1.0, 2.0, 0.5)

    def test_align_view_to_scene_target_uses_supporting_furniture_bbox(self, manipulation_context):
        graph = SceneGraph()
        graph.add_node(
            "table_1",
            "furniture",
            {
                "label": "table",
                "name": "Table 1",
                "bbox": [0.8, -0.4, 0.0, 1.4, 0.4, 1.65],
                "access_poses": [{"x": 0.0, "y": 0.0, "yaw": 0.0, "frame_id": "base_link", "rank": 0}],
            },
        )
        graph.add_node("mug_2", "object", {"label": "mug", "name": "Mug 2", "bbox": [1.0, -0.1, 1.65, 1.1, 0.1, 1.8]})
        graph.add_edge("table_1", "mug_2", "supports")
        sdc = _FakeSensorDataCenter()
        sdc.joint_state = SimpleNamespace(
            name=["arm_lift_joint", "head_pan_joint", "head_tilt_joint"],
            position=[0.0, 0.0, -0.3],
        )
        sdc.camera_info = SimpleNamespace(
            width=640,
            height=480,
            k=np.asarray([525.0, 0.0, 320.0, 0.0, 525.0, 240.0, 0.0, 0.0, 1.0]),
        )
        manipulation_context.scene_graph = graph
        manipulation_context.sensor_data_center = sdc

        action = AlignViewToSceneTarget(target="mug_2", frame_id="base_link")
        result = _run_until_done(action, manipulation_context)

        assert result.status == "success"
        assert manipulation_context.arm.last_joint_goal is not None
        assert manipulation_context.arm.last_joint_goal["arm_lift_joint"] > 0.0
        assert set(manipulation_context.arm.last_joint_goal) == {"arm_lift_joint", "head_pan_joint", "head_tilt_joint"}
        assert result.payload["metadata"]["target"] == "table_1"
        assert result.payload["metadata"]["method"] == "fov_coverage"
        assert result.payload["metadata"]["selected_lift"] == pytest.approx(manipulation_context.arm.last_joint_goal["arm_lift_joint"])

    def test_align_view_to_scene_target_lifts_toward_high_surface(self, manipulation_context):
        graph = SceneGraph()
        graph.add_node(
            "shelf_1",
            "furniture",
            {
                "label": "shelf",
                "name": "Shelf 1",
                "bbox": [0.8, -0.3, 0.0, 1.2, 0.3, 1.75],
                "access_poses": [{"x": 0.0, "y": 0.0, "yaw": 0.0, "frame_id": "base_link", "rank": 0}],
            },
        )
        sdc = _FakeSensorDataCenter()
        sdc.joint_state = SimpleNamespace(
            name=["arm_lift_joint", "head_pan_joint", "head_tilt_joint"],
            position=[0.0, 0.0, -0.3],
        )
        sdc.camera_info = SimpleNamespace(
            width=640,
            height=480,
            k=np.asarray([525.0, 0.0, 320.0, 0.0, 525.0, 240.0, 0.0, 0.0, 1.0]),
        )
        manipulation_context.scene_graph = graph
        manipulation_context.sensor_data_center = sdc

        result = _run_until_done(AlignViewToSceneTarget(target="shelf_1", frame_id="base_link"), manipulation_context)

        assert result.status == "success"
        goal = manipulation_context.arm.last_joint_goal
        assert goal is not None
        assert goal["arm_lift_joint"] > 0.5
        assert result.payload["metadata"]["bbox_top_z"] == pytest.approx(1.75)
        assert result.payload["metadata"]["desired_camera_z"] == pytest.approx(1.85)
        assert result.payload["metadata"]["selected_camera_z"] == pytest.approx(1.20 + goal["arm_lift_joint"])

    def test_align_view_to_scene_target_keeps_low_surface_near_current_lift(self, manipulation_context):
        graph = SceneGraph()
        graph.add_node(
            "low_table",
            "furniture",
            {
                "label": "table",
                "name": "Low Table",
                "bbox": [0.8, -0.3, 0.0, 1.2, 0.3, 0.65],
                "access_poses": [{"x": 0.0, "y": 0.0, "yaw": 0.0, "frame_id": "base_link", "rank": 0}],
            },
        )
        sdc = _FakeSensorDataCenter()
        sdc.joint_state = SimpleNamespace(
            name=["arm_lift_joint", "head_pan_joint", "head_tilt_joint"],
            position=[0.12, 0.0, -0.3],
        )
        manipulation_context.scene_graph = graph
        manipulation_context.sensor_data_center = sdc

        result = _run_until_done(AlignViewToSceneTarget(target="low_table", frame_id="base_link"), manipulation_context)

        assert result.status == "success"
        goal = manipulation_context.arm.last_joint_goal
        assert goal is not None
        assert goal["arm_lift_joint"] < 0.06
        assert result.payload["metadata"]["desired_camera_z"] == pytest.approx(1.20)

    def test_align_view_to_scene_target_lifts_even_when_tilt_can_aim(self, manipulation_context):
        graph = SceneGraph()
        graph.add_node(
            "counter_1",
            "furniture",
            {
                "label": "counter",
                "name": "Counter 1",
                "bbox": [0.9, -0.2, 0.0, 1.1, 0.2, 1.55],
                "access_poses": [{"x": 0.0, "y": 0.0, "yaw": 0.0, "frame_id": "base_link", "rank": 0}],
            },
        )
        sdc = _FakeSensorDataCenter()
        sdc.joint_state = SimpleNamespace(
            name=["arm_lift_joint", "head_pan_joint", "head_tilt_joint"],
            position=[0.0, 0.0, 0.0],
        )
        manipulation_context.scene_graph = graph
        manipulation_context.sensor_data_center = sdc

        action = AlignViewToSceneTarget(
            target="counter_1",
            frame_id="base_link",
            lift_samples=9,
            lift_height_weight=12.0,
        )
        result = _run_until_done(action, manipulation_context)

        assert result.status == "success"
        goal = manipulation_context.arm.last_joint_goal
        assert goal is not None
        assert goal["arm_lift_joint"] > 0.35

    def test_align_view_to_scene_target_skips_missing_bbox(self, manipulation_context):
        graph = SceneGraph()
        graph.add_node(
            "table_1",
            "furniture",
            {
                "label": "table",
                "name": "Table 1",
                "access_poses": [{"x": 0.0, "y": 0.0, "yaw": 0.0, "frame_id": "base_link", "rank": 0}],
            },
        )
        manipulation_context.scene_graph = graph

        result = _run_until_done(AlignViewToSceneTarget(target="table_1", frame_id="base_link"), manipulation_context)

        assert result.status == "success"
        assert manipulation_context.arm.last_joint_goal is None

    def test_move_to_manipulation_pose_selects_pose_and_navigates(self, manipulation_context):
        manipulation_context.planners["default"] = _AlwaysIkPlanner({})
        sdc = _FakeSensorDataCenter()
        sdc.map = _free_grid()
        manipulation_context.sensor_data_center = sdc
        target_pose = np.eye(4, dtype=np.float32)
        target_pose[:3, 3] = [1.1, 2.0, 0.75]
        manipulation_context.world_state.set_movable_object_state(
            "mug",
            pose_matrix=target_pose,
            frame_id="map",
            source_action="ObserveScene",
        )
        action = MoveToManipulationPose(
            target="mug",
            min_radius=0.5,
            max_radius=0.5,
            radius_step=0.5,
            preferred_radius=0.5,
            preferred_angles=[0.0],
            refine_iterations=0,
        )

        assert _run_until_done(action, manipulation_context).status == "success"
        stored_pose = manipulation_context.world_state.get("mug_manipulation_base_pose")
        selection = manipulation_context.world_state.get("mug_manipulation_base_selection")
        assert stored_pose is not None
        assert selection["accepted_count"] >= 1
        assert manipulation_context.arm.named_goal_history[-1] == "go"
        assert manipulation_context.nav.last_target == pytest.approx(
            (stored_pose["x"], stored_pose["y"], stored_pose["yaw"])
        )

    def test_move_to_manipulation_pose_records_soft_ik_diagnostics(self, manipulation_context):
        planner = _SoftFailIkPlanner({})
        manipulation_context.planners["default"] = planner
        sdc = _FakeSensorDataCenter()
        sdc.map = _free_grid()
        sdc.joint_state = SimpleNamespace(
            name=["arm_lift_joint", "arm_flex_joint"],
            position=[0.1, -0.2],
        )
        manipulation_context.sensor_data_center = sdc
        target_pose = np.array(
            [
                [0.0, -1.0, 0.0, 1.0],
                [1.0, 0.0, 0.0, 2.0],
                [0.0, 0.0, 1.0, 0.7],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        manipulation_context.world_state.set("mug_pose_mat", target_pose)
        manipulation_context.world_state.set_movable_object_state(
            "mug",
            pose_matrix=target_pose,
            frame_id="map",
            source_action="ObserveScene",
        )
        action = MoveToManipulationPose(
            target="mug",
            min_radius=0.5,
            max_radius=0.5,
            radius_step=0.5,
            preferred_radius=0.5,
            preferred_angles=[0.0],
            refine_iterations=0,
            use_ik=True,
            require_ik=False,
        )

        assert _run_until_done(action, manipulation_context).status == "success"
        selection = manipulation_context.world_state.get("mug_manipulation_base_selection")
        selected_metadata = selection["selected"]["metadata"]
        assert selected_metadata["ik_feasible"] is False
        assert selected_metadata["ik_reason"] == "position_reach_plan_failed"
        assert selected_metadata["ik_metadata"]["planner_metadata"]["solver_trace"] == [1, 2, 3]
        assert planner.targets
        assert planner.targets[0]["reach_target"]["x"] == pytest.approx(1.0)
        assert planner.targets[0]["reach_target"]["y"] == pytest.approx(2.0)
        assert planner.targets[0]["reach_target"]["z"] == pytest.approx(0.82)
        assert planner.targets[0]["reach_target"]["frame_id"] == "map"
        assert planner.targets[0]["candidate_base_pose"]["x"] == pytest.approx(1.5)
        assert planner.targets[0]["joint_state_seed"] == {
            "name": ["arm_lift_joint", "arm_flex_joint"],
            "position": [0.1, -0.2],
        }
        assert "target_pose_matrix" not in planner.targets[0]

    def test_move_to_manipulation_pose_reachability_ignores_object_orientation(self, manipulation_context):
        planner = _SoftFailIkPlanner({})
        manipulation_context.planners["default"] = planner
        sdc = _FakeSensorDataCenter()
        sdc.map = _free_grid()
        manipulation_context.sensor_data_center = sdc
        pose_a = np.eye(4, dtype=np.float32)
        pose_a[:3, 3] = [1.0, 2.0, 0.7]
        pose_b = np.array(
            [
                [0.0, -1.0, 0.0, 1.0],
                [1.0, 0.0, 0.0, 2.0],
                [0.0, 0.0, 1.0, 0.7],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

        for pose in (pose_a, pose_b):
            manipulation_context.world_state.set_movable_object_state(
                "mug",
                pose_matrix=pose,
                frame_id="map",
                source_action="ObserveScene",
            )
            action = MoveToManipulationPose(
                target="mug",
                min_radius=0.5,
                max_radius=0.5,
                radius_step=0.5,
                preferred_radius=0.5,
                preferred_angles=[0.0],
                refine_iterations=0,
                use_ik=True,
                require_ik=False,
            )
            assert _run_until_done(action, manipulation_context).status == "success"

        assert len(planner.targets) == 2
        assert planner.targets[0]["reach_target"]["x"] == pytest.approx(planner.targets[1]["reach_target"]["x"])
        assert planner.targets[0]["reach_target"]["y"] == pytest.approx(planner.targets[1]["reach_target"]["y"])
        assert planner.targets[0]["reach_target"]["z"] == pytest.approx(planner.targets[1]["reach_target"]["z"])
        assert "target_pose_matrix" not in planner.targets[0]
        assert "target_pose_matrix" not in planner.targets[1]

    def test_move_to_manipulation_pose_reach_target_uses_object_pcd_top(self, manipulation_context):
        planner = _SoftFailIkPlanner({})
        manipulation_context.planners["default"] = planner
        sdc = _FakeSensorDataCenter()
        sdc.map = _free_grid()
        manipulation_context.sensor_data_center = sdc
        target_pose = np.eye(4, dtype=np.float32)
        target_pose[:3, 3] = [1.0, 2.0, 0.5]
        manipulation_context.world_state.set("mug_object_pcd", np.array([[1.0, 2.0, 0.45], [1.0, 2.0, 0.9]], dtype=np.float32))
        manipulation_context.world_state.set_movable_object_state(
            "mug",
            pose_matrix=target_pose,
            frame_id="map",
            source_action="ObserveScene",
            object_pcd_key="mug_object_pcd",
        )
        action = MoveToManipulationPose(
            target="mug",
            min_radius=0.5,
            max_radius=0.5,
            radius_step=0.5,
            preferred_radius=0.5,
            preferred_angles=[0.0],
            refine_iterations=0,
            use_ik=True,
            require_ik=False,
            reach_clearance_m=0.07,
        )

        assert _run_until_done(action, manipulation_context).status == "success"

        assert planner.targets[0]["reach_target"]["z"] == pytest.approx(0.97)
        assert planner.targets[0]["reach_target"]["source"] == "object_top_plus_margin"

    def test_move_to_manipulation_pose_observes_when_object_state_is_stale(self, manipulation_context, monkeypatch):
        class FakeObserveScene(AsyncAction):
            def __init__(self, target):
                super().__init__()
                self.target = target
                self.target = target

            def start(self, context):
                pose = np.eye(4, dtype=np.float32)
                pose[:3, 3] = [1.0, 2.0, 0.7]
                context.world_state.set_movable_object_state(
                    self.target,
                    pose_matrix=pose,
                    frame_id="map",
                    source_action="ObserveScene",
                )

            def check_result(self, context):
                return ActionResult(status="success")

            def cancel(self, context):
                self._cancel_requested = True

        import src.plugin.actions.observe as observe_module

        monkeypatch.setattr(observe_module, "ObserveScene", FakeObserveScene)
        sdc = _FakeSensorDataCenter()
        sdc.map = _free_grid()
        manipulation_context.sensor_data_center = sdc
        stale_pose = np.eye(4, dtype=np.float32)
        manipulation_context.world_state.set_movable_object_state(
            "mug",
            pose_matrix=stale_pose,
            frame_id="map",
            timestamp=1.0,
        )
        action = MoveToManipulationPose(
            target="mug",
            min_radius=0.5,
            max_radius=0.5,
            radius_step=0.5,
            preferred_radius=0.5,
            preferred_angles=[0.0],
            refine_iterations=0,
            object_pose_ttl_s=0.001,
        )

        assert _run_until_done(action, manipulation_context).status == "success"
        state = manipulation_context.world_state.get_movable_object_state("mug", ttl_s=None)
        assert state.source_action == "ObserveScene"
        assert manipulation_context.nav.last_target is not None

    def test_move_to_manipulation_pose_missing_nav_data_skips_ik(self, manipulation_context):
        planner = _SoftFailIkPlanner({})
        manipulation_context.planners["default"] = planner
        manipulation_context.sensor_data_center = _FakeSensorDataCenter()
        target_pose = np.eye(4, dtype=np.float32)
        target_pose[:3, 3] = [1.1, 2.0, 0.75]
        manipulation_context.world_state.set_movable_object_state(
            "mug",
            pose_matrix=target_pose,
            frame_id="map",
            source_action="ObserveScene",
        )
        action = MoveToManipulationPose(
            target="mug",
            min_radius=0.5,
            max_radius=0.5,
            radius_step=0.5,
            preferred_radius=0.5,
            preferred_angles=[0.0],
            refine_iterations=0,
            use_ik=True,
            require_ik=True,
        )

        result = _run_until_done(action, manipulation_context)

        assert result.status == "needs_replan"
        assert "missing_local_costmap" in result.reason
        assert planner.targets == []

    def test_move_to_manipulation_pose_scores_from_one_current_pose_snapshot(self, manipulation_context):
        sdc = _FakeSensorDataCenter()
        sdc.map = _free_grid()
        sdc.robot_pose = {"pose": Pose(x=1.6, y=2.0, yaw=math.pi), "frame_id": "map"}
        manipulation_context.sensor_data_center = sdc
        manipulation_context.hardware = {
            "frame_id": "map",
            "manipulation": {
                "robot_distance_weight": 1.0,
                "robot_yaw_change_weight": 1.0,
                "nav_feasibility_missing_policy": "reject",
            },
        }
        target_pose = np.eye(4, dtype=np.float32)
        target_pose[:3, 3] = [1.1, 2.0, 0.75]
        manipulation_context.world_state.set_movable_object_state(
            "mug",
            pose_matrix=target_pose,
            frame_id="map",
            source_action="ObserveScene",
        )
        action = MoveToManipulationPose(
            target="mug",
            min_radius=0.5,
            max_radius=0.5,
            radius_step=0.5,
            preferred_radius=0.5,
            preferred_angles=[0.0],
            refine_iterations=0,
        )

        assert _run_until_done(action, manipulation_context).status == "success"
        selection = manipulation_context.world_state.get("mug_manipulation_base_selection")

        assert sdc.robot_pose_calls == 2
        assert selection["selected"]["metadata"]["robot_distance"] == pytest.approx(0.0, abs=1e-6)


    def test_grasp_and_release(self, manipulation_context):
        grasp = GraspObjectAction()
        assert tick_action(grasp, manipulation_context).status == "success"
        assert manipulation_context.gripper.is_open is False

        release = ReleaseObjectAction()
        assert tick_action(release, manipulation_context).status == "success"
        assert manipulation_context.gripper.is_open is True

    def test_pick_object_runs_arm_and_gripper_phases(self, manipulation_context):
        pick = PickAtPose(object_id="mug", x=0.2, y=0.0, z=0.8, retreat_z=0.05, force_N=0.7)
        assert _run_until_done(pick, manipulation_context).status == "success"
        assert manipulation_context.arm.last_pose is not None
        assert manipulation_context.gripper.last_open_width == 0.3
        assert manipulation_context.gripper.last_close_force == 0.7
        assert [pose.z for pose in manipulation_context.arm.pose_goals] == pytest.approx([0.85, 0.8, 0.85])
        action_names = [record.action_name for record in manipulation_context.world_state.get_action_history()]
        assert "_MoveArmToPose[approach]" in action_names
        assert "_MoveArmToPose[grasp]" in action_names
        assert "_MoveArmToPose[retreat]" in action_names

    def test_pick_object_forwards_looking_hand_constraint_to_pose_moves(self, manipulation_context):
        pick = PickAtPose(
            x=0.2,
            y=0.0,
            z=0.8,
            retreat_z=0.05,
            looking_hand_constraint=True,
        )

        assert _run_until_done(pick, manipulation_context).status == "success"
        assert manipulation_context.arm.pose_goal_options == [
            {"looking_hand_constraint": True, "planning_scope": "arm_only", "segment": "approach"},
            {
                "looking_hand_constraint": True,
                "planning_scope": "arm_only",
                "segment": "grasp",
                "collision_policy": "target_relaxed",
                "collision_relaxation": True,
                "collision_relaxation_objects": ["pick_target_collision"],
            },
            {"looking_hand_constraint": True, "planning_scope": "arm_only", "segment": "retreat"},
        ]

    def test_pick_object_forwards_per_segment_planning_flags(self, manipulation_context):
        pick = PickAtPose(
            x=0.2,
            y=0.0,
            z=0.8,
            retreat_z=0.05,
            retreat_mode="grasp_minus_x",
            grasp_planning_scope="arm_only",
            grasp_planning_motion="cartesian_line",
            retreat_planning_scope="arm_only",
            retreat_planning_motion="cartesian_line",
        )

        assert _run_until_done(pick, manipulation_context).status == "success"
        assert manipulation_context.arm.pose_goal_options == [
            {"looking_hand_constraint": False, "planning_scope": "arm_only", "segment": "approach"},
            {
                "looking_hand_constraint": False,
                "planning_scope": "arm_only",
                "planning_motion": "cartesian_line",
                "segment": "grasp",
                "line_axis": [1.0, 0.0, 0.0],
                "line_distance": 0.05,
                "collision_policy": "target_relaxed",
                "collision_relaxation": True,
                "collision_relaxation_objects": ["pick_target_collision"],
            },
            {
                "looking_hand_constraint": False,
                "planning_scope": "arm_only",
                "planning_motion": "cartesian_line",
                "segment": "retreat",
                "line_axis": [-1.0, 0.0, 0.0],
                "line_distance": 0.05,
            },
        ]

    def test_pick_object_collision_sequence_order(self, manipulation_context, monkeypatch):
        import src.plugin.actions.manipulation as manipulation

        monkeypatch.setattr(
            manipulation,
            "_create_mesh_for_collision_world",
            lambda context, pcd=None, *, convex_hull=False, inflation_factor=1.0: "/tmp/collision.stl",
        )
        manipulation_context.world_state.set("mug_surround_pcd", np.ones((4, 3)))
        manipulation_context.world_state.set("mug_object_pcd", np.ones((4, 3)))
        pick = PickAtPose(object_id="mug", x=0.2, y=0.0, z=0.8, retreat_z=0.05)
        pick.target = "mug"

        assert _run_until_done(pick, manipulation_context).status == "success"
        assert manipulation_context.arm.collision_events == [
            ("add", "pick_surround_collision"),
            ("add", "pick_target_collision"),
            ("remove", "pick_target_collision"),
            ("add", "pick_target_collision"),
            ("clear", None),
        ]

    def test_moveit_pick_keeps_target_collision_until_after_grasp(self, manipulation_context, monkeypatch):
        import src.plugin.actions.manipulation as manipulation

        class MoveItArmController(type(manipulation_context.arm)):
            def __init__(self, hardware_config=None):
                super().__init__(hardware_config)
                self._owned = set()
                self._attached = set()

            def add_collision_mesh(self, mesh_file_name, name: str = "collision_mesh", *, frame_id: str | None = None) -> bool:
                self._owned.add(name)
                return super().add_collision_mesh(mesh_file_name, name=name, frame_id=frame_id)

            def attach_collision_object(self, name: str, link_name: str | None = None, touch_links: list[str] | None = None) -> None:
                self._owned.discard(name)
                self._attached.add(name)
                super().attach_collision_object(name, link_name=link_name, touch_links=touch_links)

            def remove_collision_object(self, name: str) -> None:
                self._owned.discard(name)
                super().remove_collision_object(name)

            def clear_collision_scene(self) -> None:
                for name in sorted(self._attached):
                    super().detach_collision_object(name)
                for name in sorted(self._owned):
                    super().remove_collision_object(name)
                self._attached.clear()
                self._owned.clear()
                super().clear_collision_scene()

        manipulation_context.arm = MoveItArmController({})
        monkeypatch.setattr(
            manipulation,
            "_create_mesh_for_collision_world",
            lambda context, pcd=None, *, convex_hull=False, inflation_factor=1.0: "/tmp/collision.stl",
        )
        manipulation_context.world_state.set("mug_surround_pcd", np.ones((4, 3)))
        manipulation_context.world_state.set("mug_object_pcd", np.ones((4, 3)))
        pick = PickAtPose(object_id="mug", x=0.2, y=0.0, z=0.8, retreat_z=0.05)
        pick.target = "mug"

        assert _run_until_done(pick, manipulation_context).status == "success"
        assert manipulation_context.arm.collision_events == [
            ("add", "pick_surround_collision"),
            ("add", "pick_target_collision"),
            ("attach", "pick_target_collision"),
            ("detach", "pick_target_collision"),
            ("remove", "pick_surround_collision"),
            ("clear", None),
        ]
        assert manipulation_context.arm.allowed_collision_pairs == [
            ("pick_target_collision", "pick_surround_collision")
        ]
        grasp_options = manipulation_context.arm.pose_goal_options[1]
        assert grasp_options["moveit_group"] == "whole_body_weighted"
        assert grasp_options["moveit_goal_mode"] == "ik_joint_goal"
        assert grasp_options["constrain_base"] is True
        assert grasp_options["collision_policy"] == "target_relaxed"
        assert grasp_options["max_base_path_drift"] == {"xy": 0.08, "yaw": 0.12}

    def test_moveit_pick_clears_collision_scene_on_failure(self, manipulation_context, monkeypatch):
        import src.plugin.actions.manipulation as manipulation

        class MoveItArmController(type(manipulation_context.arm)):
            def send_pose_goal(self, request):
                super().send_pose_goal(request)
                if options.get("segment") == "grasp":
                    self.current_status = "failed"

            def get_error(self):
                return "forced grasp failure"

        manipulation_context.arm = MoveItArmController({})
        monkeypatch.setattr(
            manipulation,
            "_create_mesh_for_collision_world",
            lambda context, pcd=None, *, convex_hull=False, inflation_factor=1.0: "/tmp/collision.stl",
        )
        manipulation_context.world_state.set("mug_surround_pcd", np.ones((4, 3)))
        manipulation_context.world_state.set("mug_object_pcd", np.ones((4, 3)))
        pick = PickAtPose(object_id="mug", x=0.2, y=0.0, z=0.8, retreat_z=0.05)
        pick.target = "mug"

        result = _run_until_done(pick, manipulation_context)

        assert result.status == "failure"
        assert ("clear", None) in manipulation_context.arm.collision_events

    def test_pick_object_uses_object_height_for_offset(self, manipulation_context, monkeypatch):
        import src.plugin.actions.manipulation as manipulation

        monkeypatch.setattr(
            manipulation,
            "_create_mesh_for_collision_world",
            lambda context, pcd=None, *, convex_hull=False, inflation_factor=1.0: "/tmp/collision.stl",
        )
        object_points = np.array([[0.2, 0.0, 0.8], [0.2, 0.0, 1.05]], dtype=np.float32)
        manipulation_context.world_state.set("mug_object_pcd", object_points)
        manipulation_context.world_state.set_movable_object_state(
            "mug",
            pose_matrix=np.eye(4),
            frame_id="odom",
            object_pcd_key="mug_object_pcd",
        )

        pick = PickAtPose(object_id="mug", x=0.2, y=0.0, z=0.8, retreat_z=0.05)
        pick.target = "mug"

        assert _run_until_done(pick, manipulation_context).status == "success"
        assert manipulation_context.arm.pose_goals[0].z == pytest.approx(1.11)
        assert manipulation_context.world_state.get("mug_pick_z_increment")["object_height_m"] == pytest.approx(0.25)

    def test_pick_object_grasp_minus_x_offsets_approach_and_retreat(self, manipulation_context):
        pick = PickAtPose(
            x=0.2,
            y=0.0,
            z=0.8,
            retreat_z=0.05,
            retreat_mode="grasp_minus_x",
        )
        assert _run_until_done(pick, manipulation_context).status == "success"
        assert [(pose.x, pose.y, pose.z) for pose in manipulation_context.arm.pose_goals] == [
            pytest.approx((0.15, 0.0, 0.8)),
            pytest.approx((0.2, 0.0, 0.8)),
            pytest.approx((0.15, 0.0, 0.8)),
        ]

    def test_place_object(self, manipulation_context):
        manipulation_context.gripper.is_open = False
        place = PlaceAtPose( x=0.5, y=0.1, z=0.7)
        assert _run_until_done(place, manipulation_context).status == "success"
        assert manipulation_context.gripper.is_open is True
        assert [pose.z for pose in manipulation_context.arm.pose_goals] == pytest.approx([0.8, 0.7, 0.8])

    def test_place_object_forwards_looking_hand_constraint_to_pose_moves(self, manipulation_context):
        manipulation_context.gripper.is_open = False
        place = PlaceAtPose(
            x=0.5,
            y=0.1,
            z=0.7,
            looking_hand_constraint=True,
        )

        assert _run_until_done(place, manipulation_context).status == "success"
        assert manipulation_context.arm.pose_goal_options == [
            {"looking_hand_constraint": True, "segment": "approach"},
            {"looking_hand_constraint": True, "segment": "place"},
            {"looking_hand_constraint": True, "segment": "retreat"},
        ]

    def test_place_object_grasp_minus_x_offsets_approach_and_retreat(self, manipulation_context):
        manipulation_context.gripper.is_open = False
        place = PlaceAtPose(
            x=0.5,
            y=0.1,
            z=0.7,
            retreat_z=0.2,
            retreat_mode="grasp_minus_x",
        )
        assert _run_until_done(place, manipulation_context).status == "success"
        assert [(pose.x, pose.y, pose.z) for pose in manipulation_context.arm.pose_goals] == [
            pytest.approx((0.3, 0.1, 0.7)),
            pytest.approx((0.5, 0.1, 0.7)),
            pytest.approx((0.3, 0.1, 0.7)),
        ]

    def test_pick_target_wraps_pick_with_go_state_cleanup(self, manipulation_context):
        manipulation_context.world_state.set(
            "mug_estimated_pick_pose_mat",
            np.array(
                [
                    [1.0, 0.0, 0.0, 0.2],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.8],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ),
        )
        manipulation_context.world_state.set("mug_estimated_width", 0.04)
        pick = PickTarget(target="mug")

        assert _run_until_done(pick, manipulation_context).status == "success"
        assert manipulation_context.arm.named_goal_history == ["go"]
        assert manipulation_context.gripper.last_open_width == 0.04

    def test_pick_target_forwards_looking_hand_constraint_to_pick_at_pose(self, manipulation_context):
        manipulation_context.world_state.set(
            "mug_estimated_pick_pose_mat",
            np.array(
                [
                    [1.0, 0.0, 0.0, 0.2],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.8],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ),
        )
        manipulation_context.world_state.set("mug_estimated_width", 0.04)
        pick = PickTarget(target="mug", looking_hand_constraint=True)

        assert _run_until_done(pick, manipulation_context).status == "success"
        assert manipulation_context.arm.pose_goal_options == [
            {"looking_hand_constraint": True, "planning_scope": "arm_only", "segment": "approach"},
            {
                "looking_hand_constraint": True,
                "planning_scope": "arm_only",
                "segment": "grasp",
                "collision_policy": "target_relaxed",
                "collision_relaxation": True,
                "collision_relaxation_objects": ["pick_target_collision"],
            },
            {"looking_hand_constraint": True, "planning_scope": "arm_only", "segment": "retreat"},
        ]

    def test_move_arm_to_state_uses_manifest_named_target(self, manipulation_context):
        manipulation_context.arm.hardware_config["named_targets"] = {
            "observe": {
                "type": "joint_positions",
                "joints": {"head_tilt_joint": -0.5, "arm_flex_joint": 0.3},
            }
        }
        action = MoveArmToState(state="observe")
        assert _run_until_done(action, manipulation_context).status == "success"
        assert manipulation_context.arm.last_named_goal == "observe"
        assert manipulation_context.arm.last_joint_goal == {
            "head_tilt_joint": -0.5,
            "arm_flex_joint": 0.3,
        }

    def test_move_arm_to_state_dispatches_even_when_named_target_already_reached(self, manipulation_context):
        manipulation_context.hardware = {"arm": {"named_target_tolerance": 0.03}}
        manipulation_context.arm.hardware_config["named_targets"] = {
            "neutral": {
                "type": "joint_positions",
                "joints": {"arm_flex_joint": 0.0, "wrist_flex_joint": -1.57},
            }
        }
        sdc = _FakeSensorDataCenter()
        sdc.joint_state = type(
            "JointState",
            (),
            {
                "name": ["arm_flex_joint", "wrist_flex_joint", "unrelated_joint"],
                "position": [0.01, -1.56, 2.0],
            },
        )()
        manipulation_context.sensor_data_center = sdc

        action = MoveArmToState(state="neutral")
        result = _run_until_done(action, manipulation_context)

        assert result.status == "success"
        assert manipulation_context.arm.last_named_goal == "neutral"
        assert manipulation_context.arm.last_joint_goal == {
            "arm_flex_joint": 0.0,
            "wrist_flex_joint": -1.57,
        }

    def test_move_arm_to_state_dispatches_when_joint_state_missing(self, manipulation_context):
        manipulation_context.arm.hardware_config["named_targets"] = {
            "neutral": {
                "type": "joint_positions",
                "joints": {"arm_flex_joint": 0.0},
            }
        }
        manipulation_context.sensor_data_center = _FakeSensorDataCenter()

        action = MoveArmToState(state="neutral")
        result = _run_until_done(action, manipulation_context)

        assert result.status == "success"
        assert manipulation_context.arm.last_named_goal == "neutral"

    def test_move_to_pose_succeeds_and_cancels_when_live_pose_within_tolerance(self, manipulation_context):
        nav = _SlowNavController()
        manipulation_context.nav = nav
        manipulation_context.hardware = {
            "nav": {"xy_goal_tolerance": 0.1, "yaw_goal_tolerance": 0.2}
        }
        sdc = _FakeSensorDataCenter()
        sdc.robot_pose = {"pose": Pose(x=1.04, y=2.03, yaw=0.05)}
        manipulation_context.sensor_data_center = sdc

        action = MoveToPose(x=1.0, y=2.0, yaw=0.0, frame_id="map")
        result = tick_action(action, manipulation_context)

        assert result.status == "success"
        assert nav.cancel_count == 1
        assert nav.last_target == (1.0, 2.0, 0.0)

    def test_move_to_pose_outside_live_tolerance_continues_polling_nav(self, manipulation_context):
        nav = _SlowNavController()
        manipulation_context.nav = nav
        sdc = _FakeSensorDataCenter()
        sdc.robot_pose = {"pose": Pose(x=2.0, y=2.0, yaw=0.0)}
        manipulation_context.sensor_data_center = sdc

        action = MoveToPose(x=1.0, y=2.0, yaw=0.0, frame_id="map")
        result = tick_action(action, manipulation_context)

        assert result.status == "running"
        assert nav.cancel_count == 0

    def test_move_base_with_whole_body_uses_relative_goal(self, manipulation_context):
        sdc = _FakeSensorDataCenter()
        sdc.robot_pose = {"pose": Pose(x=1.0, y=2.0, yaw=math.pi / 2)}
        manipulation_context.sensor_data_center = sdc

        action = MoveBaseWithWholeBody(relative_x=-0.4, relative_y=0.0, relative_yaw=0.1, frame_id="odom")
        result = _run_until_done(action, manipulation_context)

        assert result.status == "success"
        goal = manipulation_context.arm.last_base_pose_goal
        assert goal.x == pytest.approx(1.0)
        assert goal.y == pytest.approx(1.6)
        assert goal.yaw == pytest.approx(math.pi / 2 + 0.1)
        assert manipulation_context.arm.base_pose_goal_options["moveit_group"] == "whole_body_weighted"

    def test_move_base_with_whole_body_can_return_to_pick_start(self, manipulation_context):
        manipulation_context.world_state.set(
            "mug_pick_start_base_pose",
            {"x": 0.5, "y": -0.2, "yaw": 0.3, "frame_id": "odom"},
        )

        action = MoveBaseWithWholeBody(target="mug", return_to_pick_start=True, frame_id="odom")
        result = _run_until_done(action, manipulation_context)

        assert result.status == "success"
        goal = manipulation_context.arm.last_base_pose_goal
        assert goal.x == pytest.approx(0.5)
        assert goal.y == pytest.approx(-0.2)
        assert goal.yaw == pytest.approx(0.3)

    def test_ask_user_stores_response_and_requests_replan(self, manipulation_context):
        manipulation_context.request_user_input = lambda prompt: "top shelf"
        action = AskUser(prompt="Which shelf?")
        result = _run_until_done(action, manipulation_context)
        assert result.status == "needs_replan"
        assert result.payload["user_input"] == "top shelf"
        assert manipulation_context.world_state.get("last_user_input") == "top shelf"

    def test_scan_scene_missing_sensor_center_reaches_user_input(self, manipulation_context):
        action = ScanScene(target="mug")
        result = _run_until_done(action, manipulation_context)
        assert result.status == "needs_user_input"
        assert result.suggested_replan_context["reason"] == "missing_sensor_data_center"

    def test_scan_scene_exploration_missing_sensor_center_fails(self, manipulation_context):
        action = ScanScene(target="mug", exploration_mode=True)
        result = _run_until_done(action, manipulation_context)
        assert result.status == "failure"
        assert result.reason == "missing_sensor_data_center"

    def test_describe_scene_missing_sensor_center_fails(self, manipulation_context):
        action = DescribeScene()
        result = _run_until_done(action, manipulation_context)
        assert result.status == "failure"
        assert result.reason == "missing_sensor_data_center"

    def test_head_actions_import_from_head_module(self):
        assert MoveHeadToPanTilt.__module__ == "src.plugin.actions.head"
        assert MoveHeadToState.__module__ == "src.plugin.actions.head"

    def test_tree_builder_instantiates_parametric_action(self, manipulation_context):
        node = ActionNode(
            action_type="move",
            name="go_to_table",
            params={"x": 1.5, "y": 0.0, "yaw": 1.57},
        )
        tree = TreeBuilder(manipulation_context).build(Plan(root=node))
        engine = ExecutionEngine(manipulation_context)
        engine._execute_plan(Plan(root=node))
        assert manipulation_context.nav.last_target == (1.5, 0.0, 1.57)


class TestPluginDiscovery:
    def test_entry_points_register_actions_and_controllers(self):
        registry = PluginRegistry()
        registry.discover()
        for name in (
            "MoveArmToState",
            "AskUser",
            "GraspObjectAction",
            "ReleaseObjectAction",
        ):
            assert registry.get_action(name) is not None
        for name in ("nav2", "moveit_arm", "moveit_gripper"):
            assert registry.get_controller(name) is not None
