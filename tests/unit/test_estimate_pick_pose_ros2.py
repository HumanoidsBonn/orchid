"""Tests for ROS2-backed GPD pick-pose estimation helpers."""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import numpy as np

from src.core.context import Context
from src.core.module_proxy import ModuleProxy
from src.core.sensor_data_center import SensorDataCenter
from src.core.world_state import WorldState
from src.plugin.actions.estimate_pick_pose_ros2 import DEFAULT_GPD_PRESETS, EstimatePickPoseRos2Service


class _Msg:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


def _install_fake_ros_msgs(monkeypatch):
    geometry_msgs = types.ModuleType("geometry_msgs")
    geometry_msgs_msg = types.ModuleType("geometry_msgs.msg")
    geometry_msgs_msg.Point = type("Point", (_Msg,), {})
    geometry_msgs_msg.Vector3 = type("Vector3", (_Msg,), {})

    class Transform(_Msg):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.translation = _Msg(x=0.0, y=0.0, z=0.0)
            self.rotation = _Msg(x=0.0, y=0.0, z=0.0, w=0.0)

    geometry_msgs_msg.Transform = Transform

    std_msgs = types.ModuleType("std_msgs")
    std_msgs_msg = types.ModuleType("std_msgs.msg")
    std_msgs_msg.Header = type("Header", (_Msg,), {})
    std_msgs_msg.Int64 = type("Int64", (_Msg,), {})

    sensor_msgs = types.ModuleType("sensor_msgs")
    sensor_msgs_msg = types.ModuleType("sensor_msgs.msg")
    sensor_msgs_msg.PointCloud2 = type("PointCloud2", (_Msg,), {})

    class PointField(_Msg):
        FLOAT32 = 7

    sensor_msgs_msg.PointField = PointField

    gpd_msgs = types.ModuleType("gpd_ros2_msgs")
    gpd_msgs_msg = types.ModuleType("gpd_ros2_msgs.msg")

    class CloudSources(_Msg):
        pass

    class CloudIndexed(_Msg):
        pass

    class GraspParams(_Msg):
        pass

    gpd_msgs_msg.CloudSources = CloudSources
    gpd_msgs_msg.CloudIndexed = CloudIndexed
    gpd_msgs_msg.GraspParams = GraspParams

    gpd_msgs_srv = types.ModuleType("gpd_ros2_msgs.srv")

    class DetectConstrainedGrasps:
        class Request(_Msg):
            USE_CFG_FILE = 0
            USE_REQUEST_PARAMS = 1

    gpd_msgs_srv.DetectConstrainedGrasps = DetectConstrainedGrasps

    for name, module in {
        "geometry_msgs": geometry_msgs,
        "geometry_msgs.msg": geometry_msgs_msg,
        "std_msgs": std_msgs,
        "std_msgs.msg": std_msgs_msg,
        "sensor_msgs": sensor_msgs,
        "sensor_msgs.msg": sensor_msgs_msg,
        "gpd_ros2_msgs": gpd_msgs,
        "gpd_ros2_msgs.msg": gpd_msgs_msg,
        "gpd_ros2_msgs.srv": gpd_msgs_srv,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)


def _install_fake_visualization_msgs(monkeypatch):
    visualization_msgs = types.ModuleType("visualization_msgs")
    visualization_msgs_msg = types.ModuleType("visualization_msgs.msg")

    class Marker:
        ARROW = 0
        ADD = 0
        DELETEALL = 3

        def __init__(self):
            self.header = _Msg(frame_id="")
            self.ns = ""
            self.id = 0
            self.type = 0
            self.action = 0
            self.pose = _Msg(
                position=_Msg(x=0.0, y=0.0, z=0.0),
                orientation=_Msg(x=0.0, y=0.0, z=0.0, w=1.0),
            )
            self.scale = _Msg(x=0.0, y=0.0, z=0.0)
            self.color = _Msg(r=0.0, g=0.0, b=0.0, a=0.0)

    class MarkerArray:
        def __init__(self):
            self.markers = []

    visualization_msgs_msg.Marker = Marker
    visualization_msgs_msg.MarkerArray = MarkerArray
    monkeypatch.setitem(sys.modules, "visualization_msgs", visualization_msgs)
    monkeypatch.setitem(sys.modules, "visualization_msgs.msg", visualization_msgs_msg)


class FakeSensorDataCenter:
    @staticmethod
    def points_to_pointcloud2(points, frame_id="odom", colors=None):
        return SensorDataCenter.points_to_pointcloud2(points, frame_id=frame_id, colors=colors)

    def get_pose_at(self, source_frame, target_frame):
        transforms = {
            ("odom", "camera_depth_frame"): {"x": -1.0, "y": 0.0, "z": 0.0},
            ("camera_depth_frame", "odom"): {"x": 1.0, "y": 0.0, "z": 0.0},
            ("camera_depth_frame", "base_link"): {"x": 0.4, "y": 0.0, "z": 1.0},
            ("odom", "base_link"): {"x": -0.6, "y": 0.0, "z": 1.0},
            ("base_link", "odom"): {"x": 0.6, "y": 0.0, "z": -1.0},
        }
        translation = transforms[(source_frame, target_frame)]
        return {
            "position": translation,
            "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
        }


def test_bbox_fraction_indices_selects_object_subregion():
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.5],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    assert EstimatePickPoseRos2Service._bbox_fraction_indices(
        points,
        {"mode": "bbox_fraction", "z": [0.5, 1.0]},
    ) == [1, 2]


def test_all_gpd_presets_disable_plane_sampling():
    assert set(DEFAULT_GPD_PRESETS) == {"fast_debug", "balanced", "high_recall"}
    assert all(preset["sample_above_plane"] == 0 for preset in DEFAULT_GPD_PRESETS.values())
    assert all(preset["voxelize"] == 0 for preset in DEFAULT_GPD_PRESETS.values())
    assert all("sample_radius_m" not in preset for preset in DEFAULT_GPD_PRESETS.values())


def test_sample_downsampling_is_deterministic_without_centroid_radial_crop():
    action = EstimatePickPoseRos2Service(target="cup")
    points = np.array(
        [
            [0.00, 0.00, 0.0],
            [0.01, 0.00, 0.0],
            [0.02, 0.00, 0.0],
            [0.20, 0.00, 0.0],
            [0.21, 0.00, 0.0],
        ],
        dtype=np.float32,
    )
    first = action._sample_object_indices(
        points,
        None,
        {"max_samples": 2},
    )
    second = action._sample_object_indices(
        points,
        None,
        {"max_samples": 2},
    )
    assert first == second
    assert len(first) == 2
    assert all(0 <= idx < len(points) for idx in first)


def test_radius_outlier_filter_removes_sparse_point_and_preserves_colors():
    action = EstimatePickPoseRos2Service(target="cup")
    points = np.array(
        [
            [0.000, 0.000, 0.000],
            [0.005, 0.000, 0.000],
            [0.000, 0.005, 0.000],
            [0.005, 0.005, 0.000],
            [1.000, 1.000, 1.000],
        ],
        dtype=np.float32,
    )
    colors = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    filtered_points, filtered_colors = action._radius_outlier_filter_points(
        points,
        colors,
        {
            "radius_outlier_filter_enabled": True,
            "radius_outlier_radius_m": 0.02,
            "radius_outlier_min_neighbors": 2,
        },
        label="object",
    )

    assert filtered_points.shape == (4, 3)
    assert filtered_colors.shape == (4, 3)
    assert not np.any(np.all(np.isclose(filtered_points, [1.0, 1.0, 1.0]), axis=1))
    assert np.allclose(filtered_colors, colors[:4])


def _cloud_xyz(cloud):
    return np.frombuffer(cloud.data, dtype=np.float32).reshape((cloud.width, cloud.point_step // 4))[:, :3]


def test_build_request_uses_sample_first_cloud_and_contiguous_indices(monkeypatch):
    _install_fake_ros_msgs(monkeypatch)
    context = Context(
        robot_id="test",
        hardware={"base_frame": "base_link"},
        modules=ModuleProxy([]),
        world_state=WorldState(max_history=5),
        sensor_data_center=FakeSensorDataCenter(),
    )
    context.world_state.set("cup_pcd_frame", "odom")
    context.world_state.set("cup_source_camera_frame", "camera_depth_frame")
    action = EstimatePickPoseRos2Service(
        target="cup",
        sample_region={"mode": "bbox_fraction", "z": [0.5, 1.0]},
        max_samples=None,
        radius_outlier_filter_enabled=False,
    )
    action.object_pcd_array = np.array(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.5], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    action.surround_pcd_array = np.array([[1.0, 1.0, 1.0]], dtype=np.float32)
    action.object_colors_array = np.array(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    action.surround_colors_array = np.array([[1.0, 1.0, 0.0]], dtype=np.float32)

    request = action._build_request(context)

    assert request.cloud_indexed.cloud_sources.cloud.width == 4
    assert request.cloud_indexed.cloud_sources.cloud.header.frame_id == "camera_depth_frame"
    assert [idx.data for idx in request.cloud_indexed.indices] == [0, 1]
    assert np.allclose(action._sample_points_array, [[-1.0, 0.0, 0.5], [-1.0, 0.0, 1.0]])
    xyz = _cloud_xyz(request.cloud_indexed.cloud_sources.cloud)
    assert np.allclose(xyz[:2], action._sample_points_array)
    assert np.allclose(xyz[2:], [[-1.0, 0.0, 0.0], [0.0, 1.0, 1.0]])
    assert np.allclose(action._object_colors_camera, [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
    assert np.allclose(action._surround_colors_camera, [[1.0, 1.0, 0.0]])
    assert [source.data for source in request.cloud_indexed.cloud_sources.camera_source] == [0, 0, 0, 0]
    assert request.cloud_indexed.cloud_sources.view_points[0].x == 0.0
    assert request.grasp_params.camera_position == [0.0, 0.0, 0.0]
    assert request.params_policy == request.USE_REQUEST_PARAMS


def test_build_request_uses_base_frame_workspace_and_camera_to_base_tf(monkeypatch):
    _install_fake_ros_msgs(monkeypatch)

    context = Context(
        robot_id="test",
        hardware={"base_frame": "base_link"},
        modules=ModuleProxy([]),
        world_state=WorldState(max_history=5),
        sensor_data_center=FakeSensorDataCenter(),
    )
    context.world_state.set("cup_pcd_frame", "odom")
    context.world_state.set("cup_source_camera_frame", "camera_depth_frame")
    action = EstimatePickPoseRos2Service(
        target="cup",
        workspace_margin=0.1,
        max_samples=None,
        radius_outlier_filter_enabled=False,
    )
    action.object_pcd_array = np.array(
        [[1.0, 2.0, 3.0], [2.0, 4.0, 6.0]],
        dtype=np.float32,
    )
    action.surround_pcd_array = np.empty((0, 3), dtype=np.float32)

    request = action._build_request(context)

    assert request.cloud_indexed.cloud_sources.view_points[0].x == 0.0
    assert request.grasp_params.camera_position == [0.0, 0.0, 0.0]
    assert request.grasp_params.transform_camera2base.translation.x == 0.4
    assert request.grasp_params.transform_camera2base.translation.z == 1.0
    assert np.allclose(
        request.grasp_params.workspace,
        [0.3, 1.5, 1.9, 4.1, 3.9, 7.1],
    )
    assert action._request_metadata["cloud_frame"] == "camera_depth_frame"
    assert action._request_metadata["world_frame"] == "odom"
    assert action._request_metadata["base_frame"] == "base_link"


def test_build_request_requires_pcd_frame(monkeypatch):
    _install_fake_ros_msgs(monkeypatch)
    context = Context(
        robot_id="test",
        hardware={"base_frame": "base_link"},
        modules=ModuleProxy([]),
        world_state=WorldState(max_history=5),
        sensor_data_center=FakeSensorDataCenter(),
    )
    action = EstimatePickPoseRos2Service(target="cup", max_samples=None, radius_outlier_filter_enabled=False)
    action.object_pcd_array = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    action.surround_pcd_array = np.empty((0, 3), dtype=np.float32)

    try:
        action._build_request(context)
    except ValueError as exc:
        assert "cup_pcd_frame" in str(exc)
    else:
        raise AssertionError("Expected missing pcd frame to fail")


def test_build_request_requires_source_camera_frame(monkeypatch):
    _install_fake_ros_msgs(monkeypatch)
    context = Context(
        robot_id="test",
        hardware={"base_frame": "base_link"},
        modules=ModuleProxy([]),
        world_state=WorldState(max_history=5),
        sensor_data_center=FakeSensorDataCenter(),
    )
    context.world_state.set("cup_pcd_frame", "odom")
    action = EstimatePickPoseRos2Service(target="cup", max_samples=None, radius_outlier_filter_enabled=False)
    action.object_pcd_array = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    action.surround_pcd_array = np.empty((0, 3), dtype=np.float32)

    try:
        action._build_request(context)
    except ValueError as exc:
        assert "cup_source_camera_frame" in str(exc)
    else:
        raise AssertionError("Expected missing source camera frame to fail")


def test_build_request_sends_xyzrgb_cloud(monkeypatch):
    _install_fake_ros_msgs(monkeypatch)
    context = Context(
        robot_id="test",
        hardware={"base_frame": "base_link"},
        modules=ModuleProxy([]),
        world_state=WorldState(max_history=5),
        sensor_data_center=FakeSensorDataCenter(),
    )
    context.world_state.set("cup_pcd_frame", "odom")
    context.world_state.set("cup_source_camera_frame", "camera_depth_frame")
    action = EstimatePickPoseRos2Service(target="cup", max_samples=None, radius_outlier_filter_enabled=False)
    action.object_pcd_array = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    action.surround_pcd_array = np.empty((0, 3), dtype=np.float32)
    action.object_colors_array = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)

    request = action._build_request(context)

    field_names = [field.name for field in request.cloud_indexed.cloud_sources.cloud.fields]
    assert field_names == ["x", "y", "z", "rgb"]
    assert request.cloud_indexed.cloud_sources.cloud.point_step == 16
    assert action._request_metadata["has_color"] is True


def test_extract_grasp_configs_response_to_arrays():
    action = EstimatePickPoseRos2Service(target="cup")
    response = {
        "grasp_configs": {
            "grasps": [
                {
                    "position": {"x": 1.0, "y": 2.0, "z": 3.0},
                    "approach": {"x": 1.0, "y": 0.0, "z": 0.0},
                    "binormal": {"x": 0.0, "y": 1.0, "z": 0.0},
                    "axis": {"x": 0.0, "y": 0.0, "z": 1.0},
                    "width": {"data": 0.08},
                    "score": {"data": 0.9},
                }
            ]
        }
    }

    tf_matrices, scores, widths = action._extract_data_from_response(response)

    assert np.allclose(tf_matrices[0][:3, 3], [1.0, 2.0, 3.0])
    assert scores.tolist() == [0.9]
    assert widths.tolist() == [0.08]


def test_response_callback_stores_filtered_candidates_and_selects_best():
    ws = WorldState(max_history=5)
    action = EstimatePickPoseRos2Service(
        target="cup",
        min_score=0.5,
        min_grasp_width=0.02,
        max_grasp_width=0.1,
        debug_visualization=False,
    )
    action.ws = ws
    action.object_pcd_array = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    action.surround_pcd_array = np.empty((0, 3), dtype=np.float32)
    action._request_metadata = {
        "camera_position_world": [0.0, 0.0, 0.0],
        "cloud_frame": "camera_depth_frame",
        "world_frame": "odom",
        "transform_camera2world": np.eye(4, dtype=np.float32).tolist(),
    }

    action.response_callback(
        {
            "grasp_configs": {
                "grasps": [
                    {
                        "position": {"x": 1.0, "y": 0.0, "z": 0.1},
                        "approach": {"x": 1.0, "y": 0.0, "z": 0.0},
                        "binormal": {"x": 0.0, "y": 1.0, "z": 0.0},
                        "axis": {"x": 0.0, "y": 0.0, "z": 1.0},
                        "width": {"data": 0.05},
                        "score": {"data": 0.9},
                    },
                    {
                        "position": {"x": 1.0, "y": 0.0, "z": 0.1},
                        "approach": {"x": 1.0, "y": 0.0, "z": 0.0},
                        "binormal": {"x": 0.0, "y": 1.0, "z": 0.0},
                        "axis": {"x": 0.0, "y": 0.0, "z": 1.0},
                        "width": {"data": 0.2},
                        "score": {"data": 0.95},
                    },
                ]
            }
        }
    )

    filtered = ws.get("cup_gpd_filtered_candidates")
    selected = ws.get("cup_selected_grasp_candidate")
    assert action.status == "success"
    assert filtered["accepted_count"] == 1
    assert filtered["rejected_count"] == 1
    assert selected["index"] == 0
    assert ws.get("cup_estimated_width") == 0.05
    assert ws.get("cup_estimated_pick_pose_frame") == "odom"
    pose = ws.get("cup_estimated_pick_pose_mat")
    assert np.allclose(pose[:3, 2], [1.0, 0.0, 0.0])
    assert ws.get("cup_selected_grasp_alignment")["ee_x_up_dot"] > 0.0


def test_response_callback_fails_on_frame_mismatch():
    ws = WorldState(max_history=5)
    action = EstimatePickPoseRos2Service(target="cup", debug_visualization=False)
    action.ws = ws
    action.object_pcd_array = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    action.surround_pcd_array = np.empty((0, 3), dtype=np.float32)
    action._request_metadata = {
        "cloud_frame": "camera_depth_frame",
        "world_frame": "odom",
        "transform_camera2world": np.eye(4, dtype=np.float32).tolist(),
    }

    action.response_callback(
        {
            "grasp_configs": {
                "header": {"frame_id": "base_link"},
                "grasps": [
                    {
                        "position": {"x": 1.0, "y": 0.0, "z": 0.1},
                        "approach": {"x": 1.0, "y": 0.0, "z": 0.0},
                        "binormal": {"x": 0.0, "y": 1.0, "z": 0.0},
                        "axis": {"x": 0.0, "y": 0.0, "z": 1.0},
                        "width": {"data": 0.05},
                        "score": {"data": 0.9},
                    }
                ],
            }
        }
    )

    assert action.status == "failure"
    assert "gpd_response_frame_mismatch" in action._failure_reason


def test_gpd_debug_uses_single_fixed_marker_array_topic(monkeypatch):
    _install_fake_visualization_msgs(monkeypatch)

    class FakePublisher:
        def __init__(self, topic):
            self.topic = topic
            self.messages = []

        def publish(self, msg):
            self.messages.append(msg)

    class FakeNode:
        def __init__(self):
            self.publishers = {}

        def create_publisher(self, msg_type, topic, depth):
            publisher = FakePublisher(topic)
            self.publishers[topic] = publisher
            return publisher

    node = FakeNode()
    action = EstimatePickPoseRos2Service(target="cup", debug_visualization=True)
    action._context = Context(
        robot_id="test",
        hardware={},
        modules=ModuleProxy([]),
        world_state=WorldState(max_history=5),
        sensor_data_center=SimpleNamespace(node=node),
    )
    action._request_metadata = {"cloud_frame": "camera_depth_frame"}
    accepted = SimpleNamespace(candidate=SimpleNamespace(index=0))
    rejected = SimpleNamespace(candidate=SimpleNamespace(index=1))
    filter_result = SimpleNamespace(accepted=[accepted], rejected=[rejected])
    grasp = np.eye(4, dtype=np.float32)

    action._publish_debug_candidates(
        [grasp, grasp],
        np.array([0.9, 0.1], dtype=np.float32),
        best_idx=0,
        filter_result=filter_result,
    )

    assert set(node.publishers.keys()) == {"/orchid/gpd/grasps"}
    markers = node.publishers["/orchid/gpd/grasps"].messages[-1].markers
    assert {marker.ns for marker in markers} == {"raw", "accepted", "rejected", "selected"}
