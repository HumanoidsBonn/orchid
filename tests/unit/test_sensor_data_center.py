"""Unit tests for SensorDataCenter."""

import math

import numpy as np
import pytest
from unittest.mock import Mock, patch, MagicMock
from src.core.sensor_data_center import SensorDataCenter, SensorMessage
from src.core.manifest import RobotManifest, ControllerMap


class FakeImage:
    """Small ROS Image stand-in for image extraction tests."""

    def __init__(self, array, encoding="rgb8", frame_id="camera_link"):
        self.height = int(array.shape[0])
        self.width = int(array.shape[1])
        self.encoding = encoding
        self.step = int(array.strides[0])
        self.data = array.tobytes()
        self.header = Mock()
        self.header.frame_id = frame_id


class FakeCameraInfo:
    """Small ROS CameraInfo stand-in."""

    def __init__(self):
        self.k = [100.0, 0.0, 1.0, 0.0, 100.0, 1.0, 0.0, 0.0, 1.0]


class FakeNumpyCameraInfo:
    """CameraInfo stand-in with the numpy matrix shape used by some ROS paths."""

    def __init__(self):
        self.k = np.array([525.0, 0.0, 319.5, 0.0, 525.0, 239.5, 0.0, 0.0, 1.0])


@pytest.fixture
def sample_manifest():
    """Create a sample manifest with sensor configuration."""
    return RobotManifest(
        robot_id="test_robot",
        robot_type="mobile_manipulation",
        capabilities=["move", "pick_object"],
        controllers=ControllerMap(nav="nav2", arm="moveit_arm", gripper="moveit_gripper"),
        hardware={
            "sensors": {
                "camera_rgb": "/robot/camera/rgb/image_raw",
                "lidar_scan": "/robot/lidar/scan",
                "imu_data": "/robot/imu/data",
            }
        },
    )


class TestSensorDataCenterSingleton:
    """Test singleton pattern for SensorDataCenter."""

    def teardown_method(self):
        """Clear singleton instances after each test."""
        SensorDataCenter._instances.clear()

    def test_get_or_create_returns_singleton(self, sample_manifest):
        """Verify get_or_create returns same instance on multiple calls."""
        center1 = SensorDataCenter.get_or_create("robot1", sample_manifest)
        center2 = SensorDataCenter.get_or_create("robot1", sample_manifest)
        assert center1 is center2

    def test_get_or_create_different_robots(self, sample_manifest):
        """Verify different robots get different instances."""
        manifest2 = sample_manifest
        manifest2.robot_id = "robot2"

        center1 = SensorDataCenter.get_or_create("robot1", sample_manifest)
        center2 = SensorDataCenter.get_or_create("robot2", manifest2)
        assert center1 is not center2
        assert center1.robot_id == "robot1"
        assert center2.robot_id == "robot2"

    def test_get_instance_returns_existing_or_none(self, sample_manifest):
        """Verify get_instance returns existing instance or None."""
        assert SensorDataCenter.get_instance("robot1") is None

        center = SensorDataCenter.get_or_create("robot1", sample_manifest)
        assert SensorDataCenter.get_instance("robot1") is center


def test_camera_intrinsics_accepts_numpy_camera_matrix():
    sdc = SensorDataCenter.__new__(SensorDataCenter)
    sdc.manifest = RobotManifest(robot_id="test_robot", robot_type="test", hardware={})

    assert sdc._camera_intrinsics(FakeNumpyCameraInfo(), 640, 480) == (525.0, 525.0, 319.5, 239.5)

    def test_thread_safety_on_creation(self, sample_manifest):
        """Verify singleton creation is thread-safe."""
        import threading

        results = []

        def create_instance():
            center = SensorDataCenter.get_or_create("robot1", sample_manifest)
            results.append(center)

        threads = [threading.Thread(target=create_instance) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All should be the same instance
        assert all(r is results[0] for r in results)


class TestSensorTopicParsing:
    """Test sensor topic parsing from manifest."""

    def test_parse_sensor_topics_from_hardware(self, sample_manifest):
        """Verify sensor topics are parsed from hardware config."""
        center = SensorDataCenter("test_robot", sample_manifest)
        assert center._sensor_topics == {
            "camera_rgb": "/robot/camera/rgb/image_raw",
            "lidar_scan": "/robot/lidar/scan",
            "imu_data": "/robot/imu/data",
        }

    def test_parse_sensor_topics_empty_hardware(self):
        """Verify empty topics when no sensors config."""
        manifest = RobotManifest(
            robot_id="test_robot",
            robot_type="mobile_manipulation",
            capabilities=[],
            controllers=ControllerMap(),
            hardware={},
        )
        center = SensorDataCenter("test_robot", manifest)
        assert center._sensor_topics == {}

    def test_parse_sensor_topics_none_hardware(self):
        """Verify empty topics when hardware is None."""
        manifest = RobotManifest(
            robot_id="test_robot",
            robot_type="mobile_manipulation",
            capabilities=[],
            controllers=ControllerMap(),
            hardware=None,
        )
        center = SensorDataCenter("test_robot", manifest)
        assert center._sensor_topics == {}


class TestMessageTypeInference:
    """Test ROS2 message type inference from topic/sensor names."""

    def test_infer_image_type(self):
        """Verify Image type inferred for camera topics."""
        msg_type = SensorDataCenter._infer_message_type("/camera/image", "camera_rgb")
        assert msg_type is not None
        # Message type will be imported if available

    def test_infer_scan_type(self):
        """Verify LaserScan type inferred for lidar topics."""
        msg_type = SensorDataCenter._infer_message_type("/lidar/scan", "lidar_scan")
        assert msg_type is not None

    def test_infer_imu_type(self):
        """Verify Imu type inferred for imu topics."""
        msg_type = SensorDataCenter._infer_message_type("/imu/data", "imu_data")
        assert msg_type is not None

    def test_infer_odom_type(self):
        """Verify Odometry type inferred for odometry topics."""
        msg_type = SensorDataCenter._infer_message_type("/odom", "odometry")
        assert msg_type is not None

    def test_infer_joint_states_type(self):
        """Verify JointState type inferred."""
        msg_type = SensorDataCenter._infer_message_type(
            "/joint_states", "joint_states"
        )
        assert msg_type is not None

    def test_infer_map_type(self):
        """Verify /map is treated as an OccupancyGrid sensor."""
        msg_type = SensorDataCenter._infer_message_type("/map", "map")
        assert msg_type is not None
        assert msg_type.__name__ == "OccupancyGrid"

    def test_infer_local_footprint_type_before_costmap_fallback(self):
        """Verify published footprint is PolygonStamped even under costmap namespace."""
        msg_type = SensorDataCenter._infer_message_type(
            "/local_costmap/published_footprint",
            "local_footprint",
        )

        assert msg_type is not None
        assert msg_type.__name__ == "PolygonStamped"


class TestSensorQoS:
    """Test subscriber QoS selection."""

    def test_map_qos_is_reliable_transient_local_keep_last(self):
        from nav_msgs.msg import OccupancyGrid
        from rclpy.qos import DurabilityPolicy, HistoryPolicy, ReliabilityPolicy

        qos = SensorDataCenter._qos_profile_for("map", "/map", OccupancyGrid)

        assert qos.history == HistoryPolicy.KEEP_LAST
        assert qos.depth == 1
        assert qos.reliability == ReliabilityPolicy.RELIABLE
        assert qos.durability == DurabilityPolicy.TRANSIENT_LOCAL

    def test_local_costmap_qos_is_not_transient_local(self):
        from nav_msgs.msg import OccupancyGrid
        from rclpy.qos import DurabilityPolicy, HistoryPolicy, ReliabilityPolicy

        qos = SensorDataCenter._qos_profile_for("local_costmap", "/local_costmap/costmap", OccupancyGrid)

        assert qos.history == HistoryPolicy.KEEP_LAST
        assert qos.depth == 1
        assert qos.reliability == ReliabilityPolicy.RELIABLE
        assert qos.durability != DurabilityPolicy.TRANSIENT_LOCAL


class TestTimestampExtraction:
    """Test timestamp extraction from ROS2 messages."""

    def test_extract_timestamp_ros2_style(self):
        """Verify ROS2 style timestamp extraction."""
        msg = Mock()
        msg.header = Mock()
        msg.header.stamp = Mock()
        msg.header.stamp.nanoseconds = 1e9  # 1 second

        timestamp = SensorDataCenter._extract_timestamp(msg)
        assert timestamp == 1.0

    def test_extract_timestamp_fallback_to_now(self):
        """Verify fallback to current time if no timestamp in message."""
        msg = Mock(spec=[])  # No attributes
        import time

        before = time.time()
        timestamp = SensorDataCenter._extract_timestamp(msg)
        after = time.time()

        assert before <= timestamp <= after


class TestFrameIdExtraction:
    """Test frame_id extraction from ROS2 messages."""

    def test_extract_frame_id_from_header(self):
        """Verify frame_id extracted from message header."""
        msg = Mock()
        msg.header = Mock()
        msg.header.frame_id = "base_link"

        frame_id = SensorDataCenter._extract_frame_id(msg)
        assert frame_id == "base_link"

    def test_extract_frame_id_none_if_missing(self):
        """Verify None returned if no frame_id."""
        msg = Mock(spec=[])  # No attributes

        frame_id = SensorDataCenter._extract_frame_id(msg)
        assert frame_id is None


class TestSensorMessageCaching:
    """Test sensor message caching behavior."""

    def test_sensor_message_creation(self, sample_manifest):
        """Verify SensorMessage creation."""
        msg = SensorMessage(
            topic="/test/topic",
            data={"key": "value"},
            timestamp=123.456,
            frame_id="base_link",
        )
        assert msg.topic == "/test/topic"
        assert msg.data == {"key": "value"}
        assert msg.timestamp == 123.456
        assert msg.frame_id == "base_link"

    def test_make_callback_caches_message(self, sample_manifest):
        """Verify callback caches sensor messages."""
        center = SensorDataCenter("test_robot", sample_manifest)
        callback = center._make_callback("camera_rgb", "/robot/camera/rgb/image_raw")

        # Create mock message
        msg = Mock()
        msg.header = Mock()
        msg.header.stamp = Mock()
        msg.header.stamp.nanoseconds = 1e9
        msg.header.frame_id = "camera_link"

        callback(msg)

        # Verify message was cached
        assert "camera_rgb" in center._messages
        cached = center._messages["camera_rgb"]
        assert cached.topic == "/robot/camera/rgb/image_raw"
        assert cached.frame_id == "camera_link"

    def test_peek_latest_returns_cached_message_without_consuming(self, sample_manifest):
        """Persistent state reads should not pop cached messages."""
        center = SensorDataCenter("test_robot", sample_manifest)
        cached = SensorMessage(
            topic="/robot/camera/rgb/image_raw",
            data={"value": 1},
            timestamp=1.0,
            frame_id="camera_link",
        )
        center._messages["camera_rgb"] = cached
        center._activate_subscriber = Mock(side_effect=AssertionError("cached peek should not activate"))
        center._deactivate_subscriber = Mock()

        assert center.peek_latest("camera_rgb", timeout_s=0.0) is cached
        assert center._messages["camera_rgb"] is cached
        center._deactivate_subscriber.assert_not_called()

    def test_peek_latest_keeps_subscription_active_after_waiting(self, sample_manifest):
        """peek_latest should activate but not deactivate persistent streams."""
        center = SensorDataCenter("test_robot", sample_manifest)
        cached = SensorMessage(
            topic="/robot/camera/rgb/image_raw",
            data={"value": 2},
            timestamp=2.0,
            frame_id="camera_link",
        )

        def activate(sensor_name, topic):
            center._messages[sensor_name] = cached
            return True

        center._activate_subscriber = Mock(side_effect=activate)
        center._deactivate_subscriber = Mock()

        assert center.peek_latest("camera_rgb", timeout_s=0.05) is cached
        assert center._messages["camera_rgb"] is cached
        center._activate_subscriber.assert_called_once_with("camera_rgb", "/robot/camera/rgb/image_raw")
        center._deactivate_subscriber.assert_not_called()


class TestRobotPoseGlobal:
    """Test robot global pose helper."""

    def test_get_robot_pose_global_uses_tf_defaults(self, sample_manifest):
        sample_manifest.hardware["map"] = "map"
        sample_manifest.hardware["base_frame"] = "base_link"
        center = SensorDataCenter("test_robot", sample_manifest)
        center.get_pose_at = Mock(
            return_value={
                "position": {"x": 1.0, "y": 2.0, "z": 0.0},
                "orientation": {
                    "x": 0.0,
                    "y": 0.0,
                    "z": math.sin(0.5),
                    "w": math.cos(0.5),
                },
                "frame_id": "map",
                "child_frame_id": "base_link",
                "timestamp": 12.0,
            }
        )

        result = center.get_robot_pose_global()

        center.get_pose_at.assert_called_once_with("base_link", "map")
        assert result["pose"].x == pytest.approx(1.0)
        assert result["pose"].y == pytest.approx(2.0)
        assert result["pose"].yaw == pytest.approx(1.0)
        assert result["frame_id"] == "map"
        assert result["child_frame_id"] == "base_link"

    def test_get_robot_pose_global_returns_none_without_tf(self, sample_manifest):
        center = SensorDataCenter("test_robot", sample_manifest)
        center.get_pose_at = Mock(return_value=None)

        assert center.get_robot_pose_global() is None

    def test_sensor_data_center_does_not_own_nav_feasibility_policy(self, sample_manifest):
        center = SensorDataCenter("test_robot", sample_manifest)

        assert not hasattr(center, "check_base_pose_feasible")


class TestROS2NodeCreation:
    """Test ROS2 node linkage and lifecycle."""

    def test_node_creation_is_deferred(self, sample_manifest):
        """Verify the session-owned ROS2 node is linked later via set_node."""
        center = SensorDataCenter("test_robot", sample_manifest)
        assert center.node is None
        assert center._rclpy_node is None

    def test_set_node_links_ros2_node(self, sample_manifest):
        """Verify set_node stores the externally owned ROS2 node."""
        center = SensorDataCenter("test_robot", sample_manifest)
        mock_node = Mock()
        center._ensure_tf_listener = Mock()

        center.set_node(mock_node)

        assert center.node is mock_node
        assert center._rclpy_node is mock_node
        center._ensure_tf_listener.assert_called_once()

    def test_shutdown_cleanup(self, sample_manifest):
        """Verify shutdown cleans up resources."""
        center = SensorDataCenter("test_robot", sample_manifest)
        # Add a mock subscription
        center._active_subscriptions.add("/test/topic")
        center._subscribers["/test/topic"] = Mock()

        center.shutdown()

        # Verify subscriptions cleared
        assert len(center._active_subscriptions) == 0


class TestListActiveSubscriptions:
    """Test listing active subscriptions."""

    def test_list_active_subscriptions_empty(self, sample_manifest):
        """Verify empty list when no active subscriptions."""
        center = SensorDataCenter("test_robot", sample_manifest)
        assert center.list_active_subscriptions() == []

    def test_list_active_subscriptions_with_active(self, sample_manifest):
        """Verify active subscriptions are listed."""
        center = SensorDataCenter("test_robot", sample_manifest)
        center._active_subscriptions.add("/topic1")
        center._active_subscriptions.add("/topic2")

        active = center.list_active_subscriptions()
        assert set(active) == {"/topic1", "/topic2"}


class TestColorAlignedExtraction:
    """Test color-aligned RGB/depth point extraction."""

    def test_extract_color_aligned_points_from_images(self, sample_manifest):
        """Verify RGB+depth extraction returns stable Nx3 points and colors."""
        center = SensorDataCenter("test_robot", sample_manifest)
        rgb = np.zeros((4, 4, 3), dtype=np.uint8)
        rgb[:, :, 0] = 10
        rgb[:, :, 1] = 20
        rgb[:, :, 2] = 30
        depth = np.ones((4, 4), dtype=np.float32)
        mask = [[[1, 1, 2, 1, 2, 2, 1, 2]]]

        result = center.extract_color_aligned_points_from_images(
            rgb_msg=FakeImage(rgb, "rgb8"),
            depth_msg=FakeImage(depth, "32FC1"),
            camera_info_msg=FakeCameraInfo(),
            bbox=[1, 1, 2, 2],
            mask=mask,
        )

        assert result.reason is None
        assert result.method == "rgb_depth"
        assert result.object_points.ndim == 2
        assert result.object_points.shape[1] == 3
        assert result.object_colors.shape[1] == 3
        assert result.object_points.shape[0] > 0

    def test_extract_color_aligned_points_missing_depth_is_recoverable(self, sample_manifest):
        """Missing depth should return a reason instead of throwing."""
        center = SensorDataCenter("test_robot", sample_manifest)
        rgb = np.zeros((4, 4, 3), dtype=np.uint8)

        result = center.extract_color_aligned_points_from_images(
            rgb_msg=FakeImage(rgb, "rgb8"),
            depth_msg=None,
            bbox=[1, 1, 2, 2],
        )

        assert result.reason == "missing_depth"
        assert result.object_points.shape == (0, 3)
        assert result.surround_points.shape == (0, 3)

    def test_transform_data_handles_empty_points(self, sample_manifest):
        """Empty point arrays should stay stable and avoid NumPy shape crashes."""
        center = SensorDataCenter("test_robot", sample_manifest)

        transformed = center.transform_data(np.array([], dtype=np.float32), "camera", "odom")

        assert transformed.shape == (0, 3)


class TestPointCloud2Helpers:
    """Test PointCloud2 construction helpers."""

    def test_points_to_pointcloud2_builds_xyzrgb_when_colors_present(self):
        points = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        colors = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)

        msg = SensorDataCenter.points_to_pointcloud2(points, frame_id="camera", colors=colors)

        assert msg.header.frame_id == "camera"
        assert msg.width == 1
        assert msg.point_step == 16
        assert [field.name for field in msg.fields] == ["x", "y", "z", "rgb"]

    def test_matrix_to_rpy_delegates_to_geometry_helper(self, monkeypatch):
        import src.core.sensor_data_center as sdc_module

        called = {}

        def fake_helper(matrix):
            called["matrix"] = matrix
            return [1, 2, 3, 4, 5, 6]

        monkeypatch.setattr(sdc_module, "matrix_to_xyz_rpy", fake_helper)
        matrix = np.eye(4)

        assert SensorDataCenter.get_tf_matrix_to_xyz_rpy(matrix) == [1, 2, 3, 4, 5, 6]
        assert called["matrix"] is matrix
