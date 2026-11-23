"""Sensor Data Center: ROS2-based sensor aggregation with lazy activation.

The SensorDataCenter manages sensor subscriptions and TF2 transformations for a robot.
It implements a singleton pattern per robot to avoid duplicate subscriptions.

Key features:
- Lazy subscriber activation: subscriptions only active when data is requested
- Dynamic subscriber spawning based on hardware config
- TF2 listener for pose transformations w.r.t. reference frames
- Thread-safe message caching
- On-demand activation/deactivation to minimize resource consumption
- Dedicated ROS2 node created per data center
"""

from __future__ import annotations

import threading
import math
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

import numpy as np

try:  # pragma: no cover - availability depends on sourced ROS environment
    import rclpy
except ImportError:  # pragma: no cover
    rclpy = None

if TYPE_CHECKING:
    from src.core.manifest import RobotManifest

import src.core.logger as log
from src.core.geometry import (
    matrix_to_xyz_rpy,
    quaternion_to_yaw,
    transform_points,
    transform_to_matrix,
)


@dataclass
class SensorMessage:
    """Container for a single sensor message with metadata."""

    topic: str
    data: Any
    timestamp: float
    frame_id: Optional[str] = None


@dataclass
class ColorPointExtraction:
    """Object/surround XYZ and color extraction result."""

    object_points: np.ndarray
    object_colors: np.ndarray
    surround_points: np.ndarray
    surround_colors: np.ndarray
    frame_id: Optional[str]
    method: str
    reason: str | None = None

    @property
    def success(self) -> bool:
        return self.reason is None


class SensorDataCenter:
    """Singleton sensor aggregation hub for a robot.

    Manages:
    - Dynamic ROS2 subscribers for hardware-defined topics
    - TF2 listener for pose transformations
    - Lazy activation (subscribers only active when queried)
    - Thread-safe message caching
    - Dedicated ROS2 node for subscriptions

    Usage:
        center = SensorDataCenter.get_or_create(robot_id, manifest)
        # Subscribers inactive by default
        msg = center.get_latest("sensor_name")  # Activates, fetches, deactivates
        pose = center.get_pose_at("frame", "map", timestamp=1234.5)
    """

    _instances: dict[str, SensorDataCenter] = {}
    _lock = threading.Lock()

    def __init__(self, robot_id: str, manifest: RobotManifest):
        """Initialize sensor data center for a robot.

        Creates a dedicated ROS2 node for sensor subscriptions and TF2 listening.

        Args:
            robot_id: Unique robot identifier
            manifest: RobotManifest declaring hardware/sensors
        """
        self.robot_id = robot_id
        self.manifest = manifest
        self._subscribers: dict[str, Any] = {}  # topic_name -> rclpy.Subscription
        self._tf_listener: Optional[Any] = None
        self._tf_buffer: Optional[Any] = None
        self._messages: dict[str, SensorMessage] = {}
        self._active_subscriptions: set[str] = set()
        self._sub_lock = threading.Lock()
        self._msg_lock = threading.Lock()
        self._rclpy_node: Optional[Any] = None
        self.node: Optional[Any] = None
        self._pose_timer = None
        self._pose_publisher = None
        self._pose_warning_at = float('-inf')
        hardware = manifest.hardware or {}
        self._pose_config = dict(hardware.get('robot_pose_publisher') or {})
        self._pose_frequency = float(self._pose_config.get('frequency_hz', 10.0))
        if not math.isfinite(self._pose_frequency) or self._pose_frequency <= 0:
            raise ValueError('robot_pose_publisher.frequency_hz must be finite and positive')

        # Parse sensor topics from hardware config
        self._sensor_topics = self._parse_sensor_topics(manifest)

        log.info(
            "SensorDataCenter initialized for robot %s with topics: %s",
            robot_id,
            list(self._sensor_topics.keys()),
        )

    def set_node(self, node: Any) -> None:
        """Set the ROS2 node to use for sensor subscriptions and transforms."""
        if self._rclpy_node is node and self._pose_timer is not None:
            return
        if self._rclpy_node is not None and self._rclpy_node is not node:
            raise RuntimeError('SensorDataCenter cannot be rebound to another live node')
        self.node = node
        self._rclpy_node = node
        self._ensure_tf_listener()
        self._start_pose_publisher()
        log.info(
            "ROS2 node linked to SensorDataCenter for robot %s",
            self.robot_id,
        )

    def _start_pose_publisher(self) -> None:
        if self._tf_buffer is None or self._pose_timer is not None:
            return
        from geometry_msgs.msg import PoseStamped
        from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
        from rclpy.clock import Clock, ClockType
        topic = self._pose_config.get('topic', '/global_pose')
        self._pose_publisher = self.node.create_publisher(
            PoseStamped, topic, QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT,
                                           durability=DurabilityPolicy.VOLATILE))
        self._pose_timer = self.node.create_timer(
            1.0 / self._pose_frequency, self._publish_robot_pose,
            clock=Clock(clock_type=ClockType.STEADY_TIME))
        log.info('Robot pose publisher topic=%s frame=map base=%s frequency_hz=%s best_effort',
                 topic, self._pose_config.get('base_frame', (self.manifest.hardware or {}).get('base_frame', 'base_link')),
                 self._pose_frequency)

    def _publish_robot_pose(self) -> None:
        publisher = self._pose_publisher
        if publisher is None:
            return
        from geometry_msgs.msg import PoseStamped
        from rclpy.time import Time
        from rclpy.duration import Duration
        base = self._pose_config.get('base_frame', (self.manifest.hardware or {}).get('base_frame', 'base_link'))
        try:
            transform = self._tf_buffer.lookup_transform('map', base, Time(), timeout=Duration())
        except Exception as exc:
            now = time.monotonic()
            if now - self._pose_warning_at >= 10.0:
                self._pose_warning_at = now
                log.warn('Robot pose TF unavailable map <- %s: %s', base, exc)
            return
        msg = PoseStamped()
        msg.header = transform.header
        msg.pose.position.x = transform.transform.translation.x
        msg.pose.position.y = transform.transform.translation.y
        msg.pose.position.z = transform.transform.translation.z
        msg.pose.orientation = transform.transform.rotation
        publisher.publish(msg)

    @classmethod
    def get_or_create(cls, robot_id: str, manifest: RobotManifest) -> SensorDataCenter:
        """Get or create singleton instance for a robot.

        Args:
            robot_id: Unique robot identifier
            manifest: RobotManifest declaring hardware/sensors

        Returns:
            SensorDataCenter singleton for this robot
        """
        with cls._lock:
            if robot_id not in cls._instances:
                cls._instances[robot_id] = cls(robot_id, manifest)
            return cls._instances[robot_id]

    @classmethod
    def get_instance(cls, robot_id: str) -> Optional[SensorDataCenter]:
        """Get existing instance for a robot, or None.

        Args:
            robot_id: Unique robot identifier

        Returns:
            SensorDataCenter instance or None if not created
        """
        with cls._lock:
            return cls._instances.get(robot_id)

    @staticmethod
    def _parse_sensor_topics(manifest: RobotManifest) -> dict[str, str]:
        """Extract sensor topics from hardware config.

        Expected hardware format:
        {
            "sensors": {
                "camera": "/robot/camera/rgb/image_raw",
                "lidar": "/robot/lidar/scan",
                "imu": "/robot/imu/data"
            }
        }

        Args:
            manifest: RobotManifest with hardware config

        Returns:
            Dict mapping sensor name -> topic name
        """
        topics = {}
        hardware = manifest.hardware or {}
        sensors_cfg = hardware.get("sensors", {})

        if isinstance(sensors_cfg, dict):
            topics.update(sensors_cfg)

        return topics

    def _ensure_tf_listener(self) -> None:
        """Initialize TF2 listener if not already created."""
        if self._tf_listener is not None or self._rclpy_node is None:
            return

        try:
            from tf2_ros import TransformListener, Buffer

            self._tf_buffer = Buffer()
            # RobotSession owns executor spinning for this node; avoid an extra
            # TF spin thread that can outlive tests or shutdown.
            self._tf_listener = TransformListener(self._tf_buffer, self._rclpy_node, spin_thread=False)
            log.info("TF2 listener initialized for robot %s", self.robot_id)
        except ImportError:
            log.warn(
                "tf2_ros not available; pose transformation disabled for robot %s",
                self.robot_id,
            )
        except Exception as e:
            log.warn(
                "Failed to initialize TF2 listener for robot %s: %s",
                self.robot_id,
                e,
            )

    def _activate_subscriber(self, sensor_name: str, topic: str) -> bool:
        """Activate a subscriber for a sensor topic (lazy loading).

        Args:
            sensor_name: Logical name of sensor
            topic: ROS2 topic to subscribe to

        Returns:
            True if activation succeeded, False otherwise
        """
        if self._rclpy_node is None:
            log.warn(
                "Cannot activate subscriber for %s: ROS2 node not set",
                sensor_name,
            )
            return False

        with self._sub_lock:
            if topic in self._active_subscriptions:
                return True

            try:
                # Infer message type from topic name (heuristic)
                msg_type = self._infer_message_type(topic, sensor_name)
                log.debug("msg type for %s %s is %s", topic, sensor_name, msg_type)
                if msg_type is None:
                    log.warn(
                        "Cannot infer message type for topic %s (sensor %s)",
                        topic,
                        sensor_name,
                    )
                    return False

                callback = self._make_callback(sensor_name, topic)
                qos_profile = self._qos_profile_for(sensor_name, topic, msg_type)
                sub = self._rclpy_node.create_subscription(
                    msg_type,
                    topic,
                    callback,
                    qos_profile,
                )
                self._subscribers[topic] = sub
                self._active_subscriptions.add(topic)
                log.debug(
                    "Subscriber activated for %s on topic %s",
                    sensor_name,
                    topic,
                )
                return True
            except Exception as e:
                log.error(
                    "Failed to activate subscriber for %s on %s: %s",
                    sensor_name,
                    topic,
                    e,
                )
                return False

    @staticmethod
    def _qos_profile_for(sensor_name: str, topic: str, msg_type: Any):
        """Return a latest-only QoS profile appropriate for this sensor."""
        from rclpy.qos import (
            DurabilityPolicy,
            HistoryPolicy,
            QoSProfile,
            ReliabilityPolicy,
            qos_profile_sensor_data,
        )

        topic_name = str(topic or "").rstrip("/")
        msg_type_name = getattr(msg_type, "__name__", "")
        if sensor_name == "map" or topic_name == "/map":
            return QoSProfile(
                history=HistoryPolicy.KEEP_LAST,
                depth=1,
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
            )
        if msg_type_name == "OccupancyGrid":
            return QoSProfile(
                history=HistoryPolicy.KEEP_LAST,
                depth=1,
                reliability=ReliabilityPolicy.RELIABLE,
            )
        return qos_profile_sensor_data

    def _deactivate_subscriber(self, topic: str) -> None:
        """Deactivate a subscriber to free resources.

        Args:
            topic: Topic to unsubscribe from
        """
        with self._sub_lock:
            if topic not in self._active_subscriptions:
                return

            if self._rclpy_node is not None and topic in self._subscribers:
                try:
                    self._rclpy_node.destroy_subscription(self._subscribers[topic])
                    log.debug("Subscriber deactivated for topic %s", topic)
                except Exception as e:
                    log.warn(
                        "Failed to destroy subscription for %s: %s",
                        topic,
                        e,
                    )

            self._subscribers.pop(topic, None)
            self._active_subscriptions.discard(topic)

    def _make_callback(self, sensor_name: str, topic: str):
        """Create a message callback that caches the latest message.

        Args:
            sensor_name: Logical sensor name
            topic: Topic name (for logging)

        Returns:
            Callback function
        """

        def callback(msg: Any) -> None:
            try:
                timestamp = self._extract_timestamp(msg)
                frame_id = self._extract_frame_id(msg)
                with self._msg_lock:
                    self._messages[sensor_name] = SensorMessage(
                        topic=topic,
                        data=msg,
                        timestamp=timestamp,
                        frame_id=frame_id,
                    )
                log.debug(
                    "Cached message for sensor %s (topic %s)",
                    sensor_name,
                    topic,
                )
            except Exception as e:
                log.error(
                    "Error processing message for sensor %s: %s",
                    sensor_name,
                    e,
                )

        return callback

    @staticmethod
    def _infer_message_type(topic: str, sensor_name: str) -> Optional[type]:
        """Infer ROS2 message type from topic name or sensor name (heuristic).

        Common patterns:
        - *image*, *camera* -> sensor_msgs/Image
        - *scan*, *lidar* -> sensor_msgs/LaserScan
        - *imu* -> sensor_msgs/Imu
        - *odom*, *odometry* -> nav_msgs/Odometry
        - *tf*, *transform* -> tf2_msgs/TFMessage
        - *joint_states* -> sensor_msgs/JointState
        - *pointcloud*, *cloud* -> sensor_msgs/PointCloud2

        Args:
            topic: Topic name
            sensor_name: Sensor name (as fallback)

        Returns:
            Message type class or None
        """
        try:
            combined = f"{topic} {sensor_name}".lower()

            if "camera_info" in combined or "camera info" in combined:
                from sensor_msgs.msg import CameraInfo

                return CameraInfo
            if any(x in combined for x in ["image", "camera"]):
                from sensor_msgs.msg import Image

                return Image
            elif any(x in combined for x in ["scan", "lidar"]):
                from sensor_msgs.msg import LaserScan

                return LaserScan
            elif "imu" in combined:
                from sensor_msgs.msg import Imu

                return Imu
            elif "footprint" in combined:
                from geometry_msgs.msg import PolygonStamped

                return PolygonStamped
            elif (
                sensor_name in {"map", "local_costmap", "global_costmap"}
                or "occupancygrid" in combined
                or "costmap" in combined
                or topic.rstrip("/") == "/map"
            ):
                from nav_msgs.msg import OccupancyGrid

                return OccupancyGrid
            elif any(x in combined for x in ["odom", "odometry"]):
                from nav_msgs.msg import Odometry

                return Odometry
            elif any(x in combined for x in ["tf", "transform"]):
                from tf2_msgs.msg import TFMessage

                return TFMessage
            elif "joint_states" in combined:
                from sensor_msgs.msg import JointState

                return JointState
            elif any(x in combined for x in ["pointcloud", "cloud", "pcl"]):
                from sensor_msgs.msg import PointCloud2

                return PointCloud2
            else:
                # Default to generic message
                from std_msgs.msg import String

                return String
        except ImportError:
            log.warn("Could not import message type for %s", topic)
            return None

    @staticmethod
    def _extract_timestamp(msg: Any) -> float:
        """Extract timestamp from ROS2 message.

        Args:
            msg: ROS2 message

        Returns:
            Timestamp as float (seconds since epoch)
        """
        try:
            # Try common timestamp locations
            if hasattr(msg, "header") and hasattr(msg.header, "stamp"):
                stamp = msg.header.stamp
                if hasattr(stamp, "nanoseconds") and isinstance(stamp.nanoseconds, (int, float)):
                    # ROS2 style (nanoseconds since epoch)
                    return float(stamp.nanoseconds) / 1e9
                if hasattr(stamp, "sec") and hasattr(stamp, "nsec"):
                    # ROS1 style
                    sec = getattr(stamp, "sec")
                    nsec = getattr(stamp, "nsec")
                    if isinstance(sec, (int, float)) and isinstance(nsec, (int, float)):
                        return float(sec) + float(nsec) / 1e9
            elif hasattr(msg, "stamp"):
                stamp = msg.stamp
                if hasattr(stamp, "nanoseconds") and isinstance(stamp.nanoseconds, (int, float)):
                    return float(stamp.nanoseconds) / 1e9
        except Exception:
            pass

        import time

        return time.time()

    @staticmethod
    def _extract_frame_id(msg: Any) -> Optional[str]:
        """Extract frame_id from ROS2 message if present.

        Args:
            msg: ROS2 message

        Returns:
            Frame ID string or None
        """
        try:
            if hasattr(msg, "header") and hasattr(msg.header, "frame_id"):
                return msg.header.frame_id
        except Exception:
            pass
        return None

    def get_latest(
        self, sensor_name: str, timeout_s: float = 5.0
    ) -> Optional[SensorMessage]:
        """Get latest message from a sensor.

        Activates subscriber, waits for a message, then deactivates.
        This minimizes resource consumption while providing on-demand access.

        Args:
            sensor_name: Logical sensor name (from hardware config)
            timeout_s: How long to wait for a message (seconds)

        Returns:
            SensorMessage with latest data, or None if timeout/error
        """
        if sensor_name not in self._sensor_topics:
            log.warn(
                "Sensor %s not found in configured topics",
                sensor_name,
            )
            return None

        topic = self._sensor_topics[sensor_name]

        # Try to get cached message first
        with self._msg_lock:
            if sensor_name in self._messages:
                msg = self._messages[sensor_name]
                self._messages.pop(sensor_name)  # Clear cache after retrieval
                return msg

        # Activate subscriber
        if not self._activate_subscriber(sensor_name, topic):
            return None

        # Wait for a message with timeout
        import time

        start = time.time()
        data_received = False
        while time.time() - start < timeout_s:
            with self._msg_lock:
                if sensor_name in self._messages:
                    msg = self._messages[sensor_name]
                    self._messages.pop(sensor_name)
                    data_received = True
                    break
            time.sleep(0.01)
        
        if not data_received:
            log.warn(
                "Timeout waiting for sensor %s on topic %s",
                sensor_name,
                topic,
            )
            msg = None

        # Deactivate subscriber
        self._deactivate_subscriber(topic)

        return msg

    def peek_latest(
        self,
        sensor_name: str,
        timeout_s: float = 0.2,
        *,
        activate: bool = True,
        keep_active: bool = True,
    ) -> Optional[SensorMessage]:
        """Return the latest message without consuming it.

        This is intended for state streams such as ``/joint_states`` where
        callers need repeated low-latency reads and should not destroy/recreate
        the subscription on every poll.
        """
        if sensor_name not in self._sensor_topics:
            log.warn(
                "Sensor %s not found in configured topics",
                sensor_name,
            )
            return None

        topic = self._sensor_topics[sensor_name]
        with self._msg_lock:
            msg = self._messages.get(sensor_name)
            if msg is not None:
                return msg

        if not activate:
            return None
        if not self._activate_subscriber(sensor_name, topic):
            return None

        import time

        start = time.time()
        msg = None
        while time.time() - start < timeout_s:
            with self._msg_lock:
                msg = self._messages.get(sensor_name)
                if msg is not None:
                    break
            time.sleep(0.01)

        if msg is None:
            log.warn(
                "Timeout waiting for sensor %s on topic %s",
                sensor_name,
                topic,
            )
            if not keep_active:
                self._deactivate_subscriber(topic)
            return None

        if not keep_active:
            self._deactivate_subscriber(topic)
        return msg

    def to_pcd_stream(self, pcd_np_array):
        import io
        def to_pcd_binary(pcd_array):
                header = (
                    "VERSION .7\n"
                    "FIELDS x y z\n"
                    "SIZE 4 4 4\n"
                    "TYPE F F F\n"
                    "COUNT 1 1 1\n"
                    f"WIDTH {len(pcd_array)}\n"
                    "HEIGHT 1\n"
                    "VIEWPOINT 0 0 0 1 0 0 0\n"
                    f"POINTS {len(pcd_array)}\n"
                    "DATA binary\n"
                )
                data = pcd_array.tobytes()
                return header.encode('ascii') + data

        return io.BytesIO(to_pcd_binary(pcd_np_array))

    @staticmethod
    def empty_points() -> np.ndarray:
        """Return a stable empty XYZ array."""
        return np.empty((0, 3), dtype=np.float32)

    @staticmethod
    def empty_colors() -> np.ndarray:
        """Return a stable empty RGB array."""
        return np.empty((0, 3), dtype=np.float32)

    def extract_color_aligned_points_from_images(
        self,
        *,
        rgb_msg: Any,
        depth_msg: Any,
        bbox: list[float] | tuple[float, float, float, float],
        mask: Any = None,
        camera_info_msg: Any | None = None,
        object_margin: float = 0.1,
        surround_box: float = 2.0,
        min_depth_m: float = 0.05,
        max_depth_m: float = 5.0,
    ) -> ColorPointExtraction:
        """Extract color-aligned 3D points from synchronized RGB/depth images.

        Points are returned in the depth/RGB camera frame. The object region is
        the segmentation mask when available, otherwise the bbox expanded by
        ``object_margin``. The surround region is an expanded bbox excluding the
        object mask.
        """
        empty = ColorPointExtraction(
            self.empty_points(),
            self.empty_colors(),
            self.empty_points(),
            self.empty_colors(),
            frame_id=self._extract_frame_id(depth_msg) or self._extract_frame_id(rgb_msg),
            method="rgb_depth",
        )
        try:
            rgb = self._rgb_image_to_array(rgb_msg)
            depth = self._depth_image_to_meters(depth_msg)
            if rgb is None:
                empty.reason = "missing_rgb"
                return empty
            if depth is None:
                empty.reason = "missing_depth"
                return empty

            img_h = min(int(rgb.shape[0]), int(depth.shape[0]))
            img_w = min(int(rgb.shape[1]), int(depth.shape[1]))
            rgb = rgb[:img_h, :img_w, :3]
            depth = depth[:img_h, :img_w]
            bbox_px = self._normalize_bbox_to_pixels(bbox, img_w, img_h)
            if bbox_px is None:
                empty.reason = "invalid_bbox"
                return empty

            object_mask, surround_mask = self._region_masks(
                bbox_px,
                img_w,
                img_h,
                mask=mask,
                object_margin=object_margin,
                surround_box=surround_box,
            )
            if not np.any(object_mask):
                empty.reason = "empty_object_mask"
                return empty

            fx, fy, cx, cy = self._camera_intrinsics(camera_info_msg, img_w, img_h)
            valid_depth = np.isfinite(depth) & (depth >= min_depth_m) & (depth <= max_depth_m)
            object_pixels = object_mask & valid_depth
            surround_pixels = surround_mask & ~object_mask & valid_depth

            object_points, object_colors = self._pixels_to_points_and_colors(
                object_pixels,
                depth,
                rgb,
                fx,
                fy,
                cx,
                cy,
            )
            surround_points, surround_colors = self._pixels_to_points_and_colors(
                surround_pixels,
                depth,
                rgb,
                fx,
                fy,
                cx,
                cy,
            )
            return ColorPointExtraction(
                object_points,
                object_colors,
                surround_points,
                surround_colors,
                frame_id=self._extract_frame_id(depth_msg) or self._extract_frame_id(rgb_msg),
                method="rgb_depth",
            )
        except Exception as exc:  # noqa: BLE001
            log.exception("RGB/depth point extraction failed: %s", exc)
            empty.reason = "rgb_depth_extraction_failed"
            return empty

    def extract_color_aligned_points_from_pointcloud(
        self,
        *,
        pcl_msg: Any,
        bbox: list[float] | tuple[float, float, float, float],
        mask: Any = None,
        object_margin: float = 0.1,
        surround_box: float = 2.0,
    ) -> ColorPointExtraction:
        """Extract object/surround XYZ and RGB from an organized PointCloud2."""
        empty = ColorPointExtraction(
            self.empty_points(),
            self.empty_colors(),
            self.empty_points(),
            self.empty_colors(),
            frame_id=self._extract_frame_id(pcl_msg),
            method="pointcloud",
        )
        try:
            img_h = int(getattr(pcl_msg, "height", 0) or 0)
            img_w = int(getattr(pcl_msg, "width", 0) or 0)
            if img_h <= 0 or img_w <= 0:
                empty.reason = "invalid_pointcloud_shape"
                return empty
            bbox_px = self._normalize_bbox_to_pixels(bbox, img_w, img_h)
            if bbox_px is None:
                empty.reason = "invalid_bbox"
                return empty
            object_mask, surround_mask = self._region_masks(
                bbox_px,
                img_w,
                img_h,
                mask=mask,
                object_margin=object_margin,
                surround_box=surround_box,
            )

            object_points = []
            object_colors = []
            surround_points = []
            surround_colors = []
            ys, xs = np.where(surround_mask)
            for v, u in zip(ys.tolist(), xs.tolist(), strict=True):
                decoded = self.decode_pcd(pcl_msg, u, v)
                if decoded is None:
                    continue
                x, y, z, color = decoded
                if object_mask[v, u]:
                    object_points.append((x, y, z))
                    object_colors.append(color)
                elif z < 3.0 and x < 3.0 and y < 3.0:  # Filter out far-away points
                    # log.warn("PointCloud2 point at (%d, %d) is too far away: (%.3f, %.3f, %.3f)", u, v, x, y, z)    
                    surround_points.append((x, y, z))
                    surround_colors.append(color)
            return ColorPointExtraction(
                self._as_point_array(object_points),
                self._as_point_array(object_colors),
                self._as_point_array(surround_points),
                self._as_point_array(surround_colors),
                frame_id=self._extract_frame_id(pcl_msg),
                method="pointcloud",
            )
        except Exception as exc:  # noqa: BLE001
            log.exception("PointCloud2 region extraction failed: %s", exc)
            empty.reason = "pointcloud_extraction_failed"
            return empty

    @staticmethod
    def _as_point_array(points: Any) -> np.ndarray:
        array = np.asarray(points, dtype=np.float32)
        if array.size == 0:
            return np.empty((0, 3), dtype=np.float32)
        if array.ndim != 2 or array.shape[1] < 3:
            return np.empty((0, 3), dtype=np.float32)
        return np.ascontiguousarray(array[:, :3], dtype=np.float32)

    @staticmethod
    def _rgb_image_to_array(msg: Any) -> np.ndarray | None:
        if msg is None:
            return None
        height = int(getattr(msg, "height", 0) or 0)
        width = int(getattr(msg, "width", 0) or 0)
        data = getattr(msg, "data", None)
        if height <= 0 or width <= 0 or data is None:
            return None
        encoding = str(getattr(msg, "encoding", "rgb8")).lower()
        channels = 4 if encoding in {"rgba8", "bgra8"} else 3
        array = np.frombuffer(bytes(data), dtype=np.uint8)
        if array.size < height * width * channels:
            return None
        array = array[: height * width * channels].reshape((height, width, channels))
        if encoding.startswith("bgr"):
            array = array[:, :, :3][:, :, ::-1]
        else:
            array = array[:, :, :3]
        return array.astype(np.float32) / 255.0

    @staticmethod
    def _depth_image_to_meters(msg: Any) -> np.ndarray | None:
        if msg is None:
            return None
        height = int(getattr(msg, "height", 0) or 0)
        width = int(getattr(msg, "width", 0) or 0)
        data = getattr(msg, "data", None)
        if height <= 0 or width <= 0 or data is None:
            return None
        encoding = str(getattr(msg, "encoding", "")).lower()
        if encoding in {"16uc1", "mono16"}:
            dtype = ">u2" if getattr(msg, "is_bigendian", False) else "<u2"
            depth = np.frombuffer(bytes(data), dtype=np.dtype(dtype), count=height * width)
            return depth.reshape((height, width)).astype(np.float32) / 1000.0
        if encoding in {"32fc1", "passthrough"}:
            dtype = ">f4" if getattr(msg, "is_bigendian", False) else "<f4"
            depth = np.frombuffer(bytes(data), dtype=np.dtype(dtype), count=height * width)
            return depth.reshape((height, width)).astype(np.float32)
        return None

    def _camera_intrinsics(
        self,
        camera_info_msg: Any | None,
        width: int,
        height: int,
    ) -> tuple[float, float, float, float]:
        if camera_info_msg is not None:
            k = getattr(camera_info_msg, "k", None)
            if k is not None and len(k) >= 6 and float(k[0]) > 0.0 and float(k[4]) > 0.0:
                return float(k[0]), float(k[4]), float(k[2]), float(k[5])
        cfg = ((self.manifest.hardware or {}).get("camera_intrinsics") or {})
        if cfg:
            return (
                float(cfg.get("fx", width)),
                float(cfg.get("fy", width)),
                float(cfg.get("cx", (width - 1) / 2.0)),
                float(cfg.get("cy", (height - 1) / 2.0)),
            )
        focal = float(max(width, height))
        return focal, focal, (width - 1) / 2.0, (height - 1) / 2.0

    @classmethod
    def _normalize_bbox_to_pixels(
        cls,
        bbox: list[float] | tuple[float, float, float, float],
        width: int,
        height: int,
    ) -> tuple[int, int, int, int] | None:
        if bbox is None or len(bbox) != 4:
            return None
        ymin, xmin, ymax, xmax = [float(v) for v in bbox]

        # To cater VLM outputs
        max_coord = max(abs(ymin), abs(xmin), abs(ymax), abs(xmax))
        if max_coord > max(width, height) and max_coord <= 1000.0:
            ymin = ymin / 1000.0 * height
            ymax = ymax / 1000.0 * height
            xmin = xmin / 1000.0 * width
            xmax = xmax / 1000.0 * width

        ymin_i = max(0, min(height - 1, int(round(min(ymin, ymax)))))
        ymax_i = max(0, min(height - 1, int(round(max(ymin, ymax)))))
        xmin_i = max(0, min(width - 1, int(round(min(xmin, xmax)))))
        xmax_i = max(0, min(width - 1, int(round(max(xmin, xmax)))))
        if ymax_i <= ymin_i or xmax_i <= xmin_i:
            return None
        return ymin_i, xmin_i, ymax_i, xmax_i

    @classmethod
    def _region_masks(
        cls,
        bbox: tuple[int, int, int, int],
        width: int,
        height: int,
        *,
        mask: Any = None,
        object_margin: float = 0.1,
        surround_box: float = 2.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        ymin, xmin, ymax, xmax = bbox
        bb_w = max(1, xmax - xmin)
        bb_h = max(1, ymax - ymin)
        if mask:
            obj = cls._mask_to_bool(mask, width, height)
        else:
            margin_x = int(object_margin * bb_w)
            margin_y = int(object_margin * bb_h)
            obj = np.zeros((height, width), dtype=bool)
            obj[
                max(0, ymin - margin_y) : min(height, ymax + margin_y + 1),
                max(0, xmin - margin_x) : min(width, xmax + margin_x + 1),
            ] = True
        surround_margin_x = int(surround_box * bb_w)
        surround_margin_y = int(surround_box * bb_h)
        surround = np.zeros((height, width), dtype=bool)
        surround[
            max(0, ymin - surround_margin_y) : min(height, ymax + surround_margin_y + 1),
            max(0, xmin - surround_margin_x) : min(width, xmax + surround_margin_x + 1),
        ] = True
        return obj, surround

    @staticmethod
    def _mask_to_bool(mask: Any, width: int, height: int) -> np.ndarray:
        result = np.zeros((height, width), dtype=np.uint8)
        try:
            import cv2
        except ImportError:
            return result
        polygons = mask if isinstance(mask, list) else []
        for poly in polygons:
            if (
                isinstance(poly, (list, tuple))
                and len(poly) == 1
                and isinstance(poly[0], (list, tuple))
            ):
                poly = poly[0]
            if not isinstance(poly, (list, tuple)) or len(poly) < 6:
                continue
            coords = np.asarray(poly, dtype=np.float32).reshape((-1, 2))
            if coords.size == 0:
                continue
            if np.nanmax(np.abs(coords)) <= 1000.0 and (
                np.nanmax(coords[:, 0]) > width or np.nanmax(coords[:, 1]) > height
            ):
                coords[:, 0] = coords[:, 0] / 1000.0 * width
                coords[:, 1] = coords[:, 1] / 1000.0 * height
            coords[:, 0] = np.clip(coords[:, 0], 0, width - 1)
            coords[:, 1] = np.clip(coords[:, 1], 0, height - 1)
            cv2.fillPoly(result, [coords.astype(np.int32).reshape((-1, 1, 2))], 1)
        return result.astype(bool)

    @staticmethod
    def _pixels_to_points_and_colors(
        pixel_mask: np.ndarray,
        depth: np.ndarray,
        rgb: np.ndarray,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        vs, us = np.where(pixel_mask)
        if len(us) == 0:
            return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.float32)
        z = depth[vs, us].astype(np.float32)
        x = ((us.astype(np.float32) - float(cx)) * z) / float(fx)
        y = ((vs.astype(np.float32) - float(cy)) * z) / float(fy)
        points = np.stack([x, y, z], axis=1).astype(np.float32)
        colors = rgb[vs, us, :3].astype(np.float32)
        return points, colors

    @staticmethod
    def points_to_pointcloud2(points: Any, frame_id: str = "odom", colors: Any | None = None):
        """Build a minimal XYZ or XYZRGB PointCloud2 message without sensor_msgs_py."""
        from sensor_msgs.msg import PointCloud2, PointField
        from std_msgs.msg import Header

        points_array = np.asarray(points, dtype=np.float32)
        if points_array.ndim != 2 or points_array.shape[1] < 3:
            raise ValueError(f"Invalid point cloud shape {points_array.shape}")
        points_array = np.ascontiguousarray(points_array[:, :3], dtype=np.float32)
        color_array = None
        if colors is not None:
            color_array = np.asarray(colors, dtype=np.float32)
            if color_array.size == 0:
                color_array = None
            elif color_array.ndim != 2 or color_array.shape[1] < 3:
                raise ValueError(f"Invalid color array shape {color_array.shape}")
            else:
                if color_array.shape[0] != points_array.shape[0]:
                    raise ValueError(
                        f"Point/color length mismatch: {points_array.shape[0]} != {color_array.shape[0]}"
                    )
                color_array = np.ascontiguousarray(color_array[:, :3], dtype=np.float32)

        cloud = PointCloud2()
        cloud.header = Header()
        cloud.header.frame_id = frame_id
        cloud.height = 1
        cloud.width = int(points_array.shape[0])
        cloud.is_bigendian = False
        if color_array is None:
            cloud.fields = [
                PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
                PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
                PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            ]
            cloud.point_step = 12
            cloud.data = points_array.tobytes()
        else:
            rgb = SensorDataCenter._pack_rgb_float32(color_array)
            packed = np.empty((points_array.shape[0], 4), dtype=np.float32)
            packed[:, :3] = points_array
            packed[:, 3] = rgb
            cloud.fields = [
                PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
                PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
                PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
                PointField(name="rgb", offset=12, datatype=PointField.FLOAT32, count=1),
            ]
            cloud.point_step = 16
            cloud.data = np.ascontiguousarray(packed).tobytes()
        cloud.row_step = cloud.point_step * cloud.width
        cloud.is_dense = True
        return cloud

    @staticmethod
    def _pack_rgb_float32(colors: np.ndarray) -> np.ndarray:
        """Pack RGB colors into PCL's float32 rgb field representation."""
        color_array = np.asarray(colors, dtype=np.float32)
        if color_array.size == 0:
            return np.empty((0,), dtype=np.float32)
        rgb = np.clip(color_array[:, :3], 0.0, 255.0)
        if float(np.nanmax(rgb)) <= 1.0:
            rgb = rgb * 255.0
        rgb_u32 = rgb.astype(np.uint32)
        packed = (
            ((rgb_u32[:, 0] & 0xFF) << 16)
            | ((rgb_u32[:, 1] & 0xFF) << 8)
            | (rgb_u32[:, 2] & 0xFF)
        ).astype(np.uint32)
        return packed.view(np.float32)
    
    def pcd_to_mesh(self, pcd, *, convex_hull: bool = False, inflation_factor: float = 1.0):
        if convex_hull:
            from src.core.collision_geometry import inflated_convex_hull, write_collision_mesh
            return write_collision_mesh(inflated_convex_hull(pcd, inflation_factor))
        try:
            log.info('Start making mesh')
            from src.core.collision_geometry import voxel_collision_mesh, write_collision_mesh
            combined_mesh = voxel_collision_mesh(pcd)
            temp_stl_path = write_collision_mesh(combined_mesh)
            log.info('Mesh done!')
            return temp_stl_path
        except Exception as e:
            log.exception(f"Error processing point cloud for collision: {e}")
            return None
    
    # @staticmethod
    def decode_pcd(self, pcl_msg, u=None, v=None, offset=None):
        import struct
        if u is not None and v is not None and offset is None:
            offset = (v * pcl_msg.row_step) + (u * pcl_msg.point_step)
        elif offset is None: 
            log.error('Impossible combination of uv and offset!')
            return None
        data_slice = pcl_msg.data[offset : offset + pcl_msg.point_step]

        # Unpack specifically according to offsets:
        # <fff = x, y, z (12 bytes)
        # x = 4 bytes of padding
        # f = rgb (4 bytes)
        # The rest (12 bytes) is more padding.
        x, y, z, rgb_packed = struct.unpack('<fff 4x f', data_slice[:20])
        if np.isnan(x) or np.isnan(y) or np.isnan(z) or z > 1.0 or z < 0.5: return None

        s = struct.pack('<f', rgb_packed)
        i = struct.unpack('<I', s)[0]
        r = (i >> 16) & 0xFF
        g = (i >> 8) & 0xFF
        b = i & 0xFF
        color = (r / 255.0, g / 255.0, b / 255.0)
        return x, y, z, color
    
    @staticmethod
    def get_tf_matrix_to_xyz_rpy(tf_matrix: np.ndarray) -> list:
        """Convert a 4x4 matrix to [X, Y, Z, Roll, Pitch, Yaw] in radians."""
        return matrix_to_xyz_rpy(tf_matrix)

    @staticmethod
    def get_transform_to_matrix(transform = None) -> Optional[np.ndarray]:
        """Return a 4x4 homogeneous matrix from a ROS Transform-like object."""
        return transform_to_matrix(transform)

    def get_pose_at(self, source_frame: str, target_frame: str = "map", timestamp: Optional[float] = None) -> Optional[dict]:
        """Get pose transformation from source frame to target frame.

        Uses TF2 listener to query pose at a specific timestamp.
        If timestamp is None, uses latest available transform.

        Args:
            source_frame: Source frame ID
            target_frame: Target frame ID (default: "map")
            timestamp: Query timestamp (seconds since epoch), or None for latest

        Returns:
            Dict with pose info or None if transform unavailable:
            {
                "position": {"x": float, "y": float, "z": float},
                "orientation": {"x": float, "y": float, "z": float, "w": float},
                "frame_id": str,
                "child_frame_id": str,
                "timestamp": float
            }
        """
        if self._tf_listener is None:
            log.warn("TF2 listener not available for robot %s", self.robot_id)
            return None

        try:
            from rclpy.duration import Duration
            from rclpy.time import Time

            # Build time for query
            if timestamp is not None:
                query_time = Time(seconds=timestamp)
            else:
                query_time = Time()

            transform = self._tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                query_time,
                timeout=Duration(seconds=1.0),
            )
            if transform is None:
                log.warn(
                    "Transform not found: %s -> %s at time %s",
                    source_frame,
                    target_frame,
                    query_time,
                )
                return None

            return {
                "position": {
                    "x": transform.transform.translation.x,
                    "y": transform.transform.translation.y,
                    "z": transform.transform.translation.z,
                },
                "orientation": {
                    "x": transform.transform.rotation.x,
                    "y": transform.transform.rotation.y,
                    "z": transform.transform.rotation.z,
                    "w": transform.transform.rotation.w,
                },
                "frame_id": transform.header.frame_id,
                "child_frame_id": transform.child_frame_id,
                "timestamp": (
                    transform.header.stamp.sec
                    + transform.header.stamp.nanosec / 1e9
                ),
                "mat": self.get_transform_to_matrix(transform.transform)
            }
        except Exception as e:
            log.exception(
                "Transform lookup failed: %s -> %s: %s",
                source_frame,
                target_frame,
                e,
            )
            return None

    def get_robot_pose_global(
        self,
        global_frame: Optional[str] = None,
        base_frame: Optional[str] = None,
    ) -> Optional[dict]:
        """Return the robot base pose in a global frame using TF."""
        hardware = self.manifest.hardware or {}
        target_frame = global_frame or hardware.get("map") or "map"
        source_frame = base_frame or hardware.get("base_frame") or "base_link"
        tf_pose = self.get_pose_at(source_frame, target_frame)
        if tf_pose is None:
            return None

        position = tf_pose.get("position") or {}
        orientation = tf_pose.get("orientation") or {}
        yaw = quaternion_to_yaw(
            float(orientation.get("x", 0.0)),
            float(orientation.get("y", 0.0)),
            float(orientation.get("z", 0.0)),
            float(orientation.get("w", 1.0)),
        )
        from src.core.controllers import Pose

        pose = Pose(
            x=float(position.get("x", 0.0)),
            y=float(position.get("y", 0.0)),
            yaw=yaw,
        )
        return {
            "pose": pose,
            "x": pose.x,
            "y": pose.y,
            "yaw": pose.yaw,
            "frame_id": tf_pose.get("frame_id", target_frame),
            "child_frame_id": tf_pose.get("child_frame_id", source_frame),
            "timestamp": tf_pose.get("timestamp"),
        }

    def transform_data(self, data: np.ndarray, src_frame: str, target_frame: str, timestamp: Optional[float] = None):
        data = self._as_point_array(data)
        if data.shape[0] == 0:
            return data
        if not src_frame or not target_frame or src_frame == target_frame:
            return data
        tf = self.get_pose_at(src_frame, target_frame)
        if tf is not None:
            return transform_points(data, tf.get("mat", np.eye(4)))
        log.warn("Cannot transform point data because TF %s -> %s is unavailable", src_frame, target_frame)
        return self.empty_points()
    
    def list_active_subscriptions(self) -> list[str]:
        """List currently active subscriber topics.

        Returns:
            List of active topic names
        """
        with self._sub_lock:
            return list(self._active_subscriptions)

    def shutdown(self) -> None:
        """Clean up all subscribers and listeners."""
        if self._pose_timer is not None:
            self.node.destroy_timer(self._pose_timer)
            self._pose_timer = None
        if self._pose_publisher is not None:
            self.node.destroy_publisher(self._pose_publisher)
            self._pose_publisher = None
        with self._sub_lock:
            topics = list(self._active_subscriptions)

        for topic in topics:
            self._deactivate_subscriber(topic)

        if self.node is not None:
            try:
                import rclpy

                rclpy.logging.get_logger(__name__).debug(
                    "Destroying ROS2 node for sensor center %s",
                    self.robot_id,
                )
                # Node cleanup is automatic via garbage collection
            except Exception as e:
                log.warn(
                    "Error during ROS2 node cleanup: %s",
                    e,
                )

        log.info("SensorDataCenter shut down for robot %s", self.robot_id)
