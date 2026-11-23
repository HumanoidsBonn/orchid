#!/usr/bin/env python3

from collections import deque

import cv2
import numpy as np
import open3d as o3d

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.qos import qos_profile_sensor_data

from cv_bridge import CvBridge

from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header

from tf2_ros import (
    Buffer,
    TransformListener,
    TransformException,
)


def quaternion_to_rotation_matrix(x, y, z, w):
    q = np.array([x, y, z, w], dtype=np.float64)
    norm = np.linalg.norm(q)

    if norm < 1e-12:
        return np.eye(3)

    q /= norm
    x, y, z, w = q

    return np.array([
        [
            1.0 - 2.0 * (y*y + z*z),
            2.0 * (x*y - z*w),
            2.0 * (x*z + y*w)
        ],
        [
            2.0 * (x*y + z*w),
            1.0 - 2.0 * (x*x + z*z),
            2.0 * (y*z - x*w)
        ],
        [
            2.0 * (x*z - y*w),
            2.0 * (y*z + x*w),
            1.0 - 2.0 * (x*x + y*y)
        ],
    ], dtype=np.float64)


def transform_msg_to_matrix(transform):
    T = np.eye(4, dtype=np.float64)

    t = transform.translation
    q = transform.rotation

    T[:3, :3] = quaternion_to_rotation_matrix(
        q.x, q.y, q.z, q.w
    )

    T[:3, 3] = [
        t.x,
        t.y,
        t.z,
    ]

    return T


class DepthToFilteredPointCloud(Node):

    def __init__(self):
        super().__init__('depth_to_filtered_pointcloud')

        # ==============================================================
        # Parameters
        # ==============================================================

        self.declare_parameter(
            'depth_topic',
            '/camera/depth/image_rect_raw'
        )

        self.declare_parameter(
            'camera_info_topic',
            '/camera/depth/camera_info'
        )

        self.declare_parameter(
            'output_topic',
            '/filtered_pointcloud'
        )

        self.declare_parameter(
            'base_frame',
            'base_link'
        )

        # --------------------------------------------------------------
        # Raw depth filtering [m]
        # --------------------------------------------------------------

        self.declare_parameter('min_depth', 0.20)
        self.declare_parameter('max_depth', 3.0)

        # --------------------------------------------------------------
        # Bounding box in base_frame
        # --------------------------------------------------------------

        self.declare_parameter('min_x', 0.0)
        self.declare_parameter('max_x', 2.0)

        self.declare_parameter('min_y', -1.5)
        self.declare_parameter('max_y', 1.5)

        self.declare_parameter('min_z', 0.02)
        self.declare_parameter('max_z', 2.0)

        # --------------------------------------------------------------
        # Projection/downsampling
        # --------------------------------------------------------------

        self.declare_parameter('pixel_stride', 2)
        self.declare_parameter('voxel_size', 0.015)

        # --------------------------------------------------------------
        # Image-space filtering
        # --------------------------------------------------------------

        self.declare_parameter('median_filter_enable', True)
        self.declare_parameter('median_kernel', 3)

        # Flying-pixel / depth-edge rejection.
        #
        # Reject a depth pixel when its local depth gradient exceeds
        # this value [m].
        #
        # <= 0 disables this filter.
        # --------------------------------------------------------------

        self.declare_parameter(
            'depth_edge_threshold',
            0.05
        )

        # --------------------------------------------------------------
        # Statistical Outlier Removal
        # --------------------------------------------------------------

        self.declare_parameter('sor_enable', True)
        self.declare_parameter('sor_neighbors', 15)
        self.declare_parameter('sor_std_ratio', 1.5)

        # --------------------------------------------------------------
        # Radius Outlier Removal
        # --------------------------------------------------------------

        self.declare_parameter(
            'radius_filter_enable',
            False
        )

        self.declare_parameter(
            'radius_neighbors',
            5
        )

        self.declare_parameter(
            'radius',
            0.03
        )

        # --------------------------------------------------------------
        # TF queue
        # --------------------------------------------------------------

        self.declare_parameter(
            'tf_queue_size',
            10
        )

        # Drop an image if TF is still unavailable after this many sec.
        self.declare_parameter(
            'tf_message_timeout',
            0.5
        )

        # How frequently we check queued images for available TF.
        self.declare_parameter(
            'tf_check_period',
            0.01
        )

        # ==============================================================
        # Get basic params
        # ==============================================================

        self.depth_topic = (
            self.get_parameter('depth_topic')
            .get_parameter_value()
            .string_value
        )

        self.camera_info_topic = (
            self.get_parameter('camera_info_topic')
            .get_parameter_value()
            .string_value
        )

        self.output_topic = (
            self.get_parameter('output_topic')
            .get_parameter_value()
            .string_value
        )

        self.base_frame = (
            self.get_parameter('base_frame')
            .get_parameter_value()
            .string_value
        )

        # ==============================================================
        # State
        # ==============================================================

        self.bridge = CvBridge()

        self.camera_info = None

        # queue elements:
        #
        #   (Image message, time image entered queue)
        #
        self.depth_queue = deque()

        # ==============================================================
        # TF
        # ==============================================================

        self.tf_buffer = Buffer(
            cache_time=Duration(seconds=10.0)
        )

        # Listens to BOTH /tf and /tf_static.
        self.tf_listener = TransformListener(
            self.tf_buffer,
            self,
            spin_thread=True
        )

        # ==============================================================
        # Subscriptions
        # ==============================================================

        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            self.camera_info_topic,
            self.camera_info_callback,
            qos_profile_sensor_data
        )

        self.depth_sub = self.create_subscription(
            Image,
            self.depth_topic,
            self.depth_callback,
            qos_profile_sensor_data
        )

        # ==============================================================
        # Publisher
        # ==============================================================

        self.cloud_pub = self.create_publisher(
            PointCloud2,
            self.output_topic,
            5
        )

        # ==============================================================
        # TF queue processing timer
        # ==============================================================

        tf_check_period = float(
            self.get_parameter('tf_check_period').value
        )

        self.tf_timer = self.create_timer(
            tf_check_period,
            self.process_tf_queue
        )

        self.get_logger().info(
            '\n'
            'Depth -> Filtered PointCloud node\n'
            f'  depth_topic       : {self.depth_topic}\n'
            f'  camera_info_topic : {self.camera_info_topic}\n'
            f'  output_topic      : {self.output_topic}\n'
            f'  base_frame        : {self.base_frame}'
        )

    # ==================================================================
    # Camera info
    # ==================================================================

    def camera_info_callback(self, msg):
        self.camera_info = msg

    # ==================================================================
    # Depth input
    # ==================================================================

    def depth_callback(self, msg):

        if self.camera_info is None:
            self.get_logger().warn(
                'Waiting for CameraInfo...',
                throttle_duration_sec=2.0
            )
            return

        max_queue = int(
            self.get_parameter('tf_queue_size').value
        )

        # Keep the queue bounded.
        while len(self.depth_queue) >= max_queue:
            self.depth_queue.popleft()

        receive_time = self.get_clock().now()

        self.depth_queue.append(
            (msg, receive_time)
        )

    # ==================================================================
    # TF-aware queue
    # ==================================================================

    def process_tf_queue(self):

        if not self.depth_queue:
            return

        timeout_sec = float(
            self.get_parameter('tf_message_timeout').value
        )

        remaining = deque()

        while self.depth_queue:

            msg, receive_time = self.depth_queue.popleft()

            camera_frame = msg.header.frame_id.lstrip('/')

            if not camera_frame:
                camera_frame = (
                    self.camera_info.header.frame_id.lstrip('/')
                )

            now = self.get_clock().now()

            age = (
                now.nanoseconds - receive_time.nanoseconds
            ) / 1e9

            # ----------------------------------------------------------
            # Drop stale image
            # ----------------------------------------------------------

            if age > timeout_sec:
                self.get_logger().warn(
                    f'Dropping depth frame because TF '
                    f'{self.base_frame} <- {camera_frame} '
                    f'was unavailable for {age:.3f} s',
                    throttle_duration_sec=2.0
                )
                continue

            # ----------------------------------------------------------
            # Ask whether transform exists AT IMAGE TIMESTAMP
            # ----------------------------------------------------------

            available = self.tf_buffer.can_transform(
                self.base_frame,
                camera_frame,
                msg.header.stamp,
                timeout=Duration(seconds=0.0)
            )

            if not available:
                remaining.append(
                    (msg, receive_time)
                )
                continue

            # ----------------------------------------------------------
            # Transform now exists
            # ----------------------------------------------------------

            try:
                tf_msg = self.tf_buffer.lookup_transform(
                    self.base_frame,
                    camera_frame,
                    msg.header.stamp,
                )

            except TransformException as exc:
                remaining.append(
                    (msg, receive_time)
                )
                continue

            self.process_depth_image(
                msg,
                tf_msg
            )

        self.depth_queue = remaining

    # ==================================================================
    # Main processing
    # ==================================================================

    def process_depth_image(self, msg, tf_msg):

        try:
            depth = self.bridge.imgmsg_to_cv2(
                msg,
                desired_encoding='passthrough'
            )

        except Exception as exc:
            self.get_logger().error(
                f'cv_bridge conversion failed: {exc}'
            )
            return

        # --------------------------------------------------------------
        # Convert depth to metres
        # --------------------------------------------------------------

        if depth.dtype == np.uint16:
            depth_m = (
                depth.astype(np.float32) * 0.001
            )
        else:
            depth_m = depth.astype(np.float32)

        # --------------------------------------------------------------
        # Median filter
        # --------------------------------------------------------------

        if bool(
            self.get_parameter(
                'median_filter_enable'
            ).value
        ):
            kernel = int(
                self.get_parameter(
                    'median_kernel'
                ).value
            )

            if kernel >= 3 and kernel % 2 == 1:
                depth_m = cv2.medianBlur(
                    depth_m,
                    kernel
                )

        # --------------------------------------------------------------
        # Flying-pixel / edge filtering
        # --------------------------------------------------------------

        edge_threshold = float(
            self.get_parameter(
                'depth_edge_threshold'
            ).value
        )

        edge_valid = np.ones(
            depth_m.shape,
            dtype=bool
        )

        if edge_threshold > 0.0:

            dx = np.abs(
                np.diff(
                    depth_m,
                    axis=1,
                    prepend=depth_m[:, :1]
                )
            )

            dy = np.abs(
                np.diff(
                    depth_m,
                    axis=0,
                    prepend=depth_m[:1, :]
                )
            )

            edge_valid = (
                (dx < edge_threshold) &
                (dy < edge_threshold)
            )

        # --------------------------------------------------------------
        # Project depth -> XYZ camera frame
        # --------------------------------------------------------------

        points_camera = self.depth_to_points(
            depth_m,
            edge_valid
        )

        if len(points_camera) == 0:
            return

        # --------------------------------------------------------------
        # Camera -> base frame
        # --------------------------------------------------------------

        T_base_camera = transform_msg_to_matrix(
            tf_msg.transform
        )

        points_base = self.transform_points(
            points_camera,
            T_base_camera
        )

        # --------------------------------------------------------------
        # Bounding box IN BASE FRAME
        # --------------------------------------------------------------

        points_base = self.apply_bbox(
            points_base
        )

        if len(points_base) == 0:
            return

        # --------------------------------------------------------------
        # PCD filtering
        # --------------------------------------------------------------

        points_base = self.filter_pointcloud(
            points_base
        )

        if len(points_base) == 0:
            return

        # --------------------------------------------------------------
        # Publish
        # --------------------------------------------------------------

        header = Header()

        # Keep sensor timestamp.
        header.stamp = msg.header.stamp
        header.frame_id = self.base_frame

        cloud_msg = point_cloud2.create_cloud_xyz32(
            header,
            points_base
        )

        self.cloud_pub.publish(
            cloud_msg
        )

    # ==================================================================
    # Depth -> XYZ
    # ==================================================================

    def depth_to_points(
        self,
        depth_m,
        edge_valid
    ):

        info = self.camera_info

        fx = info.k[0]
        fy = info.k[4]
        cx = info.k[2]
        cy = info.k[5]

        if fx == 0.0 or fy == 0.0:
            self.get_logger().error(
                'Invalid CameraInfo: fx/fy = 0'
            )
            return np.empty(
                (0, 3),
                dtype=np.float32
            )

        stride = int(
            self.get_parameter(
                'pixel_stride'
            ).value
        )

        stride = max(1, stride)

        min_depth = float(
            self.get_parameter(
                'min_depth'
            ).value
        )

        max_depth = float(
            self.get_parameter(
                'max_depth'
            ).value
        )

        sampled_depth = depth_m[
            ::stride,
            ::stride
        ]

        sampled_edges = edge_valid[
            ::stride,
            ::stride
        ]

        h, w = depth_m.shape

        v, u = np.mgrid[
            0:h:stride,
            0:w:stride
        ]

        valid = (
            np.isfinite(sampled_depth) &
            (sampled_depth > 0.0) &
            (sampled_depth >= min_depth) &
            (sampled_depth <= max_depth) &
            sampled_edges
        )

        if not np.any(valid):
            return np.empty(
                (0, 3),
                dtype=np.float32
            )

        z = sampled_depth[valid]

        u = u[valid].astype(np.float32)
        v = v[valid].astype(np.float32)

        # Standard optical frame:
        #
        #   +X right
        #   +Y down
        #   +Z forward

        x = (
            (u - cx) * z / fx
        )

        y = (
            (v - cy) * z / fy
        )

        return np.column_stack(
            (x, y, z)
        ).astype(np.float32)

    # ==================================================================
    # Transform point array
    # ==================================================================

    @staticmethod
    def transform_points(points, T):

        R = T[:3, :3]
        t = T[:3, 3]

        return (
            points @ R.T + t
        ).astype(np.float32)

    # ==================================================================
    # Base-frame ROI
    # ==================================================================

    def apply_bbox(self, points):

        min_x = float(
            self.get_parameter('min_x').value
        )
        max_x = float(
            self.get_parameter('max_x').value
        )

        min_y = float(
            self.get_parameter('min_y').value
        )
        max_y = float(
            self.get_parameter('max_y').value
        )

        min_z = float(
            self.get_parameter('min_z').value
        )
        max_z = float(
            self.get_parameter('max_z').value
        )

        keep = (
            (points[:, 0] >= min_x) &
            (points[:, 0] <= max_x) &
            (points[:, 1] >= min_y) &
            (points[:, 1] <= max_y) &
            (points[:, 2] >= min_z) &
            (points[:, 2] <= max_z)
        )

        return points[keep]

    # ==================================================================
    # PCD filtering
    # ==================================================================

    def filter_pointcloud(self, points):

        if len(points) == 0:
            return points

        cloud = o3d.geometry.PointCloud()

        cloud.points = (
            o3d.utility.Vector3dVector(
                points.astype(np.float64)
            )
        )

        # --------------------------------------------------------------
        # Voxel downsample
        # --------------------------------------------------------------

        voxel_size = float(
            self.get_parameter(
                'voxel_size'
            ).value
        )

        if voxel_size > 0.0:
            cloud = cloud.voxel_down_sample(
                voxel_size
            )

        # --------------------------------------------------------------
        # Statistical Outlier Removal
        # --------------------------------------------------------------

        if bool(
            self.get_parameter(
                'sor_enable'
            ).value
        ):

            neighbors = int(
                self.get_parameter(
                    'sor_neighbors'
                ).value
            )

            std_ratio = float(
                self.get_parameter(
                    'sor_std_ratio'
                ).value
            )

            if len(cloud.points) > neighbors:

                cloud, _ = (
                    cloud.remove_statistical_outlier(
                        nb_neighbors=neighbors,
                        std_ratio=std_ratio
                    )
                )

        # --------------------------------------------------------------
        # Radius Outlier Removal
        # --------------------------------------------------------------

        if bool(
            self.get_parameter(
                'radius_filter_enable'
            ).value
        ):

            neighbors = int(
                self.get_parameter(
                    'radius_neighbors'
                ).value
            )

            radius = float(
                self.get_parameter(
                    'radius'
                ).value
            )

            if len(cloud.points) > neighbors:

                cloud, _ = (
                    cloud.remove_radius_outlier(
                        nb_points=neighbors,
                        radius=radius
                    )
                )

        return np.asarray(
            cloud.points,
            dtype=np.float32
        )


def main(args=None):

    rclpy.init(args=args)

    node = DepthToFilteredPointCloud()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
