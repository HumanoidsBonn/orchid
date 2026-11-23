"""EstimatePickPoseRos2Service action.

This async action retrieves the target object's stored point-cloud information and
calls a ROS2 service-backed GPD implementation instead of the Flask endpoint.
The request is assembled in the action layer, while the ROS2 service dispatch and
spinning live in the shared ROS2 service module.
"""

from __future__ import annotations

import json
import random
import time
from collections import Counter
from typing import Any

import numpy as np
import open3d as o3d

from src.core.action import ActionResult, AsyncAction
from src.core.context import Context
from src.core.geometry import (
    pose_dict_to_matrix,
    rotation_matrix_to_quaternion,
    transform_direction,
    transform_points,
    transform_pose_matrix,
)
from src.core.sensor_data_center import SensorDataCenter
from src.plugin.grasp_filter import (
    GraspFilterConfig,
    GraspFilterResult,
    filter_and_rank_grasps,
    infer_support_z,
)
import src.core.logger as log

DEFAULT_GPD_SERVICE_MODULE = "gpd_ros2_service"
DEFAULT_GPD_PRESET = "fast_debug"
DEFAULT_GPD_PRESETS: dict[str, dict[str, Any]] = {
    "fast_debug": {
        "max_samples": 50,
        # "max_samples": 150,
        "sample_above_plane": 0,
        "voxelize": 0,
        "radius_outlier_filter_enabled": True,
        "radius_outlier_radius_m": 0.035,
        "radius_outlier_min_neighbors": 4,
        "num_orientations": 10,
        "num_finger_placements": 10,
        "num_selected": 15,
        # "voxel_size": 0.003,
        "voxel_size": 0.02,
        "min_grasp_width": 0.0,
        "max_grasp_width": 0.2,
        "approach_direction": [0.0, -1.0, 0.0],
        "approach_threshold_rad": None,
        "workspace_margin": 0.05,
    },
    "balanced": {
        "max_samples": 300,
        "sample_above_plane": 0,
        "voxelize": 0,
        "radius_outlier_filter_enabled": True,
        "radius_outlier_radius_m": 0.015,
        "radius_outlier_min_neighbors": 4,
        "num_orientations": 8,
        "num_finger_placements": 8,
        "num_selected": 10,
        "voxel_size": 0.003,
        "min_grasp_width": 0.0,
        "max_grasp_width": 0.2,
        "approach_direction": [0.0, -1.0, 0.0],
        "approach_threshold_rad": 2.0,
        "workspace_margin": 0.05,
    },
    "high_recall": {
        "max_samples": 600,
        "sample_above_plane": 0,
        "voxelize": 0,
        "radius_outlier_filter_enabled": True,
        "radius_outlier_radius_m": 0.015,
        "radius_outlier_min_neighbors": 4,
        "num_orientations": 12,
        "num_finger_placements": 12,
        "num_selected": 15,
        "voxel_size": 0.003,
        "min_grasp_width": 0.0,
        "max_grasp_width": 0.2,
        "approach_direction": [0.0, -1.0, 0.0],
        "approach_threshold_rad": 2.0,
        "workspace_margin": 0.05,
    },
}


class EstimatePickPoseRos2Service(AsyncAction):
    """Estimate a pick pose for a target object using the ROS2 GPD service.

    Use this after an observation action has stored odom-frame object and
    surrounding point clouds in ``WorldState`` under ``<target>_object_pcd`` and
    ``<target>_surround_pcd``. The action converts those clouds into the source
    camera frame only for the GPD request, then transforms the selected grasp
    back to odom before storing the existing pick-pose keys.

    Params:
        target: str - Object identifier used to read observation data and write
                      estimated grasp results in ``WorldState``.
        clearance: float - Optional local EE z-axis offset applied to the final
                           selected pick pose.

    Advanced GPD tuning such as sample count, PCD denoising, workspace,
    approach direction, thresholds, and speed preset should come from manifest
    defaults or manual debugging kwargs, not from normal LLM plans.
    """

    required_modules = []

    def __init__(
        self,
        target: str = '',
        clearance: float = 0.01,
        preset: str | None = None,
        sample_region: dict[str, Any] | None = None,
        workspace: list[float] | None = None,
        approach_direction: list[float] | None = None,
        approach_threshold_rad: float | None = None,
        sample_radius_m: float | None = None,
        radius_outlier_filter_enabled: bool | None = None,
        radius_outlier_radius_m: float | None = None,
        radius_outlier_min_neighbors: int | None = None,
        max_samples: int | None = None,
        min_score: float | None = None,
        dynamic_workspace: bool = True,
        workspace_margin: float | None = None,
        debug_visualization: bool = True,
        min_grasp_width: float = 0.0,
        max_grasp_width: float = 0.2,
        enable_support_filter: bool = False,
        support_clearance: float = 0.0,
        visualize: bool = True,
        sample_above_plane: bool | None = None,
        voxelize: bool | None = None,
        num_orientations: int | None = None,
        num_finger_placements: int | None = None,
        num_selected: int | None = None,
        voxel_size: float | None = None,
    ) -> None:
        self.target = target
        self.clearance = clearance
        self.preset = preset
        self.sample_region = sample_region
        self.workspace = workspace
        self.approach_direction = approach_direction
        self.approach_threshold_rad = approach_threshold_rad
        self.sample_radius_m = sample_radius_m
        self.radius_outlier_filter_enabled = radius_outlier_filter_enabled
        self.radius_outlier_radius_m = radius_outlier_radius_m
        self.radius_outlier_min_neighbors = radius_outlier_min_neighbors
        self.max_samples = max_samples
        self.min_score = min_score
        self.dynamic_workspace = dynamic_workspace
        self.workspace_margin = workspace_margin
        self.debug_visualization = debug_visualization
        self.min_grasp_width = min_grasp_width
        self.max_grasp_width = max_grasp_width
        self.enable_support_filter = enable_support_filter
        self.support_clearance = support_clearance
        self.visualize = visualize
        self.sample_above_plane = sample_above_plane
        self.voxelize = voxelize
        self.num_orientations = num_orientations
        self.num_finger_placements = num_finger_placements
        self.num_selected = num_selected
        self.voxel_size = voxel_size
        super().__init__()
        self.content = ''
        self._context: Context | None = None
        self.object_pcd_array = None
        self.surround_pcd_array = None
        self.object_colors_array = None
        self.surround_colors_array = None
        self.status = 'idle'
        self._failure_reason: str | None = None
        self._needs_replan = False
        self._request_started_at: float | None = None
        self._request_metadata: dict[str, Any] = {}
        self._gpd_settings: dict[str, Any] = {}
        self._debug_publishers: dict[str, Any] = {}
        self._sample_points_array = np.empty((0, 3), dtype=np.float32)
        self._object_points_camera = np.empty((0, 3), dtype=np.float32)
        self._surround_points_camera = np.empty((0, 3), dtype=np.float32)
        self._object_colors_camera: np.ndarray | None = None
        self._surround_colors_camera: np.ndarray | None = None

    def start(self, context: Context) -> None:
        self._context = context
        self.ws = context.world_state
        if self.ws is None:
            self._set_failure("missing_world_state")
            return

        self.object_pcd_array = self.ws.get(f"{self.target}_object_pcd")
        self.surround_pcd_array = self.ws.get(f"{self.target}_surround_pcd")
        self.object_colors_array = self.ws.get(f"{self.target}_object_pcd_colors")
        self.surround_colors_array = self.ws.get(f"{self.target}_surround_pcd_colors")

        try:
            request = self._build_request(context)
            self._publish_debug_request(context)
        except Exception as exc:  # noqa: BLE001
            self._set_failure(f"gpd_request_build_failed: {exc}", needs_replan=True)
            return

        def _on_response(response: dict[str, Any]) -> None:
            self.response_callback(response)

        def _on_error(exc: Exception) -> None:
            self._set_failure(f"gpd_service_callback_failed: {exc}", needs_replan=True)

        try:
            self._request_started_at = time.monotonic()
            log.info(
                "Calling ROS2 GPD service for target=%s cloud_points=%s object_points=%s "
                "surround_points=%s samples=%s cloud_frame=%s base_frame=%s workspace=%s "
                "approach_direction=%s approach_threshold_rad=%s",
                self.target,
                self._request_metadata.get("combined_points"),
                self._request_metadata.get("object_points"),
                self._request_metadata.get("surround_points"),
                self._request_metadata.get("sample_count"),
                self._request_metadata.get("cloud_frame"),
                self._request_metadata.get("base_frame"),
                self._request_metadata.get("workspace"),
                self._request_metadata.get("approach_direction_base"),
                self._request_metadata.get("approach_threshold_rad"),
            )
            self._context.run_in_context_loop(
                context.modules.call_async,
                DEFAULT_GPD_SERVICE_MODULE,
                payload={"request": request},
                callback=_on_response,
                error_callback=_on_error,
            )
        except Exception as exc:  # noqa: BLE001
            self._set_failure(f"gpd_service_dispatch_failed: {exc}", needs_replan=True)
            return

        self.status = "in_progress"

    def _build_request(self, context: Context):
        from geometry_msgs.msg import Point, Transform, Vector3
        from gpd_ros2_msgs.msg import CloudIndexed, CloudSources, GraspParams
        from gpd_ros2_msgs.srv import DetectConstrainedGrasps
        from std_msgs.msg import Int64

        settings = self._resolve_gpd_settings(context)
        self._gpd_settings = settings
        object_points_world = self._as_points_array(self.object_pcd_array, "object")
        surround_points_world = self._as_points_array(self.surround_pcd_array, "surround", required=False)
        object_colors = self._as_color_array(self.object_colors_array, len(object_points_world), "object", required=False)
        surround_colors = self._as_color_array(self.surround_colors_array, len(surround_points_world), "surround", required=False)

        if context.world_state is None:
            raise ValueError("Missing world state")
        world_frame = context.world_state.get(f"{self.target}_pcd_frame")
        if not world_frame:
            raise ValueError(f"Missing {self.target}_pcd_frame for GPD request")
        camera_frame = context.world_state.get(f"{self.target}_source_camera_frame")
        if not camera_frame:
            raise ValueError(f"Missing {self.target}_source_camera_frame for GPD request")
        base_frame = str(context.hardware.get("base_frame", "base_link"))

        world_to_camera = pose_dict_to_matrix(
            self._lookup_transform(context, str(world_frame), str(camera_frame))
        )
        camera_to_base = self._lookup_transform(context, str(camera_frame), base_frame)
        camera_to_base_mat = pose_dict_to_matrix(camera_to_base)
        camera_to_world_mat = pose_dict_to_matrix(
            self._lookup_transform(context, str(camera_frame), str(world_frame))
        )
        world_to_base_mat = pose_dict_to_matrix(
            self._lookup_transform(context, str(world_frame), base_frame)
        )
        base_to_world_mat = np.linalg.inv(world_to_base_mat)

        object_points = transform_points(object_points_world, world_to_camera)
        surround_points = transform_points(surround_points_world, world_to_camera)
        object_points, object_colors = self._radius_outlier_filter_points(
            object_points,
            object_colors,
            settings,
            label="object",
        )
        if object_points.shape[0] == 0:
            raise ValueError("empty_object_pointcloud_after_outlier_filter")
        surround_points, surround_colors = self._radius_outlier_filter_points(
            surround_points,
            surround_colors,
            settings,
            label="surround",
        )

        sample_source_indices = self._sample_object_indices(object_points, self.sample_region, settings)
        if not sample_source_indices:
            raise ValueError("No GPD sample points remain after bbox/downsampling")
        sample_points = object_points[sample_source_indices, :3]
        sample_colors = object_colors[sample_source_indices, :3] if object_colors is not None else None
        sample_mask = np.zeros(len(object_points), dtype=bool)
        sample_mask[sample_source_indices] = True
        remaining_object_points = object_points[~sample_mask, :3]
        remaining_object_colors = object_colors[~sample_mask, :3] if object_colors is not None else None

        combined_points = np.vstack(
            [
                sample_points,
                remaining_object_points,
                surround_points,
            ]
        )
        surround_color_block = (
            surround_colors
            if len(surround_points) > 0
            else np.empty((0, 3), dtype=np.float32)
            if sample_colors is not None and remaining_object_colors is not None
            else None
        )
        combined_colors = self._combine_cloud_blocks(
            [sample_colors, remaining_object_colors, surround_color_block],
            len(combined_points),
        )
        candidate_indices = list(range(len(sample_points)))
        self._sample_points_array = sample_points
        self._object_points_camera = np.vstack([sample_points, remaining_object_points])
        self._surround_points_camera = surround_points
        self._object_colors_camera = self._combine_cloud_blocks(
            [sample_colors, remaining_object_colors],
            len(self._object_points_camera),
        )
        self._surround_colors_camera = surround_colors

        workspace = (
            [float(v) for v in settings.get("workspace", [])]
            if settings.get("workspace")
            else self._dynamic_workspace(
                transform_points(
                    np.vstack([object_points_world, surround_points_world])
                    if surround_points_world.size
                    else object_points_world,
                    world_to_base_mat,
                ),
                float(settings["workspace_margin"]),
            )
            if bool(self.dynamic_workspace)
            else []
        )

        request = DetectConstrainedGrasps.Request()
        request.cloud_indexed = CloudIndexed()
        request.cloud_indexed.cloud_sources = CloudSources()
        cloud_builder = context.sensor_data_center or SensorDataCenter
        request.cloud_indexed.cloud_sources.cloud = cloud_builder.points_to_pointcloud2(
            combined_points,
            frame_id=str(camera_frame),
            colors=combined_colors,
        )
        request.cloud_indexed.cloud_sources.camera_source = [
            Int64(data=0) for _ in range(len(combined_points))
        ]
        request.cloud_indexed.cloud_sources.view_points = [
            Point(x=0.0, y=0.0, z=0.0)
        ]
        request.cloud_indexed.indices = [Int64(data=int(idx)) for idx in candidate_indices]

        request.grasp_params = GraspParams()
        direction = settings.get("approach_direction") or [1.0, 0.0, 0.0]
        request.grasp_params.approach_direction = Vector3(
            x=float(direction[0]),
            y=float(direction[1]),
            z=float(direction[2]),
        )
        request.grasp_params.camera_position = [
            0.0,
            0.0,
            0.0,
        ]
        request.grasp_params.transform_camera2base = Transform()
        position = camera_to_base.get("position", {})
        orientation = camera_to_base.get("orientation", {})
        request.grasp_params.transform_camera2base.translation.x = float(position.get("x", 0.0))
        request.grasp_params.transform_camera2base.translation.y = float(position.get("y", 0.0))
        request.grasp_params.transform_camera2base.translation.z = float(position.get("z", 0.0))
        request.grasp_params.transform_camera2base.rotation.x = float(orientation.get("x", 0.0))
        request.grasp_params.transform_camera2base.rotation.y = float(orientation.get("y", 0.0))
        request.grasp_params.transform_camera2base.rotation.z = float(orientation.get("z", 0.0))
        request.grasp_params.transform_camera2base.rotation.w = float(orientation.get("w", 1.0))
        request.grasp_params.workspace = workspace
        request.grasp_params.enable_approach_dir_filtering = (
            settings.get("approach_threshold_rad") is not None
        )
        request.grasp_params.approach_dir_threshold = float(settings.get("approach_threshold_rad") or 0.0)
        request.params_policy = DetectConstrainedGrasps.Request.USE_REQUEST_PARAMS
        self._request_metadata = {
            "target": self.target,
            "object_points_original": int(len(object_points_world)),
            "surround_points_original": int(len(surround_points_world)),
            "object_points": int(len(object_points)),
            "surround_points": int(len(surround_points)),
            "combined_points": int(len(combined_points)),
            "sample_count": int(len(candidate_indices)),
            "remaining_object_points": int(len(remaining_object_points)),
            "world_frame": str(world_frame),
            "cloud_frame": str(camera_frame),
            "camera_frame": str(camera_frame),
            "base_frame": base_frame,
            "camera_position": [0.0, 0.0, 0.0],
            "camera_position_world": camera_to_world_mat[:3, 3].astype(float).tolist(),
            "transform_camera2base": camera_to_base_mat.tolist(),
            "transform_camera2world": camera_to_world_mat.tolist(),
            "transform_world2camera": world_to_camera.tolist(),
            "transform_world2base": world_to_base_mat.tolist(),
            "transform_base2world": base_to_world_mat.tolist(),
            "workspace": workspace,
            "sample_region": self.sample_region,
            "max_samples": settings.get("max_samples"),
            "radius_outlier_filter": {
                "enabled": bool(settings.get("radius_outlier_filter_enabled")),
                "radius_m": settings.get("radius_outlier_radius_m"),
                "min_neighbors": settings.get("radius_outlier_min_neighbors"),
            },
            "has_color": combined_colors is not None,
            "preset": settings.get("preset"),
            "effective_values": {
                "max_samples": settings.get("max_samples"),
                "num_orientations": settings.get("num_orientations"),
                "num_finger_placements": settings.get("num_finger_placements"),
                "num_selected": settings.get("num_selected"),
                "voxel_size": settings.get("voxel_size"),
                "min_grasp_width": settings.get("min_grasp_width"),
                "max_grasp_width": settings.get("max_grasp_width"),
                "sample_above_plane": settings.get("sample_above_plane"),
                "voxelize": settings.get("voxelize"),
            },
            "approach_direction_base": direction,
            "approach_direction_world": transform_direction(direction, base_to_world_mat).tolist(),
            "approach_threshold_rad": settings.get("approach_threshold_rad"),
        }
        log.info(
            "GPD preset=%s effective=%s sample_above_plane=%s",
            settings.get("preset"),
            self._request_metadata["effective_values"],
            settings.get("sample_above_plane"),
        )
        self._log_gpd_request_geometry()
        return request

    def _publish_debug_request(self, context: Context) -> None:
        if not bool(self.debug_visualization):
            return
        cloud_frame = str(self._request_metadata.get("cloud_frame") or "")
        self._publish_debug_cloud(context, "object_cloud", self._object_points_camera, cloud_frame, self._object_colors_camera)
        self._publish_debug_cloud(context, "surround_cloud", self._surround_points_camera, cloud_frame, self._surround_colors_camera)
        self._publish_debug_cloud(context, "sample_cloud", self._sample_points_array, cloud_frame, None)
        self._publish_grasp_markers([], np.array([]), -1, None, clear=True)

    def _publish_debug_candidates(
        self,
        tf_matrices: list[np.ndarray],
        scores: np.ndarray,
        best_idx: int,
        filter_result: GraspFilterResult | None = None,
    ) -> None:
        if not bool(self.debug_visualization) or self._context is None:
            return
        node = self._debug_node(self._context)
        if node is None:
            return
        try:
            from visualization_msgs.msg import Marker, MarkerArray

            frame_id = str(self._request_metadata.get("cloud_frame") or self._context.hardware.get("odom", "odom"))
            markers = MarkerArray()
            accepted_indices = {
                record.candidate.index
                for record in (filter_result.accepted if filter_result is not None else [])
            }
            rejected_indices = {
                record.candidate.index
                for record in (filter_result.rejected if filter_result is not None else [])
            }
            for idx, tf in enumerate(tf_matrices):
                marker = self._grasp_marker(Marker, tf, frame_id, idx, scores[idx] if idx < len(scores) else 0.0)
                marker.ns = "raw"
                markers.markers.append(marker)
                if idx in accepted_indices:
                    accepted = self._grasp_marker(Marker, tf, frame_id, 10_000 + idx, scores[idx] if idx < len(scores) else 0.0)
                    accepted.ns = "accepted"
                    accepted.color.r = 0.0
                    accepted.color.g = 1.0
                    accepted.color.b = 0.0
                    markers.markers.append(accepted)
                if idx in rejected_indices:
                    rejected = self._grasp_marker(Marker, tf, frame_id, 20_000 + idx, scores[idx] if idx < len(scores) else 0.0)
                    rejected.ns = "rejected"
                    rejected.color.r = 1.0
                    rejected.color.g = 0.0
                    rejected.color.b = 0.0
                    rejected.color.a = 0.35
                    markers.markers.append(rejected)
                if idx == best_idx:
                    selected = self._grasp_marker(Marker, tf, frame_id, 30_000, scores[idx] if idx < len(scores) else 0.0)
                    selected.ns = "selected"
                    selected.color.r = 0.0
                    selected.color.g = 0.0
                    selected.color.b = 1.0
                    selected.scale.x = 0.12
                    markers.markers.append(selected)
            self._debug_publisher(node, "grasps", MarkerArray).publish(markers)
        except Exception as exc:  # noqa: BLE001
            log.warn("Failed to publish GPD debug markers: %s", exc)

    def _publish_grasp_markers(
        self,
        tf_matrices: list[np.ndarray],
        scores: np.ndarray,
        best_idx: int,
        filter_result: GraspFilterResult | None,
        *,
        clear: bool = False,
    ) -> None:
        if not bool(self.debug_visualization) or self._context is None:
            return
        node = self._debug_node(self._context)
        if node is None:
            return
        try:
            from visualization_msgs.msg import Marker, MarkerArray

            markers = MarkerArray()
            if clear:
                marker = Marker()
                marker.action = Marker.DELETEALL
                markers.markers.append(marker)
                self._debug_publisher(node, "grasps", MarkerArray).publish(markers)
                return
        except Exception as exc:  # noqa: BLE001
            log.warn("Failed to clear GPD debug markers: %s", exc)

    def _publish_debug_cloud(
        self,
        context: Context,
        name: str,
        points: Any,
        frame_id: str,
        colors: Any | None = None,
    ) -> None:
        node = self._debug_node(context)
        if node is None:
            return
        try:
            array = self._as_points_array(points, name, required=False)
            color_array = self._as_color_array(colors, len(array), name, required=False)
            cloud = SensorDataCenter.points_to_pointcloud2(array, frame_id=frame_id, colors=color_array)
            self._debug_publisher(node, name, type(cloud)).publish(cloud)
        except Exception as exc:  # noqa: BLE001
            log.warn("Failed to publish GPD debug cloud %s: %s", name, exc)

    def _debug_node(self, context: Context) -> Any | None:
        sdc = context.sensor_data_center
        node = getattr(sdc, "node", None) if sdc is not None else None
        if node is None:
            return None
        return node

    def _debug_publisher(self, node: Any, name: str, msg_type: Any) -> Any:
        key = f"{name}:{getattr(msg_type, '__name__', str(msg_type))}"
        if key not in self._debug_publishers:
            # topic_target = str(self.target or "target").replace(" ", "_")
            # topic = f"/orchid/gpd_ros2/{topic_target}/{name}"
            topic_map = {
                "object_cloud": "/orchid/gpd/object_cloud",
                "surround_cloud": "/orchid/gpd/surround_cloud",
                "sample_cloud": "/orchid/gpd/sample_cloud",
                "grasps": "/orchid/gpd/grasps",
            }
            topic = topic_map.get(name, f"/orchid/gpd/{name}")
            self._debug_publishers[key] = node.create_publisher(msg_type, topic, 1)
        return self._debug_publishers[key]

    def _grasp_marker(
        self,
        marker_type: Any,
        tf: np.ndarray,
        frame_id: str,
        marker_id: int,
        score: float,
    ) -> Any:
        marker = marker_type()
        marker.header.frame_id = frame_id
        marker.ns = "gpd_raw_grasps"
        marker.id = int(marker_id)
        marker.type = marker_type.ARROW
        marker.action = marker_type.ADD
        marker.pose.position.x = float(tf[0, 3])
        marker.pose.position.y = float(tf[1, 3])
        marker.pose.position.z = float(tf[2, 3])
        qx, qy, qz, qw = rotation_matrix_to_quaternion(tf[:3, :3])
        marker.pose.orientation.x = qx
        marker.pose.orientation.y = qy
        marker.pose.orientation.z = qz
        marker.pose.orientation.w = qw
        marker.scale.x = 0.08
        marker.scale.y = 0.01
        marker.scale.z = 0.01
        marker.color.r = 0.2
        marker.color.g = 0.55
        marker.color.b = 1.0
        marker.color.a = max(0.25, min(1.0, 0.25 + float(score)))
        return marker

    @staticmethod
    def _lookup_transform(context: Context, source_frame: str, target_frame: str) -> dict[str, Any]:
        if context.sensor_data_center is None:
            raise ValueError("Missing sensor data center for GPD frame transform")
        pose = context.sensor_data_center.get_pose_at(source_frame, target_frame)
        if pose is None:
            raise ValueError(f"Missing TF {source_frame} -> {target_frame} for GPD request")
        return pose

    @staticmethod
    def _dynamic_workspace(points: np.ndarray, margin: float) -> list[float]:
        mins = points[:, :3].min(axis=0) - float(margin)
        maxs = points[:, :3].max(axis=0) + float(margin)
        return [
            float(mins[0]),
            float(maxs[0]),
            float(mins[1]),
            float(maxs[1]),
            float(mins[2]),
            float(maxs[2]),
        ]

    def _resolve_gpd_settings(self, context: Context) -> dict[str, Any]:
        hardware_cfg = context.hardware.get("gpd_ros2", {}) if isinstance(context.hardware, dict) else {}
        preset_name = str(self.preset or hardware_cfg.get("preset") or DEFAULT_GPD_PRESET)
        presets = dict(DEFAULT_GPD_PRESETS)
        if isinstance(hardware_cfg.get("presets"), dict):
            for name, values in hardware_cfg["presets"].items():
                if isinstance(values, dict):
                    presets[str(name)] = {**presets.get(str(name), {}), **values}
        if preset_name not in presets:
            raise ValueError(f"Unknown GPD preset '{preset_name}'")

        settings = dict(presets[preset_name])
        settings["preset"] = preset_name
        for key in (
            "workspace",
            "approach_direction",
            "approach_threshold_rad",
            "max_samples",
            "min_score",
            "workspace_margin",
            "min_grasp_width",
            "max_grasp_width",
            "enable_support_filter",
            "support_clearance",
            "sample_above_plane",
            "voxelize",
            "radius_outlier_filter_enabled",
            "radius_outlier_radius_m",
            "radius_outlier_min_neighbors",
            "num_orientations",
            "num_finger_placements",
            "num_selected",
            "voxel_size",
        ):
            if key in hardware_cfg:
                settings[key] = hardware_cfg[key]
            value = getattr(self, key, None)
            if value is not None:
                settings[key] = value
        if "workspace_margin" not in settings or settings["workspace_margin"] is None:
            settings["workspace_margin"] = 0.05
        if "sample_above_plane" not in settings:
            settings["sample_above_plane"] = 0
        return settings

    def _sample_object_indices(
        self,
        object_points: np.ndarray,
        sample_region: dict[str, Any] | None,
        settings: dict[str, Any],
    ) -> list[int]:
        candidate_indices = self._bbox_fraction_indices(object_points, sample_region)
        max_samples = 500 #settings.get("max_samples")
        if max_samples is not None and len(candidate_indices) > int(max_samples):
            rng = random.Random(0)
            candidate_indices = sorted(rng.sample(candidate_indices, int(max_samples)))
        return candidate_indices

    def _radius_outlier_filter_points(
        self,
        points: np.ndarray,
        colors: np.ndarray | None,
        settings: dict[str, Any],
        *,
        label: str,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        points = np.ascontiguousarray(points[:, :3], dtype=np.float32)
        if points.shape[0] == 0:
            return points, colors
        if not bool(settings.get("radius_outlier_filter_enabled", True)):
            log.info("GPD %s radius outlier filter disabled count=%d bbox=%s", label, len(points), self._bbox(points))
            return points, colors
        radius = float(settings.get("radius_outlier_radius_m", 0.015))
        min_neighbors = int(settings.get("radius_outlier_min_neighbors", 4))
        if radius <= 0.0 or min_neighbors <= 0:
            log.info("GPD %s radius outlier filter skipped invalid radius/min_neighbors count=%d", label, len(points))
            return points, colors
        if len(points) <= min_neighbors:
            log.info(
                "GPD %s radius outlier filter skipped tiny cloud count=%d min_neighbors=%d bbox=%s",
                label,
                len(points),
                min_neighbors,
                self._bbox(points),
            )
            return points, colors

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
        _, kept_indices = pcd.remove_radius_outlier(nb_points=min_neighbors, radius=radius)
        kept = np.asarray(sorted(kept_indices), dtype=np.int64)
        if kept.size == 0:
            log.warn(
                "GPD %s radius outlier filter removed all points count=%d radius=%.4f min_neighbors=%d bbox=%s",
                label,
                len(points),
                radius,
                min_neighbors,
                self._bbox(points),
            )
            return np.empty((0, 3), dtype=np.float32), None if colors is None else np.empty((0, 3), dtype=np.float32)

        filtered_points = np.ascontiguousarray(points[kept, :3], dtype=np.float32)
        filtered_colors = None if colors is None else np.ascontiguousarray(colors[kept, :3], dtype=np.float32)
        log.info(
            "GPD %s radius outlier filter count %d -> %d removed=%d radius=%.4f min_neighbors=%d bbox_before=%s bbox_after=%s",
            label,
            len(points),
            len(filtered_points),
            len(points) - len(filtered_points),
            radius,
            min_neighbors,
            self._bbox(points),
            self._bbox(filtered_points),
        )
        return filtered_points, filtered_colors

    @staticmethod
    def _as_points_array(points: Any, label: str, required: bool = True) -> np.ndarray:
        if points is None:
            if required:
                raise ValueError(f"Missing {label} point cloud")
            return np.empty((0, 3), dtype=np.float32)
        array = np.asarray(points, dtype=np.float32)
        if array.size == 0:
            if required:
                raise ValueError(f"Empty {label} point cloud")
            return np.empty((0, 3), dtype=np.float32)
        if array.ndim != 2 or array.shape[1] < 3:
            raise ValueError(f"Invalid {label} point cloud shape {array.shape}")
        return array[:, :3]

    @staticmethod
    def _as_color_array(
        colors: Any,
        expected_len: int,
        label: str,
        required: bool = False,
    ) -> np.ndarray | None:
        if colors is None:
            if required:
                raise ValueError(f"Missing {label} color cloud")
            return None
        array = np.asarray(colors, dtype=np.float32)
        if array.size == 0:
            if required:
                raise ValueError(f"Empty {label} color cloud")
            return None
        if array.ndim != 2 or array.shape[1] < 3:
            raise ValueError(f"Invalid {label} color shape {array.shape}")
        if array.shape[0] != expected_len:
            raise ValueError(
                f"{label} color length {array.shape[0]} does not match point length {expected_len}"
            )
        return np.ascontiguousarray(array[:, :3], dtype=np.float32)

    @staticmethod
    def _combine_optional_colors(
        object_colors: np.ndarray | None,
        surround_colors: np.ndarray | None,
        expected_len: int,
    ) -> np.ndarray | None:
        if object_colors is None:
            return None
        if surround_colors is None:
            colors = object_colors
        else:
            colors = np.vstack([object_colors, surround_colors])
        if colors.shape[0] != expected_len:
            raise ValueError(
                f"Combined color length {colors.shape[0]} does not match point length {expected_len}"
            )
        return np.ascontiguousarray(colors[:, :3], dtype=np.float32)

    @staticmethod
    def _combine_cloud_blocks(
        blocks: list[np.ndarray | None],
        expected_len: int,
    ) -> np.ndarray | None:
        if all(block is None for block in blocks):
            return None
        filled: list[np.ndarray] = []
        for block in blocks:
            if block is None:
                return None
            array = np.asarray(block, dtype=np.float32)
            if array.size == 0:
                filled.append(np.empty((0, 3), dtype=np.float32))
            else:
                filled.append(np.ascontiguousarray(array[:, :3], dtype=np.float32))
        colors = np.vstack(filled) if filled else np.empty((0, 3), dtype=np.float32)
        if colors.shape[0] != expected_len:
            raise ValueError(
                f"Combined color length {colors.shape[0]} does not match point length {expected_len}"
            )
        return np.ascontiguousarray(colors[:, :3], dtype=np.float32)

    @staticmethod
    def _bbox_fraction_indices(
        object_points: np.ndarray,
        sample_region: dict[str, Any] | None,
    ) -> list[int]:
        indices = np.arange(len(object_points))
        if not sample_region:
            return indices.tolist()
        if sample_region.get("mode") != "bbox_fraction":
            raise ValueError("Only sample_region mode 'bbox_fraction' is supported")

        mins = object_points.min(axis=0)
        maxs = object_points.max(axis=0)
        spans = np.maximum(maxs - mins, 1e-9)
        mask = np.ones(len(object_points), dtype=bool)
        axis_map = {"x": 0, "y": 1, "z": 2}
        for axis, dim in axis_map.items():
            bounds = sample_region.get(axis)
            if bounds is None:
                continue
            lo, hi = float(bounds[0]), float(bounds[1])
            coord_fraction = (object_points[:, dim] - mins[dim]) / spans[dim]
            mask &= (coord_fraction >= lo) & (coord_fraction <= hi)
        return indices[mask].tolist()

    def _visualize_grasp(
        self,
        tf_matrices,
        scores,
        widths,
        best_grasp_idx,
        aligned_best_tf,
        viz: bool = False,
    ):
        if not viz:
            return
        obj_pcd = o3d.geometry.PointCloud()
        obj_pcd.points = o3d.utility.Vector3dVector(self.object_pcd_array)
        obj_pcd.paint_uniform_color([0.6, 0.6, 0.6])
        surr_pcd = o3d.geometry.PointCloud()
        surr_pcd.points = o3d.utility.Vector3dVector(self.surround_pcd_array)
        geometries = [obj_pcd, surr_pcd]

        for idx, tf in enumerate(tf_matrices):
            axis1 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.04, origin=[0, 0, 0])
            axis1.transform(tf)
            geometries.append(axis1)
            if idx == best_grasp_idx:
                axis2 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.04, origin=[0, 0, 0])
                axis2.transform(aligned_best_tf)
                geometries.append(axis2)

        o3d.visualization.draw_geometries(geometries)

    def response_callback(self, response: dict[str, Any]):
        try:
            self.content = json.dumps(response, default=str)
            elapsed_s = (
                time.monotonic() - self._request_started_at
                if self._request_started_at is not None
                else None
            )
            if self.ws is not None:
                self.ws.set(f"{self.target}_gpd_request_metadata", self._request_metadata)
                if elapsed_s is not None:
                    self.ws.set(f"{self.target}_gpd_service_elapsed_s", elapsed_s)
            log.info(response)
            if isinstance(response, dict) and response.get("error"):
                self._set_failure(f"gpd_service_error: {response['error']}", needs_replan=True)
                return
            tf_matrices_camera, scores, widths = self._extract_data_from_response(response)
            if tf_matrices_camera is None:
                self._set_failure("no_grasp_candidates", needs_replan=True)
                return
            response_frame = self._response_frame_id(response)
            request_frame = self._request_metadata.get("cloud_frame")
            if response_frame:
                self._request_metadata["response_frame"] = response_frame
            if response_frame and request_frame and response_frame != request_frame:
                self._set_failure(
                    f"gpd_response_frame_mismatch: response={response_frame} request={request_frame}",
                    needs_replan=True,
                )
                return
            camera_to_world = np.asarray(
                self._request_metadata.get("transform_camera2world"),
                dtype=np.float32,
            )
            if camera_to_world.shape != (4, 4):
                self._set_failure("missing_camera_to_world_transform", needs_replan=True)
                return
            tf_matrices_world = [
                transform_pose_matrix(matrix, camera_to_world)
                for matrix in tf_matrices_camera
            ]
            log.info(
                "ROS2 GPD returned %d candidate(s) for target=%s in %.2fs",
                len(tf_matrices_camera),
                self.target,
                elapsed_s if elapsed_s is not None else -1.0,
            )
            self._log_gpd_response_geometry(tf_matrices_camera, tf_matrices_world)

            filter_result = filter_and_rank_grasps(
                tf_matrices_world,
                scores,
                widths,
                self._grasp_filter_config(),
            )
            self._log_grasp_filter_result(filter_result)
            if self.ws is not None:
                self.ws.set(f"{self.target}_gpd_filtered_candidates", filter_result.to_dict())
            if filter_result.best is None:
                self._set_failure("no_grasp_candidates_after_filtering", needs_replan=True)
                return

            best_idx = int(filter_result.best.candidate.index)
            aligned_best_tf, alignment = self._align_gpd_pose_x_upper_half(tf_matrices_world[best_idx])
            log.info(
                "GPD selected grasp target=%s index=%s final_score=%.3f raw_score=%.3f "
                "width=%.3f pos=%s approach=%s alignment=%s",
                self.target,
                best_idx,
                float(filter_result.best.final_score),
                float(filter_result.best.candidate.score),
                float(filter_result.best.candidate.width),
                self._fmt_vec(filter_result.best.candidate.position),
                self._fmt_vec(filter_result.best.candidate.approach),
                alignment,
            )
            self._publish_debug_candidates(tf_matrices_camera, scores, int(best_idx), filter_result)
            self._visualize_grasp(
                tf_matrices_world,
                scores,
                widths,
                best_idx,
                aligned_best_tf,
                viz=bool(self.visualize),
            )
            self.ws.set(f"{self.target}_estimated_pick_pose_mat", self._apply_pickup_offset(aligned_best_tf))
            self.ws.set(f"{self.target}_estimated_pick_pose_frame", self._request_metadata.get("world_frame"))
            self.ws.set(f"{self.target}_estimated_width", widths[best_idx])
            if hasattr(self.ws, "set_movable_object_state"):
                object_pose = None
                object_points = self.ws.get(f"{self.target}_object_pcd")
                try:
                    object_array = np.asarray(object_points, dtype=np.float32)
                    if object_array.ndim == 2 and object_array.shape[0] > 0 and object_array.shape[1] >= 3:
                        object_pose = np.eye(4, dtype=np.float32)
                        object_pose[:3, 3] = object_array[:, :3].mean(axis=0)
                except Exception:  # noqa: BLE001
                    object_pose = None
                if object_pose is not None:
                    self.ws.set_movable_object_state(
                        self.target,
                        pose_matrix=object_pose,
                        frame_id=str(self._request_metadata.get("world_frame") or "odom"),
                        source_action="EstimatePickPoseRos2Service",
                        object_pcd_key=f"{self.target}_object_pcd",
                        surround_pcd_key=f"{self.target}_surround_pcd",
                        source_camera_frame=self._request_metadata.get("camera_frame"),
                        metadata={
                            "estimated_pick_pose_key": f"{self.target}_estimated_pick_pose_mat",
                            "raw_candidate_count": len(tf_matrices_camera),
                            "accepted_candidate_count": len(filter_result.accepted),
                            "selected_score": float(filter_result.best.final_score),
                        },
                    )
            self.ws.set(f"{self.target}_selected_grasp_candidate", filter_result.best.to_dict())
            self.ws.set(f"{self.target}_selected_grasp_alignment", alignment)
            self.ws.set(f"{self.target}_gpd_raw_candidates", {
                "frame": self._request_metadata.get("cloud_frame"),
                "tf_matrices": [matrix.tolist() for matrix in tf_matrices_camera],
                "scores": scores.tolist(),
                "widths": widths.tolist(),
            })
            self.ws.set(f"{self.target}_gpd_odom_candidates", {
                "frame": self._request_metadata.get("world_frame"),
                "tf_matrices": [matrix.tolist() for matrix in tf_matrices_world],
                "aligned_selected_tf": aligned_best_tf.tolist(),
            })
            self.ws.set(f"{self.target}_gpd_service_response", response)
            self.status = "success"
        except Exception as exc:  # noqa: BLE001
            log.exception("Failed to parse ROS2 GPD response: %s", exc)
            self._set_failure(f"gpd_response_parse_failed: {exc}", needs_replan=True)

    def _grasp_filter_config(self) -> GraspFilterConfig:
        camera_position = self._request_metadata.get("camera_position_world")
        settings = self._gpd_settings or {}
        enable_support_filter = bool(settings.get("enable_support_filter", self.enable_support_filter))
        support_z = (
            infer_support_z(self.object_pcd_array, self.surround_pcd_array)
            if enable_support_filter
            else None
        )
        return GraspFilterConfig(
            min_score=float(settings["min_score"]) if settings.get("min_score") is not None else None,
            min_width=float(settings.get("min_grasp_width", self.min_grasp_width)),
            max_width=float(settings.get("max_grasp_width", self.max_grasp_width)),
            enable_support_filter=enable_support_filter,
            support_z=support_z,
            support_clearance=float(settings.get("support_clearance", self.support_clearance)),
            camera_position=camera_position if camera_position else None,
            approach_direction=self._request_metadata.get("approach_direction_world"),
            approach_threshold_rad=self._request_metadata.get("approach_threshold_rad"),
        )

    def _log_grasp_filter_result(self, filter_result: GraspFilterResult, limit: int = 5) -> None:
        config = self._grasp_filter_config()
        rejection_counts: Counter[str] = Counter()
        for record in filter_result.rejected:
            if record.reasons:
                rejection_counts.update(record.reasons)
            else:
                rejection_counts.update(["rejected_without_reason"])

        log.info(
            "GPD grasp filter target=%s frame=%s preset=%s accepted=%d rejected=%d "
            "min_score=%s width=[%.3f, %.3f] support_filter=%s support_z=%s support_clearance=%.3f "
            "approach_dir=%s threshold=%s camera=%s rejection_counts=%s",
            self.target,
            self._request_metadata.get("world_frame"),
            self._request_metadata.get("preset"),
            len(filter_result.accepted),
            len(filter_result.rejected),
            config.min_score,
            float(config.min_width),
            float(config.max_width),
            config.enable_support_filter,
            config.support_z,
            float(config.support_clearance),
            self._fmt_vec(config.approach_direction),
            config.approach_threshold_rad,
            self._fmt_vec(config.camera_position),
            dict(rejection_counts),
        )

        for rank, record in enumerate(filter_result.accepted[:limit], start=1):
            log.info(
                "GPD accepted[%d] target=%s index=%s final_score=%.3f raw_score=%.3f "
                "width=%.3f pos=%s approach=%s reasons=%s",
                rank,
                self.target,
                record.candidate.index,
                float(record.final_score),
                float(record.candidate.score),
                float(record.candidate.width),
                self._fmt_vec(record.candidate.position),
                self._fmt_vec(record.candidate.approach),
                record.reasons,
            )

        for rank, record in enumerate(filter_result.rejected[:limit], start=1):
            log.info(
                "GPD rejected[%d] target=%s index=%s final_score=%.3f raw_score=%.3f "
                "width=%.3f pos=%s approach=%s reasons=%s",
                rank,
                self.target,
                record.candidate.index,
                float(record.final_score),
                float(record.candidate.score),
                float(record.candidate.width),
                self._fmt_vec(record.candidate.position),
                self._fmt_vec(record.candidate.approach),
                record.reasons,
            )

    def _log_gpd_request_geometry(self) -> None:
        sample_count = int(self._request_metadata.get("sample_count") or 0)
        first_indices = list(range(min(sample_count, 5)))
        last_indices = list(range(max(0, sample_count - 5), sample_count)) if sample_count > 5 else first_indices
        log.info(
            "GPD request layout target=%s frame=%s sample_count=%d remaining_object_count=%d "
            "surround_count=%d indices_first=%s indices_last=%s sample_bbox=%s object_bbox=%s surround_bbox=%s",
            self.target,
            self._request_metadata.get("cloud_frame"),
            sample_count,
            int(self._request_metadata.get("remaining_object_points") or 0),
            int(self._request_metadata.get("surround_points_filtered") or self._request_metadata.get("surround_points") or 0),
            first_indices,
            last_indices,
            self._bbox(self._sample_points_array),
            self._bbox(self._object_points_camera),
            self._bbox(self._surround_points_camera),
        )
        log.info(
            "GPD sample examples target=%s first=%s last=%s",
            self.target,
            self._example_points(self._sample_points_array[:5]),
            self._example_points(self._sample_points_array[-5:]),
        )

    def _log_gpd_response_geometry(
        self,
        tf_matrices_camera: list[np.ndarray],
        tf_matrices_world: list[np.ndarray],
        limit: int = 5,
    ) -> None:
        raw_positions = np.asarray([matrix[:3, 3] for matrix in tf_matrices_camera], dtype=np.float32)
        world_positions = np.asarray([matrix[:3, 3] for matrix in tf_matrices_world], dtype=np.float32)
        distances = self._nearest_distances(raw_positions[:limit], self._sample_points_array)
        log.info(
            "GPD response geometry target=%s response_frame=%s raw_grasp_bbox=%s odom_grasp_bbox=%s "
            "top_raw_to_sample_nearest_m=%s",
            self.target,
            self._request_metadata.get("response_frame") or self._request_metadata.get("cloud_frame"),
            self._bbox(raw_positions),
            self._bbox(world_positions),
            [round(float(value), 4) for value in distances],
        )

    @staticmethod
    def _fmt_vec(vector: Any) -> list[float] | None:
        if vector is None:
            return None
        array = np.asarray(vector, dtype=np.float32)
        if array.size < 3:
            return None
        return [round(float(v), 4) for v in array[:3]]

    @staticmethod
    def _bbox(points: Any) -> dict[str, list[float]] | None:
        array = np.asarray(points, dtype=np.float32)
        if array.size == 0:
            return None
        if array.ndim != 2 or array.shape[1] < 3:
            return None
        mins = np.nanmin(array[:, :3], axis=0)
        maxs = np.nanmax(array[:, :3], axis=0)
        return {
            "min": [round(float(value), 4) for value in mins],
            "max": [round(float(value), 4) for value in maxs],
        }

    @classmethod
    def _example_points(cls, points: Any) -> list[list[float]]:
        array = np.asarray(points, dtype=np.float32)
        if array.size == 0 or array.ndim != 2 or array.shape[1] < 3:
            return []
        return [cls._fmt_vec(point) or [] for point in array[:, :3]]

    @staticmethod
    def _nearest_distances(points: np.ndarray, reference: np.ndarray) -> list[float]:
        if points.size == 0 or reference.size == 0:
            return []
        reference = np.asarray(reference, dtype=np.float32)[:, :3]
        distances = []
        for point in np.asarray(points, dtype=np.float32)[:, :3]:
            distances.append(float(np.min(np.linalg.norm(reference - point, axis=1))))
        return distances

    def _align_best_pose(self, scores, tf_matrices):
        best_grasp_idx = np.argmax(scores) if len(scores) > 0 else 0
        aligned, _ = self._align_gpd_pose_x_upper_half(tf_matrices[best_grasp_idx])
        return best_grasp_idx, aligned

    @staticmethod
    def _align_gpd_pose_x_upper_half(tf: np.ndarray, up: np.ndarray | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        """Build an EE frame from raw GPD ``[approach, binormal, axis]`` axes.

        The EE z-axis follows GPD approach. The EE x-axis is selected from the
        GPD axis/binormal family so it lies in the upper half relative to odom
        or base up, without forcing it to be exactly vertical.
        """
        raw = np.asarray(tf, dtype=np.float32)
        up_vec = EstimatePickPoseRos2Service._normalize(up if up is not None else np.array([0.0, 0.0, 1.0]))
        approach = EstimatePickPoseRos2Service._normalize(raw[:3, 0])
        binormal = EstimatePickPoseRos2Service._normalize(raw[:3, 1])
        axis = EstimatePickPoseRos2Service._normalize(raw[:3, 2])
        candidates = [
            ("axis", axis),
            ("-axis", -axis),
            ("binormal", binormal),
            ("-binormal", -binormal),
        ]
        source, seed = max(candidates, key=lambda item: float(np.dot(item[1], up_vec)))
        x_axis = seed - approach * float(np.dot(seed, approach))
        fallback = None
        if np.linalg.norm(x_axis) < 1e-6:
            fallback = "projected_up"
            x_axis = up_vec - approach * float(np.dot(up_vec, approach))
        if np.linalg.norm(x_axis) < 1e-6:
            fallback = "raw_axis"
            x_axis = axis
        x_axis = EstimatePickPoseRos2Service._normalize(x_axis)
        if float(np.dot(x_axis, up_vec)) < 0.0:
            x_axis = -x_axis
            source = f"flipped_{source}"
        y_axis = EstimatePickPoseRos2Service._normalize(np.cross(approach, x_axis))
        x_axis = EstimatePickPoseRos2Service._normalize(np.cross(y_axis, approach))
        aligned = np.eye(4, dtype=np.float32)
        aligned[:3, 0] = x_axis
        aligned[:3, 1] = y_axis
        aligned[:3, 2] = approach
        aligned[:3, 3] = raw[:3, 3]
        metadata = {
            "ee_z_source": "gpd_approach",
            "ee_x_source": source,
            "ee_x_up_dot": float(np.dot(x_axis, up_vec)),
            "fallback": fallback,
            "raw_gpd_convention": "R=[approach,binormal,axis]",
        }
        return aligned, metadata

    @staticmethod
    def _normalize(vector: Any) -> np.ndarray:
        array = np.asarray(vector, dtype=np.float32)[:3]
        norm = float(np.linalg.norm(array))
        if norm < 1e-9:
            return np.zeros(3, dtype=np.float32)
        return np.ascontiguousarray(array / norm, dtype=np.float32)

    def _apply_pickup_offset(self, tf):
        clearance = float(getattr(self, "clearance", 0.01))
        T_offset = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, -clearance],
            [0.0, 0.0, 0.0, 1.0],
        ])
        return tf @ T_offset

    def _extract_data_from_response(self, response):
        if isinstance(response, dict):
            if "tf_matrices" in response:
                tf_matrices = [np.array(matrix) for matrix in response.get("tf_matrices", [])]
                scores = np.array(response.get("scores", []))
                widths = np.array(response.get("widths", []))
                if len(tf_matrices) == 0:
                    return [None, None, None]
                return [tf_matrices, scores, widths]

            grasp_configs = response.get("grasp_configs", response)
            grasps = self._get_field(grasp_configs, "grasps", [])
            tf_matrices, scores, widths = [], [], []
            for grasp in grasps:
                score = float(self._get_data_field(grasp, "score", 0.0))
                tf_matrices.append(self._grasp_to_matrix(grasp))
                scores.append(score)
                widths.append(float(self._get_data_field(grasp, "width", 0.0)))
            if len(tf_matrices) == 0:
                return [None, None, None]
            return [tf_matrices, np.array(scores), np.array(widths)]
        return [None, None, None]

    @classmethod
    def _response_frame_id(cls, response: Any) -> str | None:
        grasp_configs = response.get("grasp_configs", response) if isinstance(response, dict) else response
        header = cls._get_field(grasp_configs, "header")
        frame_id = cls._get_field(header, "frame_id")
        if frame_id:
            return str(frame_id)
        return None

    @classmethod
    def _grasp_to_matrix(cls, grasp) -> np.ndarray:
        approach = cls._vector(grasp, "approach")
        binormal = cls._vector(grasp, "binormal")
        axis = cls._vector(grasp, "axis")
        position = cls._point(grasp, "position")
        tf = np.eye(4)
        tf[:3, 0] = approach
        tf[:3, 1] = binormal
        tf[:3, 2] = axis
        tf[:3, 3] = position
        return tf

    @classmethod
    def _vector(cls, source, name: str) -> np.ndarray:
        value = cls._get_field(source, name)
        return np.array([
            float(cls._get_field(value, "x", 0.0)),
            float(cls._get_field(value, "y", 0.0)),
            float(cls._get_field(value, "z", 0.0)),
        ])

    _point = _vector

    @staticmethod
    def _get_field(source, name: str, default=None):
        if isinstance(source, dict):
            return source.get(name, default)
        return getattr(source, name, default)

    @classmethod
    def _get_data_field(cls, source, name: str, default=None):
        value = cls._get_field(source, name, default)
        return cls._get_field(value, "data", value)

    def check_result(self, context: Context) -> ActionResult:
        if self.status == "failure":
            status = "needs_replan" if self._needs_replan else "failure"
            return ActionResult(
                status=status,
                reason=self._failure_reason or "Failed to call pose estimator",
                suggested_replan_context={
                    "action": "EstimatePickPoseRos2Service",
                    "target": self.target,
                    "reason": self._failure_reason,
                } if self._needs_replan else None,
            )
        if self.status == "cancelled":
            return ActionResult(status="failure", reason="Action Cancelled")
        if self.status == "in_progress":
            return ActionResult(status="running")
        return ActionResult(status="success", payload={"response": self.content})

    def cancel(self, context):
        self.status = "cancelled"
        self._failure_reason = "Action Cancelled"

    def reset(self) -> None:
        super().reset()
        self.content = ""
        self.status = "idle"
        self._failure_reason = None
        self._needs_replan = False
        self._request_started_at = None
        self._request_metadata = {}
        self._gpd_settings = {}
        self._sample_points_array = np.empty((0, 3), dtype=np.float32)
        self._object_points_camera = np.empty((0, 3), dtype=np.float32)
        self._surround_points_camera = np.empty((0, 3), dtype=np.float32)

    def _set_failure(self, reason: str, *, needs_replan: bool = False) -> None:
        self.status = "failure"
        self._failure_reason = reason
        self._needs_replan = needs_replan
        log.warn("EstimatePickPoseRos2Service failed: %s", reason)
