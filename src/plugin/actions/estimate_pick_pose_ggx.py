"""EstimatePickPoseGgx action.

This async action retrieves the target object's stored point-cloud information
and calls an external GraspGenX HTTP inference service (module ``ggx_http``,
see ``orchid/submodules/GGX/GraspGenX/scripts_local/ggx_http_server.py``) to
obtain a pick pose. Mirrors ``EstimatePickPose``'s WorldState-read /
HTTP-call / WorldState-write structure, but talks to GraspGenX instead of GPD.

Unlike GPD, GraspGenX's own convention already matches the real gripper's
frame (approach = +Z, closing = +X, matching how ``hsr_gripper``'s config was
built in the gripper wizard), so no frame-remapping transform is applied to
the returned poses.
"""

from __future__ import annotations

import json
import os
from typing import Any

import numpy as np
import open3d as o3d

from src.core.action import ActionResult, AsyncAction
from src.core.context import Context
# Commented out for now: planners have shifted to MoveIt, which does its own
# collision checking against the planning scene (pick_target_collision /
# pick_surround_collision meshes - see manipulation.py's PickAtPose). Revisit
# this envelope pre-check if/when a GGX-selected pose is observed landing in
# collision under the MoveIt stack.
# from src.core import grasp_collision
from src.core.sensor_data_center import SensorDataCenter
import src.core.logger as log

DEFAULT_GGX_MODULE = "ggx_http"
DEFAULT_DEBUG_DUMP_DIR = "/home/orchid/orchid_server/submodules/GGX/GraspGenX/scratch_handoff"

# Keyed by id(node) - see _publish_grasp_poses for why these must outlive
# any single action instance.
_GGX_POSE_PUBLISHERS: dict[int, tuple] = {}
_GGX_PRINCIPAL_AXIS_PUBLISHERS: dict[int, Any] = {}


class EstimatePickPoseGgx(AsyncAction):
    """Estimate a pick pose for a target object using the GraspGenX HTTP service.

    Use this after an observation action has stored object and surrounding
    point clouds in ``WorldState`` under ``<target>_object_pcd`` and
    ``<target>_surround_pcd``. The action sends the object point cloud to the
    configured ``ggx_http`` module, parses returned grasp candidates, and
    stores the selected pose and gripper width back into ``WorldState`` using
    the existing pick-pose keys (so ``PickAtPose``/``PickTarget`` consume it
    identically to a GPD-produced pose).

    Params:
        target: str - Object identifier used to read observation data and write
                      estimated grasp results in ``WorldState``.
        grasp_threshold: float (default 0.7) - Minimum grasp confidence.
        num_grasps: int (default 200) - Number of grasp samples to generate.
        visualize: bool - If true, show the candidate and selected grasp frames.
        debug_dump: bool - If true, best-effort dump the object/surround point
                      clouds to ``.npy`` files (see ``DEFAULT_DEBUG_DUMP_DIR``)
                      for offline GraspGenX debugging via
                      ``scripts_local/orchid_sim_check.py``. Never blocks or
                      fails the pick - a failed/skipped dump just logs a warning.
        preferred_orientation: str | None - Testing/debug override for
                      ``_select_grasp_pose``'s bucket ordering. Default
                      (``None``) or ``"side"`` keeps the normal preference
                      (working side grasps first, top grasps as fallback).
                      Set to ``"top"`` to force a top-down grasp to be
                      selected instead whenever one exists, for testing the
                      top-grasp path independent of side-grasp
                      availability - does not change what GraspGenX
                      generates, only which candidate gets picked.
    """

    required_modules = [DEFAULT_GGX_MODULE]

    def __init__(
        self,
        target: str = '',
        grasp_threshold: float = 0.7,
        num_grasps: int = 200,
        visualize: bool = False,
        debug_dump: bool = False,
        preferred_orientation: str | None = 'side',
        max_candidate_axis_markers: int = 80,
    ) -> None:
        self.target = target
        self.grasp_threshold = grasp_threshold
        self.num_grasps = num_grasps
        self.visualize = visualize
        self.debug_dump = debug_dump
        self.preferred_orientation = preferred_orientation
        self.max_candidate_axis_markers = max_candidate_axis_markers
        super().__init__()
        self.content = ''
        self._context: Context | None = None
        self.object_pcd_array = None
        self.surround_pcd_array = None
        self.status = 'idle'
        self._failure_reason: str | None = None
        self._needs_replan = False

    def start(self, context: Context) -> None:
        self._context = context
        self.ws = context.world_state
        if self.ws is None:
            self._set_failure("missing_world_state")
            return

        self.object_pcd_array = self.ws.get(f"{self.target}_object_pcd")
        self.surround_pcd_array = self.ws.get(f"{self.target}_surround_pcd")
        if self.object_pcd_array is None:
            self._set_failure("missing_object_pointcloud", needs_replan=True)
            return

        object_points = np.asarray(self.object_pcd_array, dtype=np.float32)[:, :3]
        object_points = self._clean_object_points(object_points)
        # Keep downstream consumers (_estimate_object_width, _select_grasp_pose's
        # distance-to-object ranking, the collision-envelope check) using the
        # same cleaned cloud sent to GraspGenX - otherwise they measure the raw,
        # outlier-laden capture and get inflated extents/distances.
        self.object_pcd_array = object_points

        if self.debug_dump:
            self._dump_debug_pcd()

        payload = {
            "object_points": object_points.tolist(),
            "grasp_threshold": float(self.grasp_threshold),
            "num_grasps": int(self.num_grasps),
        }

        def _on_response(response: dict[str, Any]) -> None:
            self.response_callback(response)

        def _on_error(exc: Exception) -> None:
            self._set_failure(f"ggx_callback_failed: {exc}", needs_replan=True)

        try:
            self._context.run_in_context_loop(
                context.modules.call_async,
                DEFAULT_GGX_MODULE,
                payload=payload,
                callback=_on_response,
                error_callback=_on_error,
            )
        except Exception as exc:  # noqa: BLE001
            self._set_failure(f"ggx_dispatch_failed: {exc}", needs_replan=True)
            return

        self.status = "in_progress"

    def _dump_debug_pcd(self) -> None:
        """Best-effort dump of the target's point clouds to ``.npy`` files
        for offline GraspGenX debugging (see
        ``scripts_local/orchid_sim_check.py``), folded in from the former
        standalone ``DumpPcdForGgx`` action. Unlike that standalone action,
        this is opt-in (``debug_dump=True``) and never blocks or fails the
        pick - capturing debug data failing shouldn't stop the real grasp
        attempt, so any failure here just logs a warning.
        """
        try:
            if self.object_pcd_array is None:
                log.warn(
                    "EstimatePickPoseGgx: debug_dump requested but no object "
                    "point cloud available, skipping"
                )
                return
            os.makedirs(DEFAULT_DEBUG_DUMP_DIR, exist_ok=True)
            object_path = os.path.join(DEFAULT_DEBUG_DUMP_DIR, f"{self.target}_object.npy")
            np.save(object_path, np.asarray(self.object_pcd_array, dtype=np.float32))
            log.info(f"EstimatePickPoseGgx: debug dump wrote {object_path}")
            if self.surround_pcd_array is not None:
                surround_path = os.path.join(DEFAULT_DEBUG_DUMP_DIR, f"{self.target}_surround.npy")
                np.save(surround_path, np.asarray(self.surround_pcd_array, dtype=np.float32))
                log.info(f"EstimatePickPoseGgx: debug dump wrote {surround_path}")
        except Exception as exc:  # noqa: BLE001
            log.warn("EstimatePickPoseGgx: debug dump failed (non-blocking): %s", exc)

    @staticmethod
    def _clean_object_points(points: np.ndarray) -> np.ndarray:
        """Strip depth-sensor outlier points before sending the object cloud
        to GraspGenX.

        A small object against a background at a different depth is exactly
        the scenario where RGBD sensors produce "flying pixel" artifacts -
        mixed/interpolated depth values right at the object's silhouette
        edge - which can make a physically small object's reconstructed
        cloud span far more depth than it actually occupies, biasing
        GraspGenX's estimated grasp pose (confirmed 2026-09-04: a 4cm cube
        produced a ~20cm-tall object cloud, and the resulting grasp pose
        landed ~13cm above the object's real surface). Unlike the
        collision-mesh path (SensorDataCenter.pcd_to_mesh), nothing
        filtered this cloud before it reached GraspGenX. Uses the same
        parameters already validated in pcd_to_mesh, not new ones.
        """
        if points.shape[0] < 50:
            return points
        try:
            cloud = o3d.geometry.PointCloud()
            cloud.points = o3d.utility.Vector3dVector(points.astype(np.float64))
            cloud, _ = cloud.remove_radius_outlier(nb_points=50, radius=0.05)
            cloud, _ = cloud.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
            cleaned = np.asarray(cloud.points, dtype=np.float32)
            return cleaned if cleaned.shape[0] > 0 else points
        except Exception as exc:  # noqa: BLE001
            log.warn("EstimatePickPoseGgx: object cloud outlier removal failed, using raw cloud: %s", exc)
            return points

    def _visualize_grasp(self, tf_matrices, scores, widths, best_grasp_idx, viz: bool = False):
        if not viz:
            return
        obj_pcd = o3d.geometry.PointCloud()
        obj_pcd.points = o3d.utility.Vector3dVector(self.object_pcd_array[:, :3])
        obj_pcd.paint_uniform_color([0.6, 0.6, 0.6])
        geometries = [obj_pcd]
        if self.surround_pcd_array is not None:
            srnd_pcd = o3d.geometry.PointCloud()
            srnd_pcd.points = o3d.utility.Vector3dVector(np.asarray(self.surround_pcd_array)[:, :3])
            srnd_pcd.paint_uniform_color([0.3, 0.3, 0.3])
            geometries.append(srnd_pcd)

        for idx, tf in enumerate(tf_matrices):
            axis = o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=0.06 if idx == best_grasp_idx else 0.03, origin=[0, 0, 0]
            )
            axis.transform(tf)
            geometries.append(axis)

        o3d.visualization.draw_geometries(geometries)

    def response_callback(self, response: dict[str, Any]):
        try:
            self.content = json.dumps(response, default=str)
            # log.info(response) # very big print of GGX response, not useful in normal operation
            if isinstance(response, dict) and response.get("error"):
                self._set_failure(f"ggx_service_error: {response['error']}", needs_replan=True)
                return
            tf_matrices, scores, widths, branch_tags = self._extract_data_from_response(response)
            if tf_matrices is None:
                self._set_failure("no_grasp_candidates", needs_replan=True)
                return

            orientations = self._classify_grasp_orientations(tf_matrices)
            log.info(
                "Grasp orientation classification: %d top, %d side (of %d total)",
                orientations.count("top"), orientations.count("side"), len(orientations),
            )
            branch_tags_for_log = list(branch_tags) if branch_tags is not None else ["diffusion"] * len(orientations)
            for branch_name in sorted(set(branch_tags_for_log)):
                branch_orientations = [
                    orientations[i] for i in range(len(orientations)) if branch_tags_for_log[i] == branch_name
                ]
                log.info(
                    "  branch=%s: %d top, %d side (of %d)",
                    branch_name, branch_orientations.count("top"), branch_orientations.count("side"),
                    len(branch_orientations),
                )
            self.ws.set(f"{self.target}_grasp_orientations", orientations)

            best_idx = self._select_grasp_pose(tf_matrices, scores, branch_tags, orientations)
            log.info(
                "EstimatePickPoseGgx: selected candidate %d classified as '%s'",
                best_idx, orientations[best_idx],
            )
            log.info(
                f"Found {len(tf_matrices)} grasps. Selected grasp score: {scores[best_idx]:.4f}"
            )
            self._visualize_grasp(tf_matrices, scores, widths, best_idx, viz=bool(self.visualize))
            self._publish_grasp_poses(tf_matrices, best_idx)

            estimated_width = self._estimate_object_width(tf_matrices[best_idx])
            log.info(
                "EstimatePickPoseGgx: object-based open width %.4fm "
                "(GraspGenX reported gripper constant %.4fm)",
                estimated_width, widths[best_idx],
            )
            self.ws.set(f"{self.target}_estimated_pick_pose_mat", self._apply_pickup_offset(tf_matrices[best_idx]))
            self.ws.set(f"{self.target}_estimated_width", estimated_width)
            self.ws.set(f"{self.target}_ggx_response", response)
            self.status = "success"
        except Exception as exc:  # noqa: BLE001
            log.exception("Failed to parse GraspGenX response: %s", exc)
            self._set_failure(f"ggx_response_parse_failed: {exc}", needs_replan=True)

    def _estimate_object_width(
        self,
        grasp_tf,
        margin_m: float = 0.04,
        min_width_m: float = 0.02,
        max_width_m: float = 0.085,
    ) -> float:
        """Estimate a sensible gripper-open distance for this grasp.

        GraspGenX's returned ``widths`` is a fixed gripper-geometry constant
        (confirmed identical across every candidate in a response - it comes
        from the gripper's own bbox span, not the object) rather than
        anything object-specific. Using it directly as
        ``ReleaseObjectAction``'s open distance commands the gripper to open
        to its full mechanical span (e.g. ~0.20m after the hsr_gripper
        config fix) regardless of the target object's actual size, sweeping
        a needlessly large collision envelope during the approach/descent -
        confirmed as the direct cause of a GOAL_IN_COLLISION failure on a
        6cm-wide test object. Instead, measure the object's own point cloud
        extent along the grasp pose's local X axis (the closing direction,
        per GraspGenX's approach=+Z/closing=+X convention) and open only as
        wide as that object actually needs, plus a margin.

        margin_m=0.04 (not 0.02): the gripper opens to this width at the
        approach pose, then descends ~10cm straight down past the object to
        reach the grasp pose (PickAtPose's approach->grasp move) while still
        open. A tight margin leaves too little clearance for that vertical
        descent - confirmed 2026-09-05: a 0.02m margin produced only ~1cm of
        clearance per side, and the descending open finger clipped and
        knocked over the object before the grasp ever started.
        """
        if self.object_pcd_array is None:
            return min_width_m
        try:
            points = np.asarray(self.object_pcd_array, dtype=np.float64)[:, :3]
            grasp_inv = np.linalg.inv(np.asarray(grasp_tf))
            local = (grasp_inv[:3, :3] @ points.T).T + grasp_inv[:3, 3]
            object_extent_x = float(local[:, 0].max() - local[:, 0].min())
            width = object_extent_x + margin_m
            clamped = float(np.clip(width, min_width_m, max_width_m))
            if width != clamped:
                log.warn(
                    "EstimatePickPoseGgx: raw object extent along closing axis "
                    "%.4fm (+margin=%.4fm) clamped to %.4fm - object cloud may "
                    "extend beyond the true closing-axis width for this candidate",
                    object_extent_x, width, clamped,
                )
            return clamped
        except Exception as exc:  # noqa: BLE001
            log.warn(
                "EstimatePickPoseGgx: failed to estimate object-based width, "
                "falling back to default: %s", exc,
            )
            return min_width_m

    def _publish_grasp_poses(self, tf_matrices, best_idx: int) -> None:
        """Publish all candidate grasp poses and the selected one as
        geometry_msgs/PoseArray topics, so they can be inspected directly in
        RViz instead of read out of logs - the raw GraspGenX pose (all
        candidates and the post-offset selected pose), in the same 'odom'
        frame everything else in this action already uses.

        Topics: ``ggx_grasp_candidates`` (every candidate, post pickup
        offset) and ``ggx_selected_grasp`` (just the one chosen), both
        geometry_msgs/PoseArray - add a PoseArray display in RViz for each,
        set Fixed Frame to ``odom``. Also ``ggx_selected_grasp_axes``
        (visualization_msgs/MarkerArray, add a MarkerArray display) - a
        fixed-size RGB axis triad for the selected grasp only, so it's
        visible correctly the moment the display is added, with no manual
        RViz axis-size tuning required (unlike PoseArray, whose Axes shape
        size is a per-viewer display setting the message itself doesn't
        control). Best-effort only - never fails the action if ROS2/ the
        sensor node isn't available.
        """
        try:
            sdc = self._context.sensor_data_center if self._context else None
            node = getattr(sdc, "node", None)
            if node is None:
                return

            from geometry_msgs.msg import Point, Pose, PoseArray
            from rclpy.qos import QoSDurabilityPolicy, QoSProfile
            from visualization_msgs.msg import Marker, MarkerArray
            from src.plugin.controllers.moveit import _euler_to_quaternion

            # Cached per-node at module level, not stored on self - a new
            # EstimatePickPoseGgx object is constructed for every pick
            # attempt, and a publisher tied to a short-lived action instance
            # gets destroyed (along with its transient-local cache) the
            # moment that instance is garbage-collected, so a late RViz
            # subscriber would get nothing. Caching by node keeps one
            # long-lived publisher for the whole session instead.
            pubs = _GGX_POSE_PUBLISHERS.get(id(node))
            if pubs is None or len(pubs) < 4:
                # Transient-local so a late subscriber (e.g. `ros2 topic echo`,
                # or RViz added after the fact) immediately gets the last
                # published pose instead of nothing until the next pick.
                latched_qos = QoSProfile(
                    depth=1,
                    durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
                )
                pubs = (
                    node.create_publisher(PoseArray, "ggx_grasp_candidates", latched_qos),
                    node.create_publisher(PoseArray, "ggx_selected_grasp", latched_qos),
                    # PoseArray's axis size is an RViz *display* setting each
                    # viewer has to configure by hand (the message carries no
                    # scale) - this MarkerArray fixes the axis length/thickness
                    # in the message itself, so it's visible correctly the
                    # instant the display is added, no manual tuning needed.
                    node.create_publisher(MarkerArray, "ggx_selected_grasp_axes", latched_qos),
                    node.create_publisher(MarkerArray, "ggx_grasp_candidate_axes", latched_qos),
                )
                _GGX_POSE_PUBLISHERS[id(node)] = pubs
            self._ggx_candidates_pub, self._ggx_selected_pub, self._ggx_selected_axes_pub, self._ggx_candidate_axes_pub = pubs

            def _to_pose(tf) -> Pose:
                x, y, z, roll, pitch, yaw = SensorDataCenter.get_tf_matrix_to_xyz_rpy(np.asarray(tf))
                pose = Pose()
                pose.position.x, pose.position.y, pose.position.z = x, y, z
                pose.orientation = _euler_to_quaternion(roll, pitch, yaw)
                return pose

            stamp = node.get_clock().now().to_msg()

            candidates_msg = PoseArray()
            candidates_msg.header.frame_id = "odom"
            candidates_msg.header.stamp = stamp
            candidates_msg.poses = [_to_pose(self._apply_pickup_offset(tf)) for tf in tf_matrices]
            self._ggx_candidates_pub.publish(candidates_msg)

            selected_msg = PoseArray()
            selected_msg.header.frame_id = "odom"
            selected_msg.header.stamp = stamp
            selected_msg.poses = [_to_pose(self._apply_pickup_offset(tf_matrices[best_idx]))]
            self._ggx_selected_pub.publish(selected_msg)

            axis_colors = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))  # X, Y, Z
            selected_tf = np.asarray(self._apply_pickup_offset(tf_matrices[best_idx]))

            def _axis_markers(tf, *, ns: str, start_id: int, axis_length: float, alpha: float) -> list[Any]:
                origin = tf[:3, 3]
                result = []
                for axis_idx, color in enumerate(axis_colors):
                    direction = tf[:3, axis_idx] * axis_length
                    marker = Marker()
                    marker.header.frame_id = "odom"
                    marker.header.stamp = stamp
                    marker.ns = ns
                    marker.id = start_id + axis_idx
                    marker.type = Marker.ARROW
                    marker.action = Marker.ADD
                    marker.points = [
                        Point(x=float(origin[0]), y=float(origin[1]), z=float(origin[2])),
                        Point(
                            x=float(origin[0] + direction[0]),
                            y=float(origin[1] + direction[1]),
                            z=float(origin[2] + direction[2]),
                        ),
                    ]
                    marker.scale.x = 0.006
                    marker.scale.y = 0.012
                    marker.scale.z = 0.02
                    marker.color.r, marker.color.g, marker.color.b = color
                    marker.color.a = float(alpha)
                    result.append(marker)
                return result

            self._ggx_selected_axes_pub.publish(
                MarkerArray(markers=_axis_markers(selected_tf, ns="ggx_selected_grasp_axes", start_id=0, axis_length=0.06, alpha=1.0))
            )
            max_axes = max(0, int(getattr(self, "max_candidate_axis_markers", 80)))
            step = max(1, int(np.ceil(max(1, len(tf_matrices)) / max(1, max_axes)))) if max_axes else len(tf_matrices) + 1
            candidate_markers = []
            marker_id = 0
            for index, raw_tf in enumerate(tf_matrices):
                if index != best_idx and index % step != 0:
                    continue
                tf = np.asarray(self._apply_pickup_offset(raw_tf))
                selected = index == best_idx
                candidate_markers.extend(
                    _axis_markers(
                        tf,
                        ns="ggx_grasp_candidate_axes",
                        start_id=marker_id,
                        axis_length=0.07 if selected else 0.045,
                        alpha=1.0 if selected else 0.35,
                    )
                )
                marker_id += 3
            self._ggx_candidate_axes_pub.publish(MarkerArray(markers=candidate_markers))
        except Exception as exc:  # noqa: BLE001
            log.warn("EstimatePickPoseGgx: failed to publish grasp poses for visualization: %s", exc)

    # PickAtPose's own default `retreat_z` (see manipulation.py) - the
    # approach/retreat pose PickAtPose actually moves to is the grasp pose
    # offset by exactly this much along world +Z (mode "world_z", also
    # PickAtPose's default). Mirrored here so the pre-flight check validates
    # the same pose PickAtPose will really send the arm to, not just the
    # final grasp pose. If a caller overrides PickTarget/PickAtPose's
    # retreat_z, this pre-check will silently stop matching reality.
    _PICKATPOSE_DEFAULT_RETREAT_Z_M = 0.1

    # Below this angle from straight-down, a candidate is classified as a
    # "top" grasp; at or above it, "side". A single fixed threshold, so
    # every candidate gets exactly one label - no ambiguous middle bucket.
    _TOP_ANGLE_THRESHOLD_DEG = 45.0

    @classmethod
    def _classify_grasp_orientations(cls, tf_matrices) -> list[str]:
        """Classify each GraspGenX candidate's approach direction as a
        "top" (roughly straight-down) or "side" (roughly horizontal)
        grasp, purely from its own tf_matrix - no vague heuristics, no
        object-shape assumptions, just the geometry of the pose itself.

        GraspGenX's tf_matrix convention places the approach direction
        along the pose's local +Z axis (confirmed by
        ``_estimate_object_width``'s "approach=+Z/closing=+X convention"
        and by ``_apply_pickup_offset`` shifting along local Z to fine-tune
        pickup depth). Since these tf_matrices are already expressed in
        the same world-referenced frame as the ``object_points`` sent to
        GraspGenX (``odom`` - see ``ObserveScene``), ``tf_matrix[:3, 2]``
        directly gives the approach direction in that world frame; no
        further transform is needed.

        For each candidate:
          1. Take the local +Z axis (``tf_matrix[:3, 2]``) and normalize it
             - this is the direction the gripper travels to reach the
             grasp pose.
          2. Compute the angle, in degrees, between that vector and
             straight-down in world frame (``(0, 0, -1)``), via
             ``arccos(dot(approach_unit, down))``. This ranges from 0
             (pointing straight down - a pure top grasp) to 180 (pointing
             straight up).
          3. Label "top" if that angle is below ``_TOP_ANGLE_THRESHOLD_DEG``
             (45 degrees), else "side". A 45-degree split point means a
             candidate is labeled by whichever of "straight down" or
             "horizontal" it is closer to - there is no third category, so
             this always classifies 100% of candidates.

        A degenerate (near-zero-length) approach vector - which should
        never happen for a valid rotation matrix - is labeled "top" as
        the conservative default, rather than silently mislabeling it
        "side".

        Returns a list of "top"/"side" strings, one per input matrix, in
        the same order as ``tf_matrices``.
        """
        down = np.array([0.0, 0.0, -1.0], dtype=np.float64)
        labels: list[str] = []
        for tf in tf_matrices:
            tf_arr = np.asarray(tf, dtype=np.float64)
            approach = tf_arr[:3, 2]
            norm = float(np.linalg.norm(approach))
            if norm < 1e-9:
                labels.append("top")
                continue
            approach_unit = approach / norm
            cos_angle = float(np.clip(np.dot(approach_unit, down), -1.0, 1.0))
            angle_deg = float(np.degrees(np.arccos(cos_angle)))
            labels.append("top" if angle_deg < cls._TOP_ANGLE_THRESHOLD_DEG else "side")
        return labels

    @staticmethod
    def _compute_object_principal_axis(object_pcd_array) -> np.ndarray | None:
        """Compute the object's principal (longest) axis via PCA on its
        real captured point cloud, in the same world-referenced frame the
        cloud is already in (``odom``) - not an assumption about which
        world axis is "up", so it holds regardless of how the object is
        actually oriented.

        PCA: center the points on their mean, compute the 3x3 covariance
        matrix of the centered XYZ coordinates, and take the eigenvector
        of the largest eigenvalue - the direction the points are most
        spread out along. For a cylinder standing upright, this reliably
        comes out close to ``(0, 0, 1)`` because the points genuinely
        spread far more along its height than across its diameter.

        Returns None (never raises) if there's no usable cloud - fewer
        than 3 points, or a degenerate zero-length result - so callers
        can skip whatever check depends on it instead of failing the pick.
        """
        if object_pcd_array is None:
            return None
        points = np.asarray(object_pcd_array, dtype=np.float64)[:, :3]
        if points.shape[0] < 3:
            return None
        centered = points - points.mean(axis=0)
        cov = np.cov(centered, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        principal_axis = eigenvectors[:, int(np.argmax(eigenvalues))]
        norm = float(np.linalg.norm(principal_axis))
        if norm < 1e-9:
            return None
        return principal_axis / norm

    def _publish_principal_axis_marker(self, principal_axis: np.ndarray) -> None:
        """Publish the object's computed principal axis as a yellow line
        through its centroid (topic ``ggx_object_principal_axis``,
        visualization_msgs/Marker), sized to the object's real extent
        along that axis - so a closing-axis check result can be judged by
        eye against the actual gripper orientation in RViz, not just from
        logged numbers. Best-effort only - never fails the pick if
        ROS2/the sensor node isn't available.
        """
        try:
            sdc = self._context.sensor_data_center if self._context else None
            node = getattr(sdc, "node", None)
            if node is None or self.object_pcd_array is None:
                return

            from geometry_msgs.msg import Point
            from visualization_msgs.msg import Marker
            from rclpy.qos import QoSDurabilityPolicy, QoSProfile

            pub = _GGX_PRINCIPAL_AXIS_PUBLISHERS.get(id(node))
            if pub is None:
                latched_qos = QoSProfile(depth=1, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
                pub = node.create_publisher(Marker, "ggx_object_principal_axis", latched_qos)
                _GGX_PRINCIPAL_AXIS_PUBLISHERS[id(node)] = pub

            points = np.asarray(self.object_pcd_array, dtype=np.float64)[:, :3]
            centroid = points.mean(axis=0)
            # Project the real points onto the axis to size the line to
            # the object's actual extent, rather than guessing a fixed
            # length that wouldn't match the object in view.
            projections = (points - centroid) @ principal_axis
            p0 = centroid + principal_axis * float(projections.min())
            p1 = centroid + principal_axis * float(projections.max())

            marker = Marker()
            marker.header.frame_id = "odom"
            marker.header.stamp = node.get_clock().now().to_msg()
            marker.ns = "ggx_object_principal_axis"
            marker.id = 0
            marker.type = Marker.ARROW
            marker.action = Marker.ADD
            marker.points = [
                Point(x=float(p0[0]), y=float(p0[1]), z=float(p0[2])),
                Point(x=float(p1[0]), y=float(p1[1]), z=float(p1[2])),
            ]
            marker.scale.x = 0.02
            marker.scale.y = 0.05
            marker.scale.z = 0.08
            marker.color.r, marker.color.g, marker.color.b = (1.0, 1.0, 0.0)
            marker.color.a = 0.9
            pub.publish(marker)
        except Exception as exc:  # noqa: BLE001
            log.warn("EstimatePickPoseGgx: failed to publish principal axis marker: %s", exc)

    # A side grasp's closing axis must land within this many degrees of
    # perpendicular (90 degrees) to the object's principal axis to be
    # considered capable of actually gripping it, rather than closing
    # along its length and contacting nothing.
    _CLOSING_AXIS_PERPENDICULAR_TOLERANCE_DEG = 30.0

    @classmethod
    def _closing_axis_can_grip(cls, tf_matrix, principal_axis: np.ndarray) -> bool:
        """True if this candidate's closing axis (local +X, GraspGenX's
        "closing=+X" convention) is close enough to perpendicular to the
        object's principal axis to actually squeeze across its
        cross-section, rather than along its length.

        Only meaningful for side grasps: a genuine top-down approach
        forces the closing axis to be horizontal by simple geometry of a
        rotation matrix (if the approach/local-Z axis is vertical, the
        other two axes must be horizontal, since all three are mutually
        perpendicular) - so for an upright object this check always
        passes for real top grasps, and only ever matters for side ones,
        where the closing axis genuinely could point either way around
        the object.
        """
        tf_arr = np.asarray(tf_matrix, dtype=np.float64)
        closing = tf_arr[:3, 0]
        norm = float(np.linalg.norm(closing))
        if norm < 1e-9:
            return False
        closing_unit = closing / norm
        cos_angle = float(np.clip(np.dot(closing_unit, principal_axis), -1.0, 1.0))
        angle_deg = float(np.degrees(np.arccos(cos_angle)))
        low = 90.0 - cls._CLOSING_AXIS_PERPENDICULAR_TOLERANCE_DEG
        high = 90.0 + cls._CLOSING_AXIS_PERPENDICULAR_TOLERANCE_DEG
        return low <= angle_deg <= high

    def _get_camera_position_odom(self) -> np.ndarray | None:
        """Best-effort lookup of the camera's position, in the same frame
        ``object_pcd_array`` is already in (``odom``), for the
        back-of-object rejection check below. Reads the camera frame name
        and point-cloud frame ObserveScene already stored in WorldState
        and queries live TF for the camera's current pose in that frame.

        Returns None (never raises) on any failure, so the caller can
        just skip the back-of-object check rather than fail the pick.
        """
        try:
            camera_frame = self.ws.get(f"{self.target}_source_camera_frame")
            pcd_frame = self.ws.get(f"{self.target}_pcd_frame") or "odom"
            if not camera_frame or self._context is None:
                return None
            sdc = getattr(self._context, "sensor_data_center", None)
            if sdc is None:
                return None
            pose = sdc.get_pose_at(camera_frame, pcd_frame)
            if pose is None:
                return None
            position = pose.get("position")
            if not position:
                return None
            return np.array(
                [position["x"], position["y"], position["z"]], dtype=np.float64
            )
        except Exception as exc:  # noqa: BLE001
            log.warn(
                "EstimatePickPoseGgx: camera position lookup failed, "
                "skipping back-of-object check: %s", exc,
            )
            return None

    @staticmethod
    def _approaches_from_camera_side(tf_matrix, camera_position: np.ndarray) -> bool:
        """True if this grasp is reachable from the camera's (robot's)
        side of the object, rather than requiring the arm to reach around
        to the object's far/back side - which is very likely unreachable
        or would collide with the object itself.

        The gripper travels along its local +Z (approach) axis to arrive
        at the grasp pose, so it comes FROM the -Z side. For the grasp to
        be reachable without reaching around the object, that -Z
        ("retreat") direction should point roughly toward the camera, not
        away from it - checked via the sign of the dot product between
        the retreat direction and the vector from the grasp position to
        the camera. Positive means camera-side (reachable); negative
        means the far/back side.
        """
        tf_arr = np.asarray(tf_matrix, dtype=np.float64)
        approach = tf_arr[:3, 2]
        norm = float(np.linalg.norm(approach))
        if norm < 1e-9:
            return True
        approach_unit = approach / norm
        grasp_position = tf_arr[:3, 3]
        to_camera = camera_position - grasp_position
        cam_norm = float(np.linalg.norm(to_camera))
        if cam_norm < 1e-9:
            return True
        to_camera_unit = to_camera / cam_norm
        retreat_direction = -approach_unit
        return float(np.dot(retreat_direction, to_camera_unit)) > 0.0

    def _select_grasp_pose(self, tf_matrices, scores, branch_tags=None, orientations=None) -> int:
        """Pick the highest-scoring GraspGenX candidate whose corrected pose
        actually clears the real geometry ``PickAtPose`` will move through,
        instead of always taking the single top-scoring candidate blind to
        ``<target>_surround_pcd``/``<target>_object_pcd``.

        GraspGenX only searches the isolated OBJECT cloud - unlike GPD,
        which searches the combined object+surround cloud and so implicitly
        avoids the support surface - so GraspGenX's top-scoring candidate
        can be a perfectly valid grasp for the object's shape while still
        driving the gripper into the shelf/table it sits on.

        CURRENTLY DISABLED: the envelope check described below is commented
        out (planners have shifted to MoveIt, which does its own collision
        checking against the planning scene - see manipulation.py's
        PickAtPose). Only the OBB-distance/score ranking below is active;
        candidate selection currently returns ``order[0]`` unconditionally.
        Revisit if a GGX-selected pose is observed landing in collision.

        Uses the real gripper envelope from GraspGenX's own wizard
        calibration (``grasp_collision.gripper_envelope_clears``, the same
        geometry/frame convention GraspGenX's poses already use) rather than
        a single-point distance proxy. Checks both poses PickAtPose actually
        visits, against the collision meshes present in its planning scene
        at that point (see manipulation.py's PickAtPose.start): the
        *approach* pose is visited while both ``pick_surround_collision``
        and ``pick_target_collision`` are still present (the object's own
        mesh is only removed right before the final descent), so it must
        clear both; the *grasp* pose is visited after the target mesh is
        removed, so it only needs to clear the surround mesh. Falls back to
        the top-scoring candidate if none clear both checks - this checks
        the gripper's shape at two fixed poses, not the whole arm's swept
        path to reach them, so it narrows failures rather than eliminating
        them entirely.

        ``branch_tags`` (parallel array, "obb" or "diffusion" per candidate)
        moves "obb" candidates to the front of the walk order. GraspGenX's
        OBB branch is built directly from the object's own captured
        geometry, not the learned diffusion model - measured on real
        captures, its raw candidates land far closer to the true object
        surface (median ~4-5cm, some within 1-2cm) than diffusion
        candidates (median ~10cm+), despite scoring far lower on the same
        discriminator (mean ~0.14 vs routinely >0.9 for diffusion) - the
        discriminator's score is not a trustworthy signal here, for either
        group. So within the OBB group, candidates are instead ranked by
        real distance to the captured object cloud (closest first) - not
        score - since OBB only offers a handful of discrete stand-off
        heights (5 in a confirmed real case) and picking by score alone
        was observed to select the *farthest* of them, not the closest,
        directly contributing to an avoidable collision. Diffusion
        candidates, lacking any better signal, remain ranked by score.

        ``orientations`` (parallel array, "top" or "side" per candidate,
        from ``_classify_grasp_orientations``) then re-ranks all of the
        above into four buckets, most to least preferred: (1) side
        candidates whose closing axis can actually grip the object
        (``_closing_axis_can_grip``) together with top candidates (which
        always can, by construction), reachable from the camera's side of
        the object; (2) side candidates whose closing axis can't actually
        grip (e.g. would close along a cylinder's length); (3) anything
        requiring the arm to reach around to the object's far/back side
        (``_approaches_from_camera_side``), regardless of top/side. Each
        bucket preserves the existing OBB-distance/score ranking within
        it. This is a one-shot selection preference, not a retry/fallback
        - if the resulting top-of-order candidate later fails to plan,
        nothing here retries the next one.
        """
        if branch_tags is None:
            branch_tags = ["diffusion"] * len(scores)
        branch_tags = list(branch_tags)

        def _dist_to_object(idx) -> float:
            if self.object_pcd_array is None:
                return 0.0
            translation = np.asarray(tf_matrices[idx])[:3, 3]
            points = np.asarray(self.object_pcd_array)[:, :3]
            return float(np.linalg.norm(points - translation[None, :], axis=1).min())

        obb_indices = [i for i in range(len(scores)) if branch_tags[i] == "obb"]
        other_indices = [i for i in range(len(scores)) if branch_tags[i] != "obb"]
        obb_ranked = sorted(obb_indices, key=_dist_to_object)
        other_ranked = sorted(other_indices, key=lambda i: scores[i], reverse=True)
        order = obb_ranked + other_ranked
        if not order:
            order = [0]

        # Re-prioritize `order` into four buckets (each preserving the
        # existing OBB-distance/score ranking within it via stable
        # partitioning), from most to least preferred. This is a one-shot
        # selection preference, not a retry/fallback - if the resulting
        # top-of-order candidate later fails to plan, nothing here retries
        # the next one.
        #
        #   1. camera-side-reachable, and either a top grasp (always able
        #      to grip, by construction) or a side grasp whose closing
        #      axis can actually grip the object - side is tried first
        #      within this bucket, per current preference for side grasps.
        #   2. camera-side-reachable side grasps whose closing axis can't
        #      actually grip (e.g. would close along a cylinder's length,
        #      not across it) - as good as no grasp, so ranked below any
        #      working top grasp, but kept as a last resort over nothing.
        #   3. anything requiring the arm to reach around to the object's
        #      far/back side, regardless of top/side - only ever selected
        #      if every other candidate fails this check too.
        if orientations is not None:
            principal_axis = self._compute_object_principal_axis(self.object_pcd_array)
            if principal_axis is not None:
                self._publish_principal_axis_marker(principal_axis)
            camera_position = self._get_camera_position_odom()

            def _closing_ok(idx: int) -> bool:
                if orientations[idx] != "side":
                    return True  # top grasps: geometric no-op, always OK
                if principal_axis is None:
                    return True  # can't check - don't penalize for it
                return self._closing_axis_can_grip(tf_matrices[idx], principal_axis)

            def _camera_side_ok(idx: int) -> bool:
                if camera_position is None:
                    return True  # can't check - don't penalize for it
                return self._approaches_from_camera_side(tf_matrices[idx], camera_position)

            reachable = [i for i in order if _camera_side_ok(i)]
            unreachable = [i for i in order if not _camera_side_ok(i)]

            side_working = [i for i in reachable if orientations[i] == "side" and _closing_ok(i)]
            top_working = [i for i in reachable if orientations[i] == "top"]
            side_broken = [i for i in reachable if orientations[i] == "side" and not _closing_ok(i)]

            log.info(
                "EstimatePickPoseGgx: candidate buckets - side_working=%d "
                "top_working=%d side_broken_closing_axis=%d "
                "back_of_object_unreachable=%d (preferred_orientation=%s)",
                len(side_working), len(top_working), len(side_broken), len(unreachable),
                self.preferred_orientation,
            )

            if self.preferred_orientation == "top":
                order = top_working + side_working + side_broken + unreachable
            else:
                order = side_working + top_working + side_broken + unreachable

        # Commented out for now: planners have shifted to MoveIt, which does
        # its own collision checking against the planning scene
        # (pick_target_collision / pick_surround_collision meshes - see
        # manipulation.py's PickAtPose). Falls back to the top of `order`
        # (OBB candidates by real distance-to-object first, then diffusion
        # candidates by score) with no local envelope pre-check. Revisit if
        # a GGX-selected pose is observed landing in collision under MoveIt.
        #
        # if self.surround_pcd_array is None and self.object_pcd_array is None:
        #     return int(order[0])
        #
        # for idx in order:
        #     grasp_tf = self._apply_pickup_offset(tf_matrices[idx])
        #     # PickAtPose's "world_z" retreat_mode offsets purely along world
        #     # +Z, not the pose's own local Z - orientation is unchanged.
        #     approach_tf = grasp_tf.copy()
        #     approach_tf[:3, 3] = grasp_tf[:3, 3] + np.array(
        #         [0.0, 0.0, self._PICKATPOSE_DEFAULT_RETREAT_Z_M]
        #     )
        #     if (
        #         grasp_collision.gripper_envelope_clears(grasp_tf, self.surround_pcd_array)
        #         and grasp_collision.gripper_envelope_clears(approach_tf, self.surround_pcd_array)
        #         and grasp_collision.gripper_envelope_clears(approach_tf, self.object_pcd_array)
        #     ):
        #         return int(idx)
        #
        # log.warn(
        #     "EstimatePickPoseGgx: no candidate's gripper envelope clears "
        #     "approach/grasp collision checks; falling back to top-scoring "
        #     "candidate %d",
        #     int(order[0]),
        # )
        return int(order[0])

    @classmethod
    def _apply_pickup_offset(cls, tf):
        T_align = np.array([
            [0.0, -1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
        T_offset = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.1],
            [0.0, 0.0, 0.0, 1.0],
        ])
        # return T_align @ np.asarray(tf) @ T_offset
        return np.asarray(tf) @ T_align @ T_offset

    @staticmethod
    def _extract_data_from_response(response):
        tf_matrices = [np.array(matrix) for matrix in response.get("tf_matrices", [])]
        scores = np.array(response.get("scores", []))
        widths = np.array(response.get("widths", []))
        branch_tags = list(response.get("branch_tags", ["diffusion"] * len(tf_matrices)))
        if len(tf_matrices) == 0:
            log.warn("No grasp poses were returned by the GraspGenX service.")
            return [None, None, None, None]
        return [tf_matrices, scores, widths, branch_tags]

    def check_result(self, context: Context) -> ActionResult:
        if self.status == "failure":
            status = "needs_replan" if self._needs_replan else "failure"
            return ActionResult(
                status=status,
                reason=self._failure_reason or "Failed to call pose estimator",
                suggested_replan_context={
                    "action": "EstimatePickPoseGgx",
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

    def _set_failure(self, reason: str, *, needs_replan: bool = False) -> None:
        self.status = "failure"
        self._failure_reason = reason
        self._needs_replan = needs_replan
        log.warn("EstimatePickPoseGgx failed: %s", reason)
