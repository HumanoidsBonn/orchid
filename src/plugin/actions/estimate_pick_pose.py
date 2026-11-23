"""EstimatePickPose action.

This async action retrieves the target object's bounding box and point cloud that were stored by
`ObserveScene`, then calls an external Flask service (URL taken from the environment variable
`FLASK_ESTIMATE_POSE_URL`) to obtain a pick pose. The call is performed asynchronously via the
HTTP module, and the resulting pose is returned in the action payload.
"""

from __future__ import annotations

import os, io
import json
from typing import Any
import json, base64


from src.core.action import ActionResult, AsyncAction
from src.core.context import Context
from src.core.module_proxy import ModuleProxy
# from src.plugin.actions._helpers import 

import src.core.logger as log
import open3d as o3d
import numpy as np

DEFAULT_GPD_MODULE = "gpd"

class EstimatePickPose(AsyncAction):
    """Estimate a pick pose for a target object.

    Parameters
    ----------
    target: str - The object identifier used when storing data in ``WorldState`` by ``ObserveScene``.
    """

    required_modules = []  # Will be added at runtime if necessary.

    def __init__(self, target: str='') -> None:
        self.target = target
        super().__init__()
        self._thread = None
        self.content = ''
        self._context: Context | None = None
        self.object_pcd_array = None
        self.surround_pcd_array = None
        self.status = 'idle'
        self._failure_reason: str | None = None
        self._needs_replan = False
    
    def start(self, context: Context) -> None:
        """Kick off the async HTTP request.

        The method extracts the previously stored point‑cloud data from ``WorldState`` and
        forwards it (as JSON‑serialisable placeholders) to the Flask service. The response is
        captured in ``vlm_response_callback`` and reported via ``check_result``.
        """
        self._context = context
        self.ws = context.world_state
        if self.ws is None:
            self._set_failure("missing_world_state")
            return

        # Retrieve stored observations.
        bbox = self.ws.get(f"{self.target}_normalized_bbox")
        log.info('Entries in ws', self.ws.list())
        self.object_pcd_array = self.ws.get(f"{self.target}_object_pcd")
        self.surround_pcd_array = self.ws.get(f"{self.target}_surround_pcd")
        if self.object_pcd_array is None:
            self._set_failure("missing_object_pointcloud", needs_replan=True)
            return
        if self.surround_pcd_array is None:
            self.surround_pcd_array = np.empty((0, 3), dtype=np.float32)

        item_stream = self._context.sensor_data_center.to_pcd_stream(self.object_pcd_array)
        env_stream = self._context.sensor_data_center.to_pcd_stream(self.surround_pcd_array)
        files = {
            'item_cloud': ('item.pcd', item_stream, 'application/octet-stream'),
            'env_cloud': ('env.pcd', env_stream, 'application/octet-stream')
        }
        data = {
            'rotation_resolution': str(24),
            'top_n': str(3),
            'n_best': str(1)
        }

        def _on_response(response: dict[str, Any]) -> None:
            self.response_callback(response)

        def _on_error(exc: Exception) -> None:
            self._set_failure(f"gpd_callback_failed: {exc}", needs_replan=True)

        try:
            self._context.run_in_context_loop(
                context.modules.call_async,
                DEFAULT_GPD_MODULE,
                payload=data,
                files=files,
                callback=_on_response,
                error_callback=_on_error,
            )
        except Exception as exc:  # noqa: BLE001
            self._set_failure(f"gpd_dispatch_failed: {exc}", needs_replan=True)
            return
        self.status = "in_progress"

    def _visualize_grasp(self, tf_matrices, scores, widths, best_grasp_idx, aligned_best_tf, viz=True):
        """
        Visualizes a point cloud and a grasp pose TF matrix.
        
        object_pcd: list of points of the object cloud
        tf_matrix: (4, 4) numpy array representing the grasp pose
        """
        if not viz: return
        obj_pcd = o3d.geometry.PointCloud()
        obj_pcd.points = o3d.utility.Vector3dVector(self.object_pcd_array)
        obj_pcd.paint_uniform_color([0.6, 0.6, 0.6])
        surr_pcd = o3d.geometry.PointCloud()
        surr_pcd.points = o3d.utility.Vector3dVector(self.surround_pcd_array)
        geometries = [obj_pcd, surr_pcd]

        for idx, tf in enumerate(tf_matrices):
            # Create a coordinate frame. Size=0.04 means 4cm axes lines.
            axis1 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.04, origin=[0, 0, 0])
            axis1.transform(tf) # original pose from gpd
            geometries.append(axis1)
            log.info("Grasp %s: Score=%.2f, Gripper Width=%.4f", idx, scores[idx], widths[idx])
            if idx == best_grasp_idx:
                log.info("Grasp %s (BEST): Score=%.2f, Gripper Width=%.4f", idx, scores[idx], widths[idx])
                axis2 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.04, origin=[0, 0, 0])
                axis2.transform(aligned_best_tf) # aligned pose from gpd
                geometries.append(axis2)

        o3d.visualization.draw_geometries(geometries)
    
    def response_callback(self, response: dict[str, Any]):
        """Handle the Flask response.

        The Flask service is expected to return a JSON object with a ``pose`` field describing the
        estimated pick pose. The raw response is stored in ``self.content`` and the pose is added
        to the action payload.
        """
        if self._cancel_requested: 
            self.status = 'cancelled'
            return
        try:
            log.info(response)
            if isinstance(response, dict) and response.get("error"):
                self._set_failure(f"gpd_service_error: {response['error']}", needs_replan=True)
                return
            tf_matrices, scores, widths = self._extract_data_from_reponse(response)
            if tf_matrices is None:
                self._set_failure("no_grasp_candidates", needs_replan=True)
                return
            best_idx, aligned_best_tf = self._align_best_pose(scores, tf_matrices)
            self._visualize_grasp(tf_matrices, scores, widths, best_idx, aligned_best_tf, viz=True)
            self.ws.set(f"{self.target}_estimated_pick_pose_mat", self._apply_pickup_offset(aligned_best_tf))
            self.ws.set(f"{self.target}_estimated_width", widths[best_idx])
            log.info(self.ws.list())
            self.status = 'success'
        except Exception as exc:  # noqa: BLE001
            log.exception("Failed to process GPD response: %s", exc)
            self._set_failure(f"gpd_response_parse_failed: {exc}", needs_replan=True)
    
    def _align_best_pose(self, scores, tf_matrices):
        best_grasp_idx = np.argmax(scores) if len(scores) > 0 else 0
        log.info(f"Found {len(tf_matrices)} grasps. Best grasp score: {scores[best_grasp_idx]:.4f}")

        T_align = np.array([
            [ 0.0, 0.0, 1.0, 0.0],  # Gripper X comes from GPD Z
            [ 0.0, 1.0, 0.0, 0.0],  # Gripper Y comes from -GPD Y (Flipped!)
            [-1.0, 0.0, 0.0, 0.0],  # Gripper Z comes from GPD X
            [ 0.0, 0.0, 0.0, 1.0]
        ])
        return best_grasp_idx, tf_matrices[best_grasp_idx] @ T_align 
    
    def _apply_pickup_offset(self, tf):
        T_offset = np.array([[1.0, 0.0, 0.0,   0.0],  
                             [0.0, 1.0, 0.0,   0.0],  
                             [0.0, 0.0, 1.0, -0.01],
                             [0.0, 0.0, 0.0,   1.0]
                    ])
        return tf @ T_offset

    def _extract_data_from_reponse(self, response):
        tf_matrices = [np.array(matrix) for matrix in response.get('tf_matrices', [])]
        scores = np.array(response.get('scores', []))
        widths = np.array(response.get('widths', []))
        if len(tf_matrices) == 0:
            log.warn("No grasp poses were returned by the GPD server.")
            return [None, None, None]
        return [tf_matrices, scores, widths]

    def check_result(self, context: Context) -> ActionResult:
        if self.status == "failure":
            status = "needs_replan" if self._needs_replan else "failure"
            return ActionResult(
                status=status,
                reason=self._failure_reason or "Failed to call pose estimator",
                suggested_replan_context={
                    "action": "EstimatePickPose",
                    "target": self.target,
                    "reason": self._failure_reason,
                } if self._needs_replan else None,
            )
        elif self.status == "cancelled":
            return ActionResult(status="failure", reason="Action Cancelled")
        elif self.status == "in_progress":
            return ActionResult(status="running")
        else:
            return ActionResult(status="success", payload={"response": self.content})
    
    def cancel(self, context):
        self._cancel_requested = True
        return

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
        log.warn("EstimatePickPose failed: %s", reason)
