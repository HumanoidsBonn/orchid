"""Navigation plugin actions."""

from __future__ import annotations
from typing import Any
import json, base64
import math

try:
    import cv2
    from cv_bridge import CvBridge
except ImportError:  # pragma: no cover - ROS/perception dependency
    cv2 = None
    CvBridge = None

import py_trees
from src.tree.bt_node_wrapper import ActionScope, BtConditionLeaf
from src.core.failure import FailureInfo, FailureCode, FailureOrigin
from src.core.action import AbstractAction, ActionResult, AsyncAction
from src.core.context import Context
from src.plugin.actions.head import MoveHeadToPanTilt

from src.core.module_proxy import ModuleProxy
# from src.plugin.actions._helpers import 

import src.core.logger as log
import open3d as o3d
import numpy as np
import struct

# DEFAULT_VLM_MODULE = "gemma4_vlm"
DEFAULT_VLM_MODULE = "sam3"
# DEFAULT_SCAN_MODULE = "grounding_dino"
DEFAULT_SCAN_MODULE = "sam3"
DEFAULT_SCENE_CONTEXT_MODULE = "gemma4_vlm"

SYSTEM_PROMPT = """You are an accurate, high-performance spatial-localization vision model built for autonomous robotics. "
        "Your sole task is to detect objects and return their normalized bounding box (0 to 1000) locations in JSON format. The top left of the image should be (0,0)"
        "Never include conversational explanations, preambles, markdown formatting, or markdown code blocks like ```json. "
        "Output absolute valid JSON and nothing else."""


def _extract_response_content(response: dict[str, Any]) -> str:
    content = response.get("content") if isinstance(response, dict) else None
    if isinstance(content, str) and content.strip():
        return content
    raw = response.get("raw") if isinstance(response, dict) else None
    if isinstance(raw, dict):
        choices = raw.get("choices", [])
        if choices:
            message = choices[0].get("message", {})
            nested = message.get("content")
            if isinstance(nested, str) and nested.strip():
                return nested
    return ""


def _json_from_content(content: str) -> Any:
    clean = content.replace("```json", "").replace("```", "").strip()
    return json.loads(clean)

class ObserveScene(AsyncAction):
    """Look at the scene and get semantic information about what objects are there in the scene.
    Update them in the scene graph optionally without location information.
    Get the bounding box coordinates of a target object of interest and get its 3D location and point cloud and update in scene graph and world state. 

    Params:
        str: target - Object of interest, mostly for picking up
        bool: add_other_objects_to_sg - Whether to add other objects of interest to scene graph 
        object_margin: float (default 0.3) – extra margin (% of bounding box) around the object for storing as object point cloud.
        surround_box: float (default 2.0) – size of a box (% inflation of bounding box), surrounding the target object stored as environment point cloud.
    """

    required_modules = [DEFAULT_VLM_MODULE]

    def __init__(
        self,
        target: str = '',
        add_other_objects_to_sg: bool = True,
        object_margin: float = 0.1,
        surround_box: float = 2,
        replan_on_failure: bool = True,
        *,
        min_depth_m: float = 0.05,
        max_depth_m: float = 3.0,
        target_label: str | None = None,
    ) -> None:
        self.target = target
        self.target_label = target_label
        self.add_other_objects_to_sg = add_other_objects_to_sg
        self.object_margin = object_margin
        self.surround_box = surround_box
        self.replan_on_failure = replan_on_failure
        self.min_depth_m = min_depth_m
        self.max_depth_m = max_depth_m
        super().__init__()
        self.processed_frame = None
        self.content = ''
        self.status = 'idle'
        self._context = None
        self.sdc = None
        self._thread = None
        self.rgb_data = None
        self.depth_data = None
        self.pcl_data = None
        self.camera_info_data = None
        self._failure_reason: str | None = None
        self._needs_replan = False
    
    def _detection_query(self, context: Context) -> str:
        """Text the detector is asked to find.

        WorldState keys stay on ``self.target`` so EstimatePickPose/PickTarget
        find them under the same string the planner used, but a scene-graph
        node id such as ``apple_36`` is a poor detector prompt, so resolve it
        to the node's label when the graph knows it.
        """
        scene_graph = getattr(context, "scene_graph", None)
        if scene_graph is None or not self.target:
            return self.target
        try:
            node = scene_graph.get_node(self.target)
        except Exception:  # noqa: BLE001
            node = None
        if node is None:
            return self.target
        label = node.attributes.get("label") or node.attributes.get("name")
        if isinstance(label, str) and label.strip() and label.strip() != self.target:
            log.info("ObserveScene: querying detector for '%s' (label of node '%s')", label, self.target)
            return label.strip()
        return self.target

    def _build_user_prompt(self, target, image_data) -> list:        
        task = {"type": "text", "text": 
            f"Find '{target}' in the image. "
            f"Return JSON: {{\"status\": \"Found\", \"box\": [ymin, xmin, ymax, xmax], \"other objects\": {'<dictionary of name and bounding box>'}}} "
            f"or {{\"status\": \"Not Found\", \"other objects\": {'<dictionary of name and bounding box>'}}} at the root level, dont nest these values." 
            "Also tell me about all the objects in the scene with some semantic context about them."
            "Respond ONLY with valid JSON."
        }

        try:
            if image_data is None or getattr(image_data, "data", None) is None:
                raise ValueError("missing_rgb")
            cv_image = CvBridge().imgmsg_to_cv2(image_data.data, desired_encoding="bgr8")
            self.processed_frame = cv2.resize(cv_image, (640, 480), interpolation=cv2.INTER_AREA)
            success, encoded_img = cv2.imencode(".jpg", self.processed_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        except Exception as ex:
            log.error(f"Failed to compress image {ex}.")
            return []

        image_bytes = encoded_img.tobytes()
        base64_str = base64.b64encode(image_bytes).decode("utf-8")
        image_data_url = f"data:image/jpeg;base64,{base64_str}"
        image = {"type": "image_url", "image_url": {"url": image_data_url}}
        return [task, image]

    def start(self, context: Context) -> None:
        self._generation = getattr(self, "_generation", 0) + 1
        generation = self._generation
        self._context = context
        context.world_state.set('perception_preview_observation',None)
        self.sdc = context.sensor_data_center
        self.hware = context.hardware
        if context.sensor_data_center is None:
            self._set_failure("missing_sensor_data_center")
            return
        if context.modules is None:
            self._set_failure("missing_module_proxy")
            return

        self.rgb_data = context.sensor_data_center.get_latest("head_camera_rgb")
        if self.rgb_data is None:
            self._set_failure("missing_rgb", needs_replan=True)
            return
        self.depth_data = context.sensor_data_center.get_latest("head_camera_depth")
        self.camera_info_data = context.sensor_data_center.get_latest("head_camera_info")
        self.pcl_data = context.sensor_data_center.get_latest("head_pointcloud")

        prompt = self._build_user_prompt(self._detection_query(context), self.rgb_data)
        if not prompt:
            self._set_failure("invalid_rgb_prompt", needs_replan=True)
            return

        def _on_response(response: dict[str, Any]) -> None:
            if generation == self._generation and not self._cancel_requested and not context.execution.cancel_requested and context.execution.control is None:
                self.response_callback(response)

        def _on_error(exc: Exception) -> None:
            if generation != self._generation or self._cancel_requested:
                return
            self._set_failure(f"observe_model_callback_failed: {exc}", needs_replan=True)

        try:
            context.run_in_context_loop(
                context.modules.call_async,
                DEFAULT_VLM_MODULE,
                {
                    "system_prompt": SYSTEM_PROMPT,
                    "prompt": prompt,
                },
                callback=_on_response,
                error_callback=_on_error,
            )
        except Exception as exc:  # noqa: BLE001
            self._set_failure(f"observe_model_dispatch_failed: {exc}", needs_replan=True)
            return

        self.status = "in_progress"

    def response_callback(self, response):
        try:
            if self._cancel_requested: 
                self.status = 'cancelled'
                return
            self.content = self._extract_content(response, DEFAULT_VLM_MODULE)
            bbox, mask = self._extract_bb_and_mask(self.content) # [ymin, xmin, ymax, xmax]
            ws = getattr(self, '_context', None).world_state if getattr(self, '_context', None) else None
            if ws is None:
                self._set_failure("missing_world_state")
                return
            if bbox is None:
                self._set_failure("object_not_found", needs_replan=True)
                return
            if not mask:
                self._set_failure("invalid_model_response: missing mask", needs_replan=True)
                return

            ws.set(f"{self.target}_normalized_bbox", bbox)
            extraction = self._extract_observation_points(bbox, mask)
            if extraction.reason is not None:
                if self.pcl_data is not None:
                    ws.set(f"{self.target}_full_pcd", self.pcl_data)
                self._set_failure(extraction.reason, needs_replan=True)
                return

            source_camera_frame = extraction.frame_id
            odom_frame = str(self.hware.get("odom", "odom"))
            if not source_camera_frame:
                self._set_failure("missing_observation_frame", needs_replan=True)
                return
            try:
                object_pcd_array = self.sdc.transform_data(
                    extraction.object_points,
                    source_camera_frame,
                    odom_frame,
                )
                surround_pcd_array = self.sdc.transform_data(
                    extraction.surround_points,
                    source_camera_frame,
                    odom_frame,
                )
            except Exception as exc:  # noqa: BLE001
                self._set_failure(f"observation_transform_failed: {exc}", needs_replan=True)
                return
            if object_pcd_array.shape[0] == 0:
                self._set_failure("empty_object_pointcloud", needs_replan=True)
                return
            if surround_pcd_array.shape[0] == 0:
                self._set_failure("empty_surround_pointcloud", needs_replan=True)
                return

            ws.set(f"{self.target}_object_pcd", object_pcd_array)
            ws.set(f"{self.target}_object_pcd_colors", extraction.object_colors)
            ws.set(f"{self.target}_surround_pcd", surround_pcd_array)
            ws.set(f"{self.target}_surround_pcd_colors", extraction.surround_colors)
            ws.set(f"{self.target}_pcd_frame", odom_frame)
            ws.set(f"{self.target}_source_camera_frame", source_camera_frame)
            ws.set(f"{self.target}_observation_method", extraction.method)
            object_pose = np.eye(4, dtype=np.float32)
            object_pose[:3, 3] = object_pcd_array[:, :3].mean(axis=0)
            ws.set_movable_object_state(
                self.target,
                pose_matrix=object_pose,
                frame_id=odom_frame,
                source_action="ObserveScene",
                object_pcd_key=f"{self.target}_object_pcd",
                surround_pcd_key=f"{self.target}_surround_pcd",
                source_camera_frame=source_camera_frame,
                metadata={
                    "observation_method": extraction.method,
                    "normalized_bbox": bbox,
                    "object_point_count": int(object_pcd_array.shape[0]),
                    "surround_point_count": int(surround_pcd_array.shape[0]),
                },
            )
            from .perception_preview import observation_preview
            observation_preview(self,bbox,mask,odom_frame)
            import time
            # time.sleep(5)  # Allow time for the preview to be published and retained
            self.status = 'success'
            log.speak("%s detected",(self.target_label or self.target.replace("_"," ")))
            self._viz_pcd(
                object_pcd_array,
                extraction.object_colors,
                surround_pcd_array,
                extraction.surround_colors,
                viz=False,
            )
        except Exception as exc:  # noqa: BLE001
            log.exception("ObserveScene response handling failed: %s", exc)
            self._set_failure(f"observe_response_failed: {exc}", needs_replan=True)

    def _set_failure(self, reason: str, *, needs_replan: bool = False) -> None:
        self.status = "failure"
        self._failure_reason = reason
        self._needs_replan = needs_replan and bool(self.replan_on_failure)
        log.warn("ObserveScene failed: %s", reason)

    def _extract_observation_points(self, bbox, mask):
        if self.depth_data is not None:
            log.error(">>>>>>>>>>>>>> Extracting point cloud from RGB-D data")
            result = self.sdc.extract_color_aligned_points_from_images(
                rgb_msg=self.rgb_data.data,
                depth_msg=self.depth_data.data,
                camera_info_msg=self.camera_info_data.data if self.camera_info_data else None,
                bbox=bbox,
                mask=mask,
                object_margin=float(self.object_margin),
                surround_box=float(self.surround_box),
                min_depth_m=float(getattr(self, 'min_depth_m', 0.05)),
                max_depth_m=float(getattr(self, 'max_depth_m', 3.0)),
            )
            if result.success and result.object_points.shape[0] > 0:
                return result
            log.warn("RGB/depth extraction unavailable: %s", result.reason)

        if self.pcl_data is not None:
            log.error(">>>>>>>>>>>>>> Extracting point cloud from PCD data")
            result = self.sdc.extract_color_aligned_points_from_pointcloud(
                pcl_msg=self.pcl_data.data,
                bbox=bbox,
                mask=mask,
                object_margin=float(self.object_margin),
                surround_box=float(self.surround_box),
            )
            return result

        reason = "missing_depth_and_pointcloud" if self.depth_data is None else "missing_pointcloud"
        from src.core.sensor_data_center import ColorPointExtraction
        return ColorPointExtraction(
            self.sdc.empty_points(),
            self.sdc.empty_colors(),
            self.sdc.empty_points(),
            self.sdc.empty_colors(),
            frame_id=None,
            method="none",
            reason=reason,
        )

    def _extract_bb_and_mask(self, content):
        bbox = None
        mask = None
        try:
            data = _json_from_content(content)
            log.info(type(data))
            if isinstance(data, dict):
                log.info('status', data.get('status'))
                if data.get('status') != 'Not Found': 
                    bbox = data.get('box')
                    mask = data.get('mask')
                else: bbox = None
            elif isinstance(data, list) and len(data) == 4: bbox = data
        except Exception as exc:
            raise ValueError("invalid_model_response: invalid detection response") from exc
        if bbox is None and not (isinstance(data, dict) and data.get('status') == 'Not Found'):
            raise ValueError("invalid_model_response: missing detection box")
        
        log.info('bbox', bbox)
        return bbox, mask

    def check_result(self, context: Context) -> ActionResult:
        if self.status == "failure":
            status = "needs_replan" if self._needs_replan else "failure"
            return ActionResult(
                status=status,
                reason=self._failure_reason or "ObserveScene failed",
                failure=FailureInfo(FailureCode.NOT_FOUND if self._failure_reason == "object_not_found" else FailureCode.UNKNOWN, origin=FailureOrigin.PERCEPTION),
                suggested_replan_context={
                    "action": "ObserveScene",
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

    def _extract_content(self, response: dict[str, Any], module_name: str) -> str:        
        return _extract_response_content(response)
    
    def _rescale_bbox(self, bbox, bb_x, bb_y, img_dim, scale_factor):
        ymin, xmin, ymax, xmax = bbox
        img_h, img_w = img_dim
        scaled_w = int(scale_factor * bb_x)
        scaled_h = int(scale_factor * bb_y)
        scaled_xmin = max(0, xmin - scaled_w)
        scaled_xmax = min(img_w - 1, xmax + scaled_w)
        scaled_ymin = max(0, ymin - scaled_h)
        scaled_ymax = min(img_h - 1, ymax + scaled_h)
        return scaled_ymin, scaled_xmin, scaled_ymax, scaled_xmax 
        
    def _extract_pointcloud_region(self, bbox, pcl_msg, mask=None):
        """Extract object and surrounding point clouds directly from PointCloud2.

        Parameters
        ----------
        bbox : list[float]
            Normalized bounding box ``[ymin, xmin, ymax, xmax]`` with values in the 0‑1000 range.
        pcl_msg : PointCloud2
            The full point‑cloud message from the sensor (assumed organized like the image).

        Returns
        -------
        tuple[list[tuple[float, float, float]], list[tuple[float, float, float]]]
            ``object_points`` – points inside the bounding box.
            ``surround_points`` – points in an expanded region around the box.
        """
        try:
            ymin, xmin, ymax, xmax = bbox
            img_h = pcl_msg.height
            img_w = pcl_msg.width
            log.info('PCD frame size', 'img_h', img_h, 'img_w', img_w)
      
            xmin = max(0, min(xmin, img_w - 1))
            xmax = max(0, min(xmax, img_w - 1))
            ymin = max(0, min(ymin, img_h - 1))
            ymax = max(0, min(ymax, img_h - 1))
            bb_x = xmax - xmin
            bb_y = ymax - ymin

            if mask is None: om = getattr(self, 'object_margin', None) if getattr(self, 'object_margin', None) else 0.1 
            else: om = -0.05
            obj_ymin, obj_xmin, obj_ymax, obj_xmax = self._rescale_bbox(bbox, bb_x, bb_y, (img_h, img_w), om)
            sb = getattr(self, 'surround_box', None) if getattr(self, 'surround_box', None) else 2.0         
            surround_ymin, surround_xmin, surround_ymax, surround_xmax = self._rescale_bbox(bbox, bb_x, bb_y, (img_h, img_w), sb)

            log.info('BBox', 'ymin', ymin, 'xmin', xmin, 'ymax', ymax, 'xmax', xmax,)
            log.info('Obj BBox', 'ymin', obj_ymin, 'xmin', obj_xmin, 'ymax', obj_ymax, 'xmax', obj_xmax)
            log.info('Surround BBox', 'ymin', surround_ymin, 'xmin', surround_xmin, 'ymax', surround_ymax, 'xmax', surround_xmax)

            object_points = []
            object_colors = []
            surround_points = []
            surround_colors = []
          
            mask_img = np.zeros((img_h, img_w), dtype=np.uint8)
            for poly in mask:
                if len(poly) >= 6:
                    pts = np.array(poly, dtype=np.int32).reshape((-1, 1, 2))
                    cv2.fillPoly(mask_img, [pts], 255)

            for v in range(surround_ymin, surround_ymax + 1):
                for u in range(surround_xmin, surround_xmax + 1):
                    data = self.sdc.decode_pcd(pcl_msg, u, v)
                    if data is None: continue
                    x, y, z, color = data
                    if mask_img[v, u] == 255:
                        object_points.append((x, y, z))
                        object_colors.append(color)
                    elif surround_xmax > u > surround_xmin and surround_ymax > v > surround_ymin: 
                        surround_points.append((x, y, z))
                        surround_colors.append(color)

            object_points_arr = np.array(object_points, dtype=np.float32)
            object_colors_arr = np.array(object_colors, dtype=np.float32)
            surround_points_arr = np.array(surround_points, dtype=np.float32)
            surround_colors_arr = np.array(surround_colors, dtype=np.float32)
            log.warn(object_points_arr.shape, surround_points_arr.shape,)
            return (object_points_arr, object_colors_arr, surround_points_arr, surround_colors_arr)      
        
        except Exception as e:
            log.exception(f"Failed to extract pointcloud region: {e}")
            return None, None

    def _viz_detection(self, frame, bbox, mask=None, viz=True):
        from .perception_preview import segmentation_image
        return segmentation_image(frame,bbox,mask,self.target_label or self.target) if viz else None

    def _viz_pcd(self, object_pcd_array, object_colors_array, surround_pcd_array, surround_colors_array, viz=True):
        if not viz: return 
        try:
            log.info('object_pcd_array', object_pcd_array.shape)
            obj_pcd = o3d.geometry.PointCloud()
            obj_pcd.points = o3d.utility.Vector3dVector(object_pcd_array)
            # obj_pcd.colors = o3d.utility.Vector3dVector(object_colors_array)
            # o3d.visualization.draw_geometries([obj_pcd], window_name="Object Visualization")

            log.info('surround_pcd_array', surround_pcd_array.shape)
            srnd_pcd = o3d.geometry.PointCloud()
            srnd_pcd.points = o3d.utility.Vector3dVector(surround_pcd_array)
            srnd_pcd.colors = o3d.utility.Vector3dVector(surround_colors_array)
            o3d.visualization.draw_geometries([srnd_pcd, obj_pcd], window_name="Surround Visualization")
            
        except Exception as ex:
            log.exception(f":Exception in visualization: {ex}.")

    def cancel(self, context):
        self._cancel_requested = True
        self._generation = getattr(self, "_generation", 0) + 1
        return

    def reset(self) -> None:
        super().reset()
        self._generation = getattr(self, "_generation", 0) + 1
        self.content = ""
        self.status = "idle"
        self._failure_reason = None
        self._needs_replan = False
        self.rgb_data = None
        self.depth_data = None
        self.pcl_data = None
        self.camera_info_data = None


class _DetectTargetInView(AsyncAction):
    """Internal detector leaf for one head view in ScanScene."""

    def __init__(self, target: str, detector_module: str, pan: float, tilt: float, index: int) -> None:
        self.target = target
        self.detector_module = detector_module
        self.pan = pan
        self.tilt = tilt
        self.index = index
        super().__init__()
        self._status = 'idle'
        self._failure_reason: str | None = None
        self._content = ''

    def start(self, context: Context) -> None:
        self._generation = getattr(self, "_generation", 0) + 1
        generation = self._generation
        if context.sensor_data_center is None:
            self._fail("missing_sensor_data_center")
            return
        if context.modules is None:
            self._fail("missing_module_proxy")
            return
        rgb = context.sensor_data_center.get_latest("head_camera_rgb")
        if rgb is None:
            self._fail("missing_rgb")
            return
        prompt = self._build_scan_prompt(rgb)
        if not prompt:
            self._fail("invalid_rgb_prompt")
            return

        def _on_response(response: dict[str, Any]) -> None:
            if generation == self._generation and not self._cancel_requested and not context.execution.cancel_requested and context.execution.control is None:
                self._handle_detection_response(context, response)

        def _on_error(exc: Exception) -> None:
            if generation == self._generation and not self._cancel_requested:
                self._fail(f"scan_detector_failed: {exc}")

        try:
            context.run_in_context_loop(
                context.modules.call_async,
                self.detector_module,
                {"prompt": prompt},
                callback=_on_response,
                error_callback=_on_error,
            )
            self._status = "running"
        except Exception as exc:  # noqa: BLE001
            self._fail(f"scan_detector_dispatch_failed: {exc}")

    def _build_scan_prompt(self, rgb_msg: Any) -> list:
        task = {
            "type": "text",
            "text": (
                f"Find '{self.target}' in the image. Return JSON only: "
                "{\"status\":\"Found\",\"box\":[ymin,xmin,ymax,xmax]} or "
                "{\"status\":\"Not Found\"}."
            ),
        }
        try:
            cv_image = CvBridge().imgmsg_to_cv2(rgb_msg.data, desired_encoding="bgr8")
            image = cv2.resize(cv_image, (640, 480), interpolation=cv2.INTER_AREA)
            success, encoded_img = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
            if not success:
                return []
            image_data_url = "data:image/jpeg;base64," + base64.b64encode(encoded_img.tobytes()).decode("utf-8")
            return [task, {"type": "image_url", "image_url": {"url": image_data_url}}]
        except Exception as exc:  # noqa: BLE001
            log.warn("ScanScene failed to encode RGB image: %s", exc)
            return []

    def _handle_detection_response(self, context: Context, response: dict[str, Any]) -> None:
        try:
            self._content = _extract_response_content(response)
            data = _json_from_content(self._content)
            if isinstance(data, dict) and data.get("status") != "Not Found" and data.get("box"):
                ws = context.world_state
                if ws is not None:
                    ws.set(f"{self.target}_scan_bbox", data["box"])
                    ws.set(f"{self.target}_scan_view", {
                        "pan": float(self.pan),
                        "tilt": float(self.tilt),
                        "index": int(self.index),
                    })
                    ws.set(f"{self.target}_visible", True)
                self._status = "success"
                return
            self._fail("object_not_found" if isinstance(data, dict) and data.get("status") == "Not Found" else "invalid_detector_response")
        except Exception as exc:  # noqa: BLE001
            self._fail(f"invalid_detector_response: {exc}")

    def _fail(self, reason: str) -> None:
        self._failure_reason = reason
        self._status = "failure"

    def check_result(self, context: Context) -> ActionResult:
        if self._cancel_requested:
            return ActionResult(status="failure", reason="Action Cancelled")
        if self._status == "success":
            return ActionResult(status="success", payload={"response": self._content})
        if self._status == "failure":
            return ActionResult(status="failure", reason=self._failure_reason, failure=FailureInfo(FailureCode.NOT_FOUND if self._failure_reason == "object_not_found" else FailureCode.UNKNOWN, origin=FailureOrigin.PERCEPTION))
        return ActionResult(status="running")

    def reset(self) -> None:
        super().reset()
        self._generation = getattr(self, "_generation", 0) + 1
        self._status = "idle"
        self._failure_reason = None
        self._content = ""

    def cancel(self, context: Context) -> None:
        self._cancel_requested = True
        self._generation = getattr(self, "_generation", 0) + 1


class ScanScene(AbstractAction):
    """Scan camera views with a lightweight detector to locate a target.

    This action is intended before ``ObserveScene``. It composes per-view head
    moves with detector calls and stores the first visible target bbox/view in
    ``WorldState``. It does not segment or create point clouds.

    Params:
        target: str - object to search for.
        pan_angles: list[float] - pan angles in radians, default center/left/right.
        tilt: float - offset from the invocation's measured head tilt, radians.
        tilt_offsets: list[float] - extra scan offsets, default current/up/down.
        detector_module: str - module name for bbox detection.
        exploration_mode: bool - when true, exhausted scan returns failure so a
                                 selector can continue to another location.
    """

    required_controllers = []

    def __init__(
        self,
        target: str = '',
        pan_angles: list[float] | None = None,
        tilt: float = 0.0,
        detector_module: str = DEFAULT_SCAN_MODULE,
        exploration_mode: bool = False,
        strict_failures: bool = False,
        tilt_offsets: list[float] | None = None,
        min_tilt: float = -1.57,
        max_tilt: float = .52,
        target_label: str | None = None,
    ) -> None:
        self.target = target
        self.target_label = target_label
        self.pan_angles = pan_angles
        self.tilt = tilt
        self.detector_module = detector_module
        self.exploration_mode = exploration_mode
        self.strict_failures = strict_failures
        self.tilt_offsets = list(tilt_offsets) if tilt_offsets is not None else [0.]
        self.min_tilt, self.max_tilt = min_tilt, max_tilt
        if not self.tilt_offsets or not all(math.isfinite(v) for v in [tilt,min_tilt,max_tilt,*self.tilt_offsets,*(pan_angles or [0.,.45,-.45])]) or min_tilt > max_tilt:
            raise ValueError("ScanScene angles must be finite, offsets nonempty and tilt bounds ordered")
        self._fatal = None
        self.pan_angles = pan_angles or [0.0, 0.45, -0.45]
        super().__init__()



    def _exhausted_result(self, reason: str) -> ActionResult:
        if self.exploration_mode:
            return ActionResult(status="failure", reason=reason, failure=FailureInfo(FailureCode.NOT_FOUND if reason == "scan_exhausted" else FailureCode.INVALID_STATE))
        return ActionResult(
            status="needs_user_input",
            reason=(
                f"I could not find '{self.target_label or self.target}' after scanning the current view. "
                "Please clarify where to look next or whether to continue searching."
            ),
            suggested_replan_context={
                "action": "ScanScene",
                "target": self.target,
                "reason": reason,
            },
        )



    def build_node(self, context, name):
        views = []
        self._view_inputs = []
        offsets = self.tilt_offsets if context.head is not None else [0.]
        for index, (offset,pan) in enumerate((offset,pan) for offset in offsets for pan in self.pan_angles):
            actions = []
            head = None
            if context.head is not None:
                head = MoveHeadToPanTilt(pan=float(pan), tilt=0.)
                actions.append(head.build_node(context, f"Look {index}"))
            detector = _DetectTargetInView(target=self.target, detector_module=self.detector_module, pan=float(pan), tilt=0., index=index)
            self._view_inputs.append((head,detector,float(offset),float(pan)))
            actions.append(detector.build_node(context, f"Detect {index}"))
            view = py_trees.composites.Sequence(name=f"Scan view {index}", memory=True, children=actions)
            def complete(ctx, status, view=view, index=index):
                if status == py_trees.common.Status.SUCCESS:
                    return ActionResult(status="success")
                failed = [getattr(node, "result", None) for node in view.iterate() if node.status == py_trees.common.Status.FAILURE]
                result = next((r for r in failed if r is not None), ActionResult(status="failure", failure=FailureInfo()))
                if result.failure and result.failure.origin == FailureOrigin.CONTROLLER and result.failure.code != FailureCode.CANCELLED:
                    self._motion_failure = result
                    log.warn("ScanScene skipping view after motion failure: %s", result.reason)
                elif result.failure and result.failure.code == FailureCode.NOT_FOUND:
                    self._missed_views += 1
                    if index == 0 and ctx.head is not None and any(pan != self.pan_angles[0] for pan in self.pan_angles[1:]):
                        log.speak("%s not found, panning left-right",(self.target_label or self.target.replace("_"," ")))
                else:
                    self._backend_failure = result
                    if self.strict_failures or (result.failure and result.failure.code == FailureCode.CANCELLED):
                        self._fatal = result
                return result
            scope = ActionScope(self, context, f"Scan attempt {index}", view, on_enter=lambda ctx:None, on_complete=complete, on_cancel=lambda ctx:None)
            view = py_trees.decorators.EternalGuard(name="Scan backend healthy", child=scope, condition=lambda:self._fatal is None)
            views.append(view)
        return ActionScope(self, context, name, py_trees.composites.Selector(name="Scan views", memory=True, children=views))

    def on_enter(self, context):
        log.speak("Looking for %s",(self.target_label or self.target.replace("_"," ")))
        self._fatal = None
        self._motion_failure = None
        self._missed_views = 0
        self._backend_failure = None
        if context.world_state is not None:
            context.world_state.set(f"{self.target}_visible", False)
            context.world_state.set(f"{self.target}_scan_bbox", None)
        if context.sensor_data_center is None:
            return self._exhausted_result("missing_sensor_data_center")
        if context.modules is None:
            return self._exhausted_result("missing_module_proxy")
        if context.head is not None:
            from src.plugin.actions.navigation import _head_joint_snapshot, _view_joint_limits
            current = _head_joint_snapshot(context,timeout_s=0.) or {}
            neutral = current.get("head_tilt_joint")
            if neutral is None or not math.isfinite(neutral):
                return ActionResult(status="failure",reason="missing measured head tilt for ScanScene",failure=FailureInfo(FailureCode.INVALID_STATE))
            limits = _view_joint_limits(context)
            low = max(self.min_tilt,limits["head_tilt_joint"][0])
            high = min(self.max_tilt,limits["head_tilt_joint"][1])
            if low > high:
                return ActionResult(status="failure",reason="ScanScene tilt bounds do not overlap hardware limits",failure=FailureInfo(FailureCode.INVALID_REQUEST))
            for head,detector,offset,pan in self._view_inputs:
                head.tilt = detector.tilt = max(low,min(high,neutral+self.tilt+offset))
                head.pan = detector.pan = max(limits["head_pan_joint"][0],min(limits["head_pan_joint"][1],pan))
            log.info("ScanScene measured neutral tilt=%.3f offsets=%s effective_bounds=[%.3f, %.3f]",neutral,self.tilt_offsets,low,high)

    def on_complete(self, context, status):
        if self._fatal is not None:
            return self._fatal
        if status == py_trees.common.Status.SUCCESS:
            return ActionResult(status="success")
        if self.exploration_mode and self._backend_failure is not None:
            return self._backend_failure
        if (self.strict_failures or self.exploration_mode) and self._motion_failure is not None and not self._missed_views:
            return self._motion_failure
        if status != py_trees.common.Status.SUCCESS and self._missed_views and not self._backend_failure:
            log.speak("%s not found",(self.target_label or self.target.replace("_"," ")))
        return ActionResult(status="success") if status == py_trees.common.Status.SUCCESS else self._exhausted_result("scan_exhausted")


class DescribeScene(AsyncAction):
    """Capture a VLM scene summary for the user or replanning context.

    Use this when the planner needs semantic context rather than grasp point
    clouds. The response is stored in ``WorldState`` and can optionally request
    replanning with that context.

    Params:
        prompt: str - optional scene description prompt.
        response_key: str - world-state key for the scene summary.
        target: str - optional object/task focus.
        module: str - VLM module name.
        needs_replan: bool - return ``needs_replan`` after storing context.
    """

    def __init__(
        self,
        prompt: str | None = None,
        response_key: str = 'scene_context',
        target: str = '',
        module: str = DEFAULT_SCENE_CONTEXT_MODULE,
        needs_replan: bool = False,
    ) -> None:
        self.prompt = prompt
        self.response_key = response_key
        self.target = target
        self.module = module
        self.needs_replan = needs_replan
        super().__init__()
        self._status = 'idle'
        self._failure_reason: str | None = None
        self._content = ''

    def start(self, context: Context) -> None:
        if context.sensor_data_center is None:
            self._status = "failure"
            self._failure_reason = "missing_sensor_data_center"
            return
        if context.modules is None:
            self._status = "failure"
            self._failure_reason = "missing_module_proxy"
            return
        rgb = context.sensor_data_center.get_latest("head_camera_rgb")
        if rgb is None:
            self._status = "failure"
            self._failure_reason = "missing_rgb"
            return
        prompt = self._build_context_prompt(rgb)
        if not prompt:
            self._status = "failure"
            self._failure_reason = "invalid_rgb_prompt"
            return

        def _on_response(response: dict[str, Any]) -> None:
            self._content = _extract_response_content(response)
            if context.world_state is not None:
                context.world_state.set(self.response_key, self._content)
                if self.target:
                    context.world_state.set(f"{self.target}_scene_context", self._content)
            self._status = "success"

        def _on_error(exc: Exception) -> None:
            self._status = "failure"
            self._failure_reason = f"describe_scene_failed: {exc}"

        try:
            context.run_in_context_loop(
                context.modules.call_async,
                self.module,
                {"prompt": prompt},
                callback=_on_response,
                error_callback=_on_error,
            )
        except Exception as exc:  # noqa: BLE001
            self._status = "failure"
            self._failure_reason = f"describe_scene_dispatch_failed: {exc}"
            return
        self._status = "in_progress"

    def _build_context_prompt(self, rgb_msg: Any) -> list:
        text = self.prompt or (
            "Describe the visible scene for a mobile manipulation planner. "
            "Mention objects, approximate spatial relations, and likely next useful actions."
        )
        if self.target:
            text += f" Focus on information relevant to '{self.target}'."
        try:
            cv_image = CvBridge().imgmsg_to_cv2(rgb_msg.data, desired_encoding="bgr8")
            image = cv2.resize(cv_image, (640, 480), interpolation=cv2.INTER_AREA)
            success, encoded_img = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
            if not success:
                return []
            image_data_url = "data:image/jpeg;base64," + base64.b64encode(encoded_img.tobytes()).decode("utf-8")
            return [{"type": "text", "text": text}, {"type": "image_url", "image_url": {"url": image_data_url}}]
        except Exception as exc:  # noqa: BLE001
            log.warn("DescribeScene failed to encode RGB image: %s", exc)
            return []

    def check_result(self, context: Context) -> ActionResult:
        if self._cancel_requested:
            return ActionResult(status="failure", reason="Action Cancelled")
        if self._status == "failure":
            return ActionResult(status="failure", reason=self._failure_reason)
        if self._status != "success":
            return ActionResult(status="running")
        if self.needs_replan:
            return ActionResult(
                status="needs_replan",
                reason="Scene context available for replanning",
                suggested_replan_context={
                    "scene_context": self._content,
                    "response_key": self.response_key,
                    "target": self.target,
                },
                payload={"scene_context": self._content},
            )
        return ActionResult(status="success", payload={"scene_context": self._content})

    def reset(self) -> None:
        super().reset()
        self._status = "idle"
        self._failure_reason = None
        self._content = ""

    def cancel(self, context: Context) -> None:
        self._cancel_requested = True
