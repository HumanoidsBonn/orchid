"""Explore plugin actions: open-ended scene search for underspecified commands.

Unlike ScanScene (which checks whether one specific, named target is visible),
InspectScene asks the detector an open question about a described category of
object (e.g. "something to eat") and reports whatever it finds matching that
description.

Because an open question invites the VLM to guess, every positive answer is
verified before it is trusted: ObserveScene (SAM3-backed) is run on the label
the VLM returned, in the same camera frame. A find that SAM3 cannot segment is
treated as a hallucination; the VLM is re-asked once with that feedback, and if
it still cannot produce a verifiable object the view is abandoned and the next
head pose is tried. A verified find is already observed (point clouds written
to WorldState under the found label), so the replan can go straight to
EstimatePickPose/PickTarget.
"""

from __future__ import annotations

import base64
import json
import os
import re
import time
from typing import Any

try:
    import cv2
    from cv_bridge import CvBridge
except ImportError:  # pragma: no cover - ROS/perception dependency
    cv2 = None
    CvBridge = None

import py_trees
from src.tree.bt_node_wrapper import ActionScope, BtConditionLeaf
from src.core.failure import FailureInfo, FailureCode
from src.core.action import AbstractAction, ActionResult, AsyncAction
from src.core.context import Context
from src.plugin.actions.head import MoveHeadToPanTilt
from src.plugin.actions.observe import ObserveScene

import src.core.logger as log

DEFAULT_EXPLORE_MODULE = "deepseek_vlm"

# Directory for per-view debug dumps (JPEG sent to the VLM + raw reply).
# Unset means no dumps are written.
EXPLORE_DEBUG_DIR_ENV = "ORCHID_EXPLORE_DEBUG_DIR"

# Frame size the VLM sees; ObserveScene resizes to the same, so boxes from
# both detectors are directly comparable.
_FRAME_W, _FRAME_H = 640, 480



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


def _as_pixel_box(box: Any) -> list[float] | None:
    """Coerce a [ymin, xmin, ymax, xmax] box to pixel units on the 640x480 frame."""
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return None
    try:
        vals = [float(v) for v in box]
    except (TypeError, ValueError):
        return None
    if all(0.0 <= v <= 1.0 for v in vals):
        vals = [vals[0] * _FRAME_H, vals[1] * _FRAME_W, vals[2] * _FRAME_H, vals[3] * _FRAME_W]
    return vals


def _box_iou(a: list[float], b: list[float]) -> float:
    ay0, ax0, ay1, ax1 = a
    by0, bx0, by1, bx1 = b
    iy0, ix0 = max(ay0, by0), max(ax0, bx0)
    iy1, ix1 = min(ay1, by1), min(ax1, bx1)
    inter = max(0.0, iy1 - iy0) * max(0.0, ix1 - ix0)
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ay1 - ay0) * max(0.0, ax1 - ax0)
    area_b = max(0.0, by1 - by0) * max(0.0, bx1 - bx0)
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(text).lower()).strip("_") or "target"


def _debug_dir() -> str | None:
    debug_dir = os.environ.get(EXPLORE_DEBUG_DIR_ENV)
    if not debug_dir:
        return None
    try:
        os.makedirs(debug_dir, exist_ok=True)
    except OSError as exc:
        log.warn("InspectScene debug dir unavailable (%s): %s", debug_dir, exc)
        return None
    return debug_dir


def _run_log(entry: str) -> None:
    """Append one timestamped entry to <debug dir>/explore_log.txt.

    A single readable file per debug directory, so a whole run's VLM replies
    and verification outcomes can be reviewed after the fact without scrolling
    through the terminal. No-op when the debug dir is not configured.
    """
    debug_dir = _debug_dir()
    if debug_dir is None:
        return
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    try:
        with open(os.path.join(debug_dir, "explore_log.txt"), "a") as f:
            f.write(f"[{stamp}] {entry.rstrip()}\n")
    except OSError as exc:
        log.warn("InspectScene failed to write run log: %s", exc)


class _DetectDescribedTargetInView(AsyncAction):
    """Internal VLM leaf for one head view in InspectScene.

    Asks the VLM to first enumerate what it can actually see, then decide
    whether any of those objects matches the description. When
    ``rejected_labels`` is given, the prompt also tells the model that those
    labels were reported before and could not be verified, so it should look
    again rather than repeat them.
    """

    def __init__(
        self,
        target: str,
        detector_module: str,
        pan: float,
        tilt: float,
        index: int,
        attempt: int = 0,
        min_confidence: float = 0.0,
        rejected_labels: list[str] | None = None,
        rejected_box: list[float] | None = None,
    ) -> None:
        self.target = target
        self.detector_module = detector_module
        self.pan = pan
        self.tilt = tilt
        self.index = index
        self.attempt = attempt
        self.min_confidence = min_confidence
        self.rejected_labels = rejected_labels
        self.rejected_box = rejected_box
        self.rejected_labels = list(rejected_labels or [])
        super().__init__()
        self._status = 'idle'
        self._failure_reason: str | None = None
        self._content = ''
        self.found_label: str | None = None
        self.found_box: list[float] | None = None
        self.found_confidence: float | None = None
        self._frame_path: str | None = None

    def _log_entry(self, outcome: str, *, visible: Any = None, raw: str | None = None) -> None:
        lines = [
            f"VLM target='{self.target}' view={self.index} pan={float(self.pan):.2f} "
            f"tilt={float(self.tilt):.2f} attempt={self.attempt}"
        ]
        if self.rejected_labels:
            lines.append(f"  rejected_so_far: {self.rejected_labels}")
        if visible is not None:
            lines.append(f"  visible_objects: {json.dumps(visible)}")
        lines.append(f"  outcome: {outcome}")
        if self._frame_path:
            lines.append(f"  frame: {self._frame_path}")
        if raw is not None:
            lines.append(f"  raw: {raw.strip()}")
        _run_log("\n".join(lines))

    def start(self, context: Context) -> None:
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
        prompt = self._build_explore_prompt(rgb)
        if not prompt:
            self._fail("invalid_rgb_prompt")
            return

        def _on_response(response: dict[str, Any]) -> None:
            self._handle_detection_response(context, response)

        def _on_error(exc: Exception) -> None:
            self._fail(f"explore_detector_failed: {exc}")

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
            self._fail(f"explore_detector_dispatch_failed: {exc}")

    def _prompt_text(self) -> str:
        text = (
            f"This is a {_FRAME_W}x{_FRAME_H} image from a robot's head camera.\n"
            "Step 1: list every distinct physical object you can clearly see, each with a "
            "tight bounding box [ymin, xmin, ymax, xmax] in pixels.\n"
            f"Step 2: decide whether any object from step 1 matches '{self.target}'. "
            "Only choose an object you listed in step 1.\n"
            "If the surfaces are empty or nothing clearly matches, answer Not Found. "
            "Do not guess and do not invent objects that are not visible.\n"
            "Return JSON only, in one of these two shapes:\n"
            '{"visible_objects":[{"label":"<name>","box":[ymin,xmin,ymax,xmax]}],'
            '"status":"Found","label":"<label copied from visible_objects>",'
            '"box":[ymin,xmin,ymax,xmax],"confidence":<0.0-1.0>}\n'
            '{"visible_objects":[...],"status":"Not Found","confidence":<0.0-1.0>}'
        )
        if self.rejected_labels:
            labels = ", ".join(f"'{lbl}'" for lbl in self.rejected_labels)
            where = f" at box {self.rejected_box}" if self.rejected_box else ""
            text += (
                f"\nIMPORTANT: In this same image you previously reported {labels}{where}, "
                "but a segmentation model found no such object there, so that report was "
                "wrong. Look again carefully, list only what is actually visible, and do "
                "not report those labels again unless you can clearly see them."
            )
        return text

    def _build_explore_prompt(self, rgb_msg: Any) -> list:
        task = {"type": "text", "text": self._prompt_text()}
        try:
            cv_image = CvBridge().imgmsg_to_cv2(rgb_msg.data, desired_encoding="bgr8")
            image = cv2.resize(cv_image, (_FRAME_W, _FRAME_H), interpolation=cv2.INTER_AREA)
            success, encoded_img = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
            if not success:
                return []
            self._debug_dump_image(encoded_img.tobytes())
            image_data_url = "data:image/jpeg;base64," + base64.b64encode(encoded_img.tobytes()).decode("utf-8")
            return [task, {"type": "image_url", "image_url": {"url": image_data_url}}]
        except Exception as exc:  # noqa: BLE001
            log.warn("InspectScene failed to encode RGB image: %s", exc)
            return []

    # -- debug dumps --------------------------------------------------------

    def _debug_path(self, suffix: str) -> str | None:
        debug_dir = _debug_dir()
        if debug_dir is None:
            return None
        stamp = time.strftime("%Y%m%d_%H%M%S")
        name = f"explore_{stamp}_{_slug(self.target)}_view{self.index}_attempt{self.attempt}{suffix}"
        return os.path.join(debug_dir, name)

    def _debug_dump_image(self, jpeg_bytes: bytes) -> None:
        path = self._debug_path(".jpg")
        if path is None:
            return
        try:
            with open(path, "wb") as f:
                f.write(jpeg_bytes)
            self._frame_path = path
            log.info("InspectScene view %s attempt %s: frame saved to %s", self.index, self.attempt, path)
        except OSError as exc:
            log.warn("InspectScene failed to write debug frame: %s", exc)

    def _debug_dump_reply(self, content: str) -> None:
        path = self._debug_path(".txt")
        if path is None:
            return
        try:
            with open(path, "w") as f:
                f.write(self._prompt_text())
                f.write("\n\n===== VLM reply =====\n")
                f.write(content)
        except OSError as exc:
            log.warn("InspectScene failed to write debug reply: %s", exc)

    # -- response handling --------------------------------------------------

    def _handle_detection_response(self, context: Context, response: dict[str, Any]) -> None:
        try:
            self._content = _extract_response_content(response)
            log.info(
                "InspectScene view %s attempt %s raw VLM reply: %s",
                self.index, self.attempt, self._content,
            )
            self._debug_dump_reply(self._content)
            data = _json_from_content(self._content)
            if not isinstance(data, dict):
                self._log_entry("invalid response (not a JSON object)", raw=self._content)
                self._fail("invalid_detector_response: not a JSON object")
                return

            confidence = data.get("confidence")
            try:
                confidence = float(confidence) if confidence is not None else None
            except (TypeError, ValueError):
                confidence = None
            self.found_confidence = confidence

            visible = data.get("visible_objects")
            if isinstance(visible, list):
                log.info(
                    "InspectScene view %s attempt %s visible_objects: %s",
                    self.index, self.attempt,
                    [obj.get("label") for obj in visible if isinstance(obj, dict)],
                )

            if data.get("status") == "Not Found":
                self._log_entry(f"Not Found (confidence={confidence})", visible=visible)
                log.speak("I could not find %s", self.target)
                self._fail("object_not_found")
                return

            label = data.get("label")
            box = _as_pixel_box(data.get("box"))
            if not label or box is None:
                self._log_entry(f"Found without usable label/box -> Not Found (confidence={confidence})", visible=visible, raw=self._content)
                self._fail("object_not_found")
                return
            label = str(label).strip()

            if label.lower() in {lbl.lower() for lbl in self.rejected_labels}:
                self._log_entry(f"Found '{label}' again after rejection -> ignored", visible=visible)
                self._fail(f"repeated_rejected_label: '{label}'")
                return

            if confidence is not None and confidence < float(self.min_confidence):
                self._log_entry(
                    f"Found '{label}' box={box} confidence={confidence} < min {float(self.min_confidence)} -> ignored",
                    visible=visible,
                )
                self._fail(
                    f"low_confidence: '{label}' at {confidence:.2f} < {float(self.min_confidence):.2f}"
                )
                return

            self.found_label = label
            self.found_box = box
            log.speak("I see %s", label)
            self._log_entry(f"Found '{label}' box={box} confidence={confidence}", visible=visible)
            log.info(
                "InspectScene view %s attempt %s: VLM reports '%s' at %s (confidence=%s)",
                self.index, self.attempt, label, box, confidence,
            )
            self._status = "success"
        except Exception as exc:  # noqa: BLE001
            self._log_entry(f"unparseable response: {exc}", raw=self._content)
            self._fail(f"invalid_detector_response: {exc}")

    def _fail(self, reason: str) -> None:
        self._failure_reason = reason
        self._status = "failure"

    def check_result(self, context: Context) -> ActionResult:
        if self._cancel_requested:
            return ActionResult(status="failure", reason="Action Cancelled")
        if self._status == "success":
            return ActionResult(
                status="success",
                payload={
                    "response": self._content,
                    "found_label": self.found_label,
                    "found_bbox": self.found_box,
                    "confidence": self.found_confidence,
                },
            )
        if self._status == "failure":
            return ActionResult(status="failure", reason=self._failure_reason)
        return ActionResult(status="running")

    def reset(self) -> None:
        super().reset()
        self._status = "idle"
        self._failure_reason = None
        self._content = ""
        self.found_label = None
        self.found_box = None
        self.found_confidence = None

    def cancel(self, context: Context) -> None:
        self._cancel_requested = True


class _ExploreView(AbstractAction):
    """One head pose of InspectScene: look, ask the VLM, verify, maybe retry.

    Native sequences and bounded retries expose each observation directly
    to the mission tree and its visitors:

    1. ``MoveHeadToPanTilt`` (skipped when no head controller) then
       ``_DetectDescribedTargetInView``.
    2. If the VLM reports a find and verification is enabled: ``ObserveScene``
       on the reported label, with ``replan_on_failure=False``.
       - success -> SAM3 segmented the label, the view succeeds.
       - ``object_not_found`` -> hallucination; re-ask the VLM with that
         feedback while attempts remain, else the view fails.
       - any other failure (missing depth, empty point cloud, ...) -> the view
         fails without re-asking; that is a perception problem, not a VLM one.
    """

    def __init__(
        self,
        target: str,
        detector_module: str,
        pan: float,
        tilt: float,
        index: int,
        move_head: bool,
        max_retries: int,
        min_confidence: float,
        verify_with_observe: bool,
    ) -> None:
        self.target = target
        self.detector_module = detector_module
        self.pan = pan
        self.tilt = tilt
        self.index = index
        self.move_head = move_head
        self.max_retries = max_retries
        self.min_confidence = min_confidence
        self.verify_with_observe = verify_with_observe
        super().__init__()
        self._detector: _DetectDescribedTargetInView | None = None
        self._observer: ObserveScene | None = None
        self._attempt = 0
        self._rejected_labels: list[str] = []
        self._rejected_box: list[float] | None = None

    def display_name(self) -> str:
        return f"ExploreSceneView[{self.index}]"

    # -- stage construction -------------------------------------------------

    def _new_detector(self) -> _DetectDescribedTargetInView:
        return _DetectDescribedTargetInView(
            target=self.target,
            detector_module=self.detector_module,
            pan=float(self.pan),
            tilt=float(self.tilt),
            index=int(self.index),
            attempt=self._attempt,
            min_confidence=float(self.min_confidence),
            rejected_labels=self._rejected_labels,
            rejected_box=self._rejected_box,
        )




    # -- tick ---------------------------------------------------------------




    def _log_verify(self, label: str, outcome: str) -> None:
        _run_log(
            f"VERIFY target='{self.target}' view={self.index} attempt={self._attempt} "
            f"label='{label}'\n  outcome: {outcome}"
        )


    # -- world state --------------------------------------------------------

    def _publish_find(self, context: Context, label: str, box: list[float], *, observed: bool) -> None:
        ws = context.world_state
        if ws is None:
            return
        view = {"pan": float(self.pan), "tilt": float(self.tilt), "index": int(self.index)}
        # Under the vague description (what InspectScene was asked for)...
        ws.set(f"{self.target}_scan_bbox", box)
        ws.set(f"{self.target}_scan_label", label)
        ws.set(f"{self.target}_scan_view", view)
        ws.set(f"{self.target}_visible", True)
        ws.set(f"{self.target}_scan_observed", observed)
        # ...and under the concrete label, so follow-up actions keyed on
        # found_label see the same scan data ScanScene would have written.
        ws.set(f"{label}_scan_bbox", box)
        ws.set(f"{label}_scan_view", view)
        ws.set(f"{label}_visible", True)

    # -- lifecycle ----------------------------------------------------------




    def build_node(self, context, name):
        self._detector = self._new_detector()
        self._observer = ObserveScene(target="", add_other_objects_to_sg=False, replan_on_failure=False)
        self._detect_node = self._detector.build_node(context, "Find candidate")
        self._verify_node = self._observer.build_node(context, "Verify candidate")
        bind = BtConditionLeaf(self._bind_candidate, context)
        bind.name = "Bind detected candidate"
        children = [self._detect_node, bind]
        if self.verify_with_observe:
            children.append(self._verify_node)
        attempt = ActionScope(self, context, "Candidate attempt", py_trees.composites.Sequence(name="Candidate verification", memory=True, children=children), on_enter=self._begin_attempt, on_complete=self._complete_attempt, on_cancel=lambda context: None)
        retries = py_trees.decorators.Retry(name="Retry rejected candidates", child=attempt, num_failures=self.max_retries + 1)
        def eligible():
            result = attempt.result
            return result is None or result.status != "failure" or bool(result.failure and result.failure.retryable)
        guarded = py_trees.decorators.EternalGuard(name="Candidate retry eligibility", child=retries, condition=eligible)
        children = [guarded]
        if self.move_head and context.head is not None:
            children.insert(0, MoveHeadToPanTilt(pan=self.pan, tilt=self.tilt).build_node(context, "Look at view"))
        return ActionScope(self, context, name, py_trees.composites.Sequence(name="Explore operations", memory=True, children=children))

    def on_enter(self, context):
        self._attempt = -1
        self._rejected_labels = []
        self._rejected_box = None

    def _begin_attempt(self, context):
        self._attempt += 1
        self._detector.attempt = self._attempt
        self._detector.rejected_labels = list(self._rejected_labels)
        self._detector.rejected_box = self._rejected_box
        self._detector.found_label = None
        self._detector.found_box = None

    def _bind_candidate(self, context):
        label = self._detector.found_label
        box = self._detector.found_box
        if not label or box is None:
            return False
        if label.lower() in {value.lower() for value in self._rejected_labels}:
            return False
        self._observer.target = label
        return True

    def _complete_attempt(self, context, status):
        label, box = self._detector.found_label, self._detector.found_box
        if status == py_trees.common.Status.SUCCESS:
            self._publish_find(context, label, box, observed=self.verify_with_observe)
            return ActionResult(status="success")
        detected = self._detect_node.result
        verified = self._verify_node.result
        rejected = bool(label and box is not None and (label.lower() in {value.lower() for value in self._rejected_labels} or (verified and verified.failure and verified.failure.code == FailureCode.NOT_FOUND)))
        if rejected:
            if label.lower() not in {value.lower() for value in self._rejected_labels}:
                self._rejected_labels.append(label)
            self._rejected_box = box
            self._log_verify(label, "REJECTED: candidate verification failed")
            return ActionResult(status="failure", reason="candidate_rejected", failure=FailureInfo(FailureCode.NOT_FOUND, retryable=True))
        return verified or detected or ActionResult(status="failure", reason="object_not_found", failure=FailureInfo(FailureCode.NOT_FOUND))


class InspectScene(AbstractAction):
    """Inspect the current scene for an underspecified target.

    Pans the head through multiple views, asks a VLM which visible objects match
    the target description, and verifies positive matches with ObserveScene. It doen't have a navigation  component, it only inspects the current postion.

    Params:
        target: Description such as "something to eat".
    """

    required_controllers = []

    def __init__(
        self,
        target: str = '',
        pan_angles: list[float] | None = None,
        tilt: float = -0.35,
        detector_module: str = DEFAULT_EXPLORE_MODULE,
        exploration_mode: bool = False,
        max_retries: int = 1,
        min_confidence: float = 0.0,
        verify_with_observe: bool = True,
    ) -> None:
        self.target = target
        self.pan_angles = pan_angles
        self.tilt = tilt
        self.detector_module = detector_module
        self.exploration_mode = exploration_mode
        self.max_retries = max_retries
        self.min_confidence = min_confidence
        self.verify_with_observe = verify_with_observe
        self.pan_angles = pan_angles or [0.0, 0.45, -0.45]
        self.max_retries = max(0, int(max_retries))
        self.min_confidence = float(min_confidence)
        self.verify_with_observe = bool(verify_with_observe)
        super().__init__()



    def _found_result(self, context: Context) -> ActionResult:
        """Report a verified find as needs_replan, carrying the specific label.

        InspectScene is asked about a vague description (e.g. "something to
        eat"), not an exact name. Whatever it actually finds is only known at
        runtime, so a plan written before this ran can only reference the
        vague description. Replanning here lets the LLM write the rest of the
        task against the specific thing found instead.
        """
        ws = context.world_state
        found_label = ws.get(f"{self.target}_scan_label") if ws is not None else None
        found_label = found_label or self.target
        found_bbox = ws.get(f"{self.target}_scan_bbox") if ws is not None else None
        observed = False # bool(ws.get(f"{self.target}_scan_observed")) if ws is not None else False
        next_step = (
            "It has already been observed; continue with EstimatePickPose/PickTarget "
            "using this specific object as the target."
            if observed
            else "Continue the task by making a new plan using this specific object as the target object."
        )
        return ActionResult(
            status="needs_replan",
            reason=f"Found '{found_label}' matching '{self.target}'. {next_step}",
            suggested_replan_context={
                "action": "InspectScene",
                "requested_target": self.target,
                "found_label": found_label,
                "found_bbox": found_bbox,
                "observed": observed,
            },
            # Stop the tree here: in a fallback over furniture the selector
            # would otherwise treat this as a miss and drive on to the next one.
            force_replan=True,
        )

    def _exhausted_result(self, reason: str) -> ActionResult:
        self.exploration_mode = True
        if self.exploration_mode:
            return ActionResult(
                status="needs_replan",
                reason=(
                    f"Nothing matching '{self.target}' was found at the current location "
                    "after exploring all views. Try a different, not-yet-attempted furniture "
                    "candidate, or use AskUser if every plausible candidates have already been tried."
                ),
                suggested_replan_context={
                    "action": "InspectScene",
                    "target": self.target,
                    "reason": reason,
                    "found": False,
                },
            )
        return ActionResult(
            status="needs_user_input",
            reason=(
                f"I could not find anything matching '{self.target}' after exploring the "
                "current view. Please clarify where to look next or whether to continue searching."
            ),
            suggested_replan_context={
                "action": "InspectScene",
                "target": self.target,
                "reason": reason,
            },
        )



    def build_node(self, context, name):
        views = [_ExploreView(target=self.target, detector_module=self.detector_module, pan=float(pan), tilt=float(self.tilt), index=index, move_head=context.head is not None, max_retries=int(self.max_retries), min_confidence=float(self.min_confidence), verify_with_observe=bool(self.verify_with_observe)).build_node(context, f"Explore view {index}") for index, pan in enumerate(self.pan_angles)]
        return ActionScope(self, context, name, py_trees.composites.Selector(name="Inspect views", memory=True, children=views))

    def on_enter(self, context):
        if context.sensor_data_center is None:
            return self._exhausted_result("missing_sensor_data_center")
        if context.modules is None:
            return self._exhausted_result("missing_module_proxy")

    def on_complete(self, context, status):
        return self._found_result(context) if status == py_trees.common.Status.SUCCESS else self._exhausted_result("explore_exhausted")
